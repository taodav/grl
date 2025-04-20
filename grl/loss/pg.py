import jax.numpy as jnp
from jax import nn, lax, jit

from grl.utils.policy import deconstruct_aug_policy
from grl.utils.mdp_solver import functional_get_occupancy, get_p_s_given_o, functional_create_td_model
from grl.utils.policy_eval import functional_solve_mdp
from grl.utils.augmented_policy import deconstruct_aug_policy
from grl.utils import reverse_softmax
from grl.memory import memory_cross_product
from grl.mdp import MDP, POMDP


def pg_objective_func(pi_params: jnp.ndarray, pomdp: POMDP,
                      disc_occupancy: bool = False):
    """
    Policy gradient objective function:
    sum_{s_0} p(s_0) v_pi(s_0)
    """
    pi_abs = nn.softmax(pi_params, axis=-1)
    pi_ground = pomdp.phi @ pi_abs

    # Terminals have p(S) = 0.
    occupancy = functional_get_occupancy(pi_ground, pomdp, discounted=disc_occupancy) * (1 - pomdp.terminal_mask)

    p_pi_of_s_given_o = get_p_s_given_o(pomdp.phi, occupancy)
    T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, pomdp)
    td_model = MDP(T_obs_obs, R_obs_obs, pomdp.p0 @ pomdp.phi, gamma=pomdp.gamma)
    td_v_vals, td_q_vals = functional_solve_mdp(pi_abs, td_model)
    p_init_obs = pomdp.p0 @ pomdp.phi
    return jnp.dot(p_init_obs, td_v_vals), (td_v_vals, td_q_vals)


def augmented_pg_objective_func(augmented_pi_params: jnp.ndarray, pomdp: POMDP):
    """
    Policy gradient objective function:
    sum_{s_0} p(s_0) v_pi(s_0)
    """
    augmented_pi_probs = nn.softmax(augmented_pi_params)
    mem_probs, action_policy_probs = deconstruct_aug_policy(augmented_pi_probs)
    mem_logits = reverse_softmax(mem_probs)
    mem_aug_mdp = memory_cross_product(mem_logits, pomdp)
    return pg_objective_func(action_policy_probs, mem_aug_mdp)


def mem_pg_objective_func(augmented_pi_params: jnp.ndarray, pomdp: POMDP):
    augmented_pi_probs = nn.softmax(augmented_pi_params, axis=-1)
    mem_logits, action_policy_probs = deconstruct_aug_policy(augmented_pi_probs)
    mem_aug_mdp = memory_cross_product(mem_logits, pomdp)
    O, M, A = action_policy_probs.shape
    return pg_objective_func(reverse_softmax(action_policy_probs).reshape(O * M, A), mem_aug_mdp)


def unrolled_mem_pg_objective_func(augmented_pi_params: jnp.ndarray, pomdp: POMDP):# O, M, AM
    augmented_pi_probs_unflat = nn.softmax(augmented_pi_params, axis=-1)
    mem_logits, action_policy_probs = deconstruct_aug_policy(augmented_pi_probs_unflat)# A,O,M->M ; O,M->A
    O, M, A = action_policy_probs.shape
    mem_aug_mdp = memory_cross_product(mem_logits, pomdp)

    pi_probs = action_policy_probs.reshape(O * M, A)
    aug_pi_probs = augmented_pi_probs_unflat.reshape(O * M, A * M)

    pi_ground = mem_aug_mdp.phi @ pi_probs  # pi: (S * M, A)
    occupancy = functional_get_occupancy(pi_ground, mem_aug_mdp)  # eta: S * M
    om_occupancy = occupancy @ mem_aug_mdp.phi  # om_eta: O * M

    # Calculate our Q vals over A x O * M
    p_pi_of_s_given_o = get_p_s_given_o(mem_aug_mdp.phi, occupancy) # P(SM|OM)
    T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, mem_aug_mdp) # T(O'M'|O,M,A); R(O,M,A)
    td_model = MDP(T_obs_obs, R_obs_obs, mem_aug_mdp.p0 @ mem_aug_mdp.phi, gamma=mem_aug_mdp.gamma)
    td_v_vals, td_q_vals = functional_solve_mdp(pi_probs, td_model)  # q: (A, O * M)

    # expand over A * M
    mem_probs = nn.softmax(mem_logits, axis=-1) # (A, O, M, M)
    mem_probs_omam = jnp.moveaxis(mem_probs, 0, -2)  # (O, M, A, M)
    mem_probs_omam = mem_probs_omam.reshape(O * M, A, M)  # (OM, A, M)
    #(OM,A,M)                        #(A, OM)^T => (OM, A) => (OM, A, 1)
    am_q_vals = mem_probs_omam * jnp.expand_dims(td_q_vals.T, -1)  # (OM, A, M)
    am_q_vals = am_q_vals.reshape(O * M, A * M)  # (OM, AM)

    # Don't take gradients over eta or Q
    weighted_am_q_vals = jnp.expand_dims(om_occupancy, -1) * am_q_vals
    weighted_am_q_vals = lax.stop_gradient(weighted_am_q_vals)
    return (weighted_am_q_vals * aug_pi_probs).sum(), (td_v_vals, td_q_vals)

