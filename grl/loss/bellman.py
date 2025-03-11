from functools import partial

import jax.numpy as jnp
from jax import jit, lax

from grl.memory import memory_cross_product
from grl.mdp import POMDP
from grl.utils.mdp_solver import get_p_s_given_o, functional_create_td_model
from grl.utils.policy_eval import lstdq_lambda

from grl.loss.ld import weight_and_sum_discrep_loss


def mem_bellman_loss(
        mem_params: jnp.ndarray,
        pi: jnp.ndarray,
        pomdp: POMDP, # input non-static arrays
        value_type: str = 'q',
        error_type: str = 'l2',
        lambda_0: float = 0,
        lambda_1: float = 1.,  # NOT CURRENTLY USED!
        residual: bool = False,
        alpha: float = 1.,
        flip_count_prob: bool = False):
    mem_aug_pomdp = memory_cross_product(mem_params, pomdp)
    loss, _, _ = bellman_loss(pi,
                              mem_aug_pomdp,
                              value_type,
                              error_type,
                              alpha,
                              lambda_=lambda_0,
                              residual=residual,
                              flip_count_prob=flip_count_prob)
    return loss

@partial(jit, static_argnames=['value_type', 'error_type', 'alpha', 'residual', 'flip_count_prob'])
def bellman_loss(
        pi: jnp.ndarray,
        pomdp: POMDP, # non-state args
        value_type: str = 'q',
        error_type: str = 'l2',
        alpha: float = 1.,
        lambda_: float = 0.,
        residual: bool = False,
        flip_count_prob: bool = False): # initialize static args

    # First, calculate our TD(0) Q-values
    v_vals, q_vals, info = lstdq_lambda(pi, pomdp, lambda_=lambda_)
    vals = {'v': v_vals, 'q': q_vals}
    assert value_type == 'q'

    c_s = info['occupancy']
    # Make TD(0) model
    p_pi_of_s_given_o = get_p_s_given_o(pomdp.phi, c_s)
    T_aoo, R_aoo = functional_create_td_model(p_pi_of_s_given_o, pomdp)

    # Tensor for AxOxOxA (obs action to obs action)
    T_pi_aooa = jnp.einsum('ijk,kl->ijkl', T_aoo, pi)
    R_ao = R_aoo.sum(axis=-1)

    # Calculate the expected next value for each (a, o) pair
    expected_next_V_given_ao = jnp.einsum('ijkl,kl->ij', T_pi_aooa, q_vals.T)  # A x O

    # Our Bellman error
    target = R_ao + pomdp.gamma * expected_next_V_given_ao
    if not residual:
        target = lax.stop_gradient(target)
    diff = target - q_vals

    # R_s_o = pomdp.R @ pomdp.phi  # A x S x O

    # expanded_R_s_o = R_s_o[..., None].repeat(pomdp.action_space.n, axis=-1)  # A x S x O x A

    # repeat the Q-function over A x O
    # Multiply that with p(O', A' | s, a) and sum over O' and A' dimensions.
    # P(O' | s, a) = T @ phi, P(A', O' | s, a) = P(O' | s, a) * pi (over new dimension)
    # pr_o = pomdp.T @ pomdp.phi
    # pr_o_a = jnp.einsum('ijk,kl->ijkl', pr_o, pi)
    # expected_next_Q = jnp.einsum('ijkl,kl->ijkl', pr_o_a, q_vals.T)
    # expanded_Q = jnp.expand_dims(
    #     jnp.expand_dims(q_vals.T, 0).repeat(pr_o_a.shape[1], axis=0),
    #     0).repeat(pr_o_a.shape[0], axis=0)  # Repeat over OxA -> O x A x O x A
    # diff = expanded_R_s_o + pomdp.gamma * expected_next_Q - expanded_Q


    # set terminal counts to 0
    c_s = info['occupancy'] * (1 - pomdp.terminal_mask)
    loss = weight_and_sum_discrep_loss(diff, c_s, pi, pomdp,
                                       value_type=value_type,
                                       error_type=error_type,
                                       alpha=alpha,
                                       flip_count_prob=flip_count_prob)

    return loss, vals, vals
