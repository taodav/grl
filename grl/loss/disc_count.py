import jax.numpy as jnp
from jax.scipy.special import kl_div

from grl.memory import memory_cross_product
from grl.memory.lib import get_memory
from grl.mdp import POMDP, MDP
from grl.utils.mdp_solver import (
    functional_get_occupancy, functional_get_undiscounted_occupancy,
    get_p_s_given_o, functional_create_td_model
)
from grl.utils.policy_eval import functional_solve_mdp, functional_solve_pomdp


def disc_count_test(pi: jnp.ndarray,
                    pomdp: POMDP,
                    dist: str = 'diff'):
    pi_state = pomdp.phi @ pi
    disc_occupancy = functional_get_occupancy(pi_state, pomdp)
    undisc_occupancy = functional_get_undiscounted_occupancy(pi_state, pomdp)

    p_pi_of_s_given_o_undiscounted = get_p_s_given_o(pomdp.phi, undisc_occupancy)
    T_phi_undisc = pomdp.T @ pomdp.phi @ p_pi_of_s_given_o_undiscounted.T
    td_pomdp_undisc = MDP(T_phi_undisc, pomdp.R, pomdp.p0, gamma=pomdp.gamma)
    disc_td_occupancy_over_undisc_T_phi = functional_get_occupancy(pi_state, td_pomdp_undisc)

    p_pi_of_s_given_o_discounted = get_p_s_given_o(pomdp.phi, disc_occupancy)
    T_phi_disc = pomdp.T @ pomdp.phi @ p_pi_of_s_given_o_discounted.T
    td_pomdp_disc = MDP(T_phi_disc, pomdp.R, pomdp.p0, gamma=pomdp.gamma)
    disc_td_occupancy = functional_get_occupancy(pi_state, td_pomdp_disc)

    # test values here
    v, q = functional_solve_mdp(pi_state, pomdp)

    mc_vals_undisc_count = functional_solve_pomdp(q, p_pi_of_s_given_o_undiscounted, pi)

    mc_vals_disc_count = functional_solve_pomdp(q, p_pi_of_s_given_o_discounted, pi)

    T_obs_obs_undisc, R_obs_obs_undisc = functional_create_td_model(p_pi_of_s_given_o_undiscounted, pomdp)

    print()


def disc_count_loss(pi: jnp.ndarray,
                    pomdp: POMDP,
                    dist: str = 'kl'):
    pi_state = pomdp.phi @ pi
    disc_occupancy = functional_get_occupancy(pi_state, pomdp)
    undisc_occupancy = functional_get_undiscounted_occupancy(pi_state, pomdp)

    p_pi_of_s_given_o_undiscounted = get_p_s_given_o(pomdp.phi, undisc_occupancy)
    T_phi_undisc = pomdp.T @ pomdp.phi @ p_pi_of_s_given_o_undiscounted.T
    td_pomdp_undisc = MDP(T_phi_undisc, pomdp.R, pomdp.p0, gamma=pomdp.gamma)
    disc_td_occupancy_over_undisc_T_phi = functional_get_occupancy(pi_state, td_pomdp_undisc)

    masked_disc_occupancy = disc_occupancy * pomdp.terminal_mask
    mu = masked_disc_occupancy / masked_disc_occupancy.sum(axis=0)

    masked_disc_td_occupancy = disc_td_occupancy_over_undisc_T_phi * pomdp.terminal_mask
    mu_phi = masked_disc_td_occupancy / masked_disc_td_occupancy.sum(axis=0)

    if dist == 'kl':
        loss = kl_div(mu, mu_phi).sum()
    elif dist == 'mse':
        loss = ((mu - mu_phi)**2).mean()
    else:
        raise NotImplementedError

    return loss, None, None


def mem_disc_count_loss(
        mem_params: jnp.ndarray,
        pi: jnp.ndarray,
        pomdp: POMDP,  # input non-static arrays
        dist: str = 'kl',  # [mse, kl]
        **kwargs):
    mem_aug_pomdp = memory_cross_product(mem_params, pomdp)
    loss, _, _ = disc_count_loss(pi, mem_aug_pomdp, dist=dist)
    return loss


if __name__ == '__main__':
    import jax
    from grl.environment import load_pomdp

    jax.disable_jit(True)

    env, info = load_pomdp('tmaze_5_two_thirds_up', memory_id=0)
    pi = info['Pi_phi'][0]
    n_mem = 2
    mem_params = get_memory('0',
                            n_obs=env.observation_space.n,
                            n_actions=env.action_space.n,
                            n_mem_states=n_mem)

    # loss = disc_count_test(pi, env, dist='diff')
    mem_loss = mem_disc_count_loss(mem_params, pi.repeat(n_mem, axis=0), env)
    print()
