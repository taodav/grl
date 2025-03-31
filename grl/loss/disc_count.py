import jax.numpy as jnp

from grl.mdp import POMDP, MDP
from grl.utils.mdp_solver import (
    functional_get_occupancy, functional_get_undiscounted_occupancy,
    get_p_s_given_o, functional_create_td_model
)
from grl.utils.policy_eval import functional_solve_mdp, functional_solve_pomdp

def disc_count_loss(pi: jnp.ndarray,
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


def mem_disc_count_loss(
        mem_params: jnp.ndarray,
        pi: jnp.ndarray,
        pomdp: POMDP,  # input non-static arrays
        dist: str = 'diff',  # [diff, kl]
        **kwargs):
    pass


if __name__ == '__main__':
    import jax
    from grl.environment import load_pomdp

    jax.disable_jit(True)

    env, info = load_pomdp('tmaze_5_two_thirds_up', memory_id='18')
    pi = info['Pi_phi'][0]

    loss = disc_count_loss(pi, env, dist='diff')
