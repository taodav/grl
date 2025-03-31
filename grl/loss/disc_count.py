import jax.numpy as jnp

from grl.mdp import POMDP, MDP
from grl.utils.mdp_solver import functional_get_occupancy, get_p_s_given_o, functional_create_td_model

def disc_count_loss(pi: jnp.ndarray,
                    pomdp: POMDP,
                    dist: str = 'diff'):
    pi_state = pomdp.phi @ pi
    occupancy = functional_get_occupancy(pi_state, pomdp)

    p_pi_of_s_given_o = get_p_s_given_o(pomdp.phi, occupancy)
    T_phi = pomdp.T @ pomdp.phi @ p_pi_of_s_given_o.T
    td_pomdp = MDP(T_phi, pomdp.R, pomdp.p0, gamma=pomdp.gamma)
    td_occupancy = functional_get_occupancy(pi_state, td_pomdp)
    td_obs_occupancy = None


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
