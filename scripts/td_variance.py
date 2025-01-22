from typing import Union

from jax import jit
import jax.numpy as jnp

from grl.environment import load_pomdp
from grl.mdp import MDP, POMDP
from grl.utils.mdp import functional_get_occupancy, get_p_s_given_o, functional_create_td_model
from grl.utils.policy_eval import functional_solve_mdp

from grl.environment.policy_lib import switching_two_thirds_right_policy


# @jit
def value_second_moment(pi: jnp.ndarray, mdp: Union[POMDP, MDP]):
    # First solve for values.
    # Taken from functional_solve_mdp
    Pi_pi = pi.transpose()[..., None]
    T_pi = (Pi_pi * mdp.T).sum(axis=0) # T^π(s'|s)
    R_pi = (Pi_pi * mdp.T * mdp.R).sum(axis=0).sum(axis=-1) # R^π(s)

    # A*V_pi(s) = b
    # A = (I - \gamma (T^π))
    # b = R^π
    A = (jnp.eye(mdp.state_space.n) - mdp.gamma * T_pi)
    b = R_pi
    v_pi_s = jnp.linalg.solve(A, b)

    R_sa = (mdp.T * mdp.R).sum(axis=-1) # R(s,a)
    q_pi_s = (R_sa + (mdp.gamma * mdp.T @ v_pi_s))


    R_pi_s_s = (Pi_pi * mdp.R).sum(axis=0) # S x S

    R_pi_s_s_squared = R_pi_s_s ** 2

    R_v_s_prime = R_pi_s_s * v_pi_s[None, ...]
    R_2_pi_s_over_s_prime = T_pi * (R_pi_s_s_squared + 2 * mdp.gamma * R_v_s_prime)
    R_2_pi_s = R_2_pi_s_over_s_prime.sum(axis=-1)

    # A*V^{(2)}_pi(s) = b
    # A = (I - \gamma^2 (T_π))
    # b = R^{(2)}_π
    A = (jnp.eye(mdp.state_space.n) - (mdp.gamma ** 2) * T_pi)
    b = R_2_pi_s
    V_2_pi_s = jnp.linalg.solve(A, b)  # Second moment of V over state

    # TODO: add Q_2
    return V_2_pi_s, {'v': v_pi_s, 'q': q_pi_s}


def get_variances(pi: jnp.ndarray, pomdp: POMDP):
    pi_sa = pomdp.phi @ pi

    # Now we map to observation-based value functions
    occupancy = functional_get_occupancy(pi_sa, pomdp)

    p_pi_of_s_given_o = get_p_s_given_o(pomdp.phi, occupancy)
    # second moment over STATES
    V_2_pi_s, state_values = value_second_moment(pi_sa, pomdp)
    V_squared_pi_s = (state_values['v'] ** 2)
    var_pi_s = V_2_pi_s - V_squared_pi_s

    # MC
    V_pi_o_mc = state_values['v'] @ p_pi_of_s_given_o
    V_2_pi_o_mc = V_2_pi_s @ p_pi_of_s_given_o
    V_squared_pi_o_mc = (V_pi_o_mc ** 2)
    var_pi_o_mc = V_2_pi_o_mc - V_squared_pi_o_mc

    # TD
    T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, pomdp)
    td_model = MDP(T_obs_obs, R_obs_obs, pomdp.p0 @ pomdp.phi, gamma=pomdp.gamma)
    V_2_pi_o_td, obs_td_values = value_second_moment(pi, td_model)
    V_squared_pi_o_td = (obs_td_values['v'] ** 2)
    var_pi_o_td = V_2_pi_o_td - V_squared_pi_o_td

    variances = {
        'td': var_pi_o_td,
        'mc': var_pi_o_mc,
        'state': var_pi_s
    }
    return variances

if __name__ == "__main__":
    # env_str = 'tmaze_5_two_thirds_up'
    env_str = 'counting_wall'
    # env_str = 'switching'

    pomdp, pi_dict = load_pomdp(env_str,
                                corridor_length=5,
                                discount=0.9)

    # This is state-based variance
    if env_str == 'switching':
        pi = switching_two_thirds_right_policy()
    elif env_str == 'counting_wall':
        raise NotImplementedError
    else:
        pi = pi_dict['Pi_phi'][0]

    get_variances(pi, pomdp)

