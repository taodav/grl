import numpy as np

from grl.mdp import MDP, POMDP


def map_to_strict_aliasing(pomdp: POMDP) -> POMDP:
    # states_to_expand = {}
    # for i, phi_row in enumerate(pomdp.phi):
    #     state_maps_to_n_obs = (phi_row > 0).sum()
    #     if state_maps_to_n_obs > 1:
    #         pass
    n_states, n_obs = pomdp.state_space.n, pomdp.observation_space.n
    n_actions = pomdp.action_space.n
    p0, phi, T, R = pomdp.p0, pomdp.phi, pomdp.T, pomdp.R


    n_states_new = 0
    new_state_dict = {}
    # add a new state (s,o) only if the state s can actually
    # produce the observation o
    for s in range(n_states):
        for o in range(n_obs):
            if phi[s, o] > 0:
                new_state_dict[(s, o)] = n_states_new
                n_states_new += 1

    phi_new = np.zeros((n_states_new, n_obs))
    p0_new = np.zeros(n_states_new)
    T_new = np.zeros((n_actions, n_states_new, n_states_new))
    R_new = np.zeros((n_actions, n_states_new, n_states_new))
    for (s, o), new_s in new_state_dict.items():
        phi_new[new_s, o] = 1.0
        p0_new[new_s] = p0[s] * phi[s, o]

        for (s2, o2), new_s2 in new_state_dict.items():
            for a in range(n_actions):
                T_new[a, new_s, new_s2] = T[a, s, s2] * phi[s2, o2]
                R_new[a, new_s, new_s2] = R[a, s, s2]
    mdp = MDP(T_new, R_new, p0_new, pomdp.gamma, rand_key=pomdp.rand_key)
    new_pomdp = POMDP(mdp, phi_new)

    return new_pomdp

