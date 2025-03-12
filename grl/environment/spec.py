import inspect
from pathlib import Path
from typing import Tuple

import numpy as np

from definitions import ROOT_DIR
from grl.environment.pomdp_file import POMDPFile
from grl.mdp import MDP, POMDP
from grl.utils.math import normalize
from . import examples_lib

def load_spec(name: str, **kwargs):
    """
    Loads a pre-defined POMDP specification, as well as policies.
    :param name:            The name of the function or .POMDP file defining the POMDP.
    :param memory_id:       ID of memory function to use.
    :param n_mem_states:    Number of memory states allowed.
    :param mem_leakiness:   for memory_id="f" - how leaky do is out leaky identity function.

    The following **kwargs are specified for the following specs:
    tmaze_hyperparams:
        :param corridor_length:     Length of the maze corridor.
        :param discount:            Discount factor gamma to use.
        :param junction_up_pi:      If we specify a policy for the tmaze spec, what is the probability
                                    of traversing UP at the tmaze junction?
    """

    # Try to load from examples_lib first
    # then from pomdp_files
    spec = None
    try:
        spec_fn = getattr(examples_lib, name)
        arg_names = inspect.signature(spec_fn).parameters
        kwargs = {k: v for k, v in kwargs.items() if v is not None and k in arg_names}
        spec = spec_fn(**kwargs)

    except AttributeError:
        pass

    if spec is None:
        try:
            file_path = Path(ROOT_DIR, 'grl', 'environment', 'pomdp_files', f'{name}.POMDP')
            spec = POMDPFile(file_path).get_spec()
        except FileNotFoundError as _:
            raise AttributeError

    # Check sizes and types
    if len(spec.keys()) < 6:
        raise ValueError("POMDP specification must contain at least: T, R, gamma, p0, phi, Pi_phi")
    if len(spec['T'].shape) != 3:
        raise ValueError("T tensor must be 3d")
    if len(spec['R'].shape) != 3:
        raise ValueError("R tensor must be 3d")

    if spec['Pi_phi'] is not None:
        spec['Pi_phi'] = np.array(spec['Pi_phi']).astype('float')
        spec['Pi_phi'] = normalize(spec['Pi_phi'])
        if not np.all(len(spec['T']) == np.array([len(spec['R']), len(spec['Pi_phi'][0][0])])):
            raise ValueError("T, R, and Pi_phi must contain the same number of actions")

    # Make sure probs sum to 1
    # e.g. if they are [0.333, 0.333, 0.333], normalizing will do so
    spec['T'] = normalize(spec['T']) # terminal states had all zeros -> nan
    spec['p0'] = normalize(spec['p0'])
    spec['phi'] = normalize(spec['phi'])

    return spec

def add_rewards_in_obs(spec: dict) -> dict:
    phi, T, R = spec['phi'], spec['T'], spec['R']

    list_new_phi = []

    n_actions = T.shape[0]
    n_states, n_obs = phi.shape
    states_to_rewards = {}
    new_states = []

    new_n_states = 0
    # first we need to find all states where R[:, :, sp] has more than one unique element
    for sp in range(n_states):
        unique_rewards = np.unique(R[:, :, sp])
        states_to_rewards[sp] = unique_rewards
        new_n_states += len(unique_rewards)
        for r in unique_rewards:
            new_states.append((sp, r))

    # now we construct our new_T, with rewards
    new_T = np.eye(new_n_states)[None, ...].repeat(n_actions, axis=0)
    new_R = np.zeros_like(new_T)

    for a in range(n_actions):
        for s in range(n_states):

            # first find the new index of current state s
            matching_new_indices = [(i, s, r) for i, (s_new, r) in enumerate(new_states) if s == s_new]

            for new_s, _, r in matching_new_indices:

                if np.all(T[:, s, s] == 1):
                    # TODO: check here for terminal. If og state was terminal, this new one will be absorbing to itself as well.
                    pass
                else:
                    for sp in range(n_states):
                        # first reset the row
                        new_T[a, new_s] *= 0

                        # now we need to find the corresponding next state.
                        # There should be only one entry since (s, r) entries are unique.
                        next_new_s = new_states.index((sp, R[a, s, sp]))

                        # Now we assign the probability accordingly
                        new_T[a, new_s, next_new_s] = T[a, s, sp]
                        new_R[a, new_s, next_new_s] = r






    # now we need to make our indexing of states.


    # for o in range(n_obs):
    #     state_to_reward_map = {}
    #     for sp in range(n_states):
    #         if not np.isclose(phi[sp, o], 0):
    #             # we only check states that map to o
    #
    #             state_to_reward_map[sp] = np.unique(R[:, :, sp])
    #             # for each reward and obs, we list out all states that map to it
    #             for r in :
    #                 if r not in reward_to_state_map:
    #                     reward_to_state_map[r] = []
    #                 reward_to_state_map[r].append(sp)
    #
    #     if len(reward_to_state_map.keys()) >= 2:
    #         # if we're here, then observation o needs to be split depending on number of keys
    #         for r, states in reward_to_state_map.items():
    #             new_obs = np.zeros(n_states)
    #             new_obs[states] = 1
    #             list_new_phi.append(new_obs)
    #     else:
    #         list_new_phi.append(phi[:, o])

    print()



def load_pomdp(name: str,
               reward_in_obs: bool = True,
               rand_key: np.random.RandomState = None, **kwargs) -> Tuple[POMDP, dict]:
    """
    Wraps a MDP/POMDP specification in a POMDP
    """
    spec = load_spec(name, rand_key=rand_key, **kwargs)
    if reward_in_obs:
        # TODO
        spec = add_rewards_in_obs(spec)
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'], rand_key=rand_key)
    pomdp = POMDP(mdp, spec['phi'])
    return pomdp, {'Pi_phi': spec['Pi_phi'], 'Pi_phi_x': spec['Pi_phi_x']}

