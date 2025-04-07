from copy import deepcopy
import inspect
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm


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
    phi, T, R, p0, Pi_phi = spec['phi'], spec['T'], spec['R'], spec['p0'], spec['Pi_phi']

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
                # first reset the row
                new_T[a, new_s] *= 0

                if np.all(T[:, s, s] == 1):
                    # Check here for terminal. If og state was terminal, this new one will be absorbing to itself as well.
                    new_T[a, new_s, new_s] = 1
                else:
                    for sp in range(n_states):
                        # now we need to find the corresponding next state.
                        # There should be only one entry since (s, r) entries are unique.
                        next_new_s = new_states.index((sp, R[a, s, sp]))

                        # Now we assign the probability accordingly
                        new_T[a, new_s, next_new_s] = T[a, s, sp]
                        new_R[a, new_s, next_new_s] = R[a, s, sp]

    # now we need to make a new phi function,
    # mapping all our reward-expanded states into new observations.
    new_phi = []
    o_to_new_o_mapping = {}
    new_o = 0
    for o in range(n_obs):
        phi_o = phi[:, o]  # prob dist of shape |S| (og)
        applicable_s_for_o = np.nonzero(phi_o)[0]

        # make a unique list of all rewards for this observation.
        # This is how many new observations we should be adding.
        all_o_rewards = []
        for s in applicable_s_for_o:
            all_o_rewards += states_to_rewards[s].tolist()
        all_unique_o_rewards = np.unique(all_o_rewards)

        # for each reward
        for r in all_unique_o_rewards:
            new_phi_o_r = np.zeros(new_n_states)
            # populate our new phi column
            for s in applicable_s_for_o:
                # if s admits an r reward, set the new phi to the corresponding phi.
                # if not, it should remain 0.
                if (s, r) in new_states:
                    new_s = new_states.index((s, r))
                    new_phi_o_r[new_s] = phi[s, o]
            new_phi.append(new_phi_o_r)
            if o not in o_to_new_o_mapping:
                o_to_new_o_mapping[o] = []
            o_to_new_o_mapping[o].append(new_o)
            new_o += 1

    new_phi = np.stack(new_phi, axis=-1)

    # Now we need to expand p0.
    # we (uniformly) split up starting probabilities among split states.
    new_p0 = np.zeros(new_n_states)
    for new_s, (s, r) in enumerate(new_states):
        new_p0[new_s] = p0[s] / len(states_to_rewards[s].tolist())

    # finally, we have our policies.
    # we'll just copy our policies
    new_Pi_phi = None
    if Pi_phi is not None:
        new_Pi_phi = []
        for pi in Pi_phi:
            new_pi = []
            for o, pi_o in enumerate(pi):
                for new_o in o_to_new_o_mapping[o]:
                    new_pi.append(pi_o)

            new_Pi_phi.append(np.stack(new_pi, axis=0))
        new_Pi_phi = np.stack(new_Pi_phi, axis=0)

    new_spec = deepcopy(spec)
    new_spec.update({
        'phi': new_phi,
        'T': new_T,
        'R': new_R,
        'p0': new_p0,
        'Pi_phi': new_Pi_phi,
    })
    if 'Pi_phi_x' in new_spec and new_spec['Pi_phi_x'] is not None:
        new_spec['Pi_phi_x'] = None
    return new_spec


def load_pomdp(name: str,
               reward_in_obs: bool = False,
               rand_key: np.random.RandomState = None, **kwargs) -> Tuple[POMDP, dict]:
    """
    Wraps a MDP/POMDP specification in a POMDP
    """
    spec = load_spec(name, rand_key=rand_key, **kwargs)
    if reward_in_obs:
        spec = add_rewards_in_obs(spec)
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'], rand_key=rand_key)
    pomdp = POMDP(mdp, spec['phi'])
    return pomdp, {'Pi_phi': spec['Pi_phi'], 'Pi_phi_x': spec['Pi_phi_x']}


def augment_pomdp_gamma(pomdp: POMDP,
                        rand_key: jax.random.PRNGKey,
                        augmentation: str = 'uniform',  # uniform | normal
                        scale: float = None):
    pomdp = deepcopy(pomdp)
    """
    TODO: these augmentations assume observation-conditioned gammas, but 
    we implement them as state-conditioned gammas
    """

    o = pomdp.observation_space.n

    if augmentation == 'uniform':
        obs_gammas = jax.random.uniform(rand_key, shape=(o, ), minval=0, maxval=1)
    elif augmentation == 'normal':
        mean = 0.0
        std = 1.0
        low, high = -1.0, 1.0

        # Compute CDF bounds
        cdf_low = norm.cdf(low, loc=mean, scale=std)
        cdf_high = norm.cdf(high, loc=mean, scale=std)

        # Sample uniform values between the CDF bounds
        u = jax.random.uniform(rand_key, shape=(o,), minval=cdf_low, maxval=cdf_high)

        # Invert the CDF to get samples from the truncated normal
        obs_gammas = norm.ppf(u, loc=mean, scale=std)
    else:
        raise NotImplementedError

    if scale is not None:
        raise NotImplementedError

    # now we need to map obs_gammas to state_gammas
    state_phi_occupancy = (pomdp.phi > 0).astype(float)  # S x O
    state_gammas = state_phi_occupancy @ obs_gammas

    pomdp.gamma = state_gammas
    return pomdp





