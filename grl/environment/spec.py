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
from grl.mdp import MDP, POMDP, POMDPG
from grl.environment.aliasing import map_to_strict_aliasing
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
    # Gamma_o = spec['gamma'] * np.eye(spec['phi'].shape[-1], dtype=float)
    # Gamma_s = spec['gamma'] * np.eye(spec['T'].shape[-1], dtype=float)
    pomdp = POMDP(mdp, spec['phi'])
    return pomdp, {'Pi_phi': spec['Pi_phi'], 'Pi_phi_x': spec['Pi_phi_x']}


def augment_pomdp_gamma(pomdp: POMDP,
                        rand_key: jax.random.PRNGKey,
                        augmentation: str = 'uniform',  # uniform | normal
                        scale: float = None,
                        min_val: float = 0.,
                        max_val: float = 1.,
                        num_gammas: int = 1):
    mdp = deepcopy(pomdp.base_mdp)
    """
    TODO: these augmentations assume observation-conditioned gammas, but 
    we implement them as state-conditioned gammas
    """

    n_obs = pomdp.observation_space.n
    n_states = pomdp.state_space.n

    # assert np.all((pomdp.phi > 0).sum(axis=-1) <= 1), "States map to more than one obs!"
    if not np.all((pomdp.phi > 0).sum(axis=-1) <= 1):
        new_pomdp = map_to_strict_aliasing(pomdp)
        pomdp = new_pomdp


    if augmentation == 'uniform':
        obs_gammas = jax.random.uniform(rand_key, shape=(num_gammas, n_obs), minval=min_val, maxval=max_val)
    elif augmentation == 'normal':
        mean = 0.0
        std = 1.0
        # Compute CDF bounds
        cdf_low = norm.cdf(min_val, loc=mean, scale=std)
        cdf_high = norm.cdf(max_val, loc=mean, scale=std)

        # Sample uniform values between the CDF bounds
        u = jax.random.uniform(rand_key, shape=(num_gammas, n_obs), minval=cdf_low, maxval=cdf_high)

        # Invert the CDF to get samples from the truncated normal
        obs_gammas = norm.ppf(u, loc=mean, scale=std)
    else:
        raise NotImplementedError

    if scale is not None:
        raise NotImplementedError
    
    # TODO: this is (o, 1), and not (s, s), why?
    # pomdp.gamma = obs_gammas[..., None]
    # return pomdp

    # turn into (s, s) matrix
    #Phi = pomdp.phi
    assert np.all(np.count_nonzero(pomdp.phi, axis=1) == 1), f"custom gammas only available for strict aliasing, that is deterministic Phi"
    #phi = np.zeros(n_states, dtype=int)
    #for s in range(n_states):
    #    for o in range(n_obs):
    #        if Phi[s, o] != 0:
    #            phi[s] = o
    #assert obs_gammas.shape == (n_obs,)
    #Gamma_o = np.diag(obs_gammas)
    #state_gammas = np.zeros(n_states, dtype=float)
    #for s in range(n_states):
    #    state_gammas[s] = obs_gammas[phi[s]]
    #Gamma_s = np.diag(state_gammas)
    #new_pomdp = POMDPG(mdp, pomdp.phi, Gamma_o=Gamma_o, Gamma_s=Gamma_s)

    new_pomdp = POMDPG(mdp, pomdp.phi, gamma_o=obs_gammas)

    return new_pomdp


def make_subprob_matrix(T):
    """Find terminal states, which are those where every action just leads back to 
    the same state, and set their out-probabilities to zero, which means that we actually
    terminate after that state; this makes it so that geometric sum formulas with gamma=1
    give you the correct answer"""
    
    # assumes (a, s, s)
    _, _, n_states = T.shape

    # find terminal states
    def is_terminal(s):
        return np.allclose(T[:, s, s], 1.0) \
            and np.allclose(T[:, s, :s], 0.0) \
            and np.allclose(T[:, s, (s+1):], 0.0)

    Tnew = np.copy(T)
    for s in range(n_states):
        if is_terminal(s):
            Tnew[:, s, :] = 0
    return Tnew




