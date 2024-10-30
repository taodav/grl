import chex
import jax
from jax import random
import jax.numpy as jnp

from grl.utils.loss import discrep_loss
from grl.mdp import POMDP, MDP
from grl.environment import load_pomdp


def random_r_o_fn(rng: chex.PRNGKey, pomdp: POMDP) -> jnp.ndarray:
    """
    This function makes a random reward function in observation space (O),
    repeats it over action size (O x A), then projects this using a left multiply
    of \phi to the state space (S x A).

    We sample random numbers as per
    https://en.wikipedia.org/wiki/Random_projection#More_computationally_efficient_random_projections
    """
    obs_size = pomdp.observation_space.n
    choices = jnp.array([-1, 0, 1])
    p = jnp.array([1/6, 2/3, 1/6])
    r_o = random.choice(rng, choices, shape=(obs_size,), p=p)

    r_o_a = r_o[..., None].repeat(pomdp.action_space.n, axis=-1)
    r_a_s = jnp.einsum('so,oa->sa', pomdp.phi, r_o_a).T
    r_a_s_s = r_a_s[..., None].repeat(pomdp.state_space.n, axis=-1)

    return r_a_s_s


def random_r_o_a_fn(rng: chex.PRNGKey, pomdp: POMDP) -> jnp.ndarray:
    """
    This function makes a random reward function in observation-action space (O x A),
    then projects this using a left multiply of \phi to the state space (S x A).
    We sample random numbers as per
    https://en.wikipedia.org/wiki/Random_projection#More_computationally_efficient_random_projections
    """
    obs_size = pomdp.observation_space.n
    action_size = pomdp.action_space.n
    choices = jnp.array([-1, 0, 1])
    p = jnp.array([1/6, 2/3, 1/6])
    r_o_a = random.choice(rng, choices, shape=(obs_size, action_size), p=p)

    r_a_s = jnp.einsum('so,oa->sa', pomdp.phi, r_o_a).T
    r_a_s_s = r_a_s[..., None].repeat(pomdp.state_space.n, axis=-1)

    return r_a_s_s


def random_r_s_fn(rng: chex.PRNGKey, pomdp: POMDP) -> jnp.ndarray:
    """
    This function makes a random reward function in observation space (O),
    repeats it over action size (O x A), then projects this using a left multiply
    of \phi to the state space (S x A).
    We sample random numbers as per
    https://en.wikipedia.org/wiki/Random_projection#More_computationally_efficient_random_projections
    """
    obs_size = pomdp.state_space.n
    choices = jnp.array([-1, 0, 1])
    p = jnp.array([1/6, 2/3, 1/6])
    r_s = random.choice(rng, choices, shape=(obs_size,), p=p)

    r_a_s = r_s[None, ...].repeat(pomdp.action_space.n, axis=0)
    r_a_s_s = r_a_s[..., None].repeat(pomdp.state_space.n, axis=-1)

    return r_a_s_s


if __name__ == "__main__":
    # spec = 'tmaze_5_two_thirds_up'
    specs = ['parity_check']
    # TODO: sample policies for these. Check Cam's script for how he does this.
    # specs = [
    #     'ld_zero_by_mdp',
    #     'ld_zero_by_k_equality',
    # ]
    n_reward_functions = 30
    seed = 2027

    rng = random.PRNGKey(seed=seed)
    all_sampled_lds = {}

    for spec in enumerate(specs):
        pomdp, info = load_pomdp(spec)
        if spec == 'parity_check':
            pi = jnp.array([
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [2/3, 1/3],
                [0, 1],
            ])
        else:
            pi = info['Pi_phi'][0]

        # vmapped_random_reward_fn = jax.vmap(random_obs_reward_fn, in_axes=[0, None])
        # vmapped_random_reward_fn = jax.vmap(random_r_s_fn, in_axes=[0, None])
        vmapped_random_reward_fn = jax.vmap(random_r_o_a_fn, in_axes=[0, None])
        vmapped_discrep_loss = jax.vmap(discrep_loss, in_axes=[None, 0])

        rngs = random.split(rng, n_reward_functions + 1)
        r_rngs, rng = rngs[:-1], rngs[-1]
        random_rs = vmapped_random_reward_fn(r_rngs, pomdp)

        multi_reward_pomdp = POMDP(
            MDP(
                pomdp.T[None, ...].repeat(n_reward_functions, axis=0),
                random_rs,
                pomdp.p0[None, ...].repeat(n_reward_functions, axis=0),
                jnp.array([pomdp.gamma] * n_reward_functions),
                pomdp.terminal_mask[None, ...].repeat(n_reward_functions, axis=0)),
            pomdp.phi[None, ...].repeat(n_reward_functions, axis=0))
        lds, mc_vals, td_vals = vmapped_discrep_loss(pi, multi_reward_pomdp)
        all_sampled_lds[spec] = lds

    print()
