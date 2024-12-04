"""
Make our online variance calculations for sampled GAE(\lambda) returns
and memory-augmented value.
"""
from typing import NamedTuple

import chex
from gymnax.environments import environment, spaces
import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm

from grl.environment import load_spec
from grl.environment.jax_pomdp import POMDP
from grl.environment.policy_lib import switching_two_thirds_right_policy
from grl.memory.lib import switching_optimal_deterministic_1_bit_mem
from grl.utils import reverse_softmax


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    mem: jnp.ndarray
    info: jnp.ndarray


def compute_n_step_returns(traj_batch: Transition, last_vals, gamma: float):

    def _step(prev_vals, runner):
        terminal, reward = runner
        vals = reward + gamma * (1 - terminal) * prev_vals
        return vals, vals

    _, returns = jax.lax.scan(
        _step, last_vals,
        (traj_batch.done, traj_batch.reward),
        reverse=True
    )
    return returns


def make_runner(env: environment.Environment,
                pi: jnp.ndarray,
                n: int = int(1e6)):
    """
    n here is the n-step return.
    assume pi is of shape (O * M) x A
    """

    def runner(rng: chex.PRNGKey, mem_params: jnp.ndarray):
        # INIT ENV
        env_params = env.default_params
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng, env_params)

        # O x A x M x M
        mem_func = jax.nn.softmax(mem_params, axis=-1)
        n_mem = mem_func.shape[-1]
        mem_state = jax.array(0)

        @scan_tqdm(n)
        def _env_step(runner_state, _):
            env_state, last_obs, last_mem, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            obs_idx = last_obs * n_mem + last_mem
            pi_dist = pi[obs_idx]
            action = jax.random.choice(_rng, pi_dist.shape[-1], p=pi_dist)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

            # STEP MEM
            rng, _rng = jax.random.split(rng)
            mem_dist = mem_func[obs, action, last_mem]
            mem = jax.random.choice(_rng, n_mem, p=mem_dist)

            transition = Transition(
                done, action, reward, last_obs, last_mem, info
            )

            runner_state = (env_state, obs, mem, rng)
            return runner_state, transition

        runner_state = (env_state, obs, mem_state, rng)
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, jnp.arange(n), n
        )
        return traj_batch

    return runner


def load_jax_pomdp(name: str, fully_observable: bool = False):
    spec = load_spec(name)
    pomdp = POMDP(spec['T'], spec['R'], spec['p0'], spec['gamma'], spec['phi'],
                     fully_observable=fully_observable)
    return pomdp


def epsilon_interpolate_deterministic_dist(deterministic_dist: jnp.ndarray, eps: float = 0.1):
    ones = jnp.isclose(deterministic_dist, 1.)
    zeros = jnp.isclose(deterministic_dist, 0.)
    assert jnp.all((ones + zeros).sum(axis=-1) == deterministic_dist.shape[-1]), \
        'Deterministic distribution not filled with only 0s and 1s'

    return deterministic_dist - eps * ones + eps * zeros


if __name__ == '__main__':
    env_str = 'switching'
    n_samples = int(1e6)

    env = load_jax_pomdp(env_str)

    mem_fn = switching_optimal_deterministic_1_bit_mem()
    pi = switching_two_thirds_right_policy()

    runner_fn = make_runner(env, pi, n_samples)

    # Now we instantiate our epsilon memory params
    mem_params = reverse_softmax(mem_fn)
    pi_params = reverse_softmax(pi)

    epsilons = jnp.linspace(0, 0.5, num=32)

    # TODO: vmap this!
    epsilon = 0.1
    eps_mem_fn = epsilon_interpolate_deterministic_dist(mem_fn, epsilon)
    eps_mem_params = reverse_softmax(eps_mem_fn)

    mem_aug_pi_params = pi_params.repeat(eps_mem_params.shape[-1], axis=0)
    mem_aug_pi = jax.nn.softmax(mem_aug_pi_params, axis=-1)



