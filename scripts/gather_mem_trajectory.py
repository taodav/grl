"""
Make our online variance calculations for sampled GAE(\lambda) returns
and memory-augmented value.
"""
from pathlib import Path
from typing import NamedTuple

import chex
from flax.training import orbax_utils
from gymnax.environments import environment, spaces
import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import orbax.checkpoint

from definitions import ROOT_DIR
from grl.environment import load_spec
from grl.environment.jax_pomdp import POMDP
from grl.environment.policy_lib import switching_two_thirds_right_policy
from grl.memory.lib import switching_optimal_deterministic_1_bit_mem
from grl.utils import reverse_softmax
from grl.utils.file_system import numpyify


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
        gamma = env.gamma

        # O x A x M x M
        mem_func = jax.nn.softmax(mem_params, axis=-1)
        n_mem = mem_func.shape[-1]
        mem_state = jnp.array(0)

        n_obs = env.observation_space(env_params).n

        assert pi.shape[0] == (n_obs * n_mem)
        jax.debug.print(f"Collecting {n} environment steps")
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

        jax.debug.print(f"Calculating returns")
        # @scan_tqdm(n)
        def _calc_return_step(next_ret, x):
            _, (reward, done) = x
            ret = reward + gamma * (1 - done) * next_ret
            return ret, ret

        _, returns = jax.lax.scan(
            _calc_return_step, jnp.zeros_like(traj_batch.reward[-1]),
            (jnp.arange(n), (traj_batch.reward, traj_batch.done)), n,
            reverse=True
        )

        return traj_batch, returns

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
    # jax.disable_jit(True)
    env_str = 'switching'
    seed = 2024
    # n_samples = int(1e6)
    n_samples = int(1e6)
    res_dir = Path(ROOT_DIR, 'results', f'{env_str}_returns_dataset')

    rng = jax.random.PRNGKey(seed)

    env = load_jax_pomdp(env_str)

    mem_fn = switching_optimal_deterministic_1_bit_mem()
    pi = switching_two_thirds_right_policy()

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

    runner_fn = jax.jit(make_runner(env, mem_aug_pi, n_samples))

    rng, _rng = jax.random.split(rng)
    dataset, returns = runner_fn(_rng, eps_mem_params)
    res = {
        'dataset': dataset,
        'returns': returns
    }

    res = jax.tree.map(numpyify, res)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(res)

    print(f"Saving results to {res_dir}")
    orbax_checkpointer.save(res_dir, res, save_args=save_args)



