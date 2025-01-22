from functools import partial

import chex
import jax
import jax.numpy as jnp
from jax import random

from grl.environment.jax_pomdp import load_jax_pomdp, POMDP
from grl.environment.policy_lib import switching_two_thirds_right_policy

from scripts.td_variance import get_variances


def make_collect_samples(env_str: str, env: POMDP,
                         n_samples: int = int(100000),
                         n_envs: int = 1,
                         epsilon: float = 0.
                         ):
    env_params = env.default_params

    def get_policy(env_str: str = 'tmaze_two_thirds_up',
                   epsilon: float = 0.):

        def tabular_policy(rng: chex.PRNGKey, obs: jnp.ndarray,
                           pi_arr: jnp.ndarray, epsilon: float = 0.0) -> chex.Array:
            """
            obs: n_obs shape one-hot array.
            pi_arr: n_obs x n_actions shape policy array.
            :return: sampled action
            """
            pi_o = obs @ pi_arr
            nonzero = pi_o > 0
            epsilon_proportion = epsilon / nonzero.sum()
            pi_taken_away_epsilon = pi_o - (nonzero * epsilon_proportion)
            eps_pi = pi_taken_away_epsilon + epsilon / pi_o.shape[0]
            return random.choice(rng, pi_o.shape[0], p=eps_pi)

        if env_str == 'tmaze_5_two_thirds_up':
            pi_arr = jnp.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [2 / 3, 1 / 3, 0, 0], [1, 0, 0, 0]])
            return partial(tabular_policy, pi_arr=pi_arr, epsilon=epsilon), pi_arr
        elif env_str == 'parity_check_two_thirds_up':
            pi_arr = jnp.array([
                        [0, 0, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 1, 0],
                        [2 / 3, 1 / 3, 0, 0],
                        [1, 0, 0, 0]])
            return partial(tabular_policy, pi_arr=pi_arr, epsilon=epsilon), pi_arr
        elif env_str == 'switching':
            pi_arr = switching_two_thirds_right_policy()
            return partial(tabular_policy, pi_arr=pi_arr, epsilon=epsilon), pi_arr
        elif env_str == 'counting_wall':
            raise NotImplementedError
        raise NotImplementedError

    policy, pi = get_policy(env_str, epsilon=epsilon)
    vmap_policy = jax.vmap(policy)

    def env_step(runner_state, step):
        env_state, last_obs, last_done, rng = runner_state

        policy_rng, env_rng, rng = jax.random.split(rng, 3)

        policy_rngs = jax.random.split(rng, n_envs)
        action = vmap_policy(policy_rngs, last_obs)

        env_rngs = jax.random.split(rng, n_envs)
        obs, env_state, reward, done, info = env.step(env_rngs, env_state, action, env_params)
        experience = {
            'obs': last_obs,
            'action': action,
            'reward': reward,
            'done': done,
            'state': env_state.env_state
        }
        runner_state = (env_state, obs, done, rng)

        return runner_state, experience

    def collect(rng):
        reset_rng, rng = jax.random.split(rng)
        reset_rngs = jax.random.split(rng, n_envs)
        obs, env_state = env.reset(reset_rngs, env_params)

        scan_rng, rng = random.split(rng)
        init_runner_state = (env_state, obs, jnp.zeros((n_envs,), dtype=bool), scan_rng)

        final_runner_state, experiences = jax.lax.scan(
            env_step, init_runner_state, None, n_samples
        )

        # TODO: add a while not done here so that the end of the buffer is always on a done.

        if n_envs > 1:
            raise NotImplementedError
        else:
            experiences = jax.tree_util.tree_map(lambda x: x[:, 0], experiences)

        def get_disc_returns(next_return, idx):
            done, reward = experiences['done'][idx], experiences['reward'][idx]
            g = reward
            g += (1 - done) * pomdp.gamma * next_return
            return g, g

        init_runner_state = 0

        _, disc_returns = jax.lax.scan(
            get_disc_returns, init_runner_state, jnp.flip(jnp.arange(n_samples)), n_samples
        )
        experiences['g'] = jnp.flip(disc_returns)
        info = {
            'pi': pi
        }
        return experiences, info
    return collect


if __name__ == "__main__":
    # jax.disable_jit(True)
    env_str = 'tmaze_5_two_thirds_up'
    # env_str = 'counting_wall'
    # env_str = 'switching'

    rng = jax.random.PRNGKey(2024)

    pomdp = load_jax_pomdp(env_str)

    collect_fn = make_collect_samples(env_str, pomdp)

    collect_rng, rng = jax.random.split(rng)

    experiences, info = collect_fn(collect_rng)

    # TODO: calc variance over...
    # State, observation (MC), observation (TD). Will probably have to run TD model separately
    # vars_o

    variances = get_variances(info['pi'], pomdp)

