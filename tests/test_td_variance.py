from functools import partial

import chex
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from grl.environment import load_pomdp
from grl.environment.jax_pomdp import load_jax_pomdp, POMDP, LogWrapper, VecEnv
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
        elif env_str == 'tmaze_5_separate_goals_two_thirds_up':
            pi_arr = jnp.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [2 / 3, 1 / 3, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
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
        state = env_state.env_state

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
            'state': state
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
            r_2 = (reward ** 2) + 2 * (1 - done) * pomdp.gamma * reward * next_return
            return g, (g, r_2)

        init_runner_state = 0

        _, (disc_returns, r_2) = jax.lax.scan(
            get_disc_returns, init_runner_state, jnp.flip(jnp.arange(n_samples)), n_samples
        )
        experiences['g'] = jnp.flip(disc_returns)
        experiences['r_2'] = jnp.flip(r_2)
        # experiences['g'] = disc_returns
        info = {
            'pi': pi
        }
        return experiences, info
    return collect


if __name__ == "__main__":
    # jax.disable_jit(True)
    env_str = 'tmaze_5_separate_goals_two_thirds_up'
    # env_str = 'counting_wall'
    # env_str = 'switching'

    n_samples = int(1e6)

    rng = jax.random.PRNGKey(2024)

    # Load POMDPs
    pomdp = load_jax_pomdp(env_str)

    # collect_fn = make_collect_samples(env_str, pomdp, n_samples=int(1e7))
    collect_fn = make_collect_samples(env_str, pomdp, n_samples=n_samples)

    collect_rng, rng = jax.random.split(rng)

    mc_experiences, info = collect_fn(collect_rng)

    def calc_vars_from_experience(experiences, pomdp):
        states = experiences['state']
        one_hot_states = jnp.zeros((states.shape[0], pomdp.T.shape[-1]))
        one_hot_states = one_hot_states.at[jnp.arange(states.shape[0]), states].set(1)
        one_hot_obs = experiences['obs']

        returns = experiences['g']
        squared_returns = returns**2

        all_one_hot_obs_returns = returns[..., None] * experiences['obs']
        all_one_hot_state_returns = returns[..., None] * one_hot_states
        all_one_hot_state_r_2 = experiences['r_2'][..., None] * one_hot_states

        def one_hots_to_counts(one_hots: jnp.ndarray, axis: int = 0):
            counts = one_hots.sum(axis=axis)
            counts += (counts == 0)
            return counts

        obs_counts = one_hots_to_counts(one_hot_obs)
        state_counts = one_hots_to_counts(one_hot_states)

        obs_squared_returns = squared_returns[..., None] * one_hot_obs
        state_squared_returns = squared_returns[..., None] * one_hot_states

        state_returns = all_one_hot_state_returns.sum(axis=0) / state_counts
        obs_returns = all_one_hot_obs_returns.sum(axis=0) / obs_counts
        state_r_2 = all_one_hot_state_r_2.sum(axis=0) / state_counts

        state_sampled_var = (one_hot_states * ((all_one_hot_state_returns - state_returns[None, ...]) ** 2)).sum(axis=0) / state_counts
        obs_sampled_var = (one_hot_obs * ((all_one_hot_obs_returns - obs_returns[None, ...]) ** 2)).sum(axis=0) / obs_counts

        state_second_moment = state_squared_returns.sum(axis=0) / state_counts
        obs_second_moment = obs_squared_returns.sum(axis=0) / obs_counts
        results = {
            'state_r_2': state_r_2,
            'obs_sampled_var': obs_sampled_var,
            'obs_second_moment': obs_second_moment,
            'state_sampled_var': state_sampled_var,
            'state_second_moment': state_second_moment,
        }
        return results

    mc_var_results = calc_vars_from_experience(mc_experiences, pomdp)

    pomdp, pi_dict = load_pomdp(env_str,
                                corridor_length=5,
                                discount=0.)
    variances, info = get_variances(info['pi'], pomdp)

    td_mdp_spec = info['td_model']
    td_mdp = POMDP(td_mdp_spec.T, td_mdp_spec.R, td_mdp_spec.p0, td_mdp_spec.gamma,
                   np.eye(td_mdp_spec.state_space.n))
    td_env = LogWrapper(td_mdp, gamma=td_mdp.gamma)
    td_env = VecEnv(td_env)

    collect_td_fn = make_collect_samples(env_str, td_env, n_samples=n_samples)

    collect_rng, rng = jax.random.split(rng)

    td_experiences, td_info = collect_td_fn(collect_rng)
    td_var_results = calc_vars_from_experience(td_experiences, td_env)

    print()

