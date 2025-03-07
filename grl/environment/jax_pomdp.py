from functools import partial
from typing import Optional, Tuple, Union

import chex
from flax import struct
import jax
from jax import random, jit
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from grl.environment.spec import load_spec

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env
        if hasattr(env, '_unwrapped'):
            self._unwrapped = env._unwrapped
        else:
            self._unwrapped = env

    @property
    def unwrapped(self):
        return self._unwrapped
    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    discounted_episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_discounted_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment, gamma: float = 0.99):
        super().__init__(env)
        self.gamma = gamma

    @partial(jax.jit, static_argnums=(0, -1))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, -1))
    def step(
            self,
            key: chex.PRNGKey,
            state: environment.EnvState,
            action: Union[int, float],
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_discounted_episode_return = state.discounted_episode_returns + (self.gamma ** state.episode_lengths) * reward
        new_episode_length = state.episode_lengths + 1
        # TODO: add discounted_episode_returns here.
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            discounted_episode_returns=new_discounted_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
                                     + new_episode_return * done,
            returned_discounted_episode_returns=state.returned_discounted_episode_returns * (1 - done)
                                                + new_discounted_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
                                     + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_discounted_episode_returns"] = state.returned_discounted_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        info["reward"] = reward
        return obs, state, reward, done, info


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


class POMDP(environment.Environment):
    def __init__(self,
                 T: jnp.ndarray,
                 R: jnp.ndarray,
                 p0: jnp.ndarray,
                 gamma: float,
                 phi: jnp.ndarray,
                 fully_observable: bool = False):
        self.gamma = jnp.array(gamma)
        self.T = jnp.array(T)
        self.R = jnp.array(R)
        self.phi = jnp.array(phi)

        self.p0 = jnp.array(p0)
        self.fully_observable = fully_observable

    def observation_space(self, params: environment.EnvParams):
        if self.fully_observable:
            return spaces.Box(0, 1, (self.T.shape[-1],))
        return spaces.Box(0, 1, (self.phi.shape[-1],))

    def action_space(self, params: environment.EnvParams):
        return spaces.Discrete(self.T.shape[0])

    @property
    def default_params(self) -> environment.EnvParams:
        return environment.EnvParams(max_steps_in_episode=1000)

    def get_obs(self, key: chex.PRNGKey, s: jnp.ndarray):
        if self.fully_observable:
            n_states = self.T.shape[-1]
            obs = jnp.zeros(n_states)
            obs = obs.at[s].set(1)
            return obs

        n_obs = self.phi[s].shape[0]

        observed_idx = random.choice(key, n_obs, p=self.phi[s])
        obs = jnp.zeros(n_obs)
        obs = obs.at[observed_idx].set(1)
        return obs

    @partial(jit, static_argnums=(0, ))
    def reset_env(self, key: chex.PRNGKey, params: environment.EnvParams):
        obs_key, init_key = random.split(key)
        state = random.choice(init_key, self.p0.shape[0], p=self.p0)
        return self.get_obs(obs_key, state), state

    @partial(jit, static_argnums=(0, -2))
    def step_env(self,
                 key: chex.PRNGKey,
                 state: jnp.ndarray,
                 action: int,
                 params: environment.EnvParams):
        pr_next_s = self.T[action, state, :]

        next_state_key, obs_key = random.split(key)
        next_state = random.choice(next_state_key, pr_next_s.shape[0], p=pr_next_s)

        reward = self.R[action, state, next_state]

        # Check if next_state is absorbing state
        is_absorbing = (self.T[:, next_state, next_state] == 1)
        terminal = is_absorbing.all() # absorbing for all actions
        observation = self.get_obs(obs_key, next_state)

        return observation, next_state, reward, terminal, {}


def load_jax_pomdp(name: str, fully_observable: bool = False):
    spec = load_spec(name)
    pomdp = POMDP(spec['T'], spec['R'], spec['p0'], spec['gamma'], spec['phi'],
                     fully_observable=fully_observable)

    env = LogWrapper(pomdp, gamma=spec['gamma'])

    # Vectorize our environment
    env = VecEnv(env)
    return env
