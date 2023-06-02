import numpy as np
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from dataclasses import dataclass, fields
from typing import Union, Iterable

@register_pytree_node_class
@dataclass
class Batch:
    obs: Union[np.ndarray, Iterable]
    action: Union[np.ndarray, Iterable]
    reward: Union[np.ndarray, Iterable] = None
    next_obs: Union[np.ndarray, Iterable] = None
    next_action: Union[np.ndarray, Iterable] = None
    prev_action: Union[np.ndarray, Iterable] = None
    done: Union[np.ndarray, Iterable] = None
    gamma: Union[np.ndarray, Iterable] = None
    state: Union[np.ndarray, Iterable] = None
    next_state: Union[np.ndarray, Iterable] = None
    end: Union[np.ndarray, Iterable] = None  # End is done or max_steps == timesteps

    # MC stuff
    returns: Union[np.ndarray, Iterable] = None

    # GRU stuff
    zero_mask: Union[np.ndarray, Iterable] = None

    def tree_flatten(self):
        children = tuple(getattr(self, field.name) for field in fields(self))
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # dict_children = {k: v for k, v in children}
        # return cls(**dict_children)
        return cls(*children)

def one_hot(x, n):
    return np.eye(n)[x]

def get_idx_from_nth_dim(arr: np.ndarray, idx: int, nth_dim: int):
    ix = tuple(idx if dim == nth_dim else slice(None) for dim in range(arr.ndim))
    return arr[ix]

def compress_episode_rewards(episode_reward: jnp.ndarray) -> dict:
    """
    Compresses a sequence of episode rewards (floats).
    Returns a dict with 'most_common_reward', 'episode_length' and 'compressed_rewards',
    where 'compressed_rewards' is a list of (t, r_t), whenever r_t != the most common reward.
    """
    most_common_reward = max(set(episode_reward), key=episode_reward.count)
    compressed_rewards = [(i, rew) for i, rew in enumerate(episode_reward) if rew != most_common_reward]
    return {'episode_length': len(episode_reward),
            'most_common_reward': most_common_reward,
            'compressed_rewards': compressed_rewards}
