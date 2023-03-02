import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from typing import Union

@register_pytree_node_class
class JaxBatch:
    """
    Batch of data for training JAX-based NN agents on an MDP/AMDP.
    Meant to take in (one-hot encoded) states, actions, next_states, etc. directly from the MDP
    and convert into JAX representation internally.
    """
    def __init__(self, 
                 obs: Union[jnp.ndarray, list[jnp.ndarray]] = [],
                 actions: Union[jnp.ndarray, list[int]] = [],
                 next_obs: Union[jnp.ndarray, list[jnp.ndarray]] = [],
                 terminals: Union[jnp.ndarray, list[bool]] = [],
                 rewards: Union[jnp.ndarray, list[float]] = [],
                 next_actions: Union[jnp.ndarray, list[int]] = []) -> None:
        # (b x num_observations)
        self.obs = jnp.array(obs)
        # (b x 1)
        self.actions = jnp.array(actions, dtype=jnp.int32)
        # (b x num_observations)
        self.next_obs = jnp.array(next_obs)
        # (b x 1)
        self.terminals = jnp.array(terminals)
        # (b x 1)
        self.rewards = jnp.array(rewards)
        # (b x 1)
        self.next_actions = jnp.array(next_actions, dtype=jnp.int32)

    def tree_flatten(self):
        children = (self.obs, self.actions, self.next_obs, self.terminals, self.rewards, self.next_actions)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization
        obj = object.__new__(JaxBatch)
        obj.obs = children[0]
        obj.actions = children[1]
        obj.next_obs = children[2]
        obj.terminals = children[3]
        obj.rewards = children[4]
        obj.next_actions = children[5]
        return obj
    