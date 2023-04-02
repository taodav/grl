import numpy as np
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
                 obs: Union[np.ndarray, list[np.ndarray]] = [],
                 actions: Union[np.ndarray, list[int]] = [],
                 next_obs: Union[np.ndarray, list[np.ndarray]] = [],
                 terminals: Union[np.ndarray, list[bool]] = [],
                 rewards: Union[np.ndarray, list[float]] = [],
                 next_actions: Union[np.ndarray, list[int]] = []) -> None:
        args = [obs, actions, next_obs, terminals, rewards, next_actions]
        tmap = set(map(lambda x: type(x), args))
        assert len(tmap) == 1
        if tmap.pop() == np.ndarray:
            self.obs = obs
            self.actions = actions
            self.next_obs = next_obs
            self.terminals = terminals
            self.rewards = rewards
            self.next_actions = next_actions
        else:
            # (b x num_observations)
            self.obs = np.array(obs)
            # (b x 1)
            self.actions = np.array(actions, dtype=np.int32)
            # (b x num_observations)
            self.next_obs = np.array(next_obs)
            # (b x 1)
            self.terminals = np.array(terminals)
            # (b x 1)
            self.rewards = np.array(rewards)
            # (b x 1)
            self.next_actions = np.array(next_actions, dtype=np.int32)

    def tree_flatten(self):
        children = (self.obs, self.actions, self.next_obs, self.terminals, self.rewards, self.next_actions)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization
        obj = JaxBatch(*children)
        return obj
    
    def __str__(self) -> str:
        return f"""
            obs: {self.obs} \n
            actions: {self.actions} \n
            next_obs: {self.next_obs} \n
            terminals: {self.terminals} \n
            rewards: {self.rewards} \n
            next_actions: {self.next_actions} \n
        """
    