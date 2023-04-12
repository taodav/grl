from typing import Tuple
from dataclasses import dataclass
import jax.random as random

@dataclass
class DQNArgs:
    features_shape: Tuple[int]
    n_actions: int
    gamma: float
    rand_key: random.PRNGKey
    epsilon: float = 0.1
    epsilon_start: float = 1.
    anneal_steps: int = 0
    optimizer: str = "sgd"
    alpha: float = 0.01
    algo: str = "sarsa"
    trunc_len: int = None
    init_hidden_var = 0.