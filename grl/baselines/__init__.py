from typing import Tuple
from dataclasses import dataclass
import jax.random as random
import haiku as hk
import numpy as np

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

class SimpleNN(hk.Module):
    def __init__(self, input_size, output_size, name='basic_mlp'):
        super().__init__(name=name)
        self._internal_linear_1 = hk.nets.MLP([input_size, output_size], 
                                              w_init=hk.initializers.RandomUniform(), 
                                              b_init=hk.initializers.RandomUniform(),
                                              name='hk_linear')

    def __call__(self, x):
        return self._internal_linear_1(x)


class ManagedLSTM(hk.Module):
    """
    LSTM that expects its hidden state to be managed by the calling class.
    """
    def __init__(self, hidden_size, output_size, name='managed_lstm'):
        super().__init__(name=name)
        init = hk.initializers.VarianceScaling(np.sqrt(2), 'fan_avg', 'uniform')
        b_init = hk.initializers.Constant(0)
        self._lstm = hk.LSTM(hidden_size)
        
        self._linear = hk.Linear(output_size, w_init=init, b_init = b_init)

    def __call__(self, x, h):
        # initial_state = jnp.zeros(initial_state.shape)
        outputs, cell_state = hk.dynamic_unroll(self._lstm, x, h, time_major=False)
        return hk.BatchApply(self._linear)(outputs), cell_state
    

# TODO currently unused
class SimpleGRU(hk.Module):
    def __init__(self, hidden_size, output_size, name='basic_gru'):
        super().__init__(name=name)
        init = hk.initializers.VarianceScaling(np.sqrt(2), 'fan_avg', 'uniform')
        b_init = hk.initializers.Constant(0)
        self._lstm = hk.GRU(hidden_size, w_i_init=init, 
                        w_h_init=init, 
                        b_init=b_init)
        
        self._linear = hk.Linear(output_size, w_init=init, b_init = b_init)
        

    def __call__(self, x):
        batch_size = x.shape[0]
        initial_state = self._lstm.initial_state(batch_size)
        # initial_state = jnp.zeros(initial_state.shape)
        outputs, cell_state = hk.dynamic_unroll(self._lstm, x, initial_state, time_major=False)
        return hk.BatchApply(self._linear)(outputs), cell_state
        