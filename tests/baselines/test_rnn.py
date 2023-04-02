import jax.numpy as jnp
from jax import random
import jax
from jax.config import config
import haiku as hk
from grl.mdp import one_hot
import numpy as np

config.update('jax_platform_name', 'cpu')

from grl import MDP, AbstractMDP, environment
from grl.baselines.dqn_agent import DQNArgs
from grl.baselines.rnn_agent import RNNAgent, train_rnn_agent


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


def test_lstm_chain_pomdp():
    chain_length = 10
    spec = environment.load_spec('po_simple_chain', memory_id=None)
    n_steps = 1e6 * chain_length

    print(f"Testing LSTM with Sequential SARSA on Simple Chain MDP over {n_steps} steps")
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    pomdp = AbstractMDP(mdp, spec['phi'])
    print(pomdp.n_obs)

    ground_truth_vals = spec['gamma']**jnp.arange(chain_length - 2, -1, -1)

    def _lstm_func(x: jnp.ndarray):
        module = SimpleGRU(1, pomdp.n_actions)
        return module(x)
    
    transformed = hk.without_apply_rng(hk.transform(_lstm_func))

    rand_key = random.PRNGKey(2023)
    rand_key, subkey = random.split(rand_key)
    agent_args = DQNArgs((pomdp.n_obs,), 
                         pomdp.n_actions, 
                         pomdp.gamma, 
                         subkey, 
                         algo = "sarsa", 
                         trunc_len=chain_length, 
                         alpha=0.01)
    agent = RNNAgent(transformed, agent_args)

    agent = train_rnn_agent(pomdp, agent, n_steps, zero_obs=False)


    test_batch = jnp.array([[[1.],
                                [1.],
                                [1.],
                                [1.],
                                [1.],
                                [1.],
                                [1.],
                                [1.],
                                [1.]]])
    v = jnp.sum(agent.Qs(test_batch, agent.network_params), axis=-1)

    print(f"Calculated values: {v[0]}\n"
          f"Ground-truth values: {ground_truth_vals}")
    assert jnp.all(jnp.isclose(v[0], ground_truth_vals, atol=0.05))


def test_lstm_5len_tmaze():
    spec = environment.load_spec("tmaze_5_two_thirds_up")
    trunc_len = 10

    n_steps = 1e7

    print(f"Testing LSTM with Sequential SARSA on 5-length T-maze over {n_steps} steps with trunc length {trunc_len}")
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    pomdp = AbstractMDP(mdp, spec['phi'])
    print(pomdp.n_states)
    print(pomdp.n_obs)

    # TODO ground truths?
    #ground_truth_vals = spec['gamma']**jnp.arange(chain_length - 2, -1, -1)

    def _lstm_func(x: jnp.ndarray):
        module = SimpleGRU(12, pomdp.n_actions)
        return module(x)
    
    transformed = hk.without_apply_rng(hk.transform(_lstm_func))

    rand_key = random.PRNGKey(2023)
    rand_key, subkey = random.split(rand_key)
    agent_args = DQNArgs((pomdp.n_obs,), pomdp.n_actions, pomdp.gamma, subkey, algo = "sarsa", trunc_len=trunc_len, alpha=0.001)
    agent = RNNAgent(transformed, agent_args)

    agent = train_rnn_agent(pomdp, agent, n_steps)

    # TODO how to get ground truths here?
    # test_batch = jnp.array([[one_hot(0, pomdp.n_obs) for _ in range(trunc_len)]])
    # v = jnp.sum(agent.Qs(test_batch, agent.network_params), axis=-1)

    
    # print(f"Calculated values: {v[0][:-1]}\n"
    #       f"Ground-truth values: {ground_truth_vals}")
    # assert jnp.all(jnp.isclose(v[0][:-1], ground_truth_vals, atol=0.05))

if __name__ == "__main__":
    #test_lstm_chain_pomdp()
    test_lstm_5len_tmaze()

