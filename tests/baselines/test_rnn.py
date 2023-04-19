import jax.numpy as jnp
from jax import random
import jax
from jax.config import config
import haiku as hk
from grl.mdp import one_hot
import numpy as np

config.update('jax_platform_name', 'cpu')

from grl import MDP, AbstractMDP, environment
from grl.baselines import DQNArgs, ManagedLSTM
from grl.baselines.rnn_agent import LSTMAgent, train_rnn_agent


def test_lstm_chain_pomdp():
    chain_length = 10
    spec = environment.load_spec('po_simple_chain', memory_id=None)
    n_eps = 6.5e4
    n_hidden = 1

    print(f"Testing LSTM with Sequential SARSA on Simple Chain MDP over {n_eps * chain_length} steps")
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    pomdp = AbstractMDP(mdp, spec['phi'])
    print(pomdp.n_obs)

    ground_truth_vals = spec['gamma']**jnp.arange(chain_length - 2, -1, -1)

    def _lstm_func(x: jnp.ndarray, h: hk.LSTMState):
        module = ManagedLSTM(n_hidden, pomdp.n_actions)
        return module(x, h)
    
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
    agent = LSTMAgent(transformed, n_hidden, agent_args)

    _, agent = train_rnn_agent(pomdp, agent, n_eps, zero_obs=False)


    test_batch = jnp.array([[[1.],
                                [1.],
                                [1.],
                                [1.],
                                [1.],
                                [1.],
                                [1.],
                                [1.],
                                [1.]]])
    v = jnp.sum(agent.Qs(test_batch, agent.get_initial_hidden_state(), agent.network_params)[0], axis=-1)

    print(f"Calculated values: {v[0]}\n"
          f"Ground-truth values: {ground_truth_vals}")
    assert jnp.all(jnp.isclose(v[0], ground_truth_vals, atol=0.02))


def test_lstm_5len_tmaze():
    spec = environment.load_spec("tmaze_5_two_thirds_up")
    trunc_len = 100
    n_hidden = 12

    n_eps = 1.5e5

    print(f"Testing LSTM with Sequential SARSA on 5-length T-maze over {n_eps} episodes with trunc length {trunc_len}")
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    pomdp = AbstractMDP(mdp, spec['phi'])

    def _lstm_func(x: jnp.ndarray, h: hk.LSTMState):
        module = ManagedLSTM(n_hidden, pomdp.n_actions)
        return module(x, h)
    
    transformed = hk.without_apply_rng(hk.transform(_lstm_func))

    rand_key = random.PRNGKey(2023)
    rand_key, subkey = random.split(rand_key)
    agent_args = DQNArgs((pomdp.n_obs,), 
                         pomdp.n_actions, 
                         pomdp.gamma, 
                         subkey, 
                         algo = "sarsa", 
                         trunc_len=trunc_len, 
                         alpha=0.001, 
                         epsilon=0.1,
                         epsilon_start=1.,
                         anneal_steps=n_eps / 2)
    agent = LSTMAgent(transformed, n_hidden, agent_args)

    train_logs, agent_args = train_rnn_agent(pomdp, agent, n_eps)

    assert train_logs["final_pct_success"] >= 0.95
    assert train_logs["avg_len"][-1] < 10.

    
if __name__ == "__main__":
    test_lstm_chain_pomdp()
    test_lstm_5len_tmaze()
