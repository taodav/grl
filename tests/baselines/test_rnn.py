import jax.numpy as jnp
from jax import random
import jax
from jax.config import config
import haiku as hk
from grl.mdp import one_hot

config.update('jax_platform_name', 'cpu')

from grl import MDP, environment
from grl.baselines.dqn_agent import DQNArgs
from grl.baselines.rnn_agent import RNNAgent, train_rnn_agent


def test_lstm_chain_mdp():
    chain_length = 10
    spec = environment.load_spec('simple_chain', memory_id=None)
    n_steps = 5e4 * chain_length

    print(f"Testing LSTM with Sequential SARSA on Simple Chain MDP over {n_steps} steps")
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])

    ground_truth_vals = spec['gamma']**jnp.arange(chain_length - 2, -1, -1)

    def _lstm_func(x: jnp.ndarray):
        """Unroll an LSTM over sequences of [B, T, F]"""
        lstm = hk.LSTM(1)
        batch_size = x.shape[0]
        outputs, cell_state = hk.dynamic_unroll(lstm, x, lstm.initial_state(batch_size), time_major=False)
        return hk.BatchApply(hk.Linear(mdp.n_actions))(outputs), cell_state
    
    transformed = hk.without_apply_rng(hk.transform(_lstm_func))

    rand_key = random.PRNGKey(2023)
    rand_key, subkey = random.split(rand_key)
    agent_args = DQNArgs((mdp.n_obs,), mdp.n_actions, mdp.gamma, subkey, algo = "sarsa", trunc_len=chain_length)
    agent = RNNAgent(transformed, agent_args)

    agent = train_rnn_agent(mdp, agent, n_steps)


    test_batch = jnp.array([[one_hot(s, mdp.n_obs) for s in range(mdp.n_obs)]])
    print(test_batch)
    print(test_batch.shape)
    test_batch_Qs = agent.Qs(test_batch, agent.network_params)
    print(test_batch_Qs)
    v = jnp.sum(agent.Qs(test_batch, agent.network_params), axis=-1)
    print(v)

    print(f"Calculated values: {v[0][:-1]}\n"
          f"Ground-truth values: {ground_truth_vals}")
    assert jnp.all(jnp.isclose(v[0][:-1], ground_truth_vals, atol=1e-2))

    if __name__ == "__main__":
        test_lstm_chain_mdp()

