import jax.numpy as jnp
from jax import random
import jax
from jax.config import config
import haiku as hk
from grl.mdp import one_hot
import numpy as np

config.update('jax_platform_name', 'cpu')

from grl import MDP, AbstractMDP, environment
from grl.baselines import DQNArgs, LSTMReinforceAgent, train_reinforce_agent, create_managed_lstm_func


def test_reinforce_chain_pomdp():
    chain_length = 10
    spec = environment.load_spec('po_simple_chain', memory_id=None)
    n_eps = 1e4
    n_hidden = 1

    print(f"Testing LSTM with REINFORCE on Simple Chain MDP over {n_eps} episodes")
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    pomdp = AbstractMDP(mdp, spec['phi'])

    #ground_truth_vals = spec['gamma']**jnp.arange(chain_length - 2, -1, -1)

    _lstm_func = create_managed_lstm_func(n_hidden, pomdp.n_actions)
    
    transformed = hk.without_apply_rng(hk.transform(_lstm_func))

    rand_key = random.PRNGKey(2023)
    rand_key, subkey = random.split(rand_key)
    agent_args = DQNArgs((pomdp.n_obs,), 
                         pomdp.n_actions, 
                         pomdp.gamma, 
                         subkey, 
                         algo = "reinforce", 
                         trunc_len=chain_length, 
                         alpha=0.01)
    agent = LSTMReinforceAgent(transformed, n_hidden, agent_args)

    info, agent = train_reinforce_agent(pomdp, agent, n_eps)


    test_batch = jnp.array([[[1.],
                                [1.],
                                [1.],
                                [1.],
                                [1.],
                                [1.],
                                [1.],
                                [1.],
                                [1.]]])
    last_policy = info['final_pi']
    last_policy_logits = agent.policy_logits(agent.network_params, agent.get_initial_hidden_state(), test_batch)[0]

    print(f"Calculated policy: {last_policy}\n"
          f"Logits: {last_policy_logits}\n")
    assert jnp.all(jnp.isclose(last_policy, test_batch[0], atol=0.0001))

def test_reinforce_5len_tmaze():
    spec = environment.load_spec("tmaze_5_two_thirds_up")
    trunc_len = 100
    n_hidden = 12

    n_eps = 1e4

    print(f"Testing LSTM with REINFORCE on 5-length T-maze over {n_eps} episodes with trunc length {trunc_len}")
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    pomdp = AbstractMDP(mdp, spec['phi'])

    _lstm_func = create_managed_lstm_func(n_hidden, pomdp.n_actions)
    
    transformed = hk.without_apply_rng(hk.transform(_lstm_func))

    rand_key = random.PRNGKey(2023)
    rand_key, subkey = random.split(rand_key)
    agent_args = DQNArgs((pomdp.n_obs,), 
                         pomdp.n_actions, 
                         pomdp.gamma, 
                         subkey, 
                         algo = "reinforce", 
                         trunc_len=trunc_len, 
                         alpha=0.001)
    agent = LSTMReinforceAgent(transformed, n_hidden, agent_args)

    train_logs, agent_args = train_reinforce_agent(pomdp, agent, n_eps)

    assert train_logs["final_pct_success"] >= 0.95
    assert train_logs["avg_len"][-1] < 10.


if __name__ == "__main__":
    #test_reinforce_chain_pomdp()
    test_reinforce_5len_tmaze()
