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
from grl.baselines.reinforce import LSTMReinforceAgent, train_reinforce_agent


def test_reinforce_chain_pomdp():
    chain_length = 10
    spec = environment.load_spec('po_simple_chain', memory_id=None)
    n_eps = 1e4
    n_hidden = 1

    print(f"Testing LSTM with REINFORCE on Simple Chain MDP over {n_eps} episodes")
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    pomdp = AbstractMDP(mdp, spec['phi'])

    #ground_truth_vals = spec['gamma']**jnp.arange(chain_length - 2, -1, -1)

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




if __name__ == "__main__":
    test_reinforce_chain_pomdp()
