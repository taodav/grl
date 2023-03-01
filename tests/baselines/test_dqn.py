import jax.numpy as jnp
from jax import random
import jax
from jax.config import config
import haiku as hk

config.update('jax_platform_name', 'cpu')

from grl import MDP, environment
from grl.baselines.dqn_agent import DQNAgent, train_dqn_agent

class SimpleNN(hk.Module):
    def __init__(self, input_size, output_size, name='basic_mlp'):
        super().__init__(name=name)
        self._internal_linear_1 = hk.nets.MLP([input_size, 30, output_size], 
                                              w_init=hk.initializers.RandomUniform(), 
                                              b_init=hk.initializers.RandomUniform(),
                                              name='hk_linear')

    def __call__(self, x):
        return self._internal_linear_1(x)


def test_sarsa_chain_mdp():
    chain_length = 10
    spec = environment.load_spec('simple_chain', memory_id=None)
    n_steps = 2000 * chain_length

    print(f"Testing SARSA on Simple Chain MDP over {n_steps} steps")
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])

    ground_truth_vals = spec['gamma']**jnp.arange(chain_length - 2, -1, -1)

    def _nn_func(x):
        module = SimpleNN(mdp.n_states, mdp.n_actions)
        return module(x)
    transformed = hk.without_apply_rng(hk.transform(_nn_func))

    rand_key = random.PRNGKey(2023)
    rand_key, subkey = random.split(rand_key)
    agent = DQNAgent(transformed, (mdp.n_states,), mdp.n_actions, mdp.gamma, subkey, algo = "sarsa")

    agent = train_dqn_agent(mdp, agent, 20000)


    v = jnp.array([jnp.sum(agent.Qs(jax.nn.one_hot(jnp.array([s]), mdp.n_states), agent.network_params)) for s in range(mdp.n_states)])

    print(f"Calculated values: {v[:-1]}\n"
          f"Ground-truth values: {ground_truth_vals}")
    assert jnp.all(jnp.isclose(v[:-1], ground_truth_vals, atol=1e-2))

