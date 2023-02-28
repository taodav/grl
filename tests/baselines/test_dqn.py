import jax.numpy as jnp
from jax import random
from jax.config import config
import haiku as hk

config.update('jax_platform_name', 'cpu')

from grl import MDP, environment
from grl.baselines.dqn_agent import train_dqn_agent

class SimpleNN(hk.Module):
    def __init__(self, output_size, name='basic_mlp'):
        super().__init__(name=name)
        self._internal_linear_1 = hk.nets.MLP([1, 3, output_size], name='hk_linear')

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
        module = SimpleNN(mdp.n_actions)
        return module(x)
    transformed = hk.without_apply_rng(hk.transform(_nn_func))

    trained_agent = train_dqn_agent(mdp, transformed, n_steps, random.PRNGKey(2023), algo = "sarsa")

    print(jnp.array([trained_agent.Qs(jnp.array([s]), trained_agent.network_params) for s in range(mdp.n_states)]))

    v = jnp.array([jnp.sum(trained_agent.Qs(jnp.array([s]), trained_agent.network_params)) for s in range(mdp.n_states)])

    print(f"Calculated values: {v[:-1]}\n"
          f"Ground-truth values: {ground_truth_vals}")
    assert jnp.all(jnp.isclose(v[:-1], ground_truth_vals, atol=1e-2))

