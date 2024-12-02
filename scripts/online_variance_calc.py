"""
Make our online variance calculations for sampled GAE(\lambda) returns
and memory-augmented value.
"""
import chex
from gymnax.environments import environment, spaces
import jax
import jax.numpy as jnp

from grl.environment.jax_pomdp import POMDP
from grl.utils.policy_eval import lstdq_lambda

def make_runner(env: environment.Environment, pi: jnp.ndarray,
                n: int = 128,
                lambda_: float = 0.99):
    """
    n here is the n-step return.
    lambda_ is for GAE
    """

    lambda_v_vals, lambda_q_vals, _ = lstdq_lambda(pi, pomdp, lambda_=lambda_)


    # INIT ENV
    rng, _rng = jax.random.split(rng)
    obsv, env_state = env.reset(reset_rng, env_params)
    mem_state = jax.array(0)

    def runner(rng: chex.PRNGKey, mem_params: jnp.ndarray):
        pass

    return runner