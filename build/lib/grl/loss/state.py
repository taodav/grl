from functools import partial

import jax.numpy as jnp
from jax import jit, lax

from grl.memory import memory_cross_product
from grl.mdp import POMDP
from grl.utils.policy_eval import lstdq_lambda


@partial(jit, static_argnames=['error_type', 'residual'])
def mem_state_discrep(
        mem_params: jnp.ndarray,
        pi: jnp.ndarray,
        pomdp: POMDP, # non-state args
        alpha: float = 1.,
        value_type: str = 'q',
        error_type: str = 'l2',
        lambda_: float = 0.,
        residual: bool = False): # initialize static args
    n_mem = mem_params.shape[-1]
    assert n_mem == 2, "Haven't implemented n_mem > 2 for mem_state_discrep yet!"

    pomdp = memory_cross_product(mem_params, pomdp)
    lambda_v_vals, lambda_q_vals, info = lstdq_lambda(pi, pomdp, lambda_=lambda_)
    lambda_vals = {'v': lambda_v_vals, 'q': lambda_q_vals}
    vals = lambda_vals[value_type]

    occupancy = info['occupancy'] * (1 - pomdp.terminal_mask)

    c_o = occupancy @ pomdp.phi
    count_o = c_o / c_o.sum()

    count_mask = (1 - jnp.isclose(count_o, 0, atol=1e-12)).astype(float)
    uniform_o = (jnp.ones(pi.shape[0]) / count_mask.sum()) * count_mask
    # uniform_o = jnp.ones(pi.shape[0])

    p_o = alpha * uniform_o + (1 - alpha) * count_o

    weight = (pi * p_o[:, None]).T
    if value_type == 'v':
        weight = weight.sum(axis=0)
    weight = lax.stop_gradient(weight)

    # we repeat for every mem state
    diff = (vals[..., ::2] - vals[..., 1::2]).repeat(n_mem, axis=-1)

    if error_type == 'l2':
        unweighted_err = (diff**2)
    elif error_type == 'abs':
        unweighted_err = jnp.abs(diff)
    else:
        raise NotImplementedError(f"Error {error_type} not implemented yet in mem_loss fn.")

    weighted_err = weight * unweighted_err
    if value_type == 'q':
        weighted_err = weighted_err.sum(axis=0)

    # negative here, because we want there to be a big diff between
    # the different values of the mem states
    loss = -weighted_err.sum()
    return loss


