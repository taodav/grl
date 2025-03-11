import jax.numpy as jnp
from jax import lax

"""
The following few functions are loss function w.r.t. memory parameters, mem_params.
"""

def mse(predictions: jnp.ndarray, targets: jnp.ndarray = None, zero_mask: jnp.ndarray = None):
    if targets is None:
        targets = jnp.zeros_like(predictions)
    squared_diff = 0.5 * (predictions - targets)**2

    # if we have a zero mask, we take the mean over non-zero elements.
    if zero_mask is not None:
        masked_squared_diff = squared_diff * zero_mask
        return jnp.sum(masked_squared_diff) * (1 / zero_mask.sum())

    return jnp.mean(squared_diff)

def seq_sarsa_loss(q: jnp.ndarray, a: jnp.ndarray, r: jnp.ndarray, gamma: jnp.ndarray,
                   next_q: jnp.ndarray, next_a: jnp.ndarray):
    """
    sequential version of sarsa loss
    First axis of all tensors are the sequence length.
    """
    target = r + gamma * next_q[jnp.arange(next_a.shape[0]), next_a]
    target = lax.stop_gradient(target)
    q_vals = q[jnp.arange(a.shape[0]), a]
    return q_vals - target

def seq_sarsa_mc_loss(q: jnp.ndarray, a: jnp.ndarray, ret: jnp.ndarray):
    q_vals = q[jnp.arange(a.shape[0]), a]
    return q_vals - ret

def seq_sarsa_lambda_discrep(q_td: jnp.ndarray, q_mc: jnp.ndarray, a: jnp.ndarray):
    q_vals_td = q_td[jnp.arange(a.shape[0]), a]
    q_vals_mc = q_mc[jnp.arange(a.shape[0]), a]
    q_vals_mc = lax.stop_gradient(q_vals_mc)

    return q_vals_td - q_vals_mc





