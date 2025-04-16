from functools import partial

import jax.numpy as jnp
from jax import jit, lax

from grl.mdp import POMDP
from grl.utils.policy_eval import lstdq_lambda, analytical_pe


@partial(jit, static_argnames=['value_type', 'error_type', 'lambda_'])
def value_error(pi: jnp.ndarray,
                pomdp: POMDP,
                value_type: str = 'q',
                error_type: str = 'l2',
                lambda_: float = 1.0):
    state_vals, mc_vals, td_vals, info = analytical_pe(pi, pomdp)
    if lambda_ == 0.0:
        obs_vals = td_vals
    elif lambda_ == 1.0:
        obs_vals = mc_vals
    else:
        v_vals, q_vals, _ = lstdq_lambda(pi, pomdp, lambda_=lambda_)
        obs_vals = {'v': v_vals, 'q': q_vals}

    # Expand observation (q-)value function to state (q-)value function
    # (a,o) @ (o,s) => (a,s);  (o,) @ (o,s) => (s,)
    expanded_obs_vals = obs_vals[value_type] @ pomdp.phi.T
    diff = state_vals[value_type] - expanded_obs_vals

    c_s = info['occupancy'] * (1 - pomdp.terminal_mask)
    p_s = c_s / c_s.sum()

    pi_s = pomdp.phi @ pi
    weight = (pi_s * p_s[:, None]).T
    if value_type == 'v':
        weight = weight.sum(axis=0)
    weight = lax.stop_gradient(weight)

    if error_type == 'l2':
        unweighted_err = (diff**2)
    elif error_type == 'abs':
        unweighted_err = jnp.abs(diff)
    else:
        raise NotImplementedError(
            f"error_type {error_type} not implemented yet in value_error fn.")

    weighted_err = weight * unweighted_err
    if value_type == 'q':
        weighted_err = weighted_err.sum(axis=0)

    loss = weighted_err.sum()

    return loss, state_vals, expanded_obs_vals
