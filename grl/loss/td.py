from functools import partial

import jax.numpy as jnp
from jax import jit, lax

from grl.memory import memory_cross_product
from grl.mdp import POMDP
from grl.utils.policy_eval import lstdq_lambda


def mem_tde_loss(
        mem_params: jnp.ndarray,
        pi: jnp.ndarray,
        pomdp: POMDP, # input non-static arrays
        value_type: str = 'q',
        error_type: str = 'l2',
        lambda_0: float = 0,
        lambda_1: float = 1.,  # NOT CURRENTLY USED!
        residual: bool = False,
        alpha: float = 1.,
        flip_count_prob: bool = False):
    mem_aug_pomdp = memory_cross_product(mem_params, pomdp)
    loss, _, _ = mstd_err(pi,
                          mem_aug_pomdp,
                          # value_type,
                          error_type,
                          # alpha,
                          lambda_=lambda_0,
                          residual=residual,
                          # flip_count_prob=flip_count_prob
                          )
    return loss

@partial(jit, static_argnames=['error_type', 'residual'])
def mstd_err(
        pi: jnp.ndarray,
        pomdp: POMDP, # non-state args
        error_type: str = 'l2',
        lambda_: float = 0.,
        residual: bool = False): # initialize static args
    # First, calculate our TD(0) Q-values
    v_vals, q_vals, info = lstdq_lambda(pi, pomdp, lambda_=lambda_)
    vals = {'v': v_vals, 'q': q_vals}
    # assert value_type == 'q'

    n_states = pomdp.base_mdp.state_space.n
    n_obs = pomdp.observation_space.n
    n_actions = pomdp.action_space.n

    # Project Q-values from observations to states
    q_soa = q_vals.T[None, :, :].repeat(n_states, axis=0)
    # q_sa = pomdp.phi @ q_vals.T

    # Calculate all "potential" next q-values by repeating at the front.
    qp_soasoa = (
        q_soa[None, None, None, ...]
        .repeat(n_states, axis=0)
        .repeat(n_obs, axis=1)
        .repeat(n_actions, axis=2)
    )

    # Calculate all current q-values, repeating at the back.
    q_soasoa = (
        q_soa[..., None, None, None]
        .repeat(n_states, axis=-3)
        .repeat(n_obs, axis=-2)
        .repeat(n_actions, axis=-1)
    )

    # Expanded and repeated reward tensor
    R_sas = jnp.swapaxes(pomdp.base_mdp.R, 0, 1)
    R_soasoa = (
        R_sas[:, None, :, :, None, None]
        .repeat(n_obs, axis=1)
        .repeat(n_obs, axis=-2)
        .repeat(n_actions, axis=-1)
    )

    # Calculate targets (R + gamma * Q') and stop_grad it.
    targets = R_soasoa + pomdp.gamma * qp_soasoa
    targets = lax.cond(residual, lambda x: x, lambda x: lax.stop_gradient(x), targets)

    # Compute errors
    tde_soasoa = (targets - q_soasoa)

    if error_type == 'l2':
        mag_tde_soasoa = (tde_soasoa**2)
    elif error_type == 'abs':
        mag_tde_soasoa = jnp.abs(tde_soasoa)
    else:
        raise NotImplementedError(f"Error {error_type} not implemented yet in mem_loss fn.")

    # set terminal count to 0 and compute Pr(s)
    c_s = info['occupancy'] * (1 - pomdp.terminal_mask)
    pr_s = c_s / c_s.sum()

    # Retrieve Pr(o|s), Pr(s'|s,a)
    phi_so = pomdp.phi
    T_sas = jnp.swapaxes(pomdp.base_mdp.T, 0, 1)

    # Compute Pr(s,o,a,s',o',a')
    pr_s_soasoa =       pr_s[   :, None, None, None, None, None] # Pr(s)
    phi_soasoa =      phi_so[   :,    :, None, None, None, None] # Pr(o|s)
    pi_soasoa =           pi[None,    :,    :, None, None, None] # Pr(a|o)
    T_soasoa =         T_sas[   :, None,    :,    :, None, None] # Pr(s'|s,a)
    next_phi_soasoa = phi_so[None, None, None,    :,    :, None] # Pr(o'|s')
    next_pi_soasoa =      pi[None, None, None, None,    :,    :] # Pr(a'|o')
    # Pr(s,o,a,s',o',a') = Pr(s) * Pr(o|s) * Pr(a|o) * Pr(s'|s,a) * Pr(o'|s') * Pr(a'|o')
    pr_soasoa = (
            pr_s_soasoa * phi_soasoa * pi_soasoa * T_soasoa * next_phi_soasoa * next_pi_soasoa
    )

    # Reweight squared errors according to Pr(s,o,a,s',o',a')
    weighted_sq_tde_soasoa = pr_soasoa * mag_tde_soasoa

    # Sum over all dimensions
    loss = weighted_sq_tde_soasoa.sum()
    return loss, vals, vals

