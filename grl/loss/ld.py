from functools import partial

import jax.numpy as jnp
from jax import jit, nn, lax

from grl.memory import memory_cross_product
from grl.mdp import POMDP
from grl.utils.policy_eval import lstdq_lambda


def weight_and_sum_discrep_loss(diff: jnp.ndarray,
                                occupancy: jnp.ndarray,
                                pi: jnp.ndarray,
                                pomdp: POMDP,
                                value_type: str = 'q',
                                error_type: str = 'l2',
                                alpha: float = 1.,
                                flip_count_prob: bool = False):
    c_o = occupancy @ pomdp.phi
    count_o = c_o / c_o.sum()

    if flip_count_prob:
        count_o = nn.softmax(-count_o)

    count_mask = (1 - jnp.isclose(count_o, 0, atol=1e-12)).astype(float)
    uniform_o = (jnp.ones(pi.shape[0]) / count_mask.sum()) * count_mask
    # uniform_o = jnp.ones(pi.shape[0])

    p_o = alpha * uniform_o + (1 - alpha) * count_o

    weight = (pi * p_o[:, None]).T
    if value_type == 'v':
        weight = weight.sum(axis=0)
    weight = lax.stop_gradient(weight)

    if error_type == 'l2':
        unweighted_err = (diff**2)
    elif error_type == 'abs':
        unweighted_err = jnp.abs(diff)
    else:
        raise NotImplementedError(f"Error {error_type} not implemented yet in mem_loss fn.")

    weighted_err = weight * unweighted_err
    if value_type == 'q':
        weighted_err = weighted_err.sum(axis=0)

    loss = weighted_err.sum()
    return loss


@partial(jit,
         static_argnames=[
             'value_type', 'error_type', 'lambda_0', 'lambda_1', 'alpha', 'flip_count_prob'
         ])
def discrep_loss(
        pi: jnp.ndarray,
        pomdp: POMDP, # non-state args
        value_type: str = 'q',
        error_type: str = 'l2',
        lambda_0: float = 0.,
        lambda_1: float = 1.,
        alpha: float = 1.,
        flip_count_prob: bool = False): # initialize static args
    # if lambda_0 == 0. and lambda_1 == 1.:
    #     _, mc_vals, td_vals, info = analytical_pe(pi, pomdp)
    #     lambda_0_vals = td_vals
    #     lambda_1_vals = mc_vals
    # else:
    # TODO: info here only contains state occupancy, which should lambda agnostic.
    lambda_0_v_vals, lambda_0_q_vals, _ = lstdq_lambda(pi, pomdp, lambda_=lambda_0)
    lambda_1_v_vals, lambda_1_q_vals, info = lstdq_lambda(pi, pomdp, lambda_=lambda_1)
    lambda_0_vals = {'v': lambda_0_v_vals, 'q': lambda_0_q_vals}
    lambda_1_vals = {'v': lambda_1_v_vals, 'q': lambda_1_q_vals}

    diff = lambda_1_vals[value_type] - lambda_0_vals[value_type]
    c_s = info['occupancy'] * (1 - pomdp.terminal_mask)
    loss = weight_and_sum_discrep_loss(diff,
                                       c_s,
                                       pi,
                                       pomdp,
                                       value_type=value_type,
                                       error_type=error_type,
                                       alpha=alpha,
                                       flip_count_prob=flip_count_prob)

    return loss, lambda_1_vals, lambda_0_vals

def mem_discrep_loss(
        mem_params: jnp.ndarray,
        pi: jnp.ndarray,
        pomdp: POMDP, # input non-static arrays
        value_type: str = 'q',
        error_type: str = 'l2',
        lambda_0: float = 0.,
        lambda_1: float = 1.,
        alpha: float = 1.,
        flip_count_prob: bool = False): # initialize with partial
    mem_aug_pomdp = memory_cross_product(mem_params, pomdp)
    loss, _, _ = discrep_loss(pi,
                              mem_aug_pomdp,
                              value_type,
                              error_type,
                              lambda_0=lambda_0,
                              lambda_1=lambda_1,
                              alpha=alpha,
                              flip_count_prob=flip_count_prob)
    return loss

def obs_space_mem_discrep_loss(
        mem_params: jnp.ndarray,
        pi: jnp.ndarray,
        pomdp: POMDP, # input non-static arrays
        value_type: str = 'q',
        error_type: str = 'l2',
        lambda_0: float = 0.,
        lambda_1: float = 1.,
        alpha: float = 1.,
        flip_count_prob: bool = False):
    """
    Memory discrepancy loss on the TD(0) estimator over observation space.

    """
    mem_aug_pomdp = memory_cross_product(mem_params, pomdp)

    n_mem_states = mem_params.shape[-1]
    pi_obs = pi[::n_mem_states]
    mem_lambda_0_v_vals, mem_lambda_0_q_vals, mem_info = lstdq_lambda(pi,
                                                                      mem_aug_pomdp,
                                                                      lambda_=lambda_0)
    lambda_1_v_vals, lambda_1_q_vals, info = lstdq_lambda(pi_obs, pomdp, lambda_=lambda_1)

    counts_mem_aug_flat_obs = mem_info['occupancy'] @ mem_aug_pomdp.phi
    counts_mem_aug_flat = jnp.einsum('i,ij->ij', counts_mem_aug_flat_obs, pi).T # A x OM

    counts_mem_aug = counts_mem_aug_flat.reshape(pomdp.action_space.n, -1,
                                                 n_mem_states) # A x O x M

    denom_counts_mem_aug_unmasked = counts_mem_aug.sum(axis=-1, keepdims=True)
    denom_mask = (denom_counts_mem_aug_unmasked == 0).astype(float)
    denom_counts_mem_aug = denom_counts_mem_aug_unmasked + denom_mask
    prob_mem_given_oa = counts_mem_aug / denom_counts_mem_aug

    unflattened_lambda_0_q_vals = mem_lambda_0_q_vals.reshape(pomdp.action_space.n, -1,
                                                              n_mem_states)
    reformed_lambda_0_q_vals = (unflattened_lambda_0_q_vals * prob_mem_given_oa).sum(axis=-1)

    lambda_1_vals = {'v': lambda_1_v_vals, 'q': lambda_1_q_vals}
    lambda_0_vals = {
        'v': (reformed_lambda_0_q_vals * pi_obs.T).sum(0),
        'q': reformed_lambda_0_q_vals
    }

    diff = lambda_1_vals[value_type] - lambda_0_vals[value_type]

    # set terminal counts to 0
    c_s = info['occupancy'] * (1 - pomdp.terminal_mask)

    loss = weight_and_sum_discrep_loss(diff,
                                       c_s,
                                       pi_obs,
                                       pomdp,
                                       value_type=value_type,
                                       error_type=error_type,
                                       alpha=alpha,
                                       flip_count_prob=flip_count_prob)
    return loss

def policy_discrep_loss(pi_params: jnp.ndarray,
                        pomdp: POMDP,
                        value_type: str = 'q',
                        error_type: str = 'l2',
                        lambda_0: float = 0.,
                        lambda_1: float = 1.,
                        alpha: float = 1.,
                        flip_count_prob: bool = False): # initialize with partial
    pi = nn.softmax(pi_params, axis=-1)
    loss, mc_vals, td_vals = discrep_loss(pi,
                                          pomdp,
                                          value_type,
                                          error_type,
                                          lambda_0=lambda_0,
                                          lambda_1=lambda_1,
                                          alpha=alpha,
                                          flip_count_prob=flip_count_prob)
    return loss, (mc_vals, td_vals)
