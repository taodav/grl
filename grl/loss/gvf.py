from typing import Union

import chex
import jax
import jax.numpy as jnp

from grl.environment import load_pomdp
from grl.loss.ld import weight_and_sum_discrep_loss
# from grl.loss.sr import sr_lstd_lambda
from grl.mdp import POMDP, MDP
from grl.memory import memory_cross_product
from grl.utils.mdp_solver import get_p_s_given_o


def obs_rew_gvf_lstd_lambda(pi: jnp.ndarray, pomdp: Union[MDP, POMDP], lambda_: float = 0.9):
    pi_sa = pomdp.phi @ pi
    a, s, _ = pomdp.T.shape

    sr_as_as, info = sr_lstd_lambda(pi, pomdp, lambda_=lambda_)
    R_as, phi_as_ao, occupancy_as = info['R_as'], info['phi_as_ao'], info['occupancy_as']

    Q_LSTD_lamb_as = jnp.matmul(sr_as_as, R_as).reshape((a, s))
    phi_LSTD_lamb_as = jnp.matmul(sr_as_as, phi_as_ao).reshape((a, s, -1))

    # Compute V(s)
    V_LSTD_lamb_s = jnp.einsum('ij,ji->j', Q_LSTD_lamb_as, pi_sa)

    # Convert from states to observations
    occupancy_s = occupancy_as.reshape((a, s)).sum(0)
    p_pi_of_s_given_o = get_p_s_given_o(pomdp.phi, occupancy_s)

    Q_LSTD_lamb_ao = Q_LSTD_lamb_as @ p_pi_of_s_given_o
    V_LSTD_lamb_o = V_LSTD_lamb_s @ p_pi_of_s_given_o

    return V_LSTD_lamb_o, Q_LSTD_lamb_ao, {'occupancy': occupancy_s}


def gvf_loss(pi: jnp.ndarray,
             pomdp: Union[MDP, POMDP],
             value_type: str = 'q',
             error_type: str = 'l2',
             lambda_0: float = 0.,
             lambda_1: float = 1.,
             projection: str = 'obs_rew',  # ['obs_rew', 'obs']
             proj: jnp.ndarray = None,
             alpha: float = 1.,
             flip_count_prob: bool = False):
    pi_sa = pomdp.phi @ pi
    a, s, _ = pomdp.T.shape

    sr_as_as_0, info = sr_lstd_lambda(pi, pomdp, lambda_=lambda_0)
    sr_as_as_1, _ = sr_lstd_lambda(pi, pomdp, lambda_=lambda_1)

    R_as, phi_as_ao, occupancy_as = info['R_as'], info['phi_as_ao'], info['occupancy_as']

    # we need occupancy over s
    occupancy_s = occupancy_as.reshape((a, s)).sum(0)

    # We can calculate the gvf loss by projecting the difference between the two SRs,
    # as per the generalized LD document.
    sr_diff = jnp.abs(sr_as_as_0 - sr_as_as_1)

    if projection == 'obs_rew':
        proj = jnp.concatenate((phi_as_ao, R_as[..., None]), axis=-1)
    elif projection == 'obs':
        proj = phi_as_ao
    else:
        assert proj is not None

    projected_diff = (sr_diff @ proj).reshape((a, s, -1))  # A x S x proj_size

    # TODO: How do we reduce projection down to a single dimension? Right now we just do sum.
    a_s_diff = projected_diff.sum(axis=-1)  # A x S

    # now we need to apply W and map back down to Phi space.
    p_pi_of_s_given_o = get_p_s_given_o(pomdp.phi, occupancy_s)

    if value_type == 'v':
        s_diff = jnp.einsum('ij,ji->j', a_s_diff, pi_sa)
        o_diff = s_diff @ p_pi_of_s_given_o
        diff = o_diff
    elif value_type == 'q':
        a_o_diff = a_s_diff @ p_pi_of_s_given_o
        diff = a_o_diff
    else:
        raise NotImplementedError

    c_s = occupancy_s * (1 - pomdp.terminal_mask)

    loss = weight_and_sum_discrep_loss(diff,
                                       c_s,
                                       pi,
                                       pomdp,
                                       value_type=value_type,
                                       error_type=error_type,
                                       alpha=alpha,
                                       flip_count_prob=flip_count_prob)
    return loss, None, None


def mem_gvf_loss(
        mem_params: jnp.ndarray,
        pi: jnp.ndarray,
        pomdp: POMDP,  # input non-static arrays
        value_type: str = 'q',
        error_type: str = 'l2',
        lambda_0: float = 0.,
        lambda_1: float = 1.,
        projection: str = 'obs_rew',  # ['obs_rew', 'obs']
        proj: jnp.ndarray = None,
        alpha: float = 1.,
        flip_count_prob: bool = False):  # initialize with partial
    mem_aug_pomdp = memory_cross_product(mem_params, pomdp)
    loss, _, _ = gvf_loss(pi,
                          mem_aug_pomdp,
                          value_type=value_type,
                          error_type=error_type,
                          lambda_0=lambda_0,
                          lambda_1=lambda_1,
                          projection=projection,
                          proj=proj,
                          alpha=alpha,
                          flip_count_prob=flip_count_prob)
    return loss


if __name__ == "__main__":
    jax.disable_jit(True)

    env, info = load_pomdp('tmaze_5_two_thirds_up')
    pi = info['Pi_phi'][0]

    res = obs_rew_gvf_lstd_lambda(pi, env)
