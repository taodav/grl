from typing import Union

import chex
import jax
import jax.numpy as jnp

from grl.environment import load_pomdp
from grl.loss.ld import weight_and_sum_discrep_loss
# from grl.loss.sr import sr_lstd_lambda
from grl.loss.sr import sr_lambda, sf_lambda
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

def kron(a, b):
    return jnp.einsum("ij...,kl...->ikjl...", a, b)

def ddot(a, b):
    return jnp.tensordot(a, b, axes=2)

def sr_discrep_loss_peter(
        pi: jnp.ndarray,
        pomdp: Union[MDP, POMDP]):
    n_states = pomdp.state_space.n
    n_actions = pomdp.action_space.n
    n_obs = pomdp.observation_space.n
    T = jnp.permute_dims(pomdp.T, (1, 0, 2))

    I_S = jnp.eye(n_states)
    I_A = jnp.eye(n_actions)
    I_O = jnp.eye(n_obs)

    pi_s = pomdp.phi @ pi
    T_pi = jnp.einsum("ik,ikj->ij", pi_s, T)

    # calculate W
    Pr_s = jnp.linalg.inv(I_S - pomdp.gamma * T_pi.T).dot(pomdp.p0)
    Pr_s = Pr_s / jnp.sum(Pr_s)
    W = Pr_s[None, ...] *  pomdp.phi.T
    W = W / jnp.sum(W, axis=1)[..., None]

    # (o, o, 1) * (1, o, a) -> (o, o, a), so Pi(o1,o2,a) = delta(o1,o2) pi(a|o1)
    Pi = jnp.eye(n_obs)[..., None] * pi[None, ...]
    Pi_s = jnp.eye(n_states)[..., None] * pi_s[None, ...]
    # (o, a), (o, a, s, a) -> (s, a)
    W_Pi = ddot(Pi, kron(W, I_A))
    SR_MC_SS = jnp.linalg.inv(I_S - pomdp.Gamma_s @ ddot(Pi_s, T))
    SR_TD_SS = jnp.linalg.inv(I_S - pomdp.Gamma_s @ ddot((pomdp.phi @ W_Pi), T))

    SR_MC = I_O + pomdp.Gamma_o @ ddot(W_Pi, T) @ SR_MC_SS @ pomdp.phi
    SR_TD = I_O + pomdp.Gamma_o @ ddot(W_Pi, T) @ SR_TD_SS @ pomdp.phi

    Pr_o = Pr_s @ pomdp.phi
    return jnp.sqrt(Pr_o @ jnp.sum((SR_MC - SR_TD)**2, axis=1))

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

    # sr_as_as_0, info = sr_lstd_lambda(pi, pomdp, lambda_=lambda_0)
    # sr_as_as_1, _ = sr_lstd_lambda(pi, pomdp, lambda_=lambda_1)
    sr_s_s_0, info = sr_lambda(pomdp, pi, lambda_=lambda_0)
    sr_s_s_1, _ = sr_lambda(pomdp, pi, lambda_=lambda_1)

    W_Pi, T, c_s = info['W_Pi'], info['T'], info['c_s']
    proj_next_state_to_obs = jnp.einsum('ijk,jkl->il', W_Pi, T)

    # We can calculate the gvf loss by projecting the difference between the two SRs,
    # as per the generalized LD document.
    sr_diff = sr_s_s_0 - sr_s_s_1

    def calc_sf(sr_s_s: jnp.ndarray):
        I_O = jnp.eye(pomdp.observation_space.n)
        sr_so = sr_s_s @ pomdp.phi
        return I_O + pomdp.gamma * (proj_next_state_to_obs @ sr_so)

    if projection == 'obs_rew':
        sf_diff = calc_sf(sr_s_s_0) - calc_sf(sr_s_s_1)
        R_sa = jnp.einsum('ijk,ijk->ij', pomdp.T, pomdp.R).T
        R_s = jnp.einsum('ij,ij->i', R_sa, pi_sa)
        v_s_sr = jnp.einsum('ik,k->i', sr_diff, R_s)
        v_diff = info['W'] @ v_s_sr
        projected_diff = jnp.concatenate((sf_diff, v_diff[..., None]), axis=-1)

    elif projection == 'obs':
        sf_diff = calc_sf(sr_s_s_0) - calc_sf(sr_s_s_1)
        projected_diff = sf_diff
    else:
        assert proj is not None

    # take absolute value
    abs_projected_diff = jnp.abs(projected_diff)

    # TODO: How do we reduce projection down to a single dimension? Right now we just do sum.
    diff = abs_projected_diff.sum(axis=-1)  # O

    # # now we need to apply W and map back down to Phi space.
    # p_pi_of_s_given_o = get_p_s_given_o(pomdp.phi, occupancy_s)
    #
    # if value_type == 'v':
    #     s_diff = jnp.einsum('ij,ji->j', a_s_diff, pi_sa)
    #     o_diff = s_diff @ p_pi_of_s_given_o
    #     diff = o_diff
    # elif value_type == 'q':
    #     a_o_diff = a_s_diff @ p_pi_of_s_given_o
    #     diff = a_o_diff
    # else:
    #     raise NotImplementedError

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
