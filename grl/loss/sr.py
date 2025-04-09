from typing import Union
import jax.numpy as jnp
import numpy as np

from grl.mdp import POMDP, MDP
from grl.utils.mdp_solver import get_p_s_given_o


# def sr_lstd_lambda(pi: jnp.ndarray, pomdp: Union[MDP, POMDP], lambda_: float = 0.9):
#     """
#     Solve for the successor representation using LSTD(λ)
#
#     For the definition of LSTD(λ) see https://arxiv.org/pdf/1405.3229.pdf
#
#     We replace state features with state-action features as described in section 2 of
#     https://arxiv.org/pdf/1511.08495.pdf
#     """
#     T_ass = pomdp.T
#     R_ass = pomdp.R
#
#     a, s, _ = T_ass.shape
#     terminal_mask = jnp.diag(pomdp.terminal_mask)[None, ...].repeat(a, axis=0)
#     T_ass = T_ass * (1 - terminal_mask)
#     phi = pomdp.phi if hasattr(pomdp, 'phi') else jnp.eye(s)
#
#     o = phi.shape[1]
#     sa = s * a
#     oa = o * a
#
#     gamma = pomdp.gamma
#     s0 = pomdp.p0
#
#     pi_sa = phi @ pi
#
#     as_0 = (s0[:, None] * pi_sa).T.reshape(sa)
#
#     # State-action to state-action transition kernel
#     P_asas = jnp.einsum('ijk,kl->ijlk', T_ass, pi_sa)
#     P_as_as = P_asas.reshape((sa, sa))
#
#     # State-action reward function
#     R_as = jnp.einsum('ijk,ijk->ij', T_ass, R_ass).reshape((sa,))
#
#     # Compute the state-action distribution as a diagonal matrix
#     I = jnp.eye(sa)
#     occupancy_as = jnp.linalg.solve((I - gamma * P_as_as.T), as_0)
#     mu = occupancy_as / jnp.sum(occupancy_as)
#     D_mu = jnp.diag(mu)
#
#     # Compute the state-action to obs-action observation function
#     # (use a copy of phi for each action)
#     phi_as_ao = jnp.kron(jnp.eye(a), phi)
#
#     # Solve the linear system for Q(s,a), replacing the state features with state-action features
#     #
#     # See section 2 of https://arxiv.org/pdf/1511.08495.pdf
#     D_eps_ao = 1e-10 * jnp.eye(oa)
#     phi_D_mu = phi_as_ao.T @ D_mu
#     A = (phi_D_mu @ (I - gamma * P_as_as) @ jnp.linalg.solve(I - gamma * lambda_ * P_as_as,
#                                                              phi_as_ao))
#
#     # So instead of calculating the LSTDQ formula with a b that right-multiplies by R_as,
#     # we just calculate the inverse directly for a B term.
#     B = phi_D_mu @ jnp.linalg.solve(I - gamma * lambda_ * P_as_as, I)
#     sr_LSTD_lamb_as_as = (phi_as_ao @ jnp.linalg.solve(A + D_eps_ao, B))
#     return sr_LSTD_lamb_as_as, {'R_as': R_as, 'phi_as_ao': phi_as_ao, 'occupancy_as': occupancy_as}
#
#
# def sf_lstd_lambda(pi: jnp.ndarray, pomdp: Union[MDP, POMDP], lambda_: float = 0.9):
#     pi_s = pomdp.phi @ pi
#
#     s, a = pomdp.state_space.n, pomdp.action_space.n
#     sr_lstd_lamb_as_as, info = sr_lstd_lambda(pi, pomdp, lambda_=lambda_)
#     sr_lstd_lamb_s_a_as = sr_lstd_lamb_as_as.reshape((s, a, -1))
#     sr_lstd_lamb_s_as = jnp.einsum('ijk,ij->ik', sr_lstd_lamb_s_a_as, pi_s)
#     sr_lstd_lamb_s_s = sr_lstd_lamb_s_as.reshape((s, s, a)).sum(axis=-1)
#
#     occupancy_s = info['occupancy_as'].reshape((a, s)).sum(0)
#     p_pi_of_s_given_o = get_p_s_given_o(pomdp.phi, occupancy_s)
#
#     sf_lstd_lamb_s_o = sr_lstd_lamb_s_s @ pomdp.phi  # S x O
#     sf_lstd_lamb_o_o = p_pi_of_s_given_o.T @ sf_lstd_lamb_s_o
#
#     return sf_lstd_lamb_o_o, {
#         'sr_lstd_lamb_as_as': sr_lstd_lamb_as_as
#     }

def kron(a, b):
    return jnp.einsum("ij...,kl...->ikjl...", a, b)


def sr_lambda(
        pomdp: POMDP,
        pi: jnp.ndarray,
        lambda_: float = 0.
):
    """
    Implemented as per https://arxiv.org/abs/2407.07333.
    """
    n_states, n_obs = pomdp.state_space.n, pomdp.observation_space.n
    n_actions = pomdp.action_space.n
    # n_sa = n_states * n_actions

    T, R, p0, Phi, gamma = pomdp.T, pomdp.R, pomdp.p0, pomdp.phi, pomdp.gamma

    terminal_mask = jnp.diag(pomdp.terminal_mask)[None, ...].repeat(n_actions, axis=0)  # A x S x S
    if T.shape == (n_actions, n_states, n_states):
        T = jnp.permute_dims(T, (1, 0, 2))
        terminal_mask = jnp.permute_dims(terminal_mask, (1, 0, 2))

    T *= (1 - terminal_mask)

    I_S = jnp.eye(n_states)
    I_A = jnp.eye(n_actions)
    # I_SA = jnp.eye(n_sa)

    pi_s = Phi @ pi

    T_pi = jnp.einsum("ik,ikj->ij", pi_s, T)
    Pi = jnp.eye(len(pi))[..., None] * pi[None, ...]
    Pi_s = jnp.eye(len(pi_s))[..., None] * pi_s[None, ...]

    c_s = jnp.linalg.solve(I_S - gamma * T_pi.T, p0)

    Pr_s = c_s / jnp.sum(c_s)

    W_unnorm = jnp.einsum('i,ij->ji', Pr_s, Phi)
    W_denom = W_unnorm.sum(axis=-1, keepdims=True)
    W_denom += jnp.isclose(W_denom, 0).astype(float)
    W = W_unnorm / W_denom

    W_Pi = jnp.einsum('ijk,jklm->ilm', Pi, kron(W, I_A))

    W_phi_Pi = jnp.einsum('ij,jkl->ikl', Phi, W_Pi)

    K_pi = lambda_ * Pi_s + (1 - lambda_) * W_phi_Pi

    T_lambda = jnp.einsum('ijk,jkl->il', K_pi, T)
    SR_lambda_S_S = jnp.linalg.inv(I_S - gamma * T_lambda)

    # T_lambda = jnp.einsum('ijk,klm->ijlm', T, K_pi)
    # T_lambda_flat = T_lambda.reshape((n_sa, n_sa))
    # SR_lambda_SA_SA_flat = jnp.linalg.inv(I_SA - gamma * T_lambda_flat)
    # SR_lambda_SA_SA = SR_lambda_SA_SA_flat.reshape((T_lambda.shape))
    #
    # return SR_lambda_SA_SA, {'W_Pi': W_Pi, 'T': T, 'pi_s': pi_s}
    return SR_lambda_S_S, {'W': W, 'W_Pi': W_Pi, 'T': T, 'pi_s': pi_s, 'c_s': c_s}

# TODO: OA version of SF lambda
# def sf_lambda(pomdp: POMDP, pi: jnp.ndarray, lambda_: float = 0.):
#     n_actions = pomdp.action_space.n
#     n_obs = pomdp.observation_space.n
#     n_oa = n_obs * n_actions
#
#
#     SR_lambda_SA_SA, info = sr_lambda(pomdp, pi, lambda_=lambda_)
#
#     W_Pi, T, pi_s = info['W_Pi'], info['T'], info['pi_s']
#
#     # Compute the state-action to obs observation function
#     # (use a copy of phi for each action)
#     phi_as_o = jnp.expand_dims(pomdp.phi, axis=1).repeat(n_actions, axis=1)
#
#     Pr_next_s_given_o = jnp.einsum('ijk,jkl->il', W_Pi, T)
#
#     SR_lambda_SA_O = jnp.einsum('ijkl,klm->ijm', SR_lambda_SA_SA, phi_as_o)
#
#     next_SF = jnp.einsum('ij,jkl->ikl', Pr_next_s_given_o, SR_lambda_SA_O)
#     next_SF_flat = next_SF.reshape((n_oa, -1))
#
#     I_OA = jnp.eye(n_obs).repeat(n_actions, axis=0)
#
#     SF_lambda_OA_O_flat = I_OA + pomdp.gamma * next_SF_flat
#     SF_lambda_OA_O = SF_lambda_OA_O_flat.reshape(next_SF.shape)
#     # SR_TD = np.linalg.inv(I_O - gamma * dot(ddot(W_Pi, T), Phi))
#
#     return SF_lambda_OA_O

def sf_lambda(pomdp: POMDP, pi: jnp.ndarray, lambda_: float = 0.):
    n_obs = pomdp.observation_space.n
    I_O = jnp.eye(n_obs)

    SR_lambda_S_S, info = sr_lambda(pomdp, pi, lambda_=lambda_)

    W_Pi, T = info['W_Pi'], info['T']

    proj_next_state_to_obs = jnp.einsum('ijk,jkl->il', W_Pi, T)
    SR_lambda_SO = SR_lambda_S_S @ pomdp.phi

    SR_lambda_OO = I_O + pomdp.gamma * (proj_next_state_to_obs @ SR_lambda_SO)
    return SR_lambda_OO
