from typing import Union
import jax.numpy as jnp

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


def calculate_sr(
        pomdp: POMDP,
        pi: jnp.ndarray,
):
    n_states, n_obs = pomdp.state_space.n, pomdp.observation_space.n
    n_actions = pomdp.action_space.n

    T, R, p0, Phi, gamma = pomdp.T, pomdp.R, pomdp.p0, pomdp.phi, pomdp.gamma

    terminal_mask = jnp.diag(pomdp.terminal_mask)[None, ...].repeat(n_actions, axis=0)  # A x S x S
    if T.shape == (n_actions, n_states, n_states):
        T = jnp.permute_dims(T, (1, 0, 2))
        terminal_mask = jnp.permute_dims(terminal_mask, (1, 0, 2))

    T *= (1 - terminal_mask)

    I_S = jnp.eye(n_states)
    I_A = jnp.eye(n_actions)
    I_O = jnp.eye(n_obs)
    I_SA = jnp.eye(n_actions * n_states).reshape((n_states, n_actions, n_states, n_actions))
    Phi_A = kron(Phi, I_A)

    pi_s = dot(Phi, pi)
    assert is_prob_matrix(pi_s, (n_states, n_actions))
    T_pi = np.einsum("ik,ikj->ij", pi_s, T)
    assert is_subprob_matrix(T_pi, (n_states, n_states))
    Pi = np.eye(len(pi))[..., None] * pi[None, ...]
    Pi_s = np.eye(len(pi_s))[..., None] * pi_s[None, ...]

    # Pr_s = np.ones(n_states) / n_states
    c_s = np.linalg.solve(I_S - gamma * T_pi.T, p0)

    # Pr_s = np.linalg.inv(I_S - gamma * T_pi.T).dot(p0)
    # Pr_s = np.random.random(n_states)

    Pr_s = c_s / np.sum(c_s)
    # Pr_s[0] = (Pr_s[0] + Pr_s[1] + Pr_s[2] + Pr_s[3]) / 4
    # Pr_s[1] = Pr_s[0]
    # Pr_s[2] = Pr_s[0]
    # Pr_s[3] = Pr_s[0]

    # W = np.zeros((n_obs, n_states))
    # for i in range(n_obs):
    #     for j in range(n_states):
    #         pr_i = np.sum([Pr_s[k] * Phi[k][i] for k in range(n_states)])
    #         if np.isclose(pr_i, 0.0):
    #             # this observation is never dispensed...
    #             continue
    #         W[i, j] = (
    #                 Pr_s[j] * Phi[j][i] / pr_i
    #         )

    W_unnorm = np.einsum('i,ij->ji', Pr_s, Phi)
    W = W_unnorm / W_unnorm.sum(axis=-1, keepdims=True)

    # TODO: simplify this
    W_Pi = np.einsum('ijk,jklm->ilm', Pi, kron(W, I_A))
    # W_Pi = ddot(Pi, kron(W, I_A))

    SR_MC_SS = np.linalg.inv(I_S - gamma * T_pi)

    W_phi_Pi = np.einsum('ij,jkl->ikl', Phi, W_Pi)
    T_td = np.einsum('ijk,jkl->il', W_phi_Pi, T)
    SR_TD_SS = np.linalg.inv(I_S - gamma * T_td)

    """
    print(f"SR_MC_SS = SR_TD_SS? {np.allclose(SR_MC_SS, SR_TD_SS)}")

    A_MC = dot(SR_MC_SS, Phi)
    A_TD = dot(SR_TD_SS, Phi)

    print(f"X Phi equal? {np.allclose(A_MC, A_TD)}")

    A_MC = dot(ddot(W_Pi, T), SR_MC_SS)
    A_TD = dot(ddot(W_Pi, T), SR_TD_SS)

    print(f"WPi T X equal? {np.allclose(A_MC, A_TD)}")

    A_MC = dot(dot(T, SR_MC_SS), Phi)
    A_TD = dot(dot(T, SR_TD_SS), Phi)

    print(f"T X Phi equal? {np.allclose(A_MC, A_TD)}")

    A_MC = dot(dot(ddot(kron(W, I_A), T), SR_MC_SS), Phi)
    A_TD = dot(dot(ddot(kron(W, I_A), T), SR_TD_SS), Phi)

    print(f"W T X Phi equal? {np.allclose(A_MC, A_TD)}")

    A_MC = dot(dot(ddot(W_Pi, T), SR_MC_SS), Phi)
    A_TD = dot(dot(ddot(W_Pi, T), SR_TD_SS), Phi)

    print(f"WPi T X Phi equal? {np.allclose(A_MC, A_TD)}")
    """

    SR_MC = I_O + gamma * dot(ddot(W_Pi, T), SR_MC_SS, Phi)
    SR_TD = I_O + gamma * dot(ddot(W_Pi, T), SR_TD_SS, Phi)
    # SR_TD = np.linalg.inv(I_O - gamma * dot(ddot(W_Pi, T), Phi))

    return SR_MC, SR_TD
