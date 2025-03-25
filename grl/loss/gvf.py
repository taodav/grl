from typing import Union
import jax.numpy as jnp

from grl.environment import load_pomdp
from grl.mdp import POMDP, MDP
from grl.utils.mdp_solver import get_p_s_given_o


def obs_rew_gvf_lstd_lambda(pi: jnp.ndarray, pomdp: Union[MDP, POMDP], lambda_: float = 0.9):
    """Solve for V, Q using LSTD(λ)

    For the definition of LSTD(λ) see https://arxiv.org/pdf/1405.3229.pdf

    We replace state features with state-action features as described in section 2 of
    https://arxiv.org/pdf/1511.08495.pdf
    """
    T_ass = pomdp.T
    R_ass = pomdp.R

    a, s, _ = T_ass.shape
    phi = pomdp.phi if hasattr(pomdp, 'phi') else jnp.eye(s)

    o = phi.shape[1]
    sa = s * a
    oa = o * a

    gamma = pomdp.gamma
    s0 = pomdp.p0

    pi_sa = phi @ pi

    as_0 = (s0[:, None] * pi_sa).T.reshape(sa)

    # State-action to state-action transition kernel
    P_asas = jnp.einsum('ijk,kl->ijlk', T_ass, pi_sa)
    P_as_as = P_asas.reshape((sa, sa))

    # State-action reward function
    R_as = jnp.einsum('ijk,ijk->ij', T_ass, R_ass).reshape((sa, ))

    # Compute the state-action distribution as a diagonal matrix
    I = jnp.eye(sa)
    occupancy_as = jnp.linalg.solve((I - gamma * P_as_as.T), as_0)
    mu = occupancy_as / jnp.sum(occupancy_as)
    D_mu = jnp.diag(mu)

    # Compute the state-action to obs-action observation function
    # (use a copy of phi for each action)
    phi_as_ao = jnp.kron(jnp.eye(a), phi)

    # Solve the linear system for Q(s,a), replacing the state features with state-action features
    #
    # See section 2 of https://arxiv.org/pdf/1511.08495.pdf
    D_eps_ao = 1e-10 * jnp.eye(oa)
    phi_D_mu = phi_as_ao.T @ D_mu
    A = (phi_D_mu @ (I - gamma * P_as_as) @ jnp.linalg.solve(I - gamma * lambda_ * P_as_as,
                                                             phi_as_ao))
    b = phi_D_mu @ jnp.linalg.solve(I - gamma * lambda_ * P_as_as, R_as)
    Q_LSTD_lamb_as = (phi_as_ao @ jnp.linalg.solve(A + D_eps_ao, b)).reshape((a, s))

    # Compute V(s)
    V_LSTD_lamb_s = jnp.einsum('ij,ji->j', Q_LSTD_lamb_as, pi_sa)

    # Convert from states to observations
    occupancy_s = occupancy_as.reshape((a, s)).sum(0)
    p_pi_of_s_given_o = get_p_s_given_o(pomdp.phi, occupancy_s)

    Q_LSTD_lamb_ao = Q_LSTD_lamb_as @ p_pi_of_s_given_o
    V_LSTD_lamb_o = V_LSTD_lamb_s @ p_pi_of_s_given_o

    return V_LSTD_lamb_o, Q_LSTD_lamb_ao, {'occupancy': occupancy_s}

def mem_gvf_loss(
        mem_params: jnp.ndarray,
        pi: jnp.ndarray,
        pomdp: POMDP,  # input non-static arrays
        value_type: str = 'q',
        error_type: str = 'l2',
        lambda_0: float = 0.,
        lambda_1: float = 1.,
        alpha: float = 1.,
        flip_count_prob: bool = False):  # initialize with partial
    pass


if __name__ == "__main__":
    env, info = load_pomdp('tmaze_5_two_thirds_up')
    pi = info['Pi_phi'][0]

    res = obs_rew_gvf_lstd_lambda(pi, env)
