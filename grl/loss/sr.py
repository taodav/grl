from typing import Union
import jax.numpy as jnp

from grl.mdp import POMDP, MDP


def sr_lstd_lambda(pi: jnp.ndarray, pomdp: Union[MDP, POMDP], lambda_: float = 0.9):
    """
    Solve for the successor representation using LSTD(λ)

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
    R_as = jnp.einsum('ijk,ijk->ij', T_ass, R_ass).reshape((sa,))

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

    # So instead of calculating the LSTDQ formula with a b that right-multiplies by R_as,
    # we just calculate the inverse directly for a B term.
    B = phi_D_mu @ jnp.linalg.solve(I - gamma * lambda_ * P_as_as, I)
    sr_LSTD_lamb_as_as = (phi_as_ao @ jnp.linalg.solve(A + D_eps_ao, B))
    return sr_LSTD_lamb_as_as, {'R_as': R_as, 'phi_as_ao': phi_as_ao, 'occupancy_as': occupancy_as}

