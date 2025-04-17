from grl.mdp import MDP, POMDP, POMDPG

import numpy as np
from jax import jit, nn
import jax.numpy as jnp
import jax

@jit
def memory_cross_product(mem_params: jnp.ndarray, pomdp: POMDP):
    T_mem = nn.softmax(mem_params, axis=-1)
    n_states_m = T_mem.shape[-1]
    n_states = pomdp.state_space.n
    n_states_x = n_states_m * n_states

    # Rewards only depend on MDP (not memory function)
    R_x = pomdp.R.repeat(n_states_m, axis=1).repeat(n_states_m, axis=2)

    # T_mem_phi is like T_pi
    # It is SxAxMxM
    T_mem_phi = jnp.tensordot(pomdp.phi, T_mem.swapaxes(0, 1), axes=1)

    # Outer product that compacts the two i dimensions and the two l dimensions
    # (SxAxMxM, AxSxS -> AxSMxSM), where SM=x
    T_x = jnp.einsum('iljk,lim->lijmk', T_mem_phi, pomdp.T).reshape(pomdp.T.shape[0], n_states_x,
                                                                    n_states_x)

    # The new obs_x are the original obs times memory states
    # E.g. obs={r,b} and mem={0,1} -> obs_x={r0,r1,b0,b1}
    phi_x = jnp.kron(pomdp.phi, np.eye(n_states_m))

    # Assuming memory starts with all 0s
    p0_x = jnp.zeros(n_states_x)
    p0_x = p0_x.at[::n_states_m].set(pomdp.p0)

    new_terminal_mask = pomdp.terminal_mask.repeat(n_states_m)

    gamma = pomdp.gamma
    if not isinstance(pomdp.gamma, float) and len(gamma.shape) > 0:
        gamma = gamma.repeat(n_states_m, axis=0)

    mem_aug_mdp = MDP(T_x, R_x, p0_x, gamma=gamma, terminal_mask=new_terminal_mask)

    return POMDP(mem_aug_mdp, phi_x)

@jit
def memory_cross_product_G(mem_params: jnp.ndarray, pomdp: POMDPG):
    T_mem = nn.softmax(mem_params, axis=-1)
    n_states_m = T_mem.shape[-1]
    n_states = pomdp.state_space.n
    n_states_x = n_states_m * n_states

    # Rewards only depend on MDP (not memory function)
    R_x = pomdp.R.repeat(n_states_m, axis=1).repeat(n_states_m, axis=2)

    # T_mem_phi is like T_pi
    # It is SxAxMxM
    T_mem_phi = jnp.tensordot(pomdp.phi, T_mem.swapaxes(0, 1), axes=1)

    # Outer product that compacts the two i dimensions and the two l dimensions
    # (SxAxMxM, AxSxS -> AxSMxSM), where SM=x
    T_x = jnp.einsum('iljk,lim->lijmk', T_mem_phi, pomdp.T).reshape(pomdp.T.shape[0], n_states_x,
                                                                    n_states_x)

    # The new obs_x are the original obs times memory states
    # E.g. obs={r,b} and mem={0,1} -> obs_x={r0,r1,b0,b1}
    phi_x = jnp.kron(pomdp.phi, np.eye(n_states_m))

    # Assuming memory starts with all 0s
    p0_x = jnp.zeros(n_states_x)
    p0_x = p0_x.at[::n_states_m].set(pomdp.p0)

    new_terminal_mask = pomdp.terminal_mask.repeat(n_states_m)

    gamma = pomdp.gamma
    if not isinstance(pomdp.gamma, float) and len(gamma.shape) > 0:
        gamma = gamma.repeat(n_states_m, axis=0)

    Gamma_s = pomdp.Gamma_s
    Gamma_s = jnp.diag(jnp.repeat(jnp.diag(Gamma_s), n_states_m))
    Gamma_o = pomdp.Gamma_o
    Gamma_o = jnp.diag(jnp.repeat(jnp.diag(Gamma_o), n_states_m))
    mem_aug_mdp = MDP(T_x, R_x, p0_x, gamma=gamma, terminal_mask=new_terminal_mask)

    return POMDPG(mem_aug_mdp, phi_x, Gamma_s, Gamma_o)
