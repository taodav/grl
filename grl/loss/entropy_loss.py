from grl.mdp import POMDP
import jax.numpy as jnp
from grl.memory import memory_cross_product
import jax

def kron(a, b):
    return jnp.einsum("ij...,kl...->ikjl...", a, b)

def ddot(a, b):
    return jnp.tensordot(a, b, axes=2)

def dot(a, b):
    return jnp.tensordot(a, b, axes=1)

def categorical_entropy(p, eps=1e-12):
    p = jnp.clip(p, eps, 1.0)  # Avoid log(0)
    return -jnp.sum(p * jnp.log(p), axis=-1)

def mem_entropy_loss(
        mem_params: jnp.ndarray,
        pomdp: POMDP
    ):
    mem_aug_pomdp = memory_cross_product(mem_params, pomdp)
    loss = entropy_loss(mem_aug_pomdp)
    return loss

def mem_reward_entropy_loss(
        mem_params: jnp.ndarray,
        pomdp: POMDP
    ):
    mem_aug_pomdp = memory_cross_product(mem_params, pomdp)
    loss = reward_entropy_loss(mem_aug_pomdp)
    return loss

def entropy_loss(
        pomdp: POMDP
    ):
    # calculates sum_{omega, a} H(omega' | omega, a)
    n_s = pomdp.state_space.n
    n_o = pomdp.observation_space.n
    n_a = pomdp.action_space.n

    # (a, s, s)
    T = pomdp.T 

    # W(s | omega), shape (omega, s)
    pi = jnp.ones((n_o, n_a), dtype=float) / n_a
    pi_s = pomdp.phi @ pi
    T_pi = jnp.einsum("ik,kij->ij", pi_s, T)
    I_S = jnp.eye(n_s)
    Pr_s = jnp.linalg.inv(I_S - pomdp.gamma * T_pi.T).dot(pomdp.p0)
    Pr_s = Pr_s / jnp.sum(Pr_s)

    # this is WRONG! state includes memory, and the entire point of memory is that
    # it does NOT have the same likelihood over different aliased states
    # >>> WRONG >>> Pr_s = jnp.ones(n_s) / n_s
 
    W = Pr_s[None, ...] *  pomdp.phi.T
    W = W / jnp.maximum(jnp.sum(W, axis=1)[..., None], 1e-8)

    I_A = jnp.eye(n_a)
    W_A = kron(W, I_A)  # (o, a, s, a)
    OAS = ddot(W_A, jnp.permute_dims(T, (1, 0, 2)))  # (o, a, s)

    # this is the one-step transition operator in observation space
    T_O = dot(OAS, pomdp.phi)  # (o, a, o)

    H = categorical_entropy(T_O)  # (o, a)

    loss = jnp.sum(H * H)

    return loss

def reward_entropy_loss(
        pomdp: POMDP
    ):
    # calculates sum_{omega, a} H(r | omega, a)
    n_s = pomdp.state_space.n
    n_a = pomdp.action_space.n
    n_o = pomdp.observation_space.n

    # (a, s, s)
    T = pomdp.T 

    # W(s | omega), shape (omega, s)
    pi = jnp.ones((n_o, n_a), dtype=float) / n_a
    pi_s = pomdp.phi @ pi  # (s, a)
    T_pi = jnp.einsum("ik,kij->ij", pi_s, T)
    I_S = jnp.eye(n_s)
    Pr_s = jnp.linalg.inv(I_S - pomdp.gamma * T_pi.T).dot(pomdp.p0)
    Pr_s = Pr_s / jnp.sum(Pr_s)

    # this is WRONG! state includes memory, and the entire point of memory is that
    # it does NOT have the same likelihood over different aliased states
    # >>> WRONG >>> Pr_s = jnp.ones(n_s) / n_s

    W = Pr_s[None, ...] *  pomdp.phi.T
    W = W / jnp.maximum(jnp.sum(W, axis=1)[..., None], 1e-8)
    #print_nonzero_entries(pomdp.phi)

    I_A = jnp.eye(n_a)
    W_A = kron(W, I_A)  # (o, a, s, a)
    OAASS = jnp.einsum('ijkl,lkm->ijlkm', W_A, T)  # (o, a, a, s, s)
    
    #print_nonzero_entries(OAASS)
    #print_nonzero_entries(pomdp.R)

    H = grouped_entropy(OAASS, pomdp.R)
    #print(f"H.shape = \n{H.shape}")
    #print(f"H = \n{H}")

    loss = jnp.sum(H * H)

    return loss

def print_nonzero_entries(x):
    nonzero_indices = jnp.argwhere(x != 0)  # Get indices of non-zero elements
    values = x[tuple(nonzero_indices.T)]    # Use advanced indexing to get values
    for idx, val in zip(nonzero_indices, values):
        print(f"{tuple(idx.tolist())}: {val}")

def grouped_entropy(probs, x):
    """
    probs: (a, b, c, d, e) — probability distribution per (a,b)
    x: (c, d, e) — values over the (c, d, e) grid, possibly with duplicates
    returns: (a, b) — entropy over unique values in x
    """
    a, b, c, d, e = probs.shape
    import numpy as np
    x_flat = x.reshape(-1)  # shape (c*d*e,)
    probs_flat = probs.reshape(a, b, -1)  # shape (a, b, c*d*e)

    unique_vals, inv_indices = jnp.unique(x_flat, return_inverse=True)  # unique values and mapping
    #print(f"unique_vals: {unique_vals}")
    #print(f"inv_indices.shape: {inv_indices.shape}")
    n_unique = unique_vals.shape[0]

    # One-hot mask of shape (c*d*e, n_unique)
    one_hot = jax.nn.one_hot(inv_indices, n_unique)  # shape (c*d*e, n_unique)
    #print(jnp.argwhere(inv_indices==0))
    #print(jnp.argwhere(inv_indices==2))
    #print(jnp.sum(probs_flat[4, :, 147]))
    #print(jnp.sum(probs_flat[4, :, 177]))
    #print(jnp.sum(probs_flat[4, :, 357]))
    #print(jnp.sum(probs_flat[4, :, 387]))
    #print(jnp.sum(probs_flat[4, :, 134]))
    #print(jnp.sum(probs_flat[4, :, 164]))
    #print(jnp.sum(probs_flat[4, :, 374]))
    #print(jnp.sum(probs_flat[4, :, 404]))
    #print(jnp.sum(inv_indices == 2))
    #print_nonzero_entries(one_hot)

    # Grouped probs: (a, b, n_unique)
    grouped_probs = jnp.einsum('abc,cn->abn', probs_flat, one_hot)
    #print_nonzero_entries(grouped_probs)

    # Entropy over unique values
    grouped_probs = jnp.clip(grouped_probs, 1e-12, 1.0)
    entropy = -jnp.sum(grouped_probs * jnp.log(grouped_probs), axis=-1)  # shape (a, b)

    return entropy