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
    # calculates H(omega' | omega, a) where omega and a are given
    n_s = pomdp.state_space.n
    n_a = pomdp.action_space.n

    # (a, s, s)
    T = pomdp.T 

    # W(s | omega), shape (omega, s)
    Pr_s = jnp.ones(n_s) / n_s
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
    # calculates H(omega' | omega, a) where omega and a are given
    n_s = pomdp.state_space.n
    n_a = pomdp.action_space.n

    # (a, s, s)
    T = pomdp.T 

    # W(s | omega), shape (omega, s)
    Pr_s = jnp.ones(n_s) / n_s
    W = Pr_s[None, ...] *  pomdp.phi.T
    W = W / jnp.maximum(jnp.sum(W, axis=1)[..., None], 1e-8)

    I_A = jnp.eye(n_a)
    W_A = kron(W, I_A)  # (o, a, s, a)
    OASAS = jnp.einsum('ijkl,lkm->ijklm', W_A, T)  # (o, a, s, a, s)

    H = grouped_entropy(OASAS, pomdp.R)

    loss = jnp.sum(H * H)

    return loss

def grouped_entropy(probs, x):
    """
    probs: (a, b, c, d, e) — probability distribution per (a,b)
    x: (c, d, e) — values over the (c,d) grid, possibly with duplicates
    returns: (a, b) — entropy over unique values in x
    """
    a, b, c, d, e = probs.shape
    x_flat = x.reshape(-1)  # shape (c*d*e,)
    probs_flat = probs.reshape(a, b, -1)  # shape (a, b, c*d*e)

    unique_vals, inv_indices = jnp.unique(x_flat, return_inverse=True)  # unique values and mapping
    n_unique = unique_vals.shape[0]

    # One-hot mask of shape (c*d*e, n_unique)
    one_hot = jax.nn.one_hot(inv_indices, n_unique)  # shape (c*d*e, n_unique)

    # Grouped probs: (a, b, n_unique)
    grouped_probs = jnp.einsum('abc,cn->abn', probs_flat, one_hot)

    # Entropy over unique values
    grouped_probs = jnp.clip(grouped_probs, 1e-12, 1.0)
    entropy = -jnp.sum(grouped_probs * jnp.log(grouped_probs), axis=-1)  # shape (a, b)

    return entropy