from grl.mdp import POMDP
import jax.numpy as jnp
from grl.memory import memory_cross_product

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

    I_A = jnp.eyes(n_a)
    W_A = kron(W, I_A)  # (o, a, s, a)
    OAS = ddot(W_A, jnp.permute_dims(T, (1, 0, 2)))  # (o, a, s)

    # this is the one-step transition operator in observation space
    T_O = dot(OAS, pomdp.phi)  # (o, a, o)

    H = categorical_entropy(T_O)  # (o, a)

    loss = jnp.sum(H * H)

    return loss