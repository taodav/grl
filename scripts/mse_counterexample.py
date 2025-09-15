import jax.numpy as jnp

from grl.environment.jax_pomdp import POMDP

def ce_1():
    T_right = jnp.array([
        [0, 0, 0, 1.],
        [0, 0, 1., 0],
        [0, 0, 0, 1.],
        [0, 0, 0, 1.],
    ])

    T_up = jnp.array([
        [1., 0, 0, 0],
        [0, 1., 0, 0],
        [0, 0, 0, 1.],
        [0, 0, 0, 1.],
    ])
    T = jnp.stack([T_right, T_up], axis=0)

    R_right = jnp.array([
        [0, 0, 0, 10.],
        [0, 0, 0, 0],
        [0, 0, 0, -1.],
        [0, 0, 0, 0],
    ])

    R_up = jnp.array([
        [0, 0, 0, 10.],
        [0, 0, 0, 0],
        [0, 0, 0, 1.],
        [0, 0, 0, 0],
    ])
    R = jnp.stack([R_right, R_up], axis=0)

    p0 = jnp.array([0.5, 0.5, 0, 0])

    phi = jnp.array([
        [1., 0],
        [1., 0],
        [1., 0],
        [0, 1.],
    ])
    perfect_H_phi = jnp.array([
        [1., 0, 0],
        [1., 0, 0],
        [0, 1., 0],
        [0, 0, 1.],
    ])
    pomdp = POMDP(T, R, p0, 0.9, phi)

    optimal_pi = jnp.array([])

