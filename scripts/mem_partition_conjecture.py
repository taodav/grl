import numpy as np
import jax
import jax.numpy as jnp

from grl.environment import load_pomdp
from grl.memory import memory_cross_product
from grl.utils import reverse_softmax
from grl.utils.policy_eval import lstdq_lambda


def switching_optimal_deterministic_1_bit_mem():
    mem_fn = jnp.array([
                          # action: right
                          [
                              # obs: start
                              [[0., 1.],
                               [0., 1.]],
                              # obs: middle
                              [[1., 0.],
                               [1., 0.]],
                              # obs: terminal
                              [[1., 0.],
                               [0., 1.]],
                          ],
                          # action: left
                          [
                              # obs: start
                              [[1., 0.],
                               [0., 1.]],
                              # obs: middle
                              [[0., 1.],
                               [1., 0.]],
                              # obs: terminal
                              [[1., 0.],
                               [0., 1.]]
                          ]
        ])
    return mem_fn  # A(2) x O(3) x M(2) x M(2)


def switching_two_thirds_right_policy():
    pi = jnp.array(
        [[1., 0.],
         [2/3, 1/3],
         [1., 0.]])
    return pi


def epsilon_interpolate_deterministic_dist(deterministic_dist: jnp.ndarray, eps: float = 0.1):
    ones = jnp.isclose(deterministic_dist, 1.)
    zeros = jnp.isclose(deterministic_dist, 0.)
    assert jnp.all((ones + zeros).sum(axis=-1) == deterministic_dist.shape[-1]), \
        'Deterministic distribution not filled with only 0s and 1s'

    return deterministic_dist - eps * ones + eps * zeros


if __name__ == "__main__":
    seed = 2024
    lambda_ = 0.
    rng = jax.random.PRNGKey(seed)

    # load environment
    pomdp, _ = load_pomdp('switching', rng)

    mem_fn = switching_optimal_deterministic_1_bit_mem()
    pi = switching_two_thirds_right_policy()

    mem_params = reverse_softmax(mem_fn)
    pi_params = reverse_softmax(pi)

    epsilons = jnp.linspace(0, 0.5, num=32)

    # TODO: vmap this!
    epsilon = 0.1
    eps_mem_fn = epsilon_interpolate_deterministic_dist(mem_fn, epsilon)
    eps_mem_params = reverse_softmax(eps_mem_fn)

    eps_mem_aug_pomdp = memory_cross_product(eps_mem_params, pomdp)
    mem_aug_pi_params = pi_params.repeat(eps_mem_params.shape[-1], axis=0)
    mem_aug_pi = jax.nn.softmax(mem_aug_pi_params, axis=-1)

    lambda_v_vals, lambda_q_vals, _ = lstdq_lambda(pi, pomdp, lambda_=lambda_)

    eps_mem_lambda_v_vals, eps_mem_lambda_q_vals, _ = lstdq_lambda(mem_aug_pi, eps_mem_aug_pomdp, lambda_=lambda_)
