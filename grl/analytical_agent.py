import optax
import haiku as hk
import jax.numpy as jnp
from jax import random, jit, value_and_grad, nn
from typing import Sequence
from optax import GradientTransformation
from functools import partial

from .policy_eval import analytical_pe
from .memory import functional_memory_cross_product

# def analytical_mem_func(mem_shape: Sequence[int]):
#     # init is only called in f.init. We replace this later if we fix our memory function.
#     mem_init = hk.initializers.VarianceScaling(jnp.sqrt(2), 'fan_avg', 'uniform')
#     mem_params = hk.get_parameter("mem", mem_shape, init=mem_init)
#
#     T_mem = nn.softmax(mem_params, axis=-1)
#     T_x, R_x, p0_x, phi_x = functional_memory_cross_product(T, T_mem, phi, R, p0)
#     _, mc_vals, td_vals = analytical_pe(pi, phi, T, R, p0, gamma, T.shape[-1], phi.shape[-1])


def normalize_rows(mat: jnp.ndarray):
    # Normalize (assuming params are probability distribution)
    mat = mat.clip(0, 1)
    denom = mat.sum(axis=-1, keepdims=True)
    denom_no_zero = denom + (denom == 0).astype(denom.dtype)
    mat /= denom_no_zero
    return mat

class AnalyticalAgent:
    def __init__(self, n_actions: int,
                 obs_shape: Sequence[int],
                 rand_key: random.PRNGKey,
                 pi_function: hk.Transformed,
                 mem_input_shape: Sequence[int],
                 mem_function: hk.Transformed = None,
                 pi_optimizer: GradientTransformation = None,
                 mem_optimizer: GradientTransformation = None,
                 discrep_type: str = "l2"):
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self._rand_key, pi_rand_key, mem_rand_key = rand_key.split(rand_key, 3)
        self.discrep_type = discrep_type
        self.loss_fn = self.functional_mse_loss
        if self.discrep_type == 'max':
            self.functional_loss_fn = self.functional_max_loss

        self.pi_function = pi_function
        # TODO: refactor when we need to start appending memory.
        self.pi_params = self.pi_function.init(rng=pi_rand_key, x=jnp.zeros((1, *self.obs_shape)))

        # do we use memory?
        self.mem_function, self.mem_params = None, None
        if mem_function is not None:
            self.mem_function = mem_function
            self.mem_params = self.mem_function.init(rng=mem_rand_key, x=jnp.zeros((1, *mem_input_shape)))

        # do we learn memory?
        self.mem_optimizer = None
        if mem_optimizer is None:
            self.mem_optimizer = mem_optimizer

        # do we learn policy?
        self.pi_optimizer = None
        if pi_optimizer is None:
            self.pi_optimizer = pi_optimizer

    def policy_update(self, params: jnp.ndarray, value_type: str, lr: float, *args, **kwargs):
        return self.functional_policy_update(params, value_type, self.amdp.gamma, lr, self.amdp.T,
                                             self.amdp.R, self.amdp.phi, self.amdp.p0)

    @partial(jit, static_argnames=['self', 'gamma', 'value_type', 'lr'])
    def functional_policy_update(self, params: jnp.ndarray, value_type: str, gamma: float,
                                 lr: float, T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray,
                                 p0: jnp.ndarray):
        loss, params_grad = value_and_grad(self.functional_loss_fn,
                                           argnums=0)(params, value_type, phi, T, R, p0, gamma)
        params -= lr * params_grad

        # Normalize (assuming params are probability distribution)
        params = normalize_rows(params)
        return loss, params

    def memory_update(self, T_mem: jnp.ndarray, value_type: str, lr: float, pi: jnp.ndarray):
        return self.functional_memory_update(T_mem, value_type, self.amdp.gamma, lr, pi,
                                             self.amdp.T, self.amdp.R, self.amdp.phi, self.amdp.p0)

    @partial(jit, static_argnames=['self', 'gamma', 'value_type', 'lr'])
    def functional_memory_update(self, params: jnp.ndarray, value_type: str, gamma: float,
                                 lr: float, pi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray,
                                 phi: jnp.ndarray, p0: jnp.ndarray):
        loss, params_grad = value_and_grad(self.functional_memory_loss,
                                           argnums=0)(params, gamma, value_type, pi, T, R, phi, p0)
        params -= lr * params_grad

        # Normalize (assuming params are probability distribution)
        params = normalize_rows(params)
        return loss, params

    @partial(jit, static_argnames=['self', 'gamma', 'value_type'])
    def functional_memory_loss(self, T_mem: jnp.ndarray, gamma: float, value_type: str,
                               pi: jnp.ndarray, T: jnp.ndarray, R: jnp.ndarray, phi: jnp.ndarray,
                               p0: jnp.ndarray):
        T_x, R_x, p0_x, phi_x = functional_memory_cross_product(T, T_mem, phi, R, p0)
        return self.loss_fn(pi, value_type, phi_x, T_x, R_x, p0_x, gamma)

    @partial(jit, static_argnames=['self', 'value_type', 'gamma'])
    def functional_mse_loss(self, pi: jnp.ndarray, value_type: str, phi: jnp.ndarray,
                            T: jnp.ndarray, R: jnp.ndarray, p0: jnp.ndarray, gamma: float):
        _, mc_vals, td_vals = analytical_pe(pi, phi, T, R, p0, gamma, T.shape[-1], phi.shape[-1])
        diff = mc_vals[value_type] - td_vals[value_type]
        return (diff**2).mean()

    @partial(jit, static_argnames=['self', 'value_type', 'gamma'])
    def functional_max_loss(self, pi: jnp.ndarray, value_type: str, phi: jnp.ndarray,
                            T: jnp.ndarray, R: jnp.ndarray, p0: jnp.ndarray, gamma: float):
        _, mc_vals, td_vals = analytical_pe(pi, phi, T, R, p0, gamma, T.shape[-1], phi.shape[-1])
        return jnp.abs(mc_vals[value_type] - td_vals[value_type]).max()
