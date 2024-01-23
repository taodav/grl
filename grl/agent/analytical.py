from functools import partial
from typing import Sequence

import jax.numpy as jnp
from jax import random, jit, value_and_grad
from jax.nn import softmax
import numpy as np
import optax

from grl.mdp import POMDP
from grl.utils.loss import policy_discrep_loss, pg_objective_func, \
    mem_pg_objective_func, unrolled_mem_pg_objective_func
from grl.utils.loss import mem_discrep_loss, mem_bellman_loss, mem_tde_loss, obs_space_mem_discrep_loss
from grl.utils.math import glorot_init, reverse_softmax
from grl.utils.optimizer import get_optimizer
from grl.vi import policy_iteration_step

def new_pi_over_mem(rand_key: random.PRNGKey,
        pi_params: jnp.ndarray, n_mem: int,
        new_mem_type: str = 'random'):
    old_pi_params_shape = pi_params.shape

    pi_params = pi_params.repeat(n_mem, axis=0)

    if new_mem_type == 'random':
        # randomly init policy for new memory state
        new_mem_params = glorot_init(rand_key, old_pi_params_shape)
        pi_params = pi_params.at[1::2].set(new_mem_params)
    return pi_params

class AnalyticalAgent:
    """
    Analytical agent that learns optimal policy params based on an
    analytic policy gradient.
    """
    def __init__(self,
                 optim_str: str = 'adam',
                 pi_lr: float = 1.,
                 mi_lr: float = 1.,
                 value_type: str = 'v',
                 error_type: str = 'l2',
                 objective: str = 'discrep',
                 residual: bool = False,
                 lambda_0: float = 0.,
                 lambda_1: float = 1.,
                 alpha: float = 1.,
                 pi_softmax_temp: float = 1,
                 policy_optim_alg: str = 'policy_iter',
                 new_mem_pi: str = 'copy',
                 epsilon: float = 0.1,
                 flip_count_prob: bool = False):
        """
        :param value_type: If we optimize lambda discrepancy, what type of lambda discrepancy do we optimize? (v | q)
        :param error_type: lambda discrepancy error type (l2 | abs)
        :param objective: What objective are we trying to minimize? (discrep | bellman | tde)
        :param pi_softmax_temp: When we take the softmax over pi_params, what is the softmax temperature?
        :param policy_optim_alg: What type of policy optimization do we do? (pi | pg)
            (discrep_max: discrepancy maximization | discrep_min: discrepancy minimization
            | policy_iter: policy iteration | policy_grad: policy gradient)
        :param new_mem_pi: When we do memory iteration and add memory states, how do we initialize the new policy params
                           over the new memory states? (copy | random)
        :param epsilon: (POLICY ITERATION ONLY) When we perform policy iteration, what epsilon do we use?
        :param flip_count_prob: For our memory loss, do we flip our count probabilities??
        """
        self.policy_optim_alg = policy_optim_alg

        self.epsilon = epsilon

        self.val_type = value_type
        self.error_type = error_type
        self.objective = objective
        self.residual = residual
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.alpha = alpha
        self.flip_count_prob = flip_count_prob

        self.policy_discrep_objective_func = None
        self.memory_objective_func = None

        self.new_mem_pi = new_mem_pi

        self.mi_lr = mi_lr

        # initialize optimizers
        self.optim_str = optim_str
        self.pi_lr = pi_lr

        self.pi_softmax_temp = pi_softmax_temp

        self.pi_optim = get_optimizer(optim_str, self.pi_lr)
        self.mem_optim = get_optimizer(optim_str, self.mi_lr)

        if hasattr(self, 'weight_discrep'):
            if self.weight_discrep:
                self.alpha = 0.
            else:
                self.alpha = 1.
            self.flip_count_prob = False
            del self.weight_discrep

        partial_policy_discrep_loss = partial(policy_discrep_loss,
                                              value_type=self.val_type,
                                              error_type=self.error_type,
                                              alpha=self.alpha,
                                              flip_count_prob=self.flip_count_prob)
        self.policy_discrep_objective_func = partial_policy_discrep_loss

        mem_loss_fn = mem_discrep_loss
        partial_kwargs = {
            'value_type': self.val_type,
            'error_type': self.error_type,
            'lambda_0': self.lambda_0,
            'lambda_1': self.lambda_1,
            'alpha': self.alpha,
            'flip_count_prob': self.flip_count_prob
        }
        if hasattr(self, 'objective'):
            if self.objective == 'bellman':
                mem_loss_fn = mem_bellman_loss
                partial_kwargs['residual'] = self.residual
            elif self.objective == 'tde':
                mem_loss_fn = mem_tde_loss
                partial_kwargs['residual'] = self.residual
            elif self.objective == 'obs_space':
                mem_loss_fn = obs_space_mem_discrep_loss

        partial_mem_discrep_loss = partial(mem_loss_fn, **partial_kwargs)
        self.memory_objective_func = partial_mem_discrep_loss

    def policy(self, pi_params: jnp.ndarray) -> jnp.ndarray:
        # return the learnt policy
        return softmax(pi_params, axis=-1)

    def memory(self, mem_params: jnp.ndarray) -> jnp.ndarray:
        return softmax(mem_params, axis=-1)

    def reset_pi_params(self, rand_key: random.PRNGKey, pi_shape: Sequence[int]):
        pi_params = glorot_init(rand_key, pi_shape)
        pi_optim_state = self.pi_optim.init(pi_params)
        return pi_params, pi_optim_state

    def reset_mem_params(self, rand_key: random.PRNGKey, mem_shape: Sequence[int]):
        mem_params = glorot_init(rand_key, mem_shape)
        mem_optim_state = self.mem_optim.init(mem_params)
        return mem_params, mem_optim_state

    def policy_gradient_update(self, params: jnp.ndarray, tx_params: jnp.ndarray, pomdp: POMDP):
        pg_func = pg_objective_func
        if self.policy_optim_alg == 'policy_mem_grad':
            pg_func = mem_pg_objective_func
        elif self.policy_optim_alg == 'policy_mem_grad_unrolled':
            pg_func = unrolled_mem_pg_objective_func

        outs, params_grad = value_and_grad(pg_func, has_aux=True)(params, pomdp)
        v_0, (td_v_vals, td_q_vals) = outs

        # We add a negative here to params_grad b/c we're trying to
        # maximize the PG objective (value of start state).
        params_grad = -params_grad
        updates, tx_params = self.pi_optim.update(params_grad, tx_params, params)
        params = optax.apply_updates(params, updates)

        output = {
            'v_0': v_0,
            'prev_td_q_vals': td_q_vals,
            'prev_td_v_vals': td_v_vals
        }
        return params, tx_params, output

    def policy_discrep_update(self,
                              params: jnp.ndarray,
                              tx_params: jnp.ndarray,
                              pomdp: POMDP,
                              sign: bool = True):
        outs, params_grad = value_and_grad(self.policy_discrep_objective_func,
                                           has_aux=True)(params, pomdp)
        loss, (mc_vals, td_vals) = outs

        # it's the flip of sign b/c the optimizer already applies the negative sign
        params_grad *= (1 - float(sign))

        updates, tx_params = self.pi_optim.update(params_grad, tx_params, params)
        params = optax.apply_updates(params, updates)

        output = {'loss': loss, 'mc_vals': mc_vals, 'td_vals': td_vals}

        return params, tx_params, output
    def policy_iteration_update(self, pi_params: dict,
                                pomdp: POMDP):
        new_pi_params, prev_td_v_vals, prev_td_q_vals = policy_iteration_step(
            pi_params, pomdp, eps=self.epsilon)
        output = {'prev_td_q_vals': prev_td_q_vals, 'prev_td_v_vals': prev_td_v_vals}
        return pi_params, output

    def policy_improvement(self, pi_params: dict,
                           pi_tx_params: optax.Params,
                           pomdp: POMDP):
        if self.policy_optim_alg in ['policy_grad', 'policy_mem_grad', 'policy_mem_grad_unrolled']:
            pi_params, pi_tx_params, output = \
                self.policy_gradient_update(pi_params, pi_tx_params, pomdp)
        elif self.policy_optim_alg == 'policy_iter':
            pi_params, output = self.policy_iteration_update(pi_params, pomdp)
        elif self.policy_optim_alg == 'discrep_max' or self.policy_optim_alg == 'discrep_min':
            pi_params, pi_tx_params, output = self.policy_discrep_update(
                pi_params,
                pi_tx_params,
                pomdp,
                sign=(self.policy_optim_alg == 'discrep_max'))
        else:
            raise NotImplementedError
        return pi_params, pi_tx_params, output

    def memory_update(self, params: jnp.ndarray, optim_state: jnp.ndarray, pi_params: jnp.ndarray,
                      pomdp: POMDP):
        pi = softmax(pi_params / self.pi_softmax_temp, axis=-1)
        loss, params_grad = value_and_grad(self.memory_objective_func, argnums=0)(params, pi,
                                                                                  pomdp)

        updates, optimizer_state = self.mem_optim.update(params_grad, optim_state, params)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    # def __getstate__(self) -> dict:
    #     state = self.__dict__.copy()
    #
    #     # delete unpickleable jitted functions
    #     del state['pg_objective_func']
    #     del state['policy_iteration_update']
    #     del state['policy_discrep_objective_func']
    #     del state['memory_objective_func']
    #     del state['pi_optim']
    #     state['pi_params'] = np.array(state['pi_params'])
    #
    #     if state['mem_params'] is not None:
    #         del state['mem_optim']
    #         state['mem_params'] = np.array(state['mem_params'])
    #     return state
    #
    # def __setstate__(self, state: dict):
    #     self.__dict__.update(state)
    #
    #     # restore jitted functions
    #     self.pg_objective_func = jit(pg_objective_func)
    #     if self.policy_optim_alg == 'policy_mem_grad':
    #         self.pg_objective_func = jit(mem_pg_objective_func)
    #     elif self.policy_optim_alg == 'policy_mem_grad_unrolled':
    #         self.pg_objective_func = jit(unrolled_mem_pg_objective_func)
    #     self.policy_iteration_update = jit(policy_iteration_step, static_argnames=['eps'])
    #
    #     if 'optim_str' not in state:
    #         state['optim_str'] = 'sgd'
    #     self.pi_optim = get_optimizer(state['optim_str'], state['pi_lr'])
    #     if hasattr(self, 'mem_params'):
    #         self.mi_optim = get_optimizer(state['optim_str'], state['mi_lr'])
    #
    #     if not hasattr(self, 'val_type'):
    #         self.val_type = 'v'
    #         self.error_type = 'l2'
    #     if not hasattr(self, 'lambda_0'):
    #         self.lambda_0 = 0.
    #         self.lambda_1 = 1.
    #
    #     self.init_and_jit_objectives()
