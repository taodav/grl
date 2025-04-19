import jax.numpy as jnp

from .bellman import bellman_loss, mem_bellman_loss
from .disc_count import disc_count_loss, mem_disc_count_loss
from .gvf import gvf_loss, mem_gvf_loss, sr_discrep_loss_peter, mem_sr_discrep_loss, mem_sr_discrep_loss_for_gamma_optim
from .ld import discrep_loss, mem_discrep_loss, obs_space_mem_discrep_loss, policy_discrep_loss
from .pg import pg_objective_func, augmented_pg_objective_func, mem_pg_objective_func, unrolled_mem_pg_objective_func
from .samples import mse, seq_sarsa_loss, seq_sarsa_mc_loss, seq_sarsa_lambda_discrep
from .state import mem_state_discrep
from .td import mstd_err, mem_tde_loss
from .value import value_error
from .variance import variance_loss, mem_variance_loss
from .entropy_loss import entropy_loss, mem_entropy_loss


def mem_dummy_loss(mem_params, pi, pomdp, **kwargs):
    return jnp.array(0, dtype=float)


def dummy_loss(pi, pomdp, **kwargs):
    return jnp.array(0, dtype=float), None, None
