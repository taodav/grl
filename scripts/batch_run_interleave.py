"""
This file runs a memory iteration with a batch of randomized initial policies,
as well as the TD optimal policy, on a list of different measures.

"""
import argparse
from functools import partial
from time import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad
from jax import config
from jax.debug import print
from jax_tqdm import scan_tqdm
import optax

from grl.environment import load_pomdp
from grl.utils.lambda_discrep import log_all_measures, augment_and_log_all_measures
from grl.memory import memory_cross_product
from grl.utils.file_system import results_path, numpyify_and_save
from grl.loss import (
    pg_objective_func,
    discrep_loss,
    mstd_err,
    variance_loss,
    mem_tde_loss,
    mem_discrep_loss,
    mem_bellman_loss,
    obs_space_mem_discrep_loss,
    mem_variance_loss
)
from grl.utils.optimizer import get_optimizer
from grl.vi import policy_iteration_step


def get_args():
    # Args
    parser = argparse.ArgumentParser()
    # yapf:disable

    # hyperparams for tmaze_hperparams
    parser.add_argument('--tmaze_corridor_length',
                        default=None,
                        type=int,
                        help='Length of corridor for tmaze_hyperparams')
    parser.add_argument('--tmaze_discount',
                        default=None,
                        type=float,
                        help='Discount rate for tmaze_hyperparams')
    parser.add_argument('--tmaze_junction_up_pi',
                        default=None,
                        type=float,
                        help='probability of traversing up at junction for tmaze_hyperparams')

    parser.add_argument('--spec', default='example_11', type=str,
                        help='name of POMDP spec; evals Pi_phi policies by default')
    parser.add_argument('--mi_steps', type=int, default=20000,
                        help='For memory iteration, how many steps of memory improvement do we do per iteration?')

    parser.add_argument('--policy_optim_alg', type=str, default='policy_grad',
                        help='policy improvement algorithm to use. "policy_iter" - policy iteration, "policy_grad" - policy gradient, '
                             '"discrep_max" - discrepancy maximization, "discrep_min" - discrepancy minimization')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='What optimizer do we use? (sgd | adam | rmsprop)')

    parser.add_argument('--n_mem_states', default=2, type=int,
                        help='for memory_id = 0, how many memory states do we have?')

    parser.add_argument('--lambda_0', default=0., type=float,
                        help='First lambda parameter for lambda-discrep')
    parser.add_argument('--lambda_1', default=1., type=float,
                        help='Second lambda parameter for lambda-discrep')

    parser.add_argument('--alpha', default=1., type=float,
                        help='Temperature parameter, for how uniform our lambda-discrep weighting is')
    parser.add_argument('--pi_lr', default=0.01, type=float)
    parser.add_argument('--mi_lr', default=0.01, type=float)
    parser.add_argument('--value_type', default='q', type=str,
                        help='Do we use (v | q) for our discrepancies?')
    parser.add_argument('--error_type', default='l2', type=str,
                        help='Do we use (l2 | abs) for our discrepancies?')
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='(POLICY ITERATION AND TMAZE_EPS_HYPERPARAMS ONLY) What epsilon do we use?')
    parser.add_argument('--reward_in_obs', action='store_true',
                        help='Do we add reward into observation?')

    parser.add_argument('--objective', default='ld', choices=['ld', 'tde', 'tde_residual', 'variance'])

    parser.add_argument('--log_every', default=500, type=int,
                        help='How many logs do we keep?')
    parser.add_argument('--study_name', default=None, type=str,
                        help='name of the experiment. Results saved to results/{experiment_name} directory if not None. Else, save to results directory directly.')
    parser.add_argument('--platform', default='cpu', type=str,
                        help='What platform do we run things on? (cpu | gpu)')
    parser.add_argument('--seed', default=2024, type=int,
                        help='What is the overall seed we use?')
    parser.add_argument('--n_seeds', default=1, type=int,
                        help='How many seeds do we run?')

    args = parser.parse_args()
    return args

def make_experiment(args):

    loss_map = {
        'ld': discrep_loss,
        'tde': mstd_err,
        'tde_residual': mstd_err,
        'variance': variance_loss
    }

    # Get POMDP definition
    pomdp, pi_dict = load_pomdp(args.spec,
                                memory_id=0,
                                n_mem_states=args.n_mem_states,
                                corridor_length=args.tmaze_corridor_length,
                                discount=args.tmaze_discount,
                                junction_up_pi=args.tmaze_junction_up_pi)

    partial_kwargs = {
        'value_type': args.value_type,
        'error_type': args.error_type,
        'lambda_0': args.lambda_0,
        'lambda_1': args.lambda_1,
        'alpha': args.alpha,
    }
    mem_loss_fn = mem_discrep_loss
    if args.objective == 'bellman':
        mem_loss_fn = mem_bellman_loss
        partial_kwargs['residual'] = args.residual
    elif args.objective == 'tde':
        mem_loss_fn = mem_tde_loss
        partial_kwargs['residual'] = args.residual
    elif args.objective == 'obs_space':
        mem_loss_fn = obs_space_mem_discrep_loss
    elif args.objective == 'variance':
        mem_loss_fn = mem_variance_loss
    mem_loss_fn = partial(mem_loss_fn, **partial_kwargs)

    def experiment(rng: random.PRNGKey):
        info = {}

        rng, mem_rng = random.split(rng)
        mem_shape = (pomdp.action_space.n, pomdp.observation_space.n, args.n_mem_states, args.n_mem_states)
        mem_params = random.normal(mem_rng, shape=mem_shape) * 0.5

        beginning_info = {}
        beginning_info['mem_params'] = mem_params.copy()

        rng, pi_rng = random.split(rng)
        pi_shape = (pomdp.observation_space.n, pomdp.action_space.n)
        pi_params = random.normal(pi_rng, shape=pi_shape) * 0.5

        beginning_info['pi_params'] = pi_params.copy()
        beginning_info['measures'] = log_all_measures(pomdp, pi_params)
        mem_aug_pi_params = pi_params.repeat(mem_params.shape[-1], axis=0)
        info['beginning'] = beginning_info

        pi_optim = get_optimizer(args.optimizer, args.pi_lr)
        mi_optim = get_optimizer(args.optimizer, args.mi_lr)

        mem_tx_params = mi_optim.init(mem_params)
        pi_tx_params = pi_optim.init(mem_aug_pi_params)

        def update_pg_step(params, tx_params, pomdp):
            outs, params_grad = value_and_grad(pg_objective_func, has_aux=True)(params, pomdp)
            v_0, (td_v_vals, td_q_vals) = outs

            # We add a negative here to params_grad b/c we're trying to
            # maximize the PG objective (value of start state).
            params_grad = -params_grad
            updates, tx_params = pi_optim.update(params_grad, tx_params, params)
            params = optax.apply_updates(params, updates)
            outs = (params, tx_params, pomdp)
            return outs, {'v0': v_0, 'v': td_v_vals, 'q': td_q_vals}

        def update_policy_iter_step(params, tx_params, pomdp):
            new_pi_params, prev_td_v_vals, prev_td_q_vals = policy_iteration_step(params, pomdp, eps=args.epsilon)
            outs = (new_pi_params, _, pomdp)
            return outs, {'v': prev_td_v_vals, 'q': prev_td_q_vals}

        @scan_tqdm(args.mi_steps)
        def policy_improvement_scan_wrapper(inps, i):
            params, tx_params, pomdp = inps
            outs, info = update_pg_step(params, tx_params, pomdp)
            return outs, info

        print("Finding memoryless optimal policy")
        memoryless_pi_tx_params = pi_optim.init(pi_params)
        memoryless_pi_improvement_outs, memoryless_pi_improvement_info = \
            jax.lax.scan(policy_improvement_scan_wrapper, (pi_params, memoryless_pi_tx_params, pomdp),
                         jnp.arange(args.mi_steps), length=args.mi_steps)
        memoryless_optimal_pi_params, _, _ = memoryless_pi_improvement_outs
        after_pi_op_info = {
            'pi_params': memoryless_optimal_pi_params,
            'measures': log_all_measures(pomdp, memoryless_optimal_pi_params),
            'update_logs': jax.tree_util.tree_map(lambda x: x[::args.log_every], memoryless_pi_improvement_info)
        }
        info['after_pi_op'] = after_pi_op_info

        print("Running policy and memory improvement")

        @scan_tqdm(args.mi_steps)
        def update(inps, i):
            mem_params, pi_params, mem_tx_params, pi_tx_params = inps

            # Set up for batch memory iteration
            def update_mem_step(mem_params: jnp.ndarray,
                                pi_params: jnp.ndarray,
                                mem_tx_params: jnp.ndarray):
                pi = jax.nn.softmax(pi_params, axis=-1)
                loss, params_grad = value_and_grad(mem_loss_fn, argnums=0)(mem_params, pi, pomdp)

                updates, mem_tx_params = mi_optim.update(params_grad, mem_tx_params, mem_params)
                new_mem_params = optax.apply_updates(mem_params, updates)

                return new_mem_params, pi_params, mem_tx_params, loss

            # update memory
            new_mem_params, _, new_mem_tx_params, loss = update_mem_step(mem_params, pi_params, mem_tx_params)

            # TODO: potentially remove this? does update_mem_step actually have a mem augmented POMDP already?
            mem_pomdp = memory_cross_product(new_mem_params, pomdp)

            # update policy
            output_pi_tuple, pi_optim_info = update_pg_step(pi_params, pi_tx_params, mem_pomdp)

            new_pi_params, new_pi_tx_params, _ = output_pi_tuple

            outs = (new_mem_params, new_pi_params, new_mem_tx_params, new_pi_tx_params)
            return outs, {'pi_info': pi_optim_info, 'mem_loss': loss}

        mem_input_tuple = (mem_params, mem_aug_pi_params, mem_tx_params, pi_tx_params)

        # Memory iteration for all of our measures
        print("Starting {} iterations of interleaving λ-discrepancy minimization and pi optimization", args.mi_steps)
        out_tuple, update_info = jax.lax.scan(update, mem_input_tuple, jnp.arange(args.mi_steps), length=args.mi_steps)

        final_mem_params, final_pi_params, _, _ = out_tuple

        mem_info = {'mem_params': final_mem_params,
                    'pi_params': final_pi_params,
                    'measures': augment_and_log_all_measures(final_mem_params, pomdp, final_pi_params),
                    'update_logs': jax.tree_util.tree_map(lambda x: x[::args.log_every], update_info)}

        info['final'] = mem_info

        return info

    return experiment


if __name__ == "__main__":
    start_time = time()
    # jax.disable_jit(True)

    args = get_args()

    np.set_printoptions(precision=4, suppress=True)
    config.update('jax_platform_name', args.platform)
    config.update("jax_enable_x64", True)

    rng = random.PRNGKey(seed=args.seed)
    rngs = random.split(rng, args.n_seeds + 1)
    rng, exp_rngs = rngs[-1], rngs[:-1]

    t0 = time()
    experiment_vjit_fn = jax.jit(jax.vmap(make_experiment(args)))

    # Run the experiment!
    outs = jax.block_until_ready(experiment_vjit_fn(exp_rngs))

    time_finish = time()
    begin = outs['beginning']['measures']['values']
    begin_perf = (begin['state_vals']['v'] * begin['p0']).mean(axis=0).sum()
    pi_opt = outs['after_pi_op']['measures']['values']
    pi_opt_perf = (pi_opt['state_vals']['v'] * pi_opt['p0']).mean(axis=0).sum()
    final = outs['final']['measures']['values']
    final_perf = (final['state_vals']['v'] * final['p0']).mean(axis=0).sum()
    print(f"Performances over {args.n_seeds} seeds.\n"
          f"Beginning performance: {begin_perf.item()}\n"
          f"Memoryless optimal performance: {pi_opt_perf.item()}\n"
          f"Ending performance: {final_perf.item()}")

    results_path = results_path(args, entry_point='batch_run')
    info = {'logs': outs, 'args': args.__dict__}

    end_time = time()
    run_stats = {'start_time': start_time, 'end_time': end_time}
    info['run_stats'] = run_stats

    print(f"Saving results to {results_path}")
    numpyify_and_save(results_path, info)
