"""
This file runs a memory iteration with a batch of randomized memory-augmented initial policies,
as well as the TD optimal policy, on a list of different measures.

"""
import argparse
from functools import partial
from time import time
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad, nn
from jax import config
from jax.debug import print
from jax_tqdm import scan_tqdm
import optax

from grl.environment import load_pomdp
from grl.mdp import POMDP
from grl.utils.lambda_discrep import log_all_measures
from grl.memory import memory_cross_product
from grl.utils.file_system import results_path, numpyify_and_save
from grl.utils.math import reverse_softmax
from grl.loss import (
    pg_objective_func,
    mem_tde_loss,
    mem_discrep_loss,
    mem_bellman_loss,
    mem_state_discrep,
    obs_space_mem_discrep_loss
)
from grl.utils.optimizer import get_optimizer
from grl.utils.policy import get_unif_policies


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

    parser.add_argument('--spec', default='tmaze_5_two_thirds_up', type=str,
                        help='name of POMDP spec; evals Pi_phi policies by default')
    parser.add_argument('--mi_iterations', type=int, default=1,
                        help='For memory iteration, how many iterations of memory iterations do we do?')
    parser.add_argument('--mi_steps', type=int, default=20000,
                        help='For memory iteration, how many steps of memory improvement do we do per iteration?')
    parser.add_argument('--save_mem_freq', type=int, default=100,
                        help='How often do we save our mem params during updates?')
    parser.add_argument('--pi_steps', type=int, default=10000,
                        help='For memory iteration, how many steps of policy improvement do we do per iteration?')


    parser.add_argument('--policy_optim_alg', type=str, default='policy_iter',
                        help='policy improvement algorithm to use. "policy_iter" - policy iteration, "policy_grad" - policy gradient, '
                             '"discrep_max" - discrepancy maximization, "discrep_min" - discrepancy minimization')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='What optimizer do we use? (sgd | adam | rmsprop)')

    parser.add_argument('--random_policies', default=100, type=int,
                        help='How many random policies do we use for random kitchen sinks??')
    parser.add_argument('--leave_out_optimal', action='store_true',
                        help="Do we include the optimal policy when we select the initial policy")
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

    # CURRENTLY NOT USED
    parser.add_argument('--objective', default='ld', choices=['ld', 'tde', 'tde_residual', 'mem_state_discrep'])

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


def get_mem_aug_kitchen_sink_policy(mem_params: jnp.ndarray,
                                    mem_aug_pis: jnp.ndarray,
                                    pomdp: POMDP, mem_measure: Callable):
    # batch_measures = jax.vmap(mem_measure, in_axes=(None, 0, None))
    # all_policy_measures = batch_measures(mem_params, mem_aug_pis, pomdp)
    # return mem_aug_pis[jnp.argmax(all_policy_measures)]
    mem_policies = mem_aug_pis[:, ::mem_params.shape[-1]].repeat(mem_params.shape[-1], axis=1)
    batch_measures = jax.vmap(mem_measure, in_axes=(None, 0, None))
    all_policy_measures = batch_measures(mem_params, mem_policies, pomdp)
    return mem_policies[jnp.argmax(all_policy_measures)]


loss_map = {
    'ld': mem_discrep_loss,
    'tde': mem_tde_loss,
    'tde_residual': mem_tde_loss,
    'mem_state_discrep': mem_state_discrep
}


def make_experiment(args):

    # Get POMDP definition
    pomdp, pi_dict = load_pomdp(args.spec,
                                memory_id=0,
                                n_mem_states=args.n_mem_states,
                                corridor_length=args.tmaze_corridor_length,
                                discount=args.tmaze_discount,
                                junction_up_pi=args.tmaze_junction_up_pi)

    objective = args.objective


    def experiment(rng: random.PRNGKey):
        info = {}

        batch_log_all_measures = jax.vmap(log_all_measures, in_axes=(None, 0))

        rng, mem_rng = random.split(rng)


        beginning_info = {}
        rng, pi_rng = random.split(rng)
        mem_aug_pi_shape = (int(pomdp.observation_space.n * args.n_mem_states), pomdp.action_space.n)

        # pi_paramses is args.random_policies x (n_obs * n_mem_states) x n_actions
        pi_paramses = reverse_softmax(get_unif_policies(pi_rng, mem_aug_pi_shape, args.random_policies + 1))
        updateable_pi_params = pi_paramses[-1]

        # We initialize mem params
        mem_shape = (pomdp.action_space.n, pomdp.observation_space.n, args.n_mem_states, args.n_mem_states)
        mem_params = random.normal(mem_rng, shape=mem_shape) * 0.5

        init_mem_aug_pomdp = memory_cross_product(mem_params, pomdp)

        beginning_info['pi_params'] = pi_paramses.copy()
        beginning_info['measures'] = batch_log_all_measures(init_mem_aug_pomdp, pi_paramses)
        beginning_info['init_mem_params'] = mem_params.copy()
        info['beginning'] = beginning_info

        pi_optim = get_optimizer(args.optimizer, args.pi_lr)
        mi_optim = get_optimizer(args.optimizer, args.mi_lr)

        pi_tx_params = pi_optim.init(updateable_pi_params)

        print("Running initial policy improvement")
        @scan_tqdm(args.pi_steps)
        def update_pg_step(inps, i):
            params, tx_params, pomdp = inps
            outs, params_grad = value_and_grad(pg_objective_func, has_aux=True)(params, pomdp)
            v_0, (td_v_vals, td_q_vals) = outs

            # We add a negative here to params_grad b/c we're trying to
            # maximize the PG objective (value of start state).
            params_grad = -params_grad
            updates, tx_params = pi_optim.update(params_grad, tx_params, params)
            params = optax.apply_updates(params, updates)
            outs = (params, tx_params, pomdp)
            return outs, {'v0': v_0, 'v': td_v_vals, 'q': td_q_vals}

        output_pi_tuple, init_pi_optim_info = jax.lax.scan(update_pg_step,
                                                           (updateable_pi_params, pi_tx_params, init_mem_aug_pomdp),
                                                           jnp.arange(args.pi_steps),
                                                           length=args.pi_steps)

        init_optimal_pi_params, _, _ = output_pi_tuple

        after_pi_op_info = {}
        after_pi_op_info['initial_improvement_pi_params'] = init_optimal_pi_params
        after_pi_op_info['initial_improvement_measures'] = log_all_measures(init_mem_aug_pomdp, init_optimal_pi_params)
        print("Learnt initial improvement policy over random mem:\n{}", nn.softmax(init_optimal_pi_params, axis=-1))

        pis_with_init_optimal = pi_paramses.at[-1].set(init_optimal_pi_params)

        after_pi_op_info['all_tested_pi_params'] = pis_with_init_optimal
        info['after_pi_op'] = after_pi_op_info

        if args.leave_out_optimal:
            pis_with_init_optimal = pi_paramses[:-1]

        # now we get our kitchen sink policies
        kitchen_sinks_info = {}
        measure_pi_params = get_mem_aug_kitchen_sink_policy(mem_params, pis_with_init_optimal, pomdp, loss_map[objective])

        pis_to_learn_mem = measure_pi_params

        kitchen_sinks_info[objective] = measure_pi_params.copy()

        mem_tx_params = mi_optim.init(mem_params)

        info['after_kitchen_sinks'] = kitchen_sinks_info

        # Set up for batch memory iteration
        def update_mem_step(mem_params: jnp.ndarray,
                            pi_params: jnp.ndarray,
                            mem_tx_params: jnp.ndarray,
                            objective: str = 'ld',
                            residual: bool = False):
            partial_kwargs = {
                'value_type': args.value_type,
                'error_type': args.error_type,
                'lambda_0': args.lambda_0,
                'lambda_1': args.lambda_1,
                'alpha': args.alpha,
            }
            mem_loss_fn = mem_discrep_loss
            if objective == 'bellman':
                mem_loss_fn = mem_bellman_loss
                partial_kwargs['residual'] = residual
            elif objective == 'tde':
                mem_loss_fn = mem_tde_loss
                partial_kwargs['residual'] = residual
            elif objective == 'obs_space':
                mem_loss_fn = obs_space_mem_discrep_loss

            pi = jax.nn.softmax(pi_params, axis=-1)
            loss, params_grad = value_and_grad(mem_loss_fn, argnums=0)(mem_params, pi, pomdp)

            updates, mem_tx_params = mi_optim.update(params_grad, mem_tx_params, mem_params)
            new_mem_params = optax.apply_updates(mem_params, updates)

            return new_mem_params, pi_params, mem_tx_params, loss

        # Make our vmapped memory function
        update_step = partial(update_mem_step, objective=objective, residual='residual' in objective)

        def scan_wrapper(inps, i, f: Callable):
            mem_params, pi_params, mem_tx_params = inps
            new_mem_params, pi_params, mem_tx_params, loss = f(mem_params, pi_params, mem_tx_params)
            return (new_mem_params, pi_params, mem_tx_params), (loss, new_mem_params)

        scan_tqdm_dec = scan_tqdm(args.mi_steps)
        update_step = scan_tqdm_dec(partial(scan_wrapper, f=update_step))

        # mem_aug_pi_paramses = new_pi_over_mem(pis_to_learn_mem, args.n_mem_states)
        mem_aug_pi_paramses = pis_to_learn_mem
        mem_input_tuple = (mem_params, mem_aug_pi_paramses, mem_tx_params)

        # Memory iteration for all of our measures
        print("Starting {} iterations of memory improvement", args.mi_steps)
        updated_mem_out, (losses, all_mem_params) = jax.lax.scan(update_step, mem_input_tuple, jnp.arange(args.mi_steps), length=args.mi_steps)
        updated_mem_params, updated_mem_pi_params, _ = updated_mem_out
        updated_mem_aug_pomdp = memory_cross_product(updated_mem_params, pomdp)
        updated_mem_info = {'mems': updated_mem_params,
                            'all_mem_params': all_mem_params[::args.save_mem_freq],
                            'measures': log_all_measures(updated_mem_aug_pomdp, updated_mem_pi_params)}

        info['after_mem_op'] = updated_mem_info

        def cross_and_improve_pi(mem_params: jnp.ndarray,
                                 pi_params: jnp.ndarray,
                                 pi_tx_params: dict):
            mem_pomdp = memory_cross_product(mem_params, pomdp)

            output_pi_tuple, pi_optim_info = jax.lax.scan(update_pg_step,
                                                               (pi_params, pi_tx_params, mem_pomdp),
                                                               jnp.arange(args.pi_steps),
                                                               length=args.pi_steps)
            return output_pi_tuple, pi_optim_info

        # now we do policy improvement over the learnt memory
        # reset pi indices, and mem_augment
        rng, pi_rng = random.split(rng)
        new_pi_paramses = reverse_softmax(get_unif_policies(pi_rng, mem_aug_pi_shape, 1))[0]
        mem_aug_pi_paramses = new_pi_paramses
        # mem_aug_pi_paramses = new_pi_paramses.repeat(mem_params.shape[-1], axis=0)

        # Use the same initial random pi params across all final policy improvements.
        all_mem_aug_pi_params = mem_aug_pi_paramses
        all_mem_pi_tx_paramses = pi_optim.init(all_mem_aug_pi_params)

        # Batch policy improvement with PG
        all_improved_pi_tuple, all_improved_pi_info = cross_and_improve_pi(updated_mem_params, all_mem_aug_pi_params,
                                                                           all_mem_pi_tx_paramses)

        # Retrieve each set of learned pi params
        all_improved_pi_params, _, _ = all_improved_pi_tuple
        mem_improved_pi_params = all_improved_pi_params

        final_info = {
            'improved_mem': {'pi_params': mem_improved_pi_params,
                             'measures': log_all_measures(updated_mem_aug_pomdp, mem_improved_pi_params)},
        }

        info['final'] = final_info

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
    # results will be batched over (n_seeds, random_policies + 1).
    # The + 1 is for the TD optimal policy.
    outs = jax.block_until_ready(experiment_vjit_fn(exp_rngs))

    time_finish = time()

    results_path = results_path(args, entry_point=args.objective)
    info = {'logs': outs, 'args': args.__dict__}

    end_time = time()
    run_stats = {'start_time': start_time, 'end_time': end_time}
    info['run_stats'] = run_stats

    def perf_from_stats(stats: dict) -> float:
        return (stats['state_vals']['v'] * stats['p0']).sum(axis=-1).mean().item()

    print("Finished Memory Iteration.")
    print(f"Average performance across initial policies: {perf_from_stats(outs['beginning']['measures']['values']):.4f}")
    print(
        f"Initial improvement performance: {perf_from_stats(outs['after_pi_op']['initial_improvement_measures']['values']):.4f}"
    )
    print(f"Final performance after MI: {perf_from_stats(outs['final']['improved_mem']['measures']['values']):.4f}")

    print(f"Saving results to {results_path}")
    numpyify_and_save(results_path, info)
