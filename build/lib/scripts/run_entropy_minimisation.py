"""
This file runs a memory iteration where the optimisation objective for the
memory is to minimise H(omega_{t+1} | m_t, a_t)
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

from grl.mdp import POMDP, POMDPG
from grl.agent.analytical import new_pi_over_mem
from grl.environment import load_pomdp
from grl.environment.spec import augment_pomdp_gamma, make_subprob_matrix
from grl.mdp import POMDP
from grl.utils.lambda_discrep import log_all_measures, augment_and_log_all_measures
from grl.memory import memory_cross_product
from grl.utils.file_system import results_path, numpyify_and_save
from grl.utils.math import reverse_softmax
from grl.utils.policy_eval import functional_solve_mdp
from grl.loss import entropy_loss, mem_entropy_loss
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
    parser.add_argument('--pi_steps', type=int, default=10000,
                        help='For memory iteration, how many steps of policy improvement do we do per iteration?')

    parser.add_argument('--policy_optim_alg', type=str, default='policy_grad',
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

    parser.add_argument('--pi_lr', default=0.01, type=float)
    parser.add_argument('--mi_lr', default=0.01, type=float)
    parser.add_argument('--reward_in_obs', action='store_true',
                        help='Do we add reward into observation?')

    parser.add_argument('--study_name', default=None, type=str,
                        help='name of the experiment. Results saved to results/{experiment_name} directory if not None. Else, save to results directory directly.')
    parser.add_argument('--platform', default='cpu', type=str,
                        help='What platform do we run things on? (cpu | gpu)')
    parser.add_argument('--seed', default=2024, type=int,
                        help='What is the overall seed we use?')

    args = parser.parse_args()
    return args


def make_experiment(args, rand_key: jax.random.PRNGKey):

    pomdp, pi_dict = load_pomdp(args.spec,
                                memory_id=0,
                                n_mem_states=args.n_mem_states,
                                corridor_length=args.tmaze_corridor_length,
                                discount=args.tmaze_discount,
                                junction_up_pi=args.tmaze_junction_up_pi,
                                reward_in_obs=args.reward_in_obs)
    
    # zero out terminal rows in T
    pomdp.T = make_subprob_matrix(pomdp.T)

    n_s = pomdp.state_space.n
    n_a = pomdp.action_space.n
    n_o = pomdp.observation_space.n

    def experiment(rng: random.PRNGKey):
        rng, mem_rng = random.split(rng)
        rng, pi_rng = random.split(rng)

        #pi_shape = (n_o, n_a)
        #pi_paramses = reverse_softmax(get_unif_policies(pi_rng, pi_shape, args.random_policies + 1))

        # We initialize mem params
        mem_shape = (n_a, n_o, args.n_mem_states, args.n_mem_states)
        mem_params = random.normal(mem_rng, shape=mem_shape) * 0.5

        # memory optimisation
        mi_optim = get_optimizer(args.optimizer, args.mi_lr)
        mem_tx_params = mi_optim.init(mem_params)

        #mem_params = reverse_softmax(get_optimal_one_bit_memory_parity_check())

        @scan_tqdm(args.mi_steps)
        def update_memory_step(inps, i):
            mem_params, mem_tx_params = inps

            loss, params_grad = value_and_grad(mem_entropy_loss, argnums=0)(mem_params, pomdp)

            updates, mem_tx_params, = mi_optim.update(params_grad, mem_tx_params, mem_params)
            new_mem_params = optax.apply_updates(mem_params, updates)

            return (new_mem_params, mem_tx_params), {'loss': loss}
        
        print("Optimising memory...")
        print("Initial loss: {}", mem_entropy_loss(mem_params, pomdp))

        output_mem_tuple, losses = jax.lax.scan(
            update_memory_step,
            (mem_params, mem_tx_params),
            jnp.arange(args.mi_steps),
            length=args.mi_steps
        )

        mem_params, mem_tx_params = output_mem_tuple
        print("Final loss: {}", losses[-1])
        print("Memory function:\n{}", jax.nn.softmax(mem_params))
    
    return experiment

if __name__ == "__main__":
    args = get_args()

    np.set_printoptions(precision=4, suppress=True)
    config.update('jax_platform_name', args.platform)
    config.update("jax_enable_x64", True)

    rng = random.PRNGKey(seed=args.seed)
    rngs = random.split(rng, 2)
    rng, exp_rng = rngs[0], rngs[1]

    rng, make_rng = jax.random.split(rng)

    exp = make_experiment(args, make_rng)

    exp(exp_rng)

