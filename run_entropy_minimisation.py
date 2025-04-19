"""
This file runs a memory iteration where the optimisation objective for the
memory is to minimise H(omega_{t+1} | m_t, a_t)
"""

import argparse

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad
from jax import config
from jax.debug import print
from jax_tqdm import scan_tqdm
from tqdm import tqdm
import optax

from grl.environment import load_pomdp
from grl.utils.policy_eval import functional_solve_mdp
from grl.environment.spec import make_subprob_matrix
from grl.agent.analytical import new_pi_over_mem
from grl.mdp import POMDP
from grl.utils.math import reverse_softmax
from grl.loss import mem_entropy_loss, mem_reward_entropy_loss, pg_objective_func
from grl.utils.optimizer import get_optimizer
from grl.utils.policy import get_unif_policies
from grl.memory import memory_cross_product


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
    
    parser.add_argument('--reward_entropy', action='store_true',
                        help='whether to calculate entropy only over the next reward rather than the full next observation')

    args = parser.parse_args()
    return args

def get_optimal_one_bit_memory_parity_check():
    mem = jnp.zeros((2, 8, 2, 2))  # (action, obs, mem, mem)
    # remember first colour
    mem = mem.at[:, 0, :, 0].set(1.0)
    mem = mem.at[:, 1, :, 1].set(1.0)

    # memory = "do colour families match"
    mem = mem.at[:, 2, 0, 1].set(1.0)  # observations were 0 then 2 -> colours match
    mem = mem.at[:, 3, 0, 0].set(1.0)  # observations were 0 then 3 -> colours don't match
    mem = mem.at[:, 3, 1, 1].set(1.0)  # observations were 1 then 3 -> colours match
    mem = mem.at[:, 2, 1, 0].set(1.0)  # observations were 1 then 2 -> colours don't match
    
    # if observation is 4,5,6,7, just keep the memory
    mem = mem.at[:, 4:, 0, 0].set(1.0)
    mem = mem.at[:, 4:, 1, 1].set(1.0)

    return mem

def get_policy_performance(
        pi_params: jnp.array,
        pomdp: POMDP
    ):
    pi_s = pomdp.phi @ jax.nn.softmax(pi_params, axis=-1)
    v, _ = functional_solve_mdp(pi_s, pomdp)
    performance = (v * pomdp.p0).sum(axis=-1).mean()
    return performance
 

def make_experiment(args, rand_key: jax.random.PRNGKey):

    pomdp, pi_dict = load_pomdp(args.spec,
                                memory_id=0,
                                n_mem_states=args.n_mem_states,
                                corridor_length=args.tmaze_corridor_length,
                                discount=args.tmaze_discount,
                                junction_up_pi=args.tmaze_junction_up_pi,
                                reward_in_obs=args.reward_in_obs)
    
    loss_fn_dict = {
        0: mem_entropy_loss,
        1: mem_reward_entropy_loss
    }

    mem_loss_fn = loss_fn_dict[args.reward_entropy]
    
    # zero out terminal rows in T
    pomdp.T = make_subprob_matrix(pomdp.T)

    n_s = pomdp.state_space.n
    n_a = pomdp.action_space.n
    n_o = pomdp.observation_space.n

    def experiment(rng: random.PRNGKey):
        rng, mem_rng = random.split(rng)
        rng, pi_rng = random.split(rng)

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

            loss, params_grad = value_and_grad(mem_loss_fn, argnums=0)(mem_params, pomdp)

            updates, mem_tx_params, = mi_optim.update(params_grad, mem_tx_params, mem_params)
            new_mem_params = optax.apply_updates(mem_params, updates)

            return (new_mem_params, mem_tx_params), {'loss': loss}
        
        print("Optimising memory...")
        print("Initial loss: {}", mem_loss_fn(mem_params, pomdp))

        #output_mem_tuple, info = python_scan(
        output_mem_tuple, info = jax.lax.scan(
            update_memory_step,
            (mem_params, mem_tx_params),
            jnp.arange(args.mi_steps),
            length=args.mi_steps
        )

        mem_params, mem_tx_params = output_mem_tuple
        print("Final loss: {}", info['loss'][-1])
        print("Memory function:\n{}", jax.nn.softmax(mem_params))

        mem_pomdp = memory_cross_product(mem_params, pomdp)

        # pi optimisation
        pi_shape = (n_o, n_a)
        pi_params = random.normal(mem_rng, shape=pi_shape) * 0.5
        pi_params = new_pi_over_mem(pi_params, args.n_mem_states)

        pi_optim = get_optimizer(args.optimizer, args.pi_lr)
        pi_tx_params = pi_optim.init(pi_params)

        @scan_tqdm(args.pi_steps)
        def update_pi_step(inps, i):
            params, tx_params = inps
            outs, params_grad = value_and_grad(pg_objective_func, has_aux=True)(params, mem_pomdp)
            v_0, (td_v_vals, td_q_vals) = outs

            params_grad = -params_grad  # maximise objective

            updates, tx_params = pi_optim.update(params_grad, tx_params, params)
            params = optax.apply_updates(params, updates)
            outs = (params, tx_params)
            return outs, {'v0': v_0, 'v': td_v_vals, 'q': td_q_vals}
        
        print("Optimising policy...")
        print("Initial performance: {}", get_policy_performance(pi_params, mem_pomdp))
        #output_pi_tuple, info = python_scan(
        output_pi_tuple, info = jax.lax.scan(
            update_pi_step,
            (pi_params, pi_tx_params),
            jnp.arange(args.pi_steps),
            length=args.pi_steps
        )

        pi_params, pi_tx_params = output_pi_tuple

        print("Final performance: {}", get_policy_performance(pi_params, mem_pomdp))

    return experiment

def python_scan(step_fn, init, x, length):
    carry = init
    aux_outputs = []

    with tqdm(range(length), desc="Training", ncols=80) as pbar:
        for step in pbar:
            carry, aux = step_fn(carry, step)
            aux_outputs.append(aux)
            pbar.set_postfix(loss=f"{aux['loss']:.4f}")


    # Optionally stack the aux outputs into an array
    aux_outputs = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *aux_outputs)

    return carry, aux_outputs

if __name__ == "__main__":
    jax.disable_jit(True)
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

