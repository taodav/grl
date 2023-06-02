import argparse

import numpy as np
import jax
from jax.config import config

from grl.agent import get_agent
from grl.environment import load_spec
from grl.evaluation import test_episodes
from grl.mdp import AbstractMDP, MDP
from grl.model import get_network
from grl.utils.optimizer import get_optimizer
from grl.sample_trainer import Trainer
from grl.utils.file_system import results_path, numpyify_and_save

def parse_arguments(return_defaults: bool = False):
    parser = argparse.ArgumentParser()
    # yapf:disable
    # Environment params
    parser.add_argument('--spec', default='simple_chain', type=str,
                        help='name of POMDP spec; evals Pi_phi policies by default')
    parser.add_argument('--no_gamma_terminal', action='store_true',
                        help='Do we turn OFF gamma termination?')
    parser.add_argument('--max_episode_steps', default=1000, type=int,
                        help='Maximum number of episode steps')

    # Agent params
    parser.add_argument('--algo', default='rnn', type=str,
                        help='Algorithm to evaluate, (rnn | multihead_rnn)')
    parser.add_argument('--arch', default='gru', type=str,
                        help='Algorithm to evaluate, (gru | elman)')
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='What epsilon do we use?')
    parser.add_argument('--lr', default=0.001, type=float)

    # RNN hyperparams
    parser.add_argument('--hidden_size', default=10, type=int,
                        help='RNN hidden size')
    parser.add_argument('--value_head_layers', default=0, type=int,
                        help='For our value head network, how deep is it?')
    parser.add_argument('--trunc', default=-1, type=int,
                        help='RNN truncation length')
    parser.add_argument('--action_cond', default="cat", type=str,
                        help='Do we do (previous) action conditioning of observations? (None | cat)')

    # Multihead RNN hyperparams
    parser.add_argument('--multihead_action_mode', default='td', type=str,
                        help='What head to we use for multihead_rnn for action selection? (td | mc)')
    parser.add_argument('--multihead_loss_mode', default='both', type=str,
                        help='What mode do we use for the multihead RNN loss? (both | td | mc | split)')
    parser.add_argument('--multihead_lambda_coeff', default=0., type=float,
                        help='What is our coefficient for our lambda discrepancy loss?')

    # Replay buffer hyperparams
    parser.add_argument('--replay_size', default=1, type=int,
                        help='Replay buffer size. Set to 1 for online training.')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Replay buffer batch size. Set to 1 for online training.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='What optimizer do we use? (sgd | adam | rmsprop)')

    # Logging and checkpointing hyperparams
    parser.add_argument('--offline_eval_freq', type=int, default=None,
                        help='How often do we evaluate offline during training?')
    parser.add_argument('--offline_eval_episodes', type=int, default=1,
                        help='When we do run offline eval, how many episodes do we run?')
    parser.add_argument('--offline_eval_epsilon', type=float, default=None,
                        help='What is our evaluation epsilon? Default is greedy.')
    parser.add_argument('--checkpoint_freq', type=int, default=100,
                        help='How often do we checkpoint?')
    parser.add_argument('--total_steps', type=int, default=int(1e4),
                        help='How many total environment steps do we take in this experiment?')
    parser.add_argument('--save_all_checkpoints', action='store_true',
                        help='Do we store ALL of our checkpoints? If not, store only last.')

    # Experiment hyperparams
    parser.add_argument('--platform', type=str, default='cpu',
                        help='What platform do we use (cpu | gpu)')
    parser.add_argument('--seed', default=None, type=int,
                        help='What seed do we use to make the runs deterministic?')
    parser.add_argument('--study_name', type=str, default=None,
                        help='If study name is not specified, we just save things to the root results/ directory.')

    # For testing: PyTest doesn't like parser.parse_args(), so we just return the defaults.
    if return_defaults:
        defaults = {}
        for action in parser._actions:
            if not action.required and action.dest != "help":
                defaults[action.dest] = action.default
        return argparse.Namespace(**defaults)
    args = parser.parse_args()

    if args.offline_eval_epsilon is None:
        args.offline_eval_epsilon = args.epsilon

    return args

if __name__ == "__main__":
    args = parse_arguments()

    # configs
    np.set_printoptions(precision=4, suppress=True)
    config.update('jax_platform_name', args.platform)
    config.update("jax_enable_x64", True)

    # config.update('jax_disable_jit', True)

    rand_key = None
    if args.seed is not None:
        np.random.seed(args.seed)
        rand_key = jax.random.PRNGKey(args.seed)
    else:
        rand_key = jax.random.PRNGKey(np.random.randint(1, 10000))

    # Run
    # Get POMDP definition
    spec = load_spec(args.spec)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    env = AbstractMDP(mdp, spec['phi'])

    results_path = results_path(args)
    all_agents_dir = results_path.parent / 'agent'
    all_agents_dir.mkdir(exist_ok=True)

    agents_dir = all_agents_dir / f'{results_path.stem}'

    network = get_network(args, env.n_actions)

    optimizer = get_optimizer(args.optimizer, step_size=args.lr)

    features_shape = env.observation_space
    if args.action_cond == 'cat':
        features_shape = features_shape[:-1] + (features_shape[-1] + env.n_actions,)

    agent = get_agent(network, optimizer, features_shape, env, args)

    trainer_key, rand_key = jax.random.split(rand_key)
    trainer = Trainer(env, agent, trainer_key, args, checkpoint_dir=agents_dir)

    final_network_params, final_optimizer_params, episodes_info = trainer.train()

    # TODO: change to a parser param
    final_eval_episodes = 10
    print(f"Finished training. Evaluating over {final_eval_episodes} episodes.")

    final_eval_info, rand_key = test_episodes(agent, final_network_params, env, rand_key,
                                              n_episodes=final_eval_episodes,
                                              test_eps=0., action_cond=args.action_cond,
                                              max_episode_steps=args.max_episode_steps)

    # TODO: add in test_episodes
    summed_perf = 0
    for ep in final_eval_info['episode_rewards']:
        summed_perf += sum(ep)

    print(f"Final (averaged) greedy evaluation performance: {summed_perf / final_eval_episodes}")

    info = {
        'episodes_info': episodes_info,
        'args': vars(args),
        'final_greedy_eval_rews': final_eval_info['episode_rewards'],
        'final_eval_qs': final_eval_info['episode_qs']
    }

    print(f"Saving results to {results_path}")
    numpyify_and_save(results_path, info)




