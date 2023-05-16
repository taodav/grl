import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

from jax.nn import softmax
from jax.config import config
from pathlib import Path
from collections import namedtuple

import math

config.update('jax_platform_name', 'cpu')
np.set_printoptions(precision=4)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 18})

from grl.utils import load_info
from definitions import ROOT_DIR

def smooth_results(results, window=1):
    return np.convolve(results, np.ones(window), 'valid') / window

def make_plots(results_path, plot_separator="stat", line_separators=["lstm_mode"],
               plots_to_include=None, plot_key='reward_per_episode', smoothen=-1, all_on_one=False):

    result_dicts = []
    for fname in results_path.iterdir():
        if fname.is_dir() or fname.suffix != '.npy':
            continue
        
        info = load_info(fname)

        args = info['args']
        df_dict = {
            'reward_per_episode': info['logs']['total_reward'],
            'loss': info['logs']['loss'],
            'environment': args['spec'],
            'seed': args['seed'],
            'mode': args['lstm_mode'],
            'separator': tuple([args[key] for key in line_separators]),# (args['lstm_mode'], args['epsilon_anneal_steps'])
        }
        if info['logs'].get('aux_loss'):
            df_dict.update(info['logs'].get('aux_loss'))

        print(df_dict['separator'])
        result_dicts.append(df_dict)

    grouped_dicts = group_dicts(result_dicts, plot_key=plot_key)
    num_envs = len(grouped_dicts)
    num_rows = 3
    num_columns = math.ceil(num_envs / num_rows)
    if all_on_one:
        plt.subplots_adjust(hspace=0.5)
    for i, (env, env_results) in enumerate(grouped_dicts.items()):
        if all_on_one:
            plt.subplot(num_rows, num_columns, i + 1)
        if plots_to_include is not None and env not in plots_to_include:
            continue
        for separator, run_results in sorted(env_results.items()):
            average_results = np.mean(run_results, axis=0)
            std_dev_results = np.std(run_results, axis=0)
            num_runs = len(run_results)
            std_err = std_dev_results / np.sqrt(num_runs)
            if smoothen > 0:
                average_results = smooth_results(average_results, window=smoothen)
                std_err = smooth_results(std_err, window=smoothen)
            plt.plot(average_results, label=f"{separator} ({num_runs} seeds)")
            plt.fill_between(np.arange(len(average_results)), average_results - std_err, average_results + std_err, alpha=0.2)

        # plt.title(f"{env} ({plot_key})")
        plt.title(f"{env}")
        if not(all_on_one):
            plt.title(f"{env} ({plot_key})")
            plt.legend()
            plt.show()
    if all_on_one:
        plt.legend(bbox_to_anchor=(3.0, 1.0))
        plt.show()


def group_dicts(result_dicts, plot_key='reward_per_episode'):
    grouped_dicts = {}
    for d in result_dicts:
        environemnt = d['environment']
        # mode = d['mode']
        separator = d['separator']
        if environemnt not in grouped_dicts:
            grouped_dicts[environemnt] = {}
        if separator not in grouped_dicts[environemnt]:
            grouped_dicts[environemnt][separator] = []
        grouped_dicts[environemnt][separator].append(d[plot_key])
    return grouped_dicts


if __name__ == '__main__':
    plot_key = "reward_per_episode"

    results_path = Path(ROOT_DIR) / "shared_grl_results" / "grl_data" / "lstm_150523" / "base_reward_norm_full"
    make_plots(results_path, line_separators=["lstm_mode"], plot_key='reward_per_episode', smoothen=10, all_on_one=True)

    # results_path = Path(ROOT_DIR) / "ccv_results" / "results" / "baselines" / "lstm" / "first_comparison"
    # results_path = Path(ROOT_DIR) / "ccv_results" / "results" / "baselines" / "lstm" / "gamma_terminal_td_vs_lambda"
    # make_plots(results_path, line_separators=["lstm_mode", "epsilon_anneal_steps"], plot_key=plot_key, plots_to_include=None)

    # results_path = Path(ROOT_DIR) / "ccv_results" / "results" / "baselines" / "lstm" / "lambda_scale_sweep"
    # make_plots(results_path, line_separators=["lstm_mode", "lambda_coefficient"], plot_key=plot_key, plots_to_include=['tiger-alt-start'])

    # results_path = Path(ROOT_DIR) / "results" / "baselines" / "lstm" / "all_envs_short_normalized_reward"
    # results_path = Path(ROOT_DIR) / "results" / "baselines" / "lstm" / "bit_longer_normalized_reward"
    # make_plots(results_path, line_separators=["lstm_mode", "epsilon_anneal_steps"], plot_key='reward_per_episode')#, plots_to_include="shuttle.95")

    # results_path = Path(ROOT_DIR) / "results" / "testing" / "baselines" / "lstm" / "aux_loss_logging_100x"
    # make_plots(results_path, line_separators=["lstm_mode", "lambda_coefficient"], plot_key='lambda_loss')#, plots_to_include="shuttle.95")

    # results_path = Path(ROOT_DIR) / "ccv_results" / "results" / "baselines" / "lstm" / "lower_lambda_scale_sweep"
    # make_plots(results_path, line_separators=["lstm_mode", "lambda_coefficient"], plot_key='reward_per_episode')#, plots_to_include="shuttle.95")

    # results_path = Path(ROOT_DIR) / "results" / "baselines" / "lstm" / "rs_lc_sweep"
    # # line_separators = ["lstm_mode", "reward_scale", "lambda_coefficient"]
    # line_separators = ["lstm_mode", "reward_scale"]
    # # line_separators = ["lstm_mode", "lambda_coefficient"]
    # make_plots(results_path, line_separators=line_separators, plot_key='reward_per_episode')#, plots_to_include="shuttle.95")

    # results_path = Path(ROOT_DIR) / "shared_grl_results" / "grl_data" / "lstm_130523" / "lambda_scale_sweep"
    # make_plots(results_path, line_separators=["lstm_mode", "lambda_coefficient"], plot_key='reward_per_episode')#, plots_to_include="shuttle.95")

    results_path = Path(ROOT_DIR) / "shared_grl_results" / "grl_data" / "lstm_150523" / "base_reward_norm_full"
    make_plots(results_path, line_separators=["lstm_mode"], plot_key='reward_per_episode', smoothen=10, all_on_one=True)
    #, plots_to_include="shuttle.95")

    # results_path = Path(ROOT_DIR) / "shared_grl_results" / "grl_data" / "lstm_150523" / "lambda_coeff_reward_norm_full"
    # make_plots(results_path, line_separators=["lstm_mode", "lambda_coefficient"], plot_key='reward_per_episode')#, plots_to_include="shuttle.95")
