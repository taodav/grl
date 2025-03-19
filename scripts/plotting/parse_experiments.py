from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from grl.memory.analytical import memory_cross_product
from grl.environment import load_pomdp
from grl.utils import load_info
from grl.utils.lambda_discrep import lambda_discrep_measures
from grl.utils.math import greedify

from definitions import ROOT_DIR


kitchen_obj_map = {
    'tde': 'mstde',
    'tde_residual': 'mstde_res',
    'discrep': 'ld'
}

def parse_batch_dirs(exp_dirs: list[Path],
                     baseline_dict: dict,
                     args_to_keep: list[str]):
    all_results = []

    def parse_exp_dir(exp_dir: Path):
        print(f"Parsing {exp_dir}")
        for results_path in tqdm(list(exp_dir.iterdir())):
            if results_path.is_dir() or results_path.suffix != '.npy':
                continue

            info = load_info(results_path)
            args = info['args']
            logs = info['logs']

            if args['spec'] not in baseline_dict:
                continue

            pomdp, _ = load_pomdp(args['spec'])
            # n_random_policies = args['random_policies']

            beginning = logs['beginning']
            aim_measures = beginning['measures']
            if 'kitchen' in exp_dir.stem:
                # if we're doing kitchen sinks policies, we need to take the mean over
                # initial policies
                init_policy_perf_seeds = (aim_measures['values']['state_vals']['v'] * aim_measures['values']['p0'])
                init_policy_perf_seeds = init_policy_perf_seeds.sum(axis=-1).mean(axis=-1)
            else:
                init_policy_perf_seeds = np.einsum('ij,ij->i',
                                                   aim_measures['values']['state_vals']['v'],
                                                   aim_measures['values']['p0'])

            after_pi_op = logs['after_pi_op']
            if 'measures' not in after_pi_op:
                assert 'initial_improvement_measures' in after_pi_op
                apo_measures = after_pi_op['initial_improvement_measures']
            else:
                apo_measures = after_pi_op['measures']
            init_improvement_perf_seeds = np.einsum('ij,ij->i',
                                                    apo_measures['values']['state_vals']['v'],
                                                    apo_measures['values']['p0'])
            compare_to_perf = baseline_dict[args['spec']]

            # if isinstance(args['objectives'], str):
            #     keys = [args['objectives']]
            # else:
            #     keys = args['objectives']

            # for i, key in enumerate(keys):
            #     key = kitchen_obj_map.get(key, key)
            #     objective, residual = key, False
            #     if key == 'mstde_res':
            #         objective, residual = 'mstde', True


            objective = args['objective']
            single_res = {k: args[k] for k in args_to_keep}
            single_res['experiment'] = exp_dir.name + f'_{objective}'
            single_res['objective'] = objective

            final = logs['final']
            final_measures = final['measures']
            if 'kitchen' in exp_dir.stem:
                # For now, we assume kitchen selection objective == mem learning objective
                final_mem_perf = np.einsum('ij,ij->i',
                                           final_measures['values']['state_vals']['v'][:, i],
                                           final_measures['values']['p0'][:, i])

            else:
                final_mem_perf = np.einsum('ij,ij->i',
                                           final_measures['values']['state_vals']['v'],
                                           final_measures['values']['p0'])

            for i in range(args['n_seeds']):
                all_results.append({
                    **single_res,
                    'seed': i,
                    'init_policy_perf': init_policy_perf_seeds[i],
                    'init_improvement_perf': init_improvement_perf_seeds[i],
                    'final_mem_perf': final_mem_perf[i],
                    'compare_to_perf': compare_to_perf,
                })

    for exp_dir in exp_dirs:
        parse_exp_dir(exp_dir)

    all_res_df = pd.DataFrame(all_results)

    return all_res_df


def parse_dirs(exp_dirs: list[Path],
               baseline_dict: dict,
               args_to_keep: list[str]):
    all_results = []

    def parse_exp_dir(exp_dir: Path):
        print(f"Parsing {exp_dir}")
        for results_path in tqdm(list(exp_dir.iterdir())):
            if results_path.is_dir() or results_path.suffix != '.npy':
                continue

            info = load_info(results_path)
            args = info['args']

            if args['spec'] not in baseline_dict:
                continue

            def get_perf(info: dict):
                return (info['state_vals_v'] * info['p0']).sum()

            # Greedification
            agent_path = results_path.parent / 'agent' / f'{results_path.stem}.pkl.npy'
            agent = load_info(agent_path)
            pomdp, _ = load_pomdp(args['spec'])
            final_mem_pomdp = memory_cross_product(agent.mem_params, pomdp)

            greedy_policy = greedify(agent.policy)

            greedy_measures = lambda_discrep_measures(final_mem_pomdp, greedy_policy)
            greedy_pi_mem_perf = float(get_perf(greedy_measures))

            init_policy_info = info['logs']['initial_policy_stats']
            init_improvement_info = info['logs']['greedy_initial_improvement_stats']
            final_mem_info = info['logs']['final_mem_stats']

            single_res = {k: args[k] for k in args_to_keep}

            final_mem_perf = get_perf(final_mem_info)

            compare_to_perf = baseline_dict[args['spec']]
            init_policy_perf = get_perf(init_policy_info)
            init_improvement_perf = get_perf(init_improvement_info)

            if init_policy_perf > init_improvement_perf:
                print("Initial policy performance is better than improvement performance\n"
                      f"For {results_path}.\n"
                      "Setting the bottom line to the initial policy performance.")
                init_policy_perf = init_improvement_perf

            single_res.update({
                'experiment': exp_dir.name,
                'init_policy_perf': init_policy_perf,
                'init_improvement_perf': init_improvement_perf,
                'final_mem_perf': final_mem_perf,
                'greedy_pi_mem_perf': greedy_pi_mem_perf,
                'compare_to_perf': compare_to_perf,
            })
            all_results.append(single_res)

    for exp_dir in exp_dirs:
        parse_exp_dir(exp_dir)

    all_res_df = pd.DataFrame(all_results)

    return all_res_df

def parse_baselines(
        plot_order: list[str],
        vi_dir: Path = Path(ROOT_DIR, 'results', 'vi'),
        pomdp_files_dir: Path = Path(ROOT_DIR, 'grl', 'environment', 'pomdp_files'),
        compare_to: str = 'belief'):
    compare_to_dict = {}
    spec_to_belief_state = {'tmaze_5_two_thirds_up': 'tmaze5'}

    def load_state_val(spec: str):
        for vi_path in vi_dir.iterdir():
            if spec_to_belief_state.get(spec, spec) in vi_path.name:
                vi_info = load_info(vi_path)
                max_start_vals = vi_info['optimal_vs']
                return np.dot(max_start_vals, vi_info['p0'])

    for spec in plot_order:
        if compare_to == 'belief':

            for fname in pomdp_files_dir.iterdir():
                if 'pomdp-solver-results' in fname.stem:
                    if (fname.stem ==
                            f"{spec_to_belief_state.get(spec, spec)}-pomdp-solver-results"
                    ):
                        belief_info = load_info(fname)
                        compare_to_dict[spec] = belief_info['start_val']
                        break
            else:
                compare_to_dict[spec] = load_state_val(spec)

        elif compare_to == 'state':
            compare_to_dict[spec] = load_state_val(spec)

    return compare_to_dict

if __name__ == "__main__":
    from pathlib import Path
    from definitions import ROOT_DIR

    compare_to = 'belief'

    directory = Path(ROOT_DIR, 'results', "discrep_interleave_pg")
    # directory = Path(ROOT_DIR, 'results', "final_discrep_kitchen_sinks_pg")

    vi_results_dir = Path(ROOT_DIR, 'results', 'vi')
    pomdp_files_dir = Path(ROOT_DIR, 'grl', 'environment', 'pomdp_files')

    args_to_keep = ['spec', 'n_mem_states', 'seed', 'alpha']
    spec_plot_order = [
        'network', 'paint.95', '4x3.95', 'tiger-alt-start', 'shuttle.95', 'cheese.95', 'tmaze_5_two_thirds_up'
    ]

    compare_to_dict = parse_baselines(spec_plot_order,
                                      vi_results_dir,
                                      pomdp_files_dir,
                                      compare_to=compare_to)

    res_df = parse_batch_dirs([directory], compare_to_dict, args_to_keep)
    # res_df = parse_dirs([directory], compare_to_dict, args_to_keep)

    print()

