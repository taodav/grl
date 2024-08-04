from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from grl.environment import load_pomdp
from grl.utils import load_info

from scripts.plotting.parse_experiments import parse_baselines

from definitions import ROOT_DIR


kitchen_obj_map = {
    'tde': 'mstde',
    'tde_residual': 'mstde_res',
    'discrep': 'ld'
}


def parse_exp_dir(exp_dir: Path):
    print(f"Parsing {exp_dir}")

    df = None
    for results_path in tqdm(list(exp_dir.iterdir())):
        if results_path.is_dir() or results_path.suffix != '.npy':
            continue

        info = load_info(results_path)
        args = info['args']
        logs = info['logs']

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

        if isinstance(args['objectives'], str):
            keys = [args['objectives']]
        else:
            keys = args['objectives']

        for i, key in enumerate(keys):
            key = kitchen_obj_map.get(key, key)
            objective, residual = key, False
            if key == 'mstde_res':
                objective, residual = 'mstde', True

            args['residual'] = residual
            args['objective'] = objective

            single_res = {k: args[k] for k in args_to_keep}
            # single_res['experiment'] = exp_dir.name + f'_{objective}'
            single_res['objective'] = objective

            final = logs['final']
            final_measures = final[key]['measures']
            if 'kitchen' in exp_dir.stem:
                # For now, we assume kitchen selection objective == mem learning objective
                final_mem_perf = np.einsum('ij,ij->i',
                                           final_measures['values']['state_vals']['v'][:, i],
                                           final_measures['values']['p0'][:, i])

            else:
                final_mem_perf = np.einsum('ij,ij->i',
                                           final_measures['values']['state_vals']['v'],
                                           final_measures['values']['p0'])

            n_seeds = final_mem_perf.shape[0]
            seeds_res = {}
            for k, v in single_res.items():
                seeds_res[k] = np.array([v] * n_seeds, dtype=type(v))

            seeds_res['seed'] = np.arange(n_seeds) + single_res['seed']
            seeds_res['score'] = final_mem_perf

            def get_memoryless_and_random_df():
                memoryless_res = seeds_res.copy()

                assert 'n_mem_states' in memoryless_res
                memoryless_res['n_mem_states'] = np.zeros(n_seeds, dtype=int) + np.nan
                memoryless_res['objective'] = np.array(['memoryless'] * n_seeds)
                memoryless_res['score'] = init_improvement_perf_seeds
                memoryless_df = pd.DataFrame(memoryless_res)

                random_res = memoryless_res.copy()
                random_res['score'] = init_policy_perf_seeds
                random_res['objective'] = np.array(['random'] * n_seeds)
                random_df = pd.DataFrame(random_res)
                return pd.concat([memoryless_df, random_df])

            if df is None:
                df = get_memoryless_and_random_df()

            if df.loc[df['spec'] == args['spec']].shape[0] == 0:
                # if there's nothing with this spec, we add random and init
                df = pd.concat([df, get_memoryless_and_random_df()])

            seeds_df = pd.DataFrame(seeds_res)
            df = pd.concat([df, seeds_df])

    return df


if __name__ == "__main__":
    exp_dir = Path(ROOT_DIR, 'results', 'kitchen_leave_out_30seeds')

    csv_save_path = exp_dir.parent / f'{exp_dir.stem}.csv'

    args_to_keep = ['spec', 'n_mem_states', 'seed', 'objective']

    spec_plot_order = [
        'network', 'paint.95', '4x3.95', 'tiger-alt-start', 'shuttle.95', 'cheese.95', 'tmaze_5_two_thirds_up', 'parity_check'
    ]

    vi_results_dir = Path(ROOT_DIR, 'results', 'vi')
    pomdp_files_dir = Path(ROOT_DIR, 'grl', 'environment', 'pomdp_files')

    compare_to = 'belief'
    compare_to_dict = parse_baselines(spec_plot_order,
                                      vi_results_dir,
                                      pomdp_files_dir,
                                      compare_to=compare_to)

    df = parse_exp_dir(exp_dir)
    all_seeds = df['seed'].unique()
    n_seeds = all_seeds.shape[0]

    for env, v in compare_to_dict.items():
        env_compare_res = {
            'spec': np.array([env] * n_seeds),
            'seed': all_seeds,
            'n_mem_states': np.zeros(n_seeds) + np.nan,
            'objective': np.array([compare_to] * n_seeds),
            'score': np.zeros(n_seeds) + v
        }
        df = pd.concat([df, pd.DataFrame(env_compare_res)])

    df.to_csv(csv_save_path)
    print(f'Parsed and saved results to {csv_save_path}')
