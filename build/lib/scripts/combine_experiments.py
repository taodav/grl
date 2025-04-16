from copy import deepcopy
from itertools import product
from pathlib import Path

from flax.training import orbax_utils
import jax
import orbax.checkpoint
import numpy as np

from grl.utils.file_system import load_info, numpyify_and_save
from grl.utils.file_system import import_module_to_var

from definitions import ROOT_DIR

def filter_dict_if_list(res: dict, idxes: list[int]):
    new_res = deepcopy(res)

    for k, v in res.items():
        if isinstance(v, list):
            new_res[k] = [v[i] for i in idxes]
    return new_res


def find_file_in_dir(file_name: str, base_dir: Path) -> Path:
    for path in base_dir.rglob('*'):
        if file_name in str(path):
            return path

if __name__ == "__main__":
    exp_dir = Path(ROOT_DIR, 'results', 'variance_pg_kitchen_4_mem_states')
    new_exp_dir = exp_dir.parent / ('combined_' + exp_dir.name)
    new_exp_dir.mkdir(exist_ok=True)

    hyperparams_dir = Path(ROOT_DIR, 'scripts', 'hyperparams').resolve()
    study_hparam_filename = exp_dir.stem + '.py'
    study_hparam_path = find_file_in_dir(study_hparam_filename, hyperparams_dir)

    hparams = import_module_to_var(study_hparam_path, 'hparams')
    hparam_args = hparams['args']
    assert len(hparam_args) == 1
    hparam_args = hparam_args[0]
    
    listed_hparams = [(k, v) for k, v in hparam_args.items() if isinstance(v, list) and k not in ['spec', 'seed']]
    listed_keys = [k for k, v in listed_hparams]
    listed_values = [v for k, v in listed_hparams]

    product_listed_hparams = list(product(*listed_values))

    all_env_res = {}

    for exp_path in exp_dir.iterdir():
        restored = load_info(exp_path)

        args = restored['args']
        env = args['spec']
        args_less_seed = {k: v for k, v in args.items() if k != 'seed' and k != 'n_seeds'}

        if env not in all_env_res:

            all_env_res[env] = {
                'args_less_seed': [args_less_seed],
                'n_total_seeds': args['n_seeds'],
                'seeds': [args['seed']],
                'all_logs': [restored['logs']],
                'example_restored': restored,
                'first_path_name': exp_path.name
            }
        else:
            all_env_res[env]['args_less_seed'].append(args_less_seed)
            all_env_res[env]['n_total_seeds'] += args['n_seeds']
            all_env_res[env]['seeds'].append(args['seed'])
            all_env_res[env]['all_logs'].append(restored['logs'])

    # here we compare args
    for env, res in all_env_res.items():

        # we save a new hparam file for each product of hparams swept
        if len(listed_values) > 0:
            for hparam_vals in product_listed_hparams:
                filtered_res_idxes = []
                
                id_str = ''
                for k, v in zip(listed_keys, hparam_vals):
                    id_str += f'_{k}({str(v)})'

                for i in range(len(res['args_less_seed'])):

                    for k, v in zip(listed_keys, hparam_vals):
                        if res['args_less_seed'][i][k] != v:
                            break
                    else:
                        filtered_res_idxes.append(i)

                filtered_res = filter_dict_if_list(res, filtered_res_idxes)

                def ccat(*args):
                    return np.concatenate(args, axis=0)

                new_ckpt = filtered_res['example_restored']
                new_ckpt['args'] = filtered_res['args_less_seed'][0]
                new_ckpt['args']['n_seeds'] = int(filtered_res['n_total_seeds'] / len(product_listed_hparams))
                new_ckpt['args']['seed'] = filtered_res['seeds'][0]
                new_ckpt['logs'] = jax.tree_map(ccat, *filtered_res['all_logs'])

                seeds_concat = '+'.join(map(str, sorted(filtered_res['seeds'])))
                new_file_name = filtered_res['first_path_name'].replace(f"seed({filtered_res['seeds'][0]})", f"seed({seeds_concat})")
                new_file_name_with_id = new_file_name.replace('_seed', f'{id_str}_seed')
                new_path = new_exp_dir / new_file_name_with_id
                print(f"Saving combined results for {env} to {new_path}")

                numpyify_and_save(new_path, new_ckpt)
        else:
            assert len(res['args_less_seed']) > 1
            example_arg = res['args_less_seed'][0]

            for k in example_arg.keys():
                for arg in res['args_less_seed']:

                    if isinstance(arg[k], list):
                        for a1, a2 in zip(arg[k], example_arg[k]):
                            assert a1 == a2
                    if isinstance(arg[k], np.ndarray):
                        assert all(arg[k] == example_arg[k])
                    else:
                        assert arg[k] == example_arg[k]

            new_exp_path = Path(ROOT_DIR, 'results', 'kitchen_leave_out_8_30seeds')
            new_ckpt = res['example_restored']
            new_ckpt['args']['n_seeds'] = filtered_res['n_total_seeds']
            new_ckpt['logs'] = jax.tree_map(ccat, *res['all_logs'])

            seeds_concat = '+'.join(map(str, sorted(res['seeds'])))
            new_path = exp_dir / res['first_path_name'].replace(f"seed({res['seeds'][0]})", f"seed({seeds_concat})")
            print(f"Saving combined results for {env} to {new_path}")

            numpyify_and_save(new_path, new_ckpt)

