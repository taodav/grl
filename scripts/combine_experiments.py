from pathlib import Path

from flax.training import orbax_utils
import jax
import orbax.checkpoint
import numpy as np

from grl.utils.file_system import load_info, numpyify_and_save

from definitions import ROOT_DIR


if __name__ == "__main__":
    exp_dir = Path(ROOT_DIR, 'results', 'kitchen_leave_out_8_30seeds')

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

        def ccat(*args):
            return np.concatenate(args, axis=0)

        new_exp_path = Path(ROOT_DIR, 'results', 'kitchen_leave_out_8_30seeds')
        new_ckpt = res['example_restored']
        new_ckpt['args']['n_seeds'] = res['n_total_seeds']
        new_ckpt['logs'] = jax.tree_map(ccat, *res['all_logs'])

        seeds_concat = '+'.join(map(str, sorted(res['seeds'])))
        new_path = exp_dir / res['first_path_name'].replace(f"seed({res['seeds'][0]})", f"seed({seeds_concat})")
        print(f"Saving combined results for {env} to {new_path}")

        numpyify_and_save(new_path, new_ckpt)

