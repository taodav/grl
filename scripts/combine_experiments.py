from pathlib import Path

from flax.training import orbax_utils
import jax
import orbax.checkpoint
import numpy as np

from grl.utils.file_system import load_info, numpyify_and_save

from definitions import ROOT_DIR


if __name__ == "__main__":
    exp_path_1 = Path(ROOT_DIR, 'results', 'kitchen_leave_out_8',
                      'tmaze_5_two_thirds_up_batch_run_seed(2024)_time(20240519-152547)_9bda83c52cab992d0b090497b814293b.npy')

    exp_path_2 = Path(ROOT_DIR, 'results', 'kitchen_leave_out_8',
                      'tmaze_5_two_thirds_up_batch_run_seed(2029)_time(20240519-152551)_103dc9ead45d5310253e28cf452eee32.npy')

    new_path = Path(ROOT_DIR, 'results', 'kitchen_leave_out_8',
                    'tmaze_5_two_thirds_up_batch_run_seed(2024+2029)_time(20240519-152547)_9bda83c52cab992d0b090497b814293b.npy')

    restored_1 = load_info(exp_path_1)
    restored_2 = load_info(exp_path_2)

    args_1 = restored_1['args']
    args_2 = restored_2['args']

    # here we compare args
    for k in restored_1['args'].keys():
        if k == 'n_seeds' or k == 'seed':
            continue

        if isinstance(args_1[k], list):
            for a1, a2 in zip(args_1[k], args_2[k]):
                assert a1 == a2
        if isinstance(args_1[k], np.ndarray):
            assert all(args_1[k] == args_2[k])
        else:
            assert args_1[k] == args_2[k]

    new_args = args_1
    new_args['n_seeds'] = args_1['n_seeds'] + args_2['n_seeds']

    new_out = jax.tree_map(lambda x, y: np.concatenate([x, y], axis=0), restored_1['logs'], restored_2['logs'])

    new_checkpoint = restored_1
    new_checkpoint['args'] = new_args
    new_checkpoint['logs'] = new_out

    print(f"Saving combined results to {new_path}")
    numpyify_and_save(new_path, new_checkpoint)

