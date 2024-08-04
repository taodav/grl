from pathlib import Path

exp_name = Path(__file__).stem

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m scripts.batch_run_kitchen_sinks_ld_only',
    'args': [{
        'spec': [
            'parity_check'
        ],
        'policy_optim_alg': 'policy_grad',
        'leave_out_optimal': True,
        'value_type': 'q',
        'error_type': 'l2',
        'alpha': 1.,
        'mi_steps': 10000,
        'pi_steps': 10000,
        'optimizer': 'adam',
        'lr': 0.75,
        'n_mem_states': [2, 4],
        'mi_iterations': 1,
        'random_policies': 100,
        'n_seeds': 30,
        'platform': 'gpu'
    },

    ]
}
