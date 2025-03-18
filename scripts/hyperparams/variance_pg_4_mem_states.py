from pathlib import Path

exp_name = Path(__file__).stem

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': 'scripts/batch_run.py',
    'args': [{
        'spec': [
            '4x3.95',
            'network',
            # 'hallway'
            # 'bridge-repair',
        ],
        'policy_optim_alg': 'policy_grad',
        'value_type': 'q',
        'error_type': 'l2',
        'alpha': 1.,
        'objective': 'variance',
        'mi_steps': 20000,
        'pi_steps': 10000,
        'reward_in_obs': True,
        'optimizer': 'adam',
        'lr': 0.01,
        'n_mem_states': 4,
        'n_seeds': 1,
        'platform': 'gpu',
        'seed': [2025 + i for i in range(5)],
        # 'seed': [2020 + i for i in range(10)],
    }]
}
