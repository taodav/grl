from pathlib import Path

exp_name = Path(__file__).stem

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': 'scripts/batch_run.py',
    'args': [{
        'spec': [
            'tiger-alt-start', 'tmaze_5_two_thirds_up', 'example_7', '4x3.95',
            'cheese.95', 'network', 'shuttle.95', 'paint.95'
            # 'hallway'
            # 'bridge-repair',
        ],
        'policy_optim_alg': 'policy_grad',
        'value_type': 'q',
        'error_type': 'l2',
        'alpha': 1.,
        'objective': ['variance', 'discrep'],
        # 'objective': ['discrep', 'obs_space'],
        'mi_steps': 20000,
        'pi_steps': 10000,
        'reward_in_obs': True,
        'optimizer': 'adam',
        'lr': 0.01,
        'n_mem_states': [2, 4],
        'n_seeds': 5,
        'platform': 'gpu',
        'seed': 2025,
        # 'seed': [2020 + i for i in range(10)],
    }]
}
