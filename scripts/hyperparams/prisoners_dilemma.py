import numpy as np
from pathlib import Path

exp_name = Path(__file__).stem
game_name = 'prisoners_dilemma'

# leader_policies = ['tit_for_tat', 'extort', 'grudger2', 'majority3',
#                    'treasure_hunt', 'all_d', 'all_c', 'sugar', 'alternator']
leader_policies = ['tit_for_tat', 'all_c']

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'args': [{
        'spec': [f'{game_name}_{p}' for p in leader_policies],
        'use_memory': '0',
        'value_type': 'q',
        'lambda_0': 0.,
        'lambda_1': 1.,
        'policy_optim_alg': 'policy_iter',
        'objective': 'obs_space',
        'alpha': 1.,
        'mi_steps': 5000,
        'pi_steps': 5000,
        'init_pi': 0,
        'n_mem_states': [2, 4, 8],
        'error_type': 'l2',
        'optimizer': 'adam',
        'lr': 0.01,
        'seed': [2020 + i for i in range(10)],
    }]
}
