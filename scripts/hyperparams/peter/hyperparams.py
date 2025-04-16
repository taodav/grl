from pathlib import Path

exp_name = Path(__file__).stem

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m scripts.batch_run_kitchen_sinks_single_obj',
    'args': [{
        'spec': 'parity_check',
        'policy_optim_alg': 'policy_grad',
        'value_type': 'q',
        'error_type': 'l2',
        'alpha': 1.,
        'objective': 'ld', #['gvf_obs', 'ld', 'obs_space'],
        'gamma_type': 'uniform',
        'gamma_max': 0.99,
        'gamma_min': 0.8,
        'mi_steps': 20000,
        'pi_steps': 10000,
        'reward_in_obs': True,
        'optimizer': 'adam',
        'pi_lr': 0.01,
        'mi_lr': 0.01,
        'n_mem_states': [2, 4],
        'n_seeds': 1,
        'platform': 'cpu',
        'seed': 2025,
        # 'seed': [2020 + i for i in range(10)],
    }]
}
