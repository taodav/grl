from pathlib import Path

exp_name = Path(__file__).stem

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry':
        '-m grl.run_sample_based',
    'args': [
        {
            # Env
            'spec': [
                'tmaze_5_two_thirds_up', 'tiger-alt-start', 'cheese.95', 'network', '4x3.95',
                'shuttle.95', 'paint.95', 'hallway'
            ],
            'no_gamma_terminal': False,
            'max_episode_steps': 1000,
            'feature_encoding': 'one_hot',

            # Agent
            'algo': 'multihead_rnn',
            'epsilon': 0.1,
            'arch': ['lstm', 'gru'],
            'lr': 0.001,
            'optimizer': 'adam',

            # RNN
            'hidden_size': 12,
            'value_head_layers': 0,
            'trunc': -1, # online training
            'action_cond': 'none',

            # Multihead RNN/Lambda discrep
            'multihead_action_mode': 'td',
            'multihead_loss_mode': ['both', 'td'],
            # 'multihead_loss_mode': ['both'],
            # 'multihead_lambda_coeff': [-1, 0., 1.],
            'multihead_lambda_coeff': 0.,
            'normalize_rewards': True,

            # Replay
            'replay_size': -1,
            'batch_size': 1,

            # Logging and Checkpointing
            'offline_eval_freq': 1000,
            'offline_eval_episodes': 5,
            'offline_eval_epsilon': None, # Defaults to epsilon
            'checkpoint_freq': -1, # only save last agent

            # Experiment
            'total_steps': 150000,
            'seed': [2020 + i for i in range(10)],
            'study_name': exp_name
        },
    ]
}
