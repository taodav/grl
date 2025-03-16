hparams = {
    'file_name':
        'lstm_reinforce_all_envs.txt',
    'entry':
        '-m grl.baselines.run',
    'args': [{
        'algo': 'lstm_reinforce',
        'spec': [
            'tiger-alt-start',
            'network',
            'tmaze_5_two_thirds_up',
            'example_7',
            '4x3.95',
            'shuttle.95',
            'paint.95',
            'hallway',
            'cheese.95',
        ],
        'gamma_terminal': True,
        'num_updates': int(1.5e5),
        'hidden_size': 12,
        'alpha': 0.001,
        'trunc_len': 100, # currently unused
        'seed': [2020 + i for i in range(10)],
    }]
}
