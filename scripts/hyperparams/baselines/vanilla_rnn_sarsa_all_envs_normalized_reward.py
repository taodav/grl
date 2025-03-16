hparams = {
    'file_name':
        'vanilla_rnn_sarsa_all_envs_normalized_reward.txt',
    'entry':
        '-m grl.baselines.run',
    'args': [{
        'algo': 'vanilla_rnn_sarsa',
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
        'normalize_reward_range': True,
        'gamma_terminal': True,
        'num_updates': int(2e6),
        'hidden_size': 12,
        'start_epsilon': 0.1,
        'epsilon': 0.1,
        'alpha': 0.001,
        'trunc_len': 100, # currently unused
        'seed': [2020 + i for i in range(10)],
    }]
}
