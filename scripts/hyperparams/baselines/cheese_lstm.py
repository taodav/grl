hparams = {
    'file_name':
        'cheese_lstm.txt',
    'entry': 'grl.baselines.run',
    'args': [{
        'algo': 'lstm_sarsa',
        'spec': [
            # 'tiger-alt-start', 'network',
            #'tmaze_5_two_thirds_up',
            # 'example_7', '4x3.95', 
            'cheese.95',
            # 'shuttle.95',
            # 'paint.95'
            # 'bridge-repair',
            # 'hallway',
            #'po_simple_chain'
        ],
        'gamma_terminal': True,
        'num_updates': int(1.5e5),
        'start_epsilon': 1.,
        'epsilon': 0.1,
        'epsilon_anneal_steps': int(1.5e5 / 4),
        'alpha': 0.01,
        'trunc_len': 100, # currently unused
        'seed': [2020 + i for i in range(10)],
        'log': None,
    }]
}
