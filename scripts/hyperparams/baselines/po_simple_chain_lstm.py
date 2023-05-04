hparams = {
    'file_name':
        'po_simple_chain_lstm.txt',
    'entry':
        '-m grl.baselines.run',
    'args': [{
        'algo': 'lstm_sarsa',
        'spec': [
            # 'tiger-alt-start', 'network',
            #'tmaze_5_two_thirds_up',
            # 'example_7', '4x3.95', 'cheese.95',
            # 'shuttle.95',
            # 'paint.95'
            # 'bridge-repair',
            # 'hallway',
            'po_simple_chain'
        ],
        'num_updates': int(1e3),
        'start_epsilon': 0.1,
        'epsilon': 0.1,
        'alpha': 0.01,
        'trunc_len': 100, # currently unused
        'seed': [2020 + i for i in range(10)],
        'log': None,
    }]
}
