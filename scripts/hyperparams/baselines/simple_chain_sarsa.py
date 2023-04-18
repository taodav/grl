hparams = {
    'file_name':
        'simple_chain_sarsa.txt',
    'entry': 'grl.baselines.run',
    'args': [{
        'algo': 'dqn_sarsa',
        'spec': [
            # 'tiger-alt-start', 'network',
            #'tmaze_5_two_thirds_up',
            # 'example_7', '4x3.95', 'cheese.95',
            # 'shuttle.95',
            # 'paint.95'
            # 'bridge-repair',
            # 'hallway',
            'simple_chain'
        ],
        'num_updates': 20000,
        'alpha': 0.01,
        'seed': [2020 + i for i in range(10)],
        'log': None,
    }]
}
