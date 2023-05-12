hparams = {
    'file_name':
        'cheese_lstm.txt',
    'entry': '-m grl.baselines.run',
    'args': [{
        'algo': 'lstm_reinforce',
        'spec': [
            'cheese.95',
        ],
        'gamma_terminal': True,
        'num_updates': int(1.5e5),
        'hidden_size': 12,
        'alpha': 0.001,
        'trunc_len': 100, # currently unused
        'seed': [2020 + i for i in range(10)],
        'log': None,
    }]
}
