from pathlib import Path

from scripts.plotting.parse_experiments import parse_dirs

from definitions import ROOT_DIR

belief_perf = {
    '4x3.95': 2.001088974770953,
    'cheese.95': 3.5788796295075453,
    'network': 296.2187032020679,
    'paint.95': 3.293597084371071,
    'parity_check': 0.8099999999999998,
    'shuttle.95': 32.88972468934434,
    'tiger-alt-start': 3.7701893248807115,
    'tmaze_5_two_thirds_up': 2.1257640000000007
}


if __name__ == '__main__':
    spec_plot_order = [
        'network',
        # 'paint.95',
        '4x3.95', 'tiger-alt-start', 'shuttle.95', 'cheese.95', 'tmaze_5_two_thirds_up',
        'parity_check'
    ]

    # this assumes that each experiment dir will be a column in the bar plot
    experiment_dirs = [
        Path(ROOT_DIR, 'results', 'variance_pg'),
        Path(ROOT_DIR, 'results', 'ld_pg'),
    ]




