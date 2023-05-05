import numpy as np
from grl.utils.mdp import to_dict

def prisoners_dilemma_tit_for_tat(episode_duration: int = 10):

    T_c = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])
    T_d = np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    T = np.array([T_c, T_d])

    R = np.array(
        [
            [3, 5, 0, 1],
            [3, 5, 0, 1],
            [3, 5, 0, 1],
            [3, 5, 0, 1],
        ]
    )
    R = np.array([R, R])

    p0 = np.zeros(len(T[0]))
    p0[1] = 1

    phi = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ])

    p = 2/3
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [1 - p, p],
        ]),
    ]
    gamma = (episode_duration - 1) / episode_duration
    # gamma = 0

    return to_dict(T, R, gamma, p0, phi, Pi_phi)

def prisoners_dilemma_extort(episode_duration: int = 10):
    T_c = np.array([
        [7/8, 1/8, 0, 0],
        [7/16, 1 - 7/16, 0, 0],
        [3/8, 1 - 3/8, 0, 0],
        [0, 1, 0, 0]
    ])
    T_d = np.array([
        [0, 0, 7/8, 1/8],
        [0, 0, 7/16, 1 - 7/16],
        [0, 0, 3/8, 1 - 3/8],
        [0, 0, 0, 1]
    ])
    T = np.array([T_c, T_d])
    R = np.array(
        [
            [3, 5, 0, 1],
            [3, 5, 0, 1],
            [3, 5, 0, 1],
            [3, 5, 0, 1],
        ]
    )

    R = np.array([R, R])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ])

    p = 2/3
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [1 - p, p],
        ]),
    ]
    gamma = (episode_duration - 1) / episode_duration
    # gamma = 0

    return to_dict(T, R, gamma, p0, phi, Pi_phi)

def fully_observable_prisoners_dilemma_extort(episode_duration: int = 10):
    T_c = np.array([
        [7/8, 1/8, 0, 0],
        [7/16, 1 - 7/16, 0, 0],
        [3/8, 1 - 3/8, 0, 0],
        [0, 1, 0, 0]
    ])
    T_d = np.array([
        [0, 0, 7/8, 1/8],
        [0, 0, 7/16, 1 - 7/16],
        [0, 0, 3/8, 1 - 3/8],
        [0, 0, 0, 1]
    ])
    T = np.array([T_c, T_d])
    R = np.array(
        [
            [5, -10, 10, 0],
            [5, -10, 10, 0],
            [5, -10, 10, 0],
            [5, -10, 10, 0],
        ]
    )

    R = np.array([R, R])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.eye(len(T[0]))

    p = 2/3
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [p, 1 - p], # up, down
            [1 - p, p],
            [1 - p, p],
        ]),
    ]
    gamma = (episode_duration - 1) / episode_duration
    # gamma = 0

    return to_dict(T, R, gamma, p0, phi, Pi_phi)

def prisoners_dilemma_grudger2(episode_duration: int = 10):

    T_c = np.array([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
    ])
    T_d = np.array([
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
    ])
    T = np.array([T_c, T_d])

    R = np.array(
        [
            [0, 3, 5, 3, 5, 0, 1],
            [0, 3, 5, 3, 5, 0, 1],
            [0, 3, 5, 3, 5, 0, 1],
            [0, 3, 5, 3, 5, 0, 1],
            [0, 3, 5, 3, 5, 0, 1],
            [0, 3, 5, 3, 5, 0, 1],
            [0, 3, 5, 3, 5, 0, 1],
        ]
    )
    R = np.array([R, R])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ])

    p = 2/3
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [p, 1 - p], # up, down
            [1 - p, p],
        ]),
    ]
    gamma = (episode_duration - 1) / episode_duration
    # gamma = 0

    return to_dict(T, R, gamma, p0, phi, Pi_phi)

def prisoners_dilemma_majority3(episode_duration: int = 10):

    T_c = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
    ])
    T_d = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
    ])
    T = np.array([T_c, T_d])

    R = np.array(
        [
            [0, 3, 5, 3, 5, 0, 1, 1, 0],
            [0, 3, 5, 3, 5, 0, 1, 1, 0],
            [0, 3, 5, 3, 5, 0, 1, 1, 0],
            [0, 3, 5, 3, 5, 0, 1, 1, 0],
            [0, 3, 5, 3, 5, 0, 1, 1, 0],
            [0, 3, 5, 3, 5, 0, 1, 1, 0],
            [0, 3, 5, 3, 5, 0, 1, 1, 0],
            [0, 3, 5, 3, 5, 0, 1, 1, 0],
            [0, 3, 5, 3, 5, 0, 1, 1, 0],
        ]
    )
    R = np.array([R, R])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ])

    p = 2/3
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [p, 1 - p], # up, down
            [1 - p, p],
        ]),
    ]
    gamma = (episode_duration - 1) / episode_duration
    # gamma = 0

    return to_dict(T, R, gamma, p0, phi, Pi_phi)

def prisoners_dilemma_treasure_hunt(episode_duration: int = 10):

    T_c = np.zeros((14, 14))
    # These tuples are (state, next_state) transitions for COOPERATE
    cooperate_transitions = [
        (0, 2), (1, 4), (2, 6), (3, 4), (4, 4), (5, 4), (6, 8),
        (7, 10), (8, 12), (9, 10), (10, 10), (11, 8), (12, 12), (13, 8)
    ]
    for s, ns in cooperate_transitions:
        T_c[s, ns] = 1

    T_d = np.zeros((14, 14))
    defect_transitions = [
        (0, 1), (1, 3), (2, 5), (3, 3), (4, 3), (5, 3), (6, 7),
        (7, 9), (8, 13), (9, 9), (10, 9), (11, 11), (12, 13), (13, 11)
    ]
    for s, ns in defect_transitions:
        T_d[s, ns] = 1

    T = np.array([T_c, T_d])

    R = np.array(
        [
            [0, 0, 3, 1, 5, 0, 3, 1, 5, 0, 3, 1, 3, 0],
        ]
    )
    R = R.repeat(R.shape[-1], axis=0)
    R = np.array([R, R])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
    ])

    p = 2/3
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [p, 1 - p], # up, down
            [1 - p, p],
        ]),
    ]
    gamma = (episode_duration - 1) / episode_duration
    # gamma = 0

    return to_dict(T, R, gamma, p0, phi, Pi_phi)

def prisoners_dilemma_all_d(episode_duration: int = 10):

    T_c = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ])
    T_d = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ])
    T = np.array([T_c, T_d])

    R = np.array(
        [
            [0, 1, 5],
            [0, 1, 5],
            [0, 1, 5],
        ]
    )
    R = np.array([R, R])

    p0 = np.zeros(len(T[0]))
    p0[1] = 1

    phi = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 0, 1],
    ])

    p = 2/3
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [p, 1 - p], # up, down
            [1 - p, p],
        ]),
    ]
    gamma = (episode_duration - 1) / episode_duration
    # gamma = 0

    return to_dict(T, R, gamma, p0, phi, Pi_phi)

def prisoners_dilemma_all_c(episode_duration: int = 10):

    T_c = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ])
    T_d = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ])
    T = np.array([T_c, T_d])

    R = np.array(
        [
            [0, 0, 3],
            [0, 0, 3],
            [0, 0, 3],
        ]
    )
    R = np.array([R, R])

    p0 = np.zeros(len(T[0]))
    p0[1] = 1

    phi = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
    ])

    p = 2/3
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [p, 1 - p], # up, down
            [1 - p, p],
        ]),
    ]
    gamma = (episode_duration - 1) / episode_duration
    # gamma = 0

    return to_dict(T, R, gamma, p0, phi, Pi_phi)


def prisoners_dilemma_alternator(episode_duration: int = 10):

    T_c = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]

    ])
    T_d = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0]

    ])
    T = np.array([T_c, T_d])

    R = np.array(
        [
            [0, 0, 3, 0, 0],
        ]
    )
    R = R.repeat(R.shape[-1], axis=0)
    R = np.array([R, R])

    p0 = np.zeros(len(T[0]))
    p0[1] = 1

    phi = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ])

    p = 2/3
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [p, 1 - p], # up, down
            [1 - p, p],
        ]),
    ]
    gamma = (episode_duration - 1) / episode_duration
    # gamma = 0

    return to_dict(T, R, gamma, p0, phi, Pi_phi)

def prisoners_dilemma_sugar(episode_duration: int = 10):

    T_c = np.array([
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],


    ])
    T_d = np.array([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],

    ])
    T = np.array([T_c, T_d])

    R = np.array(
        [
            [0, 1, 0, 1, 0, 3, 5],
        ]
    )
    R = R.repeat(R.shape[-1], axis=0)
    R = np.array([R, R])

    p0 = np.zeros(len(T[0]))
    p0[1] = 1

    phi = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ])

    p = 2/3
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [p, 1 - p], # up, down
            [1 - p, p],
        ]),
    ]
    gamma = (episode_duration - 1) / episode_duration
    # gamma = 0

    return to_dict(T, R, gamma, p0, phi, Pi_phi)

