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
            [-2, 0, -10, -5],
            [-2, 0, -10, -5],
            [-2, 0, -10, -5],
            [-2, 0, -10, -5],
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

def prisoners_dilemma_extort(episode_duration: int = 100):
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
            [0, -10, 10, 0],
            [0, -10, 10, 0],
            [0, -10, 10, 0],
            [0, -10, 10, 0],
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

