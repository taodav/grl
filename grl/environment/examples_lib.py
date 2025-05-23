import numpy as np
from pathlib import Path

from grl.mdp import random_stochastic_matrix
from grl.environment.pomdp_file import POMDPFile
from grl.utils.mdp_solver import to_dict

from definitions import ROOT_DIR

from .tmaze import tmaze, slippery_tmaze, four_tmaze, fixed_four_tmaze, tmaze_two_goals
from .compass_world import compass_world

"""
Library of POMDP specifications. Each function returns a dict of the form:
    {
        T:        transition tensor,
        R:        reward tensor,
        gamma:    discount factor,
        p0:       starting state probabilities,
        phi:      observation matrix (currently the same for all actions),
        Pi_phi:   policies to evaluate in the observation space of the POMDP
        mem_params:    memory parameters. We get our memory function by calling softmax(mem_params, axis=-1)
        Pi_phi_x: policies to evaluate in the observation space of the cross product of the underyling MDP and memory function
    }

Functions named 'example_*' come from examples in the GRL workbook.
"""

def example_3():
    # b, r1, r2, t
    T_up = np.array([
        [0., 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ])
    T_down = np.array([
        [0., 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ])
    T = np.array([T_up, T_down])

    R_up = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 5],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    R_down = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 3],
        [0, 0, 0, 3],
        [0, 0, 0, 0],
    ])
    R = np.array([R_up, R_down])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    p = 0.75
    q = 0.75
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [q, 1 - q],
            [1, 0],
        ]),
    ]

    return to_dict(T, R, 0.5, p0, phi, Pi_phi)

def example_7():

    T = np.array([
        # r, b, r, t
        [0., 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ])
    T = np.array([T, T])

    R = np.array([
        [
            [0., 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    ])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])

    Pi_phi = [
        np.array([
            [1, 0], # up, down
            [1, 0],
            [1, 0],
        ]),
        # np.array([
        #     [0, 1],
        #     [0, 1],
        #     [0, 1],
        # ]),
        # np.array([
        #     [4 / 7, 3 / 7], # known location of no discrepancy
        #     [1, 0],
        #     [1, 0],
        # ])
    ]

    Pi_phi_x = [
        np.array([
            [1., 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
        ]),
        # np.array([
        #     [0., 1], # Optimal policy with memory
        #     [1, 0],
        #     [1, 0],
        #     [1, 0],
        #     [1, 0],
        #     [1, 0],
        # ]),
    ]

    return to_dict(T, R, 0.5, p0, phi, Pi_phi, Pi_phi_x)

def example_11():
    T = np.array([[
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0.5, 0, 0.5],
        [0, 0, 0, 1],
    ]])

    R = np.array([[
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    Pi_phi = [
        np.array([
            [1],
            [1],
            [1],
        ]),
    ]

    return to_dict(T, R, 0.5, p0, phi, Pi_phi)

def example_13():
    T = np.array([[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0.5, 0, 0.5], [0, 0, 0, 1]]])

    R = np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])

    Pi_phi = [
        np.array([[1], [1], [1]]),
    ]

    return to_dict(T, R, 0.5, p0, phi, Pi_phi)

def example_14():
    # b1, b2, r, t
    T_up = np.array([[0., 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
    T_down = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])
    T = np.array([T_up, T_down])

    R_up = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    R_down = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
    R = np.array([R_up, R_down])

    p0 = np.zeros(len(T[0]))
    p0[0] = 0.75
    p0[1] = 0.25

    phi = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    p = .5
    Pi_phi = [
        np.array([
            [1, 0], # up, down
            [1, 0],
            [1, 0]
        ]),
        np.array([[0, 1], [0, 1], [0, 1]]),
        np.array([[p, 1 - p], [p, 1 - p], [0, 0]]),
    ]

    return to_dict(T, R, 0.5, p0, phi, Pi_phi)

def example_16():
    # b0, b1
    T = np.array([
        [0., 1],
        [1, 0],
    ])
    T = np.array([T, T])

    R_up = np.array([
        [0., 1],
        [0, 0],
    ])
    R_down = np.array([
        [0, 0],
        [1, 0],
    ])
    R = np.array([R_up, R_down])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1],
        [1],
    ])

    Pi_phi = [
        np.array([
            [1, 0], # up, down
        ]),
        np.array([
            [0, 1],
        ]),
    ]

    return to_dict(T, R, 0.5, p0, phi, Pi_phi)

def example_16_terminal():
    gamma_top = 0.5
    # b0, b1, t
    T = np.array([
        [0, gamma_top, 1 - gamma_top],
        [gamma_top, 0, 1 - gamma_top],
        [0, 0, 1],
    ])
    T = np.array([T, T])

    R_up = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])
    R_down = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    R = np.array([R_up, R_down])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
    ])

    Pi_phi = [
        np.array([
            [1, 0], # up, down
            [1, 0],
        ]),
        np.array([
            [0, 1],
            [0, 1],
        ]),
    ]

    return to_dict(T, R, 1, p0, phi, Pi_phi)

def example_18():
    T_up = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0], # b
        [0, 0, 0, 1, 0, 0, 0, 0], # r
        [0, 0, 0, 0, 1, 0, 0, 0], # i
        [0, 0, 0, 0, 0, 1, 0, 0], # y1
        [0, 0, 0, 0, 0, 0, 1, 0], # y2
        [0, 0, 0, 0, 0, 0, 0, 1], # c
        [0, 0, 0, 0, 0, 0, 0, 1], # u
        [0, 0, 0, 0, 0, 0, 0, 1.], # term
    ])
    T_down = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1.],
    ])
    T = np.array([T_up, T_down])

    R = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0.],
    ])
    R = np.array([R, R])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ])

    p = .75
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [p, 1 - p],
            [p, 1 - p],
            [p, 1 - p],
            [p, 1 - p],
            [p, 1 - p],
            [p, 1 - p]
        ])
    ]

    return to_dict(T, R, 1, p0, phi, Pi_phi)

def example_19():
    T = np.array([[
        [0, 0.5, 0.5, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ]])

    R = np.array([[
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    Pi_phi = [
        np.array([
            [1],
            [1],
            [1],
        ]),
    ]

    return to_dict(T, R, 0.5, p0, phi, Pi_phi)

def example_20():
    T = np.array([
        [
            # "wait"
            [0, 1, 0, 0.], #r1
            [0, 0, 1, 0], #r2
            [1, 0, 0, 0], #r3
            [0, 0, 0, 1], #t
        ],
        [
            # "go"
            [0, 0, 0, 1], #r1
            [0, 0, 0, 1], #r2
            [0, 0, 0, 1], #r3
            [0, 0, 0, 1], #t
        ]
    ])

    R = np.array([
        [
            # "wait"
            [0, 0, 0, 0.], #r1
            [0, 0, 0, 0], #r2
            [0, 0, 0, 0], #r3
            [0, 0, 0, 0], #t
        ],
        [
            # "go"
            [0, 0, 0, -1], #r1
            [0, 0, 0, -1], #r2
            [0, 0, 0, 1], #r3
            [0, 0, 0, 0], #t
        ]
    ])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        #r  t
        [1, 0.], #r1
        [1, 0], #r2
        [1, 0], #r3
        [0, 1], #t
    ])

    p = .5
    Pi_phi = [
        np.array([
            # up, down
            [p, 1 - p], #r
            [p, 1 - p], #t
        ])
    ]

    Pi_phi_x = [
        np.array([
            [p, 1 - p], #r0
            [p, 1 - p], #r1
            [p, 1 - p], #t0
            [p, 1 - p], #t1
        ]),
    ]

    return to_dict(T, R, 0.999, p0, phi, Pi_phi, Pi_phi_x)

def example_21():
    T = np.array([
        [
            # "wait"
            [0, 1, 0, 0.], #r1
            [0, 0, 1, 0], #r2
            [1, 0, 0, 0], #r3
            [0, 0, 0, 1], #t
        ],
        [
            # "go"
            [0, 0, 0, 1], #r1
            [0, 0, 0, 1], #r2
            [0, 0, 0, 1], #r3
            [0, 0, 0, 1], #t
        ]
    ])

    R = np.array([
        [
            # "wait"
            [0, 0, 0, 0.], #r1
            [0, 0, 0, 0], #r2
            [0, 0, 0, 0], #r3
            [0, 0, 0, 0], #t
        ],
        [
            # "go"
            [0, 0, 0, -1], #r1
            [0, 0, 0, -1], #r2
            [0, 0, 0, 1], #r3
            [0, 0, 0, 0], #t
        ]
    ])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        #r  b  t
        [1, 0, 0.], #r1
        [0, 1, 0], #b2
        [0, 1, 0], #b3
        [0, 0, 1], #t
    ])

    p = .5
    Pi_phi = [
        np.array([
            # up, down
            [p, 1 - p], #r
            [p, 1 - p], #b
            [p, 1 - p], #t
        ])
    ]

    Pi_phi_x = [
        np.array([
            [p, 1 - p], #r0
            [p, 1 - p], #r1
            [p, 1 - p], #b0
            [p, 1 - p], #b1
            [p, 1 - p], #t0
            [p, 1 - p], #t1
        ]),
    ]

    return to_dict(T, R, 0.999, p0, phi, Pi_phi, Pi_phi_x)

def example_22():
    # r, b1, b2, t
    p = .75
    q = .75
    T_up = np.array([
        [0, p, 1 - p, 0.],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ])
    T_down = np.array([
        [0, 1 - q, q, 0.],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ])
    T = np.array([T_up, T_down])

    R_up = np.array([
        [0, 0, 0, 0.],
        [0, 0, 0, 1],
        [0, 0, 0, -1],
        [0, 0, 0, 0],
    ])
    R_down = np.array([
        [0, 0, 0, 0.],
        [0, 0, 0, -1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ])
    R = np.array([R_up, R_down])

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        # r, b, t
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    p = .25
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [p, 1 - p],
            [p, 1 - p],
        ]),
    ]

    Pi_phi_x = [
        np.array([
            [p, 1 - p], #r0
            [p, 1 - p], #r1
            [p, 1 - p], #b0
            [p, 1 - p], #b1
            [p, 1 - p], #t0
            [p, 1 - p], #t1
        ]),
    ]

    return to_dict(T, R, 0.999, p0, phi, Pi_phi, Pi_phi_x)

def example_26():
    # [r1, b1, w1, r2, b2 w2]
    T_stay = np.array([
        [0, 1, 0, 0, 0, 0.],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
    ])
    T_flip = np.array([
        [0, 0, 0, 0, 1, 0.],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
    ])
    T = np.array([T_stay, T_flip])

    R = np.array([
        [0, 0, 0, 0, 0, 0.],
        [0, 0,-1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ])
    R = np.array([R, R])

    p0 = np.zeros(len(T[0]))
    p0[0] = 0.5
    p0[3] = 0.5

    phi = np.array([
        # r, b, w
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    p = .25
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [p, 1 - p],
            [p, 1 - p],
        ]),
    ]

    Pi_phi_x = [
        np.array([
            [p, 1 - p], #r0
            [p, 1 - p], #r1
            [p, 1 - p], #b0
            [p, 1 - p], #b1
            [p, 1 - p], #w0
            [p, 1 - p], #w1
        ]),
    ]

    return to_dict(T, R, 0.9, p0, phi, Pi_phi, Pi_phi_x)

def example_26a():
    # [r1, b1, w1, r2, b2 w2]
    # but rewards are observable: distinguishes w1 from w2
    T_stay = np.array([
        [0, 1, 0, 0, 0, 0.],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
    ])
    T_flip = np.array([
        [0, 0, 0, 0, 1, 0.],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
    ])
    T = np.array([T_stay, T_flip])

    R = np.array([
        [0, 0, 0, 0, 0, 0.],
        [0, 0,-1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ])
    R = np.array([R, R])

    p0 = np.zeros(len(T[0]))
    p0[0] = 0.5
    p0[3] = 0.5

    phi = np.array([
        # r, b, w1, w2
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])

    p = .25
    Pi_phi = [
        np.array([
            [p, 1 - p], # up, down
            [p, 1 - p],
            [p, 1 - p],
            [p, 1 - p],
        ]),
    ]

    Pi_phi_x = [
        np.array([
            [p, 1 - p], #r0
            [p, 1 - p], #r1
            [p, 1 - p], #b0
            [p, 1 - p], #b1
            [p, 1 - p], #w1_0
            [p, 1 - p], #w1_1
            [p, 1 - p], #w2_0
            [p, 1 - p], #w2_1
        ]),
    ]

    return to_dict(T, R, 0.9, p0, phi, Pi_phi, Pi_phi_x)

def simple_chain(n: int = 10):
    T = np.zeros((n, n))
    states = np.arange(n)
    starts = states[:-1]
    ends = states[1:]
    T[starts, ends] = 1
    T[n - 1, n - 1] = 1
    T = np.expand_dims(T, 0)

    R = np.zeros((n, n))
    R[-2, -1] = 1
    R = np.expand_dims(R, 0)

    p0 = np.zeros(n)
    p0[0] = 1

    phi = np.eye(n)

    Pi_phi = [np.ones((n, 1))]

    return to_dict(T, R, 0.9, p0, phi, Pi_phi)

def tmaze_hyperparams(corridor_length: int = 5,
                      discount: float = 0.9,
                      junction_up_pi: float = 2 / 3,
                      **kwargs):
    """
    tmaze, except set the junction length, discount, and t-junction policy based on hyperparams
    policy is still go right everywhere except for junction.
    """
    # n_obs x n_actions
    Pi_phi = [
        np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0],
                  [junction_up_pi, 1 - junction_up_pi, 0, 0], [1, 0, 0, 0]])
    ]

    # memory policy is observations * memory bits (2) x n_actions
    Pi_phi_x = [Pi_phi[0].repeat(2, axis=0)]
    return to_dict(*tmaze(corridor_length, discount=discount), Pi_phi, Pi_phi_x)

def po_simple_chain(n: int = 10):
    spec = simple_chain(n)

    # only one observation
    phi = np.ones((n, 1))
    Pi_phi = [np.ones((1, 1))]
    spec['phi'] = phi
    spec['Pi_phi'] = Pi_phi
    return spec

def tmaze_eps_hyperparams(corridor_length: int = 5,
                          discount: float = 0.9,
                          junction_up_pi: float = 2 / 3,
                          epsilon: float = 0.1,
                          **kwargs):
    """
    tmaze, except set the junction length, discount, and t-junction policy based on hyperparams
    policy is still go right everywhere except for junction.
    """
    # n_obs x n_actions
    Pi_phi = [
        np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0],
                  [junction_up_pi, 1 - junction_up_pi, 0, 0], [1, 0, 0, 0]])
    ]

    Pi_phi[0] *= (1 - epsilon)
    Pi_phi[0] += (epsilon / Pi_phi[0].shape[-1])

    # memory policy is observations * memory bits (2) x n_actions
    Pi_phi_x = [Pi_phi[0].repeat(2, axis=0)]
    return to_dict(*tmaze(corridor_length, discount=discount), Pi_phi, Pi_phi_x)

def tmaze_5_two_thirds_up():
    # n_obs x n_actions
    n = 5
    discount = 0.9
    Pi_phi = [
        np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [2 / 3, 1 / 3, 0, 0], [1, 0, 0, 0]])
    ]

    # memory policy is observations * memory bits (2) x n_actions
    Pi_phi_x = [Pi_phi[0].repeat(2, axis=0)]
    return to_dict(*tmaze(n, discount=discount), Pi_phi, Pi_phi_x)

def fixed_four_tmaze_two_thirds_up():
    # n_obs x n_actions
    n = 0
    discount = 0.9
    Pi_phi = [
        np.array([
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [2 / 3, 1 / 3],
            [1, 0]])
    ]

    # memory policy is observations * memory bits (2) x n_actions
    Pi_phi_x = [Pi_phi[0].repeat(2, axis=0)]
    return to_dict(*fixed_four_tmaze(n, discount=discount), Pi_phi, Pi_phi_x)

def four_tmaze_two_thirds_up():
    # n_obs x n_actions
    n = 1
    discount = 0.9
    Pi_phi = [
        np.array([
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [2 / 3, 1 / 3, 0, 0],
            [1, 0, 0, 0]])
    ]

    # memory policy is observations * memory bits (2) x n_actions
    Pi_phi_x = [Pi_phi[0].repeat(2, axis=0)]
    return to_dict(*four_tmaze(n, discount=discount), Pi_phi, Pi_phi_x)

def tmaze_2_two_thirds_up():
    # n_obs x n_actions
    n = 2
    discount = 0.99999999999
    Pi_phi = [
        np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [2 / 3, 1 / 3, 0, 0], [1, 0, 0, 0]])
    ]

    # memory policy is observations * memory bits (2) x n_actions
    Pi_phi_x = [Pi_phi[0].repeat(2, axis=0)]
    tmaze_instance = tmaze(n, discount=discount, good_term_reward=1, bad_term_reward=0)
    return to_dict(*tmaze_instance, Pi_phi, Pi_phi_x)

def tmaze_5_two_thirds_up_fully_observable():
    # n_obs x n_actions
    n = 5
    discount = 0.9
    T, R, discount, p0, phi = tmaze(n, discount=discount)
    phi_fully_observable = np.eye(T.shape[-1])

    pi_obs = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [2 / 3, 1 / 3, 0, 0],
                       [1, 0, 0, 0]])
    pi = phi @ pi_obs

    Pi_phi = [pi]

    # memory policy is observations * memory bits (2) x n_actions
    Pi_phi_x = [Pi_phi[0].repeat(2, axis=0)]
    return to_dict(T, R, discount, p0, phi_fully_observable, Pi_phi, Pi_phi_x)

def tmaze_5_two_thirds_up_almost_fully_observable():
    """
    Almost fully observable t-maze.
    Goal "orientation" is always observable, so only aliased states
    are the corridor states.
    """
    # n_obs x n_actions
    n = 5
    discount = 0.9
    T, R, discount, p0, phi = tmaze(n, discount=discount)
    phi_almost_fully_observable = np.zeros((T.shape[-1], 6 + 1))
    phi_almost_fully_observable[0, 0] = 1
    phi_almost_fully_observable[1, 1] = 1
    phi_almost_fully_observable[np.arange(1, n + 1) * 2, 2] = 1
    phi_almost_fully_observable[np.arange(1, n + 1) * 2 + 1, 3] = 1
    phi_almost_fully_observable[-3, 4] = 1
    phi_almost_fully_observable[-2, 5] = 1
    phi_almost_fully_observable[-1, -1] = 1

    pi = np.zeros((phi_almost_fully_observable.shape[-1], 4))
    pi[:, 2] = 1
    pi[-2, :] = 0
    pi[-2, 0] = 2 / 3
    pi[-2, 1] = 1 / 3
    pi[-3, :] = 0
    pi[-3, 0] = 2 / 3
    pi[-3, 1] = 1 / 3

    Pi_phi = [pi]

    # memory policy is observations * memory bits (2) x n_actions
    Pi_phi_x = [Pi_phi[0].repeat(2, axis=0)]
    return to_dict(T, R, discount, p0, phi_almost_fully_observable, Pi_phi, Pi_phi_x)

def slippery_tmaze_5_two_thirds_up():
    # n_obs x n_actions
    n = 5
    discount = 0.9
    Pi_phi = [
        np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [2 / 3, 1 / 3, 0, 0], [1, 0, 0, 0]])
    ]

    # memory policy is observations * memory bits (2) x n_actions
    Pi_phi_x = [Pi_phi[0].repeat(2, axis=0)]
    return to_dict(*slippery_tmaze(n, discount=discount, slip_prob=0.3), Pi_phi, Pi_phi_x)

def slippery_tmaze_5_random():
    """
    Slippery t-maze, with a randomly sampled policy
    """
    n = 5
    discount = 0.9
    Pi_phi = [random_stochastic_matrix((5, 4))]

    # memory policy is observations * memory bits (2) x n_actions
    Pi_phi_x = [Pi_phi[0].repeat(2, axis=0)]
    return to_dict(*slippery_tmaze(n, discount=discount, slip_prob=0.3), Pi_phi, Pi_phi_x)

def tmaze_5_obs_optimal():
    """
    T-Maze, with the optimal policy given observations.
    """
    n = 5
    discount = 0.9
    Pi_phi = [
        np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0.13, 0.87, 0]])
    ]
    # memory policy is observations * memory bits (2) x n_actions
    Pi_phi_x = [Pi_phi[0].repeat(2, axis=0)]
    return to_dict(*tmaze(n, discount=discount), Pi_phi, Pi_phi_x)

def tiger_fixed_pi():
    file_path = Path(ROOT_DIR, 'grl', 'environment', 'pomdp_files', f'tiger-alt-start.POMDP')
    spec = POMDPFile(file_path).get_spec()
    Pi_phi = [np.array([
        [1, 0, 0],
        [0.1, 0.1, 0.8],
        [0.1, 0.7, 0.2],
        [0, 0, 1],
    ])]

    # memory policy is observations * memory bits (2) x n_actions
    Pi_phi_x = [Pi_phi[0].repeat(2, axis=0)]

    spec['Pi_phi'] = Pi_phi
    spec['Pi_phi_x'] = Pi_phi_x

    return spec

def count_by_n(n: int = 5):
    T_right = np.zeros((n + 1, n + 1))
    T_right[np.arange(n - 1), np.arange(1, n)] = 1
    T_right[[-2, -1], -1] = 1

    T_up = np.zeros_like(T_right)
    T_up[:, -1] = 1

    T = np.array([T_up, T_right])

    R_right = np.zeros_like(T_right)
    R_right[-2, -1] = -1

    R_up = np.zeros_like(T_up)
    # R_up[np.arange(n), -1] = -1
    R_up[-2, -1] = 1
    R = np.array([R_up, R_right])

    phi = np.zeros((n + 1, 2))
    phi[:n, 0] = 1
    phi[-1, -1] = 1

    p0 = np.zeros(n + 1)
    p0[0] = 1

    pi_phi = None

    return to_dict(T, R, 0.9, p0, phi, pi_phi)

def short_corridor():
    """
    Short corridor in func. approx as described in Example 13.1
    In the RL Book.
    """
    T_left = np.array([
        # r, b, r, t
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    T_right = np.array([
        # r, b, r, t
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ])
    T = np.array([T_left, T_right])

    R = -np.ones((2, 4, 4))
    R[:, -1, -1] = 0

    p0 = np.zeros(len(T[0]))
    p0[0] = 1

    phi = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
    ])

    Pi_phi = [
        np.array([
            [1, 0], # up, down
            [1, 0],
        ]),
    ]

    return to_dict(T, R, 0.9999, p0, phi, Pi_phi, None)


def compass_random():
    Pi_phi = [
        np.ones((5, 3)) / 3
    ]
    return to_dict(*compass_world(3), Pi_phi)


def tmaze_5_separate_goals_two_thirds_up():
    # n_obs x n_actions
    n = 5
    discount = 0.9
    Pi_phi = [
        np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [2 / 3, 1 / 3, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    ]

    # memory policy is observations * memory bits (2) x n_actions
    Pi_phi_x = [Pi_phi[0].repeat(2, axis=0)]
    return to_dict(*tmaze_two_goals(n, discount=discount), Pi_phi, Pi_phi_x)
