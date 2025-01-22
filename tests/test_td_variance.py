
from grl.environment import load_pomdp
from grl.environment.policy_lib import switching_two_thirds_right_policy

from scripts.td_variance import get_variances

if __name__ == "__main__":
    # env_str = 'tmaze_5_two_thirds_up'
    env_str = 'counting_wall'
    # env_str = 'switching'

    pomdp, pi_dict = load_pomdp(env_str,
                                corridor_length=5,
                                discount=0.9)

    # This is state-based variance
    if env_str == 'switching':
        pi = switching_two_thirds_right_policy()
    elif env_str == 'counting_wall':
        raise NotImplementedError
    else:
        pi = pi_dict['Pi_phi'][0]

    get_variances(pi, pomdp)

