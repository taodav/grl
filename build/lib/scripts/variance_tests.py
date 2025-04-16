import numpy as np
import jax

from grl.environment import load_pomdp
from grl.utils.math import reverse_softmax
from grl.memory import memory_cross_product
from grl.memory.lib import counting_walls_1_bit_mem, switching_optimal_deterministic_1_bit_mem
from grl.memory.lib import memory_18 as tmaze_optimal_mem
from grl.memory.lib import memory_20 as tmaze_two_goals_optimal_mem

from grl.loss.variance import get_variances
from grl.loss import mem_tde_loss, mem_discrep_loss

from grl.environment.policy_lib import (
    switching_two_thirds_right_policy, counting_wall_optimal_memoryless_policy,
    switching_og_two_thirds_right_policy
)


if __name__ == "__main__":
    # env_str = 'tmaze_5_two_thirds_up'
    # env_str = 'tmaze_5_separate_goals_two_thirds_up'
    # env_str = 'counting_wall'
    env_str = 'switching'

    pi_str = 'random'
    reward_in_obs = False
    rng = jax.random.PRNGKey(2024)

    pomdp, pi_dict = load_pomdp(env_str,
                                corridor_length=5,
                                discount=0.9,
                                reward_in_obs=reward_in_obs)

    # This is state-based variance
    if env_str == 'switching':
        pi = switching_two_thirds_right_policy()
        mem_fn = switching_optimal_deterministic_1_bit_mem()
        if reward_in_obs:
            # since reward_in_obs is True
            mem_fn = np.concatenate((mem_fn, mem_fn[:, -1:]), axis=1)
            pi = np.concatenate((pi, pi[-1:]), axis=0)
    elif env_str == 'switching_og':
        pi = switching_og_two_thirds_right_policy()
        # TODO: this
        mem_fn = switching_optimal_deterministic_1_bit_mem()
        if reward_in_obs:
            # since reward_in_obs is True
            mem_fn = np.concatenate((mem_fn, mem_fn[:, -1:]), axis=1)
            pi = np.concatenate((pi, pi[-1:]), axis=0)
    elif env_str == 'counting_wall':
        pi = counting_wall_optimal_memoryless_policy()
        mem_fn = counting_walls_1_bit_mem()
    else:
        pi = pi_dict['Pi_phi'][0]
        mem_fn = None
        if env_str == 'tmaze_5_two_thirds_up':
            mem_fn = tmaze_optimal_mem
            if reward_in_obs:
                # since reward_in_obs is True
                term_mem_fn = mem_fn[:, -1:].repeat(2, axis=1)
                mem_fn = np.concatenate((mem_fn, term_mem_fn), axis=1)
        elif env_str == 'tmaze_5_separate_goals_two_thirds_up':
            mem_fn = tmaze_two_goals_optimal_mem


    mem_params = reverse_softmax(mem_fn)

    # if pi_str == 'random':
    #     pi_rng, mem_pi_rng, rng = random.split(rng, 3)
    #     pi_shape = (pomdp.observation_space.n, pomdp.action_space.n)
    #     pi_params = random.normal(pi_rng, shape=pi_shape) * 0.5
    #     pi = jax.nn.softmax(pi_params, axis=-1)
    #     mem_aug_pi_shape = (pomdp.observation_space.n * mem_params.shape[-1], pomdp.action_space.n)
    #     mem_aug_pi_params = random.normal(mem_pi_rng, shape=mem_aug_pi_shape) * 0.5
    #     mem_aug_pi = jax.nn.softmax(mem_aug_pi_params, axis=-1)
    # else:
    mem_aug_pi = pi.repeat(mem_params.shape[-1], axis=0)

    variances, info = get_variances(pi, pomdp)

    mem_aug_pomdp = memory_cross_product(mem_params, pomdp)

    mem_rng, rng = jax.random.split(rng)
    n_mem_states = mem_params.shape[-1]
    mem_shape = (pomdp.action_space.n, pomdp.observation_space.n, n_mem_states, n_mem_states)
    random_mem_params = jax.random.normal(mem_rng, shape=mem_shape) * 0.5

    random_mem_aug_pomdp = memory_cross_product(random_mem_params, pomdp)

    # get variances
    random_mem_aug_variances, random_mem_aug_info = get_variances(mem_aug_pi, random_mem_aug_pomdp)
    random_mem_aug_variances, random_mem_aug_info = jax.tree.map(lambda x: np.array(x), (random_mem_aug_variances, random_mem_aug_info))

    mem_aug_variances, mem_aug_info = get_variances(mem_aug_pi, mem_aug_pomdp)
    mem_aug_variances, mem_aug_info = jax.tree.map(lambda x: np.array(x), (mem_aug_variances, mem_aug_info))

    # TD errors
    random_mem_td_err = mem_tde_loss(random_mem_params, mem_aug_pi, pomdp)
    mem_td_err = mem_tde_loss(mem_params, mem_aug_pi, pomdp)

    # LDs
    random_mem_ld = mem_discrep_loss(random_mem_params, mem_aug_pi, pomdp)
    mem_ld = mem_discrep_loss(mem_params, mem_aug_pi, pomdp)

    print()

