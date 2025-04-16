from jax import random
import numpy as np

from grl.environment import load_pomdp
from grl.memory.lib import get_memory
from grl.loss import mem_discrep_loss


if __name__ == "__main__":
    spec = 'fixed_four_tmaze_two_thirds_up'
    seed = 2020
    n_samples = 100

    rng = random.PRNGKey(seed=seed)
    np.random.seed(seed)

    pomdp, info = load_pomdp(spec)
    pi = info['Pi_phi'][0]

    max_ld = -float('inf')
    for i in range(n_samples):
        mem_params = get_memory('0', pomdp.observation_space.n, pomdp.action_space.n,
                                n_mem_states=2)

        # TODO: maybe also randomly sample?
        inp_aug_pi = pi.repeat(mem_params.shape[-1], axis=0)

        mem_ld = mem_discrep_loss(mem_params, inp_aug_pi, pomdp).item()
        if mem_ld > max_ld:
            max_ld = mem_ld

    print(f"Maximum LD over memory functions: {max_ld}")
