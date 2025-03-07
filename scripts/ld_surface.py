from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import griddata

from grl.utils.loss import discrep_loss
from grl.environment import load_pomdp

from definitions import ROOT_DIR


def maybe_spec_map(id: str):
    spec_map = {
        '4x3.95': '4x3',
        'cheese.95': 'cheese',
        'paint.95': 'paint',
        'shuttle.95': 'shuttle',
        'example_7': 'ex. 7',
        'tmaze_5_two_thirds_up': 'tmaze 5',
        'tiger-alt-start': 'tiger',
        'parity_check': 'parity'
    }

    if id not in spec_map:
        return id
    return spec_map[id]

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.sans-serif": ["Computer Modern Sans serif"],
    "font.monospace": ["Computer Modern Typewriter"],
    "axes.labelsize": 12,  # LaTeX default is 10pt
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

def dstack_product(x, y):
    return jnp.dstack(jnp.meshgrid(x, y)).reshape(-1, 2)


if __name__ == "__main__":
    n = 30
    env_str = 'tmaze_5_two_thirds_up'

    pomdp, pi_dict = load_pomdp(env_str,
                                corridor_length=5,
                                discount=0.9)
    pi = pi_dict['Pi_phi'][0]

    lambdas = jnp.linspace(0, 1, num=n)
    lambda_pairs = dstack_product(lambdas, lambdas)

    def discrep_with_lambdas(pi, pomdp, ls):
        return discrep_loss(pi, pomdp, lambda_0=ls[0], lambda_1=ls[1])

    vmap_discrep = jax.vmap(discrep_with_lambdas, in_axes=[None, None, 0])
    lds, _, _ = vmap_discrep(pi, pomdp, lambda_pairs)

    lambda_pairs, lds = np.array(lambda_pairs), np.array(lds)

    grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]

    grid_lds = griddata(lambda_pairs, lds, (grid_x, grid_y), method='cubic')

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # c1 = rax.contourf(grid_x, grid_y, grid_return_mean, levels=20, cmap="viridis",
    #                   vmin=ret_min, vmax=ret_max)
    c1 = ax.contourf(grid_x, grid_y, grid_lds, levels=30, cmap="viridis")
    cbar_1 = fig.colorbar(c1, ax=ax, label='LD')
    ax.legend()
    ax.set_title(f'LDs for {maybe_spec_map(env_str)}')

    ax.set_xlabel('$\lambda_0$')
    ax.set_ylabel('$\lambda_1$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    plt.show()

    plt.savefig(Path(ROOT_DIR, 'results', f'ld_surface_{env_str}.png'))
