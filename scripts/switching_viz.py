from pathlib import Path

import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import numpy as np
import pyvista as pv

from grl.environment import load_pomdp
from grl.memory import memory_cross_product
from grl.utils.file_system import load_info
from grl.utils.mdp import POMDP
from grl.utils.policy_eval import analytical_pe, functional_solve_mdp


@jax.jit(static_argnames=['n_obs', 'bins'])
def get_interpolated_pis(n_obs: int, bins: int):
    prob_vals = jnp.linspace(0, 1, num=bins)

    # Create n copies of the array
    grids = jnp.meshgrid(*[prob_vals] * n_obs, indexing='ij')
    # Stack and reshape to get the Cartesian product
    action_0_probs = jnp.stack(grids, axis=-1).reshape(-1, n_obs)
    action_1_probs = 1 - action_0_probs

    return jnp.stack([action_0_probs, action_1_probs], axis=-1)

def solve_large_obs_vals(obs_pis: jnp.ndarray, pomdp: POMDP,
                         batch_size: int = 100):
    vmap_analytical_pe = jax.vmap(analytical_pe, in_axes=[0, None])

    # make batches
    n_rest_obs = obs_pis.shape[0] // batch_size
    batch_obs_pis = obs_pis.reshape((n_rest_obs, batch_size, *obs_pis.shape[1:]))

    @scan_tqdm(n_rest_obs)
    @jax.jit
    def solve_obs(_, inp):
        i, obs_pi = inp
        obs_vals = vmap_analytical_pe(obs_pi, pomdp)
        return _, obs_vals

    _, obs_vals = jax.lax.scan(
        solve_obs, None, (jnp.arange(n_rest_obs), batch_obs_pis), n_rest_obs
    )
    obs_vals = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), obs_vals)
    return obs_vals

def solve_large_state_vals(state_pis: jnp.ndarray, pomdp: POMDP,
                           batch_size: int = 100):
    n_rest_state = state_pis.shape[0] // batch_size
    batch_state_pis = state_pis.reshape((n_rest_state, batch_size, *state_pis.shape[1:]))

    vmap_functional_solve_mdp = jax.vmap(functional_solve_mdp, in_axes=[0, None])

    @scan_tqdm(n_rest_state)
    @jax.jit
    def solve_state(_, inp):
        i, state_pi = inp
        state_vals = vmap_functional_solve_mdp(state_pi, pomdp)
        return _, state_vals

    _, state_vals = jax.lax.scan(
        solve_state, None, (jnp.arange(n_rest_state), batch_state_pis), n_rest_state
    )
    state_vals = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), state_vals)
    return state_vals


def get_value_fns(pomdp: POMDP, eval_obs_policy: jnp.ndarray = None, batch_size: int = 100, bins: int = 20):

    obs_pis = get_interpolated_pis(pomdp.observation_space.n, bins)
    state_pis = get_interpolated_pis(pomdp.state_space.n, bins)

    obs_vals = solve_large_obs_vals(obs_pis, pomdp, batch_size=batch_size)
    state_vals = solve_large_state_vals(state_pis, pomdp, batch_size=batch_size)

    state_v, state_q = state_vals

    pi_obs_state_vals, pi_obs_mc_vals, pi_obs_td_vals, info = obs_vals
    pi_obs_state_v, pi_obs_mc_v, pi_obs_td_v = pi_obs_state_vals['v'], pi_obs_mc_vals['v'], pi_obs_td_vals['v']

    # Here we need to copy each obs val into their corresponding state vals
    obs_to_state_mask = pomdp.phi.T > 0
    pi_obs_state_mc_v = jnp.matmul(pi_obs_mc_v, obs_to_state_mask)
    pi_obs_state_td_v = jnp.matmul(pi_obs_td_v, obs_to_state_mask)

    res = {
        'interpolated_obs_pis': obs_pis,
        'interpolated_state_pis': state_pis,
        'state_v': state_v,
        'pi_obs_state_v': pi_obs_state_v,
        'pi_obs_state_mc_v': pi_obs_state_mc_v,
        'pi_obs_state_td_v': pi_obs_state_td_v
    }

    if eval_obs_policy is not None:
        eval_pi_state_vals, eval_pi_mc_vals, eval_pi_td_vals, _ = analytical_pe(eval_obs_policy, pomdp)
        eval_pi_state_v, eval_pi_mc_v, eval_pi_td_v = eval_pi_state_vals['v'], eval_pi_mc_vals['v'], eval_pi_td_vals['v']
        res['eval_pi_state_mc_v'] = jnp.matmul(eval_pi_mc_v, obs_to_state_mask)
        res['eval_pi_state_td_v'] = jnp.matmul(eval_pi_td_v, obs_to_state_mask)

    return res

def get_mem_vals(mem_params: jnp.ndarray, pomdp: POMDP, obs_pis: jnp.ndarray,
                 batch_size: int = 100):
    n_mem = mem_params.shape[-1]
    assert n_mem == 2

    mem_aug_pomdp = memory_cross_product(mem_params, pomdp)

    mem_aug_interpolated_pis = obs_pis.repeat(n_mem, axis=1)  # 0th dimension is the stack

    mem_aug_obs_vals = solve_large_obs_vals(mem_aug_interpolated_pis, mem_aug_pomdp, batch_size=batch_size)

    mem_state_vals, mem_mc_vals, mem_td_vals, info = mem_aug_obs_vals

    # S * M,      O * M,    O * M
    mem_state_v, mem_mc_v, mem_td_v = mem_state_vals['v'], mem_mc_vals['v'], mem_td_vals['v']

    # copy each obs val into their corresponding state vals, split by memory state
    obs_to_state_mask = mem_aug_pomdp.phi.T > 0  # O * M x S * M
    mem_state_mc_v = jnp.matmul(mem_mc_v, obs_to_state_mask)  # S * M
    mem_state_td_v = jnp.matmul(mem_td_v, obs_to_state_mask)  # S * M

    # TODO: pretty easily convert to n_mem
    mem_0_state_mc_v = mem_state_mc_v[::2]
    mem_1_state_mc_v = mem_state_mc_v[1::2]

    mem_0_state_td_v = mem_state_td_v[::2]
    mem_1_state_td_v = mem_state_td_v[1::2]


    # TODO: do we actually want the vals across all pis? Or for the pi that we're optimizing on?
    # TODO: could do both. Just need to add eval_pi and do the same thing. or just pass in 1 x *obs_pi thing.
    return {
        'mem_0_state_mc_v': mem_0_state_mc_v,
        'mem_1_state_mc_v': mem_1_state_mc_v,
        'mem_0_state_td_v': mem_0_state_td_v,
        'mem_1_state_td_v': mem_1_state_td_v,
    }






if __name__ == "__main__":
    batch_size = 100
    bins = 30

    # we load our memory
    mem_opt_path = Path(
        '/Users/ruoyutao/Documents/grl/results/switching_batch_run_seed(2024)_time(20241030-100457)_fdcead0bf0726bb08b24f88e8c72f0b3.npy')

    res = load_info(mem_opt_path)
    all_mem_params = res['logs']['after_mem_op']['ld']['all_mem_params'][0]  # steps x *mem_size
    init_sampled_pi = res['logs']['after_kitchen_sinks']['ld'][0]  # obs x actions

    # repeat initial policy over num mem states
    n_mem = all_mem_params.shape[-1]
    mem_repeated_init_pi = init_sampled_pi.repeat(n_mem, axis=0)

    pomdp, pi_dict = load_pomdp('switching',
                                memory_id=0,
                                n_mem_states=2)

    assert pomdp.action_space.n == 2, "Haven't implemented pi's with action spaces > 2"

    vals = get_value_fns(pomdp, eval_obs_policy=init_sampled_pi, batch_size=batch_size, bins=bins)
    vals_term_removed = jax.tree.map(lambda x: np.array(x[:, :3]), vals)

    # Now we plot our value functions
    plotter = pv.Plotter()

    state_val_cloud = pv.PolyData(vals_term_removed['state_v'])
    plotter.add_mesh(state_val_cloud, color='maroon', point_size=3, opacity=0.025,
                     label='pi(s)')

    pi_obs_state_val_cloud = pv.PolyData(vals_term_removed['pi_obs_state_v'])
    plotter.add_mesh(pi_obs_state_val_cloud, point_size=6, color='blue',
                     label='pi(o)')

    # pi_obs_state_mc_val_cloud = pv.PolyData(vals_term_removed['pi_obs_state_mc_v'])
    # plotter.add_mesh(pi_obs_state_mc_val_cloud, point_size=6, color='orange',
    #                  label='pi(o), V_mc')
    #
    # pi_obs_state_td_val_cloud = pv.PolyData(vals_term_removed['pi_obs_state_td_v'])
    # plotter.add_mesh(pi_obs_state_td_val_cloud, point_size=6, color='cyan',
    #                  label='pi(o), V_td')

    plotter.show_grid(xlabel='start val', ylabel='middle val', zlabel='right val')
    plotter.show()

    print()



