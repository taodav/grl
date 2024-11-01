from pathlib import Path

import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import numpy as np
import pyvista as pv

from grl.environment import load_pomdp
from grl.utils.file_system import load_info
from grl.utils.policy_eval import analytical_pe, functional_solve_mdp


def get_value_fns(pomdp, eval_obs_policy: jnp.ndarray = None, batch_size: int = 100, bins: int = 20):

    def get_interpolated_pis(n_obs: int):
        prob_vals = jnp.linspace(0, 1, num=bins)

        # Create n copies of the array
        grids = jnp.meshgrid(*[prob_vals] * n_obs, indexing='ij')
        # Stack and reshape to get the Cartesian product
        action_0_probs = jnp.stack(grids, axis=-1).reshape(-1, n_obs)
        action_1_probs = 1 - action_0_probs

        return jnp.stack([action_0_probs, action_1_probs], axis=-1)

    jitted_interpolate = jax.jit(get_interpolated_pis, static_argnames=['n_obs'])

    obs_pis = jitted_interpolate(pomdp.observation_space.n)
    state_pis = jitted_interpolate(pomdp.state_space.n)

    # make batches
    n_rest_obs = obs_pis.shape[0] // batch_size
    batch_obs_pis = obs_pis.reshape((n_rest_obs, batch_size, *obs_pis.shape[1:]))

    n_rest_state = state_pis.shape[0] // batch_size
    batch_state_pis = state_pis.reshape((n_rest_state, batch_size, *state_pis.shape[1:]))

    vmap_analytical_pe = jax.vmap(analytical_pe, in_axes=[0, None])
    vmap_functional_solve_mdp = jax.vmap(functional_solve_mdp, in_axes=[0, None])

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

    state_v, state_q = state_vals

    pi_obs_state_vals, pi_obs_mc_vals, pi_obs_td_vals, info = obs_vals
    pi_obs_state_v, pi_obs_mc_v, pi_obs_td_v = pi_obs_state_vals['v'], pi_obs_mc_vals['v'], pi_obs_td_vals['v']

    # Here we need to copy each obs val into their corresponding state vals
    obs_to_state_mask = pomdp.phi.T > 0
    pi_obs_state_mc_v = jnp.matmul(pi_obs_mc_v, obs_to_state_mask)
    pi_obs_state_td_v = jnp.matmul(pi_obs_td_v, obs_to_state_mask)

    res = {
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



