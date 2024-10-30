from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm

from grl.environment import load_pomdp
from grl.utils.policy_eval import analytical_pe, functional_solve_mdp


def get_value_fns(pomdp, batch_size: int = 100, bins: int = 20):

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

    pi_obs_state_vals, pi_obs_mc_vals, pi_obs_td_vals, info = obs_vals
    pi_obs_state_v, pi_obs_mc_v, pi_obs_td_v = pi_obs_state_vals['v'], pi_obs_mc_vals['v'], pi_obs_td_vals['v']

    # TODO: here we need to copy each obs val into their corresponding state vals

    return obs_vals, state_vals


if __name__ == "__main__":
    batch_size = 100
    bins = 20

    pomdp, pi_dict = load_pomdp('switching',
                                memory_id=0,
                                n_mem_states=2)

    assert pomdp.action_space.n == 2, "Haven't implemented pi's with action spaces > 2"

    obs_vals, state_vals = get_value_fns(pomdp, batch_size=batch_size, bins=bins)

    pi_obs_state_vals, pi_obs_mc_vals, pi_obs_td_vals, info = obs_vals
    pi_obs_state_v, pi_obs_mc_v, pi_obs_td_v = pi_obs_state_vals['v'], pi_obs_mc_vals['v'], pi_obs_td_vals['v']

    state_v, state_q = state_vals

    print()



