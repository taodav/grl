from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import numpy as np

from grl.environment import load_pomdp
from grl.memory import memory_cross_product
from grl.utils.file_system import load_info, numpyify_and_save
from grl.utils.mdp_solver import POMDP
from grl.utils.policy_eval import analytical_pe, functional_solve_mdp


@partial(jax.jit, static_argnames=['n_obs', 'bins'])
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
    """
    Solve for the observation value functions for a large number of policies over observations obs_pi.
    :param obs_pis: (n_pis, n_obs, n_actions)
    """
    vmap_analytical_pe = jax.vmap(analytical_pe, in_axes=[0, None])

    # make batches
    n_rest_obs = obs_pis.shape[0] // batch_size
    batch_obs_pis = obs_pis.reshape((n_rest_obs, batch_size, *obs_pis.shape[1:]))

    # @scan_tqdm(n_rest_obs)
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
    """
    Solve for the state value functions for a large number of policies over state, state_pis.
    :param obs_pis: (n_pis, n_obs, n_actions)
    """
    n_rest_state = state_pis.shape[0] // batch_size
    batch_state_pis = state_pis.reshape((n_rest_state, batch_size, *state_pis.shape[1:]))

    vmap_functional_solve_mdp = jax.vmap(functional_solve_mdp, in_axes=[0, None])

    # @scan_tqdm(n_rest_state)
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

    return res, { 'interpolated_obs_pis': obs_pis, 'interpolated_state_pis': state_pis }


def get_mem_vals(mem_params: jnp.ndarray, pomdp: POMDP,
                 fixed_pi_params: jnp.ndarray = None,
                 batch_size: int = 128, bins: int = 8):
    n_mem = mem_params.shape[-1]
    assert n_mem == 2

    mem_aug_pomdp = memory_cross_product(mem_params, pomdp)

    mem_aug_interpolated_pis = get_interpolated_pis(mem_aug_pomdp.phi.shape[-1], bins=bins)  # 0th dimension is the stack

    mem_aug_obs_vals = solve_large_obs_vals(mem_aug_interpolated_pis, mem_aug_pomdp, batch_size=batch_size)

    mem_state_vals, mem_mc_vals, mem_td_vals, info = mem_aug_obs_vals

    # S * M,      O * M,    O * M
    mem_state_v, mem_mc_v, mem_td_v = mem_state_vals['v'], mem_mc_vals['v'], mem_td_vals['v']

    # copy each obs val into their corresponding state vals, split by memory state
    obs_to_state_mask = mem_aug_pomdp.phi.T > 0  # O * M x S * M
    mem_state_mc_v = jnp.matmul(mem_mc_v, obs_to_state_mask)  # S * M
    mem_state_td_v = jnp.matmul(mem_td_v, obs_to_state_mask)  # S * M

    next_mem_action_vals = mem_aug_pomdp.gamma * jnp.einsum('ijk,lk->ijl', mem_aug_pomdp.T, mem_state_td_v)
    R_sa = (mem_aug_pomdp.T * mem_aug_pomdp.R).sum(axis=-1)
    pi_state = jnp.einsum('ij,kjl->kil', mem_aug_pomdp.phi, mem_aug_interpolated_pis)

    one_step_mem_state_td_q = R_sa[..., None] + next_mem_action_vals
    one_step_mem_state_td_v = (one_step_mem_state_td_q.T * pi_state).sum(axis=-1)

    # Separate out memory states
    mem_0_state_mc_v = mem_state_mc_v[..., ::2]
    mem_1_state_mc_v = mem_state_mc_v[..., 1::2]

    mem_0_state_td_v = mem_state_td_v[..., ::2]
    mem_1_state_td_v = mem_state_td_v[..., 1::2]
    one_step_mem_0_state_td_v = one_step_mem_state_td_v[..., ::2]
    one_step_mem_1_state_td_v = one_step_mem_state_td_v[..., 1::2]
    res = {
        'mem_0_state_mc_v': mem_0_state_mc_v,
        'mem_1_state_mc_v': mem_1_state_mc_v,
        'mem_0_state_td_v': mem_0_state_td_v,
        'mem_1_state_td_v': mem_1_state_td_v,
        'one_step_mem_0_state_td_v': one_step_mem_0_state_td_v,
        'one_step_mem_1_state_td_v': one_step_mem_1_state_td_v,
    }
    if fixed_pi_params is not None:
        mem_aug_fixed_params = fixed_pi_params.repeat(n_mem, axis=0)
        mem_state_fixed_pi, mem_mc_fixed_pi, mem_td_fixed_pi, _ = analytical_pe(mem_aug_fixed_params, mem_aug_pomdp)
        mem_state_mc_v_fixed_pi = jnp.matmul(mem_mc_fixed_pi['v'], obs_to_state_mask)  # S * M
        mem_state_td_v_fixed_pi = jnp.matmul(mem_td_fixed_pi['v'], obs_to_state_mask)  # S * M

        res['mem_0_state_mc_v_fixed_pi'] = mem_state_mc_v_fixed_pi[::2][None, ...]
        res['mem_1_state_mc_v_fixed_pi'] = mem_state_mc_v_fixed_pi[1::2][None, ...]
        res['mem_0_state_td_v_fixed_pi'] = mem_state_td_v_fixed_pi[::2][None, ...]
        res['mem_1_state_td_v_fixed_pi'] = mem_state_td_v_fixed_pi[1::2][None, ...]

    return res


if __name__ == "__main__":
    batch_size = 100
    bins = 30

    # we load our memory

    objective = 'ld'
    mem_opt_path = Path(
        '/Users/ruoyutao/Documents/grl/results/switching/switching_ld_seed(2025)_time(20241115-150703)_b2127fcb071bd63045d31fbabfc504fd.npy'
    )

    # objective = 'tde'
    # mem_opt_path = Path(
    #     '/Users/ruoyutao/Documents/grl/results/switching/switching_tde_seed(2025)_time(20241114-095800)_6c8ac066c28f4be038a74aa65d6b4576.npy'
    # )

    # objective = 'mem_state_discrep'
    # mem_opt_path = Path(
    #     '/Users/ruoyutao/Documents/grl/results/switching/switching_mem_state_discrep_seed(2025)_time(20241115-130318)_248e60554f80fcb003d25500b68d7a30.npy'
    # )

    res = load_info(mem_opt_path)

    init_sampled_pi = jax.nn.softmax(res['logs']['after_kitchen_sinks'][objective][0], axis=-1)  # obs x actions

    all_mem_params = res['logs']['after_mem_op']['all_mem_params'][0]  # steps x *mem_size

    pomdp, pi_dict = load_pomdp('switching',
                                memory_id=0,
                                n_mem_states=2)

    if objective == 'mem_state_discrep':
        init_sampled_pi = init_sampled_pi[::2]

    assert pomdp.action_space.n == 2, "Haven't implemented pi's with action spaces > 2"

    vals, info = get_value_fns(pomdp, eval_obs_policy=init_sampled_pi, batch_size=batch_size, bins=bins)
    vals_term_removed = jax.tree.map(lambda x: np.array(x[..., :3]), vals)


    @scan_tqdm(all_mem_params.shape[0])
    def mem_vals_scan_wrapper(_, inp):
        i, mem_params = inp
        return _, get_mem_vals(mem_params, pomdp, fixed_pi_params=init_sampled_pi)


    _, mem_vals = jax.lax.scan(
        mem_vals_scan_wrapper, None, (jnp.arange(all_mem_params.shape[0]), all_mem_params), all_mem_params.shape[0]
    )
    # vmapped_get_mem_vals = jax.vmap(get_mem_vals, in_axes=[0, None])
    # mem_vals = vmapped_get_mem_vals(all_mem_params, pomdp)

    mem_vals_term_removed = jax.tree.map(lambda x: np.array(x[..., :3]), mem_vals)

    res = {
        'objective': objective,
        'parent_path': str(mem_opt_path),
        'all_mem_params': all_mem_params,
        'values': vals,
        'values_terminal_removed': vals_term_removed,
        'memory_values': mem_vals,
        'memory_values_terminal_removed': mem_vals_term_removed
    }

    res_path = mem_opt_path.parent / (mem_opt_path.stem + '_polytope.npy')

    numpyify_and_save(res_path, res)
    print(f"Saved value polytope info to {res_path}")

