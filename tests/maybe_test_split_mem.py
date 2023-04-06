"""
In this script, we try and figure out how to calculate p(m | o, a)
"""
from jax.config import config
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from grl.environment import load_spec
from grl.memory import memory_cross_product
from grl.mdp import MDP, AbstractMDP
from grl.utils.mdp import amdp_get_occupancy
from grl.utils.policy_eval import lstdq_lambda

from definitions import ROOT_DIR

def load_mem_params(agent_path: Path):
    agent = np.load(agent_path, allow_pickle=True).item()
    return agent.mem_params, agent

def test_split_mem():
    config.update('jax_platform_name', 'cpu')

    spec_name = "tmaze_eps_hyperparams"
    corridor_length = 5
    discount = 0.9
    junction_up_pi = 1.
    epsilon = 0.2
    lambda_0 = 0.
    lambda_1 = 1.
    # mem_funcs = [0, 16]
    mem_funcs = [(19, 0), (16, 1)]
    agent_path = Path(ROOT_DIR, 'results', 'agents', 'tmaze_eps_hyperparams_seed(2026)_time(20230406-131003)_6a75e7a07d0b20088902a5094ede14cc.pkl.npy')
    learnt_mem_params, learnt_agent = load_mem_params(agent_path)

    for mem, target_lambda in mem_funcs:
        spec = load_spec(spec_name,
                         memory_id=str(mem),
                         corridor_length=corridor_length,
                         discount=discount,
                         junction_up_pi=junction_up_pi,
                         epsilon=epsilon)

        mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
        amdp = AbstractMDP(mdp, spec['phi'])

        pi = spec['Pi_phi'][0]
        mem_params = spec['mem_params']
        mem_aug_pi = pi.repeat(mem_params.shape[-1], axis=0)

        mem_aug_amdp = memory_cross_product(mem_params, amdp)
        learnt_mem_aug_amdp = memory_cross_product(learnt_mem_params, amdp)

        n_mem_states = mem_params.shape[-1]

        lstd_v0, lstd_q0, lstd_info = lstdq_lambda(pi, amdp, lambda_=lambda_0)
        lstd_v1, lstd_q1, lstd_info_1 = lstdq_lambda(pi, amdp, lambda_=lambda_1)

        mem_learnt_lstd_v0, mem_learnt_lstd_q0, mem_learnt_lstd_info = lstdq_lambda(mem_aug_pi, learnt_mem_aug_amdp)
        mem_lstd_v0, mem_lstd_q0, mem_lstd_info = lstdq_lambda(mem_aug_pi, mem_aug_amdp, lambda_=lambda_0)
        mem_lstd_v1, mem_lstd_q1, mem_lstd_info_1 = lstdq_lambda(mem_aug_pi, mem_aug_amdp, lambda_=lambda_1)

        # undisc_mdp = MDP(amdp.T, amdp.R, amdp.p0, 1.)
        # undisc_amdp = AbstractMDP(undisc_mdp, amdp.phi)
        # # undisc_amdp = amdp
        #
        # undisc_mem_aug_amdp = memory_cross_product(mem_params, undisc_amdp)

        # counts_mem_aug_flat_obs = amdp_get_occupancy(mem_aug_pi, undisc_mem_aug_amdp) @ undisc_mem_aug_amdp.phi
        counts_mem_aug_flat_obs = mem_lstd_info['occupancy'] @ mem_aug_amdp.phi
        counts_mem_aug_flat = jnp.einsum('i,ij->ij', counts_mem_aug_flat_obs, mem_aug_pi).T  # A x OM

        counts_mem_aug = counts_mem_aug_flat.reshape(amdp.n_actions, -1, 2)  # A x O x M

        denom_counts_mem_aug_unmasked = counts_mem_aug.sum(axis=-1, keepdims=True)
        denom_mask = (denom_counts_mem_aug_unmasked == 0).astype(float)
        denom_counts_mem_aug = denom_counts_mem_aug_unmasked + denom_mask
        prob_mem_given_oa = counts_mem_aug / denom_counts_mem_aug

        # init_obs_action = jnp.expand_dims(amdp.p0 @ amdp.phi, 0).repeat(amdp.n_actions, 0)
        # init_obs_action = init_obs_action + (init_obs_action == 0).astype(float)
        # init_obs_action_mem = jnp.expand_dims(init_obs_action, -1).repeat(n_mem_states, -1)
        # prob_mem_given_oa = prob_mem_given_oa_less_init * init_obs_action_mem

        mem_lstd_q0_unflat = mem_lstd_q0.reshape(amdp.n_actions, -1, mem_params.shape[-1])
        reformed_q0 = (mem_lstd_q0_unflat * prob_mem_given_oa).sum(axis=-1)

        learnt_reformed_q0 = (mem_learnt_lstd_q0.reshape(amdp.n_actions, -1, mem_params.shape[-1]) * prob_mem_given_oa).sum(axis=-1)
        if target_lambda == 1:
            assert jnp.allclose(reformed_q0[:, :-1], lstd_q1[:, :-1])
        elif target_lambda == 0:
            assert jnp.allclose(reformed_q0[:, :-1], lstd_q0[:, :-1])

if __name__ == "__main__":
    test_split_mem()

