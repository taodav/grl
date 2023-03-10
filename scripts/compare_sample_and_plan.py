import numpy as np
from jax.config import config
from pathlib import Path
from functools import partial

np.set_printoptions(precision=3, suppress=True)
config.update('jax_platform_name', 'cpu')

from grl.agents.actorcritic import ActorCritic
from grl.environment import load_spec
from grl.mdp import MDP, AbstractMDP
from grl.utils.policy_eval import analytical_pe
from grl.memory import memory_cross_product
from grl.utils.loss import discrep_loss
from grl.utils.lambda_discrep import lambda_discrep_measures

from definitions import ROOT_DIR
from learning_agent.memory_iteration import converge_value_functions

if __name__ == "__main__":
    spec_name = "tmaze_eps_hyperparams"
    # spec_name = "simple_chain"
    corridor_length = 5
    discount = 0.9
    junction_up_pi = 0.
    epsilon = 0.0
    discrep_type = 'mse'
    error_type = 'l2' if discrep_type == 'mse' else discrep_type
    mem_id = 0
    uniform_weights = True

    # Sampling agent hyperparams
    buffer_size = int(4e6)
    # buffer_size = int(1e3)

    buffer_dir = Path(ROOT_DIR, 'scripts', 'results', 'sample_based')
    fname = f'replaymemory_corridor({corridor_length})_eps({epsilon})_size({buffer_size})_mem({mem_id})'
    agent_fname = f'agent_corridor({corridor_length})_eps({epsilon})_size({buffer_size})_mem({mem_id})'
    ext = '.pkl'
    agent_full_path = buffer_dir / (agent_fname + ext)

    print(f"Running sample-based and analytical comparison for Q-values on {spec_name}")
    spec = load_spec(spec_name,
                     memory_id=str(mem_id),
                     corridor_length=corridor_length,
                     discount=discount,
                     junction_up_pi=junction_up_pi,
                     epsilon=epsilon)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    env = AbstractMDP(mdp, spec['phi'])
    pi = spec['Pi_phi'][0]

    agent = ActorCritic(
        n_obs=env.n_obs,
        n_actions=env.n_actions,
        gamma=env.gamma,
        n_mem_entries=0,
        replay_buffer_size=buffer_size,
        discrep_loss=discrep_type,
        disable_importance_sampling=not uniform_weights,
    )

    # instantiate analytical lambda-discrepancy calculating function
    discrep_loss_fn = partial(discrep_loss, value_type='q', error_type=error_type, alpha=int(uniform_weights))
    get_measures = partial(lambda_discrep_measures, discrep_loss_fn=discrep_loss_fn)

    agent.set_policy(pi, logits=False)

    agent.add_memory()
    # agent.reset_memory()
    # mem_params = agent.memory_logits
    mem_params = spec['mem_params']
    agent.set_memory(mem_params, logits=True)
    mem_aug_mdp = memory_cross_product(mem_params, env)

    mem_pi = pi.repeat(2, axis=0)
    # analytical_state_vals, analytical_mc_vals, analytical_td_vals, info = analytical_pe(mem_pi, mem_aug_mdp)
    # analytical_state_vals, analytical_mc_vals, analytical_td_vals, info = analytical_pe(pi, env)

    if agent_full_path.is_file():
        agent = ActorCritic.load(agent_full_path)
    else:
        converge_value_functions(agent, env, update_policy=False)
        agent.save(buffer_dir / (agent_fname + ext))

    analytical_discrep, analytical_mc_vals, analytical_td_vals = discrep_loss(mem_pi, mem_aug_mdp)
    sampled_based_discrep = agent.evaluate_memory()

    print("Analytical Q-TD values:")
    print(analytical_td_vals['q'])

    print("Sample-based Q-TD values:")
    print(agent.q_td.q)

    print("Analytical Q-MC values:")
    print(analytical_mc_vals['q'])

    print("Sample-based Q-MC values:")
    print(agent.q_mc.q)

    print("Analytical lambda-discrepancies:")
    print(analytical_discrep)

    print("Sample-based lambda-discrepancy:")
    print(sampled_based_discrep)

    print("done")

    print(f"Saving to {buffer_dir / (fname + ext)}")
    agent.replay.save(buffer_dir, filename=fname, extension=ext)
