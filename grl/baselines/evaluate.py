from argparse import ArgumentParser
import dill
from grl import MDP, AbstractMDP, environment
from grl.mdp import one_hot
import jax
import numpy as np
from pathlib import Path
from definitions import ROOT_DIR
from grl.baselines import ManagedLSTM, DQNArgs



class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def evaluate_agent(agent, amdp, num_episodes, gamma_terminal):
    
    jit_onehot = jax.jit(one_hot, static_argnames=["n"])
    eps = 0
    all_returns = np.array([])
    while (eps < num_episodes):
        episode_rewards = []
        agent.reset()
        done = False
        
        curr_ob, _ = amdp.reset()
        curr_ob_processed = jit_onehot(curr_ob, amdp.n_obs)
        
        
    
        while not done:
            # need to wrap in batch dimension
            action = agent.act(np.array([[curr_ob_processed]]))[-1][-1]
            curr_ob, reward, done, _, _ = amdp.step(action, gamma_terminal=gamma_terminal)
            curr_ob_processed = jit_onehot(curr_ob, amdp.n_obs)
            episode_rewards.append(reward)

        # discount the rewards
        gamma_coefs = amdp.gamma**np.arange(len(episode_rewards))
        all_returns = np.append(all_returns, np.sum(np.array(episode_rewards) * gamma_coefs))

        eps += 1

    return np.average(all_returns)
        
        

    
    

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--agent", required=True, 
                   help="(relative from results directory) path to agent to evaluate")
    p.add_argument("--spec", required=True,
                   help="Environment to evaluate on, e.g. 'cheese.95'")
    p.add_argument("--num_episodes", default=1000,
                   help="Number of episodes to perform evaluation on. Script reports average discounted return across episodes.")
    p.add_argument("--gamma_terminal", action='store_true',
                   help="Terminate episodes with probability (1-gamma) at each step?")
    
    args = p.parse_args()
    
    results_dir = Path(ROOT_DIR, 'results')
    # load the environment
    spec = environment.load_spec("cheese.95")
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    pomdp = AbstractMDP(mdp, spec['phi'])
    # try to load the agent
    with open(results_dir / args.agent, 'rb') as dill_file:
        agent = dill.load(dill_file)

    args.__dict__.update({'hidden_size': agent.n_hidden})
    
    agent.epsilon = 0.
    # TODO just printing rn?
    print(evaluate_agent(agent, pomdp, args.num_episodes, args.gamma_terminal))
