"""
Basic DQN (or SARSA) agent.
Based off of David Tao's implementation at https://github.com/taodav/uncertainty/blob/main/unc/agents/dqn.py .
"""
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
from functools import partial
from jax import random, jit, vmap
from optax import sgd
from typing import Tuple
from grl import MDP
from grl.utils.batching import JaxBatch
from grl.mdp import one_hot
from grl.baselines import DQNArgs, mse, create_simple_nn_func
from grl.utils.file_system import numpyify_and_save, load_info
import pickle
from dataclasses import asdict

# Error functions from David's impl
def sarsa_error(q: jnp.ndarray, a: int, r: jnp.ndarray, g: float, q1: jnp.ndarray, next_a: int):
    
    target = r + g * q1[next_a]
    target = jax.lax.stop_gradient(target)
   
    return q[a] - target


def expected_sarsa_error(q: jnp.ndarray, a: int, r: jnp.ndarray, g: float, q1: jnp.ndarray, next_a: int,
                         eps: float = 0.1):
    next_greedy_action = q1.argmax()
    pi = jnp.ones_like(q1) * (eps / q1.shape[-1])
    pi = pi.at[next_greedy_action].add(1 - eps)
    e_q1 = (pi * q1).sum(axis=-1)
    target = r + g * e_q1
    target = jax.lax.stop_gradient(target)
    return q[a] - target


def qlearning_error(q: jnp.ndarray, a: int, r: jnp.ndarray, g: float, q1: jnp.ndarray, *args):
    target = r + g * q1.max()
    target = jax.lax.stop_gradient(target)
    return q[a] - target


class DQNAgent:
    def __init__(self, network: hk.Transformed,
                 args: DQNArgs):
        self.features_shape = args.features_shape
        self.n_actions = args.n_actions
        self.gamma = args.gamma
        self.reward_scale = args.reward_scale


        self._rand_key, network_rand_key = random.split(args.rand_key)
        self.network = network
        self.network_params = self.network.init(rng=network_rand_key, x=jnp.zeros((1, *self.features_shape), dtype=jnp.float32))
        if args.optimizer == "sgd":
            self.optimizer = sgd(args.alpha)
        else:
            raise NotImplementedError(f"Unrecognized optimizer {args.optimizer}")
        self.optimizer_state = self.optimizer.init(self.network_params)
        self.eps = args.epsilon

        self.error_fn = None
        if args.algo == 'sarsa':
            self.error_fn = sarsa_error
        elif args.algo == 'esarsa':
            self.error_fn = expected_sarsa_error
        elif args.algo == 'qlearning':
            self.error_fn = qlearning_error
        self.batch_error_fn = vmap(self.error_fn)

    def save(self, save_path):
        info = {
            'agent_type': 'DQNAgent',
            'args': asdict(self.args),
        }

        save_path.mkdir(exist_ok=True, parents=True)
        info_path = save_path / 'info'
        params_path = save_path / 'params.pkl'
        numpyify_and_save(info_path, info)
        with open(params_path, 'wb') as fp:
            pickle.dump(self.network_params, fp)


    @classmethod
    def load(cls, save_path):
        info_path = save_path / 'info.npy'
        params_path = save_path / 'params.pkl'

        info = load_info(info_path)

        with open(params_path, 'rb') as fp:
            params = pickle.load(fp)
        params = jax.device_put(params)
        transformed = hk.without_apply_rng(hk.transform(create_simple_nn_func(info['args']['n_actions'])))
        agent = cls(transformed, DQNArgs(**info['args']))
        agent.network_params = params
        return agent

    def set_epsilon(self, eps):
        self.eps = eps

    def act(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Get next epsilon-greedy action given a state, using the agent's parameters.
        :param state: (*state.shape) (batch of) state(s) to find actions for.
        """
        action, self._rand_key = self._functional_act(state, self._rand_key, self.network_params)        
        return action
    
    @partial(jit, static_argnums=0)
    def _functional_act(self, state, rand_key, network_params):
       # expand obervation dim to include batch dim
       # TODO maybe we should be doing this in the training loop?
       policy, _ = self._functional_policy(state, network_params)
       key, subkey = random.split(rand_key)
       action = random.choice(subkey, jnp.arange(self.n_actions), p=policy, shape=(state.shape[0],))
       return action, key

    def policy(self, state: jnp.ndarray):
        """
        Get policy from agent's saved parameters at the given state.
        """
        return self._functional_policy(state, self.network_params)

    @partial(jit, static_argnums=0)
    def _functional_policy(self, state: jnp.ndarray, network_params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
        probs = jnp.zeros(self.n_actions) + self.eps / self.n_actions
        greedy_idx, qs = self._greedy_act(state, network_params)
        probs = probs.at[greedy_idx].add(1 - self.eps)
        return probs, qs


    @partial(jit, static_argnums=0)
    def _greedy_act(self, state: jnp.ndarray, network_params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get greedy actions given a state
        :param state: (b x *state.shape) State to find actions
        :param network_params: NN parameters to use to calculate Q.
        :return: (b) Greedy actions
        """
        qs = self.Qs(state, network_params)
        return jnp.argmax(qs, axis=1), qs

    
    def Qs(self, state: jnp.ndarray, network_params: hk.Params) -> jnp.ndarray:
        """
        Get all Q-values given a state.
        :param state: (b x *state.shape) State to find action-values
        :param model: Optional. Potenially use another model
        :return: (b x actions) torch.tensor full of action-values.
        """
        return self.network.apply(network_params, state)

    def Q(self, state: jnp.ndarray, action: jnp.ndarray, network_params: hk.Params) -> jnp.ndarray:
        """
        Get action-values given a state and action
        :param state: (b x *state.shape) State to find action-values
        :param action: (b) Actions for action-values
        :param network_params: NN parameters to use to calculate Q 
        :return: (b) Action-values
        """
        qs = self.Qs(state, network_params)
        return qs[jnp.arange(action.shape[0]), action]
    
    def _loss(self, network_params: hk.Params,
             batch: JaxBatch):
        q_s0 = self.Qs(batch.obs, network_params)
        q_s1 = self.Qs(batch.next_obs, network_params)
    
        effective_gamma = jax.lax.select(self.args.gamma_terminal, 1., self.gamma)
        effective_rewards = batch.rewards * self.reward_scale

        td_err = self.batch_error_fn(q_s0, batch.actions, effective_rewards, jnp.where(batch.terminals, 0., effective_gamma), q_s1, batch.next_actions)
        return mse(td_err)

    @partial(jit, static_argnums=0)
    def functional_update(self,
                          network_params: hk.Params,
                          optimizer_state: hk.State,
                          batch: JaxBatch
                          ) -> Tuple[float, hk.Params, hk.State]:
        loss, grad = jax.value_and_grad(self._loss)(network_params, batch)
        updates, optimizer_state = self.optimizer.update(grad, optimizer_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        return loss, network_params, optimizer_state

    def update(self, 
               batch: JaxBatch
               ) -> float:
        """
        Update given a batch of data
        :param batch: JaxBatch of data to process.
        :return: loss
        """

        loss, self.network_params, self.optimizer_state = \
            self.functional_update(self.network_params, self.optimizer_state, batch)
        return loss

def train_dqn_agent(mdp: MDP,
                    agent: DQNAgent,
                    total_steps: int):
    """
    Training loop for a dqn agent.
    :param mdp: mdp to train on. Currently DQN does not support AMDPs.
    :param agent: DQNAgent to train.
    :param total_steps: Number of steps to train for.
    """
    args = agent.args
    jit_onehot = jax.jit(one_hot, static_argnames=["n"])
    steps = 0
    num_eps = 0
    # Not really batching just updating at each step
    states, actions, next_states, terminals, rewards, next_actions = [], [], [], [], [], []

    # logging
    avg_rewards_ep = []
    avg_rewards = []
    avg_lengths = []
    episode_lengths = []
    losses = []
    pct_success = 0.
    avg_len = 0.
    while (steps < total_steps):
        done = False
        s_0, _ = mdp.reset()
        s_0_processed = np.array([jit_onehot(s_0, mdp.n_obs)])
        a_0 = agent.act(s_0_processed)[0]
        while not done:
            s_1, r_0, done, _, _ = mdp.step(a_0, gamma_terminal=False)

            s_1_processed = np.array([jit_onehot(s_1, mdp.n_obs)])
            a_1 = agent.act(s_1_processed)[0]

            states.append(s_0_processed)
            actions.append(a_0)
            next_states.append(s_1_processed)
            terminals.append(done)
            next_actions.append(a_1)
            rewards.append(r_0)

            batch = JaxBatch(obs=states, 
                             actions=actions, 
                             next_obs=next_states, 
                             terminals=terminals, 
                             rewards=rewards, 
                             next_actions=next_actions)
            
            loss = agent.update(batch)
            s_0_processed = s_1_processed
            a_0 = a_1

            steps = steps + 1
            states, actions, next_states, terminals, rewards, next_actions = [], [], [], [], [], []
               
            avg_rewards_ep.append(np.average(rewards))

            if steps % 1000 == 0:
                # various monitoring metrics
                # TODO should I trim these down?
                rewards_good = [x for x in avg_rewards_ep if x > 0]
                rewards_bad = [x for x in avg_rewards_ep if x < 0]
                pct_success = len(rewards_good) / len(avg_rewards_ep)
                pct_fail = len(rewards_bad) / len(avg_rewards_ep)
                pct_neutral = 1 - pct_success - pct_fail
                avg_rewards.append(np.average(avg_rewards_ep))
                avg_rewards_ep = []
                
            
                avg_len = np.average(np.array(episode_lengths))
                avg_lengths.append(avg_len)
                episode_lengths = []

                losses.append(loss)

                if args.save_path:
                    agent.save(args.save_path / f'ep_{num_eps}')

                print(f"Step {steps} | Episode {num_eps} | Loss {loss}")
                       
                
       
        
        num_eps = num_eps + 1

    
    p, q = agent.policy(np.array([one_hot(s, mdp.n_obs) for s in range(mdp.n_obs)]))

    info = {"final_pct_success": pct_success, 
            "avg_len": avg_lengths, 
            "avg_reward": avg_rewards, 
            "loss": losses, 
            "final_pi": p,
            "final_q": q}

    return info, agent
