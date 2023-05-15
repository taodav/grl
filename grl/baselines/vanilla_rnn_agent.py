"""
Agent that uses a vanilla rnn.
Pretty much copy/pasted from LSTMAgent, could probably use a refactor later.
"""
import jax
# from jax.config import config
# config.update('jax_disable_jit', True)
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
from functools import partial
from jax import random, jit, vmap
from optax import sgd, adam
from typing import Tuple
from grl import AbstractMDP
from grl.utils.batching import JaxBatch
from grl.mdp import one_hot
from grl.baselines.common import DQNArgs, mse, create_managed_vanilla_rnn_func
from grl.baselines.dqn_agent import DQNAgent
from grl.baselines.rnn_agent import seq_sarsa_error
from grl.utils.file_system import numpyify_and_save, load_info
import pickle
from dataclasses import asdict


class VanillaRNNAgent(DQNAgent):
    def __init__(self, network: hk.Transformed, n_hidden: int, 
                 args: DQNArgs):
        self.args = args
        # Constructor similar to DQNAgent except that network needs to be an RNN
        # TODO this is hardcoded 1 in David's impl - should this be an arg?
        self.reward_scale = args.reward_scale
        self.batch_size = 1
        # internalize args
        num_obs = args.features_shape[0]
        self.features_shape = (num_obs * (args.n_actions + 1),)
        self.n_actions = args.n_actions
        self.gamma = args.gamma
        assert args.trunc_len is not None
        self.trunc_len = args.trunc_len # for truncated backprop through time
        self._rand_key, network_rand_key = random.split(args.rand_key)
        self.n_hidden = n_hidden
        self.init_hidden_var = args.init_hidden_var
        self.network = network
        self.reset()
        self.network_params = self.network.init(rng=network_rand_key, 
                                                x=jnp.zeros((self.batch_size, self.trunc_len, *self.features_shape), dtype=jnp.float32),
                                                h=self.hidden_state)
        if args.optimizer == "sgd":
            self.optimizer = sgd(args.alpha)
        elif args.optimizer == "adam":
            self.optimizer = adam(args.alpha)
        else:
            raise NotImplementedError(f"Unrecognized optimizer {args.optimizer}")
        self.optimizer_state = self.optimizer.init(self.network_params)
        self.eps = args.epsilon

        self.error_fn = None
        if args.algo == 'sarsa':
            self.error_fn = seq_sarsa_error
        else:
            raise NotImplementedError(f"Unrecognized learning algorithm {args.algo}")
        self.batch_error_fn = vmap(self.error_fn)

    def save(self, save_path):
        info = {
            'agent_type': 'VanillaRNNAgent',
            'args': asdict(self.args),
            'n_hidden': self.n_hidden
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
        transformed = hk.without_apply_rng(hk.transform(create_managed_vanilla_rnn_func(info['n_hidden'], info['args']['n_actions'])))
        agent = cls(transformed, info['n_hidden'], DQNArgs(**info['args']))
        agent.network_params = params
        return agent


    def get_initial_hidden_state(self):
        """Get the initial state functionally so we can use it in update
            and still retain our current hidden state.
        """
        hs = jnp.zeros([self.batch_size, self.n_hidden])
        if self.init_hidden_var > 0.:
            self._rand_key, keys = random.split(self._rand_key, num=3)
            hs = random.normal(keys[0], shape=[self.batch_size, self.n_hidden]) * self.init_hidden_var
        return hs
    
    def reset(self):
        """
        Reset LSTM hidden states.
        :return:
        """
        self.hidden_state = self.get_initial_hidden_state()

        
    def act(self, obs: jnp.ndarray, last_a: jnp.ndarray) -> jnp.ndarray:
        """
        Get next epsilon-greedy action given a obs and last action, using the agent's parameters.
        :param obs: (batch x time_steps x obs_size) obs to find actions for. 
        """        
        action, self.hidden_state, self._rand_key = self._functional_act(obs, last_a, self.hidden_state, self._rand_key, self.network_params, self.eps)        
        
        return action
    
    @partial(jit, static_argnums=0)
    def _functional_act(self, obs, last_a, h, rand_key, network_params, epsilon):
       policy, _ , new_hidden = self._functional_policy(obs, last_a, h, network_params, epsilon)
       key, subkey = random.split(rand_key)
      
       action = jax.random.categorical(subkey, jnp.log(policy), axis=-1)
       return action, new_hidden, key

    def policy(self, obs: jnp.ndarray, last_a: jnp.ndarray, h: jnp.ndarray):
        """
        Get policy from agent's saved parameters at the given obs.
        :param obs: (batch x time_steps x obs_size) obs(s) to find actions for.
        """
        return self._functional_policy(obs, last_a, h, self.network_params, self.eps)

    @partial(jit, static_argnums=0)
    def _functional_policy(self, obs: jnp.ndarray, last_a: jnp.ndarray, h: jnp.ndarray, network_params: hk.Params, epsilon: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        probs = jnp.zeros((*obs.shape[:-1], self.n_actions)) + epsilon / self.n_actions
        greedy_idx, qs, new_h = self._greedy_act(obs, last_a, h, network_params)
        # TODO I genuinely don't know a better/more efficient way to do this ATM - suggestions very welcome!
        for batch in range(probs.shape[0]):
            for ts in range(probs.shape[1]):
                probs = probs.at[(batch, ts, greedy_idx[batch][ts])].add(1 - epsilon)
        return probs, qs, new_h


    @partial(jit, static_argnums=0)
    def _greedy_act(self, obs: jnp.ndarray, last_a: jnp.ndarray, h: hk.LSTMState, network_params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get greedy actions given a obs
        :param obs: (b x time_steps x *obs.shape) obs to find actions
        :param network_params: NN parameters to use to calculate Q.
        :return: (b x time_steps) Greedy actions
        """
        qs, new_h = self.Qs(obs, last_a, h, network_params)
        return jnp.argmax(qs, -1), qs, new_h

    
    @partial(jit, static_argnums=0)
    def Qs(self, obs: jnp.ndarray, last_a: jnp.ndarray, h: hk.LSTMState, network_params: hk.Params) -> jnp.ndarray:
        """
        Get all Q-values given a obs.
        :param obs: (b x time_steps x *obs.shape) obs to find action-values
        :param model: Optional. Potenially use another model
        :return: (b x time_steps x actions) torch.tensor full of action-values.
        """
        features_2d_shape = (*self.args.features_shape, self.n_actions + 1)
        selected = jnp.zeros((obs.shape[0], obs.shape[1], self.features_shape[0]), dtype=jnp.float32)


        # TODO non for-loop way to do this?
        for b in range(selected.shape[0]):
            indices = jnp.ravel_multi_index((jnp.argmax(obs[b]), last_a[b]), features_2d_shape, mode='clip')
            selected = selected.at[b, jnp.arange(selected.shape[1]), indices].set(1.)

        res = self.network.apply(network_params, selected, h)
        return res['td0'], res['h']

    def Q(self, obs: jnp.ndarray, last_action: jnp.ndarray, action: jnp.ndarray, h: hk.LSTMState, network_params: hk.Params) -> jnp.ndarray:
        """
        Get action-values given a obs and action
        :param obs: (b x time_steps x *obs.shape) obs to find action-values
        :param action: (b x time_steps) Actions for action-values
        :param network_params: NN parameters to use to calculate Q 
        :return: (b x time_steps) Action-values
        """
        qs, _ = self.Qs(obs, last_action, h, network_params)
        return qs[jnp.arange(action.shape[0]), action]
    
    def _loss(self, network_params: hk.Params,
             initial_hidden: hk.LSTMState,
             batch: JaxBatch):
        #(B x T x A)
        q_all, _ = self.Qs(batch.all_obs, batch.actions, initial_hidden, network_params)
        q_s0 = q_all[:, :-1, :]
        q_s1 = q_all[:, 1:, :]

        effective_gamma = jax.lax.select(self.args.gamma_terminal, 1., self.gamma)
        effective_rewards = batch.rewards * self.reward_scale

        td_err = self.batch_error_fn(q_s0, batch.actions[:, 1:], effective_rewards, jnp.where(batch.terminals, 0., effective_gamma), q_s1, batch.next_actions[:, 1:])
        return mse(td_err)

    @partial(jit, static_argnums=0)
    def functional_update(self,
                          network_params: hk.Params,
                          optimizer_state: hk.State,
                          hidden_state: hk.LSTMState,
                          batch: JaxBatch
                          ) -> Tuple[float, hk.Params, hk.State]:
        loss, grad = jax.value_and_grad(self._loss)(network_params, hidden_state, batch)
        updates, optimizer_state = self.optimizer.update(grad, optimizer_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        return loss, network_params, optimizer_state

    def update(self, 
               batch: JaxBatch
               ) -> float:
        """
        Update given a batch of data, additionally resetting the LSTM state.
        :param batch: JaxBatch of data to process.
        :return: loss
        """
        loss, self.network_params, self.optimizer_state = \
            self.functional_update(self.network_params, self.optimizer_state, self.get_initial_hidden_state(), batch)
        return loss
    
def train_vanilla_rnn_agent(amdp: AbstractMDP,
                    agent: VanillaRNNAgent,
                    total_eps: int):
    """
    Training loop for an rnn agent.
    :param amdp: AbstractMDP to train on.
    :param agent: RNN agent to train.
    :param total_eps: Number of episodes to train for.
    """

    steps = 0
    num_eps = 0
    args = agent.args
    epsilon_final = args.epsilon
    anneal_steps = args.anneal_steps
    epsilon_start = args.epsilon_start
    anneal_value = (epsilon_start - epsilon_final) / anneal_steps if anneal_steps > 0 else 0

    if epsilon_start != epsilon_final:
        agent.set_epsilon(epsilon_start)


    jit_onehot = jax.jit(one_hot, static_argnames=["n"])
    
    avg_rewards = []
    avg_rewards_ep = []
    avg_lengths = []
    episode_lengths = []
    losses = []
    pct_success = 0.
    avg_len = 0.
    while (num_eps <= total_eps):
        # episode buffers
        all_obs, all_actions, terminals, rewards = [], [], [], []
        agent.reset()
        done = False
        
        o_0, _ = amdp.reset()
        o_0_processed = jit_onehot(o_0, amdp.n_obs)
        all_obs.append(o_0_processed)
        # dummy action for initial state
        all_actions.append(amdp.n_actions)
        
        # need to wrap in batch dimension
        a_0 = agent.act(np.array([[o_0_processed]]), np.array([[amdp.n_actions]]))[-1][-1]
        all_actions.append(a_0)
        
        # TODO no truncation length
        # For PR: is this functionality that we want to make optional?
        #for _ in range(agent.trunc_len):
        while not done:
            o_1, r_0, done, _, _ = amdp.step(a_0, gamma_terminal=args.gamma_terminal)
            terminals.append(done)
            rewards.append(r_0)
            steps = steps + 1  
            
            o_1_processed = jit_onehot(o_1, amdp.n_obs)
            all_obs.append(o_1_processed)

            # need to wrap in batch dimension
            a_1 = agent.act(np.array([[o_1_processed]]), np.array([[a_0]]))[-1][-1]
            all_actions.append(a_1)
            
            # if done:
            #     #print(f"Broke early after {t} steps")
            #     break
            
            
            
            o_0_processed = o_1_processed
            a_0 = a_1
            
       
        batch = JaxBatch(all_obs = [all_obs],
                         obs=[all_obs[:-1]], 
                             actions=[all_actions[:-1]], 
                             next_obs=[all_obs[1:]], 
                             terminals=[terminals], 
                             rewards=[rewards], 
                             next_actions=[all_actions[1:]])

       
        loss = agent.update(batch)
        episode_lengths.append(len(rewards))
        avg_rewards_ep.append(np.average(rewards))
           
        if num_eps % 1000 == 0:
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
            print(f"Step {steps} | Episode {num_eps} | Epsilon {agent.eps} | Loss {loss} | Avg Length {avg_len} | Reward {batch.rewards} | Success/Fail/Neutral {pct_success}/{pct_fail}/{pct_neutral} | Obs {batch.obs} | Q-vals {agent.Qs(batch.obs, batch.actions[:, :-1], agent.get_initial_hidden_state(), agent.network_params)[0]}")
        
        num_eps = num_eps + 1
       
        # Anneal Epsilon
        if anneal_steps > 0 and anneal_steps > num_eps:
            agent.set_epsilon(epsilon_start - anneal_value * num_eps)

    agent.reset()
    final_policy, final_q, _ = agent.policy(batch.obs, batch.actions[:, :-1], agent.get_initial_hidden_state())
    info = {"final_pct_success": pct_success, 
            "avg_len": avg_lengths, 
            "avg_reward": avg_rewards, 
            "loss": losses, 
            "final_pi": final_policy,
            "final_q": final_q}
    return info, agent
