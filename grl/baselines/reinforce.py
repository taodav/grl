"""
Agent that utilizes REINFORCE with a
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
from grl.baselines import DQNArgs, create_managed_lstm_func
from grl.utils.file_system import numpyify_and_save, load_info
import pickle
from dataclasses import asdict


def reinforce_error(a_t, r_t, logits, gamma):
    """
    REINFORCE error function.
    :param r_t: rewards at each step over entire trajectory, used to calculate return, (T x 1)
    :param a_t: actions at each step over trajectory, (T x 1)
    :param logits: output of policy preference network at each timestep. (T x num_actions)
    :param gamma: discount factor (scalar)
    :return error at each timestep (T x 1)
    """
    logprobs = jnp.log(jax.nn.softmax(logits))
    p_a_t = logprobs[jnp.arange(logprobs.shape[0]), a_t]
    
    returns = []
    for t in range(len(r_t)):
        g_t = jnp.sum((gamma ** jnp.arange(len(r_t) - t)) * r_t[t:])
        returns.append(g_t)
    
    returns = jax.lax.stop_gradient(jnp.array(returns))

    return returns * p_a_t



class LSTMReinforceAgent():
    def __init__(self, network: hk.Transformed, n_hidden: int, 
                 args: DQNArgs):
        self.args = args
        # TODO this is hardcoded 1 in David's impl - should this be an arg?
        self.batch_size = 1
        # internalize args
        self.features_shape = args.features_shape
        self.n_actions = args.n_actions
        self.reward_scale = args.reward_scale
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

        self.error_fn = None
        if args.algo == 'reinforce':
            self.error_fn = reinforce_error
        else:
            raise NotImplementedError(f"Unrecognized learning algorithm {args.algo}")
        self.batch_error_fn = vmap(self.error_fn)
    
    def save(self, save_path):
        info = {
            'agent_type': 'LSTMReinforceAgent',
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
        transformed = hk.without_apply_rng(hk.transform(create_managed_lstm_func(info['n_hidden'], info['args']['n_actions'])))
        agent = cls(transformed, info['n_hidden'], DQNArgs(**info['args']))
        agent.network_params = params
        return agent


    def get_initial_hidden_state(self):
        """Get the initial state functionally so we can use it in update
            and still retain our current hidden state.
        """
        hs = jnp.zeros([self.batch_size, self.n_hidden])
        cs = jnp.zeros([self.batch_size, self.n_hidden])
        if self.init_hidden_var > 0.:
            self._rand_key, keys = random.split(self._rand_key, num=3)
            hs = random.normal(keys[0], shape=[self.batch_size, self.n_hidden]) * self.init_hidden_var
            cs = random.normal(keys[1], shape=[self.batch_size, self.n_hidden]) * self.init_hidden_var
        return hk.LSTMState(hidden=hs, cell=cs)

    def reset(self):
        """
        Reset LSTM hidden states and internal logit buffer.
        :return:
        """
        self.hidden_state = self.get_initial_hidden_state()

    def act(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Get next action sampled from stochastic policy from network, as well as policy logits.
        :param obs: (batch x time_steps x obs_size) obs to find actions for. 
        :return: tuple (action, logits): (batch x time_steps) actions, (batch x time_steps x num_actions) policy logits.
        """        
        action, self.hidden_state, self._rand_key = self._functional_act(obs, 
                                                                                 self.hidden_state, 
                                                                                 self._rand_key, 
                                                                                 self.network_params)
        
        return action
    
    @partial(jit, static_argnums=0)
    def _functional_act(self, obs, h, rand_key, network_params):
       policy_probs, new_hidden = self.policy(network_params, h, obs)
       key, subkey = random.split(rand_key)
      
       action = jax.random.categorical(subkey, jnp.log(policy_probs), axis=-1)
       return action, new_hidden, key

    @partial(jit, static_argnums=0)
    def policy(self, network_params: hk.Params, h: hk.LSTMState, obs: jnp.ndarray):
        """
        Get policy given a obs.
        :param network_params: parameters of network to query for policy
        :param h: Hidden state of model to use for query
        :param obs: (b x time_steps x *obs.shape) obs to find policy
        :return: (b x time_steps x actions) torch.tensor full of policies.
        """
        logits, new_h = self.policy_logits(network_params, h, obs)
        return jax.nn.softmax(logits, axis=-1), new_h
        

    def policy_logits(self, network_params: hk.Params, h: hk.LSTMState, obs: jnp.ndarray):
        """
        Get policy preferences (logits) given a obs.
        :param network_params: parameters of network to query for policy
        :param h: Hidden state of model to use for query
        :param obs: (b x time_steps x *obs.shape) obs to find policy
        :return: Tuple: (b x time_steps x actions) torch.tensor full of policy preference logits, cell state.
        """
        return self.network.apply(network_params, obs, h)
        
    def _loss(self, network_params: hk.Params,
             initial_hidden: hk.LSTMState,
             batch: JaxBatch):
        #(B x T x A)
        obs_policy_logits, _ = self.policy_logits(network_params, initial_hidden, batch.obs)
    
        effective_gamma = jax.lax.select(self.args.gamma_terminal, 1., self.gamma)
        effective_rewards = batch.rewards * self.reward_scale

        returns_scaled = self.batch_error_fn(batch.actions, effective_rewards, obs_policy_logits, jnp.full((batch.obs.shape[0],), effective_gamma))
        return -jnp.sum(returns_scaled)

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

    
def train_reinforce_agent(amdp: AbstractMDP,
                    agent: LSTMReinforceAgent,
                    total_eps: int):
    """
    Training loop for an rnn agent.
    :param amdp: AbstractMDP to train on.
    :param agent: REINFORCE agent to train.
    :param total_eps: Number of episodes to train for.
    """

    steps = 0
    num_eps = 0
    args = agent.args

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
        
        # TODO no truncation length
        # For PR: is this functionality that we want to make optional?
        #for _ in range(agent.trunc_len):
        while not done:
            # need to wrap in batch dimension
            a_0 = agent.act(np.array([[o_0_processed]]))[-1][-1]
            all_actions.append(a_0)
            o_0, r_0, done, _, _ = amdp.step(a_0, gamma_terminal=args.gamma_terminal)
            terminals.append(done)
            rewards.append(r_0)
            steps = steps + 1  
            
            o_0_processed = jit_onehot(o_0, amdp.n_obs)
            all_obs.append(o_0_processed)
            
       
        batch = JaxBatch(all_obs = [all_obs],
                         obs=[all_obs[:-1]], 
                             actions=[all_actions], 
                             terminals=[terminals], 
                             rewards=[rewards])

       
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
            print(f"Step {steps} | Episode {num_eps} | Loss {loss} | Avg Length {avg_len} | Reward {batch.rewards} | Success/Fail/Neutral {pct_success}/{pct_fail}/{pct_neutral} | Obs {batch.obs} | Policy {agent.policy(agent.network_params, agent.get_initial_hidden_state(), batch.obs)[0]}")
        
        num_eps = num_eps + 1
       
    agent.reset()
    final_policy = agent.policy(agent.network_params, agent.get_initial_hidden_state(), batch.obs)[0]
    info = {"final_pct_success": pct_success, 
            "avg_len": avg_lengths, 
            "avg_reward": avg_rewards, 
            "loss": losses, 
            "final_pi": final_policy}
    return info, agent

