"""
SARSA agent that utilizes an RNN.
Relies on haiku's recurrent API (https://dm-haiku.readthedocs.io/en/latest/api.html#recurrent).
Also based on David Tao's implementation at https://github.com/taodav/uncertainty/blob/main/unc/agents/lstm.py .
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
from optax import sgd
from typing import Tuple
from grl import MDP
from grl.utils.batching import JaxBatch
from grl.mdp import one_hot
from .dqn_agent import DQNAgent, DQNArgs, mse

# error func from David's impl
def seq_sarsa_error(q: jnp.ndarray, a: jnp.ndarray, r: jnp.ndarray, g: jnp.ndarray, q1: jnp.ndarray, next_a: jnp.ndarray):
    """
    sequential version of sarsa loss
    First axis of all tensors are the sequence length.
    :return:
    """
    # print(a.shape)
    target = r + g * q1[jnp.arange(next_a.shape[0]), next_a]
    # print('r', r)
    # print('g', g)
    # print('target', target)
    target = jax.lax.stop_gradient(target)
    q_vals = q[jnp.arange(a.shape[0]), a]
    return q_vals - target

class RNNAgent(DQNAgent):
    def __init__(self, network: hk.Transformed, 
                 args: DQNArgs):
        # Constructor similar to DQNAgent except that network needs to be an RNN
        # TODO this is hardcoded 1 in David's impl - should this be an arg?
        self.batch_size = 1
        # internalize args
        self.features_shape = args.features_shape
        self.n_actions = args.n_actions
        self.gamma = args.gamma
        assert args.trunc_len is not None
        self.trunc_len = args.trunc_len # for truncated backprop through time
        self._rand_key, network_rand_key = random.split(args.rand_key)
        self.network = network
        self.network_params = self.network.init(rng=network_rand_key, x=jnp.zeros((self.batch_size, self.trunc_len, *self.features_shape), dtype=jnp.float32))
        if args.optimizer == "sgd":
            self.optimizer = sgd(args.alpha)
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

        
    def act(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Get next epsilon-greedy action given a obs, using the agent's parameters.
        :param obs: (batch x obs_size) obs to find actions for. 
        """        
        action, self._rand_key = self._functional_act(obs, self._rand_key, self.network_params)        
        
        return action
    
    @partial(jit, static_argnums=0)
    def _functional_act(self, obs, rand_key, network_params):
       # expand obervation dim to include batch dim
       # TODO maybe we should be doing this in the training loop?
       exp_obs = jnp.expand_dims(obs, 0)
       policy, _ = self._functional_policy(exp_obs, network_params)
       key, subkey = random.split(rand_key)
       #action_choices = jnp.full((*exp_obs.shape[:-1], self.n_actions), jnp.arange(self.n_actions))
    #    print(action_choices.shape)
    #    print(policy.shape)
       # TODO is there a better way? for some reason the following doesn't work
       # action = random.choice(key=subkey, a=action_choices, p=policy, axis=-1, shape=(*exp_obs.shape[:-1], 1))
       
       action = jnp.array([[]], dtype=jnp.int32)
       for batch in range(policy.shape[0]):
            actions_ts = []
            for ts in range(policy.shape[1]):
                key, subkey = random.split(key)
                actions_ts.append(random.choice(key=subkey, a = jnp.arange(self.n_actions), p=policy[batch][ts]))
            action = jnp.append(action, jnp.array([actions_ts]), axis=-1)
       return action, key

    def policy(self, obs: jnp.ndarray):
        """
        Get policy from agent's saved parameters at the given obs.
        :param obs: (batch x time_steps x obs_size) obs(s) to find actions for.
        """
        return self._functional_policy(obs, self.network_params)

    @partial(jit, static_argnums=0)
    def _functional_policy(self, obs: jnp.ndarray, network_params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
        probs = jnp.zeros((*obs.shape[:-1], self.n_actions)) + self.eps / self.n_actions
        greedy_idx, qs = self._greedy_act(obs, network_params)
        # TODO I genuinely don't know a better way to do this atm, 
        # but hopefully this shouldn't add much overhead?
        for batch in range(probs.shape[0]):
            for ts in range(probs.shape[1]):
                probs = probs.at[(batch, ts, greedy_idx[batch][ts])].add(1 - self.eps)
        return probs, qs


    @partial(jit, static_argnums=0)
    def _greedy_act(self, obs: jnp.ndarray, network_params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get greedy actions given a obs
        :param obs: (b x time_steps x *obs.shape) obs to find actions
        :param network_params: NN parameters to use to calculate Q.
        :return: (b x time_steps) Greedy actions
        """
        qs = self.Qs(obs, network_params)
        return jnp.argmax(qs, -1), qs

    
    def Qs(self, obs: jnp.ndarray, network_params: hk.Params) -> jnp.ndarray:
        """
        Get all Q-values given a obs.
        :param obs: (b x time_steps x *obs.shape) obs to find action-values
        :param model: Optional. Potenially use another model
        :return: (b x time_steps x actions) torch.tensor full of action-values.
        """
        return self.network.apply(network_params, obs)[0]

    def Q(self, obs: jnp.ndarray, action: jnp.ndarray, network_params: hk.Params) -> jnp.ndarray:
        """
        Get action-values given a obs and action
        :param obs: (b x time_steps x *obs.shape) obs to find action-values
        :param action: (b x time_steps) Actions for action-values
        :param network_params: NN parameters to use to calculate Q 
        :return: (b x time_steps) Action-values
        """
        qs = self.Qs(obs, network_params)
        return qs[jnp.arange(action.shape[0]), action]
    
    def _loss(self, network_params: hk.Params,
             batch: JaxBatch):
        q_s0 = self.Qs(batch.obs, network_params)
        q_s1 = self.Qs(batch.next_obs, network_params)
        # print(action)
        # print(jnp.full(action.shape, self.gamma))
    

        td_err = self.batch_error_fn(q_s0, batch.actions, batch.rewards, jnp.where(batch.terminals, 0., self.gamma), q_s1, batch.next_actions)
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
    
def train_rnn_agent(mdp: MDP,
                    agent: DQNAgent,
                    total_steps: int,
                    zero_obs = False):
    """
    Training loop for a dqn agent.
    :param mdp: mdp to train on. Currently DQN does not support AMDPs.
    :param agent: DQNAgent to train.
    :param total_steps: Number of steps to train for.
    """

    steps = 0
    num_eps = 0

    jit_onehot = jax.jit(one_hot, static_argnames=["n"])
    if zero_obs:
        jit_onehot = jax.jit(lambda x, y: jnp.zeros((mdp.n_obs,)))
    
    avg_rewards = []
    while (steps < total_steps):
        # truncation buffers
        all_obs, all_actions, terminals, rewards = [], [], [], []
        
        o_0, _ = mdp.reset()
        o_0_processed = jit_onehot(o_0, mdp.n_obs)
        all_obs.append(o_0_processed)
        
        a_0 = agent.act(np.array(all_obs))[-1][-1]
        all_actions.append(a_0)
        
        for _ in range(agent.trunc_len):
            
            o_1, r_0, done, _, _ = mdp.step(a_0, gamma_terminal=False)
            terminals.append(done)
            rewards.append(r_0)
            steps = steps + 1  
            
            o_1_processed = jit_onehot(o_1, mdp.n_obs)
            all_obs.append(o_1_processed)

            a_1 = agent.act(np.array(all_obs))[-1][-1]
            all_actions.append(a_1)
            
            if done:
                #print(f"Broke early after {t} steps")
                break
            
            
            
            o_0_processed = o_1_processed
            a_0 = a_1
            
       
        batch = JaxBatch(obs=[all_obs[:-1]], 
                             actions=[all_actions[:-1]], 
                             next_obs=[all_obs[1:]], 
                             terminals=[terminals], 
                             rewards=[rewards], 
                             next_actions=[all_actions[1:]])
        
        if len(all_actions[:-1]) > agent.trunc_len:
            print(f"Episode length {len(all_actions[:-1])} was longer than truncation len {agent.trunc_len}")
        
        avg_rewards.append(np.average(rewards))
        batch_nonzero_rewards = np.fromiter((x for x in rewards if x != 0), dtype=np.float32)
        # if avg_rewards[-1] != 0:
        #     print(batch_nonzero_rewards)
        if len(batch_nonzero_rewards) == 1 and num_eps > total_steps / 100:
            print(f"Number of nonzero rewards was {len(batch_nonzero_rewards)}")
            print(batch)
            print(agent.Qs(batch.obs, agent.network_params))
            #raise ValueError
        
        loss = agent.update(batch)
           
        if num_eps % 1000 == 0:
            rewards_good = [x for x in avg_rewards if x > 0]
            rewards_bad = [x for x in avg_rewards if x < 0]
            pct_success = len(rewards_good) / len(avg_rewards)
            pct_fail = len(rewards_bad) / len(avg_rewards)
            pct_neutral = 1 - pct_success - pct_fail
            avg_reward = np.average(np.array(avg_rewards))
            avg_rewards = []
            print(f"Step {steps} | Episode {num_eps} | Loss {loss} | Reward {batch.rewards} | Success/Fail/Neutral {pct_success}/{pct_fail}/{pct_neutral} | Obs {batch.obs} | Q-vals {agent.Qs(batch.obs, agent.network_params)}")
            #print(f"Policy {agent.policy(batch.obs)}")
        
        num_eps = num_eps + 1
        # if num_eps >= 50000:
        #     print("break")

    return agent
        
