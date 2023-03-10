"""
SARSA agent that utilizes an RNN.
Relies on haiku's recurrent API (https://dm-haiku.readthedocs.io/en/latest/api.html#recurrent).
Also based on David Tao's implementation at https://github.com/taodav/uncertainty/blob/main/unc/agents/lstm.py .
"""
import jax
import jax.numpy as jnp
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
    target = r + g * q1[jnp.arange(next_a.shape[0]), next_a]
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
        :param obs: (batch x obs_size) obs(s) to find actions for. 
        """
        # expand dimensions to include timesteps
        # TODO differing from david here?
        obs = jnp.expand_dims(obs, 0)
        policy, _ = self.policy(obs)
        self._rand_key, subkey = random.split(self._rand_key)
        action = random.choice(subkey, jnp.arange(self.n_actions), p=policy, shape=(obs.shape[1],))
        return action

    def policy(self, obs: jnp.ndarray):
        """
        Get policy from agent's saved parameters at the given obs.
        :param obs: (batch x time_steps x obs_size) obs(s) to find actions for.
        """
        return self._functional_policy(obs, self.network_params)

    @partial(jit, static_argnums=0)
    def _functional_policy(self, obs: jnp.ndarray, network_params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
        probs = jnp.zeros(self.n_actions) + self.eps / self.n_actions
        greedy_idx, qs = self._greedy_act(obs, network_params)
        probs = probs.at[greedy_idx].add(1 - self.eps)
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
        return jnp.argmax(qs, axis=2), qs

    
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
                    total_steps: int):
    """
    Training loop for a dqn agent.
    :param mdp: mdp to train on. Currently DQN does not support AMDPs.
    :param agent: DQNAgent to train.
    :param total_steps: Number of steps to train for.
    """

    steps = 0
    num_eps = 0
    
    while (steps < total_steps):
        # truncation buffers
        obs, actions, next_obs, terminals, rewards, next_actions = [], [], [], [], [], []
        o_0, _ = mdp.reset()
        o_0_processed = jnp.array(one_hot(o_0, mdp.n_obs))
        obs.append(o_0_processed)
        
        for t in range(agent.trunc_len):
            a_0 = int(agent.act(jnp.array(obs))[-1])
            actions.append(a_0)
            o_1, r_0, done, _, _ = mdp.step(a_0, gamma_terminal=False)
            terminals.append(done)
            rewards.append(r_0)
            steps = steps + 1  
            
            o_1_processed = jnp.array(one_hot(o_1, mdp.n_obs)) 
            next_obs.append(o_1_processed)

            a_1 = int(agent.act(jnp.array(next_obs))[-1])
            next_actions.append(a_1)
            
            if done:
                break
            

            #print([agent.Qs(jnp.array([s]), agent.network_params) for s in range(mdp.n_states)])
            # print()
            # print(jnp.array(states))
            # print(jnp.array(actions))
            # print(jnp.array(next_states))
            # print(jnp.array(terminals))
            # print(jnp.array(rewards))
            # print(jnp.array(next_actions))
            # print()
            # print(s_1)
            # q_s0 = agent.Qs(jnp.array(states), agent.network_params)
            # q_s1 = agent.Qs(jnp.array(next_states), agent.network_params)
      

            # td_err = agent.batch_error_fn(q_s0, jnp.array(actions), jnp.array(rewards), jnp.where(jnp.array(terminals), 0., mdp.gamma), q_s1, jnp.array(next_actions))
            
            
            
            o_0_processed = o_1_processed
            obs.append(o_0_processed)
            
        
        batch = JaxBatch(obs=[obs], 
                             actions=[actions], 
                             next_obs=[next_obs], 
                             terminals=[terminals], 
                             rewards=[rewards], 
                             next_actions=[next_actions])
        print(batch)
        loss = agent.update(batch) 
           


        if num_eps % 1000 == 0:
            print(f"Step {steps} | Episode {num_eps} | Loss {loss}")                  
        
        num_eps = num_eps + 1

    return agent
        
