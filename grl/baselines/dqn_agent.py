"""
Basic DQN (or SARSA) agent.
Based off of David Tao's implementation at https://github.com/taodav/uncertainty/blob/main/unc/agents/dqn.py .
"""
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from functools import partial
from jax import random, jit, vmap, value_and_grad
from optax import sgd, GradientTransformation
from pathlib import Path
from typing import Tuple, Callable, Iterable
from grl import MDP
from grl.utils.batching import JaxBatch

# Error functions from David's impl
def sarsa_error(q: jnp.ndarray, a: int, r: jnp.ndarray, g: float, q1: jnp.ndarray, next_a: int):
    # print(a)
    # print(next_a)
    # print(r)
    # print(g)
    
    target = r + g * q1[next_a]
    target = jax.lax.stop_gradient(target)
   
    # print(target)
    # print(a)
    # print()
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

def mse(predictions: jnp.ndarray, targets: jnp.ndarray = None):
    if targets is None:
        targets = jnp.zeros_like(predictions)
    squared_diff = 0.5 * (predictions - targets) ** 2
    return jnp.mean(squared_diff)




class DQNAgent:
    def __init__(self, network: hk.Transformed,
                 features_shape,
                 n_actions: int,
                 gamma: float,
                 rand_key: random.PRNGKey,
                 epsilon: float = 0.1,
                 optimizer: str = "sgd",
                 alpha: float = 0.01,
                 algo: str = "sarsa",):
    
        self.features_shape = features_shape
        self.n_actions = n_actions
        self.gamma = gamma

        self._rand_key, network_rand_key = random.split(rand_key)
        self.network = network
        self.network_params = self.network.init(rng=network_rand_key, x=jnp.zeros((1, *self.features_shape), dtype=jnp.float32))
        if optimizer == "sgd":
            self.optimizer = sgd(alpha)
        else:
            raise ValueError(f"Unrecognized optimizer {optimizer}")
        self.optimizer_state = self.optimizer.init(self.network_params)
        self.eps = epsilon

        self.error_fn = None
        if algo == 'sarsa':
            self.error_fn = sarsa_error
        elif algo == 'esarsa':
            self.error_fn = expected_sarsa_error
        elif algo == 'qlearning':
            self.error_fn = qlearning_error
        self.batch_error_fn = vmap(self.error_fn)

    def act(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Get next epsilon-greedy action given a state, using the agent's parameters.
        :param state: (*state.shape) (batch of) state(s) to find actions for.
        """
        policy, _ = self.policy(state)
        self._rand_key, subkey = random.split(self._rand_key)
        action = random.choice(subkey, jnp.arange(self.n_actions), p=policy, shape=(state.shape[0],))
        return action

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

def train_dqn_agent(mdp: MDP,
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
    # Not really batching just updating at each step
    batch = JaxBatch(mdp.n_states)
    while (steps < total_steps):
        done = False
        s_0, _ = mdp.reset()
        while not done:
            batch.append_observations([s_0])
            # print(s_0_onehot)
            a_0 = int(agent.act(jnp.array([batch.obs[-1]])))
            batch.append_actions([a_0])

            s_1, r_0, done, _, _ = mdp.step(a_0)
            batch.append_next_observations([s_1])
            batch.append_rewards([r_0])
            batch.append_terminals([done])

            a_1 = int(agent.act(jnp.array([batch.next_obs[-1]])))
            batch.append_next_actions([a_1])
            
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
            
            loss = agent.update(batch)
            if steps % 1000 == 0:
                print(f"Step {steps} | Episode {num_eps} | Loss {loss}")
            
            batch = JaxBatch(mdp.n_states)
            s_0 = s_1
            steps = steps + 1               
                
       
        
        num_eps = num_eps + 1

    return agent
        



            





