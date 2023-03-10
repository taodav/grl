"""
Basic DQN (or SARSA) agent.
Based off of David Tao's implementation at https://github.com/taodav/uncertainty/blob/main/unc/agents/dqn.py .
"""
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from functools import partial
from jax import random, jit, vmap
from optax import sgd
from typing import Tuple, Union
from grl import MDP
from grl.utils.batching import JaxBatch
from grl.mdp import one_hot
from dataclasses import dataclass

@dataclass
class DQNArgs:
    features_shape: Tuple[int]
    n_actions: int
    gamma: float
    rand_key: random.PRNGKey
    epsilon: float = 0.1
    optimizer: str = "sgd"
    alpha: float = 0.01
    algo: str = "sarsa"
    trunc_len: int = None


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
                 args: DQNArgs):
        self.features_shape = args.features_shape
        self.n_actions = args.n_actions
        self.gamma = args.gamma

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
    states, actions, next_states, terminals, rewards, next_actions = [], [], [], [], [], []
    while (steps < total_steps):
        done = False
        s_0, _ = mdp.reset()
        s_0_processed = jnp.array([one_hot(s_0, mdp.n_obs)])
        a_0 = int(agent.act(s_0_processed))
        while not done:
            s_1, r_0, done, _, _ = mdp.step(a_0, gamma_terminal=False)

            s_1_processed = jnp.array([one_hot(s_1, mdp.n_obs)])
            a_1 = int(agent.act(s_1_processed))

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
               


            if steps % 1000 == 0:
                print(f"Step {steps} | Episode {num_eps} | Loss {loss}")
                       
                
       
        
        num_eps = num_eps + 1

    return agent
        



            





