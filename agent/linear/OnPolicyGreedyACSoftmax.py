# Import modules
import numpy as np
from scipy.stats import multivariate_normal
import time
from scipy import special
from PyFixedReps import TileCoder
from agent.baseAgent import BaseAgent
import time
import inspect
from numba import njit


class GreedyAC(BaseAgent):
    """
    GreedyAC implements an online, on-policy, linear
    Greedy Actor-Critic algorithm.
    This algorithm uses the conditional cross-entropy optimization
    method to update its actor and TD learning to learn a Q function
    critic, both using linear function approximation and tile coding.
    The algorithm can also optionally use traces for both actor and critic.
    """
    def __init__(self, decay, env, actor_lr, critic_lr, gamma,
                 bins, num_tilings, n_actor_updates, n_critic_updates,
                 temperature=1.0, seed=None,
                 trace_type="replacing", critic_type="esarsa"):
        """
        Constructor

        Parameters
        ----------
        actor_lr : float
            The learning rate for the actor
        critic_lr : float
            The learning rate for the critic
        gamma : float
            The environmental discount factor
        seed : int
            The seed to use for the normal distribution sampler, by default
            None. If set to None, uses the integer value of the Unix time.

        TODO: finish
        """
        super().__init__()
        action_space = env.action_space

        # Set the agent's policy sampler
        if seed is None:
            seed = int(time())
        self._random = np.random.default_rng(seed=int(seed))
        np.random.seed(int(seed))
        self._seed = seed

        # Needed so that when evaluating offline, we don't explore
        self._is_training = True

        # Tile Coder
        input_ranges = list(zip(env.observation_space.low,
                                env.observation_space.high))
        dims = env.observation_space.shape[0]
        params = {
                    "dims": dims,
                    "tiles": bins,
                    "tilings": num_tilings,
                    "input_ranges": input_ranges,
                    "scale_output": False,
                }
        self._tiler = TileCoder(params)

        # Keep track of state, observation, and action dimensions
        self._state_shape = env.observation_space.shape[0]
        self._state_features = self._tiler.features() + 1
        self._state_nonzero = num_tilings + 1
        self._action_n = action_space.n
        self._avail_actions = np.array(range(self._action_n))

        # Weights
        self._actor_weights = np.zeros((self._action_n, self._state_features))
        self._critic_weights = np.zeros_like(self._actor_weights)

        # Traces
        if decay < 0:
            raise ValueError("cannot use decay < 0")
        elif decay >= 1:
            raise ValueError("cannot use decay >= 1")
        elif decay == 0:
            self._use_trace = False
        else:
            self._λ = decay
            self._use_trace = True

        if self._use_trace:
            self._critic_trace = np.zeros_like(self._actor_weights)

        self._trace_type = trace_type.lower()
        if self._trace_type not in ("accumulating", "replacing"):
            raise ValueError("trace_type must be one of 'accumulating', " +
                             "'replacing'")

        # Set learning rates and other scaling factors
        self._critic_α = critic_lr / self._state_nonzero
        self._actor_α = actor_lr / self._state_nonzero
        self._γ = gamma

        if temperature < 0:
            raise ValueError("cannot use temperature < 0")
        self._τ = temperature

        # Set the critic type
        self._critic_type = critic_type.lower()
        if self._critic_type not in ("sarsa", "esarsa"):
            raise ValueError("critic_type must be one of 'sarsa', 'esarsa")

        # How many actor and critic updates per step
        self._n_actor_updates = n_actor_updates
        self._n_critic_updates = n_critic_updates

        source = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        self.info = {"source": source}

    def update(self, state, action, reward, next_state, done_mask):
        state = np.concatenate([
                np.zeros((1,), dtype=np.int32),
                self._tiler.get_indices(state) + 1,
            ]
        )

        next_state = np.concatenate([
                np.zeros((1,), dtype=np.int32),
                self._tiler.get_indices(next_state) + 1,
            ]
        )

        for _ in range(self._n_critic_updates):
            self._update_critic(state, action, reward, next_state, done_mask)

        # ###############################################
        # Actor update
        # ###############################################
        # Get the action values of each action
        q_values = self._action_values(state)

        # Choose action of maximal value for the CCEM update, breaking
        # ties randomly
        max_value = np.max(q_values)
        max_actions = np.where(q_values == max_value)[0]
        if len(max_actions) > 1:
            action = self._random.choice(max_actions)
        else:
            action = max_actions[0]

        # Update the action
        for _ in range(self._n_actor_updates):
            actor_grad = self._actor_grad(state, action)
            self._actor_weights += (self._actor_α * actor_grad)

        # In order to be consistent across all children of BaseAgent, we
        # return all transitions with the shape B x N, where N is the number
        # of state, action, or reward dimensions and B is the batch size = 1
        reward = np.array([reward])

        return np.expand_dims(state, axis=0), np.expand_dims(action, axis=0), \
            np.expand_dims(reward, axis=0), np.expand_dims(next_state, axis=0)

    def reset(self):
        if self._use_trace:
            self._critic_trace[:] = 0

    def eval(self):
        self._is_training = False

    def train(self):
        self._is_training = True

    def sample_action(self, state):
        return self._sample_action(state)

    def get_parameters(self):
        pass

    def _get_logits(self, state_ind):
        if self._τ == 0:
            raise ValueError("cannot compute logits when τ = 0")

        logits = self._actor_weights[:, state_ind].sum(axis=-1)
        logits /= self._τ

        return logits

    def _get_probs(self, state_ind):
        if self._τ == 0:
            logits = self._action_values(state_ind)

            max_value = np.amax(logits)
            max_actions = np.where(logits == max_value)[0]

            probs = np.zeros(self._action_n)
            probs[max_actions] = 1 / len(max_actions)
            return probs

        logits = self._get_logits(state_ind)
        logits -= logits.max()  # Subtract max because SciPy breaks things
        pi = special.softmax(logits)
        return pi

    def _sample_action(self, state, tile_coded=False):
        """
        Samples an action from the actor

        Parameters
        ----------
        state : np.array
            The state observation, not tile-coded
        tile_coded : bool
            Whether the input state is tile coded or not

        Returns
        -------
        np.array of float
            The action to take
        """
        if len(state.shape) != 1:
            shape = state.shape
            raise ValueError(f"state must be a vector but got shape {shape}")

        if not tile_coded:
            state = np.concatenate(
                [
                    np.zeros((1,), dtype=np.int32),
                    self._tiler.get_indices(state) + 1,
                ]
            )

        probs = self._get_probs(state)

        # If in offline evaluation mode, return the action of maximum
        # probability
        if not self._is_training:
            actions = np.where(probs == np.max(probs))[0]
            if len(actions) == 1:
                return actions[0]
            else:
                return self._random.choice(actions)

        # Sample action from a multinoulli distribution
        action = self._random.choice(self._avail_actions, p=probs)

        return action

    def _actor_grad(self, state, action):
        """
        Compute and return the actor gradient

        Parameters
        ----------
        state : np.array[any]
            The state observation, in tile-coded non-zero indices form
        action : int
            The action to evaluate the gradient at
        """
        π = self._get_probs(state)
        π = np.reshape(π, (self._actor_weights.shape[0], 1))

        features = np.zeros_like(self._actor_weights)
        features[action, state] = 1

        grad = features
        grad[:, state] -= π
        return grad

    def _action_values(self, state_ind):
        """
        Returns the action values of each action given a tile-coded
        state observation.

        Parameters
        ----------
        state_ind : np.array[int]
            The tile-coded state
        action : int
            The action taken

        Returns
        -------
        float
            The action value
        """
        return self._critic_weights[:, state_ind].sum(axis=1)

    def _action_value(self, state_ind, action):
        """
        Returns the action value of `action` given a tile-coded
        state observation.

        Parameters
        ----------
        state_ind : np.array[int]
            The tile-coded state
        action : int
            The action taken

        Returns
        -------
        float
            The action value
        """
        return self._critic_weights[action, state_ind].sum()

    def _update_critic(self, state, action, reward, next_state, done_mask):
        """
        TODO: description

        Parameters
        ----------
        state : np.ndarray
            The indices of non-zero elements of the tile-coded feature vector
            for the state
        action : TODO
        reward : TODO
        next_state : np.ndarray
            The indices of non-zero elements of the tile-coded feature vector
            for the next state
        done : TODO

        Returns
        -------
        TODO

        """
        q = self._action_value(state, action)

        if self._critic_type == "sarsa":
            next_action = self._sample_action(next_state, tile_coded=True)
            next_q = self._action_value(next_state, next_action)

        elif self._critic_type == "esarsa":
            probs = self._get_probs(next_state)
            action_values = self._action_values(next_state)
            next_q = probs @ action_values

        else:
            raise ValueError(f"unknown critic type {self._critic_type}")

        # Construct the target and TD error
        target = next_q * self._γ * done_mask + reward
        δ = target - q

        if self._use_trace:
            self._critic_trace *= (self._γ * self._λ)

            if self._trace_type == "accumulating":
                self._critic_trace[action, state] += 1
            elif self._trace_type == "replacing":
                self._critic_trace[action, state] = 1
            else:
                raise ValueError(f"unknown trace type {self._trace_type}")

            self._critic_weights += (self._critic_α * δ * self._critic_trace)
        else:
            self._critic_weights[action, state] += (self._critic_α * δ)
