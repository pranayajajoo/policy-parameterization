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
from utils.experience_replay import NumpyBuffer as ExperienceReplay


class GreedyAC(BaseAgent):
    """
    GreedyAC implements an online, on-policy, linear
    Greedy Actor-Critic algorithm.
    This algorithm uses the conditional cross-entropy optimization
    method to update its actor and TD learning to learn a Q function
    critic, both using linear function approximation and tile coding.
    The algorithm can also optionally use traces for both actor and critic.
    """
    def __init__(self, actor_lr, critic_lr, gamma,
                 bins, num_tilings, env, replay_capacity, batch_size,
                 temperature=1.0, seed=None, critic_type="esarsa",
                 n_actor_updates=1, n_critic_updates=1):
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

        # Experience Replay
        if batch_size < 0:
            raise ValueError("cannot have batch size less than 0")
        if replay_capacity < 0:
            raise ValueError("cannot have replay capacity less than 0")

        tiled_state_size = (num_tilings + 1,)
        self._batch_size = batch_size
        self._replay = ExperienceReplay(replay_capacity, seed,
                                        tiled_state_size, 1)

        # Keep track of state, observation, and action dimensions
        self._state_shape = env.observation_space.shape[0]
        self._state_features = self._tiler.features() + 1
        self._state_nonzero = num_tilings + 1
        self._action_n = action_space.n
        self._avail_actions = np.array(range(self._action_n))

        # Weights
        self._actor_weights = np.zeros((self._action_n, self._state_features))
        self._critic_weights = np.zeros_like(self._actor_weights)

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

        self._replay.push(state, action, reward, next_state, done_mask)

        # The replay buffer is not yet sampleable, we need to wait till it
        # fills up more
        if not self._replay.is_sampleable(self._batch_size):
            return

        for _ in range(self._n_critic_updates):
            state_batch, action_batch, reward_batch, next_state_batch, \
                done_mask_batch = self._replay.sample(self._batch_size)

            self._update_critic(state_batch, action_batch, reward_batch,
                                next_state_batch, done_mask_batch)

        # ###############################################
        # Actor update
        # ###############################################
        for _ in range(self._n_actor_updates):
            state_batch, action_batch, reward_batch, next_state_batch, \
                done_mask_batch = self._replay.sample(self._batch_size)

            # Get the action values of each action
            q_values = self._action_values(state_batch)

            # Get the action of maximal value in each state)
            # Should update this code to break ties randomly
            action = np.expand_dims(np.argmax(q_values, axis=-1), 1)

            # Update the actor
            actor_grad = self._actor_grad(state_batch, action)
            self._actor_weights += (self._actor_α * actor_grad)

        # In order to be consistent across all children of BaseAgent, we
        # return all transitions with the shape B x N, where N is the number
        # of state, action, or reward dimensions and B is the batch size = 1
        reward = np.array([reward])

        return np.expand_dims(state, axis=0), np.expand_dims(action, axis=0), \
            np.expand_dims(reward, axis=0), np.expand_dims(next_state, axis=0)

    def reset(self):
        pass

    def eval(self):
        self._is_training = False

    def train(self):
        self._is_training = True

    def get_parameters(self):
        pass

    def _get_logits(self, state_ind):
        if self._τ == 0:
            raise ValueError("cannot compute logits when τ = 0")

        logits = self._actor_weights[:, state_ind].sum(axis=-1)
        if self._τ != 1.0:
            logits /= self._τ

        return logits.T

    def _get_probs(self, state_ind):
        if self._τ == 0:
            logits = self._action_values(state_ind)

            # Should fix this: if two actions have the same value we should
            # break ties evenly
            max_actions = np.argmax(logits, axis=-1)
            rows = np.array(range(max_actions.shape[0]))

            probs = np.zeros((logits.shape[0], self._action_n))
            probs[rows, max_actions] = 1 / len(max_actions)
            return probs

        logits = self._get_logits(state_ind)
        logits -= np.amax(logits, axis=-1, keepdims=True)
        pi = special.softmax(logits, axis=1)
        return pi

    def sample_action(self, state):
        if len(state.shape) != 1:
            shape = state.shape
            raise ValueError(f"state must be a vector but got shape {shape}")

        state = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self._tiler.get_indices(state) + 1,
            ]
        )
        state = np.expand_dims(state, 0)
        probs = self._get_probs(state).squeeze(0)

        # If in offline evaluation mode, return the action of maximum
        # probability
        if not self._is_training:
            actions = np.where(probs == np.max(probs))[0]
            if len(actions) == 1:
                return actions[0]
            else:
                return self._random.choice(actions)

        # Sample action from a multinoulli distribution
        if np.isnan(probs).any():
            # If there are NaNs in the probs array, then just use the no-op
            # action
            return 0
        else:
            action = self._random.choice(self._avail_actions, p=probs)
        return action

    def _sample_action_batch(self, state_ind_batch):
        if len(state_ind_batch.shape) != 2:
            shape = state_ind_batch.shape
            raise ValueError(f"state must be a matrix but got shape {shape}")

        probs = self._get_probs(state_ind_batch)

        # Sample action from a multinoulli distribution
        actions = _multinomial_rvs(probs)
        return actions

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

        return _actor_grad(self._actor_weights, state, action, π)

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
        return self._critic_weights[:, state_ind].sum(axis=-1).T

    def _action_value(self, state_ind, actions):
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
        if len(actions.shape) == 1 and len(state_ind.shape) == 2:
            actions = np.expand_dims(actions, 1)
        elif len(actions.shape) != len(state_ind.shape):
            raise ValueError("actions and state should have the same ndims")

        return self._critic_weights[actions, state_ind].sum(-1)

    def _update_critic(self, state_batch, action_batch, reward_batch,
                       next_state_batch, done_mask_batch):
        q_batch = np.expand_dims(
            self._action_value(state_batch, action_batch),
            1,
        )

        if self._critic_type == "sarsa":
            next_action_batch = self._sample_action_batch(next_state_batch)

            next_q_batch = self._critic_weights[
                next_action_batch, next_state_batch].sum(axis=-1,
                                                         keepdims=True)

        elif self._critic_type == "esarsa":
            probs = self._get_probs(next_state_batch)
            action_values = self._action_values(next_state_batch)
            next_q_batch = (probs * action_values).sum(axis=-1, keepdims=True)

        else:
            raise ValueError(f"unknown critic type {self._critic_type}")

        # Construct the target and TD error
        target = next_q_batch * self._γ * done_mask_batch + reward_batch
        δ = (target - q_batch) / self._batch_size

        self._critic_weights[action_batch, state_batch] += (self._critic_α * δ)


def _single_multinomial_sample(pvals: np.array):
    return np.random.multinomial(1, pvals=pvals, size=1).argmax(-1)


def _multinomial_rvs(pvals: np.array):
    return np.apply_along_axis(_single_multinomial_sample, 1, pvals)


@njit
def _actor_grad(actor_weights: np.array, state: np.array, action: np.array,
                π: np.array):
    batch_size = π.shape[0]
    state_ind_size = state.shape[1]

    grad = np.zeros_like(actor_weights)
    for i in range(batch_size):
        for j in range(state_ind_size):
            grad[action[i], state[i, j]] = 1

    actions = actor_weights.shape[0]

    for i in range(batch_size):
        grad[:, state[i, :]] -= (π[i].reshape(actions, 1) / batch_size)

    return grad
