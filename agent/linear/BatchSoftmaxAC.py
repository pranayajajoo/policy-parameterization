# Import modules
import numpy as np
from agent.baseAgent import BaseAgent
from PyFixedReps import TileCoder
import time
from scipy import special
import inspect
from utils.experience_replay import NumpyBuffer as ExperienceReplay
from numba import njit


class SoftmaxAC(BaseAgent):
    """
    Class SoftmaxAC implements Linear-Softmax Actor-Critic with experience
    replay. Its Gaussian counterpart is outlined in "Model-Free
    Reinforcement Learning with Continuous Action in Practice",
    which can be found at:

    https://hal.inria.fr/hal-00764281/document

    This implementation is also based off that in the RL Book.
    """
    def __init__(self, env, critic_lr, actor_lr, gamma, bins, num_tilings,
                 n_actor_updates, n_critic_updates, replay_capacity,
                 batch_size, temperature=1.0, seed=None):
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
        scaled : bool, optional
            Whether the actor learning rate should be scaled by sigma^2 for
            learning stability, by default False
        clip_stddev : float, optional
            The value at which the standard deviation is clipped in order to
            prevent numerical overflow, by default 1000. If <= 0, then
            no clipping is done.
        seed : int
            The seed to use for the normal distribution sampler, by default
            None. If set to None, uses the integer value of the Unix time.
        """
        super().__init__()
        action_space = env.action_space

        # Set the agent's policy sampler
        if seed is None:
            seed = int(time())
        self._random = np.random.default_rng(seed=int(seed))
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
        state_features = self._tiler.features() + 1

        # Experience Replay
        if batch_size < 0:
            raise ValueError("cannot have batch size less than 0")
        if replay_capacity < 0:
            raise ValueError("cannot have replay capacity less than 0")

        tiled_state_size = (num_tilings + 1,)
        self._batch_size = batch_size
        self._replay = ExperienceReplay(replay_capacity, seed,
                                        tiled_state_size, 1)

        # The weight parameters
        self._action_n = action_space.n
        self._avail_actions = np.array(range(self._action_n))
        self._state_features = state_features
        self._actor_weights = np.zeros((self._action_n, state_features))

        # Set learning rates and other scaling factors
        self._critic_α = critic_lr / (num_tilings + 1)
        self._actor_α = actor_lr / (num_tilings + 1)
        self._γ = gamma
        if temperature < 0:
            raise ValueError("cannot use temperature < 0")
        self._τ = temperature

        # How many times to update the actor and critic
        self._n_actor_updates = n_actor_updates
        self._n_critic_updates = n_critic_updates

        source = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        self.info = {"source": source}

    def _get_logits(self, state_ind):
        """
        Gets the logits of the policy in state

        Parameters
        ----------
        state : np.array
            The indices of the nonzero features in the tile coded state
            representation

        Returns
        -------
        np.array of float
            The logits of each action
        """
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

    def _sample_action_batch(self, state):  # state should be tile coded
        if state.ndim != 2:
            raise ValueError("state should be a batch of states")
        if not self._is_training:
            raise RuntimeError("do not call _sample_action_batch in eval mode")

        probs = self._get_probs(state)

        if np.isnan(probs).any():
            return np.zeros(probs.shape[0])
        else:
            return _multinomial_rvs(probs)

    def sample_action(self, state):
        if state.ndim != 1:
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

    def _actor_grad(self, state, action, delta):
        """
        Returns the gradient of the log-likelihood of taking `action` in
        `state`.

        Parameters
        ----------
        state : np.ndarray
            The state observation, not tile coded
        action : int
            The action to evaluate the gradient on
        """
        π = self._get_probs(state)
        π = np.expand_dims(π, 1)
        delta = np.expand_dims(delta, 1)

        grad = np.zeros((self._batch_size, *self._actor_weights.shape))
        batch_ind = np.array(range(action.shape[0]))
        batch_ind = np.expand_dims(batch_ind, 1)
        grad[batch_ind, action, state] = 1
        grad[batch_ind, :, state] -= π
        grad[batch_ind, :, state] *= delta

        return grad

    def _update_critic(self, state, action, reward, next_state, done_mask):
        # Calculate TD error
        next_v = self._critic_weights[next_state].sum(axis=-1, keepdims=True)
        target = reward + done_mask * self._γ * next_v
        estimate = self._critic_weights[state].sum(axis=-1, keepdims=True)

        delta = target - estimate
        grad = np.zeros((self._batch_size, self._critic_weights.shape[0]))
        np.put_along_axis(grad, state, delta, 1)
        grad = grad.mean(axis=0)
        self._critic_weights += (self._critic_α * grad)

        self._critic_weights[state] += (self._critic_α * delta)
        return

    def _update_actor(self, state, action, reward, next_state, done_mask):
        # Calculate TD error
        next_v = self._critic_weights[next_state].sum(1, keepdims=True)
        target = reward + done_mask * self._γ * next_v
        estimate = self._critic_weights[state].sum(1, keepdims=True)
        delta = target - estimate

        actor_grad = self._actor_grad(state, action, delta)
        actor_grad = actor_grad.mean(axis=0)
        self._actor_weights += (self._actor_α * actor_grad)

    def update(self, state, action, reward, next_state, done_mask):
        state = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self._tiler.get_indices(state) + 1,
            ]
        )

        next_state = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self._tiler.get_indices(next_state) + 1,
            ]
        )

        self._replay.push(state, action, reward, next_state, done_mask)

        if not self._replay.is_sampleable(self._batch_size):
            return

        # The replay buffer is not yet sampleable, we need to wait till it
        # fills up more
        if not self._replay.sampleable:
            return

        # This part of the algorithm differs from the paper
        # (https://hal.inria.fr/hal-00764281/PDF/DegrisACC2012.pdf)
        # in that the paper uses the same TD error to update the actor and the
        # critic. We use the TD error to update the critic, then recompute the
        # TD error (using the updated critic) to update the actor.
        for _ in range(self._n_critic_updates):
            state_batch, action_batch, reward_batch, next_state_batch, \
                done_mask_batch = self._replay.sample(self._batch_size)

            self._update_critic(state_batch, action_batch, reward_batch,
                                next_state_batch, done_mask_batch)

        for _ in range(self._n_actor_updates):
            state_batch, action_batch, reward_batch, next_state_batch, \
                done_mask_batch = self._replay.sample(self._batch_size)

            self._update_actor(state_batch, action_batch, reward_batch,
                               next_state_batch, done_mask_batch)

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


class SoftmaxACV(SoftmaxAC):
    """
    SoftmaxACV is SoftmaxAC with a state value critic
    """
    def __init__(self, env, critic_lr, actor_lr, gamma, bins, num_tilings,
                 n_actor_updates, n_critic_updates, replay_capacity,
                 batch_size, temperature=1.0, seed=None):
        super().__init__(env, critic_lr, actor_lr, gamma, bins, num_tilings,
                         n_actor_updates, n_critic_updates, replay_capacity,
                         batch_size, temperature, seed)

        # State value critic
        self._critic_weights = np.zeros(self._state_features)

    def _update_critic(self, state, action, reward, next_state, done_mask):
        # Calculate TD error
        next_v = self._critic_weights[next_state].sum(axis=-1, keepdims=True)
        target = reward + done_mask * self._γ * next_v
        estimate = self._critic_weights[state].sum(axis=-1, keepdims=True)

        delta = target - estimate
        grad = np.zeros((self._batch_size, self._critic_weights.shape[0]))
        np.put_along_axis(grad, state, delta, 1)
        grad = grad.mean(axis=0)
        self._critic_weights += (self._critic_α * grad)
        return

    def _update_actor(self, state, action, reward, next_state, done_mask):
        # Calculate TD error
        next_v = self._critic_weights[next_state].sum(1, keepdims=True)
        target = reward + done_mask * self._γ * next_v
        estimate = self._critic_weights[state].sum(1, keepdims=True)
        delta = target - estimate

        actor_grad = self._actor_grad(state, action, delta)
        actor_grad = actor_grad.mean(axis=0)
        self._actor_weights += (self._actor_α * actor_grad)


class SoftmaxACQ(SoftmaxAC):
    """
    SoftmaxACQ is SoftmaxAC with an action value critic
    """
    def __init__(self, env, critic_lr, actor_lr, gamma, bins, num_tilings,
                 n_actor_updates, n_critic_updates, replay_capacity,
                 batch_size, critic_type, temperature=1.0, seed=None,
                 q_from_next_state=False):
        super().__init__(env, critic_lr, actor_lr, gamma, bins, num_tilings,
                         n_actor_updates, n_critic_updates, replay_capacity,
                         batch_size, temperature, seed)

        # Action value critic
        self._critic_weights = np.zeros((self._action_n, self._state_features))
        self._q_from_next_state = q_from_next_state

        self._critic_type = critic_type
        avail_critics = ("sarsa", "esarsa")
        if self._critic_type not in avail_critics:
            raise ValueError(f"critic {critic_type} not in {avail_critics}")

    def _update_critic(self, state, action, reward, next_state, done_mask):
        if self._critic_type == "sarsa":
            next_action = self._sample_action_batch(next_state)
            next_action = np.expand_dims(next_action, 1)

            next_q = self._critic_weights[next_action, next_state].sum(
                axis=-1, keepdims=True)
        elif self._critic_type == "esarsa":
            probs = self._get_probs(next_state)
            action_values = self._critic_weights[:, next_state].sum(axis=-1).T

            next_q = (probs * action_values).sum(axis=-1, keepdims=True)

        target = reward + done_mask * self._γ * next_q
        estimate = self._critic_weights[action, state].sum(axis=-1,
                                                           keepdims=True)
        delta = target - estimate
        delta /= self._batch_size

        self._critic_weights[action, state] += (self._critic_α * delta)
        return

    def _update_actor(self, state, action, reward, next_state, done_mask):
        # Calculate the approximation of the return for the policy gradient
        if self._q_from_next_state:
            # Option (1), calculate q as q(s, a) = r + γv' =
            #                                    = r + γ𝔼[Q(S, A) ∣ S = s]
            v_next = self._critic_weights[:, next_state].sum(1).mean()
            q = reward + self._γ * v_next * done_mask
        else:
            # Option (2) calculate q as approximated by the critic
            q = self._critic_weights[action, state].sum(1)

        # Calculate the baseline as v = v(S) = 𝔼[Q(A, S) | S = s]
        v = self._critic_weights[:, state].sum(-1).mean(0)

        delta = np.expand_dims(q - v, 1)

        actor_grad = self._actor_grad(state, action, delta).mean(0)
        self._actor_weights += (self._actor_α * actor_grad)


@njit
def _actor_grad(actor_weights: np.array, state: np.array, action: np.array,
                π: np.array):
    batch_size = π.shape[0]
    state_ind_size = state.shape[1]

    grad = np.zeros((batch_size, *actor_weights.shape))
    for i in range(batch_size):
        a = action[i]
        for j in range(state_ind_size):
            grad[i, a, state[i, j]] = 1

    actions = actor_weights.shape[0]

    for i in range(batch_size):
        grad[i, :, state[i, :]] -= (π[i].reshape(actions, 1) / batch_size)

    return grad


@njit
def _single_multinomial_sample(pvals: np.array):
    return np.random.multinomial(1, pvals=pvals).argmax(-1)


@njit
def _multinomial_rvs(pvals: np.array):
    out = np.zeros(pvals.shape[0], dtype=np.int32)
    for i in range(pvals.shape[0]):
        out[i] = _single_multinomial_sample(pvals[i, :])

    return out
