# Import modules
import numpy as np
from numba import njit
from scipy.stats import multivariate_normal
from env.Bimodal import Bimodal1DEnv
from agent.baseAgent import BaseAgent
import time
from PyFixedReps import TileCoder
from utils.experience_replay import NumpyBuffer as ExperienceReplay
import warnings


class GreedyAC(BaseAgent):
    def __init__(self, env, critic_lr, actor_lr, gamma,
                 alpha, num_samples, rho, critic_bins,
                 critic_num_tilings, actor_bins, actor_num_tilings,
                 replay_capacity, batch_size, n_actor_updates=1,
                 n_critic_updates=1, clip_stddev=1000, seed=None,
                 critic_type="sarsa"):
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
        clip_stddev : float, optional
            The value at which the standard deviation is clipped in order to
            prevent numerical overflow, by default 1000. If <= 0, then
            no clipping is done.
        seed : int
            The seed to use for the normal distribution sampler, by default
            None. If set to None, uses the integer value of the Unix time.
        """
        super().__init__()
        self.batch = True

        available_critics = ("sarsa",)
        if critic_type not in available_critics:
            raise ValueError(
                f"critic type {critic_type} not in {available_critics}"
            )

        self._n_actor_updates = n_actor_updates
        self._n_critic_updates = n_critic_updates

        # Set the agent's policy sampler
        if seed is None:
            seed = int(time())
        self.random = np.random.default_rng(seed=int(seed))
        self.seed = seed

        # GreedyAC update
        self.num_samples = num_samples
        self.rho = rho
        percentile = int(num_samples * rho)
        if percentile <= 0:
            raise ValueError("cannot have percentile <= 0")

        # Entropy regularization
        # Entropy scale alpha should be re-weighted so that the effective
        # entropy scale is unchanged. In the implementation, we repeat the
        # action(s) chosen for entropy regularization `percentile` times
        self.alpha = alpha / float(percentile)

        # Needed so that when evaluating offline, we don't explore
        self.is_training = True

        # Determine standard deviation clipping
        self.clip_stddev = clip_stddev > 0
        self.clip_threshold = np.log(clip_stddev)

        # State and action dimensions
        self.state_dims = env.observation_space.shape[0]
        self.action_dims = env.action_space.shape[0]

        # Critic Tile Coder
        input_ranges = list(zip(env.observation_space.low,
                                env.observation_space.high))
        input_ranges.extend(list(zip(env.action_space.low,
                                     env.action_space.high)))
        critic_dims = env.observation_space.shape[0] + \
            env.action_space.shape[0]
        params = {
                    "dims": critic_dims,
                    "tiles": critic_bins,
                    "tilings": critic_num_tilings,
                    "input_ranges": input_ranges,
                    "scale_output": False,
                }
        self.critic_tiler = TileCoder(params)

        # Actor Tile Coder
        input_ranges = list(zip(env.observation_space.low,
                                env.observation_space.high))
        actor_dims = env.observation_space.shape[0]
        params = {
                    "dims": actor_dims,
                    "tiles": actor_bins,
                    "tilings": actor_num_tilings,
                    "input_ranges": input_ranges,
                    "scale_output": False,
                }
        self.actor_tiler = TileCoder(params)

        # Critic
        self.critic_features = self.critic_tiler.features() + 1
        self.critic_nonzero = critic_num_tilings + 1
        self.critic_weights = np.zeros(self.critic_features)

        # Actor
        self.actor_features = self.actor_tiler.features() + 1
        self.actor_nonzero = actor_num_tilings + 1
        self.sigma_weights = np.zeros((self.action_dims, self.actor_features))
        self.mu_weights = np.zeros((self.action_dims, self.actor_features))

        # Sampler
        self.s_sigma_weights = np.zeros_like(self.sigma_weights)
        self.s_mu_weights = np.zeros_like(self.mu_weights)

        # Replay
        if len(env.observation_space.shape) != 1:
            raise ValueError("GreedyAC can only be used with environments " +
                             "with vector-valued observations")
        state_size = env.observation_space.shape[0]
        self._critic_replay = _GACReplay(
            replay_capacity, seed, [self.critic_nonzero, 1, 1, state_size, 1],
            [np.int32, float, float, np.float64, bool],
        )
        self._actor_replay = _GACReplay(
            replay_capacity, seed,
            [state_size, self.actor_nonzero, 1, 1, self.actor_nonzero, 1],
            [np.float32, np.int32, float, float, np.int32, bool],
        )
        self.batch_size = batch_size

        # Set learning rates and discount
        self.critic_lr = critic_lr / self.critic_nonzero
        self.actor_lr = actor_lr / self.actor_nonzero
        self.gamma = gamma

    def get_mean(self, state_ind, actor=True):
        """
        Gets the mean of the parameterized normal distribution

        Parameters
        ----------
        state_ind : np.array
            The state_ind feature vector

        Returns
        -------
        float
            The mean of the normal distribution
        """
        batch = len(state_ind.shape) > 1

        if actor:
            mu_weights = self.mu_weights
        else:
            mu_weights = self.s_mu_weights

        if not batch:
            mu = mu_weights[:, state_ind].sum(axis=-1)
        else:
            w = mu_weights
            i = state_ind
            mu = w[:, i].sum(axis=-1).T

        return mu

    def get_stddev(self, state_ind, actor=True):
        """
        Gets the standard deviation of the parameterized normal distribution

        Parameters
        ----------
        state_ind : np.array
            The state_ind feature vector

        Returns
        -------
        float
            The standard deviation of the normal distribution
        """
        if actor:
            sigma_weights = self.sigma_weights
        else:
            sigma_weights = self.s_sigma_weights

        batch = len(state_ind.shape) > 1

        # Return un-clipped standard deviation if no clipping
        if not self.clip_stddev:
            if not batch:
                return np.exp(sigma_weights[:, state_ind].sum(axis=-1))
            else:
                w = sigma_weights
                i = state_ind
                return np.exp(w[:, i].sum(axis=-1).T)

        # Clip the standard deviation to prevent numerical overflow
        if not batch:
            log_std = np.clip(sigma_weights[:, state_ind].sum(axis=-1),
                              -self.clip_threshold, self.clip_threshold)
        else:
            log_std = np.clip(sigma_weights[:, state_ind].sum(axis=-1).T,
                              -self.clip_threshold, self.clip_threshold)

        std = np.exp(log_std)
        return std

    def sample_action(self, state, actor=True):
        state = np.concatenate([np.zeros((1,), dtype=np.int32),
                                self.actor_tiler.get_indices(state) + 1])
        mean = self.get_mean(state, actor)

        # If in offline evaluation mode, return the mean action
        if not self.is_training:
            # print("Sampled mean:", mean)
            return np.array(mean)

        stddev = self.get_stddev(state, actor)
        action = self.random.normal(loc=mean, scale=stddev)

        # Sample action from a normal distribution
        return action

    def update_sampler(self, state_ind, action, entropy_action):
        std = self.get_stddev(state_ind, False)
        mean = self.get_mean(state_ind, False)

        mean_weights, std_weights = update_sampler(mean, std, state_ind,
                                                   action, entropy_action,
                                                   self.action_dims,
                                                   self.s_mu_weights,
                                                   self.s_sigma_weights,
                                                   self.actor_lr,
                                                   self.alpha)
        self.s_mu_weights = mean_weights
        self.s_sigma_weights = std_weights

    def update_actor(self, state_ind, action):
        """
        Gets the gradient of the actor's parameters

        Parameters
        ----------
        state : np.array
            The state feature vector, tile coded
        action : np.array
            The action to compute the gradient of, tile coded

        Returns
        -------
        np.array
            The gradient vector of the actor's weights, in the form
            [grad_mu_weights^T, grad_sigma_weights^T]^T
        """
        std = self.get_stddev(state_ind, True)
        mean = self.get_mean(state_ind, True)

        mean_weights, std_weights = update_actor(
            mean, std, state_ind, action, self.action_dims, self.mu_weights,
            self.sigma_weights, self.actor_lr,
        )
        self.mu_weights = mean_weights
        self.sigma_weights = std_weights

    def action_value(self, critic_ind_batch):
        if len(critic_ind_batch.shape) == 1:
            warnings.warn("reshaping into batch")
            critic_ind_batch = critic_ind_batch.reshape(
                1, *critic_index_batch.shape,
            )

        return self.critic_weights[critic_ind_batch].sum(axis=-1)

    def action_value_from_inputs(self, state_batch, action_batch):
        critic_state_action_ind = np.zeros(
            (state_batch.shape[0], self.critic_nonzero),
            dtype=np.int32,
        )

        state_action = np.concatenate(
                [state_batch, action_batch],
                axis=1
        )

        for i in range(state_batch.shape[0]):
            critic_state_action_ind[i, 1:] = \
                    self.critic_tiler.get_indices(state_action[i, :]) + 1

        return self.action_value(critic_state_action_ind)

    def update(self, state, action, reward, next_state, done_mask):
        # Tile code the state-action pair for the critic features
        # We don't tile code the next state, because we need a next action to
        # do so
        critic_ind = self.critic_tiler.get_indices(
                np.concatenate([state, action])
        ) + 1
        critic_ind = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                critic_ind,
            ]
        )

        # I think we need to store the state as well. If so, we'll need a new
        # GreedyAC Critic Replay Buffer, which can be private to this file

        # Tile code the state for the actor features
        actor_ind = self.actor_tiler.get_indices(state) + 1
        actor_ind = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                actor_ind,
            ],
        )

        # Tile code the next state for the actor features
        next_actor_ind = self.actor_tiler.get_indices(next_state) + 1
        next_actor_ind = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                next_actor_ind,
            ],
        )

        # Keep transition in replay buffer
        self._critic_replay.push(critic_ind, action, reward, next_state,
                                 done_mask)
        self._actor_replay.push(state, actor_ind, action, reward,
                                next_actor_ind, done_mask)

        # If the critic replay buffer is not sampleable, fill it more
        # before continuing onto updating the critic weights
        if not self._critic_replay.is_sampleable(self.batch_size):
            return

        # Sample a batch from memory
        for _ in range(self._n_critic_updates):
            critic_ind_batch, action_batch, reward_batch, \
                next_state_batch, mask_batch = \
                self._critic_replay.sample(batch_size=self.batch_size)

            self._update_critic(critic_ind_batch, action_batch, reward_batch,
                                next_state_batch, mask_batch)

        # If the actor replay buffer is not sampleable, fill it more
        # before continuing onto updating the actor weights
        if not self._actor_replay.is_sampleable(self.batch_size):
            return

        # ######################################################3
        # GreedyAC
        # ######################################################3
        for _ in range(self._n_actor_updates):
            state_batch, actor_ind_batch, action_batch, reward_batch, \
                next_state_batch, mask_batch = \
                self._actor_replay.sample(batch_size=self.batch_size)

            mu = self.get_mean(actor_ind_batch, False)
            sigma = self.get_stddev(actor_ind_batch, False)
            sample = self.random.normal(
                mu, sigma,
                size=(self.num_samples, *mu.shape),
            )
            sample = sample.swapaxes(0, 1)

            # Get the action values of each sampled action
            q_values = np.zeros(sample.shape[:-1])
            for i in range(sample.shape[0]):
                current_state = state_batch[i:i+1, :].repeat(
                    self.num_samples, axis=0,
                )
                current_actions = sample[i, :]
                q_values[i, :] = self.action_value_from_inputs(current_state,
                                                               current_actions)

            # Sort actions by value for GreedyAC update
            sorted_ind = np.argsort(q_values, kind="stable", axis=1)
            percentile = int(self.num_samples * self.rho)
            if percentile == 0:
                raise ValueError("percentile must be > 0")

            best_ind = sorted_ind[:, -percentile:]
            ind = np.expand_dims(best_ind, axis=-1)
            best_actions = np.take_along_axis(sample, ind, axis=1)

            # Adjust the state and best action arrays to calculate the gradient
            repeat_s_ind = actor_ind_batch.repeat(percentile, axis=0)
            reshape_a = best_actions.reshape(self.batch_size * percentile,
                                             self.action_dims)

            self.update_actor(repeat_s_ind, reshape_a)

            ################################################
            # Sampler update
            ################################################
            entropy_actions = self.random.normal(mu, sigma)
            entropy_actions = np.repeat(entropy_actions, percentile, axis=0)

            self.update_sampler(repeat_s_ind, reshape_a, entropy_actions)

        # In order to be consistent across all children of BaseAgent, we
        # return all transitions with the shape B x N, where N is the number
        # of state, action, or reward dimensions and B is the batch size = 1
        reward = np.array([reward])

        return np.expand_dims(state, axis=0), np.expand_dims(action, axis=0), \
            np.expand_dims(reward, axis=0), np.expand_dims(next_state, axis=0)

    def _update_critic(self, critic_ind_batch, action_batch, reward_batch,
                       next_state_batch, mask_batch):
        # Calculate the q values of the sampled actions in the sampled states.
        # These are stored tile-coded in the critic replay buffer
        q_batch = self.action_value(critic_ind_batch)

        # Sample next actions for the SARSA update
        next_state_batch_ind = np.zeros((self.batch_size, self.actor_nonzero),
                                        dtype=np.int32)
        for i in range(next_state_batch.shape[0]):
            next_state_ind = self.actor_tiler.get_indices(
                next_state_batch[i, :] + 1
            )

            # First element will be index 0 -- bias unit
            next_state_batch_ind[i, 1:] = next_state_ind

        mu = self.get_mean(next_state_batch_ind)
        sigma = self.get_stddev(next_state_batch_ind)
        next_action_batch = self.random.normal(mu, sigma)

        # Next q values for SARSA update
        next_q_batch = self.action_value_from_inputs(next_state_batch,
                                                     next_action_batch)

        # Reshape to calculate TD error
        mask_batch = mask_batch.ravel()
        reward_batch = reward_batch.ravel()

        # ######################################################3
        # Critic update
        # ######################################################3
        # This could be moved to its own Numba function
        target = reward_batch + next_q_batch * self.gamma * mask_batch
        delta = np.expand_dims(target - q_batch, axis=1)
        for i in range(delta.shape[0]):
            ind = critic_ind_batch[i, :]
            d = delta[i]
            self.critic_weights[ind] += (self.critic_lr * d / self.batch_size)

    def reset(self):
        pass

    def eval(self):
        self.is_training = False

    def train(self):
        self.is_training = True

    def get_parameters(self):
        pass


# @njit
def update_actor(mean: np.array, std: np.array, state_ind: np.array,
                 action: np.array, action_dims: np.int32,
                 mean_weights: np.array, sigma_weights: np.array,
                 actor_lr: np.float64):
    # ###############################################
    # Log-likelihood update
    # ###############################################
    if len(mean.shape) > 2:
        # This code should never run
        raise ValueError("mean must be a vector")

    # Gradient of performance w.r.t. θ_μ
    if len(action.shape) == 1:
        raise NotImplementedError
    else:
        # grad_mu = scale_mu[..., None] * state_onehot[:, None, :]
        scale_mu = ((action - mean) / (std ** 2))
        b = scale_mu.shape[0]  # batch size
        scale_mu = scale_mu * (actor_lr / b)
        for i in range(b):
            scale = scale_mu[i, :]
            ind = state_ind[i, :]
            mean_weights[:, ind] += scale

    # Calculate the gradient of performance w.r.t Σ weights
    if len(action.shape) == 1:
        raise NotImplementedError
    else:
        scale_sigma = (((action - mean) / std) ** 2 - 1)
        scale_sigma = scale_sigma * (actor_lr / b)
        for i in range(b):
            scale = scale_sigma[i, :]
            scale = np.expand_dims(scale, axis=-1)
            ind = state_ind[i, :]
            sigma_weights[:, ind] += scale

    return mean_weights, sigma_weights


@njit
def update_sampler(mean: np.array, std: np.array, state_ind: np.array,
                   action: np.array, entropy_action: np.array,
                   action_dims: np.int32, mean_weights: np.array,
                   sigma_weights: np.array, actor_lr: np.float64,
                   alpha: np.float64):
    # ###############################################
    # Log-likelihood update
    # ###############################################
    # Calculate the gradient of performance w.r.t μ⃗ weights θ
    if len(mean.shape) > 2:
        # This code should never run
        raise ValueError("mean must be a vector")

    # Gradient of performance w.r.t. θ_μ
    if len(action.shape) == 1:
        raise NotImplementedError
    else:
        # grad_mu = scale_mu[..., None] * state_onehot[:, None, :]
        scale_mu = ((action - mean) / (std ** 2))
        b = scale_mu.shape[0]  # batch size
        entropy_mu = alpha * ((entropy_action - mean) / (std ** 2))
        scale_mu += entropy_mu
        scale_mu *= (actor_lr / b)
        for i in range(b):
            scale = scale_mu[i, :]
            ind = state_ind[i, :]
            mean_weights[:, ind] += scale

    # Calculate the gradient of performance w.r.t Σ weights
    if len(action.shape) == 1:
        raise NotImplementedError
    else:
        scale_sigma = (((action - mean) / std) ** 2 - 1)
        entropy_sigma = alpha * (((entropy_action - mean) / std) ** 2 - 1)
        scale_sigma += entropy_sigma
        scale_sigma *= (actor_lr / b)
        for i in range(b):
            scale = scale_sigma[i, :]
            ind = state_ind[i, :]
            sigma_weights[:, ind] += scale

    return mean_weights, sigma_weights


class _GACReplay:
    def __init__(self, capacity, seed, sizes, types):
        """
        Constructor

        Parameters
        ----------
        capacity : int
            The capacity of the buffer
        seed : int
            The random seed used for sampling from the buffer
        sizes : list[int]
            A list of size of arrays to store. The buffer will have
            `len(sizes)` sub-buffers
        types : list[type]
            The types of each replay buffer
        """
        if len(sizes) != len(types):
            raise ValueError(f"expected len(sizes) == len(types)")

        self.is_full = False
        self.position = 0
        self.capacity = capacity

        # Set the random number generator
        self.random = np.random.default_rng(seed=seed)

        # Save the size of states and actions
        self._sizes = sizes

        self._sampleable = False

        # Create buffers:
        self._buffers = []
        for i in range(len(sizes)):
            self._buffers.append(np.zeros((capacity, sizes[i]),
                                          dtype=types[i]))

        self._buffers = tuple(self._buffers)

    @property
    def sampleable(self):
        return self._sampleable

    def push(self, *items):
        """
        Pushes a trajectory onto the replay buffer

        Parameters
        ----------
        items : any
            The items to store in each buffer
        """
        if len(items) != len(self._sizes):
            msg = f"attempted to add {len(items)} items to the replay " + \
                f"buffer when {len(self._sizes)} items expected"
            raise RuntimeError(msg)

        for i in range(len(items)):
            item = items[i]
            if not isinstance(item, np.ndarray):
                item = np.array([item])

            self._buffers[i][self.position] = item

        if self.position >= self.capacity - 1:
            self.is_full = True
        self.position = (self.position + 1) % self.capacity
        self._sampleable = False

    @property
    def sampleable(self):
        return self._sampleable

    def is_sampleable(self, batch_size):
        if self.position < batch_size and not self.sampleable:
            return False
        elif not self._sampleable:
            self._sampleable = True

        return self.sampleable

    def sample(self, batch_size):
        """
        Samples a random batch from the buffer

        Parameters
        ----------
        batch_size : int
            The size of the batch to sample

        Returns
        -------
        5-tuple of torch.Tensor
            The arrays of state, action, reward, next_state, and done from the
            batch
        """
        if not self.is_sampleable(batch_size):
            return None, None, None, None, None

        # Get the indices for the batch
        if self.is_full:
            indices = self.random.integers(low=0, high=len(self),
                                           size=batch_size)
        else:
            indices = self.random.integers(low=0, high=self.position,
                                           size=batch_size)

        out = []
        for i in range(len(self._buffers)):
            out.append(self._buffers[i][indices, :])

        return out

    def __len__(self):
        """
        Gets the number of elements in the buffer

        Returns
        -------
        int
            The number of elements currently in the buffer
        """
        if not self.is_full:
            return self.position
        else:
            return self.capacity
