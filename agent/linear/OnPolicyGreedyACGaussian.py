# Import modules
import numpy as np
from numba import njit
from scipy.stats import multivariate_normal, norm
from agent.baseAgent import BaseAgent
import time
from PyFixedReps import TileCoder
import warnings


class GreedyAC(BaseAgent):
    def __init__(self, env, critic_lr, actor_lr, gamma, decay, trace_type,
                 alpha, num_samples, rho, critic_bins,
                 critic_num_tilings, actor_bins, actor_num_tilings,
                 n_actor_updates=1, n_critic_updates=1, clip_stddev=1000,
                 seed=None, critic_type="sarsa"):
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
        self.batch = False

        # Set the critic type
        available_critics = ("sarsa",)
        if critic_type not in available_critics:
            raise ValueError(
                f"critic type {critic_type} not in {available_critics}"
            )

        self._n_actor_updates = n_actor_updates
        self._n_critic_updates = n_critic_updates

        # RNG
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

        # Entropy regularization scale
        self.alpha = alpha

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

        # Eligibiity traces
        self._use_trace = decay > 0
        if self._use_trace:
            self._critic_trace = np.zeros_like(self.critic_weights)
            self._λ = decay

            available_traces = ("replacing", "accumulating")
            if trace_type not in available_traces:
                raise RuntimeError(f"unknown trace type {trace_type}")
            else:
                self._trace_type = trace_type

        # Actor
        self.actor_features = self.actor_tiler.features() + 1
        self.actor_nonzero = actor_num_tilings + 1
        self.sigma_weights = np.zeros((self.action_dims, self.actor_features))
        self.mu_weights = np.zeros((self.action_dims, self.actor_features))

        # Sampler
        self.s_sigma_weights = np.zeros_like(self.sigma_weights)
        self.s_mu_weights = np.zeros_like(self.mu_weights)

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
        if len(state_ind.shape) > 1:
            raise ValueError()

        if actor:
            mu_weights = self.mu_weights
        else:
            mu_weights = self.s_mu_weights

        mean = mu_weights[:, state_ind].sum(axis=-1)
        return mean

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
        if len(state_ind.shape) > 1:
            raise ValueError()

        if actor:
            sigma_weights = self.sigma_weights
        else:
            sigma_weights = self.s_sigma_weights

        # Return un-clipped standard deviation if no clipping
        if not self.clip_stddev:
            return np.exp(sigma_weights[:, state_ind].sum(axis=-1))

        # Clip the standard deviation to prevent numerical overflow
        log_std = np.clip(sigma_weights[:, state_ind].sum(axis=-1),
                          -self.clip_threshold, self.clip_threshold)

        std = np.exp(log_std)
        return std

    def sample_action(self, state, actor=True):
        state = self._actor_tile(state)
        mean = self.get_mean(state, actor)

        # If in offline evaluation mode, return the mean action
        if not self.is_training:
            return np.array(mean)

        stddev = self.get_stddev(state, actor)
        return self.random.normal(loc=mean, scale=stddev)

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

    def action_value(self, critic_ind):
        if len(critic_ind.shape) > 1:
            raise ValueError()

        return self.critic_weights[critic_ind].sum()

    def action_value_from_inputs(self, state, action):
        critic_state_action_ind = self._critic_tile(state, action)

        return self.action_value(critic_state_action_ind)

    def _critic_tile(self, state, action):
        state_action = np.concatenate([state, action])

        return np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self.critic_tiler.get_indices(state_action) + 1,
            ],
        )

    def _actor_tile(self, state):
        return np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self.actor_tiler.get_indices(state) + 1,
            ],
        )

    def update(self, state, action, reward, next_state, done_mask):
        critic_ind = self._critic_tile(state, action)
        actor_ind = self._actor_tile(state)
        next_actor_ind = self._actor_tile(next_state)

        # Update the critic
        for _ in range(self._n_critic_updates):
            self._update_critic(critic_ind, action, reward, next_state,
                                done_mask)

        # ######################################################3
        # GreedyAC
        # ######################################################3
        for _ in range(self._n_actor_updates):
            mu = self.get_mean(actor_ind, False)
            sigma = self.get_stddev(actor_ind, False)
            sample = self.random.normal(
                mu, sigma,
                size=(self.num_samples, *mu.shape),
            )

            # Get the action values of each sampled action
            q_values = np.zeros((self.num_samples,))
            for i in range(sample.shape[0]):
                q_values[i] = self.action_value_from_inputs(state, sample[i])

            # Sort actions by value for GreedyAC update
            sorted_ind = np.argsort(q_values, kind="stable")
            percentile = int(self.num_samples * self.rho)
            if percentile == 0:
                raise ValueError("percentile must be > 0")

            best_ind = sorted_ind[-percentile:]
            best_actions = sample[best_ind]

            # Actor update
            mean_grad = self._mean_grad(actor_ind, best_actions, True)
            stddev_grad = self._stddev_grad(actor_ind, best_actions, True)

            self.mu_weights[:, actor_ind] += (mean_grad * self.actor_lr)
            self.sigma_weights[:, actor_ind] += (stddev_grad * self.actor_lr)

            # Sampler update
            entropy_actions = np.expand_dims(
                self.random.normal(mu, sigma), axis=0,
            )
            actions = np.concatenate([best_actions, entropy_actions], axis=0)

            s_mean_grad = self._mean_grad(actor_ind, actions, False)
            s_stddev_grad = self._stddev_grad(actor_ind, actions, False)

            self.s_mu_weights[:, actor_ind] += (s_mean_grad * self.actor_lr)
            self.s_sigma_weights[:, actor_ind] += (s_stddev_grad *
                                                   self.actor_lr)

        # In order to be consistent across all children of BaseAgent, we
        # return all transitions with the shape B x N, where N is the number
        # of state, action, or reward dimensions and B is the batch size = 1
        reward = np.array([reward])

        return np.expand_dims(state, axis=0), np.expand_dims(action, axis=0), \
            np.expand_dims(reward, axis=0), np.expand_dims(next_state, axis=0)

    def _stddev_grad(self, state_ind, actions, actor=True):
        mean = self.get_mean(state_ind, actor=actor)
        stddev = self.get_stddev(state_ind, actor=actor)

        grads = ((((actions - mean) / stddev)**2) - 1)

        if not actor:
            # For the sampler, the last action in the batch should be a random
            # action to increase the probability of -- entropy regularization
            n = norm(mean, stddev)
            logprob = n.logpdf(actions[-1, :])[0]

            # ∇J = 𝔼_{I*}[∇lnπ] - α 𝔼_{π}[lnπ ∇lnπ]
            return grads[:-1, :].mean() - self.alpha * (
                grads[-1, :].item() * logprob
            )

        return grads[:-1, :].mean()

    def _mean_grad(self, state_ind, actions, actor=True):
        mean = self.get_mean(state_ind, actor=actor)
        stddev = self.get_stddev(state_ind, actor=actor)

        grads = (1 / (stddev ** 2)) * (actions - mean)

        if not actor:
            # For the sampler, the last action in the batch should be a random
            # action for entropy regularization

            n = norm(mean, stddev)
            logprob = n.logpdf(actions[-1, :])[0]

            # ∇J = 𝔼_{I*}[∇lnπ] - α 𝔼_{π}[lnπ ∇lnπ]
            return grads[:-1, :].mean() - self.alpha * (
                grads[-1, :].item() * logprob
            )

        return grads.mean()

    def _update_critic(self, critic_ind, action, reward, next_state, mask):
        # Calculate the q values of the sampled actions in the sampled states.
        # These are stored tile-coded in the critic replay buffer
        q = self.action_value(critic_ind)

        # Sample next actions for the SARSA update
        next_state_ind = self._actor_tile(next_state)
        mu = self.get_mean(next_state_ind)
        sigma = self.get_stddev(next_state_ind)
        next_action = self.random.normal(mu, sigma)

        next_critic_ind = self._critic_tile(next_state, next_action)

        # Next q values for SARSA update
        next_q = self.action_value(next_critic_ind)

        # ######################################################3
        # Critic update
        # ######################################################3
        target = reward + next_q * self.gamma * mask
        δ = target - q
        if not self._use_trace:
            self.critic_weights[critic_ind] += (self.critic_lr * δ)
        else:
            self._critic_trace *= (self.gamma * self._λ)

            if self._trace_type == "accumulating":
                self._critic_trace[critic_ind] += 1
            elif self._trace_type == "replacing":
                self._critic_trace[critic_ind] = 1
            else:
                raise ValueError("unknown trace type {self.trace_type}")

            # Update critic
            self.critic_weights += (self.critic_lr * δ * self._critic_trace)

    def reset(self):
        if self._use_trace:
            self._critic_trace = np.zeros_like(self._critic_trace)

    def eval(self):
        self.is_training = False

    def train(self):
        self.is_training = True

    def get_parameters(self):
        pass
