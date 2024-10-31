# TODO: add entropy reg
# TODO: soft Q function as well!
#
# To implement GaussianACQ we will need to do the following:
# 1. Each of the Q and V subclasses should init their own critic and actor
#    tilers
# 2. Each of the Q and V subclasses should also have an actor_tile and
#    critic_tile method

# Import modules
import numpy as np
from agent.baseAgent import BaseAgent
from time import time
from PyFixedReps import TileCoder
import inspect
from scipy.stats import norm


class GaussianAC(BaseAgent):
    """
    Class GaussianAC implements Linear-Gaussian Actor-Critic with eligibility
    trace, as outlined in "Model-Free Reinforcement Learning with Continuous
    Action in Practice", which can be found at:

    https://hal.inria.fr/hal-00764281/document

    The major difference is that this algorithm uses the discounted setting
    instead of the average reward setting as used in the above paper. This
    linear actor critic support multi-dimensional actions as well.
    """
    def __init__(self, decay, actor_lr, critic_lr,
                 gamma, bins, num_tilings,
                 env, use_critic_trace, use_actor_trace,
                 n_actor_updates, n_critic_updates, scaled=False,
                 clip_stddev=1000, seed=None, trace_type="replacing"):
        """
        Constructor

        Parameters
        ----------
        decay : float
            The eligibility decay rate, lambda
        actor_lr : float
            The learning rate for the actor
        critic_lr : float
            The learning rate for the critic
        state_features : int
            The size of the state feature vectors
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
        self.batch = False

        self._n_actor_updates = n_actor_updates
        self._n_critic_updates = n_critic_updates

        if seed is None:
            seed = int(time())
        self.random = np.random.default_rng(seed=int(seed))
        self.seed = seed

        # Needed so that when evaluating offline, we don't explore
        self.is_training = True

        # Determine standard deviation clipping
        self.clip_stddev = clip_stddev > 0
        self.clip_threshold = np.log(clip_stddev)

        # Tile Coder
        self.tiler = self._init_tiler(bins, num_tilings, env)
        self.state_features = self.tiler.features() + 1

        # The weight parameters
        action_space = env.action_space
        self.action_dims = action_space.high.shape[0]
        self.sigma_weights = np.zeros((self.action_dims, self.state_features))
        self.mu_weights = np.zeros((self.action_dims, self.state_features))

        # Set learning rates and other scaling factors
        self.scaled = scaled
        self.decay = decay
        self.critic_lr = critic_lr / (num_tilings + 1)
        self.actor_lr = actor_lr / (num_tilings + 1)
        self.gamma = gamma

        # Eligibility traces
        self.use_actor_trace = use_actor_trace
        if trace_type not in ("replacing", "accumulating"):
            raise ValueError("trace_type must be one of 'accumulating', " +
                             "'replacing'")
        self.trace_type = trace_type

        if self.use_actor_trace:
            self.mu_trace = np.zeros_like(self.mu_weights)
            self.sigma_trace = np.zeros_like(self.sigma_weights)

        self.use_critic_trace = use_critic_trace

        source = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        self.info = {"source": source}

    def sample_action(self, state, n=1, tile_coded=False):
        """
        Samples an action from the actor

        Parameters
        ----------
        state : np.array
            The observation, not tile coded
        n : int
            The number of samples to get
        tile_coded : bool
            Whether the input state is tile coded or not

        Returns
        -------
        np.array of float
            The action to take
        """
        if not tile_coded:
            state = np.concatenate(
                [
                    np.zeros((1,), dtype=np.int32),
                    self.tiler.get_indices(state) + 1,
                ]
            )
        mean = self._get_mean(state)

        # If in offline evaluation mode, return the mean action
        if not self.is_training:
            return np.array(mean)

        stddev = self._get_stddev(state)

        # Sample action from a normal distribution
        action = self.random.normal(loc=mean, scale=stddev, size=n)
        return action

    def update(self, state, action, reward, next_state, done_mask):
        state = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self.tiler.get_indices(state) + 1,
            ]
        )
        next_state = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self.tiler.get_indices(next_state) + 1,
            ]
        )

        for _ in range(self._n_critic_updates):
            self._update_critic(state, action, reward, next_state, done_mask)

        for _ in range(self._n_actor_updates):
            self._update_actor(state, action, reward, next_state, done_mask)

        # In order to be consistent across all children of BaseAgent, we
        # return all transitions with the shape B x N, where N is the number
        # of state, action, or reward dimensions and B is the batch size = 1
        reward = np.array([reward])

        return np.expand_dims(state, axis=0), np.expand_dims(action, axis=0), \
            np.expand_dims(reward, axis=0), np.expand_dims(next_state, axis=0)

    def reset(self):
        if self.use_actor_trace:
            self.mu_trace = np.zeros_like(self.mu_trace)
            self.sigma_trace = np.zeros_like(self.sigma_trace)
        if self.use_critic_trace:
            self.critic_trace = np.zeros_like(self.critic_trace)

    def eval(self):
        self.is_training = False

    def train(self):
        self.is_training = True

    def get_parameters(self):
        pass

    def _update_critic(self, state, action, reward, next_state, done_mask):
        raise NotImplementedError()

    def _update_actor(self, state, action, reward, next_state, done_mask):
        raise NotImplementedError()

    def _init_tiler(self, bins: int, num_tilings: int, env):
        raise NotImplementedError()

    def _get_mean(self, state):
        """
        Gets the mean of the parameterized normal distribution

        Parameters
        ----------
        state : np.array
            The indices of the nonzero features in the one-hot encoded state
            feature vector

        Returns
        -------
        float
            The mean of the normal distribution
        """
        return self.mu_weights[:, state].sum(axis=1)

    def _get_stddev(self, state):
        """
        Gets the standard deviation of the parameterized normal distribution

        Parameters
        ----------
        state : np.array
            The indices of the nonzero features in the one-hot encoded state
            feature vector

        Returns
        -------
        float
            The standard deviation of the normal distribution
        """
        # Return un-clipped standard deviation if no clipping
        if not self.clip_stddev:
            return np.exp(self.sigma_weights[:, state].sum(axis=1))

        # Clip the standard deviation to prevent numerical overflow
        log_std = np.clip(self.sigma_weights[:, state].sum(axis=1),
                          -self.clip_threshold, self.clip_threshold)
        return np.exp(log_std)

    def _get_actor_grad(self, state, action):
        """
        Gets the gradient of the actor's parameters

        Parameters
        ----------
        state : np.array
            The indices of the nonzero features in the one-hot encoded state
            feature vector
        action : np.array of float
            The action taken

        Returns
        -------
        np.array
            The gradient vector of the actor's weights, in the form
            [grad_mu_weights^T, grad_sigma_weights^T]^T
        """
        std = self._get_stddev(state)
        mean = self._get_mean(state)

        grad_mu = np.zeros_like(self.mu_weights)
        grad_sigma = np.zeros_like(self.sigma_weights)

        if action.shape[0] != 1:
            # Repeat state along rows to match number of action dims
            n = action.shape[0]
            state = np.expand_dims(state, 0)
            state = state.repeat(n, axis=0)

            scale_mu = (1 / (std ** 2)) * (action - mean)
            scale_sigma = ((((action - mean) / std)**2) - 1)

            # Reshape scales so we can use broadcasted multiplication
            scale_mu = np.expand_dims(scale_mu, axis=1)
            scale_sigma = np.expand_dims(scale_sigma, axis=1)

        else:
            scale_mu = (1 / (std ** 2)) * (action - mean)
            scale_sigma = ((((action - mean) / std)**2) - 1)

        grad_mu[:, state] = scale_mu
        grad_sigma[:, state] = scale_sigma

        return grad_mu, grad_sigma


class GaussianACV(GaussianAC):
    """
    Class GaussianACV is GaussianAC with a state value critic
    """
    def __init__(self, decay, actor_lr, critic_lr,
                 gamma, bins, num_tilings,
                 env, use_critic_trace, use_actor_trace,
                 n_actor_updates, n_critic_updates, scaled=False,
                 clip_stddev=1000, seed=None, trace_type="replacing"):
        super().__init__(decay, actor_lr, critic_lr, gamma, bins,
                         num_tilings, env, use_critic_trace, use_actor_trace,
                         n_actor_updates, n_critic_updates, scaled,
                         clip_stddev, seed, trace_type)
        self.critic_weights = np.zeros(self.state_features)

        if self.use_critic_trace:
            self.critic_trace = np.zeros_like(self.critic_weights)

    def _init_tiler(self, bins: int, num_tilings: int, env):
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
        return TileCoder(params)

    def _update_critic(self, state, action, reward, next_state, done_mask):
        # Calculate TD error
        v = self.critic_weights[state].sum()
        next_v = self.critic_weights[next_state].sum()
        target = reward + self.gamma * next_v * done_mask
        delta = target - v

        # Critic update
        if self.use_critic_trace:
            # Update critic eligibility trace
            self.critic_trace *= (self.gamma * self.decay)
            # self.critic_trace = (self.gamma * self.decay *
            #                      self.critic_trace) + state
            if self.trace_type == "accumulating":
                self.critic_trace[state] += 1
            elif self.trace_type == "replacing":
                self.critic_trace[state] = 1
            else:
                raise ValueError("unkown trace type {self.trace_type}")
            # Update critic
            self.critic_weights += (self.critic_lr * delta * self.critic_trace)
        else:
            self.critic_weights += (self.critic_lr * delta)

        # Clear the delta which was cached from the previous actor update,
        # forcing the algorithm to re-compute the TD error when updating the
        # actor again
        self._delta_cached = False
        self._cached_delta = None

    def _update_actor(self, state, action, reward, next_state, done_mask):
        mu_grad, sigma_grad = self._get_actor_grad(state, action)

        # Calculate TD error
        if self._delta_cached:
            delta = self._cached_delta
        else:
            v = self.critic_weights[state].sum()
            next_v = self.critic_weights[next_state].sum()
            target = reward + self.gamma * next_v * done_mask
            delta = target - v

            # We know the TD error won't change when we update the actor, since
            # the critic is not being updated. So, use the cached value of the
            # TD error from the previous actor update (which will have been
            # cleared if the critic was updated).
            if not self._delta_cached:
                self._delta_cached = True
                self._cached_delta = delta

        if self.use_actor_trace:
            # Update actor eligibility traces
            self.mu_trace *= (self.gamma * self.decay)
            self.sigma_trace *= (self.gamma * self.decay)

            if self.trace_type == "accumulating":
                self.mu_trace[:, state] += mu_grad
                self.sigma_trace[:, state] += sigma_grad
            else:
                self.mu_trace[:, state] = mu_grad[:, state]
                self.sigma_trace[:, state] = sigma_grad[:, state]

            # Update actor weights
            lr = self.actor_lr
            lr *= 1 if not self.scaled else (self._get_stddev(state) ** 2)
            self.mu_weights += (lr * delta * self.mu_trace)
            self.sigma_weights += (lr * delta * self.sigma_trace)

        else:
            lr = self.actor_lr
            lr *= 1 if not self.scaled else (self._get_stddev(state) ** 2)
            self.mu_weights += (lr * delta * mu_grad)
            self.sigma_trace = (lr * delta * sigma_grad)


class GaussianACQ(GaussianAC):
    def __init__(self, decay, actor_lr, critic_lr,
                 gamma, critic_bins, critic_num_tilings,
                 actor_bins, actor_num_tilings, alpha,
                 env, use_critic_trace, use_actor_trace,
                 n_actor_updates, n_critic_updates, scaled=False,
                 clip_stddev=1000, seed=None, trace_type="replacing",
                 baseline_n=29):
        super().__init__(decay, actor_lr, critic_lr, gamma, actor_bins,
                         actor_num_tilings, env, use_critic_trace,
                         use_actor_trace, n_actor_updates, n_critic_updates,
                         scaled, clip_stddev, seed, trace_type)

        self.alpha = alpha
        self.actor_tiler = self.tiler
        self.critic_tiler = self._init_critic_tiler(
            critic_bins, critic_num_tilings, env,
        )

        self.critic_features = self.critic_tiler.features() + 1
        self.critic_weights = np.zeros(self.critic_features)

        if self.use_critic_trace:
            self.critic_trace = np.zeros_like(self.critic_weights)

        self.baseline_n = baseline_n

    def _init_tiler(self, bins: int, num_tilings: int, env):
        return self._init_actor_tiler(bins, num_tilings, env)

    def _init_actor_tiler(self, bins: int, num_tilings: int, env):
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
        return TileCoder(params)

    def _init_critic_tiler(self, bins: int, num_tilings: int, env):
        input_ranges = list(zip(env.observation_space.low,
                                env.observation_space.high))
        input_ranges.extend(list(zip(env.action_space.low,
                                     env.action_space.high)))

        critic_dims = env.observation_space.shape[0] + \
            env.action_space.shape[0]
        params = {
                    "dims": critic_dims,
                    "tiles": bins,
                    "tilings": num_tilings,
                    "input_ranges": input_ranges,
                    "scale_output": False,
                }
        return TileCoder(params)

    def update(self, state, action, reward, next_state, done_mask):
        state_tc = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self.actor_tiler.get_indices(state) + 1,
            ]
        )
        next_state_tc = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self.actor_tiler.get_indices(next_state) + 1,
            ]
        )

        for _ in range(self._n_critic_updates):
            self._update_critic(state, action, reward, next_state,
                                next_state_tc, done_mask)

        for _ in range(self._n_actor_updates):
            self._update_actor(
                state, state_tc, action, reward, next_state, next_state_tc,
                done_mask,
            )

        # In order to be consistent across all children of BaseAgent, we
        # return all transitions with the shape B x N, where N is the number
        # of state, action, or reward dimensions and B is the batch size = 1
        reward = np.array([reward])

        return np.expand_dims(state, axis=0), np.expand_dims(action, axis=0), \
            np.expand_dims(reward, axis=0), np.expand_dims(next_state, axis=0)

    def _update_critic(self, state, action, reward, next_state,
                       next_state_tc, done_mask):

        state_action_tc = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self.critic_tiler.get_indices(
                    np.concatenate((state, action)),
                ) + 1,
            ]
        )

        next_action = self.sample_action(next_state_tc, tile_coded=True)

        next_state_action_tc = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self.critic_tiler.get_indices(
                    np.concatenate((next_state, next_action)),
                ) + 1,
            ]
        )

        q = self.critic_weights[state_action_tc].sum()
        next_q = self.critic_weights[next_state_action_tc].sum()

        target = reward + self.gamma * next_q * done_mask
        delta = target - q

        # Critic update
        if self.use_critic_trace:
            # Update critic eligibility trace
            self.critic_trace *= (self.gamma * self.decay)
            if self.trace_type == "accumulating":
                self.critic_trace[state_action_tc] += 1
            elif self.trace_type == "replacing":
                self.critic_trace[state_action_tc] = 1
            else:
                raise ValueError("unknown trace type {self.trace_type}")
            # Update critic
            self.critic_weights += (self.critic_lr * delta * self.critic_trace)
        else:
            self.critic_weights[state_action_tc] += (self.critic_lr * delta)

    def _update_actor(self, state, state_tc, action, reward, next_state,
                      next_state_tc, done_mask):
        mu_grad, sigma_grad, mu_entropy_grad, sigma_entropy_grad = \
            self._get_actor_grad(state_tc, action)

        # Get the Q value
        state_action_tc = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self.critic_tiler.get_indices(
                    np.concatenate((state, action)),
                ) + 1,
            ]
        )
        q = self.critic_weights[state_action_tc].sum()

        # Compute baseline
        baseline_q_vals = []
        for _ in range(self.baseline_n):
            baseline_act = self.sample_action(state_tc, tile_coded=True)

            baseline_sa = np.concatenate(
                [
                    np.zeros((1,), dtype=np.int32),
                    self.critic_tiler.get_indices(
                        np.concatenate((state, baseline_act)),
                    ) + 1,
                ]
            )
            baseline_q_vals.append(self.critic_weights[baseline_sa].sum())

        baseline = np.mean(baseline_q_vals)
        advantage = q - baseline

        if self.use_actor_trace:
            if self.alpha != 0:
                raise ValueError("cannot use traces with entropy reg")
            # Update actor eligibility traces
            self.mu_trace *= (self.gamma * self.decay)
            self.sigma_trace *= (self.gamma * self.decay)

            if self.trace_type == "accumulating":
                self.mu_trace[:, state_tc] += mu_grad
                self.sigma_trace[:, state_tc] += sigma_grad
            else:
                self.mu_trace[:, state_tc] = mu_grad[:, state_tc]
                self.sigma_trace[:, state_tc] = \
                    sigma_grad[:, state_tc]

            # Update actor weights
            lr = self.actor_lr
            lr *= 1 if not self.scaled else (self._get_stddev(state) ** 2)
            self.mu_weights += (lr * advantage * self.mu_trace)
            self.sigma_weights += (lr * advantage * self.sigma_trace)

        else:
            lr = self.actor_lr
            lr *= 1 if not self.scaled else (self._get_stddev(state) ** 2)

            if self.use_ent_reg:
                mu_grad = advantage * mu_grad - mu_entropy_grad
                sigma_grad = advantage * sigma_grad - sigma_entropy_grad
            else:
                mu_grad = advantage * mu_grad
                sigma_grad = advantage * sigma_grad

            self.mu_weights += (lr * mu_grad)
            self.sigma_weights += (lr * sigma_grad)

    @property
    def use_ent_reg(self):
        return self.alpha != 0

    def _get_actor_grad(self, state, action):
        std = self._get_stddev(state)
        mean = self._get_mean(state)

        if self.use_ent_reg:
            n = norm(mean, std)
            logprob = n.logpdf(action)
            grad_entropy_mu = np.zeros_like(self.mu_weights)
            grad_entropy_sigma = np.zeros_like(self.sigma_weights)

        grad_mu = np.zeros_like(self.mu_weights)
        grad_sigma = np.zeros_like(self.sigma_weights)

        if action.shape[0] != 1:
            # Repeat state along rows to match number of action dims
            n = action.shape[0]
            state = np.expand_dims(state, 0)
            state = state.repeat(n, axis=0)

            scale_mu = (1 / (std ** 2)) * (action - mean)
            scale_sigma = ((((action - mean) / std)**2) - 1)

            # Reshape scales so we can use broadcasted multiplication
            scale_mu = np.expand_dims(scale_mu, axis=1)
            scale_sigma = np.expand_dims(scale_sigma, axis=1)

        else:
            scale_mu = (1 / (std ** 2)) * (action - mean)
            scale_sigma = ((((action - mean) / std)**2) - 1)

        grad_mu[:, state] = scale_mu
        grad_sigma[:, state] = scale_sigma

        if self.use_ent_reg:
            grad_entropy_mu[:, state] = logprob * scale_mu * self.alpha
            grad_entropy_sigma[:, state] = logprob * scale_sigma * self.alpha
        else:
            grad_entropy_mu = None
            grad_entropy_sigma = None

        return grad_mu, grad_sigma, grad_entropy_mu, grad_entropy_sigma
