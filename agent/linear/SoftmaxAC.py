# Import modules
import numpy as np
from agent.baseAgent import BaseAgent
from PyFixedReps import TileCoder
import time
from scipy import special
import inspect


class SoftmaxAC(BaseAgent):
    """
    Class SoftmaxAC implements Linear-Softmax Actor-Critic with eligibility
    trace. Its Gaussian counterpart is outlined in "Model-Free
    Reinforcement Learning with Continuous Action in Practice",
    which can be found at:

    https://hal.inria.fr/hal-00764281/document

    This implementation is also based off that in the RL Book.
    """
    def __init__(self, decay, env, actor_lr, critic_lr, gamma, bins,
                 num_tilings, use_critic_trace, use_actor_trace,
                 n_actor_updates, n_critic_updates, temperature=1.0, seed=None,
                 trace_type="replacing"):
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

        # Eligibility traces
        if trace_type not in ("accumulating", "replacing"):
            raise ValueError("trace_type must be one of accumulating', " +
                             "'replacing'")
        if decay < 0:
            raise ValueError("cannot use decay < 0")
        elif decay >= 1:
            raise ValueError("cannot use decay >= 1")
        elif decay == 0:
            use_actor_trace = use_critic_trace = False
        else:
            self._λ = decay

        self._trace_type = trace_type
        self.use_actor_trace = use_actor_trace
        self.use_critic_trace = use_critic_trace
        if self.use_actor_trace:
            self._actor_trace = np.zeros((self._action_n, state_features))

        # How many times to update the actor and critic
        self._n_actor_updates = n_actor_updates
        self._n_critic_updates = n_critic_updates

        source = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        self.info = {"source": source}

    def _get_logits(self, state):
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

        logits = self._actor_weights[:, state].sum(axis=1)
        logits -= np.max(logits)  # For numerical stability
        return logits / self._τ

    def _get_probs(self, state_ind):
        if self._τ == 0:
            logits = self._actor_weights[:, state_ind].sum(axis=-1)

            # Change this to break ties randomly
            max_action = np.argmax(logits)

            probs = np.zeros(self._action_n)
            probs[max_action] = 1   # / len(max_actions)
            return probs

        logits = self._get_logits(state_ind)
        logits -= logits.max()  # Subtract max because SciPy breaks things
        pi = special.softmax(logits)
        return pi

    def sample_action(self, state, tile_coded=False):
        """
        Samples an action from the actor

        Parameters
        ----------
        state : np.array
            The state feature vector
        tile_coded : bool
            Whether or not the state feature vector is tile coded

        Returns
        -------
        np.array of float
            The action to take
        """
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

        return self._random.choice(self._action_n, p=probs)

    def _actor_grad(self, state, action):
        """
        Returns the gradient of the actor's performance in `state`
        evaluated at the action `action`

        Parameters
        ----------
        state : np.ndarray
            The state observation, not tile coded
        action : int
            The action to evaluate the gradient on
        """
        π = self._get_probs(state)
        π = np.reshape(π, (self._actor_weights.shape[0], 1))
        features = np.zeros_like(self._actor_weights)
        features[action, state] = 1

        grad = features
        grad[:, state] -= π
        return grad

    def _update_critic(self, state, action, reward, next_state, done_mask):
        raise NotImplementedError("children of SoftmaxAC should implement " +
                                  "_update_critic")

    def _update_actor(self, state, action, reward, next_state, done_mask):
        raise NotImplementedError("children of SoftmaxAC should implement " +
                                  "_update_actor")

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

        # This part of the algorithm differs from the paper
        # (https://hal.inria.fr/hal-00764281/PDF/DegrisACC2012.pdf)
        # in that the paper uses the same TD error to update the actor and the
        # critic. We use the TD error to update the critic, then recompute the
        # TD error (using the updated critic) to update the actor.
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
        """
        Resets the agent between episodes
        """
        if self.use_actor_trace:
            self._actor_trace = np.zeros_like(self._actor_trace)
        if self.use_critic_trace:
            self._critic_trace = np.zeros_like(self._critic_trace)

    def eval(self):
        """
        Sets the agent into offline evaluation mode, where the agent will not
        explore
        """
        self._is_training = False

    def train(self):
        """
        Sets the agent to online training mode, where the agent will explore
        """
        self._is_training = True

    def get_parameters(self):
        """
        Gets all learned agent parameters such that training can be resumed.

        Returns
        -------
        dict of str to array_like
            The agent's weights
        """
        pass


class SoftmaxACV(SoftmaxAC):
    """
    Class SoftmaxACV is SoftmaxAC with a state-value critic
    """
    def __init__(self, decay, env, actor_lr, critic_lr, gamma, bins,
                 num_tilings, use_critic_trace, use_actor_trace,
                 n_actor_updates, n_critic_updates,
                 temperature=1.0, seed=None, trace_type="replacing"):
        super().__init__(decay, env, critic_lr, actor_lr, gamma, bins,
                         num_tilings, use_critic_trace, use_actor_trace,
                         n_actor_updates, n_critic_updates,
                         temperature, seed, trace_type)

        # State value critic
        self._critic_weights = np.zeros(self._state_features)
        if self.use_critic_trace:
            self._critic_trace = np.zeros(self._state_features)

    def _update_critic(self, state, action, reward, next_state, done_mask):
        # Calculate TD error
        target = reward + done_mask * self._γ * \
            self._critic_weights[next_state].sum()
        estimate = self._critic_weights[state].sum()
        delta = target - estimate
        self._delta = delta

        # Critic update
        if self.use_critic_trace:
            # Update critic eligibility trace
            self._critic_trace *= (self._γ * self._λ)
            if self._trace_type == "accumulating":
                self._critic_trace[state] += 1
            elif self._trace_type == "replacing":
                self._critic_trace[state] = 1
            else:
                raise ValueError(f"unknown trace type {self._trace_type}")

            # Update critic
            self._critic_weights += (self._critic_α * delta *
                                     self._critic_trace)
        else:
            self._critic_weights[state] += (self._critic_α * delta)

        # Clear the delta which was cached from the previous actor update,
        # forcing the algorithm to re-compute the TD error when updating the
        # actor again
        self._delta_cached = False
        self._cached_delta = None

    def _update_actor(self, state, action, reward, next_state, done_mask):
        # Calculate TD error
        if self._delta_cached:
            delta = self._cached_delta
        else:
            target = reward + done_mask * self._γ * \
                self._critic_weights[next_state].sum()
            estimate = self._critic_weights[state].sum()
            delta = target - estimate

            # We know the TD error won't change when we update the actor, since
            # the critic is not being updated. So, use the cached value of the
            # TD error from the previous actor update (which will have been
            # cleared if the critic was updated).
            if not self._delta_cached:
                self._delta_cached = True
                self._cached_delta = delta

        actor_grad = self._actor_grad(state, action)
        if self.use_actor_trace:
            # Update actor eligibility traces
            self._actor_trace *= (self._γ * self._λ)
            self._actor_trace += actor_grad

            # Update actor weights
            self._actor_weights += (self._actor_α * delta * self._actor_trace)
        else:
            self._actor_weights += (self._actor_α * delta * actor_grad)


class SoftmaxACQ(SoftmaxAC):
    """
    Class SoftmaxACQ is SoftmaxAC with an action value critic
    """
    def __init__(self, decay, env, actor_lr, critic_lr, gamma, bins,
                 num_tilings, use_critic_trace, use_actor_trace,
                 n_actor_updates, n_critic_updates, critic_type,
                 temperature=1.0, seed=None, trace_type="replacing",
                 q_from_next_state=False):
        super().__init__(decay, env, critic_lr, actor_lr, gamma, bins,
                         num_tilings, use_critic_trace, use_actor_trace,
                         n_actor_updates, n_critic_updates,
                         temperature, seed, trace_type)

        # Action value critic
        self._critic_weights = np.zeros((self._action_n,
                                         self._state_features))
        if self.use_critic_trace:
            self._critic_trace = np.zeros_like(self._critic_weights)

        self._q_from_next_state = q_from_next_state

        self._critic_type = critic_type
        avail_critics = ("sarsa", "esarsa")
        if self._critic_type not in avail_critics:
            raise NotImplementedError(f"unknown critic type {critic}")

    def _update_critic(self, state, action, reward, next_state, done_mask):
        # Remove cached values for actor update
        self._delta_cached = False
        self._cached_delta = None

        # Calculate TD error
        if self._critic_type == "sarsa":
            next_action = self.sample_action(next_state, True)
            next_q = self._critic_weights[next_action, next_state].sum()
        else:  # 𝔼Sarsa
            probs = self._get_probs(next_state)
            action_values = self._critic_weights[:, next_state].sum(-1)
            next_q = probs @ action_values

        estimate = self._critic_weights[action, state].sum()
        target = reward + done_mask * self._γ * next_q
        delta = target - estimate

        # Critic update
        if self.use_critic_trace:
            # Update critic eligibility trace
            self._critic_trace *= (self._γ * self._λ)
            if self._trace_type == "accumulating":
                self._critic_trace[action, state] += 1
            elif self._trace_type == "replacing":
                self._critic_trace[action, state] = 1
            else:
                raise ValueError(f"unknown trace type {self._trace_type}")

            # Update critic
            self._critic_weights += (self._critic_α * delta *
                                     self._critic_trace)
        else:
            self._critic_weights[action, state] += (self._critic_α * delta)

    def _update_actor(self, state, action, reward, next_state, done_mask):
        if self._delta_cached:
            delta = self._cached_delta
        else:
            # Calculate the approximation of the return for the policy gradient
            if self._q_from_next_state:
                # Option (1), calculate q as q(s, a) = r + γv' =
                #                                    = r + γ𝔼[Q(S, A) ∣ S = s]
                v_next = self._critic_weights[:, next_state].sum(-1).mean()
                q = reward + self._γ * v_next * done_mask
            else:
                # Option (2) calculate q as approximated by the critic
                q = self._critic_weights[action, state].sum()

            # Calculate the baseline as v = v(S) = 𝔼[Q(A, S) | S = s]
            v = self._critic_weights[:, state].sum(-1).mean()

            delta = q - v

            # Since the critic is updated N times before the actor, there is no
            # sense in re-computing delta at each iteration; cached it instead.
            if not self._delta_cached:
                self._cached_delta = delta

        self._delta_cached = True

        actor_grad = self._actor_grad(state, action)
        if self.use_actor_trace:
            # Update actor eligibility traces
            self._actor_trace *= (self._γ * self._λ)
            self._actor_trace += actor_grad

            # Update actor weights
            trace = self._actor_α * self._actor_trace
            trace *= delta
            self._actor_weights += trace
        else:
            actor_grad *= delta
            actor_grad *= self._actor_α
            self._actor_weights += actor_grad
