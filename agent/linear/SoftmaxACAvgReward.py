#!/usr/bin/env python3

# Import modules
import numpy as np
from agent.baseAgent import BaseAgent
import time
from scipy import special

# Change from actor_lr to actor_lr_scale
# Make work with multi-dimensional actions


class SoftmaxAC(BaseAgent):
    """
    Class SoftmaxAC implements Linear-Softmax Actor-Critic with eligibility
    trace. Its Gaussian counterpart is outlined in "Model-Free
    Reinforcement Learning with Continuous Action in Practice",
    which can be found at:

    https://hal.inria.fr/hal-00764281/document

    This implementation is also based off that in the RL Book.
    """
    def __init__(self, decay, actor_lr_scale, critic_lr, avg_reward_lr,
                 state_features, gamma, accumulate_trace, action_space,
                 seed=None):
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
        avg_reward_lr : float
            The scale by which to learn the average reward (the weighting for
            the exponential recency-weighted moving average)
        state_features : int
            The size of the state feature vectors
        gamma : float
            The environmental discount factor
        accumulate_trace : bool
            Whether or not to accumulate the eligibility traces or not, which
            may be desirable if the task is continuing. If it is, then the
            eligibility trace vectors will be accumulated and not reset between
            "episodes" when calling the reset() method.
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

        # Set the agent's policy sampler
        if seed is None:
            seed = int(time())
        self.random = np.random.default_rng(seed=int(seed))
        self.seed = seed

        # Save whether or not the task is continuing
        self.accumulate_trace = accumulate_trace

        # Needed so that when evaluating offline, we don't explore
        self.is_training = True

        # The weight parameters
        self.actions = action_space.n
        self.size = state_features
        self.actor_weights = np.zeros((self.actions, state_features))
        self.critic_weights = np.zeros(state_features)  # State value critic

        # Set learning rates and other scaling factors
        self.decay = decay
        self.actor_lr = actor_lr_scale * critic_lr
        self.critic_lr = critic_lr
        self.avg_reward_lr = avg_reward_lr
        self.gamma = gamma

        # Eligibility traces
        self.actor_trace = np.zeros((self.actions, state_features))
        self.critic_trace = np.zeros(state_features)

        self.avg_reward = 0

    def get_logits(self, state):
        logits = self.actor_weights @ state
        logits -= np.max(logits)  # For numerical stability
        return logits

    def sample_action(self, state):
        """
        Samples an action from the actor

        Parameters
        ----------
        state : np.array
            The state feature vector

        Returns
        -------
        np.array of float
            The action to take
        """
        # If in offline evaluation mode, return the mean action
        if not self.is_training:
            raise ValueError("cannot set softmax VAC into eval mode")

        logits = self.get_logits(state)

        # Sample action from a multinomial distribution
        action = np.random.multinomial(1, special.softmax(logits))
        action = np.where(action == 1)
        if len(action) > 1:
            # Choose a random action
            action = np.random.choice(action)[0]
        else:
            action = action[0]
        return action[0]

    def actor_grad(self, state, action):
        # print('GRAD', action)
        action = action
        logits = self.get_logits(state)
        π = special.softmax(logits)

        features = np.zeros_like(self.actor_weights)
        features[action, :] = state

        expanded_state = np.expand_dims(state, axis=0)
        all_action_features = expanded_state.repeat(self.actions, axis=0)
        expanded_π = np.expand_dims(π, axis=1)
        expanded_π = expanded_π.repeat(state.shape[0], axis=1)

        grad = features - expanded_π * all_action_features
        return grad

    def update(self, state, action, reward, next_state, done_mask):
        """
        Takes a single update step

        Parameters
        ----------
        state : np.array or array_like of np.array
            The state feature vector
        action : np.array of float or array_like of np.array
            The action taken
        reward : float or array_like of float
            The reward seen by the agent after taking the action
        next_state : np.array or array_like of np.array
            The feature vector of the next state transitioned to after the
            agent took the argument action
        done_mask : bool or array_like of bool
            False if the agent reached the goal, True if the agent did not
            reach the goal yet the episode ended (e.g. max number of steps
            reached)
            Note: this parameter is not used; it is only kept so that the
            interface BaseAgent is consistent and can be used for both
            Soft Actor-Critic and Linear-Gaussian Actor-Critic
        """
        if self.gamma != 1.0:
            raise ValueError("gamma should be 1 in average reward setting " +
                    f"but got {self.gamma}")

        # Calculate TD error
        target = (reward - self.avg_reward) + done_mask * self.gamma * \
            self.critic_weights @ next_state
        estimate = self.critic_weights @ state
        delta =  target - estimate

        # Update average reward
        self.avg_reward += (self.avg_reward_lr * delta)

        # Update critic eligibility trace
        self.critic_trace = (self.gamma * self.decay * self.critic_trace) + \
            state

        # Update critic
        self.critic_weights += (self.critic_lr * delta * self.critic_trace)

        # Update actor eligibility traces
        actor_grad = self.actor_grad(state, action)
        self.actor_trace = (self.gamma * self.decay * self.actor_trace) + \
            actor_grad

        # Update actor weights
        self.actor_weights += (self.actor_lr * delta * self.actor_trace)

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
        if self.accumulate_trace:
            return
        self.actor_trace = np.zeros_like(self.actor_trace)
        self.critic_trace = np.zeros(self.size)

    def eval(self):
        """
        Sets the agent into offline evaluation mode, where the agent will not
        explore
        """
        self.is_training = False

    def train(self):
        """
        Sets the agent to online training mode, where the agent will explore
        """
        self.is_training = True

    def get_parameters(self):
        """
        Gets all learned agent parameters such that training can be resumed.

        Gets all parameters of the agent such that, if given the
        hyperparameters of the agent, training is resumable from this exact
        point. This include the learned average reward, the learned entropy,
        and other such learned values if applicable. This does not only apply
        to the weights of the agent, but *all* values that have been learned
        or calculated during training such that, given these values, training
        can be resumed from this exact point.

        For example, in the SoftmaxAC class, we must save not only the actor
        and critic weights, but also the accumulated eligibility traces.

        Returns
        -------
        dict of str to array_like
            The agent's weights
        """
        return {"actor_weights": self.actor_weights,
                "critic_weights": self.critic_weights,
                "actor_trace": self.actor_trace,
                "critic_trace": self.critic_trace,
                "avg_reward": self.avg_reward}


if __name__ == "__main__":
    a = SoftmaxAC(0.9, 0.1, 0.1, 0.5, 3, False)
    print(a.actor_weights, a.critic_weights)
    state = np.array([1, 2, 1])
    action = a.sample_action(state)
    a.update(state, action, 1, np.array([1, 2, 2]), 0.9)
    print(a.actor_weights, a.critic_weights)
    state = np.array([1, 2, 2])
    action = a.sample_action(state)
    a.update(state, action, 1, np.array([3, 1, 2]), 0.9)
    print(a.actor_weights, a.critic_weights)
