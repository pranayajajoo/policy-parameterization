#!/usr/bin/env python3

# Import modules
import numpy as np
from agent.baseAgent import BaseAgent
from PyFixedReps import TileCoder
import time
import warnings
import inspect


# This class has been tested to match the textbook figures in chapter 12. Once
# those experiments were run, no other changes were made.
# Experiment:
# Mountain Car
# epsilon = 0
# 10,000 step cutoffs
# 8 8x8 tilings
class Sarsa(BaseAgent):
    def __init__(self, decay, lr, gamma, epsilon,
                 action_space, bins, num_tilings, env, seed=None,
                 trace_type="replacing", policy_type="εgreedy",
                 include_bias=True):
        super().__init__()
        self.batch = False

        # Set the agent's policy sampler
        if seed is None:
            seed = int(time())
        self.random = np.random.default_rng(seed=int(seed))
        self.seed = seed

        # Needed so that when evaluating offline, we don't explore
        self.is_training = True

        # Tile Coder
        self.include_bias = include_bias
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
        self.tiler = TileCoder(params)
        state_features = self.tiler.features() + self.include_bias

        # The weight parameters
        self.actions = action_space.n
        self.weights = np.zeros((self.actions, state_features))

        # Set learning rates and other scaling factors
        if decay < 0.0:
            raise ValueError("cannot have trace decay rate < 0")
        self.decay = decay
        self.lr = lr / (num_tilings + self.include_bias)
        self.gamma = gamma
        self.epsilon = epsilon
        print(self.lr)

        if policy_type not in ("εgreedy"):
            raise ValueError("policy_type must be one of 'εgreedy'")
        self.policy_type = policy_type

        # Eligibility traces
        if trace_type not in ("accumulating", "replacing"):
            raise ValueError("trace_type must be one of 'accumulating', " +
                             "'replacing'")
        self.use_trace = self.decay > 0.0
        if self.use_trace:
            self.trace = np.zeros_like(self.weights)
            self.trace_type = trace_type

        # Keep track of the states and actions used in the SARSA update for
        # error checking
        self.sarsa_state = None
        self.sarsa_action = None
        self.first_call = True

        source = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        self.info = {"source": source}

    def sample_action(self, state):
        """
        Samples an action from the actor

        Parameters
        ----------
        state : np.array
            The state feature vector, not one hot encoded

        Returns
        -------
        np.array of float
            The action to take
        """
        if self.first_call:
            self.first_call = False
            return self._sample_action(state)
        if np.any(state != self.sarsa_state) and self.is_training:
            warnings.warn("Warning: input state was not used as " +
                          "the next state in SARSA update to select the" +
                          "next action. Sampling a new action.")
            return self._sample_action(state)
        else:
            return self.sarsa_action

    def _sample_action(self, state):
        """
        Samples an action from the actor

        Parameters
        ----------
        state : np.array
            The state feature vector, not one hot encoded

        Returns
        -------
        np.array of float
            The action to take
        """
        # Take random action with probability ε and only if in training mode
        if self.policy_type == "εgreedy" and self.epsilon != 0 and \
           self.is_training:
            if self.random.uniform() < self.epsilon:
                action = self.random.choice(self.actions)
                return action

        state = self._tiler_indices(state)
        action_vals = self.weights[:, state].sum(axis=1)

        if self.policy_type == "εgreedy":
            # Choose maximum action
            max_actions = np.where(action_vals == np.max(action_vals))[0]
            if len(max_actions) > 1:
                return max_actions[self.random.choice(len(max_actions))]
            else:
                return max_actions[0]

        else:
            raise ValueError(f"unknown policy type {self.policy_type}")

    def _tiler_indices(self, state):
        if self.include_bias:
            return np.concatenate(
                [
                    np.zeros((1,), dtype=np.int32),
                    self.tiler.get_indices(state) + 1,
                ]
            )

        return self.tiler.get_indices(state)

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

        # state = np.concatenate(
        #     [
        #         np.zeros((1,), dtype=np.int32),
        #         self.tiler.get_indices(state) + 1,
        #     ]
        # )
        state = self._tiler_indices(state)

        δ = reward
        δ -= self.weights[action, state].sum()

        # Update the trace
        if self.use_trace:
            if self.trace_type == "accumulating":
                self.trace[action, state] += 1
            elif self.trace_type == "replacing":
                self.trace[action, state] = 1
            else:
                raise ValueError(f"unknown trace type {self.trace_type}")

        # Adjust δ if we are in an intra-episode timestep
        episode_done = not done_mask
        if not episode_done:
            self.sarsa_action = next_action = self._sample_action(next_state)
            self.sarsa_state = next_state

            next_state = self._tiler_indices(next_state)

            δ += (self.gamma * self.weights[next_action, next_state].sum())

        # Update the weights
        if self.use_trace:
            self.weights += (self.lr * δ * self.trace)

            # Decay the trace
            self.trace *= (self.decay * self.gamma)
        else:
            self.weights[action, state] += (self.lr * δ)

        return

    def reset(self):
        """
        Resets the agent between episodes
        """
        self.trace = np.zeros_like(self.weights)
        self.first_call = True
        self.sarasa_action = self.sarsa_state = None

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

        For example, in the ESarsa class, we must save not only the actor
        and critic weights, but also the accumulated eligibility traces.

        Returns
        -------
        dict of str to array_like
            The agent's weights
        """
        pass
