# Import modules
import gym
import numpy as np
import torch


class Bimodal(gym.Env):
    def __init__(self, seed, reward_variance=False, stddev=0.05, center=1.0):
        self._random = np.random.default_rng(seed)

        self._reward_variance = reward_variance
        self._stddev = stddev
        self._center = center

        self._state_dim = 1
        self._state_range = np.array([0.])
        self._state_min = np.array([0.])
        self._state_max = np.array([1.])
        self._state = np.array([1.])

        self._observation_space = gym.spaces.Box(self._state_min,
                                                 self._state_max)

        self._action_dim = 1
        self._action_range = np.array([4.])
        self._action_min = np.array([-2.])
        self._action_max = np.array([2.])

        self._action_space = gym.spaces.Box(self._action_min, self._action_max)

    def seed(self, seed):
        self._random = np.random.default_rng(seed)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def step(self, action):
        self._state = self._state + action  # Terminal state
        return self._state, self.reward(action), True, {}

    def reward(self, action):
        # #####################
        # Calculate the reward
        # #####################
        maxima1 = - self._center
        maxima2 = self._center
        stddev = self._stddev

        # Reward function
        if isinstance(action, torch.Tensor):
            exp_func = torch.exp
        else:
            exp_func = np.exp
        modal1 = 1.5 * exp_func(-0.5 * ((action - maxima1) / stddev)**2)
        modal2 = 1.5 * exp_func(-0.5 * ((action - maxima2) / stddev)**2)

        # Add some random noise to the reward with mean noise 0.
        # Use more variance at the lower mode so that it looks better,
        # but in expectation is worse.
        reward_var = 0
        if self._reward_variance:
            reward_var = self._random.normal(0.0, 0.5, a.shape[0])

        reward = modal1 + modal2 + reward_var
        return reward

    def reset(self):
        self._state = np.array([1.])
        return self._state

    def close(self):
        pass


class AsymmetricBimodal(Bimodal):

    def __init__(self, seed, reward_variance=False, stddev=0.05,
                 suboptimal_reward=0.75, center=1.0):
        super().__init__(seed, reward_variance, stddev, center)
        self.suboptimal_reward = suboptimal_reward

    def reward(self, action):
        # #####################
        # Calculate the reward
        # #####################
        maxima1 = - self._center
        maxima2 = self._center
        stddev = self._stddev

        # Reward function
        if isinstance(action, torch.Tensor):
            exp_func = torch.exp
        else:
            exp_func = np.exp
        modal1 = self.suboptimal_reward * exp_func(-0.5 * ((action - maxima1) / stddev)**2)
        modal2 = 1.5 * exp_func(-0.5 * ((action - maxima2) / stddev)**2)

        # Add some random noise to the reward with mean noise 0.
        # Use more variance at the lower mode so that it looks better,
        # but in expectation is worse.
        reward_var = 0
        if self._reward_variance:
            reward_var = self._random.normal(0.0, 0.5, a.shape[0])

        reward = modal1 + modal2 + reward_var
        return reward


class StochasticBimodal(Bimodal):

    def step(self, action):
        self._state = self._state + action  # Terminal state
        return self._state, self.reward(action), True, {}

    def reward(self, action):
        # #####################
        # Calculate the reward
        # #####################
        maxima1 = - self._center
        maxima2 = self._center
        stddev = 0.2

        # Reward function, the modes are changed to have different heights
        if isinstance(action, torch.Tensor):
            exp_func = torch.exp
        else:
            exp_func = np.exp
        modal1 = 1 * exp_func(-0.5 * ((action - maxima1) / stddev)**2)
        modal2 = 2 * exp_func(-0.5 * ((action - maxima2) / stddev)**2)

        # Select a random mode
        mode = self._random.choice([1, 2])
        if mode == 1:
            reward = modal1
        else:
            reward = modal2

        # Add some random noise to the reward with mean noise 0.
        reward_var = 0
        if self._reward_variance:
            reward_var = self._random.normal(0.0, 0.5)
            reward += reward_var

        return reward
