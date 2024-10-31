import gym
import numpy as np

class ContinuousShortCorridor(gym.Env):
    def __init__(self, seed, deviation=0.05):
        self._random = np.random.default_rng(seed)

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

        self.left_action_center = -1.0
        self.right_action_center = 1.0
        self.deviation = deviation
        self.soft_transition = True

        # 0: left, 1: middle, 2: right, 3: terminal
        self._internal_state = 0

        self._action_space = gym.spaces.Box(self._action_min, self._action_max)

    def seed(self, seed):
        self._random = np.random.default_rng(seed)

    @property
    def action_space(self):
        return self._action_space
    
    @property
    def observation_space(self):
        return self._observation_space
    
    def stochastic_transition(self, action):
        mode1 = 1.0 * np.exp(-0.5 * ((action - self.left_action_center) / self.deviation)**2)
        mode2 = 1.0 * np.exp(-0.5 * ((action - self.right_action_center) / self.deviation)**2)

        transition_prob = np.clip(mode1 + mode2, 0, 1)

        if self._random.uniform(0, 1) < transition_prob:
            _action = 1 if action > 0 else -1
        else:
            _action = 0
        return _action

    def deterministic_transition(self, action):
        left_action = [self.left_action_center - self.deviation,
                       self.left_action_center + self.deviation]
        righ_action = [self.right_action_center - self.deviation,
                       self.right_action_center + self.deviation]
        if action >= left_action[0] and action <= left_action[1]:
            _action = -1
        elif action >= righ_action[0] and action <= righ_action[1]:
            _action = 1
        else:
            _action = 0
        return _action

    def step(self, action):
        if self.soft_transition:
            _action = self.stochastic_transition(action)
        else:
            _action = self.deterministic_transition(action)

        # flip the action if the internal state is 1
        if self._internal_state == 1:
            _action = -_action

        self._internal_state = self._internal_state + _action
        self._internal_state = np.clip(self._internal_state, 0, 3)

        return self._state, self._get_reward(action), self._is_terminal(), {}

    def reset(self):
        self._state = np.array([1.])
        self._internal_state = 0
        return self._state

    def close(self):
        pass

    def _get_reward(self, action):
        reward = -1.0
        # if action < 0:
        #     reward += 0.5 * (1 - abs(action + 1))
        # else:
        #     reward += 0.5 * (1 - abs(action - 1))
        return reward

    def _is_terminal(self):
        return self._internal_state == 3
