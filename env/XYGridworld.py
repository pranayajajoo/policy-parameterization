from gym.spaces import Box, Discrete
import numpy as np

class GridworldEnv:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        self.action_space = Discrete(4)
        self.observation_space = Box(low=np.zeros((2,)),
                high=np.array([self.cols, self.rows]))

        self.current_pos = 0

    def reset(self):
        self.current_pos = 0
        return self.obs()

    def obs(self):
        y = self.current_pos // self.cols
        x = self.current_pos - (y * self.cols)
        return np.array([x, y])

    def seed(self, seed):
        pass

    def step(self, action):
        current_row = self.current_pos // self.cols
        current_col = self.current_pos - (current_row * self.cols)

        if action == 0 and current_col > 0:
            self.current_pos -= 1

        if action == 1 and current_col < self.cols-1:
            self.current_pos += 1

        if action == 2 and current_row > 0:
            self.current_pos -= self.cols

        elif action == 3 and current_row < self.rows-1:
            self.current_pos += self.cols

        done = self.done()
        reward = 0.0 if done else -1.0

        return self.obs(), reward, done, {}

    def done(self):
        return self.current_pos == self.rows * self.cols - 1

    def __repr__(self):
        state = self.obs()
        state = state.reshape((self.rows, self.cols))
        return str(state)


