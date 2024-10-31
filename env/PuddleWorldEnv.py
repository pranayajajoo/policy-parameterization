import gym
from gym import spaces
import math
from gym.utils import seeding
import numpy as np


class PuddleWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            seed,
            goal=[1.0, 1.0],
            goal_threshold=0.1,
            noise=0.01,
            thrust=0.05,
            continuous=True,
    ):
        self.goal = np.array(goal)
        self.goal_threshold = goal_threshold
        self.noise = noise
        self.thrust = thrust
        self.puddle1_centre = [np.array([0.1, 0.75]), np.array([0.45, 0.75])]
        self.puddle2_centre = [np.array([0.45, 0.4]), np.array([0.45, 0.8])]
        self.radius = 0.1
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,))

        self.continuous = continuous
        if continuous:
            self.high = np.ones(2) * self.thrust
            self.low = -self.high
            self.action_space = spaces.Box(self.low, self.high)
        else:
            self.action_space = spaces.Discrete(4)
            self.actions = [np.zeros(2) for i in range(4)]
            for i in range(4):
                self.actions[i][i//2] = thrust * (i % 2 * 2 - 1)

        # So the environment wrapper plays nicely with this env
        self._max_episode_steps = None

        self.seed(seed)
        self.viewer = None

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.continuous:
            action = np.clip(action, self.low, self.high)
        else:
            assert self.action_space.contains(action), \
                    "%r (%s) invalid" % (action, type(action))
            action = self.actions[action]

        self.pos += action
        self.pos += self.np_random.normal(loc=0, scale=self.noise, size=(2,))
        self.pos = np.clip(self.pos, 0.0, 1.0)

        if np.any(self.pos < 0) or np.any(self.pos > 1):
            raise ValueError("state out of bounds")

        reward = self._get_reward(self.pos)

        done = np.linalg.norm((self.pos - self.goal),
                              ord=1) < self.goal_threshold
        # if done:
        #     reward = 10000.0

        return self.pos, reward, done, {}

    def _get_reward(self, pos):
        x, y = pos

        # Check if in puddle 1
        p1x1, p1x2 = self.puddle1_centre[0][0], self.puddle1_centre[1][0]
        p1y = self.puddle1_centre[0][1]
        in_puddle1 = (
            (p1x1 <= x <= p1x2 and
                p1y - self.radius <= y <= p1y + self.radius) or
            (x < p1x1 and
                (p1x1 - x) ** 2 + (p1y - y) ** 2 <= self.radius ** 2) or
            (x > p1x2 and
                (p1x2 - x) ** 2 + (p1y - y) ** 2 <= self.radius ** 2)
        )
        # print(p1x1, x, p1x2, "|", y, p1y, in_puddle1)
        # print(p1y - self.radius <= y, y <= p1y + self.radius)

        if in_puddle1:
            y0 = p1y - self.radius
            y1 = p1y + self.radius
            min_dist = np.min([np.abs(y0 - y), np.abs(y1 - y)])
            if x < p1x1:
                dist_end1 = self.radius - math.sqrt((p1x1 - x) ** 2 +
                                                    (p1y - y) ** 2)
                min_dist = np.min([min_dist, dist_end1])
            elif x > p1x2:
                dist_end2 = self.radius - math.sqrt((p1x2 - x) ** 2 +
                                                    (p1y - y) ** 2)
                min_dist = np.min([min_dist, dist_end2])

            reward1 = -400 * min_dist

        # Check if in puddle 2
        p2y1, p2y2 = self.puddle2_centre[0][1], self.puddle2_centre[1][1]
        p2x = self.puddle2_centre[0][0]
        in_puddle2 = (
            (p2y1 <= y <= p2y2 and
                p2x - self.radius <= x <= p2x + self.radius) or
            (y > p2y2 and
                (p2y2 - y) ** 2 + (p2x - x) ** 2 <= self.radius ** 2) or
            (y < p2y1 and
                (p2y1 - y) ** 2 + (p2x - x) ** 2 <= self.radius ** 2)
        )

        if in_puddle2:
            x0 = p2x - self.radius
            x1 = p2x + self.radius
            min_dist = np.min([np.abs(x0 - x), np.abs(x1 - x)])
            if y < p2y1:
                dist_end1 = self.radius - math.sqrt((p2y1 - y) ** 2 +
                                                    (p2x - x) ** 2)
                min_dist = np.min([min_dist, dist_end1])
            elif y > p2y2:
                dist_end2 = self.radius - math.sqrt((p2y2 - y) ** 2 +
                                                    (p2x - x) ** 2)
                min_dist = np.min([min_dist, dist_end2])

            reward2 = -400 * min_dist

        if in_puddle1 and in_puddle2:
            reward = np.min([reward1, reward2])
        elif in_puddle1:
            reward = reward1
        elif in_puddle2:
            reward = reward2
        else:
            reward = 0

        return -1 + reward

    def reset(self):
        self.pos = self.observation_space.sample()
        while np.linalg.norm((self.pos - self.goal),
                             ord=1) < self.goal_threshold:
            self.pos = self.observation_space.sample()
        return self.pos

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            from gym_puddle.shapes.image import Image
            self.viewer = rendering.Viewer(screen_width, screen_height)

            import pyglet
            img_width = 100
            img_height = 100
            fformat = 'RGB'
            pixels = np.zeros((img_width, img_height, len(fformat)))
            for i in range(img_width):
                for j in range(img_height):
                    x = float(i)/img_width
                    y = float(j)/img_height
                    pixels[j, i, :] = self._get_reward(np.array([x, y]))

            pixels -= pixels.min()
            pixels *= 255./pixels.max()
            pixels = np.floor(pixels)

            img = pyglet.image.create(img_width, img_height)
            img.format = fformat
            data = [chr(int(pixel)) for pixel in pixels.flatten()]

            img.set_data(fformat, img_width * len(fformat), ''.join(data))
            bg_image = Image(img, screen_width, screen_height)
            bg_image.set_color(1.0, 1.0, 1.0)

            self.viewer.add_geom(bg_image)

            thickness = 5
            agent_polygon = rendering.FilledPolygon([(-thickness, -thickness),
                                                     (-thickness, thickness),
                                                     (thickness, thickness),
                                                     (thickness, -thickness)])
            agent_polygon.set_color(0.0, 1.0, 0.0)
            self.agenttrans = rendering.Transform()
            agent_polygon.add_attr(self.agenttrans)
            self.viewer.add_geom(agent_polygon)

        self.agenttrans.set_translation(self.pos[0]*screen_width,
                                        self.pos[1]*screen_height)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
