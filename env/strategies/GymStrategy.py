import gym
from gym.spaces import Discrete
from env.ActionSpace import ActionSpace
from env.ObservationSpace import ObservationSpace
from env.PuddleWorldEnv import PuddleWorldEnv
from env.PendulumEnv import PendulumEnv
from env.Acrobot import AcrobotEnv
from env.Gridworld import GridworldEnv
from env.XYGridworld import GridworldEnv as XYGridworldEnv
from env.IndexGridworld import GridworldEnv as IndexGridworldEnv
from env.Bimodal import Bimodal, StochasticBimodal, AsymmetricBimodal
from env.CartpoleEnv import ContinuousCartPoleEnv
from env.ContinuousShortCorridor import ContinuousShortCorridor
from env.SparsePendulumEnv import SparsePendulumEnv
from env.DenseMountainCar import DenseContinuousMountainCarEnv
from env.DenseAcrobot import DenseAcrobotEnv
import env.MinAtar as MinAtar


class GymStrategy:
    """
    Class GymStrategy implements the functionality of class Environment using 
    OpenAI Gym environments.
    """
    def action_space(self, context):
        """
        Gets the action space of the Gym environment

        Returns
        -------
        GymActionSpace
            The action space
        """
        action_space = context.env.action_space
        
        if isinstance(action_space, Discrete):
            return action_space

        return GymActionSpace(
            action_space.shape,
            action_space.dtype,
            action_space.low,
            action_space.high,
            context.env.action_space.sample
        )

    def observation_space(self, context):
        """
        Gets the observation space of the Gym environment

        Returns
        -------
        GymObservationSpace
            The observation space
        """
        obs_space = context.env.observation_space
        return GymObservationSpace(
            obs_space.shape,
            obs_space.dtype,
            obs_space.low,
            obs_space.high
        )

    def override_builtin_timeout(self, context, steps_per_episode):
        """
        Increases the episode steps of the wrapped OpenAI gym environment so
        that this wrapper will timeout before the OpenAI gym one does
        """
        context.env._max_episode_steps = steps_per_episode + 10

    def reset(self, context, start_state):
        """
        Resets the environment by resetting the step counter to 0 and resetting
        the wrapped environment. This function also increments the total
        episode count.

        Returns
        -------
        2-tuple of array_like, dict
            The new starting state and an info dictionary
        """
        context.steps = 0
        context.episodes += 1

        state = context.env.reset()

        # If the user has inputted a fixed start state, use that instead
        if start_state.shape[0] != 0:
            state = start_state
            context.env.state = state

        return state, {"orig_state": state}

    def render(self, context):
        """
        Renders the current frame
        """
        context.env.render()

    def step(self, action, context, monitor, overwrite_rewards, rewards, steps_per_episode):
        """
        Takes a single environmental step

        Parameters
        ----------
        action : array_like of float
            The action array. The number of elements in this array should be
            the same as the action dimension.

        Returns
        -------
        float, array_like of float, bool, dict
            The reward and next state as well as a flag specifying if the
            current episode has been completed and an info dictionary
        """
        if monitor and context.steps_until_monitor < 0:
            self.render()

        context.steps += 1
        context.steps_until_monitor -= (1 if context.steps_until_monitor >= 0 else 0)

        state, reward, done, info = context.env.step(action)
        info["orig_state"] = state

        # If the episode completes, return the goal reward
        if done:
            info["steps_exceeded"] = False
            if overwrite_rewards:
                reward = rewards["goal"]
            if "shift" in rewards:
                gamma = rewards["gamma"]
                reward += rewards["shift"] * \
                    (1 - gamma**(steps_per_episode - context.steps + 1)) / (1 - gamma)
            return state, reward, done, info

        # If the user has set rewards per timestep
        if overwrite_rewards:
            reward = rewards["timestep"]
        if "shift" in rewards:
            reward += rewards["shift"]

        # If the maximum time-step was reached
        if context.steps >= steps_per_episode > 0:
            # print("Steps exceeded")
            done = True
            info["steps_exceeded"] = True

        return state, reward, done, info


class GymActionSpace(ActionSpace):
    def __init__(self, shape, dtype, low, high, sample_func):
        super().__init__(shape, dtype, low, high)
        self.sample_func = sample_func

    def sample(self):
        return self.sample_func()


class GymObservationSpace(ObservationSpace):
    def __init__(self, shape, dtype, low, high):
        super().__init__(shape, dtype, low, high)


class GymEnvFactory:
    """
    Class GymEnvFactory provides a method for instantiating OpenAI Gym 
    environments.
    """
    def make_env(self, config):
        """
        Instantiates and returns an environment given an environment name.

        Parameters
        ----------
        config : dict
            The environment config

        Returns
        -------
        gym.Env
            The environment to train on
        """
        name = config["env_name"]
        seed = config["seed"]
        env = None

        if name == "Pendulum-v0":
            env = PendulumEnv(seed=seed, continuous_action=config["continuous"])

        elif name == "SparsePendulum-v1":
            g = config.get("g", 10.0)
            tolerance = config.get("tolerance", 1.0)
            sparsity_factor = config.get("sparsity_factor", None)
            env = SparsePendulumEnv(g, tolerance, sparsity_factor)

        elif name == "DenseMountainCarContinuous-v1":
            goal_velocity = config.get("goal_velocity", 0)
            episodic = config.get("episodic", False)
            env = DenseContinuousMountainCarEnv(goal_velocity, episodic)

        elif name == "DenseAcrobot-v1":
            env = DenseAcrobotEnv(seed=seed, continuous_action=config["continuous"])

        elif name == "Bimodal" or name == "Bimodal":
            reward_variance = config.get("reward_variance", True)
            stddev = config.get("stddev", 0.05)
            center = config.get("center", 1.0)
            env = Bimodal(seed, reward_variance, stddev, center)

        elif name == "AsymmetricBimodal":
            reward_variance = config.get("reward_variance", True)
            stddev = config.get("stddev", 0.05)
            suboptimal_reward = config.get("suboptimal_reward", 0.75)
            center = config.get("center", 1.0)
            env = AsymmetricBimodal(seed, reward_variance, stddev,
                                    suboptimal_reward, center)

        elif name == "StochasticBimodal":
            reward_variance = config.get("reward_variance", True)
            env = StochasticBimodal(seed, reward_variance)

        elif name == "ContinuousShortCorridor":
            deviation = config.get("deviation", 0.05)
            env = ContinuousShortCorridor(seed, deviation)

        elif name == "ContinuousCartpole-v0":
            env = ContinuousCartPoleEnv()

        elif name == "IndexGridworld":
            env = IndexGridworldEnv(config["rows"], config["cols"])
            env.seed(seed)

        elif name == "XYGridworld":
            env = XYGridworldEnv(config["rows"], config["cols"])
            env.seed(seed)

        elif name == "Gridworld":
            env = GridworldEnv(config["rows"], config["cols"])
            env.seed(seed)

        elif name == "PuddleWorld-v1":
            env = PuddleWorldEnv(continuous=config["continuous"], seed=seed)

        elif name == "Acrobot-v1":
            env = AcrobotEnv(seed=seed, continuous_action=config["continuous"])

        elif name == "ContinuousGridWorld":
            env = ContinuousGridWorld.GridWorld()

        elif "minatar" in name.lower():
            if "/" in name:
                raise ValueError(f"specify environment as MinAtar{name} rather " +
                                 "than MinAtar/{name}")
            minimal_actions = config.get("use_minimal_action_set", True)
            stripped_name = name[7:].lower()  # Strip off "MinAtar"
            env = MinAtar.GymEnv(
                stripped_name,
                use_minimal_action_set=minimal_actions,
            )

        else:
            ctrl_cost_weight = config.get("ctrl_cost_weight", None)
            if ctrl_cost_weight is not None:
                env = gym.make(name, ctrl_cost_weight=ctrl_cost_weight).env
            else:
                env = gym.make(name).env
            env.seed(seed)

        print(config)
        if "jse_tile_coding" in config and config["use_tile_coding"]:
            raise NotImplementedError("tile coding of environments has been " +
                                      "removed")

        return env
