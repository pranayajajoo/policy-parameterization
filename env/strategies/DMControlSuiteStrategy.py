from numpy import concatenate
import os
os.environ['MUJOCO_GL']='osmesa'
from dm_control import suite
from dm_env.specs import Array
from env.ActionSpace import ActionSpace
from env.ObservationSpace import ObservationSpace

import numpy as np


ENV_NAMES = [
    "pendulum",
    "acrobot",
    "cartpole",
    "ball_in_cup",
    "reacher",
    "finger",
    "fish",
    "manipulator",
    "walker",
    "humanoid",
]

TASK_NAMES = [
    "swingup",
    "balance",
    "swingup_sparse",
    "balance_sparse",
    "catch",
    "easy",
    "turn_easy",
    "turn_hard",
    "upright",
    "swim",
    "bring_ball",
    "run",
    "walk",
]


class DMControlSuiteStrategy:
    """
    Class DMControlSutieStrategy implements the functionality of class 
    Environment using DeepMind Control Suite environments.
    """
    def action_space(self, context):
        """
        Gets the action space of the DM Control Suite environment

        Returns
        -------
        DMControlSuiteActionSpace
            The action space
        """
        action_space = context.env.action_spec()
        return DMControlSuiteActionSpace(
            action_space.shape,
            action_space.dtype,
            action_space.minimum,
            action_space.maximum
        )

    def observation_space(self, context):
        """
        Gets the observation space of the DM Control Suite environment

        Returns
        -------
        DMControlSuiteObservationSpace
            The observation space
        """

        obs_space = context.env.observation_spec()

        for k, v in obs_space.items():
            if len(v.shape) == 0:
                obs_space[k] = Array(shape=(1,), dtype=v)
            else:
                assert len(v.shape) == 1, "Observation space is not one-dimensional"

        # TODO: We've made two strong assumptions: 1) that each component of
        # the observation is one-dimensional (i.e. a vector); and 2) that 
        # each component of the observation is of the same type. These 
        # assumptions are true for Pendulum but may need to be addressed in
        # order to support other environments
        return DMControlSuiteObservationSpace(
            shape = (sum([v.shape[0] for v in obs_space.values()]),),
            dtype = obs_space[next(iter(obs_space))]
        )

    def override_builtin_timeout(self, context, steps_per_episode):
        """
        Increases the episode steps of the wrapped DM Control Suite
        environment so that this wrapper will timeout before the
        wrapped one does
        """
        context.env._step_limit = steps_per_episode + 10

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

        raw_state = context.env.reset().observation
        state = concatenate([v if len(v.shape) == 1 else np.array([v]) for v in raw_state.values()])

        # If the user has inputted a fixed start state, use that instead
        if start_state.shape[0] != 0:
            state = start_state
            context.env.state = state

        return state, {"orig_state": state}

    def render(self, context):
        """
        Renders the current frame
        """
        # TODO: Figure out rendering in dmcontrol
        pass 

    def step(
        self,
        action,
        context,
        monitor,
        overwrite_rewards,
        rewards,
        steps_per_episode
    ):
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

        timestep = context.env.step(action)
        raw_state = timestep.observation
        state = concatenate([v if len(v.shape) == 1 else np.array([v]) for v in raw_state.values()])
        # TODO: We assume reward is a scalar, which is true for Pendulum but
        # may not be for all environments; address accordingly
        reward = timestep.reward
        done = timestep.last()
        info = {}

        info["orig_state"] = state

        threshold = rewards.get("threshold", None)
        if threshold is not None:
            reward = np.clip(reward - threshold, 0, 1 - threshold) / (1 - threshold)

        # If the episode completes, return the goal reward
        if done:
            info["steps_exceeded"] = False
            if overwrite_rewards:
                reward = rewards["goal"]
            return state, reward, done, info

        # If the user has set rewards per timestep
        if overwrite_rewards:
            reward = rewards["timestep"]

        # If the maximum time-step was reached
        if context.steps >= steps_per_episode > 0:
            # print("Steps exceeded")
            done = True
            info["steps_exceeded"] = True

        return state, reward, done, info


class DMControlSuiteActionSpace(ActionSpace):
    def __init__(self, shape, dtype, low, high):
        super().__init__(shape, dtype, low, high)

    def sample(self):
        """
        Samples an action from the action space

        Returns
        -------
        array_like of float
            The sampled action
        """
        return np.random.uniform(
            self.low,
            self.high,
            size=self.shape
        )


class DMControlSuiteObservationSpace(ObservationSpace):
    def __init__(self, shape, dtype, low=None, high=None):
        super().__init__(shape, dtype, low, high)


class DMControlSuiteEnvFactory:
    """
    Class DMControlSuiteEnvFactory provides a method for instantiating DM
    Control Suite environments.
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
        dm_control.rl.control.Environment
            The environment to train on
        """
        env_name = config["env_name"]
        task_name = config.get("task_name", None)
        seed = config["seed"]
        env = None

        if "-" in env_name:
            assert task_name == None, "Task name should not be specified"
            env_name, task_name = env_name.split("-")

        # TODO: To support more environments, you should only have to 
        # accommodate potentially non-scalar rewards, and possibly also 
        # different observation structures. Inspect the dm_control codebase
        # to find out.
        if env_name not in ENV_NAMES or task_name not in TASK_NAMES:
            raise NotImplementedError("Environment and/or task is not " +
                                      "currently supported")
        else:
            env = suite.load(
                domain_name=env_name,
                task_name=task_name,
                task_kwargs={'random': seed}
            )

        return env
