# Import modules
import numpy as np


class Environment:
    """
    Class Environment is a wrapper around various environment platforms, to ensure
    logging can be done as well as to ensure that we can restrict the episode
    time steps.
    """
    def __init__(self, config, seed, strategy, env_factory, monitor=False, monitor_after=0):
        """
        Constructor

        Parameters
        ----------
        config : dict
            The environment configuration file
        seed : int
            The seed to use for all random number generators
        strategy:
            The strategy object to delegate to (e.g. GymStrategy)
        env_factory:
            The environment factory object to use; should correspond to the
            chosen strategy (e.g. if you use GymStrategy, you should
            of course also use GymEnvFactory)
        monitor : bool
            Whether or not to render the scenes as the agent learns, by
            default False
        monitor_after : int
            If monitor is True, how many timesteps should pass before
            the scene is rendered, by default 0.
        """
        self.strategy = strategy

        # Overwrite rewards and start state if necessary
        self.overwrite_rewards = config["overwrite_rewards"]
        self.rewards = config["rewards"]
        self.start_state = np.array(config["start_state"])

        self.steps = 0
        self.episodes = 0

        # Keep track of monitoring
        self.monitor = monitor
        self.steps_until_monitor = monitor_after

        # Set the gym variables
        # Ty, 2023-06-08: This does not appear to be used anywhere
        self.env_name = config["env_name"]

        self.env = env_factory.make_env(config)
        print("Seeding environment:", seed)
        # self.env.seed(seed=seed)
        self.steps_per_episode = config["steps_per_episode"]

        if "info" in dir(self.env):
            self.info = self.env.info
        else:
            self.info = {}

        self.context = EnvironmentContext(
            self.env, 
            self.steps, 
            self.episodes, 
            self.steps_until_monitor
        )

        self.override_builtin_timeout()

    @property
    def action_space(self):
        """
        Gets the action space of the environment

        Returns
        -------
        Any
            The action space according to the chosen strategy (i.e. the
            chosen environment library)
        """
        return self.strategy.action_space(self.context)

    @property
    def observation_space(self):
        """
        Gets the observation space of the environment

        Returns
        -------
        Any
            The observation space according to the chosen strategy (i.e. the
            chosen environment library)
        """
        return self.strategy.observation_space(self.context)

    def override_builtin_timeout(self):
        """
        Increases the episode steps of the wrapped environment so that this
        wrapper will timeout before the wrapped one does
        """
        self.strategy.override_builtin_timeout(self.context, self.steps_per_episode)

    def reset(self):
        """
        Resets the environment by resetting the step counter to 0 and resetting
        the wrapped environment. This function also increments the total
        episode count.

        Returns
        -------
        2-tuple of array_like, dict
            The new starting state and an info dictionary
        """
        return self.strategy.reset(self.context, self.start_state)

    def render(self):
        """
        Renders the current frame
        """
        self.strategy.render(self.context)

    def step(self, action):
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
        return self.strategy.step(
            action,
            self.context,
            self.monitor,
            self.overwrite_rewards,
            self.rewards,
            self.steps_per_episode
        )


class EnvironmentContext:
    """
    Class EnvironmentContext is a bag of environment properties that change
    over the course of an experiment.
    """
    def __init__(self, env, steps, episodes, steps_until_monitor):
        """
        Constructor

        Parameters
        ----------
        env :
            The environment object
        steps : int
            The number of steps that have passed in the current episode
        episodes : int
            The number of episodes that have passed in the current experiment
        steps_until_monitor : int
            If monitor is True, how many timesteps are left before the scene
            is rendered.
        """
        self.env = env
        self.steps = steps
        self.episodes = episodes
        self.steps_until_monitor = steps_until_monitor
