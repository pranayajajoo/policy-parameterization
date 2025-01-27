# Import modules
import os
import random
import torch
import numpy as np
from glob import glob
# from env.tile_coder import TileCoding
import pickle
from tqdm import tqdm
from copy import deepcopy
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from scipy import signal as signal
import scipy.stats
try:
    import runs
except ModuleNotFoundError:
    import utils.runs


def create_agent(agent, config):
    """
    Creates an agent given the agent name and configuration dictionary

    Parameters
    ----------
    agent : str
        The name of the agent, one of 'linearAC' or 'SAC'
    config : dict
        The agent configuration dictionary

    Returns
    -------
    baseAgent.BaseAgent
        The agent to train
    """
    # Random agent
    if agent.lower() == "random":
        from agent.Random import Random
        return Random(config["action_space"], config["seed"])

    # FKL
    if agent.lower() == "fkl":
        if "activation" in config:
            activation = config["activation"]
        else:
            activation = "relu"

        # Vanilla Actor Critic using FKL
        from agent.nonlinear.FKL import FKL
        return FKL(
            num_inputs=config["feature_size"],
            action_space=config["action_space"],
            gamma=config["gamma"], tau=config["tau"],
            alpha=config["alpha"], policy=config["policy_type"],
            target_update_interval=config["target_update_interval"],
            critic_lr=config["critic_lr"],
            actor_lr_scale=config["actor_lr_scale"],
            actor_hidden_dim=config["hidden_dim"],
            critic_hidden_dim=config["hidden_dim"],
            replay_capacity=config["replay_capacity"],
            seed=config["seed"], batch_size=config["batch_size"],
            cuda=config["cuda"], clip_stddev=config["clip_stddev"],
            init=config["weight_init"], betas=config["betas"],
            num_samples=config["num_samples"], activation="relu",
            env=config["env"],
        )

    # Soft Actor-Critic
    if agent.lower() == "SAC".lower():
        from agent.nonlinear.SAC import SAC
        return SAC(
            clip_actions=config["clip_actions"],
            baseline_actions=config["reparam_baseline"][1],
            reparameterized=config["reparam_baseline"][0],
            gamma=config["gamma"],
            tau=config["tau"],
            alpha=config["alpha"],
            policy=config["policy_type"],
            target_update_interval=config["target_update_interval"],
            critic_lr=config["critic_lr"],
            actor_lr_scale=config["actor_lr_scale"],
            alpha_lr=config["alpha_lr"],
            actor_hidden_dim=config["hidden_dim"],
            critic_hidden_dim=config["hidden_dim"],
            replay_capacity=config["replay_capacity"],
            seed=config["seed"],
            batch_size=config["batch_size"],
            automatic_entropy_tuning=config["automatic_entropy_tuning"],
            cuda=config["cuda"],
            clip_stddev=config.get("clip_stddev", 1000),
            clip_min=config.get("clip_min", 1e-6),
            clip_max=config.get("clip_max", 1e6),
            epsilon=config.get("epsilon", 1.0),
            num_components=config.get("num_components", 1),
            share_std=config.get("share_std", False),
            temperature=config.get("temperature", 0.1),
            hard=config.get("hard", False),
            impl=config.get("impl", "default"),
            eps=config.get("eps", 1e-20),
            latent_dim=config.get("latent_dim", 2),
            lmbda=config.get("lmbda", -1),
            eta=config.get("eta", 1.0),
            repulsive_coef=config.get("repulsive_coef", 0.0),
            init=config["weight_init"],
            betas=config["betas"],
            activation=config.get("activation", "relu"),
            env=config["env"],
            soft_q=config["soft_q"],
            double_q=config["double_q"],
            num_samples=config["num_samples"],
            uniform_exploration_steps=config["uniform_exploration_steps"],
            steps_before_learning=config["steps_before_learning"],
            use_true_q=config.get("use_true_q", False),
            log_actions_every=config.get("log_actions_every", 10000000),
            n_actions_logged=config.get("n_actions_logged", 1000),
            record_current_state=config.get("record_current_state", False),
            record_grad_norm=config.get("record_grad_norm", False),
            record_entropy=config.get("record_entropy", False),
            record_params=config.get("record_params", False),
            record_values=config.get("record_values", False),
            record_mixture_stat=config.get("record_mixture_stat", False),
            record_eval_state=config.get("record_eval_state", None),
            n_states_logged=config.get("n_states_logged", 1),
            state_path=config.get("state_path", None),
        )
    
    ### PJ: Epsgreedy ###
    if agent.lower() == "epsgreedy".lower():
        from agent.nonlinear.epsgreedy import EpsGreedyAgent as epsgreedy
        return epsgreedy(
            clip_actions=config["clip_actions"],
            baseline_actions=config["reparam_baseline"][1],
            reparameterized=config["reparam_baseline"][0],
            gamma=config["gamma"],
            tau=config["tau"],
            policy=config["policy_type"],
            target_update_interval=config["target_update_interval"],
            critic_lr=config["critic_lr"],
            actor_lr_scale=config["actor_lr_scale"],
            actor_hidden_dim=config["hidden_dim"],
            critic_hidden_dim=config["hidden_dim"],
            replay_capacity=config["replay_capacity"],
            seed=config["seed"],
            batch_size=config["batch_size"],
            cuda=config["cuda"],
            # clip_min=config.get("clip_min", 1e-6),
            # clip_max=config.get("clip_max", 1e6),
            # num_components=config.get("num_components", 1),
            # share_std=config.get("share_std", False),
            # temperature=config.get("temperature", 0.1),
            # hard=config.get("hard", False),
            # impl=config.get("impl", "default"),
            # eps=config.get("eps", 1e-20),
            # latent_dim=config.get("latent_dim", 2),
            # lmbda=config.get("lmbda", -1),
            # eta=config.get("eta", 1.0),
            # repulsive_coef=config.get("repulsive_coef", 0.0),
            init=config["weight_init"],
            activation=config.get("activation", "relu"),
            env=config["env"],
            # double_q=config["double_q"],
            # num_samples=config["num_samples"],
            uniform_exploration_steps=config["uniform_exploration_steps"],
            steps_before_learning=config["steps_before_learning"],
            # use_true_q=config.get("use_true_q", False),
            # log_actions_every=config.get("log_actions_every", 10000000),
            # n_actions_logged=config.get("n_actions_logged", 1000),
            # record_current_state=config.get("record_current_state", False),
            # record_grad_norm=config.get("record_grad_norm", False),
            # record_entropy=config.get("record_entropy", False),
            # record_params=config.get("record_params", False),
            # record_values=config.get("record_values", False),
            # record_mixture_stat=config.get("record_mixture_stat", False),
            # record_eval_state=config.get("record_eval_state", None),
            # n_states_logged=config.get("n_states_logged", 1),
            # state_path=config.get("state_path", None),
            ### PJ: adding epsilon hyperparam for epsgreedy method
            epsilon=config["epsilon"],
            epsilon_decay=config["epsilon_decay"],
            epsilon_min=config["epsilon_min"]
        )
    
        ### PJ: Epsgreedy SCIPY MINIMIZER ###
    if agent.lower() == "epsgreedy_optimizer".lower():
        from agent.nonlinear.epsgreedy_scipyoptimizer import EpsGreedyAgent as epsgreedy
        return epsgreedy(
            clip_actions=config["clip_actions"],
            baseline_actions=config["reparam_baseline"][1],
            reparameterized=config["reparam_baseline"][0],
            gamma=config["gamma"],
            tau=config["tau"],
            policy=config["policy_type"],
            target_update_interval=config["target_update_interval"],
            critic_lr=config["critic_lr"],
            actor_lr_scale=config["actor_lr_scale"],
            actor_hidden_dim=config["hidden_dim"],
            critic_hidden_dim=config["hidden_dim"],
            replay_capacity=config["replay_capacity"],
            seed=config["seed"],
            batch_size=config["batch_size"],
            cuda=config["cuda"],
            # clip_min=config.get("clip_min", 1e-6),
            # clip_max=config.get("clip_max", 1e6),
            # num_components=config.get("num_components", 1),
            # share_std=config.get("share_std", False),
            # temperature=config.get("temperature", 0.1),
            # hard=config.get("hard", False),
            # impl=config.get("impl", "default"),
            # eps=config.get("eps", 1e-20),
            # latent_dim=config.get("latent_dim", 2),
            # lmbda=config.get("lmbda", -1),
            # eta=config.get("eta", 1.0),
            # repulsive_coef=config.get("repulsive_coef", 0.0),
            init=config["weight_init"],
            activation=config.get("activation", "relu"),
            env=config["env"],
            # double_q=config["double_q"],
            # num_samples=config["num_samples"],
            uniform_exploration_steps=config["uniform_exploration_steps"],
            steps_before_learning=config["steps_before_learning"],
            # use_true_q=config.get("use_true_q", False),
            # log_actions_every=config.get("log_actions_every", 10000000),
            # n_actions_logged=config.get("n_actions_logged", 1000),
            # record_current_state=config.get("record_current_state", False),
            # record_grad_norm=config.get("record_grad_norm", False),
            # record_entropy=config.get("record_entropy", False),
            # record_params=config.get("record_params", False),
            # record_values=config.get("record_values", False),
            # record_mixture_stat=config.get("record_mixture_stat", False),
            # record_eval_state=config.get("record_eval_state", None),
            # n_states_logged=config.get("n_states_logged", 1),
            # state_path=config.get("state_path", None),
            ### PJ: adding epsilon hyperparam for epsgreedy method
            epsilon=config["epsilon"],
            epsilon_decay=config["epsilon_decay"]
        )
    
    ### PJ: Epsgreedy with SGD optimizer ###
    if agent.lower() == "epsgreedy_SGD".lower():
        from agent.nonlinear.epsgreedy_SGD import EpsGreedyAgent as epsgreedy
        return epsgreedy(
            clip_actions=config["clip_actions"],
            baseline_actions=config["reparam_baseline"][1],
            reparameterized=config["reparam_baseline"][0],
            gamma=config["gamma"],
            tau=config["tau"],
            policy=config["policy_type"],
            target_update_interval=config["target_update_interval"],
            critic_lr=config["critic_lr"],
            actor_lr_scale=config["actor_lr_scale"],
            actor_hidden_dim=config["hidden_dim"],
            critic_hidden_dim=config["hidden_dim"],
            replay_capacity=config["replay_capacity"],
            seed=config["seed"],
            batch_size=config["batch_size"],
            cuda=config["cuda"],
            # clip_min=config.get("clip_min", 1e-6),
            # clip_max=config.get("clip_max", 1e6),
            # num_components=config.get("num_components", 1),
            # share_std=config.get("share_std", False),
            # temperature=config.get("temperature", 0.1),
            # hard=config.get("hard", False),
            # impl=config.get("impl", "default"),
            # eps=config.get("eps", 1e-20),
            # latent_dim=config.get("latent_dim", 2),
            # lmbda=config.get("lmbda", -1),
            # eta=config.get("eta", 1.0),
            # repulsive_coef=config.get("repulsive_coef", 0.0),
            init=config["weight_init"],
            activation=config.get("activation", "relu"),
            env=config["env"],
            # double_q=config["double_q"],
            # num_samples=config["num_samples"],
            uniform_exploration_steps=config["uniform_exploration_steps"],
            steps_before_learning=config["steps_before_learning"],
            # use_true_q=config.get("use_true_q", False),
            # log_actions_every=config.get("log_actions_every", 10000000),
            # n_actions_logged=config.get("n_actions_logged", 1000),
            # record_current_state=config.get("record_current_state", False),
            # record_grad_norm=config.get("record_grad_norm", False),
            # record_entropy=config.get("record_entropy", False),
            # record_params=config.get("record_params", False),
            # record_values=config.get("record_values", False),
            # record_mixture_stat=config.get("record_mixture_stat", False),
            # record_eval_state=config.get("record_eval_state", None),
            # n_states_logged=config.get("n_states_logged", 1),
            # state_path=config.get("state_path", None),
            ### PJ: adding epsilon hyperparam for epsgreedy method
            epsilon=config["epsilon"],
            epsilon_decay=config["epsilon_decay"],
            epsilon_min=config["epsilon_min"]
        )

    # GreedyAC double ρ
    if agent.lower() == "GreedyACNoEntReg".lower():
        if "activation" in config:
            activation = config["activation"]
        else:
            activation = "relu"

        from agent.nonlinear.GreedyACNoEntReg import GreedyAC

        return GreedyAC(
            double_q=config["double_q"],
            alpha=config["alpha"],
            num_inputs=config["feature_size"],
            action_space=config["action_space"],
            gamma=config["gamma"],
            tau=config["tau"],
            policy=config["policy_type"],
            target_update_interval=config["target_update_interval"],
            critic_lr=config["critic_lr"],
            actor_lr_scale=config["actor_lr_scale"],
            actor_hidden_dim=config["hidden_dim"],
            critic_hidden_dim=config["hidden_dim"],
            replay_capacity=config["replay_capacity"],
            seed=config["seed"],
            batch_size=config["batch_size"],
            cuda=config["cuda"],
            clip_stddev=config.get("clip_stddev", 1000),
            clip_min=config.get("clip_min", 1e-6),
            clip_max=config.get("clip_max", 1e6),
            epsilon=config.get("epsilon", 1.0),
            num_components=config.get("num_components", 1),
            latent_dim=config.get("latent_dim", 2),
            lmbda=config.get("lmbda", 0.5),
            direct=config.get("direct", False),
            shared=config.get("shared", False),
            init=config["weight_init"],
            num_samples=config["n"],
            actor_rho=config["actor_rho"],
            proposal_rho_scale=config["proposal_rho_scale"],
            betas=config["betas"],
            activation=activation,
            env=config["env"],
            uniform_exploration_steps=config["uniform_exploration_steps"],
            clip_actions=config.get("clip_actions", True),
            steps_before_learning=config["steps_before_learning"],
            should_use_critic_target=config["should_use_critic_target"],
        )

    # GreedyAC
    # This version of GreedyAC can be used with discrete or continuous action
    # values. The difference between this and "Discrete GreedyAC" is that this
    # one samples actions for the update, whereas "Discrete GreedyAC" only
    # increases the probability of taking the best action, and must be used
    # with discrete actions.
    if agent.lower() == "GreedyAC".lower():
        if "activation" in config:
            activation = config["activation"]
        else:
            activation = "relu"

        from agent.nonlinear.GreedyAC import GreedyAC

        uniform_exploration_steps = config.get(
            "uniform_exploration_steps", 0
        )

        steps_before_learning = config.get(
            "steps_before_learning", 0
        )

        return GreedyAC(
            double_q=config["double_q"],
            soft_q=config["soft_q"],
            num_inputs=config["feature_size"],
            action_space=config["action_space"],
            gamma=config["gamma"],
            tau=config["tau"],
            alpha=config["alpha"],
            policy=config["policy_type"],
            target_update_interval=config["target_update_interval"],
            critic_lr=config["critic_lr"],
            actor_lr_scale=config["actor_lr_scale"],
            actor_hidden_dim=config["hidden_dim"],
            critic_hidden_dim=config["hidden_dim"],
            replay_capacity=config["replay_capacity"],
            seed=config["seed"],
            batch_size=config["batch_size"],
            cuda=config["cuda"],
            clip_stddev=config["clip_stddev"],
            init=config["weight_init"],
            rho=config["n_rho"][1],
            num_samples=config["n_rho"][0],
            betas=config["betas"], activation=activation,
            env=config["env"],
            steps_before_learning=steps_before_learning,
            uniform_exploration_steps=uniform_exploration_steps,
        )

    raise NotImplementedError("No agent " + agent)


def _calculate_mean_return_episodic(hp_returns, type_, after=0):
    """
    Calculates the mean return for an experiment run on an episodic environment
    over all runs and episodes

    Parameters
    ----------
    hp_returns : Iterable of Iterable
        A list of lists, where the outer list has a single inner list for each
        run. The inner lists store the return per episode for that run. Note
        that these returns should be for a single hyperparameter setting, as
        everything in these lists are averaged and returned as the average
        return.
    type_ : str
        Whether calculating the training or evaluation mean returns, one of
        'train', 'eval'
    after : int, optional
        Only consider episodes after this episode, by default 0

    Returns
    -------
    2-tuple of float
        The mean and standard error of the returns over all episodes and all
        runs
    """
    if type_ == "eval":
        hp_returns = [np.mean(hp_returns[i][after:], axis=-1) for i in
                      range(len(hp_returns))]

    # Calculate the average return for all episodes in the run
    run_returns = [np.mean(hp_returns[i][after:]) for i in
                   range(len(hp_returns))]

    mean = np.mean(run_returns)
    stderr = np.std(run_returns) / np.sqrt(len(hp_returns))

    return mean, stderr


def _calculate_mean_return_episodic_conf(hp_returns, type_, significance,
                                         after=0):
    """
    Calculates the mean return for an experiment run on an episodic environment
    over all runs and episodes

    Parameters
    ----------
    hp_returns : Iterable of Iterable
        A list of lists, where the outer list has a single inner list for each
        run. The inner lists store the return per episode for that run. Note
        that these returns should be for a single hyperparameter setting, as
        everything in these lists are averaged and returned as the average
        return.
    type_ : str
        Whether calculating the training or evaluation mean returns, one of
        'train', 'eval'
    significance: float
        The level of significance for the confidence interval
    after : int, optional
        Only consider episodes after this episode, by default 0

    Returns
    -------
    2-tuple of float
        The mean and standard error of the returns over all episodes and all
        runs
    """
    if type_ == "eval":
        hp_returns = [np.mean(hp_returns[i][after:], axis=-1) for i in
                      range(len(hp_returns))]

    # Calculate the average return for all episodes in the run
    run_returns = [np.mean(hp_returns[i][after:]) for i in
                   range(len(hp_returns))]

    mean = np.mean(run_returns)
    run_returns = np.array(run_returns)

    conf = bs.bootstrap(run_returns, stat_func=bs_stats.mean,
                        alpha=significance)

    return mean, conf


def _calculate_mean_return_continuing(hp_returns, type_, after=0):
    """
    Calculates the mean return for an experiment run on a continuing
    environment over all runs and episodes

    Parameters
    ----------
    hp_returns : Iterable of Iterable
        A list of lists, where the outer list has a single inner list for each
        run. The inner lists store the return per episode for that run. Note
        that these returns should be for a single hyperparameter setting, as
        everything in these lists are averaged and returned as the average
        return.
    type_ : str
        Whether calculating the training or evaluation mean returns, one of
        'train', 'eval'
    after : int, optional
        Only consider episodes after this episode, by default 0

    Returns
    -------
    2-tuple of float
        The mean and standard error of the returns over all episodes and all
        runs
    """
    hp_returns = np.stack(hp_returns)

    # If evaluating, use the mean return over all episodes for each
    # evaluation interval. That is, if 10 eval episodes for each
    # evaluation the take the average return over all these eval
    # episodes
    if type_ == "eval":
        hp_returns = hp_returns.mean(axis=-1)

    # Calculate the average return over all runs
    hp_returns = hp_returns[after:, :].mean(axis=-1)

    # Calculate the average return over all "episodes"
    stderr = np.std(hp_returns) / np.sqrt(len(hp_returns))
    mean = hp_returns.mean(axis=0)

    return mean, stderr


def _calculate_mean_return_continuing_conf(hp_returns, type_, significance,
                                           after=0):
    """
    Calculates the mean return for an experiment run on a continuing
    environment over all runs and episodes

    Parameters
    ----------
    hp_returns : Iterable of Iterable
        A list of lists, where the outer list has a single inner list for each
        run. The inner lists store the return per episode for that run. Note
        that these returns should be for a single hyperparameter setting, as
        everything in these lists are averaged and returned as the average
        return.
    type_ : str
        Whether calculating the training or evaluation mean returns, one of
        'train', 'eval'
    after : int, optional
        Only consider episodes after this episode, by default 0

    Returns
    -------
    2-tuple of float
        The mean and standard error of the returns over all episodes and all
        runs
    """
    hp_returns = np.stack(hp_returns)

    # If evaluating, use the mean return over all episodes for each
    # evaluation interval. That is, if 10 eval episodes for each
    # evaluation the take the average return over all these eval
    # episodes
    if type_ == "eval":
        hp_returns = hp_returns.mean(axis=-1)

    # Calculate the average return over all episodes
    hp_returns = hp_returns[after:, :].mean(axis=-1)

    # Calculate the average return over all runs
    mean = hp_returns.mean(axis=0)
    conf = bs.bootstrap(hp_returns, stat_func=bs_stats.mean,
                        alpha=significance)

    return mean, conf


def get_best_hp_by_file(dir, type_, after=0, env_type="continuing"):
    """
    Find the best hyperparameters from a list of files.

    Gets and returns a list of the hyperparameter settings, sorted by average
    return. This function assumes a single directory containing all data
    dictionaries, where each data dictionary contains all data of all runs for
    a *single* hyperparameter setting. There must be a single file for each
    hyperparameter setting in the argument directory.

    Note: If any retrun is NaN within the range specified by after, then the
    entire return is considered NaN.

    Parameters
    ----------
    dir : str
        The directory which contains the data dictionaries, with one data
        dictionary per hyperparameter setting
    type_ : str
        The type of return by which to compare hyperparameter settings, one of
        "train" or "eval"
    after : int, optional
        Hyperparameters will only be compared by their performance after
        training for this many episodes (in continuing tasks, this is the
        number of times the task is restarted). For example, if after = -10,
        then only the last 10 returns from training/evaluation are taken
        into account when comparing the hyperparameters. As usual, positive
        values index from the front, and negative values index from the back.
    env_type : str, optional
        The type of environment, one of 'continuing', 'episodic'. By default
        'continuing'

    Returns
    -------
        n-tuple of 2-tuple(int, float)
    A tuple with the number of elements equal to the total number of
    hyperparameter combinations. Each sub-tuple is a tuple of (hyperparameter
    setting number, mean return over all runs and episodes)
    """
    files = glob(os.path.join(dir, "*.pkl"))

    if type_ not in ("train", "eval"):
        raise ValueError("type_ should be one of 'train', 'eval'")

    return_type = "train_episode_rewards" if type_ == "train" \
        else "eval_episode_rewards"

    mean_returns = []
    # hp_settings = []
    # hp_settings = sorted(list(data["experiment_data"].keys()))
    for file in tqdm(files):
        hp_returns = []

        # Get the data
        file = open(file, "rb")
        data = pickle.load(file)

        hp_setting = next(iter(data["experiment_data"]))
        # hp_settings.append(hp_setting)
        for run in data["experiment_data"][hp_setting]["runs"]:
            hp_returns.append(run[return_type])

        # Episodic and continuing must be dealt with differently since
        # we may have many episodes for a given number of timesteps for
        # episodic tasks
        if env_type == "episodic":
            hp_returns, _ = _calculate_mean_return_episodic(hp_returns, type_,
                                                            after)

        elif env_type == "continuing":
            hp_returns, _ = _calculate_mean_return_continuing(hp_returns,
                                                              type_, after)

        # Save mean return
        mean_returns.append((hp_setting, hp_returns))

        # Close the file
        file.close()
        del data

    # Create a structured array for sorting by return
    dtype = [("setting index", int), ("return", float)]
    mean_returns = np.array(mean_returns, dtype=dtype)

    # Return the best hyperparam settings in order with the
    # mean returns sorted by hyperparmater setting performance
    # best_hp_settings = np.argsort(mean_returns)
    # mean_returns = np.array(mean_returns)[best_hp_settings]
    mean_returns = np.sort(mean_returns, order="return")

    # return tuple(zip(best_hp_settings, mean_returns))
    return mean_returns


def combine_runs(data1, data2):
    """
    Adds the runs for each hyperparameter setting in data2 to the runs for the
    corresponding hyperparameter setting in data1.

    Given two data dictionaries, this function will get each hyperparameter
    setting and extend the runs done on this hyperparameter setting and saved
    in data1 by the runs of this hyperparameter setting and saved in data2.
    In short, this function extends the lists
    data1["experiment_data"][i]["runs"] by the lists
    data2["experiment_data"][i]["runs"] for all i. This is useful if
    multiple runs are done at different times, and the two data files need
    to be combined.

    Parameters
    ----------
    data1 : dict
        A data dictionary as generated by main.py
    data2 : dict
        A data dictionary as generated by main.py

    Raises
    ------
    KeyError
        If a hyperparameter setting exists in data2 but not in data1. This
        signals that the hyperparameter settings indices are most likely
        different, so the hyperparameter index i in data1 does not correspond
        to the same hyperparameter index in data2. In addition, all other
        functions expect the number of runs to be consistent for each
        hyperparameter setting, which would be violated in this case.
    """
    for hp_setting in data1["experiment_data"]:
        if hp_setting not in list(data2.keys()):
            # Ensure consistent hyperparam settings indices
            raise KeyError("hyperparameter settings are different " +
                           "between the two experiments")

        extra_runs = data2["experiment_data"][hp_setting]["runs"]
        data1["experiment_data"][hp_setting]["runs"].extend(extra_runs)


def get_returns(data, type_, ind, env_type="continuing"):
    """
    Gets the returns seen by an agent

    Gets the online or offline returns seen by an agent trained with
    hyperparameter settings index ind.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Whether to get the training or evaluation returns, one of 'train',
        'eval'
    ind : int
        Gets the returns of the agent trained with this hyperparameter
        settings index
    env_type : str, optional
        The type of environment, one of 'continuing', 'episodic'. By default
        'continuing'

    Returns
    -------
    array_like
        The array of returns of the form (N, R, C) where N is the number of
        runs, R is the number of times a performance was measured, and C is the
        number of returns generated each time performance was measured
        (offline >= 1; online = 1). For the online setting, N is the number of
        runs, and R is the number of episodes and C = 1. For the offline
        setting, N is the number of runs, R is the number of times offline
        evaluation was performed, and C is the number of episodes run each
        time performance was evaluated offline.
    """
    if env_type == "episodic":
        # data = reduce_episodes(data, ind, type_)
        data = runs.expand_episodes(data, ind, type_)

    returns = []
    if type_ == "eval":
        # Get the offline evaluation episode returns per run
        for run in data["experiment_data"][ind]["runs"]:
            returns.append(run["eval_episode_rewards"])
        returns = np.stack(returns)

    elif type_ == "train":
        # Get the returns per episode per run
        for run in data["experiment_data"][ind]["runs"]:
            returns.append(run["train_episode_rewards"])
        # if different runs have different lengths, then we simply
        # take the minimum length of all runs
        min_len = min([len(run["train_episode_rewards"]) for run in
                        data["experiment_data"][ind]["runs"]])
        returns = np.stack([run["train_episode_rewards"][:min_len] for run in
                            data["experiment_data"][ind]["runs"]])
        returns = np.expand_dims(np.stack(returns), axis=-1)

    return returns


def get_avg_returns(data, type_, ind, after=0, before=None):
    """
    Gets the average returns over all episodes seen by an agent for each run

    Gets the online or offline returns seen by an agent trained with
    hyperparameter settings index ind.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Whether to get the training or evaluation returns, one of 'train',
        'eval'
    ind : int
        Gets the returns of the agent trained with this hyperparameter
        settings index

    Returns
    -------
    array_like
        The array of returns of the form (N, R, C) where N is the number of
        runs, R is the number of times a performance was measured, and C is the
        number of returns generated each time performance was measured
        (offline >= 1; online = 1). For the online setting, N is the number of
        runs, and R is the number of episodes and C = 1. For the offline
        setting, N is the number of runs, R is the number of times offline
        evaluation was performed, and C is the number of episodes run each
        time performance was evaluated offline.
    """
    returns = []
    if type_ == "eval":
        # Get the offline evaluation episode returns per run
        for run in data["experiment_data"][ind]["runs"]:
            if before is not None:
                run_returns = run["eval_episode_rewards"][after:before]
            else:
                run_returns = run["eval_episode_rewards"][after:before]
            returns.append(run_returns)

        returns = np.stack(returns).mean(axis=(-2, -1))

    elif type_ == "train":
        # Get the returns per episode per run
        for run in data["experiment_data"][ind]["runs"]:
            if before is not None:
                run_returns = run["train_episode_rewards"][after:before]
            else:
                run_returns = run["train_episode_rewards"][after:]
            returns.append(np.mean(run_returns))

        returns = np.array(returns)

    return returns


def get_mean_returns_with_stderr_hp_varying(dir_, type_, hp_name, combo,
                                            env_config, agent_config, after=0,
                                            env_type="continuing"):
    """
    Calculate mean and standard error of return for each hyperparameter value.

    Gets the mean returns for each variation of a single hyperparameter,
    with all other hyperparameters remaining constant. Since there are
    many different ways this can happen (the hyperparameter can vary
    with all other remaining constant, but there are many combinations
    of these constant hyperparameters), the combo argument cycles through
    the combinations of constant hyperparameters.

    Given hyperparameters a, b, and c, let's say we want to get all
    hyperparameter settings indices where a varies, and b and c are constant.
    if a, b, and c can each be 1 or 2, then there are four ways that a can
    vary with b and c remaining constant:

        [
            ((a=1, b=1, c=1), (a=2, b=1, c=1)),         combo = 0
            ((a=1, b=2, c=1), (a=2, b=2, c=1)),         combo = 1
            ((a=1, b=1, c=2), (a=2, b=1, c=2)),         combo = 2
            ((a=1, b=2, c=2), (a=2, b=2, c=2))          combo = 3
        ]

    The combo argument indexes into this list of hyperparameter settings

    Parameters
    ----------
    dir_ : str
        The directory of data dictionaries generated from running main.py,
        separated into one data dictionary per HP setting
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    hp_name : str
        The name of the hyperparameter to plot the sensitivity curves of
    combo : int
        Determines the values of the constant hyperparameters. Given that
        only one hyperparameter may vary, there are many different sets
        having this hyperparameter varying with all others remaining constant
        since each constant hyperparameter may take on many values. This
        argument cycles through all sets of hyperparameter settings indices
        that have only one hyperparameter varying and all others constant.
    env_config : dict
        The environment configuration file as a Python dictionary
    agent_config : dict
        The agent configuration file as a Python dictionary
    after : int
        Only consider returns after this episode
    """
    hp_combo = get_varying_single_hyperparam(env_config, agent_config,
                                             hp_name)[combo]

    env_name = env_config["env_name"]
    agent_name = agent_config["agent_name"]
    filename = f"{env_name}_{agent_name}_hp-" + "{hp}.pkl"

    mean_returns = []
    stderr_returns = []
    hp_values = []
    for hp in hp_combo:
        if hp is None:
            continue

        with open(os.path.join(dir_, filename.format(hp=hp)), "rb") as in_file:
            data = pickle.load(in_file)

        hp_returns = []
        return_type = f"{type_}_episode_rewards"
        for run in data["experiment_data"][hp]["runs"]:
            hp_returns.append(run[return_type])

        if env_type == "episodic":
            mean_return, stderr_return = \
                _calculate_mean_return_episodic(hp_returns, type_, after)
        elif env_type == "continuing":
            mean_return, stderr_return = \
                _calculate_mean_return_continuing(hp_returns, type_, after)

        mean_returns.append(mean_return)
        stderr_returns.append(stderr_return)
        hp_value = data["experiment_data"][hp]["agent_hyperparams"][hp_name]
        hp_values.append(hp_value)

        del data

    # Get each hp value and sort all results by hp value
    # hp_values = np.array(agent_config["parameters"][hp_name])
    hp_values = np.array(hp_values)
    indices = np.argsort(hp_values)

    mean_returns = np.array(mean_returns)[indices]
    stderr_returns = np.array(stderr_returns)[indices]
    hp_values = hp_values[indices]

    return hp_values, mean_returns, stderr_returns


def get_mean_returns_with_conf_hp_varying(dir_, type_, hp_name, combo,
                                          env_config, agent_config, after=0,
                                          env_type="continuing",
                                          significance=0.1):
    """
    Calculate mean and standard error of return for each hyperparameter value.

    Gets the mean returns for each variation of a single hyperparameter,
    with all other hyperparameters remaining constant. Since there are
    many different ways this can happen (the hyperparameter can vary
    with all other remaining constant, but there are many combinations
    of these constant hyperparameters), the combo argument cycles through
    the combinations of constant hyperparameters.

    Given hyperparameters a, b, and c, let's say we want to get all
    hyperparameter settings indices where a varies, and b and c are constant.
    if a, b, and c can each be 1 or 2, then there are four ways that a can
    vary with b and c remaining constant:

        [
            ((a=1, b=1, c=1), (a=2, b=1, c=1)),         combo = 0
            ((a=1, b=2, c=1), (a=2, b=2, c=1)),         combo = 1
            ((a=1, b=1, c=2), (a=2, b=1, c=2)),         combo = 2
            ((a=1, b=2, c=2), (a=2, b=2, c=2))          combo = 3
        ]

    The combo argument indexes into this list of hyperparameter settings

    Parameters
    ----------
    dir_ : str
        The directory of data dictionaries generated from running main.py,
        separated into one data dictionary per HP setting
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    hp_name : str
        The name of the hyperparameter to plot the sensitivity curves of
    combo : int
        Determines the values of the constant hyperparameters. Given that
        only one hyperparameter may vary, there are many different sets
        having this hyperparameter varying with all others remaining constant
        since each constant hyperparameter may take on many values. This
        argument cycles through all sets of hyperparameter settings indices
        that have only one hyperparameter varying and all others constant.
    env_config : dict
        The environment configuration file as a Python dictionary
    agent_config : dict
        The agent configuration file as a Python dictionary
    after : int
        Only consider returns after this episode
    """
    hp_combo = get_varying_single_hyperparam(env_config, agent_config,
                                             hp_name)[combo]

    env_name = env_config["env_name"]
    agent_name = agent_config["agent_name"]
    filename = f"{env_name}_{agent_name}_hp-" + "{hp}.pkl"

    mean_returns = []
    conf_returns = []
    hp_values = []
    for hp in hp_combo:
        if hp is None:
            continue

        with open(os.path.join(dir_, filename.format(hp=hp)), "rb") as in_file:
            data = pickle.load(in_file)

        hp_returns = []
        return_type = f"{type_}_episode_rewards"
        for run in data["experiment_data"][hp]["runs"]:
            hp_returns.append(run[return_type])

        if env_type == "episodic":
            mean_return, conf_return = \
                _calculate_mean_return_episodic_conf(hp_returns, type_,
                                                     significance, after)
        elif env_type == "continuing":
            mean_return, conf_return = \
                _calculate_mean_return_continuing_conf(hp_returns, type_,
                                                       significance, after)

        mean_returns.append(mean_return)
        conf_returns.append([conf_return.lower_bound, conf_return.upper_bound])
        hp_value = data["experiment_data"][hp]["agent_hyperparams"][hp_name]
        hp_values.append(hp_value)

        del data

    # Get each hp value and sort all results by hp value
    # hp_values = np.array(agent_config["parameters"][hp_name])
    hp_values = np.array(hp_values)
    indices = np.argsort(hp_values)

    mean_returns = np.array(mean_returns)[indices]
    conf_returns = np.array(conf_returns)[indices, :].transpose()
    hp_values = hp_values[indices]

    return hp_values, mean_returns, conf_returns


def get_mean_err(data, type_, ind, smooth_over, error,
                 env_type="continuing", keep_shape=False,
                 err_args={}):
    """
    Gets the timesteps, mean, and standard error to be plotted for
    a given hyperparameter settings index

    Note: This function assumes that each run has an equal number of episodes.
    This is true for continuing tasks. For episodic tasks, you will need to
    cutoff the episodes so all runs have the same number of episodes.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : int
        The hyperparameter settings index to plot
    smooth_over : int
        The number of previous data points to smooth over. Note that this
        is *not* the number of timesteps to smooth over, but rather the number
        of data points to smooth over. For example, if you save the return
        every 1,000 timesteps, then setting this value to 15 will smooth
        over the last 15 readings, or 15,000 timesteps.
    error: function
        The error function to compute the error with
    env_type : str, optional
        The type of environment the data was generated on
    keep_shape : bool, optional
        Whether or not the smoothed data should discard or keep the first
        few data points before smooth_over.
    err_args : dict
        A dictionary of keyword arguments to pass to the error function

    Returns
    -------
    3-tuple of list(int), list(float), list(float)
        The timesteps, mean episodic returns, and standard errors of the
        episodic returns
    """
    timesteps = None  # So the linter doesn't have a temper tantrum

    # Determine the timesteps to plot at
    if type_ == "eval":
        timesteps = \
            data["experiment_data"][ind]["runs"][0]["timesteps_at_eval"]

    elif type_ == "train":
        timesteps_per_ep = \
            data["experiment_data"][ind]["runs"][0]["train_episode_steps"]
        timesteps = get_cumulative_timesteps(timesteps_per_ep)

    # Get the mean over all episodes per evaluation step (for online
    # returns, this axis will have length 1 so we squeeze it)
    returns = get_returns(data, type_, ind, env_type=env_type)
    returns = returns.mean(axis=-1)

    returns = smooth(returns, smooth_over, keep_shape=keep_shape)

    # Get the standard error of mean episodes per evaluation
    # step over all runs
    if error is not None:
        err = error(returns, **err_args)
    else:
        err = None

    # Get the mean over all runs
    mean = returns.mean(axis=0)

    # Return only the valid portion of timesteps. If smoothing and not
    # keeping the first data points, then the first smooth_over columns
    # will not have any data
    if not keep_shape:
        end = len(timesteps) - smooth_over + 1
        timesteps = timesteps[:end]

    return timesteps, mean, err


def bootstrap_conf(runs, significance=0.01):
    """
    THIS NEEDS TO BE UPDATED


    Gets the bootstrap confidence interval of the distribution of mean return
    per episode for a single hyperparameter setting.

    Note that this function assumes that there are an equal number of episodes
    for each run. This is true for continuing environments. If using an
    episodic environment, ensure that the episodes have been made consistent
    across runs before running this function.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    significance : float, optional
        The significance level for the confidence interval, by default 0.01

    Returns
    -------
    array_like
        An array with two rows and n columns. The first row denotes the lower
        bound of the confidence interval and the second row denotes the upper
        bound of the confidence interval. The number of columns, n, is the
        number of episodes.
    """
    # return_type = type_ + "_episode_rewards"
    # runs = []
    # for run in data["experiment_data"][hp]["runs"]:
    #     if type_ == "eval":
    #         runs.append(run[return_type].mean())
    #     else:
    #         runs.append(run[return_type])

    # Rows are the returns for the episode number == row number for each run
    ep_conf = []
    run_returns = []
    for ep in range(runs.shape[1]):
        ep_returns = []
        for run in range(runs.shape[0]):
            ep_returns.append(np.mean(runs[run][ep]))
        run_returns.append(ep_returns)

    run_returns = np.array(run_returns)

    conf_interval = []
    for ep in range(run_returns.shape[0]):
        ep_conf = bs.bootstrap(run_returns[ep, :], stat_func=bs_stats.mean,
                               alpha=significance)
        conf_interval.append([ep_conf.lower_bound, ep_conf.upper_bound])

    return np.array(conf_interval).transpose()


def stderr(matrix, axis=0):
    """
    Calculates the standard error along a specified axis

    Parameters
    ----------
    matrix : array_like
        The matrix to calculate standard error along the rows of
    axis : int, optional
        The axis to calculate the standard error along, by default 0

    Returns
    -------
    array_like
        The standard error of each row along the specified axis

    Raises
    ------
    np.AxisError
        If an invalid axis is passed in
    """
    if axis > len(matrix.shape) - 1:
        raise np.AxisError(f"""axis {axis} is out of bounds for array with
                           {len(matrix.shape) - 1} dimensions""")

    samples = matrix.shape[axis]
    return np.std(matrix, axis=axis) / np.sqrt(samples)


def t_ci(matrix, axis=0, confidence=0.99):
    if axis > len(matrix.shape) - 1:
        raise np.AxisError(f"""axis {axis} is out of bounds for array with
                           {len(matrix.shape) - 1} dimensions""")

    samples = matrix.shape[axis]
    mean = np.mean(matrix, axis=axis)
    stderr = np.std(matrix, axis=axis, ddof=1) / np.sqrt(samples)
    ci = stderr * scipy.stats.t.ppf((1 + confidence) / 2, samples-1)
    return ci


def smooth(matrix, smooth_over, keep_shape=False, axis=1):
    """
    Smooth the rows of returns

    Smooths the rows of returns by replacing the value at index i in a
    row of returns with the average of the next smooth_over elements,
    starting at element i.

    Parameters
    ----------
    matrix : array_like
        The array to smooth over
    smooth_over : int
        The number of elements to smooth over
    keep_shape : bool, optional
        Whether the smoothed array should have the same shape as
        as the input array, by default True. If True, then for the first
        few i < smooth_over columns of the input array, the element at
        position i is replaced with the average of all elements at
        positions j <= i.

    Returns
    -------
    array_like
        The smoothed over array
    """
    if smooth_over > 1:
        # Smooth each run separately
        kernel = np.ones(smooth_over) / smooth_over
        smoothed_matrix = _smooth(matrix, kernel, "valid", axis=axis)

        # Smooth the first few episodes
        if keep_shape:
            beginning_cols = []
            for i in range(1, smooth_over):
                # Calculate smoothing over the first i columns
                beginning_cols.append(matrix[:, :i].mean(axis=1))

            # Numpy will use each smoothed col as a row, so transpose
            beginning_cols = np.array(beginning_cols).transpose()
    else:
        return matrix

    if keep_shape:
        # Return the smoothed array
        return np.concatenate([beginning_cols, smoothed_matrix],
                              axis=1)
    else:
        return smoothed_matrix


def _smooth(matrix, kernel, mode="valid", axis=0):
    """
    Performs an axis-wise convolution of matrix with kernel

    Parameters
    ----------
    matrix : array_like
        The matrix to convolve
    kernel : array_like
        The kernel to convolve on each row of matrix
    mode : str, optional
         The mode of convolution, by default "valid". One of 'valid',
         'full', 'same'
    axis : int, optional
         The axis to perform the convolution along, by default 0

    Returns
    -------
    array_like
        The convolved array

    Raises
    ------
    ValueError
        If kernel is multi-dimensional
    """
    if len(kernel.shape) != 1:
        raise ValueError("kernel must be 1D")

    def convolve(mat):
        return np.convolve(mat, kernel, mode=mode)

    return np.apply_along_axis(convolve, axis=axis, arr=matrix)


def get_cumulative_timesteps(timesteps_per_episode):
    """
    Creates an array of cumulative timesteps.

    Creates an array of timesteps, where each timestep is the cumulative
    number of timesteps up until that point. This is needed for plotting the
    training data, where  the training timesteps are stored for each episode,
    and we need to plot on the x-axis the cumulative timesteps, not the
    timesteps per episode.

    Parameters
    ----------
    timesteps_per_episode : list
        A list where each element in the list denotes the amount of timesteps
        for the corresponding episode.

    Returns
    -------
    array_like
        An array where each element is the cumulative number of timesteps up
        until that point.
    """
    timesteps_per_episode = np.array(timesteps_per_episode)
    cumulative_timesteps = [timesteps_per_episode[:i].sum()
                            for i in range(timesteps_per_episode.shape[0])]

    return np.array(cumulative_timesteps)


def combine_data_dictionaries_by_hp(dir_, env, agent, num_hp_settings,
                                    num_runs, save_dir=".", save_returns=True,
                                    env_type="continuing", offset=0):
    """
    Combines all data dictionaries by hyperparameter setting.

    Given a directory, combines all data dictionaries relating to the argument
    agent and environment, grouped by hyperparameter settings index. This way,
    each resulting data dictionary will contain all data of all runs for
    a single hyperparameter setting. This function will save one data
    dictionary, consisting of all runs, for each hyperparameter setting.

    This function looks for files named like
    "env_agent_data_start_stop_step.pkl" in the argument directory and
    combines all those whose start index refers to the same hyperparameter
    settings index.

    Parameters
    ----------
    dir_ : str
        The directory containing the data files
    env : str
        The name of the environment the experiments were run on
    agent : str
        The name of the agent in the experiments
    num_hp_settings : int
        The total number of hyperparameter settings used in the experiment
    num_runs : int
        The number of runs in the experiment
    save_dir : str, optional
        The directory to save the combined data in, by default "."
    save_returns : bool, optinal
        Whether or not to save the mean training and evaluation returns over
        all episodes and runs in a text file, by default True
    env_type : str, optional
        Whether the environment is continuing or episodic, one of 'continuing',
        'episodic'; by default 'continuing'. This determines how the average
        return is calculated. For continuing environments, each episode's
        performance is first averaged over runs and then over episodes. For
        episodic environments, the average return is calculated by first
        averaging over all episodes in each run, and then averaging over all
        runs; this is required since each run may have a different number of
        episodes.
    """
    hp_returns = []

    for hp_ind in range(num_hp_settings):
        _, train_mean, eval_mean = \
            combine_data_dictionaries_single_hp(dir_, env, agent, hp_ind,
                                                num_hp_settings, num_runs,
                                                save_dir, save_returns,
                                                env_type, offset=offset)
        if save_returns:
            hp_returns.append((hp_ind, train_mean, eval_mean))

    # Write the mean training and evaluation returns to a file
    if save_returns:
        filename = f"{env}_{agent}_avg_returns.pkl"
        with open(os.path.join(save_dir, filename), "wb") as out_file:
            # out_file.write(f"{train_mean}, {eval_mean}")
            pickle.dump(hp_returns, out_file)


def combine_data_dictionaries_single_hp(dir_, env, agent, hp_ind,
                                        num_hp_settings, num_runs,
                                        save_dir=".", calculate_returns=True,
                                        env_type="continuing", offset=0):
    filenames = f"{env}_{agent}_data_" + "{start}.pkl"

    hp_run_files = []
    hp_offset = offset * num_hp_settings
    start = hp_ind + hp_offset
    for j in range(start, start + num_hp_settings * num_runs, num_hp_settings):
        filename = os.path.join(dir_, filenames.format(start=j))
        if os.path.exists(filename):
            hp_run_files.append(filename)
    data = combine_data_dictionaries(hp_run_files, True, save_dir=save_dir,
                                     filename=f"{env}_{agent}_hp-{hp_ind}")

    if not calculate_returns:
        return hp_ind, -1., -1.

    # Get the returns for each episode in each run
    train_returns = []
    eval_returns = []
    for run in data["experiment_data"][hp_ind]["runs"]:
        train_returns.append(run["train_episode_rewards"])
        eval_returns.append(run["eval_episode_rewards"])

    # Get the mean performance
    if env_type == "continuing":
        train_mean, _ = _calculate_mean_return_continuing(train_returns,
                                                          "train")
        eval_mean, _ = _calculate_mean_return_continuing(eval_returns,
                                                         "eval")

    elif env_type == "episodic":
        train_mean, _ = _calculate_mean_return_episodic(train_returns,
                                                        "train")
        eval_mean, _ = _calculate_mean_return_episodic(eval_returns,
                                                       "eval")

    return hp_ind, train_mean, eval_mean


def combine_data_dictionaries(files, save=True, save_dir=".", filename="data"):
    """
    Combine data dictionaries given a list of filenames

    Given a list of paths to data dictionaries, combines each data dictionary
    into a single one.

    Parameters
    ----------
    files : list of str
        A list of the paths to data dictionary files to combine
    save : bool
        Whether or not to save the data
    save_dir : str, optional
        The directory to save the resulting data dictionaries in
    filename : str, optional
        The name of the file to save which stores the combined data, by default
        'data'

    Returns
    -------
    dict
        The combined dictionary
    """
    # Use first dictionary as base dictionary
    with open(files[0], "rb") as in_file:
        data = pickle.load(in_file)

    # Add data from all other dictionaries
    for file in files[1:]:
        with open(file, "rb") as in_file:
            # Read in the new dictionary
            in_data = pickle.load(in_file)

            # Add experiment data to running dictionary
            for key in in_data["experiment_data"]:
                # Check if key exists
                if key in data["experiment_data"]:
                    # Append data if existing
                    data["experiment_data"][key]["runs"] \
                        .extend(in_data["experiment_data"][key]["runs"])

                else:
                    # Key doesn't exist - add data to dictionary
                    data["experiment_data"][key] = \
                        in_data["experiment_data"][key]

    if save:
        with open(os.path.join(save_dir, f"{filename}.pkl"), "wb") as out_file:
            pickle.dump(data, out_file)

    return data


def combine_data_dictionaries_by_dir(dir):
    """
    Combines the many data dictionaries created during the concurrent
    training procedure into a single data dictionary. The combined data is
    saved as "data.pkl" in the argument dir.

    Parameters
    ----------
    dir : str
        The path to the directory containing all data dictionaries to combine

    Returns
    -------
    dict
        The combined dictionary
    """
    files = glob(os.path.join(dir, "*.pkl"))

    combine_data_dictionaries(files)
