# Import modules
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker, gridspec
import experiment_utils as exp
import numpy as np
from scipy import ndimage
from scipy import stats as st
import seaborn as sns
from collections.abc import Iterable
import pickle
import matplotlib as mpl
import hypers
import warnings
import runs

TRAIN = "train"
EVAL = "eval"


# Set up plots
params = {
      'axes.labelsize': 48,
      'axes.titlesize': 36,
      'legend.fontsize': 16,
      'xtick.labelsize': 48,
      'ytick.labelsize': 48
}
plt.rcParams.update(params)

plt.rc('text', usetex=False)  # You might want usetex=True to get DejaVu Sans
plt.rc('font', **{'family': 'sans-serif', 'serif': ['DejaVu Sans']})
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams.update({'font.size': 15})
plt.tick_params(top=False, right=False, labelsize=20)

mpl.rcParams["svg.fonttype"] = "none"


# Constants
EPISODIC = "episodic"
CONTINUING = "continuing"


# Colours
CMAP = "tab10"
# DEFAULT_COLOURS = list(sns.color_palette(CMAP, 6).as_hex())
DEFAULT_COLOURS = list(sns.color_palette("Paired", n_colors=30).as_hex())
# DEFAULT_COLOURS = ["#003f5c", "#bc5090", "#ffa600", "#ff6361", "#58cfa1"]
plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=sns.color_palette(CMAP))
# sns.set_theme(palette=CMAP)
OFFSET = 0  # The offset to start in DEFAULT_COLOURS


def episode_steps(data, type_, ind, labels, xlim=None,
                  ylim=None, colours=None, xlabel="episodes",
                  ylabel="steps to goal", figsize=(16, 9),
                  title="Steps to Goal", α=0.2):
    """
    TODO: Docstring for steps_per_episode.

    Parameters
    ----------
    data : TODO
    type_ : TODO
    ind : TODO
    smooth_over : TODO
    labels : TODO
    xlim : TODO, optional
    ylim : TODO, optional
    colours : TODO, optional
    xlabel : TODO, optional
    ylabel : TODO, optional

    Returns
    -------
    TODO

    """
    # Set the colours to be default if not set
    if colours is None:
        colours = _get_default_colours(ind)

    # Set up figure
    fig, ax = _setup_fig(None, None, figsize, xlim=xlim, ylim=ylim,
                         xlabel=xlabel, ylabel=ylabel, title=title)

    # For a single dict, then many
    for i in range(len(data)):
        for j in range(len(ind[i])):
            _episode_steps(data[i], type_, ind[i][j], colours[i][j],
                           labels[i], ax, α)

    ax.legend()
    return fig, ax


def _episode_steps(data, type_, ind, colour, label, ax, α=0.2):
    """
    TODO: Docstring for steps_per_episode.

    Parameters
    ----------
    data : TODO
    type_ : TODO
    ind : TODO
    smooth_over : TODO
    label : TODO
    xlim : TODO, optional
    ylim : TODO, optional
    colours : TODO, optional
    xlabel : TODO, optional
    ylabel : TODO, optional

    Returns
    -------
    TODO

    """
    key = type_ + "_episode_steps"

    # For a single dict, then many
    steps_per_run = []
    lengths = []
    for run in data["experiment_data"][ind]["runs"]:
        steps_per_run.append(run[key])
        lengths.append(len(steps_per_run[-1]))

    # Adjust the lengths of each run so that there are a consistent number of
    # episodes in each row
    min_length = min(lengths)
    for i in range(len(steps_per_run)):
        steps_per_run[i] = steps_per_run[i][0:min_length]
    steps_per_run = np.array(steps_per_run)

    mean = steps_per_run.mean(axis=0)
    std_err = np.std(steps_per_run, axis=0, ddof=1) / \
        np.sqrt(steps_per_run.shape[0])

    print(f"Final steps to goal for {label}:", mean[-1])

    _plot_shaded(ax, np.arange(mean.shape[0]), mean, std_err, colour,
                 label, α)


def hyper_sensitivity(data_dicts, hyper, type_=TRAIN, figsize=(16, 9),
                      labels=None, metric="return", colors=dict(),
                      line_styles=dict(), markers=dict()):
    """
    Plots the hyperparameter sensitivity curves

    Parameters
    ----------
    data_dicts : list[dict]
        A list of data dictionaries resulting from some experiments
    hyper : str
        The hyper to plot the sensitivity of
    type_ : str
        The type of data to plot, one of train or eval
    figsize : tuple[int]
        The figure size
    labels : list[str]
        A list of labels, of the same length as data_dicts. If None, then the
        agent name is used
    metric : str
        The metric to gauge sensitivity by, one of 'return', 'steps'

    Returns
    -------
    plt.figure, plt.Axes
        The figure and axes plotted on
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    if type_ not in (TRAIN, EVAL):
        raise ValueError(f"type_ must be one of '{TRAIN}', '{EVAL}'")

    metric = metric.lower()
    if metric not in ("return", "steps"):
        raise ValueError(f"metric must be one of 'return', 'steps'")

    key = type_ + "_episode_" + ("rewards" if metric == "return" else "steps")

    for j, ag in enumerate(data_dicts):
        config = ag["experiment"]["agent"]
        num_settings = hypers.total(config["parameters"])
        hps = config["parameters"][hyper]

        max_returns = [None] * len(hps)
        max_inds = [-1] * len(hps)
        runs = -1

        for i in range(num_settings):
            setting = hypers.sweeps(config["parameters"], i)[0]
            ind = hps.index(setting[hyper])

            # Get the average return for the run. If no such data exists, we
            # assume that the agent diverged and we give it minimum performance
            if i not in ag["experiment_data"].keys():
                avg_return = np.finfo(np.float64).min
                continue

            # Store the total number of runs for each setting, which will be
            # needed in the final loop
            if len(ag["experiment_data"][i]["runs"]) > runs:
                runs = len(ag["experiment_data"][i]["runs"])

            AFTER = -int(3000 * 1.05)
            avg_return = []
            for run in ag["experiment_data"][i]["runs"]:
                avg_return.append(run[key][AFTER:])

            avg_run_return = [np.mean(run) for run in avg_return]
            avg_return = np.mean(avg_run_return)

            if max_returns[ind] is None or (
               metric == "return" and avg_return > max_returns[ind]) or (
               metric == "steps" and avg_return < max_returns[ind]):
                max_inds[ind] = i
                max_returns[ind] = avg_return

        # Go through each best hyper and get the mean performance + std err
        # per run. If no data exists due to divergence, then just append nans
        returns = []
        for index in max_inds:
            if index not in ag["experiment_data"]:
                returns.append([np.nan] * runs)
                continue

            index_returns = []
            for run in ag["experiment_data"][index]["runs"]:
                index_returns.append(run[key][AFTER:].mean())
            returns.append(index_returns)

            # Warn the user if some hyper setting does not have the expected
            # number of runs
            n = len(index_returns)
            if n != runs:
                warnings.warn(f"hyper setting {index} has only {n} " +
                              f"runs when {runs} runs expected")
                # for i in range(runs - n):
                #     index_returns.append(np.finfo(np.float32).min)

        # To deal with hyper settings which don't have the full number of runs,
        # we take each mean and standard error separately before adding to an
        # array.
        mean = np.array([np.mean(r) for r in returns])
        std_err = np.array([np.std(r, ddof=1) / len(r) for r in returns])

        ag_name = ag["experiment"]["agent"]["agent_name"]

        # Any runs that failed due to invalid hypers and resulted in nans
        # should have low performance. We make it 10 * lower than the lowest
        # performance
        std_err[np.where(np.isnan(std_err))] = 0
        min_ = np.min(mean[np.where(~np.isnan(mean))])
        mean[np.where(np.isnan(mean))] = min_ * (10 if min_ < 0 else 0.1)

        if not labels:
            label = ag_name
        else:
            label = labels[j]
        ax.plot(hps, mean, label=label, color=colors.get(label, None),
                linestyle=line_styles.get(label, None),
                marker=markers.get(label, None))
        ax.fill_between(hps, mean-std_err, mean+std_err, alpha=0.1,
                        color=colors.get(label, None))

        ylabel = "Steps to Goal" if metric == "steps" else "Average Return"
        ax.set_ylabel(ylabel)
        ax.set_xlabel(hyper)

    return fig, ax


def multi_hyper_sensitivity(data_dicts, _hypers=None, type_=TRAIN, figsize=(4, 3),
                            labels=None, metric="return", colors=dict(),
                            line_styles=dict(), markers=dict(), transpose=False):
    """
    Plots the hyperparameter sensitivity curves across multiple hyperparameters
    
    Parameters
    ----------
    data_dicts : list[dict]
        A list of data dictionaries resulting from some experiments
    type_ : str
        The type of data to plot, one of train or eval
    figsize : tuple[int]
        The figure size
    labels : list[str]
        A list of labels, of the same length as data_dicts. If None, then the
        agent name is used
    metric : str
        The metric to gauge sensitivity by, one of 'return', 'steps'

    Returns
    -------
    plt.figure, plt.Axes
        The figure and axes plotted on
    """
    if _hypers is None:
        _hypers = ['alpha', 'critic_lr', 'actor_lr_scale']
    if type_ not in (TRAIN, EVAL):
        raise ValueError(f"type_ must be one of '{TRAIN}', '{EVAL}'")

    metric = metric.lower()
    if metric not in ("return", "steps"):
        raise ValueError(f"metric must be one of 'return', 'steps'")
    
    key = type_ + "_episode_" + ("rewards" if metric == "return" else "steps")
    
    # each ag corresponds to plots in the same row    
    num_rows = len(data_dicts)
    num_cols = len(data_dicts[0]["experiment"]["agent"]["parameters"][_hypers[0]])
    if transpose:
        num_rows, num_cols = num_cols, num_rows
    figsize = (figsize[0] * num_cols, figsize[1] * num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True)
    for grid_x, ag in enumerate(data_dicts):
        config = ag["experiment"]["agent"]
        # num_settings = hypers.total(config["parameters"])
        grid_y_hps = config["parameters"][_hypers[0]]

        # each grid_x_hp correspond to one plot
        for grid_y, grid_y_hp in enumerate(grid_y_hps):
            if transpose:
                grid_x, grid_y = grid_y, grid_x

            if num_rows == 1:
                ax = axes[grid_y]
            elif num_cols == 1:
                ax = axes[grid_x]
            else:
                ax = axes[grid_x][grid_y]
            if grid_x == 0:
                ax.set_title(f"{_hypers[0]}={grid_y_hp}")
            # ax.set_xlabel(_hypers[2])
            # ax.set_ylabel('Average Return')

            hps = config["parameters"][_hypers[1]]
            for j, hp in enumerate(hps):
                x_hps = config["parameters"][_hypers[2]]

                max_returns = [None] * len(x_hps)
                max_inds = [-1] * len(x_hps)
                runs = -1

                def _f(_config):
                    if _config[_hypers[0]] != grid_y_hp:
                        return False
                    if _config[_hypers[1]] != hp:
                        return False
                    return True
                satisfies_settings, _ = hypers.satisfies(ag, _f)
                for i in satisfies_settings:
                    setting = hypers.sweeps(config["parameters"], i)[0]
                    assert setting[_hypers[0]] == grid_y_hp and setting[_hypers[1]] == hp
                    ind = x_hps.index(setting[_hypers[2]])

                    # Get the average return for the run. If no such data exists, we
                    # assume that the agent diverged and we give it minimum performance
                    if i not in ag["experiment_data"].keys():
                        avg_return = np.finfo(np.float64).min
                        continue

                    # Store the total number of runs for each setting, which will be
                    # needed in the final loop
                    if len(ag["experiment_data"][i]["runs"]) > runs:
                        runs = len(ag["experiment_data"][i]["runs"])

                    AFTER = -int(3000 * 1.05)
                    avg_return = []
                    for run in ag["experiment_data"][i]["runs"]:
                        avg_return.append(run[key][AFTER:])

                    avg_run_return = [np.mean(run) for run in avg_return]
                    avg_return = np.mean(avg_run_return)

                    if max_returns[ind] is None or (
                    metric == "return" and avg_return > max_returns[ind]) or (
                    metric == "steps" and avg_return < max_returns[ind]):
                        max_inds[ind] = i
                        max_returns[ind] = avg_return

                # Go through each best hyper and get the mean performance + std err
                # per run. If no data exists due to divergence, then just append nans
                returns = []
                for index in max_inds:
                    if index not in ag["experiment_data"]:
                        returns.append([np.nan] * runs)
                        continue

                    index_returns = []
                    for run in ag["experiment_data"][index]["runs"]:
                        index_returns.append(run[key][AFTER:].mean())
                    returns.append(index_returns)

                    # Warn the user if some hyper setting does not have the expected
                    # number of runs
                    n = len(index_returns)
                    if n != runs:
                        warnings.warn(f"hyper setting {index} has only {n} " +
                                    f"runs when {runs} runs expected")
                        # for i in range(runs - n):
                        #     index_returns.append(np.finfo(np.float32).min)

                # To deal with hyper settings which don't have the full number of runs,
                # we take each mean and standard error separately before adding to an
                # array.
                mean = np.array([np.mean(r) for r in returns])
                std_err = np.array([np.std(r, ddof=1) / len(r) for r in returns])

                # Any runs that failed due to invalid hypers and resulted in nans
                # should have low performance. We make it 10 * lower than the lowest
                # performance
                std_err[np.where(np.isnan(std_err))] = 0
                min_ = np.min(mean[np.where(~np.isnan(mean))])
                mean[np.where(np.isnan(mean))] = min_ * (10 if min_ < 0 else 0.1)

                label = hp
                ax.plot(x_hps, mean, label=f"{_hypers[1]} {label}", color=colors.get(label, None),
                        linestyle=line_styles.get(label, None),
                        marker=markers.get(label, None))
                ax.fill_between(x_hps, mean-std_err, mean+std_err, alpha=0.1,
                                color=colors.get(label, None))
    return fig, axes


def multi_learning_curves(data_dicts, _hypers, type_=TRAIN, figsize=(4, 3),
                          metric="return", transpose=False, smooth_over=1,
                          alpha=0.1, env_type="continuing", skip=1,
                          plot='return', log_every=100):
    """
    Plots the learning curves for each hyperparameter setting for multiple agent

    Parameters
    ----------
    data_dicts : list[dict]
        A list of data dictionaries resulting from some experiments
    type_ : str
        The type of data to plot, one of train or eval
    figsize : tuple[int]
        The figure size
    metric : str
        The metric to plot, one of 'return', 'steps'
    colors : dict
        A dictionary mapping labels to colours
    line_styles : dict
        a dictionary mapping labels to line styles
    markers : dict
        a dictionary mapping labels to markers
    transpose : bool
        Whether to transpose the plot

    Returns
    -------
    plt.figure, plt.Axes
        The figure and axes plotted on
    """
    # assert len(data_dicts) == 1, "only one agent is supported"
    assert len(_hypers) == 2, "only two hypers are supported"
    if type_ not in (TRAIN, EVAL):
        raise ValueError(f"type_ must be one of '{TRAIN}', '{EVAL}'")

    metric = metric.lower()
    if metric not in ("return", "steps"):
        raise ValueError(f"metric must be one of 'return', 'steps'")

    key = type_ + "_episode_" + ("rewards" if metric == "return" else "steps")

    # each ag corresponds to plots in the same row
    num_rows = len(data_dicts)
    num_cols = len(data_dicts[0]["experiment"]["agent"]["parameters"][_hypers[0]])
    if transpose:
        num_rows, num_cols = num_cols, num_rows
    figsize = (figsize[0] * num_cols, figsize[1] * num_rows)
    sharey = 'col' if plot == 'grad_norm' else True
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=sharey)
    for grid_x, ag in enumerate(data_dicts):
        config = ag["experiment"]["agent"]
        # num_settings = hypers.total(config["parameters"])
        grid_y_hps = config["parameters"][_hypers[0]]

        # each grid_x_hp correspond to one plot
        for grid_y, grid_y_hp in enumerate(grid_y_hps):
            if transpose:
                grid_x, grid_y = grid_y, grid_x

            if num_rows == 1:
                ax = axes[grid_y]
            elif num_cols == 1:
                ax = axes[grid_x]
            else:
                ax = axes[grid_x][grid_y]
            if grid_x == 0:
                ax.set_title(f"{_hypers[0]}={grid_y_hp}")

            hps = config["parameters"][_hypers[1]]
            colours = sns.color_palette("tab10", len(hps))
            for j, hp in enumerate(hps):
                max_return = None
                max_ind = -1
                runs = -1

                # find the best hyper setting for each hp                
                def _f(_config):
                    if _config[_hypers[0]] != grid_y_hp:
                        return False
                    if _config[_hypers[1]] != hp:
                        return False
                    return True
                satisfies_settings, _ = hypers.satisfies(ag, _f)
                for i in satisfies_settings:
                    setting = hypers.sweeps(config["parameters"], i)[0]
                    assert setting[_hypers[0]] == grid_y_hp and setting[_hypers[1]] == hp

                    # Get the average return for the run. If no such data exists, we
                    # assume that the agent diverged and we give it minimum performance
                    if i not in ag["experiment_data"].keys():
                        avg_return = np.finfo(np.float64).min
                        continue

                    # Store the total number of runs for each setting, which will be
                    # needed in the final loop
                    if len(ag["experiment_data"][i]["runs"]) > runs:
                        runs = len(ag["experiment_data"][i]["runs"])

                    AFTER = -int(3000 * 1.05)
                    avg_return = []
                    for run in ag["experiment_data"][i]["runs"]:
                        avg_return.append(run[key][AFTER:])

                    avg_run_return = [np.mean(run) for run in avg_return]
                    avg_return = np.mean(avg_run_return)

                    if max_return is None or (
                    metric == "return" and avg_return > max_return) or (
                    metric == "steps" and avg_return < max_return):
                        max_ind = i
                        max_return = avg_return

                if plot == 'return':
                    if env_type == "continuing":
                        plot_fn = _plot_mean_with_stderr_continuing
                    else:
                        plot_fn = _plot_mean_with_stderr_episodic
                    plot_fn(ag, type_, [max_ind], smooth_over, fig=fig, ax=ax,
                            alpha=alpha, name=f"{_hypers[1]}={hp}",
                            colours=[colours[j]], skip=skip)
                else:
                    _plot_metric(ag, max_ind, plot, fig=fig, ax=ax,
                                 colour=colours[j], log_every=log_every,
                                 name=f"{_hypers[1]}={hp}")
    return fig, axes


def _plot_metric(data, ind, key, fig=None, ax=None, colour=None, log_every=100,
                 name=None):
    data = data["experiment_data"][ind]["runs"]
    num_runs = len(data)
    y = []
    for i, run in enumerate(data):
        y.append(run[key])
    y = np.array(y)
    y = np.mean(y, axis=0)
    x = np.arange(y.shape[0]) * log_every
    ax.plot(x, y, color=colour, label=name)


def multi_individual_runs(data_dicts, _hypers, type_=TRAIN, figsize=(4, 3),
                          smooth_over=1, metric="return", alpha=0.1,
                          env_type="continuing"):
    assert len(data_dicts) == 1, "only one agent is supported"
    assert len(_hypers) == 2, "only two hypers are supported"
    if type_ not in (TRAIN, EVAL):
        raise ValueError(f"type_ must be one of '{TRAIN}', '{EVAL}'")

    metric = metric.lower()
    if metric not in ("return", "steps"):
        raise ValueError(f"metric must be one of 'return', 'steps'")

    key = type_ + "_episode_" + ("rewards" if metric == "return" else "steps")

    # each ag corresponds to plots in the same row
    ag = data_dicts[0]
    config = ag["experiment"]["agent"]
    grid_x_hps = config["parameters"][_hypers[1]]
    grid_y_hps = config["parameters"][_hypers[0]]
    num_rows = len(grid_x_hps)
    num_cols = len(grid_y_hps)
    figsize = (figsize[0] * num_cols, figsize[1] * num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True)
    colours = sns.color_palette("tab10", num_rows)
    for grid_x, grid_x_hp in enumerate(grid_x_hps):

        # each grid_x_hp correspond to one plot
        for grid_y, grid_y_hp in enumerate(grid_y_hps):

            if num_rows == 1:
                ax = axes[grid_y]
            elif num_cols == 1:
                ax = axes[grid_x]
            else:
                ax = axes[grid_x][grid_y]
            if grid_x == 0:
                ax.set_title(f"{_hypers[0]}={grid_y_hp}")
            if grid_y == 0:
                ax.set_ylabel(f"{_hypers[1]}={grid_x_hp}")

            max_return = None
            max_ind = -1
            runs = -1

            # find the best hyper setting for each hp                
            def _f(_config):
                if _config[_hypers[0]] != grid_y_hp:
                    return False
                if _config[_hypers[1]] != grid_x_hp:
                    return False
                return True
            satisfies_settings, _ = hypers.satisfies(ag, _f)
            for i in satisfies_settings:
                setting = hypers.sweeps(config["parameters"], i)[0]
                assert setting[_hypers[0]] == grid_y_hp and setting[_hypers[1]] == grid_x_hp

                # Get the average return for the run. If no such data exists, we
                # assume that the agent diverged and we give it minimum performance
                if i not in ag["experiment_data"].keys():
                    avg_return = np.finfo(np.float64).min
                    continue

                # Store the total number of runs for each setting, which will be
                # needed in the final loop
                if len(ag["experiment_data"][i]["runs"]) > runs:
                    runs = len(ag["experiment_data"][i]["runs"])

                AFTER = -int(3000 * 1.05)
                avg_return = []
                for run in ag["experiment_data"][i]["runs"]:
                    avg_return.append(run[key][AFTER:])

                avg_run_return = [np.mean(run) for run in avg_return]
                avg_return = np.mean(avg_run_return)

                if max_return is None or (
                metric == "return" and avg_return > max_return) or (
                metric == "steps" and avg_return < max_return):
                    max_ind = i
                    max_return = avg_return

            _plot_mean_with_runs(ag, type_, [max_ind], smooth_over,
                                    None, fig=fig, ax=ax, alpha=alpha,
                                    colours=[colours[grid_x]],
                                    env_type=env_type)
    return fig, axes


def multi_metric(data, ind, figsize=(16, 9), colour=None, show_curve=False,
                 title="Actions", alpha=0.2, steps=[0, 1, 3, 15, 63, 249],
                 smooth_over=1, num_runs=None, metric="action", log_every=100,):
    """
    Plots the actions taken by the agent over time
    """
    assert isinstance(data, dict), "only one agent is supported, " \
        "data must be a dictionary"
    data = data["experiment_data"][ind]["runs"]
    num_runs = len(data) if num_runs is None else num_runs
    if metric == "action":
        num_rows = len(steps)
        sharey = True
    else:
        num_rows = len(metric) if isinstance(metric, Iterable) else 1
        sharey = 'row'
    fig, axs = plt.subplots(num_rows + int(show_curve), num_runs, figsize=figsize,
                            sharex=True, sharey=sharey)
    for i, run in enumerate(data):
        if i >= num_runs:
            break

        if show_curve:
            ax = axs[0][i]
            twin_ax_x = ax.twinx()
            twin_ax = twin_ax_x.twiny()

            if i == 0:
                ax.set_ylabel("Learn' Curve")

            # hide the axes labels
            if i != num_runs - 1:
                twin_ax.set_yticklabels([])
            returns = run["train_episode_rewards"]
            returns = exp.smooth(returns, smooth_over=smooth_over, axis=0)
            xvalues = np.arange(returns.shape[0])
            twin_ax.plot(xvalues, returns, color=colour)
            twin_ax.tick_params(axis='x', labelsize=6)
            twin_ax_x.tick_params(axis='y', labelsize=6)
            ## for bimodal
            # twin_ax.set_ylim([-0.05, 1.55])
            # set the y ticks
            # twin_ax.set_yticks([0, 1.5])

            ## for Ant
            # twin_ax.set_ylim([-50, 5500])
            ## for SparsePendulum
            # twin_ax.set_ylim([-5, 205])
            ## for Pendulum
            # twin_ax.set_ylim([-1600, -100])
            ## for DenseMC
            twin_ax.set_ylim([-600, 600])
    
            ax.set_title(f"run={i}")

        if metric == [None]:
            continue
        elif metric == "action":
            for j, step in enumerate(steps):
                actions = run["actions_sampled"][step]
                ax = axs[j+int(show_curve)][i]
                sns.histplot(actions, ax=ax, color=colour, kde=True, bins=10,
                            stat="probability")
                if i == 0:
                    ax.set_ylabel(f"step={(step+1) * log_every}")
                if j == 0 and not show_curve:
                    ax.set_title(f"run={i}")
        else:
            if not isinstance(metric, Iterable):
                metric = [metric]
            # colours = sns.color_palette("tab10", len(metric) + 1)
            for j, _metric in enumerate(metric):
                ax = axs[j+int(show_curve)][i]
                y = np.array(run[_metric])
                x = np.arange(y.shape[0]) * log_every
                ax.plot(x, y)
                if i == 0:
                    if "mixture_" in _metric:
                        _metric = _metric[len("mixture_"):]
                    ax.set_ylabel(_metric)
    fig.suptitle(title)
    return fig, axs


def criteria1(data, target=1, use_last_percent=0.05):
    """
    if the number of actions closed to 1 is more than 50% of the total actions
    then the agent finds the optimal action; return True
    """
    actions = np.array(data["actions_sampled"])
    num_steps = actions.shape[0]
    actions = actions[-int(num_steps * use_last_percent):].flatten()
    return np.sum(np.abs(actions - target) < 0.2) / len(actions) > 0.50


def criteria2(data):
    """
    check if the agent has followed the suboptimal action
    """
    actions = np.array(data["actions_sampled"])
    for i in range(actions.shape[0]):
        if np.sum(np.abs(actions[i] + 1) < 0.2) / len(actions[i]) > 0.50:
            return True
    return False


def action_count(data, _hypers, labels, figsize=(16, 9), colours=None):
    """
    Plots the actions the agent converged to
    """
    assert len(_hypers) == 2, "only two hypers are supported"
    key = "train_episode_rewards"

    if colours is None:
        colours = sns.color_palette("tab10", len(data))

    fig, ax = plt.subplots(figsize=figsize)
    example_config = data[0]["experiment"]["agent"]
    x_hps = example_config["parameters"][_hypers[0]]
    for i, ag in enumerate(data):
        config = ag["experiment"]["agent"]
        y1s, y2s, y3s = [], [], []

        for x_hp in x_hps:
            max_return = None
            max_ind = -1

            # find the best hyper for each x_hp
            def _f(_config):
                if _config[_hypers[0]] != x_hp:
                    return False
                return True
            satisfies_settings, _ = hypers.satisfies(ag, _f)
            for j in satisfies_settings:
                setting = hypers.sweeps(config["parameters"], j)[0]
                assert setting[_hypers[0]] == x_hp
                # Get the average return for the run. If no such data exists, we
                # assume that the agent diverged and we give it minimum performance
                if j not in ag["experiment_data"].keys():
                    avg_return = np.finfo(np.float64).min
                    continue

                AFTER = -int(3000 * 1.05)
                avg_return = []
                for run in ag["experiment_data"][j]["runs"]:
                    avg_return.append(run[key][AFTER:])

                avg_run_return = [np.mean(run) for run in avg_return]
                avg_return = np.mean(avg_run_return)

                if max_return is None or avg_return > max_return:
                    max_ind = j
                    max_return = avg_return

            # print the configuration of the best hyper
            print(hypers.sweeps(config["parameters"], max_ind)[0])

            y1 = 0
            y2 = 0
            y3 = 0
            for run in ag["experiment_data"][max_ind]["runs"]:
                achieve_optimal = criteria1(run, target=1)
                y1 += achieve_optimal
                y2 += criteria1(run, target=-1)
                if achieve_optimal:
                    y3 += criteria2(run)
            y1s.append(y1)
            y2s.append(y2)
            y3s.append(y3)
        xs = np.arange(len(y1s))
        ax.plot(xs, y1s, label=f'{labels[i]} optimal action', color=colours[i])
        ax.plot(xs, y2s, label=f'{labels[i]} suboptimal action', color=colours[i], linestyle="--")
        ax.plot(xs, y3s, label=f'{labels[i]} switch', color=colours[i], linestyle=":")
        ax.set_xticks(xs)
        # set the x labels to be log scale
        ax.set_xticklabels(x_hps)
        num_runs = len(ag["experiment_data"][max_ind]["runs"])
        ax.axhline(y=num_runs, color='black', linestyle='--', linewidth=1.0)
    ax.set_xlabel(_hypers[0])
    ax.set_ylabel("Number of Actions")
    ax.legend()
    return fig, ax


def mean_with_bootstrap_conf(data, type_, ind, smooth_over, names,
                             fig=None, ax=None, figsize=(12, 6),
                             xlim=None, ylim=None, alpha=0.1,
                             colours=None, env_type="continuing",
                             significance=0.05, keep_shape=False,
                             xlabel=None, ylabel=None):
    """
    Plots the average training or evaluation return over all runs with
    confidence intervals.

    Given a list of data dictionaries of the form returned by main.py, this
    function will plot each episodic return for the list of hyperparameter
    settings ind each data dictionary. The ind argument is a list, where each
    element is a list of hyperparameter settings to plot for the data
    dictionary at the same index as this list. For example, if ind[i] = [1, 2],
    then plots will be generated for the data dictionary at location i
    in the data argument for hyperparameter settings ind[i] = [1, 2].
    The smooth_over argument tells how many previous data points to smooth
    over

    Parameters
    ----------
    data : list of dict
        The Python data dictionaries generated from running main.py for the
        agents
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : iter of iter of int
        The list of lists of hyperparameter settings indices to plot for
        each agent. For example [[1, 2], [3, 4]] means that the first agent
        plots will use hyperparameter settings indices 1 and 2, while the
        second will use 3 and 4.
    smooth_over : list of int
        The number of previous data points to smooth over for the agent's
        plot for each data dictionary. Note that this is *not* the number of
        timesteps to smooth over, but rather the number of data points to
        smooth over. For example, if you save the return every 1,000
        timesteps, then setting this value to 15 will smooth over the last
        15 readings, or 15,000 timesteps. For example, [1, 2] will mean that
        the plots using the first data dictionary will smooth over the past 1
        data points, while the second will smooth over the passed 2 data
        points for each hyperparameter setting.
    fig : plt.figure
        The figure to plot on, by default None. If None, creates a new figure
    ax : plt.Axes
        The axis to plot on, by default None, If None, creates a new axis
    figsize : tuple(int, int)
        The size of the figure to plot
    names : list of str
        The name of the agents, used for the legend
    xlim : float, optional
        The x limit for the plot, by default None
    ylim : float, optional
        The y limit for the plot, by default None
    alpha : float, optional
        The alpha channel for the plot, by default 0.1
    colours : list of list of str
        The colours to use for each hyperparameter settings plot for each data
        dictionary
    env_type : str, optional
        The type of environment, one of 'continuing', 'episodic'. By default
        'continuing'
    significance : float, optional
        The significance level for the confidence interval, by default 0.01

    Returns
    -------
    plt.figure, plt.Axes
        The figure and axes of the plot
    """
    fig, ax = _setup_fig(fig, ax, figsize, None, xlim, ylim, xlabel, ylabel)

    # Set the colours to be default if not set
    if colours is None:
        colours = _get_default_colours(ind)

    # Track the total timesteps per hyperparam setting over all episodes and
    # the cumulative timesteps per episode per data dictionary (timesteps
    # should be consistent between all hp settings in a single data dict)
    total_timesteps = []
    cumulative_timesteps = []

    for i in range(len(data)):
        if type_ == "train":
            cumulative_timesteps.append(exp.get_cumulative_timesteps(data[i]
                                        ["experiment_data"][ind[i][0]]["runs"]
                                        [0]["train_episode_steps"]))
        elif type_ == "eval":
            cumulative_timesteps.append(data[i]["experiment_data"][ind[i][0]]
                                        ["runs"][0]["timesteps_at_eval"])
        else:
            raise ValueError("type_ must be one of 'train', 'eval'")
        total_timesteps.append(cumulative_timesteps[-1][-1])

    # Find the minimum of total trained-for timesteps. Each plot will only
    # be plotted on the x-axis until this value
    min_timesteps = min(total_timesteps)

    # For each data dictionary, find the minimum index where the timestep at
    # that index is >=  minimum timestep
    ind_ge_min_timesteps = []
    for cumulative_timesteps_per_data in cumulative_timesteps:
        final_ind = np.where(cumulative_timesteps_per_data >=
                             min_timesteps)[0][0]
        # Since indexing will stop right before the minimum, increment it
        ind_ge_min_timesteps.append(final_ind + 1)

    # Plot all data for all HP settings, only up until the minimum index
    # fig, ax = None, None
    if env_type == "continuing":
        plot_fn = _plot_mean_with_conf_continuing
    else:
        plot_fn = _plot_mean_with_conf_episodic

    for i in range(len(data)):
        fig, ax = \
            plot_fn(data=data[i], type_=type_,
                    ind=ind[i], smooth_over=smooth_over[i], name=names[i],
                    fig=fig, ax=ax, figsize=figsize, xlim=xlim, ylim=ylim,
                    last_ind=ind_ge_min_timesteps[i], alpha=alpha,
                    colours=colours[i], significance=significance,
                    keep_shape=keep_shape)

    return fig, ax


def _plot_mean_with_conf_continuing(data, type_, ind, smooth_over, fig=None,
                                    ax=None, figsize=(12, 6), name="",
                                    last_ind=-1, xlabel="Timesteps",
                                    ylabel="Average Return", xlim=None,
                                    ylim=None, alpha=0.1, colours=None,
                                    significance=0.05, keep_shape=False):
    """
    Plots the average training or evaluation return over all runs for a single
    data dictionary on a continuing environment. Bootstrap confidence intervals
    are plotted as shaded regions.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : iter of int
        The list of hyperparameter settings indices to plot
    smooth_over : int
        The number of previous data points to smooth over. Note that this
        is *not* the number of timesteps to smooth over, but rather the number
        of data points to smooth over. For example, if you save the return
        every 1,000 timesteps, then setting this value to 15 will smooth
        over the last 15 readings, or 15,000 timesteps.
    fig : plt.figure
        The figure to plot on, by default None. If None, creates a new figure
    ax : plt.Axes
        The axis to plot on, by default None, If None, creates a new axis
    figsize : tuple(int, int)
        The size of the figure to plot
    name : str, optional
        The name of the agent, used for the legend
    last_ind : int, optional
        The index of the last element to plot in the returns list,
        by default -1. This is useful if you want to plot many things on the
        same axis, but all of which have a different number of elements. This
        way, we can plot the first last_ind elements of each returns for each
        agent.
    timestep_multiply : int, optional
        A value to multiply each timstep by, by default 1. This is useful if
        your agent does multiple updates per timestep and you want to plot
        performance vs. number of updates.
    xlim : float, optional
        The x limit for the plot, by default None
    ylim : float, optional
        The y limit for the plot, by default None
    alpha : float, optional
        The alpha channel for the plot, by default 0.1
    colours : list of str
        The colours to use for each plot of each hyperparameter setting
    env_type : str, optional
        The type of environment, one of 'continuing', 'episodic'. By default
        'continuing'
    significance : float, optional
        The significance level for the confidence interval, by default 0.01

    Returns
    -------
    plt.figure, plt.Axes
        The figure and axes of the plot

    Raises
    ------
    ValueError
        When an axis is passed but no figure is passed
        When an appropriate number of colours is not specified to cover all
        hyperparameter settings
    """
    # This should be the exact same as the episodic version except without
    # reducing the episodes. Follow the same structure as the episodic function
    # and the continuing function with standard error.
    raise NotImplementedError


def _plot_mean_with_conf_episodic(data, type_, ind, smooth_over, fig=None,
                                  ax=None, figsize=(12, 6), name="",
                                  last_ind=-1, xlabel="Timesteps",
                                  ylabel="Average Return", xlim=None,
                                  ylim=None, alpha=0.1, colours=None,
                                  significance=0.05, keep_shape=False):
    """
    Plots the average training or evaluation return over all runs for a single
    data dictionary on an episodic environment. Bootstrap confidence intervals
    are plotted as shaded regions.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : iter of int
        The list of hyperparameter settings indices to plot
    smooth_over : int
        The number of previous data points to smooth over. Note that this
        is *not* the number of timesteps to smooth over, but rather the number
        of data points to smooth over. For example, if you save the return
        every 1,000 timesteps, then setting this value to 15 will smooth
        over the last 15 readings, or 15,000 timesteps.
    fig : plt.figure
        The figure to plot on, by default None. If None, creates a new figure
    ax : plt.Axes
        The axis to plot on, by default None, If None, creates a new axis
    figsize : tuple(int, int)
        The size of the figure to plot
    name : str, optional
        The name of the agent, used for the legend
    last_ind : int, optional
        The index of the last element to plot in the returns list,
        by default -1. This is useful if you want to plot many things on the
        same axis, but all of which have a different number of elements. This
        way, we can plot the first last_ind elements of each returns for each
        agent.
    xlim : float, optional
        The x limit for the plot, by default None
    ylim : float, optional
        The y limit for the plot, by default None
    alpha : float, optional
        The alpha channel for the plot, by default 0.1
    colours : list of str
        The colours to use for each plot of each hyperparameter setting
    env_type : str, optional
        The type of environment, one of 'continuing', 'episodic'. By default
        'continuing'
    significance : float, optional
        The significance level for the confidence interval, by default 0.01

    Returns
    -------
    plt.figure, plt.Axes
        The figure and axes of the plot

    Raises
    ------
    ValueError
        When an axis is passed but no figure is passed
        When an appropriate number of colours is not specified to cover all
        hyperparameter settings
    """
    if colours is not None and len(colours) != len(ind):
        raise ValueError("must have one colour for each hyperparameter " +
                         "setting")

    if colours is None:
        colours = _get_default_colours(ind)

    # Set up figure
    # if ax is None and fig is None:
    #     fig = plt.figure(figsize=figsize)
    #     ax = fig.add_subplot()

    # if xlim is not None:
    #     ax.set_xlim(xlim)
    # if ylim is not None:
    #     ax.set_ylim(ylim)

    conf_level = "{:.2f}".format(1-significance)
    title = f"""Average {type_.title()} Return per Run with {conf_level}
                 Confidence Intervals"""
    fig, ax = _setup_fig(fig, ax, figsize, title, xlim, ylim, xlabel, ylabel)

    # Plot with bootstrap confidence interval
    for i in range(len(ind)):
        data = runs.expand_episodes(data, ind[i], type_=type_)

        _, mean, conf = exp.get_mean_err(data, type_, ind[i], smooth_over,
                                         exp.bootstrap_conf,
                                         err_args={
                                            "significance": significance,
                                         },
                                         keep_shape=keep_shape)

        mean = mean[:last_ind]
        conf = conf[:, :last_ind]

        episodes = np.arange(mean.shape[0])

        # Plot based on colours
        label = f"{name}"
        print(mean.shape, conf.shape, episodes.shape)
        _plot_shaded(ax, episodes, mean, conf, colours[i], label, alpha)

    ax.legend()
    conf_level = "{:.2f}".format(1-significance)
    ax.set_title(f"""Average {type_.title()} Return per Run with {conf_level}
                 Confidence Intervals""")
    # ax.set_ylabel(ylabel)
    # ax.set_xlabel(xlabel)

    fig.show()
    return fig, ax


def plot_mean_with_runs(data, type_, ind, smooth_over, names, colours=None,
                        figsize=(12, 6), xlim=None, ylim=None, alpha=0.1,
                        plot_avg=True, env_type="continuing", linewidth=1,
                        keep_shape=False, fig=None, ax=None, skip=1):
    """
    Plots the mean return over all runs and the return for each run for a list
    of data dictionaries and hyperparameter indices

    Plots both the mean return per episode (over runs) as well as the return
    for each individual run (including "mini-runs", if a set of concurrent
    episodes were run for all runs, e.g. multiple evaluation episodes per
    run at set intervals)

    Note that this function takes in a list of data dictionaries and will
    plot the runs for each ind (which is a list of lists, where each super-list
    refers to a data dictionary and each sub-list refers to the indices for
    that data dictionary to plot).

    Example
    -------
    plot_mean_with_runs([sac_data, linear_data], "train", [[3439], [38, 3]],
    smooth_over=[5, 2], names=["SAC", "LAC"], figsize=(12, 6), alpha=0.2,
    plot_avg=True, env_type="episodic")

    will plot hyperparameter index 3439 for the sac_data, smoothing over the
    last 5 episodes, and the label will have the term "SAC" in it; also plots
    the mean and each individual run on the linear_data for hyperparameter
    settings 38 and 3, smoothing over the last 2 episodes for each and with
    the term "LAC" in the labels.

    Parameters
    ----------
    data : list of dict
        The Python data dictionaries generated from running main.py for the
        agents
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : iter of iter of int
        The list of lists of hyperparameter settings indices to plot for
        each agent. For example [[1, 2], [3, 4]] means that the first agent
        plots will use hyperparameter settings indices 1 and 2, while the
        second will use 3 and 4.
    smooth_over : list of int
        The number of previous data points to smooth over for the agent's
        plot for each data dictionary. Note that this is *not* the number of
        timesteps to smooth over, but rather the number of data points to
        smooth over. For example, if you save the return every 1,000
        timesteps, then setting this value to 15 will smooth over the last
        15 readings, or 15,000 timesteps. For example, [1, 2] will mean that
        the plots using the first data dictionary will smooth over the past 1
        data points, while the second will smooth over the passed 2 data
        points for each hyperparameter setting.
    figsize : tuple(int, int)
        The size of the figure to plot
    names : list of str
        The name of the agents, used for the legend
    colours : list of list of str, optional
        The colours to use for each hyperparameter settings plot for each data
        dictionary, by default None
    xlim : float, optional
        The x limit for the plot, by default None
    ylim : float, optional
        The y limit for the plot, by default None
    alpha : float, optional
        The alpha to use for plots of the runs, by default 0.1
    plot_avg : bool, optional
        If concurrent episodes are executed in each run (e.g. multiple
        evaluation episodes are run at set intervals), then whether to plot the
        performance of each separately or the average performance over all
    env_type : str, optional
        The type of environment, one of 'continuing', 'episodic'. By default
        'continuing'
    fig : plt.Figure
        The figure to plot on
    ax : plt.Axes
        The axis to plot on

    Returns
    -------
    tuple of plt.figure, plt.Axes
        The figure and axis plotted on
    """
    # Set the colours to be default if not set
    if colours is None:
        colours = _get_default_colours(ind)

    # Set up figure
    # fig = plt.figure(figsize=figsize)
    # ax = fig.add_subplot()
    if env_type == "continuing":
        xlabel = "Timesteps"
        ylabel = "Average Reward"
    else:
        xlabel = "Timesteps"
        ylabel = "Return"
    title = "Mean Return with Runs"

    fig, ax = _setup_fig(fig, ax, figsize, title, xlim, ylim, xlabel, ylabel)

    # Plot for each data dictionary given
    legend_lines = []
    legend_labels = []
    for i in range(len(data)):
        for _ in range(len(ind[i])):
            if len(data) == 1:
                colours_ = None
            else:
                colours_ = colours[i] * len(ind[i])
                # print(colours_)
            fig, ax, labels, lines = \
                _plot_mean_with_runs(data[i], type_, ind[i], smooth_over[i],
                                     names[i], colours_, figsize, xlim, ylim,
                                     alpha, plot_avg, env_type, fig, ax,
                                     keep_shape, skip=skip, linewidth=linewidth)

            legend_lines.extend(lines)
            legend_labels.extend(labels)

    ax.legend(legend_lines, legend_labels)
    fig.show()

    return fig, ax


def _plot_mean_with_runs(data, type_, ind, smooth_over, name, colours=None,
                         figsize=(12, 6), xlim=None, ylim=None, alpha=0.1,
                         plot_avg=True, env_type="continuing", fig=None,
                         ax=None, keep_shape=False, skip=1, linewidth=1):
    """
    Plots the mean return over all runs and the return for each run for a
    single data dictionary and for each in a list of hyperparameter settings.

    Similar to plot_mean_with_runs, except that this function takes in only
    a single data dictionary.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : iter of int
        The list of hyperparameter settings indices to plot for
        each agent. For example [1, 2] means that the agent plots will use
        hyperparameter settings indices 1 and 2.
    smooth_over : int
        The number of previous data points to smooth over for the agent's
        plot. Note that this is *not* the number of timesteps to smooth over,
        but rather the number of data points to smooth over. For example,
        if you save the return every 1,000 timesteps, then setting this value
        to 15 will smooth over the last 15 readings, or 15,000 timesteps.
    figsize : tuple(int, int)
        The size of the figure to plot
    name : str
        The name of the agents, used for the legend
    colours : list of list of str, optional
        The colours to use for each hyperparameter settings plot for each data
        dictionary, by default None
    xlim : float, optional
        The x limit for the plot, by default None
    ylim : float, optional
        The y limit for the plot, by default None
    alpha : float, optional
        The alpha to use for plots of the runs, by default 0.1
    plot_avg : bool, optional
        If concurrent episodes are executed in each run (e.g. multiple
        evaluation episodes are run at set intervals), then whether to plot the
        performance of each separately or the average performance over all
    env_type : str, optional
        The type of environment, one of 'continuing', 'episodic'. By default
        'continuing'

    Returns
    -------
    tuple of plt.figure, plt.Axes, list of str, list of mpl.Lines2D
        The figure and axis plotted on as well as the list of strings to use
        as labels and the list of lines to include in the legend
    """
    # Set up figure and axis
    fig, ax = _setup_fig(fig, ax, figsize, None, xlim, ylim)

    if colours is None:
        # colours = _get_default_colours(ind)
        colours = [None]

    # Store the info to keep in the legend
    legend_labels = []
    legend_lines = []

    # Plot each selected hyperparameter setting in the data dictionary
    for j in range(len(ind)):
        fig, ax, labels, lines = \
            _plot_mean_with_runs_single_hp(data, type_, ind[j], smooth_over,
                                           name, colours[j], figsize, xlim,
                                           ylim, alpha, plot_avg, env_type,
                                           fig, ax, keep_shape, skip=skip,
                                           linewidth=linewidth)
        legend_labels.extend(labels)
        legend_lines.extend(lines)

    return fig, ax, legend_labels, legend_lines


def _plot_mean_with_runs_single_hp(data, type_, ind, smooth_over, names,
                                   colour=None, figsize=(12, 6), xlim=None,
                                   ylim=None, alpha=0.1, plot_avg=True,
                                   env_type="continuing", fig=None, ax=None,
                                   keep_shape=False, skip=1, linewidth=1):
    """
    Plots the mean return over all runs and the return for each run for a
    single data dictionary and a single hyperparameter setting.

    Similar to _plot_mean_with_runs, except that this function takes in only
    a single data dictionary and a single hyperparameter setting.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : int
        The hyperparameter settings indices to plot for the agent. For example
        5 means that the agent plots will use hyperparameter settings index 5.
    smooth_over : int
        The number of previous data points to smooth over for the agent's
        plot. Note that this is *not* the number of timesteps to smooth over,
        but rather the number of data points to smooth over. For example,
        if you save the return every 1,000 timesteps, then setting this value
        to 15 will smooth over the last 15 readings, or 15,000 timesteps.
    figsize : tuple(int, int)
        The size of the figure to plot
    name : str
        The name of the agents, used for the legend
    colours : list of list of str, optional
        The colours to use for each hyperparameter settings plot for each data
        dictionary, by default None
    xlim : float, optional
        The x limit for the plot, by default None
    ylim : float, optional
        The y limit for the plot, by default None
    alpha : float, optional
        The alpha to use for plots of the runs, by default 0.1
    plot_avg : bool, optional
        If concurrent episodes are executed in each run (e.g. multiple
        evaluation episodes are run at set intervals), then whether to plot the
        performance of each separately or the average performance over all
    env_type : str, optional
        The type of environment, one of 'continuing', 'episodic'. By default
        'continuing'

    Returns
    -------
    tuple of plt.figure, plt.Axes, list of str, list of mpl.Lines2D
        The figure and axis plotted on as well as the list of strings to use
        as labels and the list of lines to include in the legend
    """
    # if env_type == "episodic":
    #     data = runs.expand_episodes(data, ind, type_=type_)

    # Set up figure and axis
    fig, ax = _setup_fig(fig, ax, figsize, None, xlim, ylim)

    if colour is None:
        # colour = _get_default_colours([ind])[0]
        pass

    # Determine the timesteps to plot at
    if type_ == "eval":
        timesteps = \
            data["experiment_data"][ind]["runs"][0]["timesteps_at_eval"]

    elif type_ == "train":
        timesteps_per_ep = \
            data["experiment_data"][ind]["runs"][0]["train_episode_steps"]
        timesteps = exp.get_cumulative_timesteps(timesteps_per_ep)

    # Plot the average reward
    if env_type == "continuing":
        episode_steps = data["experiment"]["environment"]["steps_per_episode"]

    # Get returns
    if env_type == "episodic":
        _check_shifted_rewards_episodic(data, type_, ind)
    else:
        _check_shifted_rewards_continuing(data, type_, ind)
    all_returns = exp.get_returns(data, type_, ind, env_type)

    # If concurrent episodes are run in each run then average them if
    # appropriate
    if type_ == "eval" and plot_avg:
        all_returns = all_returns.mean(axis=-1)
    elif type_ == "eval" and not plot_avg:
        all_returns = np.concatenate(all_returns, axis=1)
        all_returns = all_returns.transpose()
    elif type_ == "train":
        all_returns = np.squeeze(all_returns)

    if all_returns.ndim == 1:
        all_returns = all_returns.reshape(1, -1)

    # Smooth returns
    all_returns = exp.smooth(all_returns, smooth_over, keep_shape)

    # Plot the average reward
    if env_type == "continuing":
        episode_steps = data["experiment"]["environment"]
        episode_steps = episode_steps["steps_per_episode"]
        # all_returns /= episode_steps

    # Determine whether to plot episodes or timesteps on the x-axis, which is
    # dependent on the environment type
    if env_type == "episodic":
        xvalues = np.arange(all_returns.shape[1])  # episodes
    else:
        xvalues = timesteps[:all_returns.shape[1]]

    if colour is None:
        # colour = _get_default_colours([all_returns.shape[0]])
        colour = [_get_default_colours([i])[0] for i in range(all_returns.shape[0])]
    else:
        colour = [colour] * all_returns.shape[0]

    xvalues = xvalues[::skip]

    # Plot each run
    for run in range(all_returns.shape[0]):
        returns = all_returns[run][::skip]
        print(all_returns[run].shape, returns.shape)
        print(run, colour[run])
        ax.plot(xvalues, returns, color=colour[run], linestyle="-",
                alpha=alpha, linewidth=linewidth, label=f"{names} Run {run}")

    # Plot the mean
    mean_colour = "black"
    # mean = all_returns.mean(axis=0)
    # ax.plot(xvalues, mean, color=mean_colour)

    # Store legend identifiers for the run
    legend_labels = []
    legend_lines = []
    legend_labels.append("Individual Runs")
    legend_lines.append(Line2D([0], [0], color=colour[run], linestyle="--",
                               alpha=alpha))

    # Set up the legend variables for the mean over all runs
    label = f"{names}"
    legend_labels.append(label)
    legend_lines.append(Line2D([0], [0], color=mean_colour, linestyle="-"))

    return fig, ax, legend_labels, legend_lines


def mean_with_stderr(data, type_, ind, smooth_over, names,
                     fig=None, ax=None, figsize=(12, 6),
                     xlim=None, ylim=None, alpha=0.1,
                     colours=None, linestyles=None, markers=None,
                     env_type="continuing",
                     keep_shape=False, xlabel="", ylabel="",
                     skip=1, errfn=exp.stderr, linewidth=1,
                     markersize=1, markevery=10):
    """
    Plots the average training or evaluation return over all runs with standard
    error.

    Given a list of data dictionaries of the form returned by main.py, this
    function will plot each episodic return for the list of hyperparameter
    settings ind each data dictionary. The ind argument is a list, where each
    element is a list of hyperparameter settings to plot for the data
    dictionary at the same index as this list. For example, if ind[i] = [1, 2],
    then plots will be generated for the data dictionary at location i
    in the data argument for hyperparameter settings ind[i] = [1, 2].
    The smooth_over argument tells how many previous data points to smooth
    over

    Parameters
    ----------
    data : list of dict
        The Python data dictionaries generated from running main.py for the
        agents
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : iter of iter of int
        The list of lists of hyperparameter settings indices to plot for
        each agent. For example [[1, 2], [3, 4]] means that the first agent
        plots will use hyperparameter settings indices 1 and 2, while the
        second will use 3 and 4.
    smooth_over : list of int
        The number of previous data points to smooth over for the agent's
        plot for each data dictionary. Note that this is *not* the number of
        timesteps to smooth over, but rather the number of data points to
        smooth over. For example, if you save the return every 1,000
        timesteps, then setting this value to 15 will smooth over the last
        15 readings, or 15,000 timesteps. For example, [1, 2] will mean that
        the plots using the first data dictionary will smooth over the past 1
        data points, while the second will smooth over the passed 2 data
        points for each hyperparameter setting.
    fig : plt.figure
        The figure to plot on, by default None. If None, creates a new figure
    ax : plt.Axes
        The axis to plot on, by default None, If None, creates a new axis
    figsize : tuple(int, int)
        The size of the figure to plot
    names : list of str
        The name of the agents, used for the legend
    xlim : float, optional
        The x limit for the plot, by default None
    ylim : float, optional
        The y limit for the plot, by default None
    alpha : float, optional
        The alpha channel for the plot, by default 0.1
    colours : list of list of str
        The colours to use for each hyperparameter settings plot for each data
        dictionary
    env_type : str, optional
        The type of environment, one of 'continuing', 'episodic'. By default
        'continuing'

    Returns
    -------
    plt.figure, plt.Axes
        The figure and axes of the plot
    """
    # Set the colours to be default if not set
    if colours is None:
        colours = _get_default_colours(ind)
    if linestyles is None:
        linestyles = _get_default_styles(ind)
    if markers is None:
        markers = _get_default_markers(ind)

    # Set up figure
    title = f"Average {type_.title()} Return per Run with Standard Error"
    fig, ax = _setup_fig(fig, ax, figsize, xlim=xlim, ylim=ylim, xlabel=xlabel,
                         ylabel=ylabel, title=title)

    # Track the total timesteps per hyperparam setting over all episodes and
    # the cumulative timesteps per episode per data dictionary (timesteps
    # should be consistent between all hp settings in a single data dict)
    total_timesteps = []
    cumulative_timesteps = []

    for i in range(len(data)):
        if type_ == "train":
            cumulative_timesteps.append(exp.get_cumulative_timesteps(data[i]
                                        ["experiment_data"][ind[i][0]]["runs"]
                                        [0]["train_episode_steps"]))
        elif type_ == "eval":
            cumulative_timesteps.append(data[i]["experiment_data"][ind[i][0]]
                                        ["runs"][0]["timesteps_at_eval"])
        else:
            raise ValueError("type_ must be one of 'train', 'eval'")
        total_timesteps.append(cumulative_timesteps[-1][-1])

    # Find the minimum of total trained-for timesteps. Each plot will only
    # be plotted on the x-axis until this value
    min_timesteps = min(total_timesteps)

    # For each data dictionary, find the minimum index where the timestep at
    # that index is >=  minimum timestep
    ind_ge_min_timesteps = []
    for cumulative_timesteps_per_data in cumulative_timesteps:
        final_ind = np.where(cumulative_timesteps_per_data >=
                             min_timesteps)[0][0]
        # Since indexing will stop right before the minimum, increment it
        ind_ge_min_timesteps.append(final_ind + 1)

    # Plot all data for all HP settings, only up until the minimum index
    # fig, ax = None, None
    plot_fn = _plot_mean_with_stderr_continuing if env_type == "continuing" \
        else _plot_mean_with_stderr_episodic
    for i in range(len(data)):
        fig, ax = \
            plot_fn(data=data[i], type_=type_,
                    ind=ind[i], smooth_over=smooth_over[i], name=names[i],
                    fig=fig, ax=ax, figsize=figsize, xlim=xlim, ylim=ylim,
                    last_ind=ind_ge_min_timesteps[i], alpha=alpha,
                    colours=colours[i], keep_shape=keep_shape, skip=skip,
                    linestyles=linestyles[i], markers=markers[i],
                    errfn=errfn, linewidth=linewidth, markersize=markersize,
                    markevery=markevery)

    return fig, ax


def _check_shifted_rewards_continuing(data, type_, ind):
    if "shift" not in data['experiment']['environment']['rewards']:
        # nothing happen
        return

    if type_ == "eval":
        raise ValueError("Cannot plot evaluation returns with shifted reward")

    # otherwise we need to shift the return
    shift = data['experiment']['environment']['rewards']['shift']

    print("Cleaning returns")
    for run in data["experiment_data"][ind]["runs"]:
        episode_returns = run["train_episode_rewards"]
        episode_time_steps = run["train_episode_steps"]
        # print("Episode returns: ", len(episode_returns))
        # print("Episode time steps: ", len(episode_time_steps))
        correction = episode_time_steps * shift
        episode_returns -= correction
        # print("Correction: ", correction)
        run["train_episode_rewards"] = episode_returns


def _plot_mean_with_stderr_continuing(data, type_, ind, smooth_over, fig=None,
                                      ax=None, figsize=(12, 6), xlim=None,
                                      ylim=None, xlabel=None, ylabel=None,
                                      name="", last_ind=-1,
                                      timestep_multiply=None, alpha=0.1,
                                      colours=None, linestyles=None,
                                      markers=None,
                                      keep_shape=False,
                                      skip=1,
                                      errfn=exp.stderr,
                                      linewidth=1,
                                      markersize=1,
                                      markevery=10):
    """
    Plots the average training or evaluation return over all runs for a single
    data dictionary on a continuing environment. Standard error
    is plotted as shaded regions.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : iter of int
        The list of hyperparameter settings indices to plot
    smooth_over : int
        The number of previous data points to smooth over. Note that this
        is *not* the number of timesteps to smooth over, but rather the number
        of data points to smooth over. For example, if you save the return
        every 1,000 timesteps, then setting this value to 15 will smooth
        over the last 15 readings, or 15,000 timesteps.
    fig : plt.figure
        The figure to plot on, by default None. If None, creates a new figure
    ax : plt.Axes
        The axis to plot on, by default None, If None, creates a new axis
    figsize : tuple(int, int)
        The size of the figure to plot
    name : str, optional
        The name of the agent, used for the legend
    last_ind : int, optional
        The index of the last element to plot in the returns list,
        by default -1. This is useful if you want to plot many things on the
        same axis, but all of which have a different number of elements. This
        way, we can plot the first last_ind elements of each returns for each
        agent.
    timestep_multiply : array_like of float, optional
        A value to multiply each timstep by, by default None. This is useful if
        your agent does multiple updates per timestep and you want to plot
        performance vs. number of updates.
    xlim : float, optional
        The x limit for the plot, by default None
    ylim : float, optional
        The y limit for the plot, by default None
    alpha : float, optional
        The alpha channel for the plot, by default 0.1
    colours : list of str
        The colours to use for each plot of each hyperparameter setting
    env_type : str, optional
        The type of environment, one of 'continuing', 'episodic'. By default
        'continuing'

    Returns
    -------
    plt.figure, plt.Axes
        The figure and axes of the plot

    Raises
    ------
    ValueError
        When an axis is passed but no figure is passed
        When an appropriate number of colours is not specified to cover all
        hyperparameter settings
    """
    if skip == 0:
        skip = 1

    if colours is not None and len(colours) != len(ind):
        raise ValueError("must have one colour for each hyperparameter " +
                         "setting")
    if linestyles is not None and len(linestyles) != len(ind):
        raise ValueError("must have one linestyle for each hyperparameter " +
                         "setting")
    if markers is not None and len(markers) != len(ind):
        raise ValueError("must have one marker for each hyperparameter " +
                         "setting")

    if timestep_multiply is None:
        timestep_multiply = [1] * len(ind)

    if ax is not None and fig is None:
        raise ValueError("must pass figure when passing axis")

    if colours is None:
        colours = _get_default_colours(ind)
    if linestyles is None:
        linestyles = _get_default_styles(ind)
    if markers is None:
        markers = _get_default_markers(ind)

    # Set up figure
    if ax is None and fig is None:
        title = f"Average {type_.title()} Return per Run with Standard Error"
        fig, ax = _setup_fig(fig, ax, figsize, xlim=xlim, ylim=ylim,
                             xlabel=xlabel, ylabel=ylabel, title=title)

    episode_length = data["experiment"]["environment"]["steps_per_episode"]

    # Plot with the standard error
    for i in range(len(ind)):
        _check_shifted_rewards_continuing(data, type_, ind[i])
        timesteps, mean, std = exp.get_mean_err(data, type_, ind[i],
                                                smooth_over,
                                                # exp.t_ci,
                                                errfn,
                                                keep_shape=keep_shape)
        if last_ind != -1:
            timesteps = timesteps[:last_ind]
        timesteps = np.array(timesteps) * timestep_multiply[i]
        # mean = mean[:last_ind] / episode_length
        # std = std[:last_ind] / episode_length
        timesteps = timesteps[::skip]
        mean = mean[::skip]
        std = std[::skip]

        print(mean.shape, std.shape, "HERE")
        print("Final perf:", mean[-10:].mean())

        # dof = len(data["experiment_data"][ind[i]]["runs"])-1
        # shaded_min = [st.t(loc=mean[i], scale=std[i], df=dof).ppf(0.025)
        #               for i in range(len(mean))]
        # shaded_max = [st.t(loc=mean[i], scale=std[i], df=dof).ppf(0.975)
        #               for i in range(len(mean))]
        # std = np.stack([shaded_min, shaded_max])

        # Plot based on colours
        label = f"{name}"
        colour = colours[i] if colours is not None else None
        linestyle = linestyles[i] if linestyles is not None else None
        marker = markers[i] if markers is not None else None
        _plot_shaded(ax, timesteps, mean, std, colour, label, alpha,
                        linestyle=linestyle, marker=marker,
                        linewidth=linewidth, markersize=markersize,
                        markevery=markevery)

    ax.legend()

    fig.show()
    return fig, ax


def _check_shifted_rewards_episodic(data, type_, ind):
    if "gamma" not in data['experiment']['environment']['rewards']:
        # nothing happen
        return

    if type_ == "eval":
        raise ValueError("Cannot plot evaluation returns with shifted reward")

    # otherwise we need to shift the return
    shift = data['experiment']['environment']['rewards']['shift']
    gamma = data['experiment']['environment']['rewards']['gamma']
    steps_per_episode = data['experiment']['environment']['steps_per_episode']

    print("Cleaning returns")
    for run in data["experiment_data"][ind]["runs"]:
        episode_returns = run["train_episode_rewards"]
        episode_time_steps = run["train_episode_steps"]
        # print("Episode returns: ", len(episode_returns))
        # print("Episode time steps: ", len(episode_time_steps))
        correction = (1 - gamma**(steps_per_episode - episode_time_steps + 1)) / (1 - gamma)
        correction += episode_time_steps - 1
        correction *= shift
        episode_returns -= correction
        # print("Correction: ", correction)
        run["train_episode_rewards"] = episode_returns


def _plot_mean_with_stderr_episodic(data, type_, ind, smooth_over, fig=None,
                                    ax=None, figsize=(12, 6), name="",
                                    last_ind=-1, xlabel="Timesteps",
                                    ylabel="Average Return", xlim=None,
                                    ylim=None, alpha=0.1, colours=None,
                                    keep_shape=False, skip=1, linestyles=None,
                                    markers=None, errfn=exp.stderr,
                                    linewidth=1, markersize=1, markevery=10):
    """
    Plots the average training or evaluation return over all runs for a
    single data dictionary on an episodic environment. Plots shaded retions
    as standard error.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : iter of int
        The list of hyperparameter settings indices to plot
    smooth_over : int
        The number of previous data points to smooth over. Note that this
        is *not* the number of timesteps to smooth over, but rather the number
        of data points to smooth over. For example, if you save the return
        every 1,000 timesteps, then setting this value to 15 will smooth
        over the last 15 readings, or 15,000 timesteps.
    fig : plt.figure The figure to plot on, by default None.
    If None, creates a new figure
    ax : plt.Axes
        The axis to plot on, by default None, If None, creates a new axis
    figsize : tuple(int, int)
        The size of the figure to plot
    name : str, optional
        The name of the agent, used for the legend
    xlim : float, optional
        The x limit for the plot, by default None
    ylim : float, optional
        The y limit for the plot, by default None
    alpha : float, optional
        The alpha channel for the plot, by default 0.1
    colours : list of str
        The colours to use for each plot of each hyperparameter setting
    env_type : str, optional
        The type of environment, one of 'continuing', 'episodic'. By default
        'continuing'

    Returns
    -------
    plt.figure, plt.Axes
        The figure and axes of the plot

    Raises
    ------
    ValueError
        When an axis is passed but no figure is passed
        When an appropriate number of colours is not specified to cover all
        hyperparameter settings
    """
    if colours is not None and len(colours) != len(ind):
        raise ValueError("must have one colour for each hyperparameter " +
                         "setting")
    if linestyles is not None and len(colours) != len(ind):
        raise ValueError("must have one linestyle for each hyperparameter " +
                         "setting")
    if markers is not None and len(markers) != len(ind):
        raise ValueError("must have one marker for each hyperparameter " +
                         "setting")

    if ax is not None and fig is None:
        raise ValueError("must pass figure when passing axis")

    if colours is None:
        colours = _get_default_colours(ind)
    if linestyles is None:
        linestyles = _get_default_styles(ind)
    if markers is None:
        markers = _get_default_markers(ind)

    # Set up figure
    if ax is None and fig is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        print("DEBUG: xlim is None")
    if ylim is not None:
        ax.set_ylim(ylim)

    # Plot with the standard error
    for i in range(len(ind)):
        _check_shifted_rewards_episodic(data, type_, i)

        # data = exp.reduce_episodes(data, ind[i], type_=type_)
        if type_ == "train":
            data = runs.expand_episodes(data, ind[i], type_=type_, skip=skip)

        # data has consistent # of episodes, so treat as env_type="continuing"
        _, mean, std = exp.get_mean_err(data, type_, ind[i], smooth_over,
                                        errfn,
                                        keep_shape=keep_shape)
        print(mean.shape, std.shape, "HERE")
        episodes = np.arange(mean.shape[0])
        print(mean.shape, episodes[0], episodes[-1])
        print("Final perf:", mean[-10:].mean())

        # dof = len(data["experiment_data"][ind[i]]["runs"])-1
        # shaded_min = [st.t(loc=mean[i], scale=std[i], df=dof).ppf(0.025)
        #               for i in range(len(mean))]
        # shaded_max = [st.t(loc=mean[i], scale=std[i], df=dof).ppf(0.975)
        #               for i in range(len(mean))]
        # std = np.stack([shaded_min, shaded_max])

        # Plot based on colours
        label = f"{name}"
        if colours is not None and linestyles is not None:
            _plot_shaded(ax, episodes, mean, std, colours[i], label, alpha,
                         linestyle=linestyles[i], marker=markers[i],
                         linewidth=linewidth, markersize=markersize,
                         markevery=markevery)
        elif colours is not None:
            _plot_shaded(ax, episodes, mean, std, colours[i], label, alpha,
                         marker=markers[i], linewidth=linewidth,
                         markersize=markersize, markevery=markevery)
        elif linestyles is not None:
            _plot_shaded(ax, episodes, mean, std, None, label, alpha,
                         linestyle=linestyles[i], marker=markers[i],
                         linewidth=linewidth, markersize=markersize,
                         markevery=markevery)
        else:
            _plot_shaded(ax, episodes, mean, std, None, label, alpha,
                         marker=markers[i], linewidth=linewidth,
                         markersize=markersize, markevery=markevery)

    ax.legend()
    # ax.set_title(f"Average {type_.title()} Return per Run with Standard Error")
    # ax.set_ylabel(ylabel)
    # ax.set_xlabel(xlabel)

    # fig.show()
    return fig, ax


def plot_mean_with_stderr_summary(data, type_, labels,
                                  key='alpha', values=[0.0001, 0.001, 0.01, 0.1],
                                  figsize=(12, 6), xlim=None, ylim=None,
                                  alpha=0.1, colours=None, linestyles=None,
                                  markers=None, env_type="continuing",
                                  keep_shape=False, xlabel="", ylabel="",
                                  errfn=exp.stderr, linewidth=1,
                                  markersize=1, to=-1, fontsize=12,
                                  tune_on_last_x_percent=1.0, title=None):
    # find best hypers for each alpha
    def hold_constant(in_data, constant_dict):
        _config = hypers.hold_constant(in_data, constant_dict)[1]
        return hypers.renumber(in_data, _config)

    def get_best_hp(data):
        best_hp, settings = hypers.best(data, to=to, perf=type_,
                                        tune_on_last_x_percent=tune_on_last_x_percent)
        return best_hp

    if to != -1:
        raise ValueError("to must be -1 for this function")

    means, stds = [], []    # mean and standard error for each data on each alpha
    for i, d in enumerate(data):
        means_per_data, stds_per_data = [], []
        for value in values:
            result = hold_constant(d, {key: value})
            best_hp = get_best_hp(result)
            returns = exp.get_returns(result, type_, best_hp, env_type=env_type)
            # extract the last tune_on_last_x_percent of the data
            returns = returns[:, int(returns.shape[1] * (1 - tune_on_last_x_percent)):]
            returns = returns.mean(axis=1)  # 0 for each run, 1 for each step, 2 for nothing?

            std = errfn(returns, axis=0)
            mean = returns.mean(axis=0)

            means_per_data.append(mean)
            stds_per_data.append(std)
        means.append(means_per_data)
        stds.append(stds_per_data)
    means = np.array(means).squeeze()
    stds = np.array(stds).squeeze()

    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=fontsize+2)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # Plot with the standard error
    for i in range(len(data)):
        ax.plot(values, means[i], label=labels[i], color=colours[i][0],
                linestyle=linestyles[i][0], linewidth=linewidth,
                markersize=markersize)
        ax.errorbar(values, means[i], yerr=stds[i], fmt='o', color=colours[i][0],
                    linestyle=linestyles[i][0], markersize=markersize)
        ax.set_xscale('log')

    ax.legend(fontsize=fontsize-2)
    return fig, ax


def return_distribution(data, type_, hp_ind, bins, figsize=(12, 6), xlim=None,
                        ylim=None, after=0, before=-1):
    """
    Plots the distribution of returns on either an episodic or continuing
    environment

    Parameters
    ----------
    data : dict
        The data dictionary containing the runs of a single hyperparameter
        setting
    type_ : str, optional
        The type of surface to plot, by default "surface". One of 'surface',
        'wireframe', or 'bar'
    hp_ind : int, optional
        The hyperparameter settings index in the data dictionary to use for
        the plot, by default -1. If less than 0, then the first hyperparameter
        setting in the dictionary is used.
    bins : Iterable, int
        The bins to use for the plot. If an Iterable, then each value in the
        Iterable is considered as a cutoff for bins. If an integer, separates
        the returns into that many bins
    figsize : tuple, optional
        The size of the figure to plot, by default (15, 10)
    xlim : 2-tuple of float, optional
        The cutoff points for the x-axis to plot between, by default None
    ylim : 2-tuple of float, optional
        The cutoff points for the y-axis to plot between, by default None

    Returns
    -------
    plt.figure, plt.Axes3D
        The figure and axis plotten on
    """
    # Get the episode returns for each run
    run_returns = []
    return_type = type_ + "_episode_rewards"
    for run in data["experiment_data"][hp_ind]["runs"]:
        run_returns.append(np.mean(run[return_type][after:before]))

    title = f"Learning Curve Distribution - HP Settings {hp_ind}"
    return _return_histogram(run_returns, bins, figsize, title, xlim, ylim)


def _return_histogram(run_returns, bins, figsize, title, xlim, ylim, kde=True):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    ax.set_title(title)
    ax.set_xlabel("Average Return Per Run")
    ax.set_ylabel("Relative Frequency")
    _ = sns.histplot(run_returns, bins=bins, kde=kde)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Plot relative frequency on the y-axis
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:
                                 "{:.2f}".format(x / len(run_returns))))

    fig.show()
    return fig, ax


def _get_default_markers(iter_):
    markers = []

    # Calculate the number of lists at the current level to go through
    paths = range(len(iter_))

    # Return a list of colours if the elements of the list are not lists
    if not isinstance(iter_[0], Iterable):
        return ["" for i in paths]

    # For each list at the current level, get the colours corresponding to
    # this level
    for i in paths:
        markers.append(_get_default_markers(iter_[i]))

    return markers


def _get_default_styles(iter_):
    styles = []

    # Calculate the number of lists at the current level to go through
    paths = range(len(iter_))

    # Return a list of colours if the elements of the list are not lists
    if not isinstance(iter_[0], Iterable):
        return ["-" for i in paths]

    # For each list at the current level, get the colours corresponding to
    # this level
    for i in paths:
        styles.append(_get_default_styles(iter_[i]))

    return styles


def _get_default_colours(iter_):
    """
    Recursively turns elements of an Iterable into strings representing
    colours.

    This function will turn each element of an Iterable into strings that
    represent colours, recursively. If the elements of an Iterable are
    also Iterable, then this function will recursively descend all the way
    through every Iterable until it finds an Iterable with non-Iterable
    elements. These elements will be replaced by strings that represent
    colours. In effect, this function keeps the data structure, but replaces
    non-Iterable elements by strings representing colours. Note that this
    funcion assumes that all elements of an Iterable are of the same type,
    and so it only checks if the first element of an Iterable object is
    Iterable or not to stop the recursion.

    Parameters
    ----------
    iter_ : collections.Iterable
        The top-level Iterable object to turn into an Iterable of strings of
        colours, recursively.

    Returns
    -------
    list of list of ... of strings
        A data structure that has the same architecture as the input Iterable
        but with all non-Iterable elements replaced by strings.
    """
    colours = []

    # Calculate the number of lists at the current level to go through
    paths = range(len(iter_))

    # Return a list of colours if the elements of the list are not lists
    if not isinstance(iter_[0], Iterable):
        global OFFSET
        col = [DEFAULT_COLOURS[(OFFSET + i) % len(DEFAULT_COLOURS)]
               for i in paths]
        OFFSET += len(paths)
        return col

    # For each list at the current level, get the colours corresponding to
    # this level
    for i in paths:
        colours.append(_get_default_colours(iter_[i]))

    return colours


def _plot_shaded(ax, x, y, region, colour, label, alpha, linestyle="-",
                 marker="", linewidth=1, markersize=1, markevery=10):
    """
    Plots a curve with a shaded region.

    Parameters
    ----------
    ax : plt.Axes
        The axis to plot on
    x : Iterable
        The points on the x-axis
    y : Iterable
        The points on the y-axis
    region : list or array_like
        The region to shade about the y points. The shaded region will be
        y +/- region. If region is a list or 1D np.ndarray, then the region
        is used both for the + and - portions. If region is a 2D np.ndarray,
        then the first row will be used as the lower bound (-) and the
        second row will be used for the upper bound (+). That is, the region
        between (lower_bound, upper_bound) will be shaded, and there will be
        no subtraction/adding of the y-values.
    colour : str
        The colour to plot with
    label : str
        The label to use for the plot
    alpha : float
        The alpha value for the shaded region
    """
    if colour is None:
        colour = DEFAULT_COLOURS[0]

    ax.plot(x, y, color=colour, label=label, linestyle=linestyle,
            marker=marker, linewidth=linewidth, markersize=markersize,
            markevery=markevery)
    if type(region) == list:
        ax.fill_between(x, y-region, y+region, alpha=alpha, color=colour)
    elif type(region) == np.ndarray and len(region.shape) == 1:
        ax.fill_between(x, y-region, y+region, alpha=alpha, color=colour)
    elif type(region) == np.ndarray and len(region.shape) == 2:
        ax.fill_between(x, region[0, :], region[1, :], alpha=alpha,
                        color=colour)


def _setup_fig(fig, ax, figsize=None, title=None, xlim=None, ylim=None,
               xlabel=None, ylabel=None, xscale=None, yscale=None, xbase=None,
               ybase=None):
    if fig is None:
        if ax is not None:
            raise ValueError("Must specify figure when axis given")
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()

    if ax is None:
        ax = fig.add_subplot()

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xscale is not None:
        if xbase is not None:
            ax.set_xscale(xscale, base=xbase)
        else:
            ax.set_xscale(xscale)

    if yscale is not None:
        if ybase is not None:
            ax.set_yscale(yscale, base=ybase)
        else:
            ax.set_yscale(yscale)

    return fig, ax


def reset():
    """
    Resets the colours offset
    """
    global OFFSET
    OFFSET = 0
