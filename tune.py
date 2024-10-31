import utils.runs as runs
import pickle
from pprint import pprint
import seaborn as sns
from copy import deepcopy
import functools
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
import json
import sys
import plot_utils as plot
import matplotlib as mpl
import hypers
from copy import deepcopy
from glob import glob


global alg
global discrete_hypers
optimal_policy = {
        "MountainCar-v0": -83.,
        "MountainCarContinuous-v0": -65.,
        "PendulumContinuous-v0": 930.1230848912774,
        "PendulumDiscrete-v0": 932.5350252370129,
        "Acrobot-v1": -56.,
}


def is_discrete(data):
    return "discrete" in data["experiment"]["agent"]["agent_name"].lower()


def match_hypers(data):
    if "greedyac" not in data[0]["experiment"]["agent"]["agent_name"].lower():
        raise NotImplementedError

    discrete_indices = []
    for i, d in enumerate(data):
        if not is_discrete(d):
            cont_data = d

        if is_discrete(d):
            discrete_indices.append(i)

    for j in discrete_indices:
        disc_data = data[j]

        new = {}
        new["experiment_data"] = {}
        new["experiment"] = disc_data["experiment"]

        for ch in cont_data["experiment_data"].keys():
            cont_hyper = cont_data["experiment_data"][ch]["agent_hyperparams"]

            for dh in disc_data["experiment_data"].keys():
                disc_hyper = \
                    disc_data["experiment_data"][dh]["agent_hyperparams"]

                same = (
                    cont_hyper["critic_lr"] == disc_hyper["critic_lr"] and
                    cont_hyper["actor_lr_scale"] ==
                    disc_hyper["actor_lr_scale"]
                )

                if same:
                    new["experiment_data"][ch] = \
                        disc_data["experiment_data"][dh]
                    new["experiment_data"][ch]["agent_hyperparams"] = \
                        cont_data["experiment_data"][ch]["agent_hyperparams"]

        data[j] = new

    return data


def best_hyper(data):
    scores = np.zeros((len(data), len(data[0]["experiment_data"])))
    scores += np.finfo(np.float64).min

    for i, d in enumerate(data):
        for h in d["experiment_data"].keys():
            perf = hypers.get_performance(d, h, repeat=False)
            run_perf = []
            for run in range(perf.shape[0]):
                run_perf.append(np.mean(perf[run]))

            run_perf = np.array(run_perf)

            # ################################
            # Get the optimal return
            # ################################
            env = d['experiment']['environment']['env_name']
            config = d['experiment']['environment']
            continuous = "Continuous" if (("continuous" in config and
                                           config["continuous"]) or
                                          "continuous" in env) else "Discrete"
            if "pendulum" in env.lower():
                env = env[:8] + continuous + env[8:]

            optimal_return = optimal_policy[env]
            # ################################

            # Normalize performance
            run_perf = optimal_return - run_perf
            run_perf /= np.abs(optimal_return)
            run_perf = 1 - run_perf
            run_perf = run_perf.mean()

            scores[i, h] = run_perf

    # Find the best hypers per-environment
    best = np.argmax(scores, axis=1)
    print("=== Best Per-Env Hypers ===")
    for i, hyper in enumerate(best):
        env_config = data[i]["experiment"]["environment"]
        env_name = env_config["env_name"]
        continuous = "continuous" in env_name.lower()
        continuous = continuous or (
            "continuous" in env_config and
            env_config["continuous"]
        )

        full_hy = data[i]["experiment_data"][hyper]["agent_hyperparams"]
        hy = {}
        hy["actor_lr_scale"] = full_hy["actor_lr_scale"]
        hy["alpha"] = full_hy["alpha"]
        hy["critic_lr"] = full_hy["critic_lr"]

        print("==========")
        print(env_name, "continuous" if continuous else "discrete")
        pprint(hy)
        print()

    # Get the best hypers across-env
    print("=== Best Across-Env Hypers ===")
    scores = scores.mean(axis=0)
    best = np.argmax(scores)

    # Print out the best score, as well as the hyper setting for both the
    # continuous and discrete versions
    if alg == "GreedyAC":
        for d in data:
            if not is_discrete(d):
                setting = d["experiment_data"][best]["agent_hyperparams"]
                pprint(setting)
                print("Continuous:", best)

                indices = hypers.index_of(discrete_hypers, setting)
                print("Discrete:", indices)
                pprint(hypers.sweeps(discrete_hypers, indices[0]))
                return

    else:
        setting = d["experiment_data"][best]["agent_hyperparams"]
        pprint(setting)
        print("Continuous:", best)
        return


if __name__ == "__main__":
    global alg
    alg = "SAC"
    data_files = glob(f"results/fixedEnt/*{alg}*")

    if alg == "vac":
        data_files.extend(glob(f"results/fixedEnt/*VAC*"))

    data = []

    for dir__ in data_files:
        with open(os.path.join(dir__, "data.pkl"), "rb") as infile:
            single_data = pickle.load(infile)
            runs.to(single_data, 10)
            data.append(single_data)

        if is_discrete(single_data):
            global discrete_hypers
            discrete_hypers = deepcopy(
                single_data["experiment"]["agent"]["parameters"]
            )

    if alg == "GreedyAC":
        data = match_hypers(data)

    best_hyper(data)
