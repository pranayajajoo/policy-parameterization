# Plot mean with standard error
# Input to script is the data files to consider

import pickle
import seaborn as sns
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import hypers
import plot_utils as plot
import matplotlib as mpl

mpl.rcParams["font.size"] = 8
mpl.rcParams["svg.fonttype"] = "none"

CONTINUING = "continuing"
EPISODIC = "episodic"
type_map = {
        "MountainCarContinuous-v0": EPISODIC,
        "Pendulum-v1": CONTINUING,
        "Acrobot-v1": EPISODIC,
        }

env = "Acrobot-v1"

DATA_FILES = [
    f"results/SGD_optimizer/Acrobot-v1_epsgreedy_SGD_DeterministicAction/data.pkl",
]


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot()

DATA = []
for f in tqdm(DATA_FILES):
    with open(f, "rb") as infile:
        d = pickle.load(infile)
    DATA.append(d)

# Find best hypers
BEST_IND = []
for agent in DATA:
    best_hp, settings = hypers.best(agent, to=-1, perf="train")
    print(agent["experiment_data"][best_hp]["agent_hyperparams"])
    BEST_IND.append(best_hp)

# Generate plot labels
labels = [
    "eps greedy",
]

CMAP = "tab10"
colours = ["black", "red", "blue", "gold"]
colours = list(map(lambda x: [x], colours))
plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=sns.color_palette(CMAP))

# Plot the mean + standard error
print("=== Plotting mean with standard error")
PLOT_TYPE = "train"
TYPE = "online" if PLOT_TYPE == "train" else "offline"
best_ind = list(map(lambda x: [x], BEST_IND))

fig, ax = plot.mean_with_stderr(
    DATA,
    PLOT_TYPE,
    best_ind,
    [0]*len(best_ind),
    
    labels,
    env_type=type_map[env].lower(),
    figsize=(6, 4),
    colours=colours,
    fig=fig,
    ax=ax,
)

ax.set_title("Acrobot-v1 (no initial exploration)")
ax.legend()

# create directory if it doesn't exist
if not os.path.exists("./plots/png"):
    os.makedirs("./plots/png")
fig.savefig(f"./plots/SGD_actor/Acrobot-v1_epsgreedy_DeterministicAction_woexploration.png", bbox_inches="tight")
print(f"Saved to ./plots/SGD_actor/Pendulum-v0_epsgreedy_DeterministicAction_woexploration.png")
