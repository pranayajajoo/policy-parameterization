# policy-params

# Dependencies

## Local

To install the dependencies locally, run the following command:
```
pip install -r requirements.txt
```

You can run this command in a virtual environment to avoid conflicts with other projects.

## Compute Canada

Before running the below commands, make sure to change the `$user_id` in the `install_environment.sh` and `activate_environment.sh` files to your Compute Canada user id.

To install the dependencies on Compute Canada, run the following command:
```
bash install_environment.sh
```
This will install the dependencies in a virtual environment in the `~/policy_params` directory. If it does not install correctly in the first run, try to rerun the commands from the script.

To activate the virtual environment, run the following command:
```
. activate_environment.sh
```

# Usage
The file main.py trains an agent for a specified number of runs, based on an environment and agent configuration file count in config/environment/ or config/agent/ respectively. The data is saved in the results directory, with a name similar to the environment and agent name.

For more information on how to use the main.py program, see the `--help` option:
```
Usage: main.py [OPTIONS]

  Given an environment, agent name, and an integer index, trains the agent on
  the environment for the hyperparameter setting with corresponding to the
  integer index. Indices higher than the number of hyperparameter settings
  will wrap around and perform subsequent runs. For example, if there are 10
  hyperparameter settings and the integer index is 11, then this will
  correspond to the 2nd run of the first hyperparameter setting, with a
  different random seed from the first run

Options:
  --env-json TEXT      Path to the environment json configuration file
                       [required]
  --agent-json TEXT    Path to the agent json configuration file  [required]
  --index INTEGER      The index of the hyperparameter to run
  -m, --monitor        Whether or not to render the scene as the agent trains.
  -a, --after INTEGER  How many timesteps (training) should pass before
                       rendering the scene
  --save-dir TEXT      Which directory to save the results file in
  --help               Show this message and exit.
```

Example:
```
python main.py --agent-json config/agent/SAC_classic_best_pendulum.json --env-json config/environment/PendulumContinuous-v1.json --index 0
```

# Hyperparameter settings
The hyperparameter settings are laid out in the agent configuration files.
The files are laid out such that each setting is a list of values, and the
total number of hyperparameter settings is the product of the lengths of each
of these lists. For example, if the agent config file looks like:
```
{
    "agent_name": "linearAC",
    "parameters":
    {
        "decay": [0.5],
        "critic_lr": [0.005, 0.1, 0.3],
        "actor_lr": [0.005, 0.1, 0.3],
        "avg_reward_lr": [0.1, 0.3, 0.5, 0.9],
        "scaled": [true],
        "clip_stddev": [1000]
    }
}
```
then, there are `1 x 3 x 3 x 4 x 1 x 1 = 36` different hyperparameter
settings. Each hyperparameter setting is given a specific index. For example
hyperparameter setting index `1` would have the following hyperparameters:
```
{
    "agent_name": "linearAC",
    "parameters":
    {
        "decay": 0.5,
        "critic_lr": 0.005,
        "actor_lr": 0.005,
        "avg_reward_lr": 0.1,
        "scaled": true,
        "clip_stddev": 1000
    }
}
```
The hyperparameter settings indices are actually implemented `mod x`,
where `x` is the maximum number of hyperparameter settings (in the example
about, `36`). So, in the example above, the hyperparameter settings with
indices `1, 37, 73, ...` all refer to the same hyperparameter settings since
`1 = 37 = 73 = ... mod 36`. The difference is that the consecutive indices
have a different seed. So, each time we run experiments with hyperparameter
setting `1`, it will have the same seed. If we run with hyperparameter setting
`37`, it will be the same hyperparameter settings as `1`, but with a different
seed, and this seed will be the same every time we run the experiment with
hyperparameter settings `37`. This is what Martha and her students
have done with their Actor-Expert implementation, and I find that it works
nicely for hyperparameter sweeps.

How the runs are done is the following: imagine that the maximum number of
hyperparameter settings is `x`. Then, when you call `./main.py ... --runs y`
from the command line, the file will run all hyperparameter settings
from `0` to `x * y` stepping by a value of `1`. This way, we run each
hyperparameter setting for `y` runs, effectively sweeping over the
hyperparameters. In addition, if we ever want to run a specific run again
with the exact same seed and hyperparameters, it is very easy.

# Scheduling on Compute Canada
```
python schedule.py --runs $NUM_RUNS clusters/CLUSTER_FILE main.py ./ $ENV_CONFIG $SAVE_DIR $AGENT_CONFIG
```

For example, you can sweep over the hyperparameters for the SAC agent on the Pendulum environment with the following command:
```
python schedule.py --runs 10 clusters/cedar.json main.py ./ config/environment/PendulumContinuous-v1.json classic_sweep config/agent/SAC_classic_sweep.json
```

The results will be saved in the `results/classic_sweep` directory.

# Saved Data
__Slightly outdated, but still gives the general idea of how data is saved__

Each experiment saves all the data as a Python dictionary. The dictionary is
designed so that we store all information about the experiment, including all
agent hyperparameters and environment settings so that the experiment is
exactly reproducible.

If the data dictionary is called `data`, then the main data for the experiment
is stored in `data["experiment_data"]`, which is a dictionary mapping from
hyperparameter settings indices to agent parameters and experiment runs.
`data["experiment_data"][i]["agent_params"]` is a dictionary storing the
agent's hyperparameters (hyperparameter settings index `i`) for the experiment.
`data["experiment_data"][i]["runs]` is a list storing the runs for the
`i-th` hyperparameter setting. Each element of the list is a dictionary, giving
all the information for that run and hyperparameter setting. For example,
`data["experiment_data"][i]["runs"][j]` will give all the information on
the `j-th` run of hyperparameter settings `i`.

Below is a tree diagram of the data structure:
```
data
├─── "experiment"
│       ├─── "environment": environment configuration file
│       └─── "agent": agent configuration file
└─── "experiment_data": dictionary of hyperparameter setting *index* to runs
        ├─── "agent_params": the hyperparameters settings
        └─── "runs": a list containing all the runs for this hyperparameter setting (each run is a dictionary of elements)
                └─── index i: information on the ith run
                		├─── "run_number": the run number
                        ├─── "random_seed": the random seed used for the run
                        ├─── "total_timesteps": the total number of timesteps in the run
                        ├─── "eval_interval_timesteps": the interval of timesteps to pass before running offline evaluation
                        ├─── "episodes_per_eval": the number of episodes run at each offline evaluation
                        ├─── "eval_episode_rewards": list of the returns (np.array) from each evaluation episode if there are 10 episodes per eval,
                        │     then this will be a list of np.arrays where each np.array has 10 elements (one per eval episode)
                        ├─── "eval_episode_steps": the number of timesteps per evaluation episode, with the same form as "eval_episode_rewards"
                        ├─── "timesteps_at_eval": the number of training steps that passed at each evaluation. For example, if there were 10
                        │    offline evaluations, then this will be a list of 10 integers, each stating how many training steps passed before each
                        │    evaluation.
                        ├─── "train_episode_rewards": the return seen for each training episode
                        ├─── "train_episode_steps": the number of timesteps passed for each training episode
                        ├─── "train_time": the total amount of training time in seconds
                        ├─── "eval_time": the total amount of evaluation time in seconds
                        ├─── "total_train_episodes": the total number of training episodes for the run
                        ├─── "learned_params": dict of all parameters learned during training, this includes weights, average reward, entropy, etc. when appropriate
                        ├─── "train_states": the state seen at each timestep during training
                        ├─── "train_rewards": the rewards seen at each timestep during training
                        ├─── "train_actions": the actions taken at each timestep during training
						├─── "eval_states": the state seen at each timestep during evaluation
                        ├─── "update_states": the states used in each update
                        ├─── "update_actions": the actions used in each update
                        ├─── "update_rewards": the rewards used in each update
                        └─── "update_next_states": the next states used in each update
```

For example, here is `data["experiment_data"][i]["runs"][j]` for a mock run
of the Linear-Gaussian Actor-Critic agent on MountainCarContinuous-v0:
```
{'random_seed': 0,
 'total_timesteps': 1000,
 'eval_interval_timesteps': 500,
 'episodes_per_eval': 10,
 'eval_episode_rewards': array([[-200., -200., -200., -200., -200., -200., -200., -200., -200.,
         -200.],
        [-200., -200., -200., -200., -200., -200., -200., -200., -200.,
         -200.]]),
 'eval_episode_steps': array([[200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
        [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]]),
 'timesteps_at_eval': array([  0, 600]),
 'train_episode_steps': array([200, 200, 200, 200, 200]),
 'train_episode_rewards': array([-200., -200., -200., -200., -200.]),
 'train_time': 0.12098526954650879,
 'eval_time': 0.044415950775146484,
 'total_train_episodes': 5,
 ...}
```

# Configuration files
Each configuration file is a JSON file and has a few properties. There
are also templates in each configuration directory for the files.

# Plotting

## Preprocessing

Each run is saved as a sperate dictionary in a JSON file. To plot the data, we need to preprocess the data into a single dictionary. This can be done using the `simple_combine.py` script. The script takes in the path to the directory containing the JSON files and the path to the output file. For example:
```
python simple_combine.py results/Pendulum-v1_SAC_SquashedGaussian
```

It will save the combined data in the `results/Pendulum-v1_SAC_SquashedGaussian/data.json` file.

## Plotting

The `utils/plot_mse_data.py` script can be used to plot the data. You will need to modify the script to change the data that is plotted. Right now, it plots the average return for the best hyperparameter setting for each agent for the results stored in `results/Pendulum-v1_SAC_SquashedGaussian/data.json`. By running
```
python utils/plot_mse_data.py
```

it will plot the data and save the plot in the `plots/png/Pendulum-v1.png` file.

This is just an example of how to plot the data. There are also other plotting scripts in the `utils/plot_utils.py` file that can be used to plot different plots.

# ToDos

- [ ] We should likely begin be upgrading from `gym` to `gymnasium`, and
  upgrading from `-v3` to `-v4` Mujoco environments.
