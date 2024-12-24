#!/bin/bash

user_id="pranayaj"

module load StdEnv/2020
module load python/3.9.6 mujoco
source ~/policy_params/bin/activate
export MUJOCO_PY_MUJOCO_PATH=/home/$user_id/Downloads/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$user_id/Downloads/mujoco210/bin:/usr/lib/nvidia
