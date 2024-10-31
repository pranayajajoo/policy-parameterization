#!/bin/bash

user_id="jiamin"

module load StdEnv/2020
module load python/3.9.6 mujoco
virtualenv --no-download ~/policy_params

source ~/policy_params/bin/activate
mkdir -p ~/Downloads
wget -P ~/Downloads https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -zvxf ~/Downloads/mujoco210-linux-x86_64.tar.gz -C ~/Downloads/
export MUJOCO_PY_MUJOCO_PATH=/home/$user_id/Downloads/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$user_id/Downloads/mujoco210/bin:/usr/lib/nvidia

pip install --no-index --upgrade pip
pip install -r requirements_cc.txt
