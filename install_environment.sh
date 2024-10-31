#!/bin/bash

$user_id=$1
$pi_id="def-whitem"

module load python/3.9.6 mujoco
virtualenv --no-download ~/policy_params

source ~/policy_params/bin/activate
mkdir -p ~/Downloads
wget -P ~/Downloads https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -zvxf ~/Downloads/mujoco210-linux-x86_64.tar.gz -C ~/Downloads/
export MUJOCO_PY_MUJOCO_PATH=/home/$user_id/Downloads/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$user_id/Downloads/mujoco210/bin:/usr/lib/nvidia

cd ~/project/$pi_id/$user_id/policy-params
pip install --no-index --upgrade pip
pip install -r requirements.txt
pip install gym==0.21.0 dm_control==1.0.11 mujoco-py==2.1.2.14
