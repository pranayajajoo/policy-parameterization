#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=/home/pranayaj/scratch/slurm_trash/%x-%j.out
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --job-name='eps-greedy-uniform-exploration-sweep'
#SBATCH --account='rrg-whitem'


source ~/.bashrc 
cd /home/pranayaj/projects/def-whitem/pranayaj/scratch/work/policy-parameterization
user_id="pranayaj"

module load StdEnv/2020
module load python/3.9.6 mujoco
source ~/policy_params/bin/activate
export MUJOCO_PY_MUJOCO_PATH=/home/$user_id/Downloads/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$user_id/Downloads/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${PWD}:$PYTHONPATH


# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 0 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 1 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 2 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 3 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 4 

python main.py --save-dir nn_optimizer/eps-greedy-best-hypers-woexploration --agent-json config/agent/epsgreedy_best_acrobot.json --env-json config/environment/Acrobot-v1.json --index 5 &
# python main.py --save-dir nn_optimizer/eps-greedy-best-hypers-woexploration --agent-json config/agent/epsgreedy_best_acrobot.json --env-json config/environment/Acrobot-v1.json --index 6 &
# python main.py --save-dir nn_optimizer/eps-greedy-best-hypers-woexploration --agent-json config/agent/epsgreedy_best_acrobot.json --env-json config/environment/Acrobot-v1.json --index 7 &
python main.py --save-dir nn_optimizer/eps-greedy-best-hypers-woexploration --agent-json config/agent/epsgreedy_best_acrobot.json --env-json config/environment/Acrobot-v1.json --index 8 &
# python main.py --save-dir nn_optimizer/eps-greedy-best-hypers-woexploration --agent-json config/agent/epsgreedy_best_acrobot.json --env-json config/environment/Acrobot-v1.json --index 9

# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 10 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 11 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 12 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 13 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 14 

# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 15 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 16 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 17 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 18 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 19

# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 20 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 21 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 22 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 23 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 24 

# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 25 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 26 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 27 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 28 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 29

# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 30 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 31 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 32 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 33 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 34 

# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 35 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 36 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 37 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 38 &
# python main.py --save-dir results/jan5/eps-greedy-uniform-exploration-sweep --agent-json config/agent/epsgreedy_classic_sweep.json --env-json config/environment/Acrobot-v1_dense.json --index 39


# Benchmark CLOSER with chi + increase decrease Q
# python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.7 --dual_alpha=0.001 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_0.001_alpha_0.7_tau_norm_obs & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.7 --dual_alpha=0.01 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_0.01_alpha_0.7_tau_norm_obs & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.7 --dual_alpha=0.1 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_0.1_alpha_0.7_tau_norm_obs & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.7 --dual_alpha=0.5 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_0.5_alpha_0.7_tau_norm_obs & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.7 --dual_alpha=1.0 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_1.0_alpha_0.7_tau_norm_obs & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.7 --dual_alpha=5.0 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_5.0_alpha_0.7_tau_norm_obs & 

# python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.1 --dual_alpha=0.001 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_0.001_alpha_0.1_tau_norm_obs & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.1 --dual_alpha=0.01 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_0.01_alpha_0.1_tau_norm_obs & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.1 --dual_alpha=0.1 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_0.1_alpha_0.1_tau_norm_obs & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.1 --dual_alpha=0.5 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_0.5_alpha_0.1_tau_norm_obs & 
# # python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.9 --dual_alpha=1.0 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_1.0_alpha_0.9_tau_norm_obs & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.9 --dual_alpha=5.0 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_5.0_alpha_0.9_tau_norm_obs & 


# python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.1 --dual_alpha=1.0 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_1.0_alpha_0.1_tau_norm_obs &
# python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.2 --dual_alpha=1.0 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_1.0_alpha_0.2_tau_norm_obs &
# python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.3 --dual_alpha=1.0 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_1.0_alpha_0.3_tau_norm_obs &
# python main.py --env-name=hopper-random-v2 --log-dir=dump --normalize_observations --value_loss=chi --tau 0.4 --dual_alpha=1.0 --seed=0 --exp_name=corr_chi_increase_expert_v_decrease_agent_v_1.0_alpha_0.4_tau_norm_obs &


# Benchmark CLOSER with gumber + increase decrease Q
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --value_loss=gumbel --normalize_observations --tau 1.0 --dual_alpha=0.001 --seed=0 --exp_name=gumbel_corr_increase_expert_v_decrease_agent_v_0.001_alpha_norm_obs & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --value_loss=gumbel --normalize_observations --tau 1.0 --dual_alpha=0.01 --seed=0 --exp_name=gumbel_corr_increase_expert_v_decrease_agent_v_0.01_alpha_norm_obs & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --value_loss=gumbel --normalize_observations --tau 1.0 --dual_alpha=0.1 --seed=0 --exp_name=gumbel_corr_increase_expert_v_decrease_agent_v_0.1_alpha_norm_obs & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --value_loss=gumbel --normalize_observations --tau 1.0 --dual_alpha=1.0 --seed=0 --exp_name=gumbel_corr_increase_expert_v_decrease_agent_v_1.0_alpha_norm_obs & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --value_loss=gumbel --normalize_observations --tau 1.0 --dual_alpha=5.0 --seed=0 --exp_name=gumbel_corr_increase_expert_v_decrease_agent_v_5.0_alpha_norm_obs & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --value_loss=gumbel --normalize_observations --tau 1.0 --dual_alpha=10.0 --seed=0 --exp_name=gumbel_corr_increase_expert_v_decrease_agent_v_10.0_alpha_norm_obs & 





# Benchmark CLOSER with vanilla increase decrease Q on mujoco

# python main.py --env-name=antmaze-umaze-v2 --normalize_observations --log-dir=dump --dual_alpha=0.001 --exp_name=corr_imitate_increase_expert_v_decrease_agent_v_0.001_alpha & 
# python main.py --env-name=antmaze-umaze-v2 --normalize_observations --log-dir=dump --dual_alpha=0.01 --exp_name=corr_imitate_increase_expert_v_decrease_agent_v_0.01_alpha & 
# python main.py --env-name=antmaze-umaze-v2 --normalize_observations --log-dir=dump --dual_alpha=0.1 --exp_name=corr_imitate_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=antmaze-umaze-v2 --normalize_observations --log-dir=dump --dual_alpha=0.5 --exp_name=corr_imitate_increase_expert_v_decrease_agent_v_0.5_alpha & 
# python main.py --env-name=antmaze-umaze-v2 --normalize_observations --log-dir=dump --dual_alpha=1.0 --exp_name=corr_imitate_increase_expert_v_decrease_agent_v_1.0_alpha & 

# python main.py --env-name=antmaze-umaze-v2 --log-dir=dump --dual_alpha=2.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha & 
# python main.py --env-name=antmaze-umaze-v2 --log-dir=dump --dual_alpha=5.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha & 


# python main.py --env-name=antmaze-umaze-v2 --log-dir=dump --dual_alpha=1.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 
# python main.py --env-name=antmaze-umaze-v2 --log-dir=dump --dual_alpha=2.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha & 
# python main.py --env-name=antmaze-umaze-v2 --log-dir=dump --dual_alpha=5.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha & 

# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=2.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=5.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha &

# sleep 2
# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=0.001 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.001_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=0.01 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.01_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=0.1 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.5_alpha & 


# python main.py --env-name=hopper-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.001 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.001_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.01 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.01_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.1 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.5 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.5_alpha & 

# python main.py --env-name=hopper-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.001 --seed=1 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.001_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.001 --seed=2 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.001_alpha & 

# python main.py --env-name=hopper-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.01 --seed=1 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.01_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.01 --seed=2 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.01_alpha & 

# python main.py --env-name=hopper-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.1 --seed=1 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.1 --seed=2 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.1_alpha & 

# python main.py --env-name=hopper-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.5 --seed=1 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.5_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.5 --seed=2 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.5_alpha & 


# python main.py --env-name=walker2d-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.001 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.001_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.001 --seed=1 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.001_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.001 --seed=2 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.001_alpha & 

# python main.py --env-name=walker2d-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.01 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.01_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.01 --seed=1 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.01_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.01 --seed=2 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.01_alpha & 

# python main.py --env-name=walker2d-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.1 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.1 --seed=1 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.1 --seed=2 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.1_alpha & 

# python main.py --env-name=walker2d-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.5 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.5_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.5 --seed=1 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.5_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.5 --seed=2 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.5_alpha & 



# python main.py --env-name=ant-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.001 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.001_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.001 --seed=1 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.001_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.001 --seed=2 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.001_alpha & 

# python main.py --env-name=ant-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.01 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.01_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.01 --seed=1 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.01_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.01 --seed=2 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.01_alpha & 

# python main.py --env-name=ant-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.1 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.1 --seed=1 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.1 --seed=2 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.1_alpha & 

# python main.py --env-name=ant-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.5 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.5_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.5 --seed=1 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.5_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.5 --seed=2 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.5_alpha & 



# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.001 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.001_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.01 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.01_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.1 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --value_loss=iql_rank --dual_alpha=0.5 --seed=0 --exp_name=imitate_rank_increase_expert_v_decrease_agent_v_0.5_alpha & 



# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=0.001 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.001_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=0.01 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.01_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=0.1 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.5_alpha & 


# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=0.1 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=0.1 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.1_alpha & 

# python main.py --env-name=ant-random-v2 --log-dir=dump --dual_alpha=0.1 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --dual_alpha=0.1 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.1_alpha & 

# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=0.1 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=0.1 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.1_alpha & 

# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=0.1 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=0.1 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.1_alpha & 



# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.5_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.5_alpha & 

# python main.py --env-name=ant-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.5_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.5_alpha & 

# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.5_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.5_alpha & 

# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.5_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.5_alpha & 


# Hyperparam tuning with normalized observations, with and without annealing lr, and with and without policy lr = 3e-5
# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=0 --normalize_observations --exp_name=imitate_increase_expert_v_decrease_agent_v_1.0_alpha_norm_obs & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=0 --normalize_observations --no-policy-scheduler --exp_name=imitate_increase_expert_v_decrease_agent_v_1.0_alpha_norm_obs_nops & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=0 --normalize_observations --no-policy-scheduler --policy-learning-rate=3e-5 --exp_name=imitate_increase_expert_v_decrease_agent_v_1.0_alpha_norm_obs_nops_plr_1e-5 & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=0 --normalize_observations  --policy-learning-rate=3e-5 --exp_name=imitate_increase_expert_v_decrease_agent_v_1.0_alpha_norm_obs_plr_1e-5 & 

# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=0 --normalize_observations --exp_name=imitate_increase_expert_v_decrease_agent_v_1.0_alpha_norm_obs & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=0 --normalize_observations --no-policy-scheduler --exp_name=imitate_increase_expert_v_decrease_agent_v_1.0_alpha_norm_obs_nops & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=0 --normalize_observations --no-policy-scheduler --policy-learning-rate=3e-5 --exp_name=imitate_increase_expert_v_decrease_agent_v_1.0_alpha_norm_obs_nops_plr_1e-5 & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=0 --normalize_observations  --policy-learning-rate=3e-5 --exp_name=imitate_increase_expert_v_decrease_agent_v_1.0_alpha_norm_obs_plr_1e-5 & 





# python main.py --env-name=ant-random-v2 --log-dir=dump --dual_alpha=0.0001 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.0001_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --dual_alpha=0.001 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.001_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --dual_alpha=0.01 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.01_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --dual_alpha=0.1 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.5_alpha & 



# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=3 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 

# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=2.0 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=2.0 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=2.0 --seed=3 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha & 

# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=5.0 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=5.0 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=5.0 --seed=3 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha & 

# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=7.0 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_7_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=7.0 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_7_alpha & 
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=7.0 --seed=3 --exp_name=imitate_increase_expert_v_decrease_agent_v_7_alpha & 



# python main.py --env-name=ant-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 
# python main.py --env-name=ant-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=3 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 

# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=7.0 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_7_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=7.0 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_7_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=7.0 --seed=3 --exp_name=imitate_increase_expert_v_decrease_agent_v_7_alpha & 



# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=3 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 

# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=10.0 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_10_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=10.0 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_10_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=10.0 --seed=3 --exp_name=imitate_increase_expert_v_decrease_agent_v_10_alpha & 




# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=0.001 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.001_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=0.01 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.01_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=0.1 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.1_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=0.2 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.2_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=0.5 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.5_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=0.7 --seed=0 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.7_alpha & 





# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 
# python main.py --env-name=halfcheetah-random-v2 --log-dir=dump --dual_alpha=1.0 --seed=3 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 




# python main.py --env-name=halfcheetah-random-timed-v2 --log-dir=dump --dual_alpha=1.0 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 
# python main.py --env-name=halfcheetah-random-timed-v2 --log-dir=dump --dual_alpha=1.0 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 
# python main.py --env-name=halfcheetah-random-timed-v2 --log-dir=dump --dual_alpha=1.0 --seed=3 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha & 

# python main.py --env-name=halfcheetah-random-timed-v2 --log-dir=dump --dual_alpha=10.0 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_10_alpha & 
# python main.py --env-name=halfcheetah-random-timed-v2 --log-dir=dump --dual_alpha=10.0 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_10_alpha & 
# python main.py --env-name=halfcheetah-random-timed-v2 --log-dir=dump --dual_alpha=10.0 --seed=3 --exp_name=imitate_increase_expert_v_decrease_agent_v_10_alpha & 


# python main.py --env-name=halfcheetah-random-timed-v2 --log-dir=dump --dual_alpha=2.0 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha & 
# python main.py --env-name=halfcheetah-random-timed-v2 --log-dir=dump --dual_alpha=2.0 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha & 
# python main.py --env-name=halfcheetah-random-timed-v2 --log-dir=dump --dual_alpha=2.0 --seed=3 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha & 

# python main.py --env-name=halfcheetah-random-timed-v2 --log-dir=dump --dual_alpha=5.0 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha & 
# python main.py --env-name=halfcheetah-random-timed-v2 --log-dir=dump --dual_alpha=5.0 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha & 
# python main.py --env-name=halfcheetah-random-timed-v2 --log-dir=dump --dual_alpha=5.0 --seed=3 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha & 


# python main.py --env-name=ant-random-timed-v2 --log-dir=dump --dual_alpha=2.0 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha & 
# python main.py --env-name=ant-random-timed-v2 --log-dir=dump --dual_alpha=2.0 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha & 
# python main.py --env-name=ant-random-timed-v2 --log-dir=dump --dual_alpha=2.0 --seed=3 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha & 

# python main.py --env-name=ant-random-timed-v2 --log-dir=dump --dual_alpha=5.0 --seed=1 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha & 
# python main.py --env-name=ant-random-timed-v2 --log-dir=dump --dual_alpha=5.0 --seed=2 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha & 
# python main.py --env-name=ant-random-timed-v2 --log-dir=dump --dual_alpha=5.0 --seed=3 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha & 




# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=2.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha & 
# python main.py --env-name=hopper-random-v2 --log-dir=dump --dual_alpha=5.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha &
# sleep 2
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=1.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha &
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=2.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha &
# python main.py --env-name=walker2d-random-v2 --log-dir=dump --dual_alpha=5.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha &

# python main.py --env-name=ant-random-v2 --log-dir=dump --dual_alpha=1.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha &
# python main.py --env-name=ant-random-v2 --log-dir=dump --dual_alpha=2.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha &
# python main.py --env-name=ant-random-v2 --log-dir=dump --dual_alpha=5.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha &

# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --dual_alpha=1.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --dual_alpha=2.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --dual_alpha=5.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha &

# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --normalize_observations --dual_alpha=1.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_1_alpha_normalize &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --normalize_observations --dual_alpha=2.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_2_alpha_normalize &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --normalize_observations --dual_alpha=5.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_5_alpha_normalize &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --normalize_observations --dual_alpha=7.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_7_alpha_normalize &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --normalize_observations --dual_alpha=10.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_10_alpha_normalize &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --normalize_observations --dual_alpha=15.0 --exp_name=imitate_increase_expert_v_decrease_agent_v_15_alpha_normalize &


# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --beta=0.5 --normalize_observations --max-episode-steps=280 --dual_alpha=0.01 --exp_name=corr_terminal_imitate_increase_expert_v_decrease_agent_v_0.01_alpha_normalize_beta_0.5 &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --beta=0.5 --normalize_observations --max-episode-steps=280 --dual_alpha=0.1 --exp_name=corr_terminal_imitate_increase_expert_v_decrease_agent_v_0.1_alpha_normalize_beta_0.5 &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --beta=0.5 --normalize_observations --max-episode-steps=280 --dual_alpha=1.0 --exp_name=corr_terminal_imitate_increase_expert_v_decrease_agent_v_1_alpha_normalize_beta_0.5 &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --beta=0.5 --normalize_observations --max-episode-steps=280 --dual_alpha=2.0 --exp_name=corr_terminal_imitate_increase_expert_v_decrease_agent_v_2_alpha_normalize_beta_0.5 &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --beta=0.5 --normalize_observations --max-episode-steps=280 --dual_alpha=5.0 --exp_name=corr_terminal_imitate_increase_expert_v_decrease_agent_v_5_alpha_normalize_beta_0.5 &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --beta=0.5 --normalize_observations --max-episode-steps=280 --dual_alpha=7.0 --exp_name=corr_terminal_imitate_increase_expert_v_decrease_agent_v_7_alpha_normalize_beta_0.5 &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --beta=0.5 --normalize_observations --max-episode-steps=280 --dual_alpha=10.0 --exp_name=corr_terminal_imitate_increase_expert_v_decrease_agent_v_10_alpha_normalize_beta_0.5 &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --beta=0.5 --normalize_observations --max-episode-steps=280 --dual_alpha=15.0 --exp_name=corr_terminal_imitate_increase_expert_v_decrease_agent_v_15_alpha_normalize_beta_0.5 &




# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --beta=0.5 --normalize_observations --max-episode-steps=280 --dual_alpha=0.001 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.001_alpha_normalize_beta_0.5_wd_1e-4 &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --beta=0.5 --normalize_observations --max-episode-steps=280 --dual_alpha=0.01 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.01_alpha_normalize_beta_0.5_wd_1e-4 &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --beta=0.5 --normalize_observations --max-episode-steps=280 --dual_alpha=0.1 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.1_alpha_normalize_beta_0.5_wd_1e-4 &
# python main.py --env-name=kitchen-mixed-v0 --log-dir=dump --beta=0.5 --normalize_observations --max-episode-steps=280 --dual_alpha=0.5 --exp_name=imitate_increase_expert_v_decrease_agent_v_0.5_alpha_normalize_beta_0.5_wd_1e-4 &


wait