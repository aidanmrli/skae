#!/bin/bash

#SBATCH --job-name=lqr_dec_s1
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH -o /network/scratch/l/lia/skae/lqr_dec_s1-%A_%a.out
#SBATCH --array=0-35
#SBATCH --requeue

# ============================================================================
# Stage 1: Structure-effect isolation
# Arms: BD-C1, BD-C2, AH-ISO
# Seeds: 0,1,2,3
# B_proxy: 8,13,20
# target_size: 256
# System: Lyapunov
# ============================================================================

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

ARMS=(bd_c1 bd_c2 ah_iso)
B_PROXIES=(8 13 20)
SEEDS=(0 1 2 3)

idx=$SLURM_ARRAY_TASK_ID
seed_idx=$(( idx % ${#SEEDS[@]} ))
idx=$(( idx / ${#SEEDS[@]} ))
b_idx=$(( idx % ${#B_PROXIES[@]} ))
idx=$(( idx / ${#B_PROXIES[@]} ))
arm_idx=$(( idx % ${#ARMS[@]} ))

ARM="${ARMS[$arm_idx]}"
B_PROXY="${B_PROXIES[$b_idx]}"
SEED="${SEEDS[$seed_idx]}"

export STAGE=1
export SYSTEM=lyapunov
export TARGET_SIZE=256
export B_PROXY
export ARM
export SEED

export BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lqr_decision}"
export DEVICE="${DEVICE:-cuda}"
export NUM_STEPS="${NUM_STEPS:-10000}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export DIM="${DIM:-8}"
export NUM_BASINS_GT="${NUM_BASINS_GT:-13}"

echo "Stage 1 task ${SLURM_ARRAY_TASK_ID}: arm=${ARM}, b_proxy=${B_PROXY}, seed=${SEED}"

bash scripts/run_lqr_decision_trial.sh
