#!/bin/bash

#SBATCH --job-name=lqr_dec_s2
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH -o /network/scratch/l/lia/skae/lqr_dec_s2-%A_%a.out
#SBATCH --array=0-143
#SBATCH --requeue

# ============================================================================
# Stage 2: Practical head-to-head (main decision stage)
# Arms: BD*, AH-PRAG
#   BD* defaults to bd_c1; override with BD_STAR=bd_c2.
# Seeds: 0..7
# B_proxy: 8,13,20
# target_size: 128,256,512
# System: Lyapunov
# ============================================================================

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BD_STAR="${BD_STAR:-bd_c1}"
if [ "${BD_STAR}" != "bd_c1" ] && [ "${BD_STAR}" != "bd_c2" ]; then
  echo "Invalid BD_STAR='${BD_STAR}'. Expected bd_c1 or bd_c2"
  exit 1
fi

ARMS=("${BD_STAR}" ah_prag)
TARGET_SIZES=(128 256 512)
B_PROXIES=(8 13 20)
SEEDS=(0 1 2 3 4 5 6 7)

idx=$SLURM_ARRAY_TASK_ID
seed_idx=$(( idx % ${#SEEDS[@]} ))
idx=$(( idx / ${#SEEDS[@]} ))
b_idx=$(( idx % ${#B_PROXIES[@]} ))
idx=$(( idx / ${#B_PROXIES[@]} ))
ts_idx=$(( idx % ${#TARGET_SIZES[@]} ))
idx=$(( idx / ${#TARGET_SIZES[@]} ))
arm_idx=$(( idx % ${#ARMS[@]} ))

ARM="${ARMS[$arm_idx]}"
TARGET_SIZE="${TARGET_SIZES[$ts_idx]}"
B_PROXY="${B_PROXIES[$b_idx]}"
SEED="${SEEDS[$seed_idx]}"

export STAGE=2
export SYSTEM=lyapunov
export TARGET_SIZE
export B_PROXY
export ARM
export SEED

export BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lqr_decision}"
export DEVICE="${DEVICE:-cuda}"
export NUM_STEPS="${NUM_STEPS:-10000}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export DIM="${DIM:-8}"
export NUM_BASINS_GT="${NUM_BASINS_GT:-13}"

# AH-PRAG lock values (explicitly frozen from current repo practice)
export LAMBDA_GLOBAL_PRAG="${LAMBDA_GLOBAL_PRAG:-1e-4}"
export LAMBDA_LOCAL_PRAG="${LAMBDA_LOCAL_PRAG:-1e-3}"

echo "Stage 2 task ${SLURM_ARRAY_TASK_ID}: arm=${ARM}, ts=${TARGET_SIZE}, b_proxy=${B_PROXY}, seed=${SEED}"
echo "BD*=${BD_STAR}, AH-PRAG lambda_global=${LAMBDA_GLOBAL_PRAG}, lambda_local=${LAMBDA_LOCAL_PRAG}"

bash scripts/run_lqr_decision_trial.sh
