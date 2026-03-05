#!/bin/bash
#
# Simple-environment forecasting benchmark: generic_sparse with sequence loss (L=8).
# Grid: 13 systems x 3 seeds = 39 runs.
#
# Submit:
#   sbatch scripts/sweep_simple_envs_seq8_generic_sparse.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/simple_envs_seq8/generic_sparse
#   NUM_STEPS=10000 BATCH_SIZE=256 TARGET_SIZE=256
#   RECONST_COEFF=0.5 PRED_COEFF=1.0 SPARSITY_COEFF=0.01
#   SEQUENCE_LENGTH=8 EVAL_PROFILE=full
#
# NOTE:
#   This script intentionally avoids observation/action normalization flags.
#
#SBATCH --job-name=simple_gs_s8
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/simple-gs-s8-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-38

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/simple_envs_seq8/generic_sparse}"
mkdir -p "${BASE_OUT}"

NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
RECONST_COEFF="${RECONST_COEFF:-0.5}"
PRED_COEFF="${PRED_COEFF:-1.0}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.01}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

SYSTEMS=(
  parabolic
  duffing
  lotka_volterra
  pendulum
  lorenz63
  multiwell_gradient
  multiwell_rotational
  multiwell_energy
  multiwell_strong_transition
  multiwell_gradient_hd
  multiwell_rotational_hd
  multiwell_energy_hd
  multiwell_strong_transition_hd
)
SEEDS=(0 1 2)

NUM_SYSTEMS=${#SYSTEMS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_SYSTEMS * NUM_SEEDS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_JOBS )); then
  echo "Task ${TASK_ID} out of range for TOTAL_JOBS=${TOTAL_JOBS}. Exiting."
  exit 0
fi

SYSTEM_IDX=$((TASK_ID / NUM_SEEDS))
SEED_IDX=$((TASK_ID % NUM_SEEDS))

SYSTEM=${SYSTEMS[$SYSTEM_IDX]}
SEED=${SEEDS[$SEED_IDX]}
LOG_DIR="${BASE_OUT}/${SYSTEM}/seed_${SEED}"

echo "============================================="
echo "Simple Envs Sequence-L8 | Generic Sparse"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "System: ${SYSTEM}"
echo "Seed: ${SEED}"
echo "Start Time: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "LOG_DIR: ${LOG_DIR}"
echo "============================================="

TRAIN_ARGS=(
  --config generic_sparse
  --env "${SYSTEM}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --target_size "${TARGET_SIZE}"
  --reconst_coeff "${RECONST_COEFF}"
  --pred_coeff "${PRED_COEFF}"
  --sparsity_coeff "${SPARSITY_COEFF}"
  --sequence_length "${SEQUENCE_LENGTH}"
  --eval_profile "${EVAL_PROFILE}"
  --seed "${SEED}"
  --device cuda
  --log_dir "${LOG_DIR}"
)

uv run python tools/train.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
