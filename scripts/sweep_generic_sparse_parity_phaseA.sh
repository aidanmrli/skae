#!/bin/bash
#
# Phase-A matched anchor sweep for generic_sparse.
# Grid: 4 systems x 3 seeds = 12 runs (same systems/seeds as parity depth pilot).
#
# Submit:
#   sbatch scripts/sweep_generic_sparse_parity_phaseA.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/lista_parity_generic_sparse
#   ANCHOR_ROOT=/network/scratch/l/lia/skae/lista_parity_generic_sparse/phaseA_anchor_generic_sparse
#   NUM_STEPS=10000 BATCH_SIZE=256
#   DYSTS_CACHE_PROFILE=full DYSTS_CACHE_DIR=/network/scratch/l/lia/skae/dysts_native_cache
#
#SBATCH --job-name=gs_parity_a
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/gs-parity-a-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-11

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_parity_generic_sparse}"
ANCHOR_ROOT="${ANCHOR_ROOT:-${BASE_OUT}/phaseA_anchor_generic_sparse}"
mkdir -p "${ANCHOR_ROOT}"

NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE:-full}"
DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS:-4}"
DYSTS_CACHE_REUSE="${DYSTS_CACHE_REUSE:-1}"

SYSTEM_KEYS=(
  "dysts:Duffing"
  "dysts:Sakarya"
  "multiwell_gradient"
  "multiwell_strong_transition"
)
SYSTEM_LABELS=(
  "Duffing"
  "Sakarya"
  "multiwell_gradient"
  "multiwell_strong_transition"
)
SEEDS=(0 1 2)

NUM_SYSTEMS=${#SYSTEM_KEYS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_SYSTEMS * NUM_SEEDS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_JOBS )); then
  echo "Task ${TASK_ID} out of range for TOTAL_JOBS=${TOTAL_JOBS}. Exiting."
  exit 0
fi

SYSTEM_IDX=$((TASK_ID / NUM_SEEDS))
SEED_IDX=$((TASK_ID % NUM_SEEDS))

SYSTEM=${SYSTEM_KEYS[$SYSTEM_IDX]}
SYSTEM_LABEL=${SYSTEM_LABELS[$SYSTEM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

LOG_DIR="${ANCHOR_ROOT}/${SYSTEM_LABEL}/seed_${SEED}"

echo "============================================="
echo "Generic Sparse Parity Anchor Phase-A"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "System: ${SYSTEM}"
echo "System Label: ${SYSTEM_LABEL}"
echo "Seed: ${SEED}"
echo "Start Time: $(date)"
echo "ANCHOR_ROOT: ${ANCHOR_ROOT}"
echo "LOG_DIR: ${LOG_DIR}"
echo "============================================="

TRAIN_ARGS=(
  --config generic_sparse
  --env "${SYSTEM}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --sequence_length 1
  --eval_profile "${EVAL_PROFILE}"
  --seed "${SEED}"
  --device cuda
  --log_dir "${LOG_DIR}"
)

if [[ "${SYSTEM}" == dysts:* ]]; then
  TRAIN_ARGS+=(
    --standardize
    --dysts_ic_noise_scale 0.2
    --dysts_native_cache
    --dysts_cache_profile "${DYSTS_CACHE_PROFILE}"
    --dysts_cache_dir "${DYSTS_CACHE_DIR}"
    --dysts_cache_num_workers "${DYSTS_CACHE_NUM_WORKERS}"
  )
  if [[ "${DYSTS_CACHE_REUSE}" == "1" ]]; then
    TRAIN_ARGS+=(--dysts_cache_reuse)
  fi
fi

uv run python tools/train.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
