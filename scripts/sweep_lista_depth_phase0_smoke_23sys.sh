#!/bin/bash
#
# Phase-0 smoke validation for LISTA depth-first plan:
# 4 systems x depth=3 x seed=0.
#
# Submit:
#   sbatch scripts/sweep_lista_depth_phase0_smoke_23sys.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/lista_depth_first_23sys
#   NUM_STEPS=1000 BATCH_SIZE=256 TARGET_SIZE=256
#   DYSTS_CACHE_PROFILE=smoke
#   DYSTS_CACHE_STEPS=5000 DYSTS_CACHE_TRAJECTORIES=32 DYSTS_CACHE_WARMUP=200
#
#SBATCH --job-name=lista_d0_smoke
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH -o /network/scratch/l/lia/skae/lista-d0-smoke-23sys-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-3

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_depth_first_23sys}"
PHASE1_ROOT="${PHASE1_ROOT:-${BASE_OUT}/phase1_depth}"
mkdir -p "${PHASE1_ROOT}"

NUM_STEPS="${NUM_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.10}"
LISTA_ALPHA="${LISTA_ALPHA:-0.15}"
RECONST_COEFF="${RECONST_COEFF:-0.5}"
PRED_COEFF="${PRED_COEFF:-1.0}"
DEPTH="${DEPTH:-3}"
SEED="${SEED:-0}"
EVAL_PROFILE="${EVAL_PROFILE:-smoke}"

DYSTS_CACHE_STEPS="${DYSTS_CACHE_STEPS:-5000}"
DYSTS_CACHE_TRAJECTORIES="${DYSTS_CACHE_TRAJECTORIES:-32}"
DYSTS_CACHE_WARMUP="${DYSTS_CACHE_WARMUP:-200}"
DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE:-smoke}"
DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS:-4}"
DYSTS_CACHE_REUSE="${DYSTS_CACHE_REUSE:-1}"

SYSTEM_KEYS=(
  "dysts:Dadras"
  "dysts:Chua"
  "multiwell_gradient"
  "multiwell_gradient_hd"
)

SYSTEM_LABELS=(
  "Dadras"
  "Chua"
  "multiwell_gradient"
  "multiwell_gradient_hd"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#SYSTEM_KEYS[@]} )); then
  echo "Task ${TASK_ID} out of range for smoke set size ${#SYSTEM_KEYS[@]}. Exiting."
  exit 0
fi

SYSTEM=${SYSTEM_KEYS[$TASK_ID]}
SYSTEM_LABEL=${SYSTEM_LABELS[$TASK_ID]}

LOG_DIR="${PHASE1_ROOT}/depth_${DEPTH}/${SYSTEM_LABEL}/seed_${SEED}"

echo "============================================="
echo "LISTA Depth-First Phase-0 Smoke"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "Depth: ${DEPTH}"
echo "System: ${SYSTEM}"
echo "System Label: ${SYSTEM_LABEL}"
echo "Seed: ${SEED}"
echo "Eval Profile: ${EVAL_PROFILE}"
echo "Start Time: $(date)"
echo "PHASE1_ROOT: ${PHASE1_ROOT}"
echo "LOG_DIR: ${LOG_DIR}"
echo "============================================="

TRAIN_ARGS=(
  --config lista_nonlinear
  --env "${SYSTEM}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --target_size "${TARGET_SIZE}"
  --reconst_coeff "${RECONST_COEFF}"
  --pred_coeff "${PRED_COEFF}"
  --sparsity_coeff "${SPARSITY_COEFF}"
  --lista_alpha "${LISTA_ALPHA}"
  --lista_num_loops "${DEPTH}"
  --lista_final_op relu
  --pairwise
  --eval_every 250
  --eval_num_steps 100
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
    --dysts_cache_steps "${DYSTS_CACHE_STEPS}"
    --dysts_cache_trajectories "${DYSTS_CACHE_TRAJECTORIES}"
    --dysts_cache_warmup "${DYSTS_CACHE_WARMUP}"
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
