#!/bin/bash
#
# Generic Sparse (MLP encoder) forecasting benchmark on 15 dysts systems x 3 seeds.
#
# Submit:
#   sbatch scripts/sweep_dysts_forecast_generic_sparse_ts256_cachebench.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/dysts_forecast_generic_sparse_ts256_cachebench
#   NUM_STEPS=10000 BATCH_SIZE=256 TARGET_SIZE=256 SPARSITY_COEFF=0.01
#   DYSTS_CACHE_PROFILE=full
#   DYSTS_CACHE_DIR=/network/scratch/l/lia/skae/dysts_native_cache
#   DYSTS_CACHE_NUM_WORKERS=4 DYSTS_CACHE_REUSE=1
#
#SBATCH --job-name=dysts_gs_ts256
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/dysts-gs-ts256-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-44

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/dysts_forecast_generic_sparse_ts256_cachebench}"
mkdir -p "${BASE_OUT}"

NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.01}"
RECONST_COEFF="${RECONST_COEFF:-0.5}"
PRED_COEFF="${PRED_COEFF:-1.0}"

DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE:-full}"
DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS:-4}"
DYSTS_CACHE_REUSE="${DYSTS_CACHE_REUSE:-1}"

SYSTEMS=(
  Dadras
  Duffing
  QiChen
  Sakarya
  SprottTorus
  Chua
  MultiChua
  DequanLi
  LuChenCheng
  SanUmSrisuchinwong
  WangSun
  ShimizuMorioka
  LorenzCoupled
  RikitakeDynamo
  Hadley
)
SEEDS=(0 1 2)

NUM_SEEDS=${#SEEDS[@]}
SYS_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

SYSTEM=${SYSTEMS[$SYS_IDX]}
SEED=${SEEDS[$SEED_IDX]}

if [[ -z "${SYSTEM}" || -z "${SEED}" ]]; then
  echo "Invalid mapping for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

LOG_DIR="${BASE_OUT}/${SYSTEM}/seed_${SEED}"

echo "============================================="
echo "Generic Sparse Forecast Benchmark (Dysts)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "System: ${SYSTEM}"
echo "Seed: ${SEED}"
echo "Start Time: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "LOG_DIR: ${LOG_DIR}"
echo "============================================="

TRAIN_ARGS=(
  --config generic_sparse
  --env "dysts:${SYSTEM}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --target_size "${TARGET_SIZE}"
  --reconst_coeff "${RECONST_COEFF}"
  --pred_coeff "${PRED_COEFF}"
  --sparsity_coeff "${SPARSITY_COEFF}"
  --pairwise
  --standardize
  --dysts_ic_noise_scale 0.2
  --dysts_native_cache
  --dysts_cache_profile "${DYSTS_CACHE_PROFILE}"
  --dysts_cache_dir "${DYSTS_CACHE_DIR}"
  --dysts_cache_num_workers "${DYSTS_CACHE_NUM_WORKERS}"
  --seed "${SEED}"
  --device cuda
  --log_dir "${LOG_DIR}"
)

if [[ "${DYSTS_CACHE_REUSE}" == "1" ]]; then
  TRAIN_ARGS+=(--dysts_cache_reuse)
fi

uv run python tools/train.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
