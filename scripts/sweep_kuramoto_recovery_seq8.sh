#!/bin/bash
#
# Dedicated Kuramoto recovery sweep for the intrinsic-HD Seq8 baseline.
#
# Default mode:
#   - block-diagonal LISTA recovery pilot
#   - current kuramoto defaults
#   - 20k steps
#   - sp in {0.0005, 0.0010, 0.0025}
#   - alpha = 0.15
#   - loops in {1, 3, 5}
#   - k_block_size = 16
#   - seeds = 0,1,2
#
# Example submissions:
#   sbatch scripts/sweep_kuramoto_recovery_seq8.sh
#   sbatch --array=0-2 --export=ALL,MODEL_VARIANT=generic_sparse,SPARSITY_COEFFS_CSV=0.0025,NUM_STEPS_CSV=20000 scripts/sweep_kuramoto_recovery_seq8.sh
#
# If you override the grids, update --array to cover the expanded job count.
#
#SBATCH --job-name=kuramoto_rec
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/kuramoto-rec-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-26

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/kuramoto_recovery_seq8_20260305}"
MODEL_VARIANT="${MODEL_VARIANT:-lista_blockdiag}"

NUM_STEPS_CSV="${NUM_STEPS_CSV:-20000}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
SPARSITY_COEFFS_CSV="${SPARSITY_COEFFS_CSV:-0.0005,0.0010,0.0025}"
LISTA_ALPHAS_CSV="${LISTA_ALPHAS_CSV:-0.15}"
LISTA_NUM_LOOPS_CSV="${LISTA_NUM_LOOPS_CSV:-1,3,5}"
K_BLOCK_SIZES_CSV="${K_BLOCK_SIZES_CSV:-16}"

BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

RES_COEFF="${RES_COEFF:-1.0}"
RECONST_COEFF="${RECONST_COEFF:-0.03}"
PRED_COEFF="${PRED_COEFF:-1.0}"
LISTA_FINAL_OP="${LISTA_FINAL_OP:-relu}"

IFS=',' read -r -a NUM_STEPS_LIST <<< "${NUM_STEPS_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a SPARSITY_COEFFS <<< "${SPARSITY_COEFFS_CSV}"
IFS=',' read -r -a LISTA_ALPHAS <<< "${LISTA_ALPHAS_CSV}"
IFS=',' read -r -a LISTA_NUM_LOOPS_LIST <<< "${LISTA_NUM_LOOPS_CSV}"
IFS=',' read -r -a K_BLOCK_SIZES <<< "${K_BLOCK_SIZES_CSV}"

tagify() {
  local raw="$1"
  raw="${raw//-/m}"
  raw="${raw//./p}"
  echo "${raw}"
}

NUM_STEP_OPTIONS=${#NUM_STEPS_LIST[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_SPARSITY=${#SPARSITY_COEFFS[@]}
NUM_ALPHAS=${#LISTA_ALPHAS[@]}
NUM_LOOPS=${#LISTA_NUM_LOOPS_LIST[@]}
NUM_BLOCKS=${#K_BLOCK_SIZES[@]}

case "${MODEL_VARIANT}" in
  generic_sparse)
    CONFIG="generic_sparse"
    TOTAL_JOBS=$((NUM_STEP_OPTIONS * NUM_SPARSITY * NUM_SEEDS))
    ;;
  lista_blockdiag)
    CONFIG="lista_parity_generic_sparse"
    TOTAL_JOBS=$((NUM_STEP_OPTIONS * NUM_SPARSITY * NUM_ALPHAS * NUM_LOOPS * NUM_BLOCKS * NUM_SEEDS))
    ;;
  *)
    echo "Unknown MODEL_VARIANT='${MODEL_VARIANT}'. Expected one of: generic_sparse, lista_blockdiag"
    exit 2
    ;;
esac

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_JOBS )); then
  echo "Task ${TASK_ID} out of range for TOTAL_JOBS=${TOTAL_JOBS}. Exiting."
  exit 0
fi

INDEX=${TASK_ID}
SEED_IDX=$((INDEX % NUM_SEEDS))
INDEX=$((INDEX / NUM_SEEDS))
SPARSITY_IDX=$((INDEX % NUM_SPARSITY))
INDEX=$((INDEX / NUM_SPARSITY))
STEP_IDX=$((INDEX % NUM_STEP_OPTIONS))
INDEX=$((INDEX / NUM_STEP_OPTIONS))

SEED=${SEEDS[$SEED_IDX]}
SPARSITY_COEFF=${SPARSITY_COEFFS[$SPARSITY_IDX]}
NUM_STEPS=${NUM_STEPS_LIST[$STEP_IDX]}

LISTA_ALPHA="${LISTA_ALPHAS[0]}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS_LIST[0]}"
K_BLOCK_SIZE="${K_BLOCK_SIZES[0]}"

if [[ "${MODEL_VARIANT}" == "lista_blockdiag" ]]; then
  ALPHA_IDX=$((INDEX % NUM_ALPHAS))
  INDEX=$((INDEX / NUM_ALPHAS))
  LOOP_IDX=$((INDEX % NUM_LOOPS))
  INDEX=$((INDEX / NUM_LOOPS))
  BLOCK_IDX=$((INDEX % NUM_BLOCKS))

  LISTA_ALPHA=${LISTA_ALPHAS[$ALPHA_IDX]}
  LISTA_NUM_LOOPS=${LISTA_NUM_LOOPS_LIST[$LOOP_IDX]}
  K_BLOCK_SIZE=${K_BLOCK_SIZES[$BLOCK_IDX]}
fi

SP_TAG=$(tagify "${SPARSITY_COEFF}")
STEP_TAG=$(tagify "${NUM_STEPS}")
ALPHA_TAG=$(tagify "${LISTA_ALPHA}")

LOG_DIR="${BASE_OUT}/${MODEL_VARIANT}/kuramoto/steps_${STEP_TAG}/sp_${SP_TAG}"
if [[ "${MODEL_VARIANT}" == "lista_blockdiag" ]]; then
  LOG_DIR="${LOG_DIR}/alpha_${ALPHA_TAG}/loops_${LISTA_NUM_LOOPS}/block_${K_BLOCK_SIZE}"
fi
LOG_DIR="${LOG_DIR}/seed_${SEED}"
mkdir -p "${LOG_DIR}"

echo "============================================="
echo "Kuramoto Recovery Seq8"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "TOTAL_JOBS: ${TOTAL_JOBS}"
echo "Model Variant: ${MODEL_VARIANT}"
echo "Config: ${CONFIG}"
echo "System: kuramoto"
echo "Seed: ${SEED}"
echo "NUM_STEPS: ${NUM_STEPS}"
echo "SPARSITY_COEFF: ${SPARSITY_COEFF}"
if [[ "${MODEL_VARIANT}" == "lista_blockdiag" ]]; then
  echo "LISTA_ALPHA: ${LISTA_ALPHA}"
  echo "LISTA_NUM_LOOPS: ${LISTA_NUM_LOOPS}"
  echo "K_BLOCK_SIZE: ${K_BLOCK_SIZE}"
fi
echo "LOG_DIR: ${LOG_DIR}"
echo "Start Time: $(date)"
echo "============================================="

TRAIN_ARGS=(
  --config "${CONFIG}"
  --env kuramoto
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --target_size "${TARGET_SIZE}"
  --res_coeff "${RES_COEFF}"
  --reconst_coeff "${RECONST_COEFF}"
  --pred_coeff "${PRED_COEFF}"
  --sparsity_coeff "${SPARSITY_COEFF}"
  --sequence_length "${SEQUENCE_LENGTH}"
  --eval_profile "${EVAL_PROFILE}"
  --seed "${SEED}"
  --device cuda
  --log_dir "${LOG_DIR}"
)

if [[ "${MODEL_VARIANT}" == "lista_blockdiag" ]]; then
  TRAIN_ARGS+=(
    --k_structure block_diagonal
    --k_block_size "${K_BLOCK_SIZE}"
    --lista_alpha "${LISTA_ALPHA}"
    --lista_num_loops "${LISTA_NUM_LOOPS}"
    --lista_final_op "${LISTA_FINAL_OP}"
  )
fi

uv run python tools/train.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
