#!/bin/bash
#
# Sequence-L8 high-dimensional benchmark sweep on intrinsic-HD systems from
# docs/planning/high_dim_benchmarks_plan.md.
#
# Systems: kuramoto, hopfield, competitive_lv
# Seeds:   0,1,2
# Tasks:   3 systems x 3 seeds = 9
#
# Submit examples:
#   sbatch --export=ALL,MODEL_VARIANT=generic_sparse scripts/sweep_high_dim_benchmarks_seq8.sh
#   sbatch --export=ALL,MODEL_VARIANT=lista_dense scripts/sweep_high_dim_benchmarks_seq8.sh
#   sbatch --export=ALL,MODEL_VARIANT=lista_blockdiag scripts/sweep_high_dim_benchmarks_seq8.sh
#
#SBATCH --job-name=highdim_seq8
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/highdim-seq8-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-8

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/high_dim_benchmarks_seq8}"
MODEL_VARIANT="${MODEL_VARIANT:-generic_sparse}"

NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

RES_COEFF="${RES_COEFF:-1.0}"
RECONST_COEFF="${RECONST_COEFF:-0.03}"
PRED_COEFF="${PRED_COEFF:-1.0}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.0025}"

LISTA_ALPHA="${LISTA_ALPHA:-0.1}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS:-5}"
LISTA_FINAL_OP="${LISTA_FINAL_OP:-relu}"
K_BLOCK_SIZE="${K_BLOCK_SIZE:-32}"

SYSTEMS=(
  kuramoto
  hopfield
  competitive_lv
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
LOG_DIR="${BASE_OUT}/${MODEL_VARIANT}/${SYSTEM}/seed_${SEED}"
mkdir -p "${LOG_DIR}"

case "${MODEL_VARIANT}" in
  generic_sparse)
    CONFIG="generic_sparse"
    ;;
  lista_dense)
    CONFIG="lista_parity_generic_sparse"
    ;;
  lista_blockdiag)
    CONFIG="lista_parity_generic_sparse"
    ;;
  *)
    echo "Unknown MODEL_VARIANT='${MODEL_VARIANT}'. Expected one of: generic_sparse, lista_dense, lista_blockdiag"
    exit 2
    ;;
esac

echo "============================================="
echo "High-Dim Benchmarks Seq8"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "Model Variant: ${MODEL_VARIANT}"
echo "Config: ${CONFIG}"
echo "System: ${SYSTEM}"
echo "Seed: ${SEED}"
echo "TARGET_SIZE: ${TARGET_SIZE}"
echo "Start Time: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "LOG_DIR: ${LOG_DIR}"
echo "============================================="

TRAIN_ARGS=(
  --config "${CONFIG}"
  --env "${SYSTEM}"
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

if [[ "${MODEL_VARIANT}" == "lista_dense" ]]; then
  TRAIN_ARGS+=(
    --k_structure dense
    --lista_alpha "${LISTA_ALPHA}"
    --lista_num_loops "${LISTA_NUM_LOOPS}"
    --lista_final_op "${LISTA_FINAL_OP}"
  )
fi

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
