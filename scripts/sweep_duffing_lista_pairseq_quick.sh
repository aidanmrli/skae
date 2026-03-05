#!/bin/bash
#
# Duffing unstructured LISTA parity quick gate: L=1 vs L=8.
# Grid: 2 sequence lengths x 3 seeds = 6 runs.
#
# Submit:
#   sbatch scripts/sweep_duffing_lista_pairseq_quick.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_pairseq_quick_20260303
#   NUM_STEPS=3000 BATCH_SIZE=256 TARGET_SIZE=256
#   RES_COEFF=1.0 RECONST_COEFF=0.03 PRED_COEFF=1.0 SPARSITY_COEFF=0.0025
#   LISTA_ALPHA=0.1 LISTA_NUM_LOOPS=5 K_STRUCTURE=dense
#   EVAL_PROFILE=smoke
#
# NOTE:
#   This uses unstructured LISTA with parity-style settings and only varies sequence length.
#
#SBATCH --job-name=duf_lista_qk
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH -o /network/scratch/l/lia/skae/duf-lista-qk-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-5

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_pairseq_quick_20260303}"
mkdir -p "${BASE_OUT}"

NUM_STEPS="${NUM_STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
RES_COEFF="${RES_COEFF:-1.0}"
RECONST_COEFF="${RECONST_COEFF:-0.03}"
PRED_COEFF="${PRED_COEFF:-1.0}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.0025}"
LISTA_ALPHA="${LISTA_ALPHA:-0.1}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS:-5}"
K_STRUCTURE="${K_STRUCTURE:-dense}"
EVAL_PROFILE="${EVAL_PROFILE:-smoke}"

SYSTEM="duffing"
SEQUENCE_LENGTHS=(1 8)
SEEDS=(0 1 2)

NUM_L=${#SEQUENCE_LENGTHS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_L * NUM_SEEDS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_JOBS )); then
  echo "Task ${TASK_ID} out of range for TOTAL_JOBS=${TOTAL_JOBS}. Exiting."
  exit 0
fi

L_IDX=$((TASK_ID / NUM_SEEDS))
SEED_IDX=$((TASK_ID % NUM_SEEDS))

SEQUENCE_LENGTH=${SEQUENCE_LENGTHS[$L_IDX]}
SEED=${SEEDS[$SEED_IDX]}
LOG_DIR="${BASE_OUT}/L${SEQUENCE_LENGTH}/${SYSTEM}/seed_${SEED}"

echo "============================================="
echo "Duffing LISTA PairSeq Quick"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "System: ${SYSTEM}"
echo "Sequence length: ${SEQUENCE_LENGTH}"
echo "Seed: ${SEED}"
echo "Start Time: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "LOG_DIR: ${LOG_DIR}"
echo "============================================="

TRAIN_ARGS=(
  --config lista_parity_generic_sparse
  --env "${SYSTEM}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --target_size "${TARGET_SIZE}"
  --res_coeff "${RES_COEFF}"
  --reconst_coeff "${RECONST_COEFF}"
  --pred_coeff "${PRED_COEFF}"
  --sparsity_coeff "${SPARSITY_COEFF}"
  --lista_alpha "${LISTA_ALPHA}"
  --lista_num_loops "${LISTA_NUM_LOOPS}"
  --k_structure "${K_STRUCTURE}"
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
