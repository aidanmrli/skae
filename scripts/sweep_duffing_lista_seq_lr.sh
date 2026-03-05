#!/bin/bash
#
# Duffing unstructured LISTA: L=8 vs L=10 learning rate sweep.
# Grid: 2 sequence lengths x 3 LRs x 3 seeds = 18 runs.
#
# Submit:
#   sbatch scripts/sweep_duffing_lista_seq_lr.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_seq_lr_20260303
#   NUM_STEPS=20000 BATCH_SIZE=256 TARGET_SIZE=256
#   RES_COEFF=1.0 RECONST_COEFF=0.03 PRED_COEFF=1.0 SPARSITY_COEFF=0.0025
#   LISTA_ALPHA=0.1 LISTA_NUM_LOOPS=5 K_STRUCTURE=dense
#   EVAL_PROFILE=full
#
#SBATCH --job-name=duf_lista_slr
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/duf-lista-slr-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-17

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_seq_lr_20260303}"
mkdir -p "${BASE_OUT}"

NUM_STEPS="${NUM_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
RES_COEFF="${RES_COEFF:-1.0}"
RECONST_COEFF="${RECONST_COEFF:-0.03}"
PRED_COEFF="${PRED_COEFF:-1.0}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.0025}"
LISTA_ALPHA="${LISTA_ALPHA:-0.1}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS:-5}"
K_STRUCTURE="${K_STRUCTURE:-dense}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

SYSTEM="duffing"
SEQUENCE_LENGTHS=(8 10)
LRS=(3e-5 1e-4 3e-4)
SEEDS=(0 1 2)

NUM_L=${#SEQUENCE_LENGTHS[@]}
NUM_LRS=${#LRS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_L * NUM_LRS * NUM_SEEDS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_JOBS )); then
  echo "Task ${TASK_ID} out of range for TOTAL_JOBS=${TOTAL_JOBS}. Exiting."
  exit 0
fi

L_IDX=$((TASK_ID / (NUM_LRS * NUM_SEEDS)))
LR_IDX=$(( (TASK_ID / NUM_SEEDS) % NUM_LRS ))
SEED_IDX=$((TASK_ID % NUM_SEEDS))

SEQUENCE_LENGTH=${SEQUENCE_LENGTHS[$L_IDX]}
LR=${LRS[$LR_IDX]}
SEED=${SEEDS[$SEED_IDX]}
LOG_DIR="${BASE_OUT}/L${SEQUENCE_LENGTH}/lr${LR}/${SYSTEM}/seed_${SEED}"

echo "============================================="
echo "Duffing LISTA Seq LR Sweep"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "System: ${SYSTEM}"
echo "Sequence length: ${SEQUENCE_LENGTH}"
echo "Learning rate: ${LR}"
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
  --lr "${LR}"
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
