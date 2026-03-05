#!/bin/bash
#
# Duffing LISTA Queue-1 full sweep (survivor alphas only):
# - L=8, 50k steps, 3 seeds
# - ReLU enforced as LISTA final op
#
# Submit:
#   sbatch --array=0-8 scripts/sweep_duffing_lista_alpha_full.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_alpha_full_50k_20260304
#   ALPHAS_CSV=0.10,0.20,0.30
#   NUM_STEPS=50000 BATCH_SIZE=256 TARGET_SIZE=256
#
#SBATCH --job-name=duf_ls1_full
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH -o /network/scratch/l/lia/skae/duf-ls1-full-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-8

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_alpha_full_50k_20260304}"
mkdir -p "${BASE_OUT}"

NUM_STEPS="${NUM_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
ALPHAS_CSV="${ALPHAS_CSV:-0.10,0.20,0.30}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS:-5}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"

IFS=',' read -r -a ALPHAS <<< "${ALPHAS_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

SYSTEM="duffing"
NUM_ALPHAS=${#ALPHAS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_ALPHAS * NUM_SEEDS))

if (( NUM_ALPHAS == 0 )); then
  echo "No alphas provided in ALPHAS_CSV='${ALPHAS_CSV}'"
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_JOBS )); then
  echo "Task ${TASK_ID} out of range for TOTAL_JOBS=${TOTAL_JOBS}. Exiting."
  exit 0
fi

ALPHA_IDX=$((TASK_ID / NUM_SEEDS))
SEED_IDX=$((TASK_ID % NUM_SEEDS))

ALPHA=${ALPHAS[$ALPHA_IDX]}
SEED=${SEEDS[$SEED_IDX]}
ALPHA_TAG="${ALPHA//./p}"
ARM="alpha_${ALPHA_TAG}"
LOG_DIR="${BASE_OUT}/${ARM}/${SYSTEM}/seed_${SEED}"

echo "============================================="
echo "Duffing LISTA Queue-1 Full"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "System: ${SYSTEM}"
echo "Alpha: ${ALPHA}"
echo "Arm: ${ARM}"
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
  --sequence_length "${SEQUENCE_LENGTH}"
  --eval_profile "${EVAL_PROFILE}"
  --seed "${SEED}"
  --device cuda
  --log_dir "${LOG_DIR}"
  --res_coeff 1.0
  --reconst_coeff 0.5
  --pred_coeff 1.0
  --sparsity_coeff 0.01
  --k_structure dense
  --lista_alpha "${ALPHA}"
  --lista_num_loops "${LISTA_NUM_LOOPS}"
  --lista_final_op relu
)

echo "TRAIN_ARGS: ${TRAIN_ARGS[*]}"
echo "============================================="

uv run python tools/train.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
