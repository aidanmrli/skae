#!/bin/bash
#
# Duffing LISTA post-change baseline (Queue 0):
# - Same protocol as Experiment K LISTA-matched arm
# - ReLU enforced as LISTA final op
# - L=8, 50k steps, 3 seeds
#
# Submit:
#   sbatch scripts/sweep_duffing_lista_relu_baseline_50k.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_relu_baseline_50k_20260304
#   NUM_STEPS=50000 BATCH_SIZE=256 TARGET_SIZE=256
#   LISTA_ALPHA=0.10 LISTA_NUM_LOOPS=5
#
#SBATCH --job-name=duf_ls0_50k
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH -o /network/scratch/l/lia/skae/duf-ls0-50k-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-2

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_relu_baseline_50k_20260304}"
mkdir -p "${BASE_OUT}"

NUM_STEPS="${NUM_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
LISTA_ALPHA="${LISTA_ALPHA:-0.10}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS:-5}"

SYSTEM="duffing"
ARM="lista_relu_baseline"
SEEDS=(0 1 2)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= ${#SEEDS[@]} )); then
  echo "Task ${TASK_ID} out of range for ${#SEEDS[@]} seeds. Exiting."
  exit 0
fi

SEED=${SEEDS[$TASK_ID]}
LOG_DIR="${BASE_OUT}/${ARM}/${SYSTEM}/seed_${SEED}"

echo "============================================="
echo "Duffing LISTA Queue-0 Baseline"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "System: ${SYSTEM}"
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
  --lista_alpha "${LISTA_ALPHA}"
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
