#!/bin/bash
#
# Duffing LISTA Queue-3 depth/capacity sweep:
# - Fixed LISTA alpha (0.15 from Queue-1 winner)
# - Fixed sparsity coeff (0.0005 from Queue-2 winner)
# - Sweep LISTA num_loops at L=8, 50k steps, 3 seeds
#
# Submit:
#   sbatch scripts/sweep_duffing_lista_loops_q3.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_q03_loops_50k_20260304
#   LISTA_NUM_LOOPS_CSV=1,2,3,5,7
#   LISTA_ALPHA=0.15 SPARSITY_COEFF=0.0005
#   NUM_STEPS=50000 BATCH_SIZE=256 TARGET_SIZE=256
#
#SBATCH --job-name=duf_ls3_lp
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH -o /network/scratch/l/lia/skae/duf-ls3-loop-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-14

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_q03_loops_50k_20260304}"
mkdir -p "${BASE_OUT}"

NUM_STEPS="${NUM_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
LISTA_ALPHA="${LISTA_ALPHA:-0.15}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.0005}"
LISTA_NUM_LOOPS_CSV="${LISTA_NUM_LOOPS_CSV:-1,2,3,5,7}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"

IFS=',' read -r -a LOOPS <<< "${LISTA_NUM_LOOPS_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

SYSTEM="duffing"
NUM_LOOPS=${#LOOPS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_LOOPS * NUM_SEEDS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_JOBS )); then
  echo "Task ${TASK_ID} out of range for TOTAL_JOBS=${TOTAL_JOBS}. Exiting."
  exit 0
fi

LOOP_IDX=$((TASK_ID / NUM_SEEDS))
SEED_IDX=$((TASK_ID % NUM_SEEDS))

NUM_LOOP=${LOOPS[$LOOP_IDX]}
SEED=${SEEDS[$SEED_IDX]}
ARM="loops_${NUM_LOOP}"
LOG_DIR="${BASE_OUT}/${ARM}/${SYSTEM}/seed_${SEED}"

echo "============================================="
echo "Duffing LISTA Queue-3 Depth/Capacity Sweep"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "System: ${SYSTEM}"
echo "LISTA num_loops: ${NUM_LOOP}"
echo "Alpha: ${LISTA_ALPHA}"
echo "Sparsity coeff: ${SPARSITY_COEFF}"
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
  --sparsity_coeff "${SPARSITY_COEFF}"
  --k_structure dense
  --lista_alpha "${LISTA_ALPHA}"
  --lista_num_loops "${NUM_LOOP}"
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
