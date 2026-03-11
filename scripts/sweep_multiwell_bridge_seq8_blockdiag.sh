#!/bin/bash
#
# Focused seq8 block-diagonal LISTA sweep for the 2D multi-well bridge systems.
# Default grid:
#   - systems: multiwell_gradient, multiwell_rotational
#   - target_size in {64, 128, 256}
#   - lista_num_loops in {1, 3, 5}
#   - seeds in {0, 1, 2}
#   - seq8 training, 20k steps
#   - parity-style coefficient family (reconst-heavy, pred=0)
#
# Submit:
#   sbatch scripts/sweep_multiwell_bridge_seq8_blockdiag.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/multiwell_bridge_blockdiag_seq8_20260305
#   NUM_STEPS=20000
#   TARGET_SIZES_CSV=64,128,256
#   LISTA_NUM_LOOPS_CSV=1,3,5
#   SEEDS_CSV=0,1,2
#   SYSTEMS_CSV=multiwell_gradient,multiwell_rotational
#   SPARSITY_COEFF=0.01
#   LISTA_ALPHA=0.10
#   K_BLOCK_SIZE=16
#
# If you override the grids, update --array to cover the expanded job count.
#
#SBATCH --job-name=mwb_s8_bd
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/mwb-s8-bd-%A_%a.out
#SBATCH -e /network/scratch/l/lia/skae/mwb-s8-bd-%A_%a.err
#SBATCH --requeue
#SBATCH --array=0-53

set -euo pipefail

WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORK_DIR}"

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/multiwell_bridge_blockdiag_seq8_20260305}"
mkdir -p "${BASE_OUT}"

NUM_STEPS="${NUM_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZES_CSV="${TARGET_SIZES_CSV:-64,128,256}"
LISTA_NUM_LOOPS_CSV="${LISTA_NUM_LOOPS_CSV:-1,3,5}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
SYSTEMS_CSV="${SYSTEMS_CSV:-multiwell_gradient,multiwell_rotational}"

RES_COEFF="${RES_COEFF:-1.0}"
RECONST_COEFF="${RECONST_COEFF:-0.5}"
PRED_COEFF="${PRED_COEFF:-0.0}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.01}"
LISTA_ALPHA="${LISTA_ALPHA:-0.10}"
LISTA_FINAL_OP="${LISTA_FINAL_OP:-relu}"
K_BLOCK_SIZE="${K_BLOCK_SIZE:-16}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

IFS=',' read -r -a TARGET_SIZES <<< "${TARGET_SIZES_CSV}"
IFS=',' read -r -a LISTA_NUM_LOOPS_LIST <<< "${LISTA_NUM_LOOPS_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a SYSTEMS <<< "${SYSTEMS_CSV}"

NUM_TARGETS=${#TARGET_SIZES[@]}
NUM_LOOPS=${#LISTA_NUM_LOOPS_LIST[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_SYSTEMS=${#SYSTEMS[@]}
TOTAL_JOBS=$((NUM_TARGETS * NUM_LOOPS * NUM_SEEDS * NUM_SYSTEMS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_JOBS )); then
  echo "Task ${TASK_ID} out of range for TOTAL_JOBS=${TOTAL_JOBS}. Exiting."
  exit 0
fi

INDEX=${TASK_ID}
SEED_IDX=$((INDEX % NUM_SEEDS))
INDEX=$((INDEX / NUM_SEEDS))
SYSTEM_IDX=$((INDEX % NUM_SYSTEMS))
INDEX=$((INDEX / NUM_SYSTEMS))
LOOP_IDX=$((INDEX % NUM_LOOPS))
INDEX=$((INDEX / NUM_LOOPS))
TARGET_IDX=$((INDEX % NUM_TARGETS))

SEED=${SEEDS[$SEED_IDX]}
SYSTEM=${SYSTEMS[$SYSTEM_IDX]}
LISTA_NUM_LOOPS=${LISTA_NUM_LOOPS_LIST[$LOOP_IDX]}
TARGET_SIZE=${TARGET_SIZES[$TARGET_IDX]}

LOG_DIR="${BASE_OUT}/ts_${TARGET_SIZE}/loops_${LISTA_NUM_LOOPS}/${SYSTEM}/seed_${SEED}"
mkdir -p "${LOG_DIR}"

echo "============================================="
echo "MultiWell Bridge Seq8 BlockDiag Sweep"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "TOTAL_JOBS: ${TOTAL_JOBS}"
echo "Host: $(hostname)"
echo "System: ${SYSTEM}"
echo "Seed: ${SEED}"
echo "TARGET_SIZE: ${TARGET_SIZE}"
echo "LISTA_NUM_LOOPS: ${LISTA_NUM_LOOPS}"
echo "NUM_STEPS: ${NUM_STEPS}"
echo "SPARSITY_COEFF: ${SPARSITY_COEFF}"
echo "LISTA_ALPHA: ${LISTA_ALPHA}"
echo "RECONST_COEFF: ${RECONST_COEFF}"
echo "PRED_COEFF: ${PRED_COEFF}"
echo "LOG_DIR: ${LOG_DIR}"
echo "Git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Start Time: $(date)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
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
  --lista_final_op "${LISTA_FINAL_OP}"
  --k_structure block_diagonal
  --k_block_size "${K_BLOCK_SIZE}"
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
