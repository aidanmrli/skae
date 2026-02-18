#!/bin/bash

#SBATCH --job-name=lista_op_p1
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH -o /network/scratch/l/lia/skae/lista-op-p1-%A_%a.out
#SBATCH --array=0-95
#SBATCH --requeue

# ============================================================================
# LISTA Final-Op Phase 1: Core Encoder-Isolation Sweep
# ============================================================================
# Grid:
#   system      in {lyapunov, duffing}
#   target_size in {128, 256, 512}
#   final_op    in {shrink, relu}
#   seed        in {0..7}
# Total: 96 jobs
# ============================================================================

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

SYSTEMS=(lyapunov duffing)
TARGET_SIZES=(128 256 512)
FINAL_OPS=(shrink relu)
SEEDS=(0 1 2 3 4 5 6 7)

SYSTEM_IDX=$(( SLURM_ARRAY_TASK_ID / 48 ))
REM1=$(( SLURM_ARRAY_TASK_ID % 48 ))
TARGET_SIZE_IDX=$(( REM1 / 16 ))
REM2=$(( REM1 % 16 ))
FINAL_OP_IDX=$(( REM2 / 8 ))
SEED_IDX=$(( REM2 % 8 ))

SYSTEM="${SYSTEMS[$SYSTEM_IDX]}"
TARGET_SIZE="${TARGET_SIZES[$TARGET_SIZE_IDX]}"
FINAL_OP="${FINAL_OPS[$FINAL_OP_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SPARSITY_COEFF="${SPARSITY_COEFF:-1.0}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_final_op_experiment/phase1_core}"
DEVICE="${DEVICE:-cuda}"

echo "============================================="
echo "Phase 1 Core Sweep"
echo "Job: ${SLURM_JOB_ID}  Task: ${SLURM_ARRAY_TASK_ID}"
echo "System: ${SYSTEM}"
echo "Target size: ${TARGET_SIZE}"
echo "Final op: ${FINAL_OP}"
echo "Seed: ${SEED}"
echo "============================================="

PHASE=1 \
SYSTEM="${SYSTEM}" \
FINAL_OP="${FINAL_OP}" \
TARGET_SIZE="${TARGET_SIZE}" \
SEED="${SEED}" \
NUM_STEPS="${NUM_STEPS}" \
BATCH_SIZE="${BATCH_SIZE}" \
SPARSITY_COEFF="${SPARSITY_COEFF}" \
K_STRUCTURE="dense" \
BASE_OUT="${BASE_OUT}" \
DEVICE="${DEVICE}" \
bash scripts/run_lista_final_op_trial.sh
