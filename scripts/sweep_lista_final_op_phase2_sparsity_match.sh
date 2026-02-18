#!/bin/bash

#SBATCH --job-name=lista_op_p2
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH -o /network/scratch/l/lia/skae/lista-op-p2-%A_%a.out
#SBATCH --array=0-23
#SBATCH --requeue

# ============================================================================
# LISTA Final-Op Phase 2: ReLU Sparsity-Matching Sweep
# ============================================================================
# Grid (Lyapunov only):
#   target_size    in {256, 512}
#   sparsity_coeff in {1.0, 1.5, 2.0}
#   seed           in {0, 1, 2, 3}
# final_op fixed to relu
# Total: 24 jobs
# ============================================================================

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

TARGET_SIZES=(256 512)
SPARSITY_COEFFS=(1.0 1.5 2.0)
SEEDS=(0 1 2 3)

TARGET_SIZE_IDX=$(( SLURM_ARRAY_TASK_ID / 12 ))
REM1=$(( SLURM_ARRAY_TASK_ID % 12 ))
SPARSITY_IDX=$(( REM1 / 4 ))
SEED_IDX=$(( REM1 % 4 ))

SYSTEM="lyapunov"
TARGET_SIZE="${TARGET_SIZES[$TARGET_SIZE_IDX]}"
SPARSITY_COEFF="${SPARSITY_COEFFS[$SPARSITY_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_final_op_experiment/phase2_sparsity_match}"
DEVICE="${DEVICE:-cuda}"

echo "============================================="
echo "Phase 2 ReLU Sparsity Match"
echo "Job: ${SLURM_JOB_ID}  Task: ${SLURM_ARRAY_TASK_ID}"
echo "System: ${SYSTEM}"
echo "Target size: ${TARGET_SIZE}"
echo "Sparsity coeff: ${SPARSITY_COEFF}"
echo "Seed: ${SEED}"
echo "============================================="

PHASE=2 \
SYSTEM="${SYSTEM}" \
FINAL_OP="relu" \
TARGET_SIZE="${TARGET_SIZE}" \
SEED="${SEED}" \
NUM_STEPS="${NUM_STEPS}" \
BATCH_SIZE="${BATCH_SIZE}" \
SPARSITY_COEFF="${SPARSITY_COEFF}" \
K_STRUCTURE="dense" \
BASE_OUT="${BASE_OUT}" \
DEVICE="${DEVICE}" \
bash scripts/run_lista_final_op_trial.sh
