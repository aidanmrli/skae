#!/bin/bash

#SBATCH --job-name=lista_op_p0
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=4:00:00
#SBATCH -o /network/scratch/l/lia/skae/lista-op-p0-%A_%a.out
#SBATCH --array=0-7
#SBATCH --requeue

# ============================================================================
# LISTA Final-Op Phase 0: Smoke Test
# ============================================================================
# Grid:
#   system in {lyapunov, duffing}
#   final_op in {shrink, relu}
#   seed in {0, 1}
# Total: 8 jobs
# ============================================================================

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

SYSTEMS=(lyapunov duffing)
FINAL_OPS=(shrink relu)
SEEDS=(0 1)

SYSTEM_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
REM1=$(( SLURM_ARRAY_TASK_ID % 4 ))
FINAL_OP_IDX=$(( REM1 / 2 ))
SEED_IDX=$(( REM1 % 2 ))

SYSTEM="${SYSTEMS[$SYSTEM_IDX]}"
FINAL_OP="${FINAL_OPS[$FINAL_OP_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

TARGET_SIZE="${TARGET_SIZE:-256}"
NUM_STEPS="${NUM_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SPARSITY_COEFF="${SPARSITY_COEFF:-1.0}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_final_op_experiment/phase0_smoke}"
DEVICE="${DEVICE:-cuda}"

echo "============================================="
echo "Phase 0 Smoke"
echo "Job: ${SLURM_JOB_ID}  Task: ${SLURM_ARRAY_TASK_ID}"
echo "System: ${SYSTEM}"
echo "Final op: ${FINAL_OP}"
echo "Seed: ${SEED}"
echo "Target size: ${TARGET_SIZE}"
echo "Steps: ${NUM_STEPS}"
echo "============================================="

PHASE=0 \
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
