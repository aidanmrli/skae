#!/bin/bash

#SBATCH --job-name=lista_op_p3
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=14G
#SBATCH --time=10:00:00
#SBATCH -o /network/scratch/l/lia/skae/lista-op-p3-%A_%a.out
#SBATCH --array=0-95
#SBATCH --requeue

# ============================================================================
# LISTA Final-Op Phase 3: Structured-K Transfer Sweep
# ============================================================================
# Grid:
#   system      in {lyapunov, duffing}
#   target_size in {256, 512}
#   k_structure in {diagonal, block_diagonal, arrowhead_no_excl}
#   final_op    in {shrink, relu}
#   seed        in {0, 1, 2, 3}
# Total: 96 jobs
# ============================================================================

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

SYSTEMS=(lyapunov duffing)
TARGET_SIZES=(256 512)
K_STRUCTURES=(diagonal block_diagonal arrowhead_no_excl)
FINAL_OPS=(shrink relu)
SEEDS=(0 1 2 3)

SYSTEM_IDX=$(( SLURM_ARRAY_TASK_ID / 48 ))
REM1=$(( SLURM_ARRAY_TASK_ID % 48 ))
TARGET_SIZE_IDX=$(( REM1 / 24 ))
REM2=$(( REM1 % 24 ))
KSTRUCT_IDX=$(( REM2 / 8 ))
REM3=$(( REM2 % 8 ))
FINAL_OP_IDX=$(( REM3 / 4 ))
SEED_IDX=$(( REM3 % 4 ))

SYSTEM="${SYSTEMS[$SYSTEM_IDX]}"
TARGET_SIZE="${TARGET_SIZES[$TARGET_SIZE_IDX]}"
K_STRUCTURE="${K_STRUCTURES[$KSTRUCT_IDX]}"
FINAL_OP="${FINAL_OPS[$FINAL_OP_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SPARSITY_COEFF="${SPARSITY_COEFF:-1.0}"
NUM_BASINS_PROXY="${NUM_BASINS_PROXY:-20}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_final_op_experiment/phase3_structured_transfer}"
DEVICE="${DEVICE:-cuda}"

echo "============================================="
echo "Phase 3 Structured Transfer"
echo "Job: ${SLURM_JOB_ID}  Task: ${SLURM_ARRAY_TASK_ID}"
echo "System: ${SYSTEM}"
echo "Target size: ${TARGET_SIZE}"
echo "K structure: ${K_STRUCTURE}"
echo "Final op: ${FINAL_OP}"
echo "Seed: ${SEED}"
echo "============================================="

PHASE=3 \
SYSTEM="${SYSTEM}" \
FINAL_OP="${FINAL_OP}" \
TARGET_SIZE="${TARGET_SIZE}" \
SEED="${SEED}" \
NUM_STEPS="${NUM_STEPS}" \
BATCH_SIZE="${BATCH_SIZE}" \
SPARSITY_COEFF="${SPARSITY_COEFF}" \
K_STRUCTURE="${K_STRUCTURE}" \
NUM_BASINS_PROXY="${NUM_BASINS_PROXY}" \
BASE_OUT="${BASE_OUT}" \
DEVICE="${DEVICE}" \
bash scripts/run_lista_final_op_trial.sh
