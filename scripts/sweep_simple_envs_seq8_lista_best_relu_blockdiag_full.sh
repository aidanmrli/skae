#!/bin/bash
#
# Full-run simple-environment benchmark: tuned LISTAKM (ReLU final op, block-diagonal K), sequence loss (L=8).
# Grid: 13 systems x 1 seed = 13 runs.
#
# Submit:
#   sbatch scripts/sweep_simple_envs_seq8_lista_best_relu_blockdiag_full.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/simple_envs_seq8_full/lista_best_relu_blockdiag
#   NUM_STEPS=10000 BATCH_SIZE=256 TARGET_SIZE=128
#   RES_COEFF=1.0 RECONST_COEFF=0.03 PRED_COEFF=1.0 SPARSITY_COEFF=0.0025
#   LISTA_ALPHA=0.1 LISTA_NUM_LOOPS=5 LISTA_FINAL_OP=relu
#   K_STRUCTURE=block_diagonal K_BLOCK_SIZE=16
#   SEQUENCE_LENGTH=8 EVAL_PROFILE=full SEED=0
#
# NOTE:
#   This script intentionally avoids observation/action normalization flags.
#
#SBATCH --job-name=full_lista_s8_best
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/full-lista-s8-best-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-12

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/simple_envs_seq8_full/lista_best_relu_blockdiag}"
mkdir -p "${BASE_OUT}"

NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-128}"
RES_COEFF="${RES_COEFF:-1.0}"
RECONST_COEFF="${RECONST_COEFF:-0.03}"
PRED_COEFF="${PRED_COEFF:-1.0}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.0025}"
LISTA_ALPHA="${LISTA_ALPHA:-0.1}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS:-5}"
LISTA_FINAL_OP="${LISTA_FINAL_OP:-relu}"
K_STRUCTURE="${K_STRUCTURE:-block_diagonal}"
K_BLOCK_SIZE="${K_BLOCK_SIZE:-16}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
SEED="${SEED:-0}"

SYSTEMS=(
  parabolic
  duffing
  lotka_volterra
  pendulum
  lorenz63
  multiwell_gradient
  multiwell_rotational
  multiwell_energy
  multiwell_strong_transition
  multiwell_gradient_hd
  multiwell_rotational_hd
  multiwell_energy_hd
  multiwell_strong_transition_hd
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_SYSTEMS=${#SYSTEMS[@]}
if (( TASK_ID >= NUM_SYSTEMS )); then
  echo "Task ${TASK_ID} out of range for NUM_SYSTEMS=${NUM_SYSTEMS}. Exiting."
  exit 0
fi

SYSTEM=${SYSTEMS[$TASK_ID]}
LOG_DIR="${BASE_OUT}/${SYSTEM}/seed_${SEED}"

echo "============================================="
echo "Simple Envs Sequence-L8 | LISTA Best ReLU BlockDiag | Full"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "System: ${SYSTEM}"
echo "Seed: ${SEED}"
echo "K_STRUCTURE: ${K_STRUCTURE}"
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
  --lista_final_op "${LISTA_FINAL_OP}"
  --k_structure "${K_STRUCTURE}"
  --sequence
  --sequence_length "${SEQUENCE_LENGTH}"
  --eval_profile "${EVAL_PROFILE}"
  --seed "${SEED}"
  --device cuda
  --log_dir "${LOG_DIR}"
)

if [[ "${K_STRUCTURE}" == "block_diagonal" ]]; then
  TRAIN_ARGS+=(--k_block_size "${K_BLOCK_SIZE}")
fi

uv run python tools/train.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}

