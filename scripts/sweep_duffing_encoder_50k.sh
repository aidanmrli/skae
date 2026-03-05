#!/bin/bash
#
# Duffing 50k-step encoder comparison: LISTA-current vs LISTA-matched vs generic_sparse.
# Grid: 3 arms x 3 seeds = 9 runs, all L=8.
#
# Arms:
#   0 = lista_current:  LISTA with current tuned coefficients (reconst=0.03, sparsity=0.0025, k=dense)
#   1 = lista_matched:  LISTA with generic_sparse-matched coefficients (reconst=0.5, sparsity=0.01, k=dense)
#   2 = generic_sparse: MLP sparse encoder baseline (reconst=0.5, sparsity=0.01, k=dense)
#
# Submit:
#   sbatch scripts/sweep_duffing_encoder_50k.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_encoder_50k_20260303
#   NUM_STEPS=50000 BATCH_SIZE=256 TARGET_SIZE=256
#
#SBATCH --job-name=duf_enc_50k
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH -o /network/scratch/l/lia/skae/duf-enc-50k-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-8

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_encoder_50k_20260303}"
mkdir -p "${BASE_OUT}"

NUM_STEPS="${NUM_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

SYSTEM="duffing"
ARMS=(lista_current lista_matched generic_sparse)
SEEDS=(0 1 2)

NUM_ARMS=${#ARMS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_ARMS * NUM_SEEDS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_JOBS )); then
  echo "Task ${TASK_ID} out of range for TOTAL_JOBS=${TOTAL_JOBS}. Exiting."
  exit 0
fi

ARM_IDX=$((TASK_ID / NUM_SEEDS))
SEED_IDX=$((TASK_ID % NUM_SEEDS))

ARM=${ARMS[$ARM_IDX]}
SEED=${SEEDS[$SEED_IDX]}
LOG_DIR="${BASE_OUT}/${ARM}/${SYSTEM}/seed_${SEED}"

echo "============================================="
echo "Duffing Encoder 50k Comparison"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "Arm: ${ARM}"
echo "System: ${SYSTEM}"
echo "Seed: ${SEED}"
echo "Start Time: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "LOG_DIR: ${LOG_DIR}"
echo "============================================="

# Build arm-specific training arguments
TRAIN_ARGS=(
  --env "${SYSTEM}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --target_size "${TARGET_SIZE}"
  --sequence_length "${SEQUENCE_LENGTH}"
  --eval_profile "${EVAL_PROFILE}"
  --seed "${SEED}"
  --device cuda
  --log_dir "${LOG_DIR}"
)

case "${ARM}" in
  lista_current)
    TRAIN_ARGS+=(
      --config lista_parity_generic_sparse
      --reconst_coeff 0.03
      --pred_coeff 1.0
      --sparsity_coeff 0.0025
      --res_coeff 1.0
      --k_structure dense
      --lista_alpha 0.1
      --lista_num_loops 5
    )
    ;;
  lista_matched)
    TRAIN_ARGS+=(
      --config lista_parity_generic_sparse
      --reconst_coeff 0.5
      --pred_coeff 1.0
      --sparsity_coeff 0.01
      --res_coeff 1.0
      --k_structure dense
      --lista_alpha 0.1
      --lista_num_loops 5
    )
    ;;
  generic_sparse)
    TRAIN_ARGS+=(
      --config generic_sparse
      --pred_coeff 1.0
    )
    ;;
esac

echo "TRAIN_ARGS: ${TRAIN_ARGS[*]}"
echo "============================================="

uv run python tools/train.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
