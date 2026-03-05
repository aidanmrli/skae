#!/bin/bash
#
# Duffing HyperLISTA Queue-5 adaptive-threshold sweep:
# - Joint sweep over sparsity coeff and HyperLISTA c_theta
# - Fixed loops (default 1), L=8, 50k steps, 3 seeds
#
# Submit:
#   sbatch scripts/sweep_duffing_hyperlista_q5.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_hyperlista_q05_adaptive_50k_20260304
#   SPARSITY_COEFFS=0.0040,0.0060
#   LISTA_NUM_LOOPS_CSV=1
#   HYPER_C_THETAS=0.0040,0.0060,0.0100,0.0200
#   HYPER_C_BETA=0.0001
#   HYPER_C_SS=0.5
#   NUM_STEPS=50000 BATCH_SIZE=256 TARGET_SIZE=256
#
#SBATCH --job-name=duf_hl5_sw
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH -o /network/scratch/l/lia/skae/duf-hl5-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-23

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_hyperlista_q05_adaptive_50k_20260304}"
mkdir -p "${BASE_OUT}"

NUM_STEPS="${NUM_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

SPARSITY_COEFFS="${SPARSITY_COEFFS:-0.0040,0.0060}"
LISTA_NUM_LOOPS_CSV="${LISTA_NUM_LOOPS_CSV:-1}"
HYPER_C_THETAS="${HYPER_C_THETAS:-0.0040,0.0060,0.0100,0.0200}"
HYPER_C_BETA="${HYPER_C_BETA:-0.0001}"
HYPER_C_SS="${HYPER_C_SS:-0.5}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"

IFS=',' read -r -a COEFFS <<< "${SPARSITY_COEFFS}"
IFS=',' read -r -a LOOPS <<< "${LISTA_NUM_LOOPS_CSV}"
IFS=',' read -r -a THETAS <<< "${HYPER_C_THETAS}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

SYSTEM="duffing"
NUM_COEFFS=${#COEFFS[@]}
NUM_LOOPS=${#LOOPS[@]}
NUM_THETAS=${#THETAS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_COEFFS * NUM_LOOPS * NUM_THETAS * NUM_SEEDS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_JOBS )); then
  echo "Task ${TASK_ID} out of range for TOTAL_JOBS=${TOTAL_JOBS}. Exiting."
  exit 0
fi

COEFF_STRIDE=$((NUM_LOOPS * NUM_THETAS * NUM_SEEDS))
COEFF_IDX=$((TASK_ID / COEFF_STRIDE))
REM1=$((TASK_ID % COEFF_STRIDE))
LOOP_STRIDE=$((NUM_THETAS * NUM_SEEDS))
LOOP_IDX=$((REM1 / LOOP_STRIDE))
REM2=$((REM1 % LOOP_STRIDE))
THETA_IDX=$((REM2 / NUM_SEEDS))
SEED_IDX=$((REM2 % NUM_SEEDS))

SPARSITY_COEFF=${COEFFS[$COEFF_IDX]}
NUM_LOOP=${LOOPS[$LOOP_IDX]}
HYPER_C_THETA=${THETAS[$THETA_IDX]}
SEED=${SEEDS[$SEED_IDX]}

SP_TAG="${SPARSITY_COEFF//./p}"
THETA_TAG="${HYPER_C_THETA//./p}"
ARM="sp_${SP_TAG}_loops_${NUM_LOOP}_ct_${THETA_TAG}"
LOG_DIR="${BASE_OUT}/${ARM}/${SYSTEM}/seed_${SEED}"

echo "============================================="
echo "Duffing HyperLISTA Queue-5 Adaptive Sweep"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "System: ${SYSTEM}"
echo "Sparsity coeff: ${SPARSITY_COEFF}"
echo "LISTA num_loops: ${NUM_LOOP}"
echo "HyperLISTA c_theta: ${HYPER_C_THETA}"
echo "HyperLISTA c_beta: ${HYPER_C_BETA}"
echo "HyperLISTA c_ss: ${HYPER_C_SS}"
echo "Arm: ${ARM}"
echo "Seed: ${SEED}"
echo "Start Time: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "LOG_DIR: ${LOG_DIR}"
echo "============================================="

TRAIN_ARGS=(
  --config hyperlista_parity_generic_sparse
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
  --lista_num_loops "${NUM_LOOP}"
  --hyperlista_c_theta "${HYPER_C_THETA}"
  --hyperlista_c_beta "${HYPER_C_BETA}"
  --hyperlista_c_ss "${HYPER_C_SS}"
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

