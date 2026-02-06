#!/bin/bash

#SBATCH --job-name=lqr_dec_s2_diag_eval
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -o /network/scratch/l/lia/skae/lqr_dec_s2_diag_eval-%A_%a.out
#SBATCH --array=0-71
#SBATCH --requeue

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

TARGET_SIZES=(128 256 512)
B_PROXIES=(8 13 20)
SEEDS=(0 1 2 3 4 5 6 7)

idx=$SLURM_ARRAY_TASK_ID
seed_idx=$(( idx % ${#SEEDS[@]} ))
idx=$(( idx / ${#SEEDS[@]} ))
b_idx=$(( idx % ${#B_PROXIES[@]} ))
idx=$(( idx / ${#B_PROXIES[@]} ))
ts_idx=$(( idx % ${#TARGET_SIZES[@]} ))

ARM="diag_c1"
SYSTEM="lyapunov"
TARGET_SIZE="${TARGET_SIZES[$ts_idx]}"
B_PROXY="${B_PROXIES[$b_idx]}"
SEED="${SEEDS[$seed_idx]}"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lqr_decision}"
DEVICE="${DEVICE:-cuda}"

LOG_DIR="${BASE_OUT}/stage2_${SYSTEM}_${ARM}_bp${B_PROXY}_ts${TARGET_SIZE}_seed${SEED}"
CKPT=$(ls -t "${LOG_DIR}"/*/checkpoint.pt 2>/dev/null | head -1 || true)
if [ -z "${CKPT}" ]; then
  echo "No checkpoint found in ${LOG_DIR}"
  exit 1
fi
RUN_DIR=$(dirname "${CKPT}")

echo "Stage 2 diag eval task ${SLURM_ARRAY_TASK_ID}: ts=${TARGET_SIZE}, b_proxy=${B_PROXY}, seed=${SEED}"
echo "Using checkpoint: ${CKPT}"

uv run python tools/evaluate_lqr_readiness.py \
  --checkpoint "${CKPT}" \
  --system "${SYSTEM}" \
  --b_proxy "${B_PROXY}" \
  --num_trajectories 128 \
  --trajectory_length 300 \
  --horizon_h 20 \
  --lqr_horizon 40 \
  --max_state_dim 32 \
  --control_dim 8 \
  --device "${DEVICE}" \
  --stage 2 \
  --arm "${ARM}" \
  --run_seed "${SEED}" \
  --target_size "${TARGET_SIZE}" \
  --output_dir "${RUN_DIR}/lqr_readiness"

echo "Completed LQR eval for stage2_${SYSTEM}_${ARM}_bp${B_PROXY}_ts${TARGET_SIZE}_seed${SEED}"
