#!/bin/bash

#SBATCH --job-name=lqr_dec_excl_recov
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -o /network/scratch/l/lia/skae/lqr_dec_excl_recov-%A_%a.out
#SBATCH --array=0-14
#SBATCH --requeue

# ============================================================================
# Recovery for 15 failed runs from the exclusivity ablation sweep (job 8622549)
# Training + eigenvalue analysis completed; need support_eval + cosine_diag + lqr_readiness
# ============================================================================

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

# Enumerate the 15 failed (arm, ts, bp, seed) tuples explicitly
FAILED_ARMS=(ah_prag ah_prag ah_prag ah_prag ah_prag ah_prag ah_prag ah_prag_no_excl ah_prag_no_excl ah_prag_no_excl ah_prag_no_excl ah_prag_no_excl ah_prag_no_excl ah_prag_no_excl ah_prag_no_excl)
FAILED_TS=(128 128 128 128 128 128 256 128 128 128 128 128 128 128 128)
FAILED_BP=(8 8 8 8 8 8 8 8 8 8 8 8 8 8 8)
FAILED_SEEDS=(0 1 2 4 5 6 5 0 1 2 3 4 5 6 7)

idx=$SLURM_ARRAY_TASK_ID
ARM="${FAILED_ARMS[$idx]}"
TARGET_SIZE="${FAILED_TS[$idx]}"
B_PROXY="${FAILED_BP[$idx]}"
SEED="${FAILED_SEEDS[$idx]}"
SYSTEM="lyapunov"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lqr_decision_excl_ablation}"
DEVICE="${DEVICE:-cuda}"

LOG_DIR="${BASE_OUT}/stage2_${SYSTEM}_${ARM}_bp${B_PROXY}_ts${TARGET_SIZE}_seed${SEED}"
CKPT=$(ls -t "${LOG_DIR}"/*/checkpoint.pt 2>/dev/null | head -1 || true)
if [ -z "${CKPT}" ]; then
  echo "No checkpoint found in ${LOG_DIR}"
  exit 1
fi
RUN_DIR=$(dirname "${CKPT}")

echo "Recovery task ${SLURM_ARRAY_TASK_ID}: arm=${ARM}, ts=${TARGET_SIZE}, bp=${B_PROXY}, seed=${SEED}"
echo "Using checkpoint: ${CKPT}"

echo "[1/3] support + cosine diagnostics"
uv run python tools/evaluate_support_uniqueness.py \
  --checkpoint "${CKPT}" \
  --system "${SYSTEM}" \
  --support_mode mean \
  --cosine_diag \
  --threshold_sweep \
  --output_dir "${RUN_DIR}/support_eval" \
  --device "${DEVICE}"

echo "[2/3] cosine sensitivity diagnostics"
uv run python tools/diagnose_cosine_separation.py \
  --checkpoint "${CKPT}" \
  --system "${SYSTEM}" \
  --output_dir "${RUN_DIR}/cosine_diag" \
  --device "${DEVICE}"

echo "[3/3] LQR readiness evaluation"
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

echo "Completed recovery for stage2_${SYSTEM}_${ARM}_bp${B_PROXY}_ts${TARGET_SIZE}_seed${SEED}"
