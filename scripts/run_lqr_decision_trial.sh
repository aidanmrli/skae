#!/bin/bash
set -euo pipefail

# Run one arm/system/seed trial for the LQR decision plan.
#
# Required env vars:
#   STAGE ARM SYSTEM TARGET_SIZE B_PROXY SEED BASE_OUT DEVICE
#
# Optional env vars:
#   DIM NUM_BASINS_GT NUM_STEPS BATCH_SIZE SUPPORT_THRESHOLD
#   LAMBDA_GLOBAL_PRAG LAMBDA_LOCAL_PRAG

if [ -z "${STAGE:-}" ] || [ -z "${ARM:-}" ] || [ -z "${SYSTEM:-}" ] || [ -z "${TARGET_SIZE:-}" ] || [ -z "${B_PROXY:-}" ] || [ -z "${SEED:-}" ]; then
  echo "Missing required env vars. Need: STAGE ARM SYSTEM TARGET_SIZE B_PROXY SEED"
  exit 1
fi

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lqr_decision}"
DEVICE="${DEVICE:-cuda}"
DIM="${DIM:-8}"
NUM_BASINS_GT="${NUM_BASINS_GT:-13}"
NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-1e-3}"
LAMBDA_GLOBAL_PRAG="${LAMBDA_GLOBAL_PRAG:-1e-4}"
LAMBDA_LOCAL_PRAG="${LAMBDA_LOCAL_PRAG:-1e-3}"

EXP_NAME="stage${STAGE}_${SYSTEM}_${ARM}_bp${B_PROXY}_ts${TARGET_SIZE}_seed${SEED}"
LOG_DIR="${BASE_OUT}/${EXP_NAME}"

COMMON_ARGS=(
  --config lista_nonlinear
  --env "${SYSTEM}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --seed "${SEED}"
  --sequence_length 1
  --monitor_support
  --support_monitor_every 500
  --support_threshold "${SUPPORT_THRESHOLD}"
  --device "${DEVICE}"
  --log_dir "${LOG_DIR}"
  --skip_eval
  --skip_basin_eval
)

if [ "${SYSTEM}" = "lyapunov" ]; then
  COMMON_ARGS+=(
    --lyapunov_dim "${DIM}"
    --lyapunov_num_basins "${NUM_BASINS_GT}"
    --lyapunov_extend_mode embed
    --lyapunov_points_mode fixed
  )
fi

ARM_ARGS=()

case "${ARM}" in
  diag_c1)
    ARM_ARGS=(
      --target_size "${TARGET_SIZE}"
      --sparsity_coeff 1.0
      --k_structure diagonal
    )
    ;;

  bd_c1)
    K_BLOCK_SIZE=$(( TARGET_SIZE / B_PROXY ))
    if [ "${K_BLOCK_SIZE}" -lt 1 ]; then
      K_BLOCK_SIZE=1
    fi
    ARM_ARGS=(
      --target_size "${TARGET_SIZE}"
      --sparsity_coeff 1.0
      --k_structure block_diagonal
      --k_block_size "${K_BLOCK_SIZE}"
      --block_balance_loss kl_uniform
      --block_balance_weight 1.0
    )
    ;;

  bd_c2)
    K_BLOCK_SIZE=$(( TARGET_SIZE / B_PROXY ))
    if [ "${K_BLOCK_SIZE}" -lt 1 ]; then
      K_BLOCK_SIZE=1
    fi
    ARM_ARGS=(
      --target_size "${TARGET_SIZE}"
      --sparsity_coeff 1.0
      --k_structure block_diagonal
      --k_block_size "${K_BLOCK_SIZE}"
      --block_balance_loss kl_uniform
      --block_balance_weight 1.0
      --block_one_block_loss top1_margin
      --block_top1_margin 0.05
      --block_one_block_weight 0.3
    )
    ;;

  ah_iso)
    D_BASIN=$(( TARGET_SIZE / B_PROXY ))
    if [ "${D_BASIN}" -lt 1 ]; then
      D_BASIN=1
    fi
    D_GLOBAL=$(( TARGET_SIZE - B_PROXY * D_BASIN ))
    if [ "${D_GLOBAL}" -lt 0 ]; then
      D_GLOBAL=0
    fi
    ARM_ARGS=(
      --structured
      --d_global "${D_GLOBAL}"
      --num_basins "${B_PROXY}"
      --d_basin "${D_BASIN}"
      --lambda_exclusivity 0.0
      --lambda_global 0.0
      --lambda_local 0.0
      --lambda_sparsity 1.0
      --excl_warmup_steps 0
    )
    ;;

  ah_prag)
    D_BASIN=$(( TARGET_SIZE / B_PROXY ))
    if [ "${D_BASIN}" -lt 1 ]; then
      D_BASIN=1
    fi
    D_GLOBAL=$(( TARGET_SIZE - B_PROXY * D_BASIN ))
    if [ "${D_GLOBAL}" -lt 0 ]; then
      D_GLOBAL=0
    fi
    ARM_ARGS=(
      --structured
      --d_global "${D_GLOBAL}"
      --num_basins "${B_PROXY}"
      --d_basin "${D_BASIN}"
      --lambda_exclusivity 0.05
      --lambda_sparsity 0.3
      --lambda_global "${LAMBDA_GLOBAL_PRAG}"
      --lambda_local "${LAMBDA_LOCAL_PRAG}"
      --excl_warmup_steps 2000
    )
    ;;

  ah_prag_no_excl)
    D_BASIN=$(( TARGET_SIZE / B_PROXY ))
    if [ "${D_BASIN}" -lt 1 ]; then
      D_BASIN=1
    fi
    D_GLOBAL=$(( TARGET_SIZE - B_PROXY * D_BASIN ))
    if [ "${D_GLOBAL}" -lt 0 ]; then
      D_GLOBAL=0
    fi
    ARM_ARGS=(
      --structured
      --d_global "${D_GLOBAL}"
      --num_basins "${B_PROXY}"
      --d_basin "${D_BASIN}"
      --lambda_exclusivity 0.0
      --lambda_sparsity 0.3
      --lambda_global "${LAMBDA_GLOBAL_PRAG}"
      --lambda_local "${LAMBDA_LOCAL_PRAG}"
      --excl_warmup_steps 0
    )
    ;;

  *)
    echo "Unknown ARM='${ARM}'. Expected: diag_c1, bd_c1, bd_c2, ah_iso, ah_prag, ah_prag_no_excl"
    exit 1
    ;;
esac

echo "============================================="
echo "LQR Decision Trial"
echo "Stage: ${STAGE}"
echo "Arm: ${ARM}"
echo "System: ${SYSTEM}"
echo "Target size: ${TARGET_SIZE}"
echo "B_proxy: ${B_PROXY}"
echo "Seed: ${SEED}"
echo "Log dir: ${LOG_DIR}"
if [ "${ARM}" = "ah_prag" ] || [ "${ARM}" = "ah_prag_no_excl" ]; then
  echo "Arrowhead lock: lambda_global=${LAMBDA_GLOBAL_PRAG}, lambda_local=${LAMBDA_LOCAL_PRAG}"
fi
echo "============================================="

uv run python tools/train.py "${COMMON_ARGS[@]}" "${ARM_ARGS[@]}"

CKPT=$(ls -t "${LOG_DIR}"/*/checkpoint.pt 2>/dev/null | head -1)
if [ -z "${CKPT}" ]; then
  echo "No checkpoint found in ${LOG_DIR}"
  exit 1
fi
RUN_DIR=$(dirname "${CKPT}")

echo "Run directory: ${RUN_DIR}"

echo "[1/5] checkpoint evaluation"
uv run python tools/evaluate_checkpoints.py \
  --run_dir "${RUN_DIR}" \
  --system "${SYSTEM}" \
  --checkpoints checkpoint.pt \
  --device "${DEVICE}"

echo "[2/5] eigenvalue analysis"
EIGEN_ARGS=(
  --checkpoint "${CKPT}"
  --system "${SYSTEM}"
  --num_trajectories 100
  --output_dir "${RUN_DIR}/eigenvalue_analysis"
  --device "${DEVICE}"
  --seed 42
)
if [ "${SYSTEM}" = "lyapunov" ]; then
  EIGEN_ARGS+=(--correlate_basins)
fi
uv run python tools/analyze_k_eigenvalues.py "${EIGEN_ARGS[@]}"

echo "[3/5] support + cosine diagnostics"
uv run python tools/evaluate_support_uniqueness.py \
  --checkpoint "${CKPT}" \
  --system "${SYSTEM}" \
  --support_mode mean \
  --cosine_diag \
  --threshold_sweep \
  --output_dir "${RUN_DIR}/support_eval" \
  --device "${DEVICE}"

echo "[4/5] cosine sensitivity diagnostics"
uv run python tools/diagnose_cosine_separation.py \
  --checkpoint "${CKPT}" \
  --system "${SYSTEM}" \
  --output_dir "${RUN_DIR}/cosine_diag" \
  --device "${DEVICE}"

echo "[5/5] LQR readiness evaluation"
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
  --stage "${STAGE}" \
  --arm "${ARM}" \
  --run_seed "${SEED}" \
  --target_size "${TARGET_SIZE}" \
  --output_dir "${RUN_DIR}/lqr_readiness"

echo "Completed ${EXP_NAME}"
