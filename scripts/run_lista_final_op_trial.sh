#!/bin/bash
set -euo pipefail

# Run one LISTA final-op experiment trial.
#
# Required env vars:
#   PHASE SYSTEM FINAL_OP TARGET_SIZE SEED
#
# Optional env vars:
#   BASE_OUT DEVICE NUM_STEPS BATCH_SIZE SUPPORT_THRESHOLD
#   SPARSITY_COEFF K_STRUCTURE DIM NUM_BASINS_GT NUM_BASINS_PROXY
#   LAMBDA_GLOBAL LAMBDA_LOCAL LAMBDA_SPARSE_STRUCT

if [ -z "${PHASE:-}" ] || [ -z "${SYSTEM:-}" ] || [ -z "${FINAL_OP:-}" ] || [ -z "${TARGET_SIZE:-}" ] || [ -z "${SEED:-}" ]; then
  echo "Missing required env vars. Need: PHASE SYSTEM FINAL_OP TARGET_SIZE SEED"
  exit 1
fi

if [ "${FINAL_OP}" != "shrink" ] && [ "${FINAL_OP}" != "relu" ]; then
  echo "Invalid FINAL_OP='${FINAL_OP}'. Expected shrink or relu."
  exit 1
fi

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_final_op_experiment}"
DEVICE="${DEVICE:-cuda}"
NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-1e-3}"
SPARSITY_COEFF="${SPARSITY_COEFF:-1.0}"
K_STRUCTURE="${K_STRUCTURE:-dense}"
DIM="${DIM:-8}"
NUM_BASINS_GT="${NUM_BASINS_GT:-13}"
NUM_BASINS_PROXY="${NUM_BASINS_PROXY:-20}"
LAMBDA_GLOBAL="${LAMBDA_GLOBAL:-1e-4}"
LAMBDA_LOCAL="${LAMBDA_LOCAL:-1e-3}"
LAMBDA_SPARSE_STRUCT="${LAMBDA_SPARSE_STRUCT:-0.3}"

SP_TAG="${SPARSITY_COEFF//./p}"
K_TAG="${K_STRUCTURE//_/-}"
EXP_NAME="phase${PHASE}__sys-${SYSTEM}__k-${K_TAG}__ts-${TARGET_SIZE}__op-${FINAL_OP}__sp-${SP_TAG}__seed-${SEED}"
LOG_DIR="${BASE_OUT}/${EXP_NAME}"

COMMON_ARGS=(
  --config lista_nonlinear
  --env "${SYSTEM}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --target_size "${TARGET_SIZE}"
  --seed "${SEED}"
  --pairwise
  --sparsity_coeff "${SPARSITY_COEFF}"
  --lista_final_op "${FINAL_OP}"
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

K_ARGS=()
case "${K_STRUCTURE}" in
  dense)
    ;;
  diagonal)
    K_ARGS+=(--k_structure diagonal)
    ;;
  block_diagonal)
    K_BLOCK_SIZE=$(( TARGET_SIZE / NUM_BASINS_PROXY ))
    if [ "${K_BLOCK_SIZE}" -lt 1 ]; then
      K_BLOCK_SIZE=1
    fi
    K_ARGS+=(
      --k_structure block_diagonal
      --k_block_size "${K_BLOCK_SIZE}"
    )
    ;;
  arrowhead_no_excl)
    D_BASIN=$(( TARGET_SIZE / NUM_BASINS_PROXY ))
    if [ "${D_BASIN}" -lt 1 ]; then
      D_BASIN=1
    fi
    D_GLOBAL=$(( TARGET_SIZE - NUM_BASINS_PROXY * D_BASIN ))
    if [ "${D_GLOBAL}" -lt 0 ]; then
      D_GLOBAL=0
    fi
    K_ARGS+=(
      --structured
      --d_global "${D_GLOBAL}"
      --num_basins "${NUM_BASINS_PROXY}"
      --d_basin "${D_BASIN}"
      --lambda_exclusivity 0.0
      --lambda_global "${LAMBDA_GLOBAL}"
      --lambda_local "${LAMBDA_LOCAL}"
      --lambda_sparsity "${LAMBDA_SPARSE_STRUCT}"
      --excl_warmup_steps 0
    )
    ;;
  *)
    echo "Unknown K_STRUCTURE='${K_STRUCTURE}'. Expected dense|diagonal|block_diagonal|arrowhead_no_excl."
    exit 1
    ;;
esac

echo "============================================="
echo "LISTA Final-Op Trial"
echo "Phase: ${PHASE}"
echo "System: ${SYSTEM}"
echo "K structure: ${K_STRUCTURE}"
echo "Target size: ${TARGET_SIZE}"
echo "Final op: ${FINAL_OP}"
echo "Sparsity coeff: ${SPARSITY_COEFF}"
echo "Seed: ${SEED}"
echo "Log dir: ${LOG_DIR}"
echo "============================================="

uv run python tools/train.py "${COMMON_ARGS[@]}" "${K_ARGS[@]}"

CKPT=$(ls -t "${LOG_DIR}"/*/checkpoint.pt 2>/dev/null | head -1)
if [ -z "${CKPT}" ]; then
  LAST_CKPT=$(ls -t "${LOG_DIR}"/*/last.pt 2>/dev/null | head -1)
  if [ -n "${LAST_CKPT}" ]; then
    echo "checkpoint.pt not found; falling back to last.pt and creating checkpoint alias"
    CKPT_DIR=$(dirname "${LAST_CKPT}")
    cp "${LAST_CKPT}" "${CKPT_DIR}/checkpoint.pt"
    CKPT="${CKPT_DIR}/checkpoint.pt"
  else
    echo "No checkpoint found in ${LOG_DIR} (checked checkpoint.pt and last.pt)"
    exit 1
  fi
fi
RUN_DIR=$(dirname "${CKPT}")
echo "Run directory: ${RUN_DIR}"

echo "[1/3] checkpoint evaluation"
uv run python tools/evaluate_checkpoints.py \
  --run_dir "${RUN_DIR}" \
  --system "${SYSTEM}" \
  --checkpoints checkpoint.pt \
  --device "${DEVICE}"

echo "[2/3] eigenvalue analysis"
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

echo "[3/3] support + cosine diagnostics"
uv run python tools/evaluate_support_uniqueness.py \
  --checkpoint "${CKPT}" \
  --system "${SYSTEM}" \
  --support_mode mean \
  --cosine_diag \
  --threshold_sweep \
  --output_dir "${RUN_DIR}/support_eval" \
  --device "${DEVICE}"

echo "Completed ${EXP_NAME}"
