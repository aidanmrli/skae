#!/bin/bash

#SBATCH --job-name=seq_sr_sweep
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -o /network/scratch/l/lia/skae/seq_sr_sweep-%A_%a.out
#SBATCH --array=0-47
#SBATCH --requeue

# ============================================================================
# Sequence-Length Spectral-Stability Sweep (Lyapunov + LISTA variants)
# ============================================================================
# Hypothesis:
#   Longer sequence training windows (L) provide multi-step gradient pressure
#   that pushes Koopman spectral radius toward stability (SR <= 1).
#
# Grid (default):
#   sequence_length: 4, 8, 12, 16
#   k_structure:     dense, diagonal, block_diagonal, arrowhead
#   latent_dim:      64, 128, 256
# Total jobs: 4 * 4 * 3 = 48 (array 0-47)
#
# For each run:
#   1) Train with sequence loss (--sequence --sequence_length L)
#   2) Run long-horizon eval (checkpoint.pt)
#   3) Run Koopman eigenvalue analysis (spectral radius)
# ============================================================================

module load cuda/12.6.0
source .venv/bin/activate

SEQUENCE_LENGTHS=(4 8 12 16)
K_STRUCTURES=(dense diagonal block_diagonal arrowhead)
TARGET_SIZES=(64 128 256)

NUM_SEQ=${#SEQUENCE_LENGTHS[@]}
NUM_K=${#K_STRUCTURES[@]}
NUM_TS=${#TARGET_SIZES[@]}

IDX=${SLURM_ARRAY_TASK_ID}
TS_IDX=$(( IDX % NUM_TS ))
IDX=$(( IDX / NUM_TS ))
K_IDX=$(( IDX % NUM_K ))
IDX=$(( IDX / NUM_K ))
SEQ_IDX=$(( IDX % NUM_SEQ ))

SEQUENCE_LENGTH=${SEQUENCE_LENGTHS[$SEQ_IDX]}
K_STRUCTURE=${K_STRUCTURES[$K_IDX]}
TARGET_SIZE=${TARGET_SIZES[$TS_IDX]}

DIM="${DIM:-8}"
NUM_BASINS="${NUM_BASINS:-13}"
NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"

# Keep core loss weights matched across structures.
RECONST_COEFF="${RECONST_COEFF:-1.0}"
PRED_COEFF="${PRED_COEFF:-0.0}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.3}"

# StructuredLISTAKM-specific regularizers.
LAMBDA_EXCLUSIVITY="${LAMBDA_EXCLUSIVITY:-0.05}"
LAMBDA_SPARSITY="${LAMBDA_SPARSITY:-0.3}"
EXCL_WARMUP="${EXCL_WARMUP:-2000}"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/sequence_length_spectral_sweep}"
EXP_NAME="dim${DIM}_nb${NUM_BASINS}_L${SEQUENCE_LENGTH}_ts${TARGET_SIZE}_${K_STRUCTURE}"
LOG_DIR="${BASE_OUT}/${EXP_NAME}"

echo "============================================="
echo "Sequence-Length Spectral-Stability Sweep"
echo "Job ID: $SLURM_JOB_ID  Array Task: $SLURM_ARRAY_TASK_ID"
echo "SEQUENCE_LENGTH: $SEQUENCE_LENGTH"
echo "K_STRUCTURE: $K_STRUCTURE"
echo "TARGET_SIZE: $TARGET_SIZE"
echo "NUM_STEPS: $NUM_STEPS  BATCH_SIZE: $BATCH_SIZE"
echo "RECONST_COEFF: $RECONST_COEFF  PRED_COEFF: $PRED_COEFF  SPARSITY_COEFF: $SPARSITY_COEFF"
echo "LOG_DIR: $LOG_DIR"
echo "============================================="

COMMON_ARGS=(
  --config lista_nonlinear
  --env lyapunov
  --lyapunov_dim "$DIM"
  --lyapunov_num_basins "$NUM_BASINS"
  --lyapunov_extend_mode embed
  --lyapunov_points_mode fixed
  --num_steps "$NUM_STEPS"
  --batch_size "$BATCH_SIZE"
  --reconst_coeff "$RECONST_COEFF"
  --pred_coeff "$PRED_COEFF"
  --sparsity_coeff "$SPARSITY_COEFF"
  --sequence
  --sequence_length "$SEQUENCE_LENGTH"
  --skip_eval
  --skip_basin_eval
  --device cuda
  --log_dir "$LOG_DIR"
)

if [ "$K_STRUCTURE" = "arrowhead" ]; then
  # Match total latent size as closely as possible:
  #   total_dim = d_global + NUM_BASINS * d_basin ~= TARGET_SIZE
  D_BASIN=$(( TARGET_SIZE / NUM_BASINS ))
  if [ "$D_BASIN" -lt 1 ]; then
    D_BASIN=1
  fi
  D_GLOBAL=$(( TARGET_SIZE - NUM_BASINS * D_BASIN ))
  if [ "$D_GLOBAL" -lt 0 ]; then
    D_GLOBAL=0
  fi

  echo "Arrowhead layout: d_global=$D_GLOBAL d_basin=$D_BASIN num_basins=$NUM_BASINS"
  uv run python tools/train.py \
    "${COMMON_ARGS[@]}" \
    --structured \
    --d_global "$D_GLOBAL" \
    --num_basins "$NUM_BASINS" \
    --d_basin "$D_BASIN" \
    --lambda_exclusivity "$LAMBDA_EXCLUSIVITY" \
    --lambda_sparsity "$LAMBDA_SPARSITY" \
    --excl_warmup_steps "$EXCL_WARMUP"
else
  K_ARGS=(--k_structure "$K_STRUCTURE")
  if [ "$K_STRUCTURE" = "block_diagonal" ]; then
    K_BLOCK_SIZE=$(( TARGET_SIZE / NUM_BASINS ))
    if [ "$K_BLOCK_SIZE" -lt 1 ]; then
      K_BLOCK_SIZE=1
    fi
    K_ARGS+=(--k_block_size "$K_BLOCK_SIZE")
    echo "Block size: $K_BLOCK_SIZE"
  fi

  uv run python tools/train.py \
    "${COMMON_ARGS[@]}" \
    --target_size "$TARGET_SIZE" \
    "${K_ARGS[@]}"
fi

TRAIN_EXIT=$?

CKPT=$(ls -t "${LOG_DIR}"/*/checkpoint.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ] && [ $TRAIN_EXIT -eq 0 ]; then
  RUN_DIR=$(dirname "$CKPT")

  echo ""
  echo ">>> Running long-horizon evaluation..."
  uv run python tools/evaluate_checkpoints.py \
    --run_dir "$RUN_DIR" \
    --system lyapunov \
    --checkpoints checkpoint.pt \
    --device cuda
  EVAL_EXIT=$?

  echo ""
  echo ">>> Running eigenvalue analysis (spectral radius)..."
  uv run python tools/analyze_k_eigenvalues.py \
    --checkpoint "$CKPT" \
    --system lyapunov \
    --correlate_basins \
    --num_trajectories 100 \
    --output_dir "${RUN_DIR}/eigenvalue_analysis" \
    --device cuda
  EIGEN_EXIT=$?

  if [ $EVAL_EXIT -eq 0 ] && [ $EIGEN_EXIT -eq 0 ]; then
    FINAL_EXIT=0
  else
    FINAL_EXIT=1
  fi
else
  EVAL_EXIT=1
  EIGEN_EXIT=1
  FINAL_EXIT=$TRAIN_EXIT
fi

echo "============================================="
echo "Experiment: $EXP_NAME"
echo "TRAIN_EXIT=$TRAIN_EXIT EVAL_EXIT=$EVAL_EXIT EIGEN_EXIT=$EIGEN_EXIT"
echo "End Time: $(date)"
echo "Exit Code: $FINAL_EXIT"
echo "============================================="
exit $FINAL_EXIT
