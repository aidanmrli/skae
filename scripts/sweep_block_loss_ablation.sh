#!/bin/bash

#SBATCH --job-name=lyap_blk_loss
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH -o /network/scratch/l/lia/skae/lyap_blk_loss-%A_%a.out
#SBATCH --array=0-23
#SBATCH --requeue

# ============================================================================
# Block Loss Ablation Sweep (LISTAKM + block_diagonal K on Lyapunov-HD)
# ============================================================================
# Loss conditions (6) x target sizes (4) = 24 jobs
#   Conditions: control, low_entropy, pairwise_overlap, top1_margin,
#               usage_entropy, kl_uniform
#   Target sizes: 64, 128, 256, 512
#
# Each job enables ONE block loss at a time (no combos).
# ============================================================================

module load cuda/12.6.0
source .venv/bin/activate

TARGET_SIZES=(64 128 256 512)
LOSS_NAMES=(control low_entropy pairwise_overlap top1_margin usage_entropy kl_uniform)

LOSS_COUNT=${#LOSS_NAMES[@]}
TSIZE_IDX=$(( SLURM_ARRAY_TASK_ID / LOSS_COUNT ))
LOSS_IDX=$(( SLURM_ARRAY_TASK_ID % LOSS_COUNT ))

TARGET_SIZE="${TARGET_SIZES[$TSIZE_IDX]}"
LOSS_NAME="${LOSS_NAMES[$LOSS_IDX]}"

DIM="${DIM:-8}"
NUM_BASINS="${NUM_BASINS:-13}"
NUM_BLOCKS="${NUM_BLOCKS:-20}"
SPARSITY="${SPARSITY:-1.0}"
NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-1e-3}"

ONE_WEIGHT="${ONE_WEIGHT:-1.0}"
BAL_WEIGHT="${BAL_WEIGHT:-1.0}"
TOP1_MARGIN="${TOP1_MARGIN:-0.1}"
ENERGY_NORM="${ENERGY_NORM:-l2}"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lyapunov_block_loss_sweep}"
EXP_NAME="dim${DIM}_nb${NUM_BASINS}_ts${TARGET_SIZE}_blkdiag_${LOSS_NAME}"
LOG_DIR="${BASE_OUT}/${EXP_NAME}"

# Block-diagonal K with block size derived from NUM_BLOCKS (not GT basins)
K_BLOCK_SIZE=$(( TARGET_SIZE / NUM_BLOCKS ))
if [ "$K_BLOCK_SIZE" -lt 1 ]; then
  K_BLOCK_SIZE=1
fi

ONE_BLOCK_LOSS="none"
BALANCE_LOSS="none"
ONE_BLOCK_WEIGHT="0"
BALANCE_WEIGHT="0"
USE_BLOCK_LOSS=0

case "$LOSS_NAME" in
  control)
    ;;
  low_entropy)
    ONE_BLOCK_LOSS="low_entropy"
    ONE_BLOCK_WEIGHT="$ONE_WEIGHT"
    USE_BLOCK_LOSS=1
    ;;
  pairwise_overlap)
    ONE_BLOCK_LOSS="pairwise_overlap"
    ONE_BLOCK_WEIGHT="$ONE_WEIGHT"
    USE_BLOCK_LOSS=1
    ;;
  top1_margin)
    ONE_BLOCK_LOSS="top1_margin"
    ONE_BLOCK_WEIGHT="$ONE_WEIGHT"
    USE_BLOCK_LOSS=1
    ;;
  usage_entropy)
    BALANCE_LOSS="usage_entropy"
    BALANCE_WEIGHT="$BAL_WEIGHT"
    USE_BLOCK_LOSS=1
    ;;
  kl_uniform)
    BALANCE_LOSS="kl_uniform"
    BALANCE_WEIGHT="$BAL_WEIGHT"
    USE_BLOCK_LOSS=1
    ;;
  *)
    echo "Unknown loss: $LOSS_NAME"
    exit 1
    ;;
esac

echo "============================================="
echo "Block Loss Ablation Sweep"
echo "Job ID: $SLURM_JOB_ID  Array Task: $SLURM_ARRAY_TASK_ID"
echo "DIM: $DIM  NUM_BASINS: $NUM_BASINS"
echo "TARGET_SIZE: $TARGET_SIZE  K_STRUCTURE: block_diagonal"
echo "NUM_BLOCKS: $NUM_BLOCKS  K_BLOCK_SIZE: $K_BLOCK_SIZE"
echo "LOSS_NAME: $LOSS_NAME"
echo "ONE_BLOCK_LOSS: $ONE_BLOCK_LOSS (w=$ONE_BLOCK_WEIGHT)"
echo "BALANCE_LOSS: $BALANCE_LOSS (w=$BALANCE_WEIGHT)"
echo "TOP1_MARGIN: $TOP1_MARGIN"
echo "ENERGY_NORM: $ENERGY_NORM"
echo "NUM_STEPS: $NUM_STEPS  BATCH_SIZE: $BATCH_SIZE"
echo "LOG_DIR: $LOG_DIR"
echo "============================================="

BLOCK_ARGS=()
if [ "$USE_BLOCK_LOSS" -eq 1 ]; then
  BLOCK_ARGS+=(--block_loss)
  BLOCK_ARGS+=(--block_one_block_loss "$ONE_BLOCK_LOSS")
  BLOCK_ARGS+=(--block_one_block_weight "$ONE_BLOCK_WEIGHT")
  BLOCK_ARGS+=(--block_top1_margin "$TOP1_MARGIN")
  BLOCK_ARGS+=(--block_balance_loss "$BALANCE_LOSS")
  BLOCK_ARGS+=(--block_balance_weight "$BALANCE_WEIGHT")
  BLOCK_ARGS+=(--block_energy_norm "$ENERGY_NORM")
fi

uv run python tools/train.py \
  --config lista_nonlinear \
  --env lyapunov \
  --lyapunov_dim "$DIM" \
  --lyapunov_num_basins "$NUM_BASINS" \
  --lyapunov_extend_mode embed \
  --lyapunov_points_mode fixed \
  --num_steps "$NUM_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --target_size "$TARGET_SIZE" \
  --sparsity_coeff "$SPARSITY" \
  --k_structure block_diagonal \
  --k_block_size "$K_BLOCK_SIZE" \
  --sequence_length 1 \
  --monitor_support \
  --support_monitor_every 500 \
  --support_threshold "$SUPPORT_THRESHOLD" \
  "${BLOCK_ARGS[@]}" \
  --device cuda \
  --log_dir "$LOG_DIR"

TRAIN_EXIT=$?

CKPT=$(ls -t "${LOG_DIR}"/*/checkpoint.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ] && [ $TRAIN_EXIT -eq 0 ]; then
  echo ""
  echo ">>> Running cosine + threshold sweep evaluation..."
  EVAL_DIR="${LOG_DIR}/support_eval"
  mkdir -p "$EVAL_DIR"
  uv run python tools/evaluate_support_uniqueness.py \
    --checkpoint "$CKPT" \
    --system lyapunov \
    --support_mode mean \
    --cosine_diag \
    --threshold_sweep \
    --output_dir "$EVAL_DIR" \
    --device cuda
fi

echo "============================================="
echo "Experiment: $EXP_NAME"
echo "End Time: $(date)"
echo "Exit Code: $TRAIN_EXIT"
echo "============================================="
