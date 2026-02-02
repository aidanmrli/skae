#!/bin/bash

#SBATCH --job-name=lyap_hd_tsize
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH -o /network/scratch/l/lia/skae/lyap_hd_tsize-%A_%a.out
#SBATCH --array=0-4
#SBATCH --requeue

# ============================================================================
# Lyapunov-HD Target Size Sweep (simple LISTA baseline)
# ============================================================================
# Goal: find target_size that yields unique basin supports.
# ============================================================================

module load cuda/12.6.0
source .venv/bin/activate

TARGET_SIZES=(64 128 256 512 1024)
TARGET_SIZE="${TARGET_SIZES[$SLURM_ARRAY_TASK_ID]}"

DIM="${DIM:-8}"
NUM_BASINS="${NUM_BASINS:-13}"
SPARSITY="${SPARSITY:-1.0}"
NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-1e-3}"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lyapunov_hd_target_sweep}"
EXP_NAME="dim${DIM}_nb${NUM_BASINS}_ts${TARGET_SIZE}_sp${SPARSITY}"
LOG_DIR="${BASE_OUT}/${EXP_NAME}"

echo "============================================="
echo "Lyapunov-HD Target Size Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "DIM: $DIM  NUM_BASINS: $NUM_BASINS"
echo "TARGET_SIZE: $TARGET_SIZE  SPARSITY: $SPARSITY"
echo "NUM_STEPS: $NUM_STEPS  BATCH_SIZE: $BATCH_SIZE"
echo "LOG_DIR: $LOG_DIR"
echo "============================================="

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
  --pairwise \
  --monitor_support \
  --support_monitor_every 500 \
  --support_threshold "$SUPPORT_THRESHOLD" \
  --device cuda \
  --log_dir "$LOG_DIR"

TRAIN_EXIT=$?

CKPT=$(ls -t "${LOG_DIR}"/*/checkpoint.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ] && [ $TRAIN_EXIT -eq 0 ]; then
  echo ""
  echo ">>> Running support uniqueness evaluation..."
  EVAL_DIR="${LOG_DIR}/support_eval"
  mkdir -p "$EVAL_DIR"
  uv run python tools/evaluate_support_uniqueness.py \
    --checkpoint "$CKPT" \
    --system lyapunov \
    --support_threshold "$SUPPORT_THRESHOLD" \
    --support_mode mean \
    --output_dir "$EVAL_DIR" \
    --device cuda
fi

echo "============================================="
echo "Experiment: $EXP_NAME"
echo "End Time: $(date)"
echo "Exit Code: $TRAIN_EXIT"
echo "============================================="
