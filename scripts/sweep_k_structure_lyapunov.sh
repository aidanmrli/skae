#!/bin/bash

#SBATCH --job-name=lyap_kstruct
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH -o /network/scratch/l/lia/skae/lyap_kstruct-%A_%a.out
#SBATCH --array=0-14
#SBATCH --requeue

# ============================================================================
# Koopman Structure × Target Size Sweep (LISTAKM on Lyapunov-HD)
# ============================================================================
# 5 target sizes × 3 K structures = 15 jobs
#   K structures: dense, diagonal, block_diagonal
#   Target sizes: 64, 128, 256, 512, 1024
#
# Post-training: runs threshold sweep + cosine similarity evaluation.
# ============================================================================

module load cuda/12.6.0
source .venv/bin/activate

TARGET_SIZES=(64 128 256 512 1024)
K_STRUCTURES=(dense diagonal block_diagonal)

# Decode array index -> (target_size, k_structure)
TSIZE_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
KSTRUCT_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
TARGET_SIZE="${TARGET_SIZES[$TSIZE_IDX]}"
K_STRUCTURE="${K_STRUCTURES[$KSTRUCT_IDX]}"

DIM="${DIM:-8}"
NUM_BASINS="${NUM_BASINS:-13}"
SPARSITY="${SPARSITY:-1.0}"
NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-1e-3}"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lyapunov_k_structure_sweep}"
EXP_NAME="dim${DIM}_nb${NUM_BASINS}_ts${TARGET_SIZE}_${K_STRUCTURE}"
LOG_DIR="${BASE_OUT}/${EXP_NAME}"

echo "============================================="
echo "Koopman Structure Sweep"
echo "Job ID: $SLURM_JOB_ID  Array Task: $SLURM_ARRAY_TASK_ID"
echo "DIM: $DIM  NUM_BASINS: $NUM_BASINS"
echo "TARGET_SIZE: $TARGET_SIZE  K_STRUCTURE: $K_STRUCTURE"
echo "NUM_STEPS: $NUM_STEPS  BATCH_SIZE: $BATCH_SIZE"
echo "LOG_DIR: $LOG_DIR"
echo "============================================="

# Build K-structure args
K_ARGS="--k_structure $K_STRUCTURE"
if [ "$K_STRUCTURE" = "block_diagonal" ]; then
  # Auto block size: target_size // num_basins
  K_BLOCK_SIZE=$(( TARGET_SIZE / NUM_BASINS ))
  if [ "$K_BLOCK_SIZE" -lt 1 ]; then
    K_BLOCK_SIZE=1
  fi
  K_ARGS="$K_ARGS --k_block_size $K_BLOCK_SIZE"
  echo "Block size: $K_BLOCK_SIZE"
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
  $K_ARGS \
  --sequence_length 1 \
  --monitor_support \
  --support_monitor_every 500 \
  --support_threshold "$SUPPORT_THRESHOLD" \
  --device cuda \
  --log_dir "$LOG_DIR"

TRAIN_EXIT=$?

CKPT=$(ls -t "${LOG_DIR}"/*/checkpoint.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ] && [ $TRAIN_EXIT -eq 0 ]; then
  echo ""
  echo ">>> Running support uniqueness evaluation with threshold sweep..."
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
