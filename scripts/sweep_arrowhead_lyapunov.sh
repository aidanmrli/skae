#!/bin/bash

#SBATCH --job-name=lyap_arrowhead
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH -o /network/scratch/l/lia/skae/lyap_arrowhead-%A_%a.out
#SBATCH --array=0-4
#SBATCH --requeue

# ============================================================================
# Arrowhead Koopman Sweep (StructuredLISTAKM on Lyapunov-HD)
# ============================================================================
# 5 jobs matching total latent dims to the K structure sweep:
#   64, 128, 256, 512, 1024
#
# For each total dim N with B=13 basins:
#   d_global + B * d_basin = N
#   We set d_global = N mod B (absorb remainder), d_basin = N / B
#
# Post-training: runs threshold sweep + cosine similarity evaluation.
# ============================================================================

module load cuda/12.6.0
source .venv/bin/activate

# Total latent dims matching the K structure sweep
TOTAL_DIMS=(64 128 256 512 1024)
TOTAL_DIM="${TOTAL_DIMS[$SLURM_ARRAY_TASK_ID]}"

B=13  # number of basin slots = number of GT basins

# Compute d_basin and d_global so that d_global + B * d_basin = TOTAL_DIM
D_BASIN=$(( TOTAL_DIM / B ))
D_GLOBAL=$(( TOTAL_DIM - B * D_BASIN ))
# If d_basin rounds to 0 (TOTAL_DIM < B), fall back to minimal
if [ "$D_BASIN" -lt 1 ]; then
  D_BASIN=1
  D_GLOBAL=$(( TOTAL_DIM - B ))
  if [ "$D_GLOBAL" -lt 0 ]; then
    D_GLOBAL=0
  fi
fi

DIM="${DIM:-8}"
NUM_BASINS="${NUM_BASINS:-13}"
NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LAMBDA_EXCLUSIVITY="${LAMBDA_EXCLUSIVITY:-0.05}"
LAMBDA_SPARSITY="${LAMBDA_SPARSITY:-0.3}"
EXCL_WARMUP="${EXCL_WARMUP:-2000}"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lyapunov_k_structure_sweep}"
EXP_NAME="dim${DIM}_nb${NUM_BASINS}_ts${TOTAL_DIM}_arrowhead"
LOG_DIR="${BASE_OUT}/${EXP_NAME}"

echo "============================================="
echo "Arrowhead Koopman (StructuredLISTAKM) Sweep"
echo "Job ID: $SLURM_JOB_ID  Array Task: $SLURM_ARRAY_TASK_ID"
echo "DIM: $DIM  NUM_BASINS_GT: $NUM_BASINS"
echo "TOTAL_DIM: $TOTAL_DIM  D_GLOBAL: $D_GLOBAL  D_BASIN: $D_BASIN  B: $B"
echo "Actual total: $(( D_GLOBAL + B * D_BASIN ))"
echo "LAMBDA_EXCLUSIVITY: $LAMBDA_EXCLUSIVITY  LAMBDA_SPARSITY: $LAMBDA_SPARSITY"
echo "LOG_DIR: $LOG_DIR"
echo "============================================="

uv run python tools/train.py \
  --config lista_nonlinear \
  --env lyapunov \
  --lyapunov_dim "$DIM" \
  --lyapunov_num_basins "$NUM_BASINS" \
  --lyapunov_extend_mode embed \
  --lyapunov_points_mode fixed \
  --structured \
  --d_global "$D_GLOBAL" \
  --num_basins "$B" \
  --d_basin "$D_BASIN" \
  --lambda_exclusivity "$LAMBDA_EXCLUSIVITY" \
  --lambda_sparsity "$LAMBDA_SPARSITY" \
  --excl_warmup_steps "$EXCL_WARMUP" \
  --num_steps "$NUM_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --pairwise \
  --monitor_support \
  --support_monitor_every 500 \
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
    --threshold_sweep \
    --output_dir "$EVAL_DIR" \
    --device cuda
fi

echo "============================================="
echo "Experiment: $EXP_NAME"
echo "End Time: $(date)"
echo "Exit Code: $TRAIN_EXIT"
echo "============================================="
