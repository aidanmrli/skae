#!/bin/bash

#SBATCH --job-name=lyap_arrow_noexcl
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH -o /network/scratch/l/lia/skae/lyap_arrow_noexcl-%A_%a.out
#SBATCH --array=0-4
#SBATCH --requeue

# ============================================================================
# Arrowhead K Structure ONLY — No Exclusivity Loss (Fair Comparison)
# ============================================================================
# Purpose: Isolate the effect of arrowhead K structure on basin separation.
#
# The previous arrowhead sweep used exclusivity loss (lambda_excl=0.05),
# structured sparsity (lambda_global, lambda_local), and a warmup schedule.
# These extra losses confound the comparison with dense/diagonal/block_diagonal
# which only use standard L1 sparsity (sparsity_coeff=1.0, alpha=1.0).
#
# This sweep uses StructuredLISTAKM but sets:
#   - lambda_exclusivity = 0.0  (no exclusivity penalty)
#   - lambda_global = 0.0       (no structured global sparsity)
#   - lambda_local = 0.0        (no structured local sparsity)
#   - lambda_sparsity = 1.0     (matches LISTAKM: sparsity_coeff * alpha = 1.0)
#   - excl_warmup_steps = 0     (no warmup — sparsity at full strength from step 0,
#                                 matching LISTAKM which has no warmup)
#
# The ONLY difference from the dense/diagonal/block_diagonal runs is the
# arrowhead structure on K.
# ============================================================================

module load cuda/12.6.0
source .venv/bin/activate

TOTAL_DIMS=(64 128 256 512 1024)
TOTAL_DIM="${TOTAL_DIMS[$SLURM_ARRAY_TASK_ID]}"

B=13

D_BASIN=$(( TOTAL_DIM / B ))
D_GLOBAL=$(( TOTAL_DIM - B * D_BASIN ))
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

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lyapunov_k_structure_sweep}"
EXP_NAME="dim${DIM}_nb${NUM_BASINS}_ts${TOTAL_DIM}_arrowhead_no_excl"
LOG_DIR="${BASE_OUT}/${EXP_NAME}"

echo "============================================="
echo "Arrowhead K (NO exclusivity) Sweep"
echo "Job ID: $SLURM_JOB_ID  Array Task: $SLURM_ARRAY_TASK_ID"
echo "DIM: $DIM  NUM_BASINS_GT: $NUM_BASINS"
echo "TOTAL_DIM: $TOTAL_DIM  D_GLOBAL: $D_GLOBAL  D_BASIN: $D_BASIN  B: $B"
echo "Actual total: $(( D_GLOBAL + B * D_BASIN ))"
echo "lambda_exclusivity=0.0, lambda_sparsity=1.0 (matches LISTAKM)"
echo "lambda_global=0.0, lambda_local=0.0 (no structured sparsity)"
echo "excl_warmup_steps=0 (no warmup)"
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
  --lambda_exclusivity 0.0 \
  --lambda_sparsity 1.0 \
  --lambda_global 0.0 \
  --lambda_local 0.0 \
  --excl_warmup_steps 0 \
  --num_steps "$NUM_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --sequence_length 1 \
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
