#!/bin/bash

#SBATCH --job-name=support_thresh
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH -o /network/scratch/l/lia/skae/support_thresh-%A_%a.out
#SBATCH --array=0-6
#SBATCH --requeue

# ============================================================================
# Support Threshold Sweep (post-hoc evaluation)
# ============================================================================
# Usage:
#   sbatch --export=ALL,CKPT=/path/to/checkpoint.pt,OUT_BASE=/path/to/out \
#     scripts/sweep_support_threshold.sh
# Optional:
#   SYSTEM=lyapunov | duffing
#   SUPPORT_MODE=mean | last | median | majority
# ============================================================================

module load cuda/12.6.0
source .venv/bin/activate

THRESHOLDS=(1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1)
THRESH="${THRESHOLDS[$SLURM_ARRAY_TASK_ID]}"

CKPT="${CKPT:-}"
if [ -z "$CKPT" ]; then
  echo "ERROR: CKPT is not set. Provide checkpoint path via --export=ALL,CKPT=..."
  exit 1
fi

OUT_BASE="${OUT_BASE:-/network/scratch/l/lia/skae/support_threshold_sweep}"
SUPPORT_MODE="${SUPPORT_MODE:-mean}"
SYSTEM="${SYSTEM:-}"

RUN_NAME="$(basename "$(dirname "$CKPT")")"
OUT_DIR="${OUT_BASE}/${RUN_NAME}/thresh_${THRESH}"
mkdir -p "$OUT_DIR"

echo "============================================="
echo "Support Threshold Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Checkpoint: $CKPT"
echo "Threshold: $THRESH"
echo "Support Mode: $SUPPORT_MODE"
echo "Output Dir: $OUT_DIR"
echo "============================================="

CMD=(uv run python tools/evaluate_support_uniqueness.py
  --checkpoint "$CKPT"
  --support_threshold "$THRESH"
  --support_mode "$SUPPORT_MODE"
  --output_dir "$OUT_DIR"
  --device cpu
)

if [ -n "$SYSTEM" ]; then
  CMD+=(--system "$SYSTEM")
fi

"${CMD[@]}"

echo "============================================="
echo "Done: thresh=$THRESH"
echo "End Time: $(date)"
echo "============================================="
