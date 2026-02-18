#!/bin/bash
#
#SBATCH --job-name=backfill_dysts_eval_p1
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/backfill-dysts-fcast-p1-%A.out

set -euo pipefail

source .venv/bin/activate

PHASE1_ROOT="${PHASE1_ROOT:-/network/scratch/l/lia/skae/dysts_forecast_phase1_lista_relu_sp15_ts256}"

echo "============================================="
echo "Backfill Dysts Forecast Phase 1 Evaluations"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "PHASE1_ROOT: $PHASE1_ROOT"
echo "============================================="

if [ ! -d "$PHASE1_ROOT" ]; then
  echo "Phase-1 root not found: $PHASE1_ROOT"
  exit 1
fi

TOTAL=0
DONE=0
SKIP=0
FAIL=0

while IFS= read -r RUN_DIR; do
  TOTAL=$((TOTAL + 1))

  if [ ! -f "$RUN_DIR/checkpoint.pt" ] || [ ! -f "$RUN_DIR/final_metrics.json" ]; then
    echo "[skip] incomplete run dir: $RUN_DIR"
    SKIP=$((SKIP + 1))
    continue
  fi

  if [ -f "$RUN_DIR/evaluation_results_best.json" ]; then
    echo "[skip] already evaluated: $RUN_DIR"
    SKIP=$((SKIP + 1))
    continue
  fi

  echo "[eval] $RUN_DIR"
  if uv run python tools/evaluate_checkpoints.py \
    --run_dir "$RUN_DIR" \
    --checkpoints checkpoint.pt \
    --device cuda; then
    DONE=$((DONE + 1))
  else
    echo "[fail] evaluation failed: $RUN_DIR"
    FAIL=$((FAIL + 1))
  fi
done < <(find "$PHASE1_ROOT" -mindepth 2 -maxdepth 2 -type d | sort)

echo "============================================="
echo "End Time: $(date)"
echo "Scanned: $TOTAL"
echo "Evaluated: $DONE"
echo "Skipped: $SKIP"
echo "Failed: $FAIL"
echo "============================================="

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
