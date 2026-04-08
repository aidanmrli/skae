#!/bin/bash
#
# Resolve dt rescue requests for the transition-rich basin-partition sweep and
# emit the next rescue task TSV.
#
# Required env vars:
#   ROWS_CSV=<forecasting_rows.csv>
#   OUT_DIR=<resolution output directory>
#   CURRENT_PASS=<0,1,...>
#
# Optional:
#   MAX_HALVINGS=6
#   THRESHOLD=50
#   MIN_SEEDS=1
#   NEXT_TASK_TSV=<path for next rescue task table>
#   MANIFEST_JSON=<optional manifest snapshot path>
#   SEEDS_CSV=<comma-separated seeds>
#   EVAL_PROFILE=full
#
#SBATCH --job-name=resolve_tr_bp
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o /network/scratch/l/lia/skae/resolve-transition-rich-%A.out
#SBATCH -e /network/scratch/l/lia/skae/resolve-transition-rich-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"

ROWS_CSV="${ROWS_CSV:?ROWS_CSV is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
CURRENT_PASS="${CURRENT_PASS:?CURRENT_PASS is required}"
MAX_HALVINGS="${MAX_HALVINGS:-6}"
THRESHOLD="${THRESHOLD:-50}"
MIN_SEEDS="${MIN_SEEDS:-1}"
NEXT_TASK_TSV="${NEXT_TASK_TSV:-${OUT_DIR}/next_tasks.tsv}"
MANIFEST_JSON="${MANIFEST_JSON:-${OUT_DIR}/transition_rich_manifest.json}"
SEEDS_CSV="${SEEDS_CSV:-}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

echo "============================================="
echo "Resolve Transition-Rich DT"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "ROWS_CSV: ${ROWS_CSV}"
echo "OUT_DIR: ${OUT_DIR}"
echo "CURRENT_PASS: ${CURRENT_PASS}"
echo "MAX_HALVINGS: ${MAX_HALVINGS}"
echo "THRESHOLD: ${THRESHOLD}"
echo "MIN_SEEDS: ${MIN_SEEDS}"
echo "============================================="

uv run python tools/resolve_transition_rich_basin_partition_dt.py \
  --rows_csv "${ROWS_CSV}" \
  --output_dir "${OUT_DIR}" \
  --current_pass "${CURRENT_PASS}" \
  --max_halvings "${MAX_HALVINGS}" \
  --threshold "${THRESHOLD}" \
  --min_seeds "${MIN_SEEDS}"

if (( CURRENT_PASS < MAX_HALVINGS )); then
  REQUEST_TSV="${OUT_DIR}/dt_rescue_request_pass$((CURRENT_PASS + 1)).tsv"
  BUILD_ARGS=(
    --output_tsv "${NEXT_TASK_TSV}"
    --output_manifest_json "${MANIFEST_JSON}"
    --phase_label "rescue_pass$((CURRENT_PASS + 1))"
    --eval_profile "${EVAL_PROFILE}"
    --dt_table "${REQUEST_TSV}"
    --dt_column requested_dt
  )
  if [[ -n "${SEEDS_CSV}" ]]; then
    BUILD_ARGS+=(--seeds_csv "${SEEDS_CSV}")
  fi
  uv run python tools/build_transition_rich_basin_partition_tasks.py "${BUILD_ARGS[@]}"
fi
