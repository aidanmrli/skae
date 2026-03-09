#!/bin/bash
#
# Resolve per-system dt settings for the research-paper benchmark and emit the
# next task table.
#
# Required env vars:
#   ROWS_CSV=<forecasting_rows.csv>
#   OUT_DIR=<resolution output directory>
#   CURRENT_PASS=<0|1|2>
#
# Optional:
#   MAX_HALVINGS=2
#   THRESHOLD=1.0
#   NEXT_TASK_TSV=<path for rescue/full task table>
#   MANIFEST_JSON=<optional manifest snapshot path>
#
#SBATCH --job-name=resolve_paper
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o /network/scratch/l/lia/skae/resolve-paper-%A.out
#SBATCH -e /network/scratch/l/lia/skae/resolve-paper-%A.err

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
MAX_HALVINGS="${MAX_HALVINGS:-2}"
THRESHOLD="${THRESHOLD:-1.0}"
NEXT_TASK_TSV="${NEXT_TASK_TSV:-${OUT_DIR}/next_tasks.tsv}"
MANIFEST_JSON="${MANIFEST_JSON:-${OUT_DIR}/paper_manifest.json}"

echo "============================================="
echo "Resolve Paper Benchmark DT"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "ROWS_CSV: ${ROWS_CSV}"
echo "OUT_DIR: ${OUT_DIR}"
echo "CURRENT_PASS: ${CURRENT_PASS}"
echo "MAX_HALVINGS: ${MAX_HALVINGS}"
echo "THRESHOLD: ${THRESHOLD}"
echo "============================================="

uv run python tools/resolve_paper_benchmark_dt.py \
  --rows_csv "${ROWS_CSV}" \
  --output_dir "${OUT_DIR}" \
  --current_pass "${CURRENT_PASS}" \
  --max_halvings "${MAX_HALVINGS}" \
  --threshold "${THRESHOLD}"

if (( CURRENT_PASS < MAX_HALVINGS )); then
  REQUEST_TSV="${OUT_DIR}/dt_rescue_request_pass$((CURRENT_PASS + 1)).tsv"
  uv run python tools/build_paper_benchmark_tasks.py \
    --phase rescue \
    --phase_label "rescue_pass$((CURRENT_PASS + 1))" \
    --dt_table "${REQUEST_TSV}" \
    --output_tsv "${NEXT_TASK_TSV}" \
    --output_manifest_json "${MANIFEST_JSON}"
else
  SELECTED_TSV="${OUT_DIR}/selected_dt.tsv"
  uv run python tools/build_paper_benchmark_tasks.py \
    --phase full \
    --dt_table "${SELECTED_TSV}" \
    --output_tsv "${NEXT_TASK_TSV}" \
    --output_manifest_json "${MANIFEST_JSON}"
fi
