#!/bin/bash
#
# Reduce the paper's fixed controlled basin/support alignment protocol.
#
# Required env vars:
#   ROWS_CSV=<forecasting_rows.csv>
#   OUT_DIR=<output directory>
#
# Optional execution-scope vars:
#   ROOT_LABELS_CSV=<comma-separated paper row identifiers>
#   ROOT_LABELS_FILE=<MODEL_VARIANT=PATH rows; used if CSV is empty>
#   SYSTEMS_CSV=<comma-separated controlled systems>
#   SEEDS_CSV=<comma-separated seeds>
#   DEVICE=cpu
#   PROGRESS_EVERY_RUNS=1
#   FLUSH_EVERY_RUNS=0
#
# Scientific settings are owned by experiments.neurips_2026.alignment and are
# recorded in each reducer manifest; this worker exposes execution scope only.
#
#SBATCH --job-name=tr_align_reduce
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH -o slurm-%x-%A.out
#SBATCH -e slurm-%x-%A.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source .venv/bin/activate

ROWS_CSV="${ROWS_CSV:?ROWS_CSV is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
ROOT_LABELS_CSV="${ROOT_LABELS_CSV:-}"
ROOT_LABELS_FILE="${ROOT_LABELS_FILE:-}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
DEVICE="${DEVICE:-cpu}"
PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS:-1}"
FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS:-0}"

if [[ -z "${ROOT_LABELS_CSV}" ]]; then
  if [[ -n "${ROOT_LABELS_FILE}" ]]; then
    mapfile -t ROOT_LABELS < <(
      awk -F= 'NF>=1 && $1!="" && !seen[$1]++ {print $1}' "${ROOT_LABELS_FILE}"
    )
    ROOT_LABELS_CSV="$(IFS=,; echo "${ROOT_LABELS[*]}")"
  fi
fi

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Rows: ${ROWS_CSV}"
echo "Output: ${OUT_DIR}"
echo "Roots: ${ROOT_LABELS_CSV:-<canonical paper rows>}"
echo "Systems: ${SYSTEMS_CSV:-<all paper systems>}"
echo "Seeds: ${SEEDS_CSV:-<all discovered seeds>}"
echo "Protocol: experiments.neurips_2026.alignment"

ARGS=(
  --rows_csv "${ROWS_CSV}"
  --output_dir "${OUT_DIR}"
  --device "${DEVICE}"
  --progress_every_runs "${PROGRESS_EVERY_RUNS}"
  --flush_every_runs "${FLUSH_EVERY_RUNS}"
)
if [[ -n "${ROOT_LABELS_CSV}" ]]; then
  ARGS+=(--root_labels "${ROOT_LABELS_CSV}")
fi
if [[ -n "${SYSTEMS_CSV}" ]]; then
  ARGS+=(--systems "${SYSTEMS_CSV}")
fi
if [[ -n "${SEEDS_CSV}" ]]; then
  ARGS+=(--seeds "${SEEDS_CSV}")
fi

uv run skae-paper alignment reduce "${ARGS[@]}"
