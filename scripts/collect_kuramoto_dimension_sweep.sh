#!/bin/bash
#
# Collect forecasting artifacts for the Kuramoto dimension scaling sweep.
#
# Required env vars:
#   ROOT_SPECS_FILE=<text file with LABEL=PATH per line>
#   OUT_DIR=<output directory>
#
# Optional:
#   HORIZONS_CSV=100,500,1000
#
#SBATCH --job-name=collect_kura_dim
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-kura-dim-%A.out
#SBATCH -e /network/scratch/l/lia/skae/collect-kura-dim-%A.err

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

ROOT_SPECS_FILE="${ROOT_SPECS_FILE:?ROOT_SPECS_FILE is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
HORIZONS_CSV="${HORIZONS_CSV:-100,500,1000}"

mapfile -t ROOT_SPECS < "${ROOT_SPECS_FILE}"
IFS=',' read -r -a HORIZONS <<< "${HORIZONS_CSV}"

echo "============================================="
echo "Collect Kuramoto Dimension Sweep"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "ROOT_SPECS_FILE: ${ROOT_SPECS_FILE}"
echo "OUT_DIR: ${OUT_DIR}"
echo "HORIZONS: ${HORIZONS_CSV}"
echo "============================================="

uv run python tools/collect_forecasting_roots.py \
  --run_roots "${ROOT_SPECS[@]}" \
  --output_dir "${OUT_DIR}" \
  --horizons "${HORIZONS[@]}" \
  --good_threshold 10 \
  --essential_factor 10 \
  --select latest

uv run python tools/summarize_kuramoto_dimension_sweep.py \
  --rows_csv "${OUT_DIR}/forecasting_rows.csv" \
  --output_dir "${OUT_DIR}" \
  --horizons "${HORIZONS[@]}"
