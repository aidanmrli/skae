#!/bin/bash
#
# Collect forecasting artifacts for the controlled paper benchmark.
#
# Required env vars:
#   ROOT_SPECS_FILE=<text file with MODEL_VARIANT=PATH per line>
#   OUT_DIR=<output directory>
#
# Optional:
#   HORIZONS_CSV=100,500,1000
#   GOOD_THRESHOLD=50
#
#SBATCH --job-name=collect_tr_bp
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o slurm-%x-%A.out
#SBATCH -e slurm-%x-%A.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

source .venv/bin/activate

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"

ROOT_SPECS_FILE="${ROOT_SPECS_FILE:?ROOT_SPECS_FILE is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
HORIZONS_CSV="${HORIZONS_CSV:-100,500,1000}"
GOOD_THRESHOLD="${GOOD_THRESHOLD:-50}"

mapfile -t ROOT_SPECS < "${ROOT_SPECS_FILE}"
IFS=',' read -r -a HORIZONS <<< "${HORIZONS_CSV}"

echo "============================================="
echo "Collect Transition-Rich Basin Partition"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "ROOT_SPECS_FILE: ${ROOT_SPECS_FILE}"
echo "OUT_DIR: ${OUT_DIR}"
echo "HORIZONS: ${HORIZONS_CSV}"
echo "GOOD_THRESHOLD: ${GOOD_THRESHOLD}"
echo "============================================="

uv run skae-paper collect controlled \
  --run_roots "${ROOT_SPECS[@]}" \
  --output_dir "${OUT_DIR}" \
  --horizons "${HORIZONS[@]}" \
  --good_threshold "${GOOD_THRESHOLD}" \
  --essential_factor 10 \
  --select latest
