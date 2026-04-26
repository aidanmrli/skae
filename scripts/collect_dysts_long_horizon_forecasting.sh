#!/bin/bash
#
# Collect long-horizon Dysts reevaluation outputs.
#
# Required env vars:
#   TASK_TSV=<path>
#   OUT_DIR=<path>
#
# Optional env vars:
#   OUTPUT_TAG=dysts_long_horizon_h5000_h10000_h20000_h30000
#   CHECKPOINT_NAME=checkpoint
#
#SBATCH --job-name=dysts_long_collect
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH -o /network/scratch/l/lia/skae/dysts-long-collect-%j.out
#SBATCH -e /network/scratch/l/lia/skae/dysts-long-collect-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
OUTPUT_TAG="${OUTPUT_TAG:-dysts_long_horizon_h5000_h10000_h20000_h30000}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint}"

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Date: $(date)"
echo "TASK_TSV: ${TASK_TSV}"
echo "OUT_DIR: ${OUT_DIR}"

uv run python tools/collect_dysts_long_horizon_forecasting.py \
  --task-tsv "${TASK_TSV}" \
  --out-dir "${OUT_DIR}" \
  --output-tag "${OUTPUT_TAG}" \
  --checkpoint-name "${CHECKPOINT_NAME}"
