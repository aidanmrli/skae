#!/bin/bash
#
# Collect long-horizon Dysts evaluation outputs.
#
# Required env vars:
#   TASK_TSV=<path>
#   OUT_DIR=<path>
#
# Optional env vars:
#   OUTPUT_TAG=dysts_long_horizon_h5000_h10000_h20000_h30000
#   CHECKPOINT_NAME=checkpoint
#   HORIZONS="100 500 1000 1500 2000 3000 4000 5000"
#
#SBATCH --job-name=dysts_long_collect
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

source .venv/bin/activate

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
OUTPUT_TAG="${OUTPUT_TAG:-dysts_long_horizon_h5000_h10000_h20000_h30000}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint}"
HORIZONS="${HORIZONS:-100 500 1000 1500 2000 3000 4000 5000}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-}"
TASK_TSV_SHA256="${TASK_TSV_SHA256:-}"
REQUIRE_COMPLETE="${REQUIRE_COMPLETE:-0}"
EXPECTED_TASK_COUNT="${EXPECTED_TASK_COUNT:-}"

if [[ -n "${SOURCE_MANIFEST}" ]]; then
  sha256sum -c "${SOURCE_MANIFEST}"
fi
if [[ -n "${TASK_TSV_SHA256}" ]]; then
  printf '%s  %s\n' "${TASK_TSV_SHA256}" "${TASK_TSV}" | sha256sum -c -
fi

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Date: $(date)"
echo "TASK_TSV: ${TASK_TSV}"
echo "OUT_DIR: ${OUT_DIR}"

COLLECT_ARGS=(
  --task-tsv "${TASK_TSV}" \
  --out-dir "${OUT_DIR}" \
  --output-tag "${OUTPUT_TAG}" \
  --checkpoint-name "${CHECKPOINT_NAME}" \
  --horizons ${HORIZONS}
)
if [[ "${REQUIRE_COMPLETE}" == "1" ]]; then
  COLLECT_ARGS+=(--require-complete)
fi
if [[ -n "${EXPECTED_TASK_COUNT}" ]]; then
  COLLECT_ARGS+=(--expected-task-count "${EXPECTED_TASK_COUNT}")
fi

uv run skae-paper collect dysts "${COLLECT_ARGS[@]}"
