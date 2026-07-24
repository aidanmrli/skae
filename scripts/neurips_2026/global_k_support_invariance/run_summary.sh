#!/usr/bin/env bash
#SBATCH --job-name=k-support-sum
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
PROJECT_DIR="${ROOT_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
SUMMARY_DIR="${OUTPUT_ROOT}/summary"

if [[ -e "${SUMMARY_DIR}" ]]; then
  echo "Refusing to overwrite ${SUMMARY_DIR}." >&2
  exit 1
fi
cd "${ROOT_DIR}"
echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
sha256sum \
  experiments/neurips_2026/global_k_support_invariance.py \
  experiments/neurips_2026/global_k_support_invariance_card.json \
  experiments/neurips_2026/summarize_global_k_support_invariance.py

PYTHONPATH="${ROOT_DIR}" uv run --project "${PROJECT_DIR}" python \
  experiments/neurips_2026/summarize_global_k_support_invariance.py \
  --input_dir "${OUTPUT_ROOT}" \
  --output_dir "${SUMMARY_DIR}"

sha256sum "${SUMMARY_DIR}/run_rows.csv" "${SUMMARY_DIR}/system_rows.csv" \
  "${SUMMARY_DIR}/decision.json"
