#!/usr/bin/env bash

#SBATCH --job-name=k_law_summary
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
cd "${ROOT_DIR}"

echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "gpu_count=0"
sha256sum \
  experiments/neurips_2026/global_k_distinct_laws_card.json \
  experiments/neurips_2026/global_k_distinct_laws.py \
  experiments/neurips_2026/summarize_global_k_distinct_laws.py

env PYTHONPATH="${ROOT_DIR}" uv run python \
  experiments/neurips_2026/summarize_global_k_distinct_laws.py \
  --output_dir "${OUTPUT_DIR}"
