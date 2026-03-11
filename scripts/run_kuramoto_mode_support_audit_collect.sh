#!/bin/bash
#SBATCH --job-name=kuramoto_mode_collect
#SBATCH --partition=long
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/network/scratch/l/lia/skae/kuramoto-mode-collect-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/kuramoto-mode-collect-%j.err

set -euo pipefail

TASK_TSV="${TASK_TSV:?TASK_TSV not set}"
SUMMARY_DIR="${SUMMARY_DIR:?SUMMARY_DIR not set}"

cd /home/mila/l/lia/skae

uv run python tools/collect_kuramoto_mode_support_audit.py \
  --task_tsv "${TASK_TSV}" \
  --summary_dir "${SUMMARY_DIR}"
