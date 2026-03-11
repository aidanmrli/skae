#!/bin/bash
#SBATCH --job-name=lfc_v2_collect
#SBATCH --partition=long
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/network/scratch/l/lia/skae/lfc-v2-collect-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/lfc-v2-collect-%j.err

set -euo pipefail

TASK_TSV="${TASK_TSV:?TASK_TSV not set}"
BASE_OUT="${BASE_OUT:?BASE_OUT not set}"
SUMMARY_DIR="${SUMMARY_DIR:?SUMMARY_DIR not set}"

cd /home/mila/l/lia/skae

uv run python tools/collect_label_free_clustering_v2.py \
    --task_tsv "${TASK_TSV}" \
    --base_out "${BASE_OUT}" \
    --output_dir "${SUMMARY_DIR}"
