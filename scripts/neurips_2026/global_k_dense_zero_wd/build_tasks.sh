#!/bin/bash
# Build either the frozen utilization-smoke or full dense zero-WD task table.

#SBATCH --job-name=dense0wd_tasks
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source scripts/common/cluster_env.sh

MODE="${MODE:?MODE must be smoke or full}"
TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
TASK_MANIFEST="${TASK_MANIFEST:?TASK_MANIFEST is required}"
CARD_PATH="${CARD_PATH:-experiments/neurips_2026/global_k_dense_zero_wd_card.json}"

uv run python experiments/neurips_2026/global_k_dense_zero_wd_tasks.py \
  --card "${CARD_PATH}" \
  --mode "${MODE}" \
  --output_tsv "${TASK_TSV}" \
  --output_manifest "${TASK_MANIFEST}"
