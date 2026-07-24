#!/bin/bash
# Summarize all twenty H/G evaluation shards only after array completion.

#SBATCH --job-name=gkv2_summary
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source scripts/common/cluster_env.sh

CARD_PATH="${CARD_PATH:-experiments/neurips_2026/global_k_distinct_laws_v2_card.json}"
SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH:?SOURCE_LOCK_PATH is required}"
EXPECTED_SOURCE_LOCK_SHA="${EXPECTED_SOURCE_LOCK_SHA:?EXPECTED_SOURCE_LOCK_SHA is required}"
TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
AUDIT_DIR="${AUDIT_DIR:?AUDIT_DIR is required}"
INPUT_DIR="${INPUT_DIR:?INPUT_DIR is required}"
OUTPUT_JSON="${OUTPUT_JSON:?OUTPUT_JSON is required}"

uv run python -m experiments.neurips_2026.summarize_global_k_distinct_laws_v2 \
  --card "${CARD_PATH}" \
  --source_lock "${SOURCE_LOCK_PATH}" \
  --expected_source_lock_sha "${EXPECTED_SOURCE_LOCK_SHA}" \
  --task_tsv "${TASK_TSV}" \
  --audit_dir "${AUDIT_DIR}" \
  --input_dir "${INPUT_DIR}" \
  --output "${OUTPUT_JSON}"
