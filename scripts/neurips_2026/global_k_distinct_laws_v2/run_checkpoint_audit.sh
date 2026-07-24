#!/bin/bash
# Audit one scientific checkpoint, or summarize all twenty audit shards.

#SBATCH --job-name=gkv2_audit
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source scripts/common/cluster_env.sh

CARD_PATH="${CARD_PATH:-experiments/neurips_2026/global_k_distinct_laws_v2_card.json}"
SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH:?SOURCE_LOCK_PATH is required}"
EXPECTED_SOURCE_LOCK_SHA="${EXPECTED_SOURCE_LOCK_SHA:?EXPECTED_SOURCE_LOCK_SHA is required}"
TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
AUDIT_MODE="${AUDIT_MODE:-shard}"

if [[ "$(sha256sum "${SOURCE_LOCK_PATH}" | awk '{print $1}')" != "${EXPECTED_SOURCE_LOCK_SHA}" ]]; then
  echo "Source-lock hash mismatch." >&2
  exit 3
fi
uv run python -m experiments.neurips_2026.global_k_distinct_laws_v2_source_lock \
  --lock "${SOURCE_LOCK_PATH}"

ARGS=(
  --card "${CARD_PATH}"
  --task_tsv "${TASK_TSV}"
  --base_out "${BASE_OUT}"
  --output_dir "${OUTPUT_DIR}"
)
if [[ "${AUDIT_MODE}" == "summary" ]]; then
  ARGS+=(--summarize)
else
  ARGS+=(--task_index "${SLURM_ARRAY_TASK_ID:?Array task ID is required}")
fi
uv run python -m experiments.neurips_2026.global_k_distinct_laws_v2_checkpoint_audit "${ARGS[@]}"
