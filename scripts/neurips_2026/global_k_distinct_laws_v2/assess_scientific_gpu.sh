#!/bin/bash
# Authenticate outcome-blind scientific lifecycle and GPU telemetry before audit.

#SBATCH --job-name=gkv2_gpu_audit
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

uv run python -m experiments.neurips_2026.assess_global_k_distinct_laws_v2_scientific_gpu \
  --card "${CARD_PATH:-experiments/neurips_2026/global_k_distinct_laws_v2_card.json}" \
  --source_lock "${SOURCE_LOCK_PATH:?SOURCE_LOCK_PATH is required}" \
  --expected_source_lock_sha "${EXPECTED_SOURCE_LOCK_SHA:?EXPECTED_SOURCE_LOCK_SHA is required}" \
  --task_tsv "${TASK_TSV:?TASK_TSV is required}" \
  --task_manifest "${TASK_MANIFEST:?TASK_MANIFEST is required}" \
  --pack_root "${PACK_ROOT:?PACK_ROOT is required}" \
  --output "${OUTPUT_JSON:?OUTPUT_JSON is required}"
