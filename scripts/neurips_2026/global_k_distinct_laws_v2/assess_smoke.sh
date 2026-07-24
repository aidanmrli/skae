#!/bin/bash
# Assess only frozen lifecycle/GPU gates from the outcome-quarantined smoke.

#SBATCH --job-name=gkv2_smoke_gate
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

CARD_PATH="${CARD_PATH:-experiments/neurips_2026/global_k_distinct_laws_v2_card.json}"
SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH:?SOURCE_LOCK_PATH is required}"
EXPECTED_SOURCE_LOCK_SHA="${EXPECTED_SOURCE_LOCK_SHA:?EXPECTED_SOURCE_LOCK_SHA is required}"
TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
TASK_MANIFEST="${TASK_MANIFEST:?TASK_MANIFEST is required}"
PACK_ROOT="${PACK_ROOT:?PACK_ROOT is required}"
OUTPUT_JSON="${OUTPUT_JSON:?OUTPUT_JSON is required}"

uv run python -m experiments.neurips_2026.assess_global_k_distinct_laws_v2_smoke \
  --card "${CARD_PATH}" \
  --source_lock "${SOURCE_LOCK_PATH}" \
  --expected_source_lock_sha "${EXPECTED_SOURCE_LOCK_SHA}" \
  --task_tsv "${TASK_TSV}" \
  --task_manifest "${TASK_MANIFEST}" \
  --status_dir "${PACK_ROOT}/status" \
  --task_log_dir "${PACK_ROOT}/quarantined_task_logs" \
  --telemetry_csv "${PACK_ROOT}/gpu_telemetry.csv" \
  --pack_timing "${PACK_ROOT}/pack_timing.tsv" \
  --output "${OUTPUT_JSON}"
