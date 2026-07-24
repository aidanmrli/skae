#!/bin/bash
# Build the authenticated compact V2 packet after the complete summary.

#SBATCH --job-name=gkv2_packet
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

uv run python -m experiments.neurips_2026.build_global_k_distinct_laws_v2_packet \
  --card "${CARD_PATH:-experiments/neurips_2026/global_k_distinct_laws_v2_card.json}" \
  --source_lock "${SOURCE_LOCK_PATH:?SOURCE_LOCK_PATH is required}" \
  --expected_source_lock_sha "${EXPECTED_SOURCE_LOCK_SHA:?EXPECTED_SOURCE_LOCK_SHA is required}" \
  --task_tsv "${TASK_TSV:?TASK_TSV is required}" \
  --audit_summary "${AUDIT_SUMMARY:?AUDIT_SUMMARY is required}" \
  --evaluation_dir "${EVALUATION_DIR:?EVALUATION_DIR is required}" \
  --decision "${DECISION_JSON:?DECISION_JSON is required}" \
  --telemetry_assessment "${TELEMETRY_ASSESSMENT:?TELEMETRY_ASSESSMENT is required}" \
  --output "${OUTPUT_JSON:?OUTPUT_JSON is required}"
