#!/usr/bin/env bash
#SBATCH --job-name=gkrf-summary
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00

set -euo pipefail
umask 077

PROJECT_DIR="/home/mila/l/lia/skae"
PACKAGE_DIR="${PROJECT_DIR}/experiments/neurips_2026/global_k_residual_forecast"
OUTPUT_ROOT="${OUTPUT_ROOT:?required}"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?required}"
EXPECTED_TASK_SHA256="${EXPECTED_TASK_SHA256:?required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?required}"
AUTHORIZATION="${AUTHORIZE_GLOBAL_K_RESIDUAL_FORECAST:?required}"
EXPECTED_AUTHORIZATION="root-redteam-approved:${EXPECTED_CARD_SHA256}:${EXPECTED_SOURCE_MANIFEST_SHA256}:${EXPECTED_TASK_SHA256}"
CARD_PATH="${PACKAGE_DIR}/prediction_card.json"
TASK_PATH="${PACKAGE_DIR}/task_manifest.json"
SOURCE_PATH="${PACKAGE_DIR}/source_manifest.sha256"

if [[ "${AUTHORIZATION}" != "${EXPECTED_AUTHORIZATION}" ]]; then
  echo "Independent-redteam authorization does not bind the exact freeze." >&2
  exit 1
fi
cd "${PROJECT_DIR}"
printf '%s  %s\n' \
  "${EXPECTED_CARD_SHA256}" "${CARD_PATH}" \
  "${EXPECTED_TASK_SHA256}" "${TASK_PATH}" \
  "${EXPECTED_SOURCE_MANIFEST_SHA256}" "${SOURCE_PATH}" \
  | sha256sum --check --strict --status -
sha256sum --check --strict --status "${SOURCE_PATH}"
ROOT_DIR="${PROJECT_DIR}"
source scripts/common/cluster_env.sh
CUDA_VISIBLE_DEVICES="" uv run python -m \
  experiments.neurips_2026.global_k_residual_forecast.preflight \
  --stage summary \
  --output-root "${OUTPUT_ROOT}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --expected-task-sha256 "${EXPECTED_TASK_SHA256}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}"
export CUDA_VISIBLE_DEVICES=""
uv run python -m experiments.neurips_2026.global_k_residual_forecast.summarize \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --expected-task-sha256 "${EXPECTED_TASK_SHA256}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --input-dir "${OUTPUT_ROOT}/scientific/shards" \
  --smoke-assessment "${OUTPUT_ROOT}/smoke/gpu_assessment.json" \
  --scientific-telemetry "${OUTPUT_ROOT}/scientific/gpu_assessment.json" \
  --output "${OUTPUT_ROOT}/summary/decision.json"
