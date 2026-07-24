#!/usr/bin/env bash
#SBATCH --job-name=gkrf-prepare
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
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
  --stage prepare \
  --output-root "${OUTPUT_ROOT}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --expected-task-sha256 "${EXPECTED_TASK_SHA256}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}"
export CUDA_VISIBLE_DEVICES=""
uv run python -m experiments.neurips_2026.global_k_residual_forecast.prepare_data \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --expected-task-sha256 "${EXPECTED_TASK_SHA256}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --output-root "${OUTPUT_ROOT}"
