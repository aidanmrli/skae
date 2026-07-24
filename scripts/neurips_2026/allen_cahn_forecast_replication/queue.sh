#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/mila/l/lia/skae"
SCRIPT_DIR="${PROJECT_DIR}/scripts/neurips_2026/allen_cahn_forecast_replication"
CARD_PATH="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_forecast_replication/prediction_card.json"
SOURCE_MANIFEST="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_forecast_replication/source_manifest.sha256"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_forecast_replication_v1_20260720"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?EXPECTED_CARD_SHA256 is required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?EXPECTED_SOURCE_MANIFEST_SHA256 is required}"
AUTHORIZATION="${AUTHORIZE_ALLEN_FORECAST_REPLICATION:?explicit root authorization is required}"
EXPECTED_AUTHORIZATION="root-approved:${EXPECTED_CARD_SHA256}:${EXPECTED_SOURCE_MANIFEST_SHA256}"

if [[ "${AUTHORIZATION}" != "${EXPECTED_AUTHORIZATION}" ]]; then
  echo "Scientific launch authorization token does not match the exact roots." >&2
  exit 1
fi
if [[ "$(sha256sum "${CARD_PATH}" | awk '{print $1}')" != "${EXPECTED_CARD_SHA256}" ]]; then
  echo "Prediction card differs from the root-approved hash." >&2
  exit 1
fi
if [[ "$(sha256sum "${SOURCE_MANIFEST}" | awk '{print $1}')" != "${EXPECTED_SOURCE_MANIFEST_SHA256}" ]]; then
  echo "Source manifest differs from the root-approved hash." >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing pre-existing output root ${OUTPUT_ROOT}." >&2
  exit 1
fi

EXPORTS="ALL,EXPECTED_CARD_SHA256=${EXPECTED_CARD_SHA256},EXPECTED_SOURCE_MANIFEST_SHA256=${EXPECTED_SOURCE_MANIFEST_SHA256},AUTHORIZE_ALLEN_FORECAST_REPLICATION=${AUTHORIZATION}"
GPU_JOB_ID="$(sbatch --parsable --export="${EXPORTS}" "${SCRIPT_DIR}/run_generate_evaluate.sh")"
SUMMARY_JOB_ID="$(sbatch --parsable --dependency="afterok:${GPU_JOB_ID}" --export="${EXPORTS}" "${SCRIPT_DIR}/run_summary.sh")"
printf 'gpu_job_id=%s\nsummary_job_id=%s\n' "${GPU_JOB_ID}" "${SUMMARY_JOB_ID}"
