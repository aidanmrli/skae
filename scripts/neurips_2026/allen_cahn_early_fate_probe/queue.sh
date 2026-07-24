#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/mila/l/lia/skae"
SCRIPT_DIR="${PROJECT_DIR}/scripts/neurips_2026/allen_cahn_early_fate_probe"
CARD_PATH="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_early_fate_probe/prediction_card.json"
SOURCE_MANIFEST="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_early_fate_probe/source_manifest.sha256"
TASK_MANIFEST="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_early_fate_probe/task_manifest.json"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_early_fate_probe_20260721_v1"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?required}"
EXPECTED_TASK_MANIFEST_SHA256="${EXPECTED_TASK_MANIFEST_SHA256:?required}"
AUTHORIZATION="${AUTHORIZE_ALLEN_EARLY_FATE:?required}"
EXPECTED_AUTHORIZATION="root-approved:${EXPECTED_CARD_SHA256}:${EXPECTED_SOURCE_MANIFEST_SHA256}:${EXPECTED_TASK_MANIFEST_SHA256}"

if [[ "${AUTHORIZATION}" != "${EXPECTED_AUTHORIZATION}" ]]; then
  echo "Launch authorization does not match exact roots." >&2
  exit 1
fi
if [[ "$(sha256sum "${CARD_PATH}" | awk '{print $1}')" != "${EXPECTED_CARD_SHA256}" ]] || \
   [[ "$(sha256sum "${SOURCE_MANIFEST}" | awk '{print $1}')" != "${EXPECTED_SOURCE_MANIFEST_SHA256}" ]] || \
   [[ "$(sha256sum "${TASK_MANIFEST}" | awk '{print $1}')" != "${EXPECTED_TASK_MANIFEST_SHA256}" ]]; then
  echo "One or more launch roots changed after authorization." >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing pre-existing output root ${OUTPUT_ROOT}." >&2
  exit 1
fi

EXPORTS="ALL,EXPECTED_CARD_SHA256=${EXPECTED_CARD_SHA256},EXPECTED_SOURCE_MANIFEST_SHA256=${EXPECTED_SOURCE_MANIFEST_SHA256},EXPECTED_TASK_MANIFEST_SHA256=${EXPECTED_TASK_MANIFEST_SHA256},AUTHORIZE_ALLEN_EARLY_FATE=${AUTHORIZATION}"
PROFILE_JOB_ID="$(sbatch --parsable --export="${EXPORTS}" "${SCRIPT_DIR}/run_profile.sh")"
EXTRACT_JOB_ID="$(sbatch --parsable --dependency="afterok:${PROFILE_JOB_ID}" --export="${EXPORTS}" "${SCRIPT_DIR}/run_extract.sh")"
SUMMARY_JOB_ID="$(sbatch --parsable --dependency="afterok:${EXTRACT_JOB_ID}" --export="${EXPORTS}" "${SCRIPT_DIR}/run_summary.sh")"
printf 'profile_job_id=%s\nextract_job_id=%s\nsummary_job_id=%s\n' \
  "${PROFILE_JOB_ID}" "${EXTRACT_JOB_ID}" "${SUMMARY_JOB_ID}"
