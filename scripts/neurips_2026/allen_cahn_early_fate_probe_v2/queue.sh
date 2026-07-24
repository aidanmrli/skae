#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/mila/l/lia/skae"
SCRIPT_DIR="${PROJECT_DIR}/scripts/neurips_2026/allen_cahn_early_fate_probe_v2"
PACKAGE_DIR="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_early_fate_probe_v2"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_early_fate_probe_20260721_v2"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?required}"
EXPECTED_TASK_MANIFEST_SHA256="${EXPECTED_TASK_MANIFEST_SHA256:?required}"
AUTHORIZATION="${AUTHORIZE_ALLEN_EARLY_FATE_V2:?required}"
EXPECTED_AUTHORIZATION="root-redteam-approved:${EXPECTED_CARD_SHA256}:${EXPECTED_SOURCE_MANIFEST_SHA256}:${EXPECTED_TASK_MANIFEST_SHA256}"

if [[ "${AUTHORIZATION}" != "${EXPECTED_AUTHORIZATION}" ]]; then
  echo "Independent-redteam launch authorization does not match exact roots." >&2
  exit 1
fi
if [[ "$(sha256sum "${PACKAGE_DIR}/prediction_card.json" | awk '{print $1}')" != "${EXPECTED_CARD_SHA256}" ]] || \
   [[ "$(sha256sum "${PACKAGE_DIR}/source_manifest.sha256" | awk '{print $1}')" != "${EXPECTED_SOURCE_MANIFEST_SHA256}" ]] || \
   [[ "$(sha256sum "${PACKAGE_DIR}/task_manifest.json" | awk '{print $1}')" != "${EXPECTED_TASK_MANIFEST_SHA256}" ]]; then
  echo "One or more V2 launch roots changed after red-team authorization." >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing pre-existing V2 output root." >&2
  exit 1
fi
for seed in 1039834201 1541075401 1816972447; do
  if find /network/scratch/l/lia/skae -maxdepth 5 -name "*${seed}*" -print -quit | grep -q .; then
    echo "Prospective V2 seed already appears in a scratch filename: ${seed}." >&2
    exit 1
  fi
done

EXPORTS="ALL,EXPECTED_CARD_SHA256=${EXPECTED_CARD_SHA256},EXPECTED_SOURCE_MANIFEST_SHA256=${EXPECTED_SOURCE_MANIFEST_SHA256},EXPECTED_TASK_MANIFEST_SHA256=${EXPECTED_TASK_MANIFEST_SHA256},AUTHORIZE_ALLEN_EARLY_FATE_V2=${AUTHORIZATION}"
GPU_JOB_ID="$(sbatch --parsable --export="${EXPORTS}" "${SCRIPT_DIR}/run_generate_extract.sh")"
TELEMETRY_JOB_ID="$(sbatch --parsable --dependency="afterok:${GPU_JOB_ID}" --export="${EXPORTS},EXPECTED_GPU_JOB_ID=${GPU_JOB_ID}" "${SCRIPT_DIR}/run_telemetry.sh")"
CPU_JOB_ID="$(sbatch --parsable --dependency="afterok:${TELEMETRY_JOB_ID}" --export="${EXPORTS}" "${SCRIPT_DIR}/run_summary.sh")"
printf 'gpu_job_id=%s\ntelemetry_job_id=%s\ncpu_job_id=%s\n' \
  "${GPU_JOB_ID}" "${TELEMETRY_JOB_ID}" "${CPU_JOB_ID}"
