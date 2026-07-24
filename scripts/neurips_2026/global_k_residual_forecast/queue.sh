#!/usr/bin/env bash
# Submit outcome-free data -> outcome-blind smoke -> ten-seed science -> summary.

#SBATCH --job-name=gkrf-queue
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail
umask 077

PROJECT_DIR="/home/mila/l/lia/skae"
PACKAGE_DIR="${PROJECT_DIR}/experiments/neurips_2026/global_k_residual_forecast"
SCRIPT_DIR="${PROJECT_DIR}/scripts/neurips_2026/global_k_residual_forecast"
OUTPUT_ROOT="/network/scratch/l/lia/skae/global_k_residual_forecast_v3_20260721"
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
  --stage queue \
  --output-root "${OUTPUT_ROOT}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --expected-task-sha256 "${EXPECTED_TASK_SHA256}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing pre-existing scientific output root: ${OUTPUT_ROOT}" >&2
  exit 1
fi
for seed in 310947201 517204603 801336907 901337023 1101337049 314713093; do
  if find /network/scratch/l/lia/skae -maxdepth 6 -name "*${seed}*" -print -quit | grep -q .; then
    echo "A prospective trajectory seed already occurs in a scratch filename: ${seed}" >&2
    exit 1
  fi
done

mkdir --mode=700 "${OUTPUT_ROOT}"
mkdir --mode=700 "${OUTPUT_ROOT}/slurm" "${OUTPUT_ROOT}/summary"
EXPORTS="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},EXPECTED_CARD_SHA256=${EXPECTED_CARD_SHA256},EXPECTED_TASK_SHA256=${EXPECTED_TASK_SHA256},EXPECTED_SOURCE_MANIFEST_SHA256=${EXPECTED_SOURCE_MANIFEST_SHA256},AUTHORIZE_GLOBAL_K_RESIDUAL_FORECAST=${AUTHORIZATION}"
PREP_JOB_ID="$(sbatch --parsable --export="${EXPORTS}" --output="${OUTPUT_ROOT}/slurm/prepare_%j.out" --error="${OUTPUT_ROOT}/slurm/prepare_%j.err" "${SCRIPT_DIR}/run_prepare.sh")"
SMOKE_JOB_ID="$(sbatch --parsable --dependency="afterok:${PREP_JOB_ID}" --export="${EXPORTS},MODE=smoke" --output="${OUTPUT_ROOT}/slurm/smoke_%j.out" --error="${OUTPUT_ROOT}/slurm/smoke_%j.err" "${SCRIPT_DIR}/run_forecast.sh")"
SMOKE_GATE_JOB_ID="$(sbatch --parsable --dependency="afterok:${SMOKE_JOB_ID}" --export="${EXPORTS},MODE=smoke" --output="${OUTPUT_ROOT}/slurm/smoke_gate_%j.out" --error="${OUTPUT_ROOT}/slurm/smoke_gate_%j.err" "${SCRIPT_DIR}/run_telemetry.sh")"
SCIENCE_JOB_ID="$(sbatch --parsable --dependency="afterok:${SMOKE_GATE_JOB_ID}" --array=0-9%10 --export="${EXPORTS},MODE=scientific" --output="${OUTPUT_ROOT}/slurm/science_%A_%a.out" --error="${OUTPUT_ROOT}/slurm/science_%A_%a.err" "${SCRIPT_DIR}/run_forecast.sh")"
SCIENCE_GATE_JOB_ID="$(sbatch --parsable --dependency="afterok:${SCIENCE_JOB_ID}" --export="${EXPORTS},MODE=scientific" --output="${OUTPUT_ROOT}/slurm/science_gate_%j.out" --error="${OUTPUT_ROOT}/slurm/science_gate_%j.err" "${SCRIPT_DIR}/run_telemetry.sh")"
SUMMARY_JOB_ID="$(sbatch --parsable --dependency="afterok:${SCIENCE_GATE_JOB_ID}" --export="${EXPORTS}" --output="${OUTPUT_ROOT}/slurm/summary_%j.out" --error="${OUTPUT_ROOT}/slurm/summary_%j.err" "${SCRIPT_DIR}/run_summary.sh")"

printf 'prepare=%s\nsmoke=%s\nsmoke_gate=%s\nscience=%s\nscience_gate=%s\nsummary=%s\n' \
  "${PREP_JOB_ID}" "${SMOKE_JOB_ID}" "${SMOKE_GATE_JOB_ID}" \
  "${SCIENCE_JOB_ID}" "${SCIENCE_GATE_JOB_ID}" "${SUMMARY_JOB_ID}"
