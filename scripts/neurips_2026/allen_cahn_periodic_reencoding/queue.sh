#!/usr/bin/env bash
# Submit this dormant launcher with sbatch only after independent freeze approval.
#SBATCH --job-name=ac-periodic-queue
#SBATCH --output=/network/scratch/l/lia/skae/ac-periodic-queue-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/ac-periodic-queue-%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
umask 077

PROJECT_DIR="/home/mila/l/lia/skae"
PACKAGE_DIR="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_periodic_reencoding"
SCRIPT_DIR="${PROJECT_DIR}/scripts/neurips_2026/allen_cahn_periodic_reencoding"
CARD_PATH="${PACKAGE_DIR}/prediction_card.json"
SOURCE_MANIFEST="${PACKAGE_DIR}/source_manifest.sha256"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_periodic_reencoding_confirmation_20260721_v1"
SMOKE_RECEIPT="/network/scratch/l/lia/skae/allen_cahn_periodic_reencoding_smoke_20260721_v1/smoke_receipt.json"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?set the independently approved card SHA-256}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?set the independently approved source-manifest SHA-256}"
EXPECTED_SMOKE_RECEIPT_SHA256="${EXPECTED_SMOKE_RECEIPT_SHA256:?set the passing smoke-receipt SHA-256}"
AUTHORIZATION="${SKAE_PERIODIC_REENCODING_AUTHORIZE:-}"
EXPECTED_AUTHORIZATION="YES_I_AUTHORIZE_FROZEN_PERIODIC_REENCODING_V1"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit queue.sh with sbatch; direct shell execution is forbidden." >&2
  exit 1
fi
if [[ "${AUTHORIZATION}" != "${EXPECTED_AUTHORIZATION}" ]]; then
  echo "Periodic-reencoding launch is dormant. Exact authorization is absent." >&2
  exit 1
fi
if [[ ! "${EXPECTED_CARD_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
   [[ ! "${EXPECTED_SOURCE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Authorized hashes must be lowercase SHA-256 digests." >&2
  exit 1
fi
if [[ "$(sha256sum "${CARD_PATH}" | awk '{print $1}')" != "${EXPECTED_CARD_SHA256}" ]] || \
   [[ "$(sha256sum "${SOURCE_MANIFEST}" | awk '{print $1}')" != "${EXPECTED_SOURCE_MANIFEST_SHA256}" ]]; then
  echo "Card or source manifest differs from the independently approved freeze." >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing pre-existing scientific output root: ${OUTPUT_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${SMOKE_RECEIPT}" ]] || \
   [[ "$(sha256sum "${SMOKE_RECEIPT}" | awk '{print $1}')" != "${EXPECTED_SMOKE_RECEIPT_SHA256}" ]] || \
   [[ "$(jq -r '.status' "${SMOKE_RECEIPT}")" != "passed_outcome_free_gpu_smoke" ]] || \
   [[ "$(jq -r '.card_sha256' "${SMOKE_RECEIPT}")" != "${EXPECTED_CARD_SHA256}" ]] || \
   [[ "$(jq -r '.source_manifest_sha256' "${SMOKE_RECEIPT}")" != "${EXPECTED_SOURCE_MANIFEST_SHA256}" ]]; then
  echo "A current hash-bound passing outcome-free smoke receipt is required." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
sha256sum --check --strict --status "${SOURCE_MANIFEST}"
CUDA_VISIBLE_DEVICES="" uv run python -m \
  experiments.neurips_2026.allen_cahn_periodic_reencoding.source_lock \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}"
EXPORTS="ALL,EXPECTED_CARD_SHA256=${EXPECTED_CARD_SHA256},EXPECTED_SOURCE_MANIFEST_SHA256=${EXPECTED_SOURCE_MANIFEST_SHA256},EXPECTED_SMOKE_RECEIPT_SHA256=${EXPECTED_SMOKE_RECEIPT_SHA256},SKAE_PERIODIC_REENCODING_AUTHORIZE=${AUTHORIZATION}"
SCIENTIFIC_JOB_ID="$(sbatch --parsable --export="${EXPORTS}" "${SCRIPT_DIR}/run_generate_select_evaluate.sh")"
SUMMARY_JOB_ID="$(sbatch --parsable --dependency="afterok:${SCIENTIFIC_JOB_ID}" --export="${EXPORTS},EXPECTED_SCIENTIFIC_JOB_ID=${SCIENTIFIC_JOB_ID}" "${SCRIPT_DIR}/run_summary.sh")"

printf 'freeze_card_sha256=%s\nfreeze_source_manifest_sha256=%s\n' \
  "${EXPECTED_CARD_SHA256}" "${EXPECTED_SOURCE_MANIFEST_SHA256}"
printf 'dependency_chain=scientific_generate_select_evaluate_and_telemetry:%s -> dependent_summary:%s\n' \
  "${SCIENTIFIC_JOB_ID}" "${SUMMARY_JOB_ID}"
printf 'scientific_job_id=%s\nsummary_job_id=%s\n' \
  "${SCIENTIFIC_JOB_ID}" "${SUMMARY_JOB_ID}"
