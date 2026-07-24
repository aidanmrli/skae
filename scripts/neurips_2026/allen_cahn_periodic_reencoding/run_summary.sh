#!/usr/bin/env bash
#SBATCH --job-name=ac-periodic-summary
#SBATCH --output=/network/scratch/l/lia/skae/ac-periodic-summary-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/ac-periodic-summary-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
umask 077

PROJECT_DIR="/home/mila/l/lia/skae"
PACKAGE_DIR="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_periodic_reencoding"
CARD_PATH="${PACKAGE_DIR}/prediction_card.json"
SOURCE_MANIFEST="${PACKAGE_DIR}/source_manifest.sha256"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_periodic_reencoding_confirmation_20260721_v1"
GUARD_PATH="${OUTPUT_ROOT}/outcome_guard_receipt.json"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?required}"
EXPECTED_SCIENTIFIC_JOB_ID="${EXPECTED_SCIENTIFIC_JOB_ID:?required}"
AUTHORIZATION="${SKAE_PERIODIC_REENCODING_AUTHORIZE:-}"
EXPECTED_AUTHORIZATION="YES_I_AUTHORIZE_FROZEN_PERIODIC_REENCODING_V1"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit this dependent summary with sbatch." >&2
  exit 1
fi
if [[ "${AUTHORIZATION}" != "${EXPECTED_AUTHORIZATION}" ]]; then
  echo "The frozen periodic-reencoding summary remains dormant." >&2
  exit 1
fi
if [[ "$(sha256sum "${CARD_PATH}" | awk '{print $1}')" != "${EXPECTED_CARD_SHA256}" ]] || \
   [[ "$(sha256sum "${SOURCE_MANIFEST}" | awk '{print $1}')" != "${EXPECTED_SOURCE_MANIFEST_SHA256}" ]]; then
  echo "A frozen scientific root changed before summary." >&2
  exit 1
fi
if [[ ! -f "${GUARD_PATH}" ]]; then
  echo "The scientific job did not issue an outcome guard." >&2
  exit 1
fi
if [[ "$(jq -r '.slurm_job_id' "${GUARD_PATH}")" != "${EXPECTED_SCIENTIFIC_JOB_ID}" ]]; then
  echo "Outcome guard is not bound to the queued scientific job." >&2
  exit 1
fi
EXPECTED_GUARD_SHA256="$(sha256sum "${GUARD_PATH}" | awk '{print $1}')"

cd "${PROJECT_DIR}"
sha256sum --check --strict --status "${SOURCE_MANIFEST}"
CUDA_VISIBLE_DEVICES="" uv run python -m \
  experiments.neurips_2026.allen_cahn_periodic_reencoding.summarize \
  --root "${OUTPUT_ROOT}" \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --expected-guard-sha256 "${EXPECTED_GUARD_SHA256}"
