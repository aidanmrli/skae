#!/usr/bin/env bash
#SBATCH --job-name=ac-periodic-v5-summary
#SBATCH --output=/network/scratch/l/lia/skae/ac-periodic-v5-summary-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/ac-periodic-v5-summary-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
umask 077

PROJECT_DIR="/home/mila/l/lia/skae"
PACKAGE_DIR="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_periodic_reencoding_v5"
CARD_PATH="${PACKAGE_DIR}/prediction_card.json"
SOURCE_MANIFEST="${PACKAGE_DIR}/source_manifest.sha256"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_periodic_reencoding_confirmation_20260721_v5"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?required}"
EXPECTED_SCIENTIFIC_JOB_ID="${EXPECTED_SCIENTIFIC_JOB_ID:?required}"
AUTHORIZATION="${SKAE_PERIODIC_REENCODING_V5_AUTHORIZE:-}"
EXPECTED_AUTHORIZATION="YES_I_AUTHORIZE_FROZEN_PERIODIC_REENCODING_V5"

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
cd "${PROJECT_DIR}"
sha256sum --check --strict --status "${SOURCE_MANIFEST}"
EXPECTED_GUARD_SHA256="$(CUDA_VISIBLE_DEVICES="" uv run python -m \
  experiments.neurips_2026.allen_cahn_periodic_reencoding_v5.guard outcome \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --output-root "${OUTPUT_ROOT}" \
  --expected-scientific-job-id "${EXPECTED_SCIENTIFIC_JOB_ID}")"
if [[ ! "${EXPECTED_GUARD_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Strict outcome guard did not emit exactly one SHA-256 digest." >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="" uv run python -m \
  experiments.neurips_2026.allen_cahn_periodic_reencoding.summarize \
  --root "${OUTPUT_ROOT}" \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --expected-guard-sha256 "${EXPECTED_GUARD_SHA256}"
