#!/usr/bin/env bash
#SBATCH --job-name=ac-newic-summary
#SBATCH --output=/network/scratch/l/lia/skae/ac-newic-summary-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/ac-newic-summary-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

PROJECT_DIR="/home/mila/l/lia/skae"
CARD_PATH="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_forecast_replication/prediction_card.json"
SOURCE_MANIFEST="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_forecast_replication/source_manifest.sha256"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_forecast_replication_v1_20260720"
RECEIPT="${OUTPUT_ROOT}/outcome_guard_receipt.json"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?EXPECTED_CARD_SHA256 is required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?EXPECTED_SOURCE_MANIFEST_SHA256 is required}"
AUTHORIZATION="${AUTHORIZE_ALLEN_FORECAST_REPLICATION:?explicit root authorization is required}"
EXPECTED_AUTHORIZATION="root-approved:${EXPECTED_CARD_SHA256}:${EXPECTED_SOURCE_MANIFEST_SHA256}"

if [[ "${AUTHORIZATION}" != "${EXPECTED_AUTHORIZATION}" ]]; then
  echo "Scientific launch authorization token does not match the exact roots." >&2
  exit 1
fi
if [[ ! -f "${RECEIPT}" ]]; then
  echo "The GPU job did not issue an outcome-guard receipt." >&2
  exit 1
fi
EXPECTED_RECEIPT_SHA256="$(sha256sum "${RECEIPT}" | awk '{print $1}')"

cd "${PROJECT_DIR}"
uv run python -m experiments.neurips_2026.allen_cahn_forecast_replication.summarize \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --output-root "${OUTPUT_ROOT}" \
  --outcome-guard-receipt "${RECEIPT}" \
  --expected-outcome-guard-receipt-sha256 "${EXPECTED_RECEIPT_SHA256}"

