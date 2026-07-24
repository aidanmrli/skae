#!/usr/bin/env bash
#SBATCH --job-name=ac-fate-summary
#SBATCH --output=/network/scratch/l/lia/skae/ac-fate-summary-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/ac-fate-summary-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

PROJECT_DIR="/home/mila/l/lia/skae"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_early_fate_probe_20260721_v1"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?required}"
EXPECTED_TASK_MANIFEST_SHA256="${EXPECTED_TASK_MANIFEST_SHA256:?required}"
AUTHORIZATION="${AUTHORIZE_ALLEN_EARLY_FATE:?required}"
EXPECTED_AUTHORIZATION="root-approved:${EXPECTED_CARD_SHA256}:${EXPECTED_SOURCE_MANIFEST_SHA256}:${EXPECTED_TASK_MANIFEST_SHA256}"
RECEIPT="${OUTPUT_ROOT}/field_only/telemetry_receipt.json"

if [[ "${AUTHORIZATION}" != "${EXPECTED_AUTHORIZATION}" ]]; then
  echo "Launch authorization does not match exact roots." >&2
  exit 1
fi
if [[ ! -f "${RECEIPT}" ]]; then
  echo "Authenticated GPU receipt is missing." >&2
  exit 1
fi
EXPECTED_RECEIPT_SHA256="$(sha256sum "${RECEIPT}" | awk '{print $1}')"

cd "${PROJECT_DIR}"
uv run python -m experiments.neurips_2026.allen_cahn_early_fate_probe.summarize \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --expected-task-manifest-sha256 "${EXPECTED_TASK_MANIFEST_SHA256}" \
  --output-root "${OUTPUT_ROOT}" \
  --expected-telemetry-receipt-sha256 "${EXPECTED_RECEIPT_SHA256}"
