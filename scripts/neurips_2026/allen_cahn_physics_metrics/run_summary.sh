#!/usr/bin/env bash
#SBATCH --job-name=ac-physics-sum
#SBATCH --output=/network/scratch/l/lia/skae/ac-physics-sum-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/ac-physics-sum-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

PROJECT_DIR="/home/mila/l/lia/skae"
CARD_PATH="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_physics_metrics/prediction_card.json"
SOURCE_MANIFEST="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_physics_metrics/source_manifest.sha256"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_physics_metrics_v1_20260721"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?required}"
EXPECTED_OUTCOME_RECEIPT_SHA256="${EXPECTED_OUTCOME_RECEIPT_SHA256:?required after independent telemetry audit}"

cd "${PROJECT_DIR}"
uv run python -m experiments.neurips_2026.allen_cahn_physics_metrics.summarize \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --outcome-receipt "${OUTPUT_ROOT}/outcome_guard_receipt.json" \
  --expected-outcome-receipt-sha256 "${EXPECTED_OUTCOME_RECEIPT_SHA256}" \
  --output-root "${OUTPUT_ROOT}"
