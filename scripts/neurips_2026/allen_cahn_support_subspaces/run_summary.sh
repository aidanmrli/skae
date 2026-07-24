#!/usr/bin/env bash
#SBATCH --job-name=ac-support-summary
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/network/scratch/l/lia/skae/allen_cahn_support_subspaces_20260720_v4/slurm/summary_%j.out
#SBATCH --error=/network/scratch/l/lia/skae/allen_cahn_support_subspaces_20260720_v4/slurm/summary_%j.err

set -euo pipefail

REPO_ROOT=/home/mila/l/lia/skae
OUTPUT_ROOT=${OUTPUT_ROOT:-/network/scratch/l/lia/skae/allen_cahn_support_subspaces_20260720_v4}
: "${EXPECTED_CARD_SHA256:?missing EXPECTED_CARD_SHA256}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?missing EXPECTED_SOURCE_MANIFEST_SHA256}"
: "${EXPECTED_PROFILE_DECISION_SHA256:?missing EXPECTED_PROFILE_DECISION_SHA256}"
MANIFEST=${REPO_ROOT}/experiments/neurips_2026/allen_cahn_support_subspaces/source_manifest.sha256
CARD=${REPO_ROOT}/experiments/neurips_2026/allen_cahn_support_subspaces/prediction_card.json
PROFILE=${OUTPUT_ROOT}/profile/decision.json
test "$(sha256sum "${CARD}" | awk '{print $1}')" = "${EXPECTED_CARD_SHA256}"
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
test "$(sha256sum "${PROFILE}" | awk '{print $1}')" = "${EXPECTED_PROFILE_DECISION_SHA256}"
cd "${REPO_ROOT}"
BATCH_SIZE=$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_batch_size"])' "${PROFILE}")
uv run python -m experiments.neurips_2026.allen_cahn_support_subspaces.summarize \
  --input_root "${OUTPUT_ROOT}" --output_dir "${OUTPUT_ROOT}/summary" \
  --profile_decision "${PROFILE}" --telemetry_dir "${OUTPUT_ROOT}/telemetry" \
  --batch_size "${BATCH_SIZE}" --expected_card_sha256 "${EXPECTED_CARD_SHA256}" \
  --expected_source_manifest_sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --expected_profile_decision_sha256 "${EXPECTED_PROFILE_DECISION_SHA256}"
uv run python -m experiments.neurips_2026.allen_cahn_support_subspaces.build_evidence build \
  --result_dir "${OUTPUT_ROOT}/summary" --evidence_dir "${OUTPUT_ROOT}/evidence"
uv run python -m experiments.neurips_2026.allen_cahn_support_subspaces.build_evidence check \
  --evidence_dir "${OUTPUT_ROOT}/evidence"
