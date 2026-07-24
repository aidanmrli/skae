#!/usr/bin/env bash
#SBATCH --job-name=ac-bridge-data-freeze
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source scripts/common/cluster_env.sh
REPO_ROOT="${ROOT_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SKAE_SCRATCH_ROOT}/allen_cahn_mechanistic_bridge_20260720_v2}"
: "${EXPECTED_CARD_SHA256:?missing EXPECTED_CARD_SHA256}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?missing EXPECTED_SOURCE_MANIFEST_SHA256}"
: "${MECHANISM_DECISION:?missing MECHANISM_DECISION}"
: "${EXPECTED_MECHANISM_DECISION_SHA256:?missing EXPECTED_MECHANISM_DECISION_SHA256}"

cd "${REPO_ROOT}"
uv run python -m experiments.neurips_2026.allen_cahn_mechanistic_bridge.freeze_datasets \
  --output "${OUTPUT_ROOT}/dataset_manifest.json" \
  --decision "${MECHANISM_DECISION}" \
  --expected_decision_sha256 "${EXPECTED_MECHANISM_DECISION_SHA256}" \
  --expected_card_sha256 "${EXPECTED_CARD_SHA256}" \
  --expected_source_manifest_sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}"
