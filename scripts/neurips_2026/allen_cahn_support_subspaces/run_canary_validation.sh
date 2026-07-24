#!/usr/bin/env bash
#SBATCH --job-name=ac-support-canary-check
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/network/scratch/l/lia/skae/allen_cahn_support_subspaces_20260720_v4/slurm/canary_check_%j.out
#SBATCH --error=/network/scratch/l/lia/skae/allen_cahn_support_subspaces_20260720_v4/slurm/canary_check_%j.err

set -euo pipefail

REPO_ROOT=/home/mila/l/lia/skae
OUTPUT_ROOT=${OUTPUT_ROOT:-/network/scratch/l/lia/skae/allen_cahn_support_subspaces_20260720_v4}
: "${EXPECTED_CARD_SHA256:?missing EXPECTED_CARD_SHA256}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?missing EXPECTED_SOURCE_MANIFEST_SHA256}"
: "${EXPECTED_PROFILE_DECISION_SHA256:?missing EXPECTED_PROFILE_DECISION_SHA256}"
cd "${REPO_ROOT}"
uv run python -m experiments.neurips_2026.allen_cahn_support_subspaces.validate_canary \
  --output_root "${OUTPUT_ROOT}" \
  --expected_card_sha256 "${EXPECTED_CARD_SHA256}" \
  --expected_source_manifest_sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --expected_profile_decision_sha256 "${EXPECTED_PROFILE_DECISION_SHA256}"
