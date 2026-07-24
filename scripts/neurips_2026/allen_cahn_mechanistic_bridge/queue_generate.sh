#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source scripts/common/cluster_env.sh
REPO_ROOT="${ROOT_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SKAE_SCRATCH_ROOT}/allen_cahn_mechanistic_bridge_20260720_v2}"
CARD=${REPO_ROOT}/experiments/neurips_2026/allen_cahn_mechanistic_bridge/prediction_card.json
MANIFEST=${REPO_ROOT}/experiments/neurips_2026/allen_cahn_mechanistic_bridge/source_manifest.sha256
MECHANISM_DECISION="${MECHANISM_DECISION:-${SKAE_SCRATCH_ROOT}/allen_cahn_support_subspaces_20260720_v4/summary/decision.json}"
EXPECTED_CARD_SHA256=caa247afdffa666591d6694d6e51286b02b3d80d4083b6e6d71af2bdb9a4fb4b
EXPECTED_SOURCE_MANIFEST_SHA256=8585c4184e6d4a2e5738b47dd9cd2a3a5f0e4b0045b74a72598758af926098d8
EXPECTED_MECHANISM_DECISION_SHA256=__FREEZE_AFTER_MECHANISM__
test "$(sha256sum "${CARD}" | awk '{print $1}')" = "${EXPECTED_CARD_SHA256}"
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
test "$(sha256sum "${MECHANISM_DECISION}" | awk '{print $1}')" = "${EXPECTED_MECHANISM_DECISION_SHA256}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/slurm"
GEN_JOB=$(sbatch --parsable --export=ALL,OUTPUT_ROOT="${OUTPUT_ROOT}",MECHANISM_DECISION="${MECHANISM_DECISION}",EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256}",EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256}",EXPECTED_MECHANISM_DECISION_SHA256="${EXPECTED_MECHANISM_DECISION_SHA256}" "${REPO_ROOT}/scripts/neurips_2026/allen_cahn_mechanistic_bridge/run_generate.sh")
sbatch --parsable --dependency=afterok:"${GEN_JOB}" --export=ALL,OUTPUT_ROOT="${OUTPUT_ROOT}",MECHANISM_DECISION="${MECHANISM_DECISION}",EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256}",EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256}",EXPECTED_MECHANISM_DECISION_SHA256="${EXPECTED_MECHANISM_DECISION_SHA256}" "${REPO_ROOT}/scripts/neurips_2026/allen_cahn_mechanistic_bridge/run_dataset_manifest.sh"
