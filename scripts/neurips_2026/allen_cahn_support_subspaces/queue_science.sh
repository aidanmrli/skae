#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source scripts/common/cluster_env.sh
REPO_ROOT="${ROOT_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SKAE_SCRATCH_ROOT}/allen_cahn_support_subspaces_20260720_v4}"
EXPECTED_CARD_SHA256=fafa3b1a0e8f63095c3926171673fa62f2baec6e2af36a954cbca83d35f35743
EXPECTED_SOURCE_MANIFEST_SHA256=e4219ecb3b2e25d08f9f1e5afc51a16f84d94409baf62280651cc101fc3f7024
EXPECTED_PROFILE_DECISION_SHA256=043ee246bdfcc8d4ef50431d234274404bfd2438114c8755d513a62f5a04b993
PROFILE=${OUTPUT_ROOT}/profile/decision.json
test "$(sha256sum "${REPO_ROOT}/experiments/neurips_2026/allen_cahn_support_subspaces/prediction_card.json" | awk '{print $1}')" = "${EXPECTED_CARD_SHA256}"
test "$(sha256sum "${REPO_ROOT}/experiments/neurips_2026/allen_cahn_support_subspaces/source_manifest.sha256" | awk '{print $1}')" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
test "$(sha256sum "${PROFILE}" | awk '{print $1}')" = "${EXPECTED_PROFILE_DECISION_SHA256}"
CANARY_JOB=$(sbatch --parsable --array=0 --export=ALL,OUTPUT_ROOT="${OUTPUT_ROOT}",EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256}",EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256}",EXPECTED_PROFILE_DECISION_SHA256="${EXPECTED_PROFILE_DECISION_SHA256}" \
  "${REPO_ROOT}/scripts/neurips_2026/allen_cahn_support_subspaces/run_array.sh")
CANARY_CHECK_JOB=$(sbatch --parsable --dependency=afterok:"${CANARY_JOB}" --export=ALL,OUTPUT_ROOT="${OUTPUT_ROOT}",EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256}",EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256}",EXPECTED_PROFILE_DECISION_SHA256="${EXPECTED_PROFILE_DECISION_SHA256}" \
  "${REPO_ROOT}/scripts/neurips_2026/allen_cahn_support_subspaces/run_canary_validation.sh")
REST_JOB=$(sbatch --parsable --array=1-9%8 --dependency=afterok:"${CANARY_CHECK_JOB}" --export=ALL,OUTPUT_ROOT="${OUTPUT_ROOT}",EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256}",EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256}",EXPECTED_PROFILE_DECISION_SHA256="${EXPECTED_PROFILE_DECISION_SHA256}" \
  "${REPO_ROOT}/scripts/neurips_2026/allen_cahn_support_subspaces/run_array.sh")
SUMMARY_JOB=$(sbatch --parsable --dependency=afterok:"${REST_JOB}" --export=ALL,OUTPUT_ROOT="${OUTPUT_ROOT}",EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256}",EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256}",EXPECTED_PROFILE_DECISION_SHA256="${EXPECTED_PROFILE_DECISION_SHA256}" \
  "${REPO_ROOT}/scripts/neurips_2026/allen_cahn_support_subspaces/run_summary.sh")
printf 'canary_job=%s canary_check_job=%s rest_job=%s summary_job=%s\n' \
  "${CANARY_JOB}" "${CANARY_CHECK_JOB}" "${REST_JOB}" "${SUMMARY_JOB}"
