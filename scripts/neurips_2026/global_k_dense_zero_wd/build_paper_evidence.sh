#!/bin/bash
# Normalize and verify the completed dense-specificity packet for the paper.

#SBATCH --job-name=dense0wd_evid
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source scripts/common/cluster_env.sh

RESULT_ROOT="${RESULT_ROOT:?RESULT_ROOT is required}"
SMOKE_DECISION="${SMOKE_DECISION:?SMOKE_DECISION is required}"
SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH:-experiments/neurips_2026/global_k_dense_specificity_source_lock.json}"
EXPECTED_SOURCE_LOCK_SHA="${EXPECTED_SOURCE_LOCK_SHA:?EXPECTED_SOURCE_LOCK_SHA is required}"
BUILDER="experiments/neurips_2026/evidence/global_k_dense_specificity.py"
CARD="experiments/neurips_2026/global_k_dense_zero_wd_card.json"
TASK_MANIFEST="${RESULT_ROOT}/tasks/full_manifest.json"
DATA_DIR="docs/figures/neurips_paper_2026/_data"
TABLE="docs/figures/neurips_paper_2026/_tables/table_global_k_dense_zero_wd_specificity.tex"

assert_sha256() {
  local label="$1"
  local path="$2"
  local expected="$3"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "${label} SHA256 drift: expected=${expected} actual=${actual} path=${path}" >&2
    exit 3
  fi
}

assert_sha256 "Source lock" "${SOURCE_LOCK_PATH}" "${EXPECTED_SOURCE_LOCK_SHA}"
if [[ "$(jq -r '.protocol_id // empty' "${SOURCE_LOCK_PATH}")" != "global_k_dense_zero_wd_specificity_v1" ]]; then
  echo "Unexpected dense-specificity source-lock protocol ID." >&2
  exit 3
fi
while IFS=$'\t' read -r source_name source_path source_sha; do
  assert_sha256 "Locked source ${source_name}" "${source_path}" "${source_sha}"
done < <(
  jq -er '.sources | to_entries[] | [.key, .value.path, .value.sha256] | @tsv' \
    "${SOURCE_LOCK_PATH}"
)
assert_sha256 \
  "Evidence launcher" \
  scripts/neurips_2026/global_k_dense_zero_wd/build_paper_evidence.sh \
  "$(jq -er '.sources.evidence_launcher.sha256' "${SOURCE_LOCK_PATH}")"
assert_sha256 \
  "Evidence builder" "${BUILDER}" \
  "$(jq -er '.sources.evidence_builder.sha256' "${SOURCE_LOCK_PATH}")"
assert_sha256 \
  "Frozen card" "${CARD}" \
  "$(jq -er '.sources.frozen_card.sha256' "${SOURCE_LOCK_PATH}")"
assert_sha256 \
  "Task manifest" "${TASK_MANIFEST}" \
  "$(jq -er '.external_inputs.full_task_manifest.sha256' "${SOURCE_LOCK_PATH}")"
assert_sha256 \
  "Smoke decision" "${SMOKE_DECISION}" \
  "$(jq -er '.external_inputs.smoke_decision.sha256' "${SOURCE_LOCK_PATH}")"

uv run python "${BUILDER}" build \
  --card "${CARD}" \
  --source_lock "${SOURCE_LOCK_PATH}" \
  --task_manifest "${TASK_MANIFEST}" \
  --smoke_decision "${SMOKE_DECISION}" \
  --evaluation_dir "${RESULT_ROOT}/specificity/evaluation" \
  --summary_dir "${RESULT_ROOT}/specificity/summary" \
  --output_data_dir "${DATA_DIR}" \
  --output_table "${TABLE}"

uv run python "${BUILDER}" check \
  --provenance "${DATA_DIR}/global_k_dense_zero_wd_specificity_provenance.json"

sha256sum \
  "${DATA_DIR}"/global_k_dense_zero_wd_specificity_* \
  "${TABLE}"
