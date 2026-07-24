#!/bin/bash
# Evaluate one prospective dense checkpoint against paired sparse cardinalities.

#SBATCH --job-name=dense0wd_spec
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source scripts/common/cluster_env.sh

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
CARD_PATH="${CARD_PATH:-experiments/neurips_2026/global_k_dense_zero_wd_card.json}"
SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH:-experiments/neurips_2026/global_k_dense_specificity_source_lock.json}"
EXPECTED_SOURCE_LOCK_SHA="${EXPECTED_SOURCE_LOCK_SHA:?EXPECTED_SOURCE_LOCK_SHA is required}"

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
  "Evaluator launcher" \
  scripts/neurips_2026/global_k_dense_zero_wd/run_specificity_array.sh \
  "$(jq -er '.sources.evaluator_launcher.sha256' "${SOURCE_LOCK_PATH}")"
assert_sha256 \
  "Dense evaluator" \
  experiments/neurips_2026/global_k_dense_specificity.py \
  "$(jq -er '.sources.dense_evaluator.sha256' "${SOURCE_LOCK_PATH}")"
assert_sha256 \
  "Imported sparse evaluator" \
  experiments/neurips_2026/global_k_support_invariance.py \
  "$(jq -er '.sources.imported_sparse_evaluator.sha256' "${SOURCE_LOCK_PATH}")"
assert_sha256 \
  "Dense task module" \
  experiments/neurips_2026/global_k_dense_zero_wd_tasks.py \
  "$(jq -er '.sources.dense_task_module.sha256' "${SOURCE_LOCK_PATH}")"
assert_sha256 \
  "Frozen card" "${CARD_PATH}" \
  "$(jq -er '.sources.frozen_card.sha256' "${SOURCE_LOCK_PATH}")"
assert_sha256 \
  "Full task table" "${TASK_TSV}" \
  "$(jq -er '.external_inputs.full_task_tsv.sha256' "${SOURCE_LOCK_PATH}")"

echo "Evaluator SHA256: $(sha256sum experiments/neurips_2026/global_k_dense_specificity.py | awk '{print $1}')"
echo "Imported sparse evaluator SHA256: $(sha256sum experiments/neurips_2026/global_k_support_invariance.py | awk '{print $1}')"
echo "Task module SHA256: $(sha256sum experiments/neurips_2026/global_k_dense_zero_wd_tasks.py | awk '{print $1}')"
echo "Card SHA256: $(sha256sum "${CARD_PATH}" | awk '{print $1}')"
echo "Task-table SHA256: $(sha256sum "${TASK_TSV}" | awk '{print $1}')"

SKAE_GIT_COMMIT="$(git rev-parse HEAD)" \
uv run python experiments/neurips_2026/global_k_dense_specificity.py \
  --card "${CARD_PATH}" \
  --source_lock "${SOURCE_LOCK_PATH}" \
  --task_tsv "${TASK_TSV}" \
  --base_out "${BASE_OUT}" \
  --output_dir "${OUTPUT_DIR}" \
  --task_index "${SLURM_ARRAY_TASK_ID}"
