#!/bin/bash
# Aggregate the matched dense specificity shards without pseudoreplication.

#SBATCH --job-name=dense0wd_sum
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

INPUT_DIR="${INPUT_DIR:?INPUT_DIR is required}"
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
  "Summary launcher" \
  scripts/neurips_2026/global_k_dense_zero_wd/run_specificity_summary.sh \
  "$(jq -er '.sources.summary_launcher.sha256' "${SOURCE_LOCK_PATH}")"
assert_sha256 \
  "Dense specificity summarizer" \
  experiments/neurips_2026/summarize_global_k_dense_specificity.py \
  "$(jq -er '.sources.dense_summarizer.sha256' "${SOURCE_LOCK_PATH}")"
assert_sha256 \
  "Dense task module" \
  experiments/neurips_2026/global_k_dense_zero_wd_tasks.py \
  "$(jq -er '.sources.dense_task_module.sha256' "${SOURCE_LOCK_PATH}")"
assert_sha256 \
  "Frozen card" "${CARD_PATH}" \
  "$(jq -er '.sources.frozen_card.sha256' "${SOURCE_LOCK_PATH}")"

EXPECTED_TASK_TSV_SHA="$(jq -er '.external_inputs.full_task_tsv.sha256' "${SOURCE_LOCK_PATH}")"

mapfile -t SHARDS < <(find "${INPUT_DIR}/shards" -maxdepth 1 -type f -name 'task_*.json' | sort)
if (( ${#SHARDS[@]} != 45 )); then
  echo "Expected 45 dense-specificity shards, found ${#SHARDS[@]}." >&2
  exit 3
fi
for shard in "${SHARDS[@]}"; do
  if [[ "$(jq -r '.task_tsv_sha256 // empty' "${shard}")" != "${EXPECTED_TASK_TSV_SHA}" ]]; then
    echo "Shard task-table hash mismatch: ${shard}" >&2
    exit 3
  fi
done

echo "Summarizer SHA256: $(sha256sum experiments/neurips_2026/summarize_global_k_dense_specificity.py | awk '{print $1}')"
echo "Task module SHA256: $(sha256sum experiments/neurips_2026/global_k_dense_zero_wd_tasks.py | awk '{print $1}')"
echo "Card SHA256: $(sha256sum "${CARD_PATH}" | awk '{print $1}')"
echo "Task-table SHA256: ${EXPECTED_TASK_TSV_SHA}"

uv run python experiments/neurips_2026/summarize_global_k_dense_specificity.py \
  --card "${CARD_PATH}" \
  --source_lock "${SOURCE_LOCK_PATH}" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}"
