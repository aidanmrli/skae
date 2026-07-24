#!/usr/bin/env bash
#SBATCH --job-name=ac-support-audit
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%8
#SBATCH --output=/network/scratch/l/lia/skae/allen_cahn_support_subspaces_20260720_v4/slurm/eval_%A_%a.out
#SBATCH --error=/network/scratch/l/lia/skae/allen_cahn_support_subspaces_20260720_v4/slurm/eval_%A_%a.err

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
SEED=$((64 + SLURM_ARRAY_TASK_ID))
RAW_TELEMETRY=${OUTPUT_ROOT}/telemetry/raw_seed_${SEED}.csv
SUMMARY_TELEMETRY=${OUTPUT_ROOT}/telemetry/seed_${SEED}.json
READY_FILE=${OUTPUT_ROOT}/telemetry/seed_${SEED}.ready
RELEASE_FILE=${OUTPUT_ROOT}/telemetry/seed_${SEED}.release
GPU_START_FILE=${OUTPUT_ROOT}/telemetry/seed_${SEED}.gpu_start.json
GPU_DONE_FILE=${OUTPUT_ROOT}/telemetry/seed_${SEED}.gpu_done.json
test ! -e "${RAW_TELEMETRY}"
test ! -e "${SUMMARY_TELEMETRY}"
test ! -e "${READY_FILE}"
test ! -e "${RELEASE_FILE}"
test ! -e "${GPU_START_FILE}"
test ! -e "${GPU_DONE_FILE}"
mkdir -p "${OUTPUT_ROOT}/telemetry"
ALLOCATED_GPU_IDS=${SLURM_JOB_GPUS:?missing SLURM_JOB_GPUS}
[[ "${ALLOCATED_GPU_IDS}" != *,* ]]
mapfile -t VISIBLE_GPU_UUIDS < <(
  nvidia-smi --query-gpu=uuid --format=csv,noheader
)
[[ "${#VISIBLE_GPU_UUIDS[@]}" -eq 1 ]]
GPU_SELECTOR=${VISIBLE_GPU_UUIDS[0]}
[[ "${GPU_SELECTOR}" == GPU-* ]]

cleanup() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

uv run python -m experiments.neurips_2026.allen_cahn_support_subspaces.evaluate \
  --output_root "${OUTPUT_ROOT}" --task_index "${SLURM_ARRAY_TASK_ID}" \
  --batch_size "${BATCH_SIZE}" --profile_decision "${PROFILE}" \
  --expected_card_sha256 "${EXPECTED_CARD_SHA256}" \
  --expected_source_manifest_sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --expected_profile_decision_sha256 "${EXPECTED_PROFILE_DECISION_SHA256}" \
  --ready_file "${READY_FILE}" --release_file "${RELEASE_FILE}" \
  --gpu_start_file "${GPU_START_FILE}" --gpu_done_file "${GPU_DONE_FILE}" &
EVALUATOR_PID=$!
for _ in $(seq 1 600); do
  [[ -e "${READY_FILE}" ]] && break
  kill -0 "${EVALUATOR_PID}" 2>/dev/null || { wait "${EVALUATOR_PID}"; exit 1; }
  sleep 0.1
done
test -e "${READY_FILE}"
touch "${RELEASE_FILE}"
for _ in $(seq 1 600); do
  [[ -e "${GPU_START_FILE}" ]] && break
  kill -0 "${EVALUATOR_PID}" 2>/dev/null || { wait "${EVALUATOR_PID}"; exit 1; }
  sleep 0.1
done
test -e "${GPU_START_FILE}"
(
  while [[ ! -e "${GPU_DONE_FILE}" ]]; do
    sleep 1
    [[ -e "${GPU_DONE_FILE}" ]] && break
    SAMPLE=$(nvidia-smi -i "${GPU_SELECTOR}" \
      --query-gpu=uuid,timestamp,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits)
    [[ -e "${GPU_DONE_FILE}" ]] && break
    printf '%s\n' "${SAMPLE}" >> "${RAW_TELEMETRY}"
  done
) &
MONITOR_PID=$!
for _ in $(seq 1 144000); do
  [[ -e "${GPU_DONE_FILE}" ]] && break
  kill -0 "${EVALUATOR_PID}" 2>/dev/null || { wait "${EVALUATOR_PID}"; exit 1; }
  sleep 0.1
done
test -e "${GPU_DONE_FILE}"
wait "${EVALUATOR_PID}"
wait "${MONITOR_PID}"
MONITOR_PID=
trap - EXIT
DEVICE_NAME=$(nvidia-smi -i "${GPU_SELECTOR}" --query-gpu=name --format=csv,noheader | head -n 1)
uv run python -m experiments.neurips_2026.allen_cahn_support_subspaces.summarize_gpu_telemetry \
  --telemetry "${RAW_TELEMETRY}" --output "${SUMMARY_TELEMETRY}" --seed "${SEED}" \
  --slurm_job_id "${SLURM_JOB_ID}" --device_name "${DEVICE_NAME}" \
  --gpu_uuid "${GPU_SELECTOR}" --visible_gpu_count "${#VISIBLE_GPU_UUIDS[@]}" \
  --gpu_start_file "${GPU_START_FILE}" --gpu_done_file "${GPU_DONE_FILE}" \
  --expected_card_sha256 "${EXPECTED_CARD_SHA256}" \
  --expected_source_manifest_sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}"
