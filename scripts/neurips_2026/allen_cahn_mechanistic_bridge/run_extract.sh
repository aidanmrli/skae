#!/usr/bin/env bash
#SBATCH --job-name=ac-bridge-field
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --array=0-29%8
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

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
: "${DATASET_MANIFEST:?missing DATASET_MANIFEST}"
: "${EXPECTED_DATASET_MANIFEST_SHA256:?missing EXPECTED_DATASET_MANIFEST_SHA256}"
: "${PROFILE_DECISION:?missing PROFILE_DECISION}"
: "${EXPECTED_PROFILE_DECISION_SHA256:?missing EXPECTED_PROFILE_DECISION_SHA256}"
: "${BATCH_SIZE:?missing BATCH_SIZE}"

MODEL_SEED=$((64 + SLURM_ARRAY_TASK_ID / 3))
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
DATASET_SEEDS=(20260729 20260730 20260731)
DATASET_SEED=${DATASET_SEEDS[$DATASET_INDEX]}
STEM=model_${MODEL_SEED}_data_${DATASET_SEED}
RAW=${OUTPUT_ROOT}/telemetry/${STEM}_raw.csv
SUMMARY=${OUTPUT_ROOT}/telemetry/${STEM}.json
READY=${OUTPUT_ROOT}/telemetry/${STEM}.ready
RELEASE=${OUTPUT_ROOT}/telemetry/${STEM}.release
START=${OUTPUT_ROOT}/telemetry/${STEM}.start
DONE=${OUTPUT_ROOT}/telemetry/${STEM}.done
for path in "${RAW}" "${SUMMARY}" "${READY}" "${RELEASE}" "${START}" "${DONE}"; do
  test ! -e "${path}"
done
mkdir -p "${OUTPUT_ROOT}/telemetry"

ALLOCATED_GPU_IDS=${SLURM_JOB_GPUS:?missing SLURM_JOB_GPUS}
[[ "${ALLOCATED_GPU_IDS}" != *,* ]]
mapfile -t VISIBLE_GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[[ "${#VISIBLE_GPU_UUIDS[@]}" -eq 1 ]]
GPU_SELECTOR=${VISIBLE_GPU_UUIDS[0]}
[[ "${GPU_SELECTOR}" == GPU-* ]]

MONITOR_PID=
cleanup() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "${REPO_ROOT}"
uv run python -m experiments.neurips_2026.allen_cahn_mechanistic_bridge.extract_field_only \
  --output_root "${OUTPUT_ROOT}" --task_index "${SLURM_ARRAY_TASK_ID}" \
  --dataset_manifest "${DATASET_MANIFEST}" \
  --expected_dataset_manifest_sha256 "${EXPECTED_DATASET_MANIFEST_SHA256}" \
  --decision "${MECHANISM_DECISION}" \
  --expected_decision_sha256 "${EXPECTED_MECHANISM_DECISION_SHA256}" \
  --profile_decision "${PROFILE_DECISION}" \
  --expected_profile_decision_sha256 "${EXPECTED_PROFILE_DECISION_SHA256}" \
  --expected_card_sha256 "${EXPECTED_CARD_SHA256}" \
  --expected_source_manifest_sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --batch_size "${BATCH_SIZE}" --ready_file "${READY}" \
  --release_file "${RELEASE}" --start_file "${START}" \
  --done_file "${DONE}" &
EXTRACT_PID=$!
for _ in $(seq 1 1200); do
  [[ -e "${READY}" ]] && break
  kill -0 "${EXTRACT_PID}" 2>/dev/null || { wait "${EXTRACT_PID}"; exit 1; }
  sleep 0.1
done
test -e "${READY}"
touch "${RELEASE}"
for _ in $(seq 1 600); do
  [[ -e "${START}" ]] && break
  kill -0 "${EXTRACT_PID}" 2>/dev/null || { wait "${EXTRACT_PID}"; exit 1; }
  sleep 0.1
done
test -e "${START}"
(
  while [[ ! -e "${DONE}" ]]; do
    sleep 1
    [[ -e "${DONE}" ]] && break
    SAMPLE=$(nvidia-smi -i "${GPU_SELECTOR}" \
      --query-gpu=uuid,timestamp,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits)
    [[ -e "${DONE}" ]] && break
    printf '%s\n' "${SAMPLE}" >> "${RAW}"
  done
) &
MONITOR_PID=$!
for _ in $(seq 1 27000); do
  [[ -e "${DONE}" ]] && break
  kill -0 "${EXTRACT_PID}" 2>/dev/null || { wait "${EXTRACT_PID}"; exit 1; }
  sleep 0.1
done
test -e "${DONE}"
cleanup
MONITOR_PID=
wait "${EXTRACT_PID}"
test -s "${RAW}"
DEVICE_NAME=$(nvidia-smi -i "${GPU_SELECTOR}" --query-gpu=name --format=csv,noheader | head -n 1)
uv run python -m experiments.neurips_2026.allen_cahn_mechanistic_bridge.telemetry \
  --raw "${RAW}" --output "${SUMMARY}" --model_seed "${MODEL_SEED}" \
  --dataset_seed "${DATASET_SEED}" --slurm_job_id "${SLURM_JOB_ID}" \
  --device_name "${DEVICE_NAME}" --profile_decision "${PROFILE_DECISION}" \
  --gpu_start_file "${START}" --gpu_done_file "${DONE}" \
  --gpu_uuid "${GPU_SELECTOR}" --visible_gpu_count "${#VISIBLE_GPU_UUIDS[@]}" \
  --expected_profile_decision_sha256 "${EXPECTED_PROFILE_DECISION_SHA256}" \
  --expected_card_sha256 "${EXPECTED_CARD_SHA256}" \
  --expected_source_manifest_sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}"
trap - EXIT
