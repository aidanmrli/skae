#!/usr/bin/env bash
#SBATCH --job-name=ac-bridge-data
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --array=0-2%3
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

DATASET_SEEDS=(20260729 20260730 20260731)
DATASET_SEED=${DATASET_SEEDS[$SLURM_ARRAY_TASK_ID]}
DATASET=${OUTPUT_ROOT}/data/allen_cahn_grid16_dt0p1_t40_seed${DATASET_SEED}.pt
RECORD=${DATASET}.summary.json
RAW=${OUTPUT_ROOT}/telemetry/generate_seed_${DATASET_SEED}_raw.csv
SUMMARY=${OUTPUT_ROOT}/telemetry/generate_seed_${DATASET_SEED}.json
START=${OUTPUT_ROOT}/telemetry/generate_seed_${DATASET_SEED}.start
DONE=${OUTPUT_ROOT}/telemetry/generate_seed_${DATASET_SEED}.done
for path in "${RAW}" "${SUMMARY}" "${START}" "${DONE}"; do
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
START_TIME=$(date +%s.%N)
printf '{"event":"gpu_invocation_start","seed":%s,"slurm_job_id":"%s","unix_time":%s}\n' \
  "${DATASET_SEED}" "${SLURM_JOB_ID}" "${START_TIME}" > "${START}"
(
  while true; do
    nvidia-smi -i "${GPU_SELECTOR}" \
      --query-gpu=uuid,timestamp,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits >> "${RAW}"
    sleep 2
  done
) &
MONITOR_PID=$!
uv run python -m experiments.neurips_2026.allen_cahn_mechanistic_bridge.generate \
  --task_index "${SLURM_ARRAY_TASK_ID}" \
  --decision "${MECHANISM_DECISION}" \
  --expected_decision_sha256 "${EXPECTED_MECHANISM_DECISION_SHA256}" \
  --expected_card_sha256 "${EXPECTED_CARD_SHA256}" \
  --expected_source_manifest_sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}"
cleanup
MONITOR_PID=
DONE_TIME=$(date +%s.%N)
printf '{"event":"gpu_invocation_done","seed":%s,"slurm_job_id":"%s","unix_time":%s}\n' \
  "${DATASET_SEED}" "${SLURM_JOB_ID}" "${DONE_TIME}" > "${DONE}"
DEVICE_NAME=$(nvidia-smi -i "${GPU_SELECTOR}" --query-gpu=name --format=csv,noheader | head -n 1)
uv run python -m experiments.neurips_2026.allen_cahn_mechanistic_bridge.generation_telemetry \
  --raw "${RAW}" --output "${SUMMARY}" --dataset "${DATASET}" \
  --generation_record "${RECORD}" --seed "${DATASET_SEED}" \
  --slurm_job_id "${SLURM_JOB_ID}" --device_name "${DEVICE_NAME}" \
  --gpu_start_file "${START}" --gpu_done_file "${DONE}" \
  --gpu_uuid "${GPU_SELECTOR}" --visible_gpu_count "${#VISIBLE_GPU_UUIDS[@]}" \
  --expected_card_sha256 "${EXPECTED_CARD_SHA256}" \
  --expected_source_manifest_sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}"
trap - EXIT
