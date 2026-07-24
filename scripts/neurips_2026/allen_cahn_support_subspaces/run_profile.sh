#!/usr/bin/env bash
#SBATCH --job-name=ac-support-profile
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/network/scratch/l/lia/skae/allen_cahn_support_subspaces_20260720_v4/slurm/profile_%j.out
#SBATCH --error=/network/scratch/l/lia/skae/allen_cahn_support_subspaces_20260720_v4/slurm/profile_%j.err

set -euo pipefail

REPO_ROOT=/home/mila/l/lia/skae
OUTPUT_ROOT=${OUTPUT_ROOT:-/network/scratch/l/lia/skae/allen_cahn_support_subspaces_20260720_v4}
: "${EXPECTED_CARD_SHA256:?missing EXPECTED_CARD_SHA256}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?missing EXPECTED_SOURCE_MANIFEST_SHA256}"
PROFILE_DIR=${OUTPUT_ROOT}/profile
MANIFEST=${REPO_ROOT}/experiments/neurips_2026/allen_cahn_support_subspaces/source_manifest.sha256
CARD=${REPO_ROOT}/experiments/neurips_2026/allen_cahn_support_subspaces/prediction_card.json

test "$(sha256sum "${CARD}" | awk '{print $1}')" = "${EXPECTED_CARD_SHA256}"
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
test ! -e "${PROFILE_DIR}"
mkdir -p "${PROFILE_DIR}"
cd "${REPO_ROOT}"
ALLOCATED_GPU_IDS=${SLURM_JOB_GPUS:?missing SLURM_JOB_GPUS}
[[ "${ALLOCATED_GPU_IDS}" != *,* ]]
mapfile -t VISIBLE_GPU_UUIDS < <(
  nvidia-smi --query-gpu=uuid --format=csv,noheader
)
[[ "${#VISIBLE_GPU_UUIDS[@]}" -eq 1 ]]
GPU_SELECTOR=${VISIBLE_GPU_UUIDS[0]}
[[ "${GPU_SELECTOR}" == GPU-* ]]

MONITOR_PID=
stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=
  fi
}
trap stop_monitor EXIT

for BATCH_SIZE in 128 256; do
  TELEMETRY=${PROFILE_DIR}/batch_${BATCH_SIZE}_nvidia_smi.csv
  READY_FILE=${PROFILE_DIR}/batch_${BATCH_SIZE}.ready
  START_FILE=${PROFILE_DIR}/batch_${BATCH_SIZE}.start
  uv run python -m experiments.neurips_2026.allen_cahn_support_subspaces.profile \
    --output "${PROFILE_DIR}/batch_${BATCH_SIZE}.json" \
    --batch_size "${BATCH_SIZE}" --minimum_seconds 50 \
    --gpu_selector "${GPU_SELECTOR}" --ready_file "${READY_FILE}" --start_file "${START_FILE}" \
    --expected_card_sha256 "${EXPECTED_CARD_SHA256}" \
    --expected_source_manifest_sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" &
  PROFILE_PID=$!
  for _ in $(seq 1 600); do
    [[ -e "${READY_FILE}" ]] && break
    kill -0 "${PROFILE_PID}" 2>/dev/null || { wait "${PROFILE_PID}"; exit 1; }
    sleep 0.1
  done
  test -e "${READY_FILE}"
  (
    while true; do
      nvidia-smi -i "${GPU_SELECTOR}" --query-gpu=uuid,timestamp,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits >> "${TELEMETRY}"
      sleep 2
    done
  ) &
  MONITOR_PID=$!
  touch "${START_FILE}"
  wait "${PROFILE_PID}"
  stop_monitor
done

uv run python -m experiments.neurips_2026.allen_cahn_support_subspaces.select_profile \
  --profile_dir "${PROFILE_DIR}" --output "${PROFILE_DIR}/decision.json" \
  --source_manifest_sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --expected_card_sha256 "${EXPECTED_CARD_SHA256}"
sha256sum "${PROFILE_DIR}/decision.json"
