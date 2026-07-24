#!/usr/bin/env bash
#SBATCH --job-name=gkrf-gpu
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
umask 077

PROJECT_DIR="/home/mila/l/lia/skae"
PACKAGE_DIR="${PROJECT_DIR}/experiments/neurips_2026/global_k_residual_forecast"
MODE="${MODE:?required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?required}"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?required}"
EXPECTED_TASK_SHA256="${EXPECTED_TASK_SHA256:?required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?required}"
AUTHORIZATION="${AUTHORIZE_GLOBAL_K_RESIDUAL_FORECAST:?required}"
EXPECTED_AUTHORIZATION="root-redteam-approved:${EXPECTED_CARD_SHA256}:${EXPECTED_SOURCE_MANIFEST_SHA256}:${EXPECTED_TASK_SHA256}"
CARD_PATH="${PACKAGE_DIR}/prediction_card.json"
TASK_PATH="${PACKAGE_DIR}/task_manifest.json"
SOURCE_PATH="${PACKAGE_DIR}/source_manifest.sha256"

if [[ "${AUTHORIZATION}" != "${EXPECTED_AUTHORIZATION}" ]]; then
  echo "Independent-redteam authorization does not bind the exact freeze." >&2
  exit 1
fi
if [[ "${MODE}" != "smoke" && "${MODE}" != "scientific" ]]; then
  echo "MODE must be smoke or scientific." >&2
  exit 1
fi
if [[ "${MODE}" == "scientific" ]]; then
  if [[ ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]] || \
     (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 9 )); then
    echo "Scientific mode requires an explicit array task ID in 0..9." >&2
    exit 1
  fi
  TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
else
  TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
  if [[ ! "${TASK_INDEX}" =~ ^[0-9]+$ ]] || (( TASK_INDEX != 0 )); then
    echo "Smoke mode requires task ID 0." >&2
    exit 1
  fi
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
  echo "Exactly one visible GPU is required." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
printf '%s  %s\n' \
  "${EXPECTED_CARD_SHA256}" "${CARD_PATH}" \
  "${EXPECTED_TASK_SHA256}" "${TASK_PATH}" \
  "${EXPECTED_SOURCE_MANIFEST_SHA256}" "${SOURCE_PATH}" \
  | sha256sum --check --strict --status -
sha256sum --check --strict --status "${SOURCE_PATH}"
ROOT_DIR="${PROJECT_DIR}"
source scripts/common/cluster_env.sh
CUDA_VISIBLE_DEVICES="" uv run python -m \
  experiments.neurips_2026.global_k_residual_forecast.preflight \
  --stage forecast \
  --mode "${MODE}" \
  --task-index "${TASK_INDEX}" \
  --output-root "${OUTPUT_ROOT}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --expected-task-sha256 "${EXPECTED_TASK_SHA256}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}"

MODE_ROOT="${OUTPUT_ROOT}/${MODE}"
mkdir -p "${MODE_ROOT}/shards" "${MODE_ROOT}/telemetry" "${MODE_ROOT}/compute_windows"
TRACE="${MODE_ROOT}/telemetry/task_$(printf '%02d' "${TASK_INDEX}").csv"
SHARD="${MODE_ROOT}/shards/task_$(printf '%02d' "${TASK_INDEX}").json"
WINDOW="${MODE_ROOT}/compute_windows/task_$(printf '%02d' "${TASK_INDEX}").json"
if [[ -e "${TRACE}" || -e "${SHARD}" || -e "${WINDOW}" ]]; then
  echo "Refusing pre-existing task artifacts for ${MODE}/${TASK_INDEX}." >&2
  exit 1
fi

(
  echo "epoch_seconds,gpu_uuid,gpu_name,utilization_gpu,memory_used_mib,memory_total_mib"
  while true; do
    EPOCH="$(date +%s.%N)"
    GPU_ROW="$(nvidia-smi --query-gpu=uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)"
    echo "${EPOCH},${GPU_ROW}"
    sleep 1
  done
) > "${TRACE}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

export SKAE_GIT_COMMIT="$(git rev-parse HEAD)"
uv run python -m experiments.neurips_2026.global_k_residual_forecast.evaluate \
  --mode "${MODE}" \
  --task-index "${TASK_INDEX}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --expected-task-sha256 "${EXPECTED_TASK_SHA256}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --output-root "${OUTPUT_ROOT}" \
  --output "${SHARD}" \
  --compute-window "${WINDOW}"

if ! kill -0 "${MONITOR_PID}" 2>/dev/null; then
  wait "${MONITOR_PID}" || MONITOR_STATUS=$?
  echo "GPU telemetry monitor exited unexpectedly with status ${MONITOR_STATUS:-0}." >&2
  exit 1
fi
if ! kill "${MONITOR_PID}" 2>/dev/null; then
  echo "GPU telemetry monitor disappeared before controlled shutdown." >&2
  exit 1
fi
set +e
wait "${MONITOR_PID}"
MONITOR_STATUS=$?
set -e
if [[ "${MONITOR_STATUS}" -ne 0 && "${MONITOR_STATUS}" -ne 143 ]]; then
  echo "GPU telemetry monitor failed with status ${MONITOR_STATUS}." >&2
  exit 1
fi
trap - EXIT

# An explicit post-compute sample brackets the synchronized end marker. This is
# telemetry coverage outside the measured compute window, not workload padding.
EPOCH="$(date +%s.%N)"
GPU_ROW="$(nvidia-smi --query-gpu=uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)"
echo "${EPOCH},${GPU_ROW}" >> "${TRACE}"
