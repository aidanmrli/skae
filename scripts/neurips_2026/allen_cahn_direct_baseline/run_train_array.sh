#!/usr/bin/env bash
#SBATCH --job-name=ac-direct-train
#SBATCH --output=/network/scratch/l/lia/skae/allen_cahn_direct_baseline_v1_20260721/logs/train-%A_%a.out
#SBATCH --error=/network/scratch/l/lia/skae/allen_cahn_direct_baseline_v1_20260721/logs/train-%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --array=0-9%10

set -euo pipefail

REPO_ROOT="/home/mila/l/lia/skae"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_direct_baseline_v1_20260721"
TASK_LOCK="${REPO_ROOT}/experiments/neurips_2026/allen_cahn_direct_baseline/task_lock.json"
TASK_LOCK_SHA256="${TASK_LOCK_SHA256:?TASK_LOCK_SHA256 must be exported by the authenticated launcher}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
SEED=$((64 + TASK_ID))
SEED_ROOT="${OUTPUT_ROOT}/training/seed_${SEED}"
MODEL_ROOT="${SEED_ROOT}/model"
TELEMETRY="${SEED_ROOT}/raw_gpu_telemetry.csv"

if [[ -e "${SEED_ROOT}" ]]; then
  echo "Refusing to overwrite ${SEED_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${SEED_ROOT}"
cd "${REPO_ROOT}"
echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID} task=${TASK_ID} seed=${SEED}"
sha256sum "${TASK_LOCK}"
nvidia-smi

printf '%s\n' 'unix_time_seconds,gpu_index,gpu_uuid,gpu_name,utilization_gpu_percent,utilization_memory_percent,memory_used_mib,memory_total_mib,power_draw_w,power_limit_w' > "${TELEMETRY}"
monitor_gpu() {
  while true; do
    epoch="$(date +%s.%N)"
    row="$(nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit --format=csv,noheader,nounits)"
    printf '%s,%s\n' "${epoch}" "${row}" >> "${TELEMETRY}"
    sleep 1
  done
}
monitor_gpu &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

uv run python -m experiments.neurips_2026.allen_cahn_direct_baseline.train \
  --task-lock "${TASK_LOCK}" \
  --expected-task-lock-sha256 "${TASK_LOCK_SHA256}" \
  --seed "${SEED}" \
  --run-dir "${MODEL_ROOT}" \
  --device cuda

sleep 2
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT

uv run python -m experiments.neurips_2026.allen_cahn_direct_baseline.telemetry \
  --telemetry "${TELEMETRY}" \
  --phase-start "${MODEL_ROOT}/training_phase_start.json" \
  --phase-end "${MODEL_ROOT}/training_phase_end.json" \
  --output "${SEED_ROOT}/telemetry_audit.json" \
  --minimum-core-samples 30 \
  --task-lock-sha256 "${TASK_LOCK_SHA256}" \
  --seed "${SEED}" \
  --artifact-role scientific_training \
  --slurm-job-id "${SLURM_JOB_ID}"

sha256sum "${MODEL_ROOT}/checkpoint.pt" "${MODEL_ROOT}/training_summary.json" "${SEED_ROOT}/telemetry_audit.json"

