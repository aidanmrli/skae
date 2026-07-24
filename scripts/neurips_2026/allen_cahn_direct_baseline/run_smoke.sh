#!/usr/bin/env bash
#SBATCH --job-name=ac-direct-smoke
#SBATCH --output=/network/scratch/l/lia/skae/allen_cahn_direct_baseline_v1_20260721/logs/smoke-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/allen_cahn_direct_baseline_v1_20260721/logs/smoke-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G

set -euo pipefail

REPO_ROOT="/home/mila/l/lia/skae"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_direct_baseline_v1_20260721"
TASK_LOCK="${REPO_ROOT}/experiments/neurips_2026/allen_cahn_direct_baseline/task_lock.json"
TASK_LOCK_SHA256="${TASK_LOCK_SHA256:?TASK_LOCK_SHA256 must be exported by the authenticated launcher}"
RUN_DIR="${OUTPUT_ROOT}/smoke/job_${SLURM_JOB_ID}"
TELEMETRY="${RUN_DIR}/raw_gpu_telemetry.csv"

if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite ${RUN_DIR}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${RUN_DIR}"
cd "${REPO_ROOT}"

echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "git_commit=$(git rev-parse HEAD)"
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
  --seed 64 \
  --run-dir "${RUN_DIR}/training" \
  --device cuda \
  --smoke \
  --smoke-steps 80

sleep 2
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT

uv run python -m experiments.neurips_2026.allen_cahn_direct_baseline.telemetry \
  --telemetry "${TELEMETRY}" \
  --phase-start "${RUN_DIR}/training/training_phase_start.json" \
  --phase-end "${RUN_DIR}/training/training_phase_end.json" \
  --output "${RUN_DIR}/telemetry_audit.json" \
  --minimum-core-samples 5 \
  --task-lock-sha256 "${TASK_LOCK_SHA256}" \
  --seed 64 \
  --artifact-role non_scientific_gpu_smoke \
  --slurm-job-id "${SLURM_JOB_ID}"

sha256sum \
  "${RUN_DIR}/training/run_manifest.json" \
  "${RUN_DIR}/training/training_summary.json" \
  "${RUN_DIR}/telemetry_audit.json" \
  "${TELEMETRY}"
