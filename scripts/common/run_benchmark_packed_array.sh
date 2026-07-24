#!/bin/bash
#
# Packed wrapper for scripts/common/run_benchmark_task.sh.
#
# Each SLURM array element runs PACK_SIZE task-table rows sequentially on one GPU.
# This keeps campaigns under accounting job-count limits without changing the
# underlying single-row runner.
#
# Required env vars:
#   TASK_TSV=<path>
#   BASE_OUT=<output root>
#
# Optional:
#   ARRAY_OFFSET=0
#   PACK_SIZE=12
#   PACK_CONCURRENCY=1
#
#SBATCH --job-name=paper_bench_pack
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err
#SBATCH --requeue

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source scripts/common/gpu_guard.sh
trap gpu_guard_stop_sampler EXIT

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
PACK_SIZE="${PACK_SIZE:-12}"
PACK_CONCURRENCY="${PACK_CONCURRENCY:-1}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-}"
TASK_TSV_SHA256="${TASK_TSV_SHA256:-}"

if [[ -n "${SOURCE_MANIFEST}" ]]; then
  sha256sum -c "${SOURCE_MANIFEST}"
fi
if [[ -n "${TASK_TSV_SHA256}" ]]; then
  printf '%s  %s\n' "${TASK_TSV_SHA256}" "${TASK_TSV}" | sha256sum -c -
fi

if (( PACK_SIZE <= 0 )); then
  echo "PACK_SIZE must be positive, got ${PACK_SIZE}"
  exit 1
fi
if (( PACK_CONCURRENCY <= 0 )); then
  echo "PACK_CONCURRENCY must be positive, got ${PACK_CONCURRENCY}"
  exit 1
fi

PACK_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START_TASK=$((ARRAY_OFFSET + PACK_TASK_ID * PACK_SIZE))
PACKED_TASK_THREADS=$(( ${SLURM_CPUS_PER_TASK:-4} / PACK_CONCURRENCY ))
if (( PACKED_TASK_THREADS < 1 )); then
  PACKED_TASK_THREADS=1
fi

echo "============================================="

TELEMETRY_DIR="${BASE_OUT}/gpu_telemetry"
mkdir -p "${TELEMETRY_DIR}"
gpu_guard_start_sampler \
  "${TELEMETRY_DIR}/job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${PACK_TASK_ID}.csv" \
  "${GPU_TELEMETRY_INTERVAL:-60}"
gpu_guard_phase "packed benchmark start array_task=${PACK_TASK_ID}"
echo "Packed Paper Benchmark Runner"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${PACK_TASK_ID}"
echo "Start Task Row: ${START_TASK}"
echo "PACK_SIZE: ${PACK_SIZE}"
echo "PACK_CONCURRENCY: ${PACK_CONCURRENCY}"
echo "PACKED_TASK_THREADS: ${PACKED_TASK_THREADS}"
echo "TASK_TSV: ${TASK_TSV}"
echo "BASE_OUT: ${BASE_OUT}"
echo "Start Time: $(date)"
echo "============================================="

RUNNING_TASKS=0
FAILED_TASKS=0

wait_for_packed_slot() {
  local status=0
  set +e
  wait -n
  status=$?
  set -e
  if (( status != 0 )); then
    FAILED_TASKS=1
  fi
  RUNNING_TASKS=$((RUNNING_TASKS - 1))
}

for ((PACK_INDEX = 0; PACK_INDEX < PACK_SIZE; PACK_INDEX++)); do
  GLOBAL_TASK_ID=$((START_TASK + PACK_INDEX))
  LINE_NO=$((GLOBAL_TASK_ID + 2))
  TASK_LINE="$(sed -n "${LINE_NO}p" "${TASK_TSV}" || true)"
  if [[ -z "${TASK_LINE}" ]]; then
    echo "No task row for global task ${GLOBAL_TASK_ID}; stopping pack."
    break
  fi

  echo "----- packed row ${PACK_INDEX}/${PACK_SIZE}: global task ${GLOBAL_TASK_ID} -----"
  (
    OMP_NUM_THREADS="${PACKED_TASK_THREADS}" \
    MKL_NUM_THREADS="${PACKED_TASK_THREADS}" \
    NUMEXPR_NUM_THREADS="${PACKED_TASK_THREADS}" \
    SLURM_ARRAY_TASK_ID="${GLOBAL_TASK_ID}" \
    ARRAY_OFFSET=0 \
    TASK_TSV="${TASK_TSV}" \
    BASE_OUT="${BASE_OUT}" \
    PACK_LEVEL_GPU_GUARD=1 \
    SOURCE_MANIFEST="${SOURCE_MANIFEST}" \
    TASK_TSV_SHA256="${TASK_TSV_SHA256}" \
    bash scripts/common/run_benchmark_task.sh
  ) &
  RUNNING_TASKS=$((RUNNING_TASKS + 1))
  if (( RUNNING_TASKS >= PACK_CONCURRENCY )); then
    wait_for_packed_slot
  fi
done

while (( RUNNING_TASKS > 0 )); do
  wait_for_packed_slot
done

echo "============================================="
echo "Packed runner end time: $(date)"
echo "============================================="
gpu_guard_phase "packed benchmark end array_task=${PACK_TASK_ID} failed=${FAILED_TASKS}"
gpu_guard_stop_sampler
exit "${FAILED_TASKS}"
