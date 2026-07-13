#!/usr/bin/env bash
#
# Packed wrapper for scripts/run_spatialized_reaction_diffusion_array.sh.
#
# Each SLURM array element receives one GPU and runs PACK_SIZE independent
# spatialized PDE training rows concurrently on that GPU. This avoids the
# previous one-small-model-per-GPU launch pattern, which underused GPU memory
# and compute for grid-16 models.
#
# Required env vars:
#   TASK_TSV=<path>
#
# Optional:
#   ARRAY_OFFSET=0
#   PACK_SIZE=6
#   GPU_MONITOR=1
#   GPU_MONITOR_DIR=<path>
#
#SBATCH --job-name=spatial-rd-pack
#SBATCH --output=/network/scratch/l/lia/skae/spatial-rd-pack-%A_%a.out
#SBATCH --error=/network/scratch/l/lia/skae/spatial-rd-pack-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
PACK_SIZE="${PACK_SIZE:-6}"
GPU_MONITOR="${GPU_MONITOR:-1}"
GPU_MONITOR_DIR="${GPU_MONITOR_DIR:-${ROOT_DIR}/results/gpu_monitor}"

if (( PACK_SIZE <= 0 )); then
  echo "PACK_SIZE must be positive, got ${PACK_SIZE}" >&2
  exit 1
fi

TOTAL_TASKS=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TOTAL_TASKS <= 0 )); then
  echo "No tasks found in ${TASK_TSV}" >&2
  exit 1
fi

PACK_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_TASK=$((ARRAY_OFFSET + PACK_TASK_ID * PACK_SIZE))

echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse --short HEAD || true)"
echo "task_tsv=${TASK_TSV}"
echo "total_tasks=${TOTAL_TASKS}"
echo "pack_task_id=${PACK_TASK_ID}"
echo "start_task=${START_TASK}"
echo "pack_size=${PACK_SIZE}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "slurm_job_id=${SLURM_JOB_ID:-local}"

pids=()
monitor_pid=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  exit "${status}"
}
trap cleanup EXIT INT TERM

if [[ "${GPU_MONITOR}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  mkdir -p "${GPU_MONITOR_DIR}"
  monitor_log="${GPU_MONITOR_DIR}/gpu-util-${SLURM_JOB_ID:-local}_${PACK_TASK_ID}.csv"
  echo "gpu_monitor_log=${monitor_log}"
  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv \
    -l 5 \
    > "${monitor_log}" 2>&1 &
  monitor_pid=$!
  nvidia-smi || true
fi

launched=0
for ((pack_index = 0; pack_index < PACK_SIZE; pack_index++)); do
  global_task_id=$((START_TASK + pack_index))
  if (( global_task_id >= TOTAL_TASKS )); then
    echo "No task row for global task ${global_task_id}; stopping pack."
    break
  fi

  echo "launching_pack_index=${pack_index} global_task_id=${global_task_id}"
  (
    export SLURM_ARRAY_TASK_ID="${global_task_id}"
    export ARRAY_OFFSET=0
    export TASK_TSV="${TASK_TSV}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
    bash scripts/run_spatialized_reaction_diffusion_array.sh
  ) &
  pids+=("$!")
  launched=$((launched + 1))
done

if (( launched == 0 )); then
  echo "No tasks launched for packed array element ${PACK_TASK_ID}."
  exit 0
fi

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "One or more packed training rows failed." >&2
  exit 1
fi

echo "completed_at=$(date --iso-8601=seconds)"
