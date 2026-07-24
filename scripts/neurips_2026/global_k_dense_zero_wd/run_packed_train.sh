#!/bin/bash
# Run several independent zero-WD dense controls concurrently on one GPU.

#SBATCH --job-name=dense0wd_pack
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err
#SBATCH --requeue

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source scripts/common/gpu_guard.sh

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"
PACK_TELEMETRY_LOG="${PACK_TELEMETRY_LOG:?PACK_TELEMETRY_LOG is required}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
PACK_SIZE="${PACK_SIZE:-15}"
PACK_CONCURRENCY="${PACK_CONCURRENCY:-15}"
GPU_TELEMETRY_INTERVAL="${GPU_TELEMETRY_INTERVAL:-30}"
CARD_PATH="${CARD_PATH:-experiments/neurips_2026/global_k_dense_zero_wd_card.json}"

if (( PACK_SIZE <= 0 || PACK_CONCURRENCY <= 0 )); then
  echo "PACK_SIZE and PACK_CONCURRENCY must be positive."
  exit 2
fi

PACK_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START_TASK=$((ARRAY_OFFSET + PACK_TASK_ID * PACK_SIZE))
PACK_TELEMETRY_LOG="${PACK_TELEMETRY_LOG//\{array_task_id\}/${PACK_TASK_ID}}"
TASK_THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} / PACK_CONCURRENCY ))
if (( TASK_THREADS < 1 )); then
  TASK_THREADS=1
fi

echo "Dense zero-WD packed training"
echo "Job: ${SLURM_JOB_ID:-local}; array task: ${PACK_TASK_ID}"
echo "Task range starts at ${START_TASK}; pack=${PACK_SIZE}; concurrency=${PACK_CONCURRENCY}"
echo "Task table: ${TASK_TSV}"
echo "Base output: ${BASE_OUT}"
echo "Card SHA256: $(sha256sum "${CARD_PATH}" | awk '{print $1}')"
echo "Task-table SHA256: $(sha256sum "${TASK_TSV}" | awk '{print $1}')"
echo "Git commit: $(git rev-parse HEAD)"

gpu_guard_assert_cuda_visible "dense zero-WD packed training"
gpu_guard_print_context "Dense zero-WD packed training"
mkdir -p "$(dirname "${PACK_TELEMETRY_LOG}")"
gpu_guard_start_sampler "${PACK_TELEMETRY_LOG}" "${GPU_TELEMETRY_INTERVAL}"
trap gpu_guard_stop_sampler EXIT

RUNNING_TASKS=0
FAILED_TASKS=0

wait_for_slot() {
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
    break
  fi
  (
    OMP_NUM_THREADS="${TASK_THREADS}" \
    MKL_NUM_THREADS="${TASK_THREADS}" \
    NUMEXPR_NUM_THREADS="${TASK_THREADS}" \
    GPU_TELEMETRY=0 \
    SKAE_DENSE_CARD_SHA="$(sha256sum "${CARD_PATH}" | awk '{print $1}')" \
    SLURM_ARRAY_TASK_ID="${GLOBAL_TASK_ID}" \
    ARRAY_OFFSET=0 \
    TASK_TSV="${TASK_TSV}" \
    BASE_OUT="${BASE_OUT}" \
    bash scripts/common/run_benchmark_task.sh
  ) &
  RUNNING_TASKS=$((RUNNING_TASKS + 1))
  if (( RUNNING_TASKS >= PACK_CONCURRENCY )); then
    wait_for_slot
  fi
done

while (( RUNNING_TASKS > 0 )); do
  wait_for_slot
done

gpu_guard_stop_sampler
echo "Dense zero-WD packed training finished; failed=${FAILED_TASKS}."
exit "${FAILED_TASKS}"
