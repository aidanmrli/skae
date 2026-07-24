#!/bin/bash
# Run the frozen 10-sparse + 10-dense pack concurrently on one A100 80GB.

#SBATCH --job-name=gkv2_mix20
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
export SKAE_GIT_COMMIT="$(git rev-parse HEAD)"
source scripts/common/cluster_env.sh
source scripts/common/gpu_guard.sh

MODE="${MODE:?MODE must be smoke or full}"
TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
TASK_MANIFEST="${TASK_MANIFEST:?TASK_MANIFEST is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"
PACK_ROOT="${PACK_ROOT:?PACK_ROOT is required}"
SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH:?SOURCE_LOCK_PATH is required}"
EXPECTED_SOURCE_LOCK_SHA="${EXPECTED_SOURCE_LOCK_SHA:?EXPECTED_SOURCE_LOCK_SHA is required}"
CARD_PATH="${CARD_PATH:-experiments/neurips_2026/global_k_distinct_laws_v2_card.json}"
PACK_SIZE="${PACK_SIZE:-20}"
PACK_CONCURRENCY="${PACK_CONCURRENCY:-20}"
GPU_TELEMETRY_INTERVAL="${GPU_TELEMETRY_INTERVAL:-2}"

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "MODE must be smoke or full" >&2
  exit 2
fi
if [[ "${PACK_SIZE}" != "20" || "${PACK_CONCURRENCY}" != "20" ]]; then
  echo "Approved protocol requires pack size/concurrency 20." >&2
  exit 2
fi
uv run python -m experiments.neurips_2026.global_k_distinct_laws_v2_preflight mixed \
  --mode "${MODE}" \
  --card "${CARD_PATH}" \
  --source_lock "${SOURCE_LOCK_PATH}" \
  --expected_source_lock_sha "${EXPECTED_SOURCE_LOCK_SHA}" \
  --task_tsv "${TASK_TSV}" \
  --task_manifest "${TASK_MANIFEST}"

gpu_guard_assert_cuda_visible "distinct-law V2 mixed pack"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
GPU_MEMORY_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ "${GPU_NAME}" != *A100* || "${GPU_MEMORY_MIB}" -lt 80000 ]]; then
  echo "Approved smoke requires one A100 80GB; observed ${GPU_NAME} ${GPU_MEMORY_MIB} MiB." >&2
  exit 3
fi

STATUS_DIR="${PACK_ROOT}/status"
TASK_LOG_DIR="${PACK_ROOT}/quarantined_task_logs"
TELEMETRY_LOG="${PACK_ROOT}/gpu_telemetry.csv"
PACK_TIMING="${PACK_ROOT}/pack_timing.tsv"
mkdir -p "${STATUS_DIR}" "${TASK_LOG_DIR}"
if find "${STATUS_DIR}" -type f -name 'task_*.tsv' -print -quit | grep -q .; then
  echo "Refusing to reuse a nonempty status directory: ${STATUS_DIR}" >&2
  exit 3
fi

echo -e "event\tepoch_seconds" > "${PACK_TIMING}"
echo -e "pack_start\t$(date +%s.%N)" >> "${PACK_TIMING}"
gpu_guard_print_context "Distinct-law V2 mixed ${MODE} pack"
gpu_guard_start_sampler "${TELEMETRY_LOG}" "${GPU_TELEMETRY_INTERVAL}"
trap gpu_guard_stop_sampler EXIT

declare -a PIDS=()
for ((TASK_ID = 0; TASK_ID < PACK_SIZE; TASK_ID++)); do
  TASK_LOG="${TASK_LOG_DIR}/task_${TASK_ID}.log"
  STATUS_PATH="${STATUS_DIR}/task_${TASK_ID}.tsv"
  (
    START_EPOCH="$(date +%s.%N)"
    set +e
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    GPU_TELEMETRY=0 \
    SKIP_COMPLETED=0 \
    RESUME_FROM_LATEST=0 \
    SLURM_ARRAY_TASK_ID="${TASK_ID}" \
    ARRAY_OFFSET=0 \
    TASK_TSV="${TASK_TSV}" \
    BASE_OUT="${BASE_OUT}" \
    bash scripts/common/run_benchmark_task.sh >"${TASK_LOG}" 2>&1
    EXIT_CODE=$?
    set -e
    END_EPOCH="$(date +%s.%N)"
    {
      echo -e "task_id\tstart_epoch_seconds\tend_epoch_seconds\texit_code"
      echo -e "${TASK_ID}\t${START_EPOCH}\t${END_EPOCH}\t${EXIT_CODE}"
    } > "${STATUS_PATH}.tmp"
    mv "${STATUS_PATH}.tmp" "${STATUS_PATH}"
    exit "${EXIT_CODE}"
  ) &
  PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAILED=1
  fi
done
gpu_guard_stop_sampler
echo -e "pack_end\t$(date +%s.%N)" >> "${PACK_TIMING}"
echo "Mixed ${MODE} pack complete; failed=${FAILED}; task outcomes remain quarantined."
exit "${FAILED}"
