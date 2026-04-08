#!/bin/bash
#
# Build and queue the fixed transition-rich basin-partition LISTA sweep from a
# compute node. Submit with `sbatch scripts/queue_transition_rich_basin_partition.sh`
# or run inside an existing `salloc` allocation.
#
# Optional env vars:
#   SYSTEMS_CSV=gated_local_linear,claude:transition_routes_4
#   MODEL_VARIANTS_CSV=lista_dense_basin_partition,lista_blockdiag_basin_partition
#   SEEDS_CSV=0,1,2
#
#SBATCH --job-name=queue_tr_basin
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH -o /network/scratch/l/lia/skae/queue-transition-rich-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-transition-rich-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run on a compute node."
  echo "Submit it with: sbatch scripts/queue_transition_rich_basin_partition.sh"
  echo "Or run it inside an existing salloc allocation."
  exit 2
fi

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-transition_rich_basin_partition_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-transition_rich_basin_partition}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

mkdir -p "${TASK_DIR}"

TASK_TSV="${TASK_DIR}/transition_rich_basin_partition.tsv"
MANIFEST_JSON="${TASK_DIR}/transition_rich_basin_partition_manifest.json"

BUILD_ARGS=(
  --output_tsv "${TASK_TSV}"
  --output_manifest_json "${MANIFEST_JSON}"
  --phase_label "${PHASE_LABEL}"
  --eval_profile "${EVAL_PROFILE}"
)

if [[ -n "${SYSTEMS_CSV}" ]]; then
  BUILD_ARGS+=(--systems_csv "${SYSTEMS_CSV}")
fi
if [[ -n "${MODEL_VARIANTS_CSV}" ]]; then
  BUILD_ARGS+=(--model_variants_csv "${MODEL_VARIANTS_CSV}")
fi
if [[ -n "${SEEDS_CSV}" ]]; then
  BUILD_ARGS+=(--seeds_csv "${SEEDS_CSV}")
fi

uv run python tools/build_transition_rich_basin_partition_tasks.py "${BUILD_ARGS[@]}"

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi

ARRAY_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" \
    sbatch --array=0-$((TASK_COUNT - 1)) scripts/run_paper_benchmark_array.sh | awk '{print $4}'
)

echo "Queued transition-rich basin-partition sweep."
echo "Array job: ${ARRAY_JOB_ID}"
echo "Task TSV: ${TASK_TSV}"
echo "Results dir: ${RESULTS_DIR}"
