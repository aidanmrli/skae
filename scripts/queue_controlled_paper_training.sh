#!/bin/bash
# Queue the frozen 15-system x six-row x 15-seed controlled paper matrix.
#
# Submit with:
#   sbatch scripts/queue_controlled_paper_training.sh
#
# Optional repair subsets:
#   SYSTEMS_CSV=gated_local_linear,claude:cal_square_4
#   MODEL_VARIANTS_CSV=mlp_sparse_blockdiag_hardinit_basin_partition_control
#   SEEDS_CSV=3,9

#SBATCH --job-name=queue_controlled
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH -o /network/scratch/l/lia/skae/queue-controlled-paper-%j.out
#SBATCH -e /network/scratch/l/lia/skae/queue-controlled-paper-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit this launcher with sbatch scripts/queue_controlled_paper_training.sh"
  exit 2
fi

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-controlled_paper_training_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-transition_rich_basin_partition}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
QUEUE_DIR="${QUEUE_DIR:-${RESULTS_DIR}/queue}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
NUM_STEPS_OVERRIDE="${NUM_STEPS_OVERRIDE:-}"
TRAIN_PACK_SIZE="${TRAIN_PACK_SIZE:-12}"
TRAIN_ARRAY_PARALLEL="${TRAIN_ARRAY_PARALLEL:-90}"
TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT:-3-00:00:00}"

if (( TRAIN_PACK_SIZE <= 0 || TRAIN_ARRAY_PARALLEL <= 0 )); then
  echo "TRAIN_PACK_SIZE and TRAIN_ARRAY_PARALLEL must be positive."
  exit 2
fi

mkdir -p "${TASK_DIR}" "${QUEUE_DIR}"
TASK_TSV="${TASK_DIR}/controlled_paper_tasks.tsv"
MANIFEST_JSON="${TASK_DIR}/controlled_paper_manifest.json"

BUILD_ARGS=(
  tools/build_transition_rich_basin_partition_tasks.py
  --paper_protocol
  --phase_label "${PHASE_LABEL}"
  --output_tsv "${TASK_TSV}"
  --output_manifest_json "${MANIFEST_JSON}"
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
if [[ -n "${NUM_STEPS_OVERRIDE}" ]]; then
  BUILD_ARGS+=(--num_steps_override "${NUM_STEPS_OVERRIDE}")
fi

uv run python "${BUILD_ARGS[@]}"

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No controlled paper tasks were built."
  exit 1
fi
TRAIN_ARRAY_COUNT=$(( (TASK_COUNT + TRAIN_PACK_SIZE - 1) / TRAIN_PACK_SIZE ))

TRAIN_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" \
  BASE_OUT="${BASE_OUT}" \
  PACK_SIZE="${TRAIN_PACK_SIZE}" \
    sbatch --parsable \
      --partition=long \
      --time="${TRAIN_TIME_LIMIT}" \
      --array=0-$((TRAIN_ARRAY_COUNT - 1))%"${TRAIN_ARRAY_PARALLEL}" \
      scripts/run_paper_benchmark_packed_array.sh
)

cat > "${QUEUE_DIR}/launch_record.json" <<EOF
{
  "experiment_tag": "${EXPERIMENT_TAG}",
  "git_commit": "$(git rev-parse HEAD)",
  "task_tsv": "${TASK_TSV}",
  "manifest_json": "${MANIFEST_JSON}",
  "base_out": "${BASE_OUT}",
  "task_count": ${TASK_COUNT},
  "train_pack_size": ${TRAIN_PACK_SIZE},
  "train_array_count": ${TRAIN_ARRAY_COUNT},
  "train_job_id": "${TRAIN_JOB_ID}"
}
EOF

echo "Queued ${TASK_COUNT} controlled paper tasks in ${TRAIN_ARRAY_COUNT} GPU packs."
echo "Training array: ${TRAIN_JOB_ID}"
echo "Task table: ${TASK_TSV}"
echo "Manifest: ${MANIFEST_JSON}"
