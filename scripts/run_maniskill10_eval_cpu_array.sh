#!/usr/bin/env bash
# CPU-only periodic evaluation/support pass for ManiSkill-10 checkpoints.
# Submit this as a dependency after packed GPU training to avoid holding GPUs
# during low-utilization evaluation and support-family bookkeeping.
#SBATCH --job-name=mskill10_eval_cpu
#SBATCH --partition=long
#SBATCH --output=logs/maniskill10_eval_cpu_%A_%a.out
#SBATCH --error=logs/maniskill10_eval_cpu_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-119%16
#SBATCH --requeue

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs

export CUDA_VISIBLE_DEVICES=""
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

MANIFEST="${MANIFEST:-experiments/maniskill10_default_tasks.tsv}"
DATA_ROOT="${DATA_ROOT:-data/maniskill/default_tasks}"
RUN_ROOT="${RUN_ROOT:-}"
TASK_INDICES="${TASK_INDICES:-0,1,2,3,4,5,6,7,8,9}"
SEEDS="${SEEDS:-0,1,2}"
PERIODIC_REENCODE_PERIODS="${PERIODIC_REENCODE_PERIODS:-1,2,5,10,20,50,100}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-0.2}"
FAMILY_JACCARD="${FAMILY_JACCARD:-0.4}"

if [[ -z "${RUN_ROOT}" ]]; then
  echo "RUN_ROOT is required and must point to the checkpoint root from GPU training." >&2
  exit 2
fi

manifest_row() {
  local index="$1"
  awk -F '\t' 'BEGIN {i=0} /^[[:space:]]*#/ {next} NF >= 3 {if (i == idx) {print; exit} i++}' idx="${index}" "${MANIFEST}"
}

TASKS=()
add_model_rows_for_task_seed() {
  local task_index="$1"
  local seed="$2"
  local row task_id horizons dataset
  row="$(manifest_row "${task_index}")"
  if [[ -z "${row}" ]]; then
    echo "No manifest row for task index ${task_index}" >&2
    exit 2
  fi
  task_id="$(printf '%s\n' "${row}" | awk -F '\t' '{print $1}')"
  horizons="$(printf '%s\n' "${row}" | awk -F '\t' '{print $3}')"
  dataset="${DATA_ROOT}/${task_id}/${task_id}_state_compact_seed0.npz"
  if [[ ! -f "${dataset}" ]]; then
    echo "Missing compact dataset for ${task_id}: ${dataset}" >&2
    exit 2
  fi
  TASKS+=("${task_id}|${dataset}|${horizons}|dense_tanh_lr5em4_wd0|${seed}")
  TASKS+=("${task_id}|${dataset}|${horizons}|sparse_mlp_relu_sp0p003_lr5em4_wd0|${seed}")
  TASKS+=("${task_id}|${dataset}|${horizons}|sparse_mlp_relu_sp0p01_lr5em4_wd0|${seed}")
  TASKS+=("${task_id}|${dataset}|${horizons}|lista_relu_a0p01_sp1em4_std|${seed}")
}

IFS=',' read -r -a TASK_INDEX_ARRAY <<< "${TASK_INDICES}"
IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
for seed in "${SEED_ARRAY[@]}"; do
  for task_index in "${TASK_INDEX_ARRAY[@]}"; do
    add_model_rows_for_task_seed "${task_index}" "${seed}"
  done
done

TASK_COUNT="${#TASKS[@]}"
ROW_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( ROW_ID >= TASK_COUNT )); then
  echo "No evaluation row for array id ${ROW_ID}; task_count=${TASK_COUNT}."
  exit 0
fi

IFS='|' read -r TASK_ID DATASET HORIZONS RUN_TAG SEED <<< "${TASKS[${ROW_ID}]}"
RUN_DIR="${RUN_ROOT}/${TASK_ID}/${RUN_TAG}/seed${SEED}"
CHECKPOINT="${RUN_DIR}/checkpoint.pt"
EVAL_DIR="${RUN_DIR}/eval_test_periodic"

echo "date=$(date)"
echo "host=$(hostname)"
echo "commit=$(git rev-parse --short HEAD || true)"
echo "slurm_job_id=${SLURM_JOB_ID:-}"
echo "slurm_array_task_id=${ROW_ID}"
echo "run_root=${RUN_ROOT}"
echo "task=${TASK_ID}"
echo "run_tag=${RUN_TAG}"
echo "seed=${SEED}"
echo "dataset=${DATASET}"
echo "checkpoint=${CHECKPOINT}"
echo "horizons=${HORIZONS}"
echo "periodic_reencode_periods=${PERIODIC_REENCODE_PERIODS}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

if [[ "${FORCE:-0}" != "1" && -f "${EVAL_DIR}/metrics_summary.json" ]]; then
  echo "Existing eval summary found; skipping ${TASK_ID}/${RUN_TAG}/seed${SEED}."
  exit 0
fi

mkdir -p "${EVAL_DIR}"
uv run python tools/evaluate_maniskill_controlled_lista.py \
  --dataset "${DATASET}" \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${EVAL_DIR}" \
  --device cpu \
  --split test \
  --horizons "${HORIZONS}" \
  --periodic_reencode_periods "${PERIODIC_REENCODE_PERIODS}" \
  --support_threshold "${SUPPORT_THRESHOLD}" \
  --family_jaccard "${FAMILY_JACCARD}"

echo "summary=${EVAL_DIR}/metrics_summary.json"
