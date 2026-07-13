#!/usr/bin/env bash
# Packed GPU smoke for ManiSkill-10 default-task controlled-Koopman forecasting.
#
# GPU audit: these controlled KAE/LISTA models are small, so this launcher packs
# four independent rows on one GPU and records nvidia-smi telemetry. The array
# throttle is one GPU at a time by default. Keep PACK_CONCURRENCY >= 2 unless
# BATCH_SIZE is raised substantially. By default this script trains only; run
# periodic evaluation/support in a dependent CPU job to avoid holding a GPU
# during low-utilization bookkeeping.
#SBATCH --job-name=mskill10_5k_gpu
#SBATCH --partition=long
#SBATCH --output=logs/maniskill10_5k_gpu_%A_%a.out
#SBATCH --error=logs/maniskill10_5k_gpu_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%1
#SBATCH --requeue

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs

source scripts/slurm_gpu_guard.sh

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

MANIFEST="${MANIFEST:-experiments/maniskill10_default_tasks.tsv}"
DATA_ROOT="${DATA_ROOT:-data/maniskill/default_tasks}"
RUN_ROOT="${RUN_ROOT:-runs/maniskill10_default/forecast_5k_smoke_${SLURM_ARRAY_JOB_ID:-local}}"
TASK_INDICES="${TASK_INDICES:-0,9}"
SEED="${SEED:-0}"
PACK_SIZE="${PACK_SIZE:-4}"
PACK_CONCURRENCY="${PACK_CONCURRENCY:-4}"
NUM_STEPS="${NUM_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-10}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-2}"
EVAL_EVERY="${EVAL_EVERY:-500}"
PERIODIC_REENCODE_PERIODS="${PERIODIC_REENCODE_PERIODS:-1,2,5,10,20,50,100}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-0.2}"
FAMILY_JACCARD="${FAMILY_JACCARD:-0.4}"
RUN_EVAL_IN_GPU_JOB="${RUN_EVAL_IN_GPU_JOB:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"

if (( PACK_SIZE <= 0 || PACK_CONCURRENCY <= 0 )); then
  echo "PACK_SIZE and PACK_CONCURRENCY must be positive" >&2
  exit 2
fi
if (( PACK_CONCURRENCY < 2 && BATCH_SIZE < 2048 )); then
  echo "Refusing likely GPU-wasteful config: PACK_CONCURRENCY=${PACK_CONCURRENCY}, BATCH_SIZE=${BATCH_SIZE}" >&2
  exit 2
fi

manifest_row() {
  local index="$1"
  awk -F '\t' 'BEGIN {i=0} /^[[:space:]]*#/ {next} NF >= 3 {if (i == idx) {print; exit} i++}' idx="${index}" "${MANIFEST}"
}

roundup_z_dim() {
  local obs_dim="$1"
  local z_dim=$(( ((4 * obs_dim + 63) / 64) * 64 ))
  if (( z_dim < 320 )); then
    z_dim=320
  fi
  printf '%s' "${z_dim}"
}

TASKS=()
add_model_rows_for_task() {
  local task_index="$1"
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
  TASKS+=("${task_id}|${dataset}|${horizons}|dense_tanh_lr5em4_wd0|dense|tanh|0|0.05|2|5e-4|0")
  TASKS+=("${task_id}|${dataset}|${horizons}|sparse_mlp_relu_sp0p003_lr5em4_wd0|dense|relu|0.003|0.05|2|5e-4|0")
  TASKS+=("${task_id}|${dataset}|${horizons}|sparse_mlp_relu_sp0p01_lr5em4_wd0|dense|relu|0.01|0.05|2|5e-4|0")
  TASKS+=("${task_id}|${dataset}|${horizons}|lista_relu_a0p01_sp1em4_std|lista|relu|1e-4|0.01|2|3e-4|1e-4")
}

IFS=',' read -r -a TASK_INDEX_ARRAY <<< "${TASK_INDICES}"
for task_index in "${TASK_INDEX_ARRAY[@]}"; do
  add_model_rows_for_task "${task_index}"
done

TASK_COUNT="${#TASKS[@]}"
PACK_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_TASK=$((PACK_TASK_ID * PACK_SIZE))
PACKED_TASK_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / PACK_CONCURRENCY ))
if (( PACKED_TASK_THREADS < 1 )); then
  PACKED_TASK_THREADS=1
fi

if (( START_TASK >= TASK_COUNT )); then
  echo "No packed tasks for array id ${PACK_TASK_ID}; task_count=${TASK_COUNT}, start_task=${START_TASK}."
  exit 0
fi

gpu_guard_assert_cuda_visible "ManiSkill-10 5k packed smoke"
mkdir -p "${RUN_ROOT}/gpu_telemetry"
gpu_guard_print_context "ManiSkill-10 5k packed smoke"
gpu_guard_start_sampler "${RUN_ROOT}/gpu_telemetry/job_${SLURM_JOB_ID:-local}_${PACK_TASK_ID}.csv" 5

echo "date=$(date)"
echo "host=$(hostname)"
echo "commit=$(git rev-parse --short HEAD || true)"
echo "slurm_job_id=${SLURM_JOB_ID:-}"
echo "slurm_array_task_id=${PACK_TASK_ID}"
echo "manifest=${MANIFEST}"
echo "data_root=${DATA_ROOT}"
echo "run_root=${RUN_ROOT}"
echo "task_indices=${TASK_INDICES}"
echo "task_count=${TASK_COUNT}"
echo "pack_size=${PACK_SIZE}"
echo "pack_concurrency=${PACK_CONCURRENCY}"
echo "packed_task_threads=${PACKED_TASK_THREADS}"
echo "num_steps=${NUM_STEPS}"
echo "batch_size=${BATCH_SIZE}"
echo "periodic_reencode_periods=${PERIODIC_REENCODE_PERIODS}"
echo "run_eval_in_gpu_job=${RUN_EVAL_IN_GPU_JOB}"
echo "eval_device=${EVAL_DEVICE}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"

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

run_one_task() {
  local global_task_id="$1"
  IFS='|' read -r TASK_ID DATASET HORIZONS RUN_TAG ENCODER_KIND ACTIVATION SPARSITY_WEIGHT LISTA_ALPHA LISTA_LOOPS LR WEIGHT_DECAY <<< "${TASKS[${global_task_id}]}"
  local summary="${DATASET}.summary.json"
  if [[ ! -f "${summary}" ]]; then
    echo "Missing dataset summary: ${summary}" >&2
    return 2
  fi
  local obs_dim z_dim run_dir eval_dir
  obs_dim="$(uv run python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["obs_dim"])' "${summary}")"
  z_dim="$(roundup_z_dim "${obs_dim}")"
  run_dir="${RUN_ROOT}/${TASK_ID}/${RUN_TAG}/seed${SEED}"
  eval_dir="${run_dir}/eval_test_periodic"
  mkdir -p "${run_dir}" "${eval_dir}"

  if [[ "${ENCODER_KIND}" == "dense" && "${SPARSITY_WEIGHT}" == "0" && "${ACTIVATION}" != "tanh" ]]; then
    echo "Dense no-sparsity baseline must use tanh, got ${ACTIVATION} for ${RUN_TAG}" >&2
    return 2
  fi

  echo "----- task ${global_task_id}/${TASK_COUNT}: ${TASK_ID} ${RUN_TAG} seed${SEED} -----"
  echo "dataset=${DATASET}"
  echo "obs_dim=${obs_dim} z_dim=${z_dim} horizons=${HORIZONS}"
  echo "encoder=${ENCODER_KIND} activation=${ACTIVATION} sparsity=${SPARSITY_WEIGHT} alpha=${LISTA_ALPHA} loops=${LISTA_LOOPS}"
  echo "optimizer lr=${LR} weight_decay=${WEIGHT_DECAY}"

  if [[ "${FORCE:-0}" != "1" && -f "${eval_dir}/metrics_summary.json" ]]; then
    echo "Existing eval summary found; skipping ${TASK_ID}/${RUN_TAG}."
    return 0
  fi

  if [[ "${FORCE:-0}" == "1" || ! -f "${run_dir}/checkpoint.pt" ]]; then
    uv run python tools/train_maniskill_controlled_lista.py \
      --dataset "${DATASET}" \
      --run_dir "${run_dir}" \
      --seed "${SEED}" \
      --encoder_kind "${ENCODER_KIND}" \
      --activation "${ACTIVATION}" \
      --num_steps "${NUM_STEPS}" \
      --batch_size "${BATCH_SIZE}" \
      --sequence_length "${SEQUENCE_LENGTH}" \
      --z_dim "${z_dim}" \
      --hidden_dim "${HIDDEN_DIM}" \
      --num_hidden_layers "${NUM_HIDDEN_LAYERS}" \
      --lista_loops "${LISTA_LOOPS}" \
      --lista_alpha "${LISTA_ALPHA}" \
      --lr "${LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --eval_every "${EVAL_EVERY}" \
      --sparsity_weight "${SPARSITY_WEIGHT}" \
      --device cuda
  fi

  if [[ "${RUN_EVAL_IN_GPU_JOB}" != "1" ]]; then
    echo "Skipping evaluation in GPU allocation; submit CPU evaluation for checkpoint=${run_dir}/checkpoint.pt"
    return 0
  fi
  if [[ "${EVAL_DEVICE}" == "cuda" && "${ALLOW_GPU_EVAL:-0}" != "1" ]]; then
    echo "Refusing CUDA evaluation in GPU training job without ALLOW_GPU_EVAL=1; use CPU evaluation to avoid GPU waste." >&2
    return 2
  fi

  uv run python tools/evaluate_maniskill_controlled_lista.py \
    --dataset "${DATASET}" \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --output_dir "${eval_dir}" \
    --device "${EVAL_DEVICE}" \
    --split test \
    --horizons "${HORIZONS}" \
    --periodic_reencode_periods "${PERIODIC_REENCODE_PERIODS}" \
    --support_threshold "${SUPPORT_THRESHOLD}" \
    --family_jaccard "${FAMILY_JACCARD}"

  echo "summary=${eval_dir}/metrics_summary.json"
}

for ((PACK_INDEX = 0; PACK_INDEX < PACK_SIZE; PACK_INDEX++)); do
  GLOBAL_TASK_ID=$((START_TASK + PACK_INDEX))
  if (( GLOBAL_TASK_ID >= TASK_COUNT )); then
    echo "No task for global task ${GLOBAL_TASK_ID}; stopping pack."
    break
  fi
  (
    export OMP_NUM_THREADS="${PACKED_TASK_THREADS}"
    export MKL_NUM_THREADS="${PACKED_TASK_THREADS}"
    export NUMEXPR_NUM_THREADS="${PACKED_TASK_THREADS}"
    run_one_task "${GLOBAL_TASK_ID}"
  ) &
  RUNNING_TASKS=$((RUNNING_TASKS + 1))
  if (( RUNNING_TASKS >= PACK_CONCURRENCY )); then
    wait_for_packed_slot
  fi
done

while (( RUNNING_TASKS > 0 )); do
  wait_for_packed_slot
done

gpu_guard_stop_sampler
echo "end_time=$(date)"
exit "${FAILED_TASKS}"
