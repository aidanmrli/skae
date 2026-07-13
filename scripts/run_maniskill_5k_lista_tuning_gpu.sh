#!/usr/bin/env bash
# Packed 5k-step ManiSkill LISTA tuning pilot with periodic re-encoding eval.
#
# GPU audit: individual controlled-ManiSkill MLP/LISTA models are too small for
# one run per accelerator. This launcher therefore packs concurrent workers on a
# single GPU and writes nvidia-smi telemetry for posthoc utilization checks.
# Keep PACK_CONCURRENCY >= 2 unless BATCH_SIZE is made large enough to fill the
# GPU with a single process.
#SBATCH --job-name=mskill5k_gpu
#SBATCH --partition=long
#SBATCH --output=logs/maniskill_5k_gpu_%A_%a.out
#SBATCH --error=logs/maniskill_5k_gpu_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%1
#SBATCH --requeue

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs

source scripts/slurm_gpu_guard.sh

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

DATASET="${DATASET:-data/maniskill/perturbation_assessment_seed0_e20/all_setups.npz}"
RUN_ROOT="${RUN_ROOT:-runs/maniskill_insertion/perturbation_e20_5k_lista_tuning_${SLURM_ARRAY_JOB_ID:-local}}"
TASK_SET="${TASK_SET:-initial}"
SEEDS="${SEEDS:-0}"
PACK_SIZE="${PACK_SIZE:-4}"
PACK_CONCURRENCY="${PACK_CONCURRENCY:-4}"
NUM_STEPS="${NUM_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-10}"
Z_DIM="${Z_DIM:-320}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-2}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-0.2}"
FAMILY_JACCARD="${FAMILY_JACCARD:-0.4}"
HORIZONS="${HORIZONS:-10,20,30,40,50,75,100,125}"
PERIODIC_REENCODE_PERIODS="${PERIODIC_REENCODE_PERIODS:-10,20,50,100}"

if (( PACK_SIZE <= 0 )); then
  echo "PACK_SIZE must be positive, got ${PACK_SIZE}" >&2
  exit 2
fi
if (( PACK_CONCURRENCY <= 0 )); then
  echo "PACK_CONCURRENCY must be positive, got ${PACK_CONCURRENCY}" >&2
  exit 2
fi
if (( PACK_CONCURRENCY < 2 && BATCH_SIZE < 2048 )); then
  echo "Refusing likely GPU-wasteful config: PACK_CONCURRENCY=${PACK_CONCURRENCY}, BATCH_SIZE=${BATCH_SIZE}" >&2
  exit 2
fi

TASKS=()

add_task() {
  local tag="$1"
  local encoder="$2"
  local activation="$3"
  local sparsity="$4"
  local alpha="$5"
  local loops="$6"
  local seed="$7"
  local pred_weight="$8"
  local recon_weight="$9"
  local latent_weight="${10}"
  local k_weight="${11}"
  local task_lr="${12:-${LR}}"
  local task_weight_decay="${13:-${WEIGHT_DECAY}}"
  TASKS+=("${tag}|${encoder}|${activation}|${sparsity}|${alpha}|${loops}|${seed}|${pred_weight}|${recon_weight}|${latent_weight}|${k_weight}|${task_lr}|${task_weight_decay}")
}

build_initial_tasks() {
  local seed
  for seed in ${SEEDS//,/ }; do
    add_task "dense_tanh_sp0" "dense" "tanh" "0" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"

    add_task "lista_tanh_a0p001_sp0" "lista" "tanh" "0" "0.001" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_tanh_a0p001_sp1em5" "lista" "tanh" "1e-5" "0.001" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_tanh_a0p001_sp1em4" "lista" "tanh" "1e-4" "0.001" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_tanh_a0p003_sp0" "lista" "tanh" "0" "0.003" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_tanh_a0p003_sp1em5" "lista" "tanh" "1e-5" "0.003" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_tanh_a0p003_sp1em4" "lista" "tanh" "1e-4" "0.003" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_tanh_a0p01_sp0" "lista" "tanh" "0" "0.01" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_tanh_a0p01_sp1em4" "lista" "tanh" "1e-4" "0.01" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_tanh_a0p03_sp1em4" "lista" "tanh" "1e-4" "0.03" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"

    add_task "lista_relu_a0p001_sp0" "lista" "relu" "0" "0.001" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a0p001_sp1em4" "lista" "relu" "1e-4" "0.001" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a0p003_sp0" "lista" "relu" "0" "0.003" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a0p003_sp1em4" "lista" "relu" "1e-4" "0.003" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a0p01_sp1em4" "lista" "relu" "1e-4" "0.01" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"

    add_task "lista_tanh_a0p003_sp1em4_loops1" "lista" "tanh" "1e-4" "0.003" "1" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_tanh_a0p003_sp1em4_loops3" "lista" "tanh" "1e-4" "0.003" "3" "${seed}" "1.0" "0.1" "0.1" "1e-4"
  done
}

build_weight_tasks() {
  local seed
  for seed in ${SEEDS//,/ }; do
    add_task "dense_tanh_sp0" "dense" "tanh" "0" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_tanh_a0p001_sp0_pred2_rec0p03_lat0p03" "lista" "tanh" "0" "0.001" "2" "${seed}" "2.0" "0.03" "0.03" "1e-4"
    add_task "lista_tanh_a0p003_sp0_pred2_rec0p03_lat0p03" "lista" "tanh" "0" "0.003" "2" "${seed}" "2.0" "0.03" "0.03" "1e-4"
    add_task "lista_tanh_a0p003_sp1em4_pred2_rec0p03_lat0p03" "lista" "tanh" "1e-4" "0.003" "2" "${seed}" "2.0" "0.03" "0.03" "1e-4"
    add_task "lista_tanh_a0p001_sp0_pred5_rec0p03_lat0p01" "lista" "tanh" "0" "0.001" "2" "${seed}" "5.0" "0.03" "0.01" "1e-4"
    add_task "lista_tanh_a0p003_sp0_pred5_rec0p03_lat0p01" "lista" "tanh" "0" "0.003" "2" "${seed}" "5.0" "0.03" "0.01" "1e-4"
    add_task "lista_relu_a0p001_sp0_pred2_rec0p03_lat0p03" "lista" "relu" "0" "0.001" "2" "${seed}" "2.0" "0.03" "0.03" "1e-4"
    add_task "lista_relu_a0p003_sp0_pred2_rec0p03_lat0p03" "lista" "relu" "0" "0.003" "2" "${seed}" "2.0" "0.03" "0.03" "1e-4"
  done
}

build_confirm_best_tasks() {
  local seed
  for seed in ${SEEDS//,/ }; do
    add_task "dense_tanh_sp0" "dense" "tanh" "0" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a0p01_sp1em4" "lista" "relu" "1e-4" "0.01" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
  done
}

build_ultra_low_alpha_tasks() {
  local seed
  for seed in ${SEEDS//,/ }; do
    add_task "dense_tanh_sp0" "dense" "tanh" "0" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"

    add_task "lista_relu_a1em5_sp0" "lista" "relu" "0" "1e-5" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a1em5_sp1em6" "lista" "relu" "1e-6" "1e-5" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a1em5_sp1em5" "lista" "relu" "1e-5" "1e-5" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a1em4_sp0" "lista" "relu" "0" "1e-4" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a1em4_sp1em6" "lista" "relu" "1e-6" "1e-4" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a1em4_sp1em5" "lista" "relu" "1e-5" "1e-4" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a3em4_sp0" "lista" "relu" "0" "3e-4" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a3em4_sp1em5" "lista" "relu" "1e-5" "3e-4" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"

    add_task "lista_relu_a1em4_sp0_loops1" "lista" "relu" "0" "1e-4" "1" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a1em4_sp0_loops3" "lista" "relu" "0" "1e-4" "3" "${seed}" "1.0" "0.1" "0.1" "1e-4"

    add_task "lista_relu_a1em5_sp0_pred2_rec0p03_lat0p03" "lista" "relu" "0" "1e-5" "2" "${seed}" "2.0" "0.03" "0.03" "1e-4"
    add_task "lista_relu_a1em4_sp0_pred2_rec0p03_lat0p03" "lista" "relu" "0" "1e-4" "2" "${seed}" "2.0" "0.03" "0.03" "1e-4"
    add_task "lista_relu_a1em4_sp0_pred1_rec0p03_lat0p03" "lista" "relu" "0" "1e-4" "2" "${seed}" "1.0" "0.03" "0.03" "1e-4"
  done
}

build_confirm_ultra_low_best_tasks() {
  local seed
  for seed in ${SEEDS//,/ }; do
    add_task "dense_tanh_sp0" "dense" "tanh" "0" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a1em4_sp1em6" "lista" "relu" "1e-6" "1e-4" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a1em4_sp0_pred1_rec0p03_lat0p03" "lista" "relu" "0" "1e-4" "2" "${seed}" "1.0" "0.03" "0.03" "1e-4"
    add_task "lista_relu_a0p01_sp1em4" "lista" "relu" "1e-4" "0.01" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
  done
}

build_forecast_compare_tasks() {
  local seed
  for seed in ${SEEDS//,/ }; do
    add_task "dense_tanh_sp0" "dense" "tanh" "0" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "sparse_mlp_relu_sp1em4" "dense" "relu" "1e-4" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "sparse_mlp_relu_sp0p003" "dense" "relu" "0.003" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "sparse_mlp_relu_sp0p01" "dense" "relu" "0.01" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a0p01_sp1em4" "lista" "relu" "1e-4" "0.01" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "lista_relu_a1em4_sp1em6" "lista" "relu" "1e-6" "1e-4" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
  done
}

tagify_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//+/p}"
  printf '%s' "${value}"
}

build_lista_forecast_refine_tasks() {
  local seed
  local alpha
  local sparsity
  local loops
  local alpha_tag
  local sparsity_tag
  for seed in ${SEEDS//,/ }; do
    add_task "dense_tanh_sp0" "dense" "tanh" "0" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"
    add_task "sparse_mlp_relu_sp0p01" "dense" "relu" "0.01" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4"

    for loops in 1 2 3; do
      for alpha in 0.005 0.0075 0.01 0.015 0.02; do
        alpha_tag="$(tagify_float "${alpha}")"
        for sparsity in 1e-5 3e-5 1e-4 3e-4 1e-3; do
          sparsity_tag="$(tagify_float "${sparsity}")"
          add_task "lista_relu_a${alpha_tag}_sp${sparsity_tag}_loops${loops}" \
            "lista" "relu" "${sparsity}" "${alpha}" "${loops}" "${seed}" \
            "1.0" "0.1" "0.1" "1e-4"
        done
      done
    done
  done
}

build_lista_forecast_optimizer_refine_tasks() {
  local seed
  local lr
  local weight_decay
  local k_weight
  local lr_tag
  local wd_tag
  local k_tag
  for seed in ${SEEDS//,/ }; do
    add_task "dense_tanh_sp0" "dense" "tanh" "0" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4" "3e-4" "1e-4"
    add_task "sparse_mlp_relu_sp0p01" "dense" "relu" "0.01" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4" "3e-4" "1e-4"

    for lr in 1e-4 2e-4 3e-4 5e-4 7e-4; do
      lr_tag="$(tagify_float "${lr}")"
      for weight_decay in 0 1e-5 1e-4; do
        wd_tag="$(tagify_float "${weight_decay}")"
        for k_weight in 0 1e-5 1e-4; do
          k_tag="$(tagify_float "${k_weight}")"
          add_task "lista_relu_a0p01_sp1em4_loops2_lr${lr_tag}_wd${wd_tag}_k${k_tag}" \
            "lista" "relu" "1e-4" "0.01" "2" "${seed}" \
            "1.0" "0.1" "0.1" "${k_weight}" "${lr}" "${weight_decay}"
        done
      done
    done
  done
}

build_optimizer_fairness_tasks() {
  local seed
  for seed in ${SEEDS//,/ }; do
    add_task "dense_tanh_sp0_std" "dense" "tanh" "0" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4" "3e-4" "1e-4"
    add_task "dense_tanh_sp0_lr5em4_wd0" "dense" "tanh" "0" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4" "5e-4" "0"

    add_task "sparse_mlp_relu_sp0p01_std" "dense" "relu" "0.01" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4" "3e-4" "1e-4"
    add_task "sparse_mlp_relu_sp0p01_lr5em4_wd0" "dense" "relu" "0.01" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4" "5e-4" "0"
    add_task "sparse_mlp_relu_sp0p003_std" "dense" "relu" "0.003" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4" "3e-4" "1e-4"
    add_task "sparse_mlp_relu_sp0p003_lr5em4_wd0" "dense" "relu" "0.003" "0.05" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4" "5e-4" "0"

    add_task "lista_relu_a0p01_sp1em4_loops2_std" "lista" "relu" "1e-4" "0.01" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4" "3e-4" "1e-4"
    add_task "lista_relu_a0p01_sp1em4_loops2_lr5em4_wd0" "lista" "relu" "1e-4" "0.01" "2" "${seed}" "1.0" "0.1" "0.1" "1e-4" "5e-4" "0"
  done
}

case "${TASK_SET}" in
  initial)
    build_initial_tasks
    ;;
  weights)
    build_weight_tasks
    ;;
  confirm_best)
    build_confirm_best_tasks
    ;;
  ultra_low_alpha)
    build_ultra_low_alpha_tasks
    ;;
  confirm_ultra_low_best)
    build_confirm_ultra_low_best_tasks
    ;;
  forecast_compare)
    build_forecast_compare_tasks
    ;;
  lista_forecast_refine)
    build_lista_forecast_refine_tasks
    ;;
  lista_forecast_optimizer_refine)
    build_lista_forecast_optimizer_refine_tasks
    ;;
  optimizer_fairness)
    build_optimizer_fairness_tasks
    ;;
  *)
    echo "Unknown TASK_SET=${TASK_SET}; expected initial, weights, confirm_best, ultra_low_alpha, confirm_ultra_low_best, forecast_compare, lista_forecast_refine, lista_forecast_optimizer_refine, or optimizer_fairness" >&2
    exit 2
    ;;
esac

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

gpu_guard_assert_cuda_visible "ManiSkill 5k packed tuning"
mkdir -p "${RUN_ROOT}/gpu_telemetry"
gpu_guard_print_context "ManiSkill 5k packed tuning"
gpu_guard_start_sampler "${RUN_ROOT}/gpu_telemetry/job_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.csv" 5

echo "date=$(date)"
echo "host=$(hostname)"
echo "commit=$(git rev-parse --short HEAD || true)"
echo "slurm_job_id=${SLURM_JOB_ID:-}"
echo "slurm_array_task_id=${PACK_TASK_ID}"
echo "task_set=${TASK_SET}"
echo "task_count=${TASK_COUNT}"
echo "start_task=${START_TASK}"
echo "pack_size=${PACK_SIZE}"
echo "pack_concurrency=${PACK_CONCURRENCY}"
echo "packed_task_threads=${PACKED_TASK_THREADS}"
echo "dataset=${DATASET}"
echo "run_root=${RUN_ROOT}"
echo "num_steps=${NUM_STEPS}"
echo "batch_size=${BATCH_SIZE}"
echo "sequence_length=${SEQUENCE_LENGTH}"
echo "z_dim=${Z_DIM}"
echo "hidden_dim=${HIDDEN_DIM}"
echo "horizons=${HORIZONS}"
echo "periodic_reencode_periods=${PERIODIC_REENCODE_PERIODS}"
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
  IFS='|' read -r RUN_TAG ENCODER_KIND ACTIVATION SPARSITY_WEIGHT LISTA_ALPHA LISTA_LOOPS SEED PREDICTION_WEIGHT RECONSTRUCTION_WEIGHT LATENT_WEIGHT K_STABILITY_WEIGHT TASK_LR TASK_WEIGHT_DECAY <<< "${TASKS[${global_task_id}]}"
  local run_dir="${RUN_ROOT}/${RUN_TAG}/seed${SEED}"
  local eval_dir="${run_dir}/eval_test_periodic"
  mkdir -p "${run_dir}" "${eval_dir}"

  if [[ "${ENCODER_KIND}" == "dense" && "${SPARSITY_WEIGHT}" == "0" && "${ACTIVATION}" != "tanh" ]]; then
    echo "Dense no-sparsity baselines must use tanh activations, got activation=${ACTIVATION} for ${RUN_TAG}." >&2
    return 2
  fi

  echo "----- task ${global_task_id}/${TASK_COUNT}: ${RUN_TAG} seed${SEED} -----"
  echo "encoder_kind=${ENCODER_KIND} activation=${ACTIVATION} alpha=${LISTA_ALPHA} sparsity=${SPARSITY_WEIGHT} loops=${LISTA_LOOPS}"
  echo "weights pred=${PREDICTION_WEIGHT} rec=${RECONSTRUCTION_WEIGHT} latent=${LATENT_WEIGHT} k=${K_STABILITY_WEIGHT}"
  echo "optimizer lr=${TASK_LR} weight_decay=${TASK_WEIGHT_DECAY}"
  echo "run_dir=${run_dir}"

  if [[ "${FORCE:-0}" != "1" && -f "${eval_dir}/metrics_summary.json" ]]; then
    echo "Existing eval summary found; skipping ${RUN_TAG} seed${SEED}."
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
      --z_dim "${Z_DIM}" \
      --hidden_dim "${HIDDEN_DIM}" \
      --num_hidden_layers "${NUM_HIDDEN_LAYERS}" \
      --lista_loops "${LISTA_LOOPS}" \
      --lista_alpha "${LISTA_ALPHA}" \
      --lr "${TASK_LR}" \
      --weight_decay "${TASK_WEIGHT_DECAY}" \
      --eval_every "${EVAL_EVERY}" \
      --prediction_weight "${PREDICTION_WEIGHT}" \
      --reconstruction_weight "${RECONSTRUCTION_WEIGHT}" \
      --latent_weight "${LATENT_WEIGHT}" \
      --sparsity_weight "${SPARSITY_WEIGHT}" \
      --k_stability_weight "${K_STABILITY_WEIGHT}" \
      --device cuda
  else
    echo "Existing checkpoint found; evaluating ${RUN_TAG} seed${SEED}."
  fi

  uv run python tools/evaluate_maniskill_controlled_lista.py \
    --dataset "${DATASET}" \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --output_dir "${eval_dir}" \
    --device cuda \
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
