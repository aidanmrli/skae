#!/usr/bin/env bash
# Matched 50k-step ManiSkill controlled-Koopman sweep.
#
# GPU audit: this queue is intentionally CPU-only. The controlled ManiSkill
# models are small MLPs (<1M parameters with the default z_dim=320), prior
# 2k-step CPU jobs completed quickly, and one run per GPU would underutilize
# accelerator memory and compute. If future runs use GPUs, pack concurrent
# workers and add nvidia-smi utilization monitoring.
#SBATCH --job-name=mskill50k_cpu
#SBATCH --partition=long
#SBATCH --output=logs/maniskill_50k_cpu_%A_%a.out
#SBATCH --error=logs/maniskill_50k_cpu_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-59%12

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

# Keep this job CPU-only even if it lands on a node with visible CUDA devices.
export CUDA_VISIBLE_DEVICES=""

DATASET="${DATASET:-data/maniskill/perturbation_assessment_seed0_e20/all_setups.npz}"
RUN_ROOT="${RUN_ROOT:-runs/maniskill_insertion/perturbation_e20_50k_cpu_20260603}"
NUM_STEPS="${NUM_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-10}"
Z_DIM="${Z_DIM:-320}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-2}"
LISTA_LOOPS="${LISTA_LOOPS:-2}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-0.2}"
FAMILY_JACCARD="${FAMILY_JACCARD:-0.4}"
HORIZONS="${HORIZONS:-10,25,50,100}"

float_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  echo "${value}"
}

TASKS=()
for seed in 0 1 2; do
  TASKS+=("dense|dense|0|0.05|${seed}|dense_tanh_sp0")

  for sparsity in 0.003 0.01 0.03; do
    sp_tag="$(float_tag "${sparsity}")"
    TASKS+=("sparse_mlp|dense|${sparsity}|0.05|${seed}|sparse_mlp_sp${sp_tag}")
  done

  for alpha in 0.03 0.05 0.1 0.2; do
    alpha_tag="$(float_tag "${alpha}")"
    for sparsity in 0 0.003 0.01 0.03; do
      sp_tag="$(float_tag "${sparsity}")"
      TASKS+=("lista|lista|${sparsity}|${alpha}|${seed}|lista_a${alpha_tag}_sp${sp_tag}")
    done
  done
done

TASK_COUNT="${#TASKS[@]}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK_COUNT}" -ne 60 ]]; then
  echo "Internal task table error: expected 60 tasks, got ${TASK_COUNT}" >&2
  exit 2
fi
if (( TASK_ID < 0 || TASK_ID >= TASK_COUNT )); then
  echo "Invalid task id ${TASK_ID}; task count is ${TASK_COUNT}" >&2
  exit 2
fi

IFS='|' read -r MODEL_FAMILY ENCODER_KIND SPARSITY_WEIGHT LISTA_ALPHA SEED RUN_TAG <<< "${TASKS[${TASK_ID}]}"
RUN_DIR="${RUN_ROOT}/${RUN_TAG}/seed${SEED}"

mkdir -p "${RUN_DIR}"

echo "date=$(date)"
echo "host=$(hostname)"
echo "commit=$(git rev-parse --short HEAD || true)"
echo "slurm_job_id=${SLURM_JOB_ID:-}"
echo "slurm_array_task_id=${TASK_ID}"
echo "dataset=${DATASET}"
echo "run_dir=${RUN_DIR}"
echo "model_family=${MODEL_FAMILY}"
echo "encoder_kind=${ENCODER_KIND}"
echo "activation_policy=auto_dense_baseline_tanh"
echo "seed=${SEED}"
echo "num_steps=${NUM_STEPS}"
echo "batch_size=${BATCH_SIZE}"
echo "sequence_length=${SEQUENCE_LENGTH}"
echo "z_dim=${Z_DIM}"
echo "hidden_dim=${HIDDEN_DIM}"
echo "num_hidden_layers=${NUM_HIDDEN_LAYERS}"
echo "lista_alpha=${LISTA_ALPHA}"
echo "sparsity_weight=${SPARSITY_WEIGHT}"
echo "support_threshold=${SUPPORT_THRESHOLD}"
echo "family_jaccard=${FAMILY_JACCARD}"
echo "horizons=${HORIZONS}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

uv run python tools/train_maniskill_controlled_lista.py \
  --dataset "${DATASET}" \
  --run_dir "${RUN_DIR}" \
  --seed "${SEED}" \
  --encoder_kind "${ENCODER_KIND}" \
  --num_steps "${NUM_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --sequence_length "${SEQUENCE_LENGTH}" \
  --z_dim "${Z_DIM}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --num_hidden_layers "${NUM_HIDDEN_LAYERS}" \
  --lista_loops "${LISTA_LOOPS}" \
  --lista_alpha "${LISTA_ALPHA}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --eval_every "${EVAL_EVERY}" \
  --sparsity_weight "${SPARSITY_WEIGHT}" \
  --device cpu

uv run python tools/evaluate_maniskill_controlled_lista.py \
  --dataset "${DATASET}" \
  --checkpoint "${RUN_DIR}/checkpoint.pt" \
  --output_dir "${RUN_DIR}/eval_test" \
  --device cpu \
  --split test \
  --horizons "${HORIZONS}" \
  --support_threshold "${SUPPORT_THRESHOLD}" \
  --family_jaccard "${FAMILY_JACCARD}"

echo "summary=${RUN_DIR}/eval_test/metrics_summary.json"
