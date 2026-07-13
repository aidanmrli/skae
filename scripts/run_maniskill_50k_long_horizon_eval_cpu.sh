#!/usr/bin/env bash
# Focused long-horizon evaluation for completed 50k ManiSkill checkpoints.
# The existing perturbation e20 packet stores at most ~220 transitions, so
# horizons above 220 require regenerating longer rollouts rather than only
# reevaluating these checkpoints.
#SBATCH --job-name=mskill50k_long_eval
#SBATCH --partition=long
#SBATCH --output=logs/maniskill_50k_long_eval_%A_%a.out
#SBATCH --error=logs/maniskill_50k_long_eval_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-11%6

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"
export CUDA_VISIBLE_DEVICES=""

DATASET="${DATASET:-data/maniskill/perturbation_assessment_seed0_e20/all_setups.npz}"
RUN_ROOT="${RUN_ROOT:-runs/maniskill_insertion/perturbation_e20_50k_cpu_20260603}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/maniskill_insertion/perturbation_e20_50k_long_eval_20260603}"
HORIZONS="${HORIZONS:-10,25,50,100,125,150,175,200,220}"
PERIODIC_REENCODE_PERIODS="${PERIODIC_REENCODE_PERIODS:-}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-0.2}"
FAMILY_JACCARD="${FAMILY_JACCARD:-0.4}"
SETTINGS_CSV="${SETTINGS_CSV:-dense_tanh_sp0,lista_a0p03_sp0p003,lista_a0p03_sp0p01,sparse_mlp_sp0p003}"

IFS=',' read -r -a SETTINGS <<< "${SETTINGS_CSV}"
SEEDS=(0 1 2)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SETTING_INDEX=$((TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((TASK_ID % ${#SEEDS[@]}))

if (( SETTING_INDEX < 0 || SETTING_INDEX >= ${#SETTINGS[@]} )); then
  echo "Invalid setting index ${SETTING_INDEX} from task ${TASK_ID}" >&2
  exit 2
fi

SETTING="${SETTINGS[${SETTING_INDEX}]}"
SEED="${SEEDS[${SEED_INDEX}]}"
CHECKPOINT="${RUN_ROOT}/${SETTING}/seed${SEED}/checkpoint.pt"
OUTPUT_DIR="${OUTPUT_ROOT}/${SETTING}/seed${SEED}/eval_test_long"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

echo "date=$(date)"
echo "host=$(hostname)"
echo "commit=$(git rev-parse --short HEAD || true)"
echo "slurm_job_id=${SLURM_JOB_ID:-}"
echo "slurm_array_task_id=${TASK_ID}"
echo "dataset=${DATASET}"
echo "checkpoint=${CHECKPOINT}"
echo "output_dir=${OUTPUT_DIR}"
echo "setting=${SETTING}"
echo "seed=${SEED}"
echo "horizons=${HORIZONS}"
echo "periodic_reencode_periods=${PERIODIC_REENCODE_PERIODS}"
echo "support_threshold=${SUPPORT_THRESHOLD}"
echo "family_jaccard=${FAMILY_JACCARD}"

uv run python tools/evaluate_maniskill_controlled_lista.py \
  --dataset "${DATASET}" \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --device cpu \
  --split test \
  --horizons "${HORIZONS}" \
  --periodic_reencode_periods "${PERIODIC_REENCODE_PERIODS}" \
  --support_threshold "${SUPPORT_THRESHOLD}" \
  --family_jaccard "${FAMILY_JACCARD}"

echo "summary=${OUTPUT_DIR}/metrics_summary.json"
