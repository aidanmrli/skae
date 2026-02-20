#!/bin/bash
#
# Phase-2 sparsity sweep with fixed depth_star over the same 23-system set.
#
# Default mode is coarse Step-2A from the plan:
#   sparsity_coeff in {0.05, 0.10, 0.20, 0.40, 0.80}, alpha fixed at 0.15
#
# Alpha mode (Step-2B) can be enabled with:
#   PHASE2_MODE=alpha SPARSITY_STAR=0.20 sbatch --array=0-275 scripts/sweep_lista_sparsity_phase2_23sys.sh
#
# DEPTH_STAR must be provided:
#   DEPTH_STAR=3 sbatch scripts/sweep_lista_sparsity_phase2_23sys.sh
#
# Optional cache controls:
#   DYSTS_CACHE_DIR=/network/scratch/l/lia/skae/dysts_native_cache
#   DYSTS_CACHE_NUM_WORKERS=4 DYSTS_CACHE_REUSE=1
#
#SBATCH --job-name=lista_d2_23sys
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/lista-d2-23sys-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-344

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

if [[ -z "${DEPTH_STAR:-}" ]]; then
  echo "DEPTH_STAR is required. Example: DEPTH_STAR=3 sbatch $0"
  exit 1
fi

PHASE2_MODE="${PHASE2_MODE:-coarse}"
ALPHA_FIXED="${ALPHA_FIXED:-0.15}"
SPARSITY_STAR="${SPARSITY_STAR:-}"

CONFIG_TAGS=()
CONFIG_SPARSITY=()
CONFIG_ALPHA=()

if [[ "${PHASE2_MODE}" == "coarse" ]]; then
  SP_GRID=(0.05 0.10 0.20 0.40 0.80)
  for sp in "${SP_GRID[@]}"; do
    CONFIG_TAGS+=("sp${sp}_a${ALPHA_FIXED}")
    CONFIG_SPARSITY+=("${sp}")
    CONFIG_ALPHA+=("${ALPHA_FIXED}")
  done
elif [[ "${PHASE2_MODE}" == "alpha" ]]; then
  if [[ -z "${SPARSITY_STAR}" ]]; then
    echo "SPARSITY_STAR is required when PHASE2_MODE=alpha"
    exit 1
  fi
  ALPHA_GRID=(0.10 0.15 0.25 0.35)
  for alpha in "${ALPHA_GRID[@]}"; do
    CONFIG_TAGS+=("sp${SPARSITY_STAR}_a${alpha}")
    CONFIG_SPARSITY+=("${SPARSITY_STAR}")
    CONFIG_ALPHA+=("${alpha}")
  done
else
  echo "Unsupported PHASE2_MODE='${PHASE2_MODE}'. Use 'coarse' or 'alpha'."
  exit 1
fi

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_depth_first_23sys}"
PHASE2_ROOT="${PHASE2_ROOT:-${BASE_OUT}/phase2_sparsity}"
mkdir -p "${PHASE2_ROOT}"

NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
RECONST_COEFF="${RECONST_COEFF:-0.5}"
PRED_COEFF="${PRED_COEFF:-1.0}"
DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS:-4}"
DYSTS_CACHE_REUSE="${DYSTS_CACHE_REUSE:-1}"

SYSTEM_KEYS=(
  "dysts:Dadras"
  "dysts:Duffing"
  "dysts:QiChen"
  "dysts:Sakarya"
  "dysts:SprottTorus"
  "dysts:Chua"
  "dysts:MultiChua"
  "dysts:DequanLi"
  "dysts:LuChenCheng"
  "dysts:SanUmSrisuchinwong"
  "dysts:WangSun"
  "dysts:ShimizuMorioka"
  "dysts:LorenzCoupled"
  "dysts:RikitakeDynamo"
  "dysts:Hadley"
  "multiwell_gradient"
  "multiwell_rotational"
  "multiwell_energy"
  "multiwell_strong_transition"
  "multiwell_gradient_hd"
  "multiwell_rotational_hd"
  "multiwell_energy_hd"
  "multiwell_strong_transition_hd"
)

SYSTEM_LABELS=(
  "Dadras"
  "Duffing"
  "QiChen"
  "Sakarya"
  "SprottTorus"
  "Chua"
  "MultiChua"
  "DequanLi"
  "LuChenCheng"
  "SanUmSrisuchinwong"
  "WangSun"
  "ShimizuMorioka"
  "LorenzCoupled"
  "RikitakeDynamo"
  "Hadley"
  "multiwell_gradient"
  "multiwell_rotational"
  "multiwell_energy"
  "multiwell_strong_transition"
  "multiwell_gradient_hd"
  "multiwell_rotational_hd"
  "multiwell_energy_hd"
  "multiwell_strong_transition_hd"
)

SEEDS=(0 1 2)

NUM_CONFIGS=${#CONFIG_TAGS[@]}
NUM_SYSTEMS=${#SYSTEM_KEYS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_CONFIGS * NUM_SYSTEMS * NUM_SEEDS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_JOBS )); then
  echo "Task ${TASK_ID} out of range for TOTAL_JOBS=${TOTAL_JOBS}. Exiting."
  exit 0
fi

CFG_STRIDE=$((NUM_SYSTEMS * NUM_SEEDS))
CFG_IDX=$((TASK_ID / CFG_STRIDE))
REM=$((TASK_ID % CFG_STRIDE))
SYSTEM_IDX=$((REM / NUM_SEEDS))
SEED_IDX=$((REM % NUM_SEEDS))

CONFIG_TAG=${CONFIG_TAGS[$CFG_IDX]}
SPARSITY_COEFF=${CONFIG_SPARSITY[$CFG_IDX]}
LISTA_ALPHA=${CONFIG_ALPHA[$CFG_IDX]}
SYSTEM=${SYSTEM_KEYS[$SYSTEM_IDX]}
SYSTEM_LABEL=${SYSTEM_LABELS[$SYSTEM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

LOG_DIR="${PHASE2_ROOT}/${CONFIG_TAG}/${SYSTEM_LABEL}/seed_${SEED}"

echo "============================================="
echo "LISTA Depth-First Phase-2 Sweep (23 systems)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "Mode: ${PHASE2_MODE}"
echo "depth_star: ${DEPTH_STAR}"
echo "config_tag: ${CONFIG_TAG}"
echo "sparsity_coeff: ${SPARSITY_COEFF}"
echo "lista_alpha: ${LISTA_ALPHA}"
echo "System: ${SYSTEM}"
echo "Seed: ${SEED}"
echo "Start Time: $(date)"
echo "PHASE2_ROOT: ${PHASE2_ROOT}"
echo "LOG_DIR: ${LOG_DIR}"
echo "============================================="

TRAIN_ARGS=(
  --config lista_nonlinear
  --env "${SYSTEM}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --target_size "${TARGET_SIZE}"
  --reconst_coeff "${RECONST_COEFF}"
  --pred_coeff "${PRED_COEFF}"
  --sparsity_coeff "${SPARSITY_COEFF}"
  --lista_alpha "${LISTA_ALPHA}"
  --lista_num_loops "${DEPTH_STAR}"
  --lista_final_op relu
  --pairwise
  --seed "${SEED}"
  --device cuda
  --log_dir "${LOG_DIR}"
)

if [[ "${SYSTEM}" == dysts:* ]]; then
  TRAIN_ARGS+=(
    --standardize
    --dysts_ic_noise_scale 0.2
    --dysts_native_cache
    --dysts_cache_dir "${DYSTS_CACHE_DIR}"
    --dysts_cache_num_workers "${DYSTS_CACHE_NUM_WORKERS}"
    --dysts_cache_warmup 2000
  )
  if [[ "${DYSTS_CACHE_REUSE}" == "1" ]]; then
    TRAIN_ARGS+=(--dysts_cache_reuse)
  fi
fi

uv run python tools/train.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
