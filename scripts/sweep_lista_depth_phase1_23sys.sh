#!/bin/bash
#
# Phase-1 depth sweep over 23 systems x 3 seeds x 5 depths = 345 runs.
#
# Submit:
#   sbatch scripts/sweep_lista_depth_phase1_23sys.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/lista_depth_first_23sys
#   NUM_STEPS=10000 BATCH_SIZE=256 TARGET_SIZE=256
#   SPARSITY_COEFF=0.10 LISTA_ALPHA=0.15
#   DYSTS_CACHE_PROFILE=full
#   DYSTS_CACHE_DIR=/network/scratch/l/lia/skae/dysts_native_cache
#   DYSTS_CACHE_NUM_WORKERS=4 DYSTS_CACHE_REUSE=1
#
#SBATCH --job-name=lista_d1_23sys
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/lista-d1-23sys-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-344

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_depth_first_23sys}"
PHASE1_ROOT="${PHASE1_ROOT:-${BASE_OUT}/phase1_depth}"
mkdir -p "${PHASE1_ROOT}"

NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.10}"
LISTA_ALPHA="${LISTA_ALPHA:-0.15}"
RECONST_COEFF="${RECONST_COEFF:-0.5}"
PRED_COEFF="${PRED_COEFF:-1.0}"
DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE:-full}"
DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS:-4}"
DYSTS_CACHE_REUSE="${DYSTS_CACHE_REUSE:-1}"

DEPTHS=(1 2 3 5 7)

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

NUM_DEPTHS=${#DEPTHS[@]}
NUM_SYSTEMS=${#SYSTEM_KEYS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_DEPTHS * NUM_SYSTEMS * NUM_SEEDS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_JOBS )); then
  echo "Task ${TASK_ID} out of range for TOTAL_JOBS=${TOTAL_JOBS}. Exiting."
  exit 0
fi

DEPTH_STRIDE=$((NUM_SYSTEMS * NUM_SEEDS))
DEPTH_IDX=$((TASK_ID / DEPTH_STRIDE))
REM=$((TASK_ID % DEPTH_STRIDE))
SYSTEM_IDX=$((REM / NUM_SEEDS))
SEED_IDX=$((REM % NUM_SEEDS))

DEPTH=${DEPTHS[$DEPTH_IDX]}
SYSTEM=${SYSTEM_KEYS[$SYSTEM_IDX]}
SYSTEM_LABEL=${SYSTEM_LABELS[$SYSTEM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

LOG_DIR="${PHASE1_ROOT}/depth_${DEPTH}/${SYSTEM_LABEL}/seed_${SEED}"

echo "============================================="
echo "LISTA Depth-First Phase-1 Sweep (23 systems)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "Depth: ${DEPTH}"
echo "System: ${SYSTEM}"
echo "System Label: ${SYSTEM_LABEL}"
echo "Seed: ${SEED}"
echo "Start Time: $(date)"
echo "PHASE1_ROOT: ${PHASE1_ROOT}"
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
  --lista_num_loops "${DEPTH}"
  --lista_final_op relu
  --sequence_length 1
  --seed "${SEED}"
  --device cuda
  --log_dir "${LOG_DIR}"
)

if [[ "${SYSTEM}" == dysts:* ]]; then
  TRAIN_ARGS+=(
    --standardize
    --dysts_ic_noise_scale 0.2
    --dysts_native_cache
    --dysts_cache_profile "${DYSTS_CACHE_PROFILE}"
    --dysts_cache_dir "${DYSTS_CACHE_DIR}"
    --dysts_cache_num_workers "${DYSTS_CACHE_NUM_WORKERS}"
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
