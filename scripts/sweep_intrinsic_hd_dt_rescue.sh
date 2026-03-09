#!/bin/bash
#
# Focused intrinsic-HD dt-rescue pilot for the unresolved high-dimensional systems.
#
# Default mode:
#   - systems: kuramoto, hopfield
#   - dt: 0.025, 0.0125
#   - num_steps: 20000
#   - seeds: 0,1,2
#   - generic_sparse sparsity: 0.0005, 0.0025
#   - lista_blockdiag sparsity: 0.0005, 0.0010
#   - lista alpha: 0.15
#   - lista loops: 1
#   - block size: 16
#
# Example submissions:
#   sbatch --export=ALL,MODEL_VARIANT=generic_sparse scripts/sweep_intrinsic_hd_dt_rescue.sh
#   sbatch --export=ALL,MODEL_VARIANT=lista_blockdiag scripts/sweep_intrinsic_hd_dt_rescue.sh
#
# If you override the grids, update --array to cover the expanded job count.
#
#SBATCH --job-name=hd_dt_rescue
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/hd-dt-rescue-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-0

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/intrinsic_hd_dt_rescue_${DATE_TAG}}"
MODEL_VARIANT="${MODEL_VARIANT:-generic_sparse}"

SYSTEMS_CSV="${SYSTEMS_CSV:-kuramoto,hopfield}"
ENV_DTS_CSV="${ENV_DTS_CSV:-0.025,0.0125}"
NUM_STEPS_CSV="${NUM_STEPS_CSV:-20000}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"

GENERIC_SPARSITY_COEFFS_CSV="${GENERIC_SPARSITY_COEFFS_CSV:-0.0005,0.0025}"
BLOCKDIAG_SPARSITY_COEFFS_CSV="${BLOCKDIAG_SPARSITY_COEFFS_CSV:-0.0005,0.0010}"
LISTA_ALPHAS_CSV="${LISTA_ALPHAS_CSV:-0.15}"
LISTA_NUM_LOOPS_CSV="${LISTA_NUM_LOOPS_CSV:-1}"
K_BLOCK_SIZES_CSV="${K_BLOCK_SIZES_CSV:-16}"

BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
KURAMOTO_NUM_OSCILLATORS="${KURAMOTO_NUM_OSCILLATORS:-}"
HOPFIELD_NUM_NEURONS="${HOPFIELD_NUM_NEURONS:-}"
HOPFIELD_NUM_PATTERNS="${HOPFIELD_NUM_PATTERNS:-}"
COMPETITIVE_LV_NUM_SPECIES="${COMPETITIVE_LV_NUM_SPECIES:-}"

RES_COEFF="${RES_COEFF:-1.0}"
RECONST_COEFF="${RECONST_COEFF:-0.03}"
PRED_COEFF="${PRED_COEFF:-1.0}"
LISTA_FINAL_OP="${LISTA_FINAL_OP:-relu}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a SYSTEMS <<< "${SYSTEMS_CSV}"
IFS=',' read -r -a ENV_DTS <<< "${ENV_DTS_CSV}"
IFS=',' read -r -a NUM_STEPS_LIST <<< "${NUM_STEPS_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a GENERIC_SPARSITY_COEFFS <<< "${GENERIC_SPARSITY_COEFFS_CSV}"
IFS=',' read -r -a BLOCKDIAG_SPARSITY_COEFFS <<< "${BLOCKDIAG_SPARSITY_COEFFS_CSV}"
IFS=',' read -r -a LISTA_ALPHAS <<< "${LISTA_ALPHAS_CSV}"
IFS=',' read -r -a LISTA_NUM_LOOPS_LIST <<< "${LISTA_NUM_LOOPS_CSV}"
IFS=',' read -r -a K_BLOCK_SIZES <<< "${K_BLOCK_SIZES_CSV}"

tagify() {
  local raw="$1"
  raw="${raw//-/m}"
  raw="${raw//./p}"
  echo "${raw}"
}

NUM_SYSTEMS=${#SYSTEMS[@]}
NUM_DTS=${#ENV_DTS[@]}
NUM_STEP_OPTIONS=${#NUM_STEPS_LIST[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_GENERIC_SPARSITY=${#GENERIC_SPARSITY_COEFFS[@]}
NUM_BLOCKDIAG_SPARSITY=${#BLOCKDIAG_SPARSITY_COEFFS[@]}
NUM_ALPHAS=${#LISTA_ALPHAS[@]}
NUM_LOOPS=${#LISTA_NUM_LOOPS_LIST[@]}
NUM_BLOCKS=${#K_BLOCK_SIZES[@]}

case "${MODEL_VARIANT}" in
  generic_sparse)
    CONFIG="generic_sparse"
    TOTAL_JOBS=$((NUM_SYSTEMS * NUM_DTS * NUM_STEP_OPTIONS * NUM_GENERIC_SPARSITY * NUM_SEEDS))
    ;;
  lista_blockdiag)
    CONFIG="lista_parity_generic_sparse"
    TOTAL_JOBS=$((NUM_SYSTEMS * NUM_DTS * NUM_STEP_OPTIONS * NUM_BLOCKDIAG_SPARSITY * NUM_ALPHAS * NUM_LOOPS * NUM_BLOCKS * NUM_SEEDS))
    ;;
  *)
    echo "Unknown MODEL_VARIANT='${MODEL_VARIANT}'. Expected one of: generic_sparse, lista_blockdiag"
    exit 2
    ;;
esac

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_JOBS )); then
  echo "Task ${TASK_ID} out of range for TOTAL_JOBS=${TOTAL_JOBS}. Exiting."
  exit 0
fi

INDEX=${TASK_ID}
SEED_IDX=$((INDEX % NUM_SEEDS))
INDEX=$((INDEX / NUM_SEEDS))

if [[ "${MODEL_VARIANT}" == "generic_sparse" ]]; then
  SPARSITY_IDX=$((INDEX % NUM_GENERIC_SPARSITY))
  INDEX=$((INDEX / NUM_GENERIC_SPARSITY))
else
  SPARSITY_IDX=$((INDEX % NUM_BLOCKDIAG_SPARSITY))
  INDEX=$((INDEX / NUM_BLOCKDIAG_SPARSITY))
fi

STEP_IDX=$((INDEX % NUM_STEP_OPTIONS))
INDEX=$((INDEX / NUM_STEP_OPTIONS))
DT_IDX=$((INDEX % NUM_DTS))
INDEX=$((INDEX / NUM_DTS))
SYSTEM_IDX=$((INDEX % NUM_SYSTEMS))
INDEX=$((INDEX / NUM_SYSTEMS))

SEED=${SEEDS[$SEED_IDX]}
NUM_STEPS=${NUM_STEPS_LIST[$STEP_IDX]}
ENV_DT=${ENV_DTS[$DT_IDX]}
SYSTEM=${SYSTEMS[$SYSTEM_IDX]}

LISTA_ALPHA="${LISTA_ALPHAS[0]}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS_LIST[0]}"
K_BLOCK_SIZE="${K_BLOCK_SIZES[0]}"

if [[ "${MODEL_VARIANT}" == "generic_sparse" ]]; then
  SPARSITY_COEFF=${GENERIC_SPARSITY_COEFFS[$SPARSITY_IDX]}
else
  ALPHA_IDX=$((INDEX % NUM_ALPHAS))
  INDEX=$((INDEX / NUM_ALPHAS))
  LOOP_IDX=$((INDEX % NUM_LOOPS))
  INDEX=$((INDEX / NUM_LOOPS))
  BLOCK_IDX=$((INDEX % NUM_BLOCKS))

  SPARSITY_COEFF=${BLOCKDIAG_SPARSITY_COEFFS[$SPARSITY_IDX]}
  LISTA_ALPHA=${LISTA_ALPHAS[$ALPHA_IDX]}
  LISTA_NUM_LOOPS=${LISTA_NUM_LOOPS_LIST[$LOOP_IDX]}
  K_BLOCK_SIZE=${K_BLOCK_SIZES[$BLOCK_IDX]}
fi

SP_TAG=$(tagify "${SPARSITY_COEFF}")
STEP_TAG=$(tagify "${NUM_STEPS}")
DT_TAG=$(tagify "${ENV_DT}")
ALPHA_TAG=$(tagify "${LISTA_ALPHA}")

LOG_DIR="${BASE_OUT}/${MODEL_VARIANT}/${SYSTEM}/dt_${DT_TAG}/steps_${STEP_TAG}/sp_${SP_TAG}"
if [[ "${MODEL_VARIANT}" == "lista_blockdiag" ]]; then
  LOG_DIR="${LOG_DIR}/alpha_${ALPHA_TAG}/loops_${LISTA_NUM_LOOPS}/block_${K_BLOCK_SIZE}"
fi
if [[ "${SYSTEM}" == "kuramoto" && -n "${KURAMOTO_NUM_OSCILLATORS}" ]]; then
  LOG_DIR="${LOG_DIR}/n_${KURAMOTO_NUM_OSCILLATORS}"
fi
if [[ "${SYSTEM}" == "hopfield" && -n "${HOPFIELD_NUM_NEURONS}" ]]; then
  LOG_DIR="${LOG_DIR}/n_${HOPFIELD_NUM_NEURONS}"
fi
if [[ "${SYSTEM}" == "competitive_lv" && -n "${COMPETITIVE_LV_NUM_SPECIES}" ]]; then
  LOG_DIR="${LOG_DIR}/n_${COMPETITIVE_LV_NUM_SPECIES}"
fi
LOG_DIR="${LOG_DIR}/seed_${SEED}"
mkdir -p "${LOG_DIR}"

echo "============================================="
echo "Intrinsic-HD DT Rescue"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "TOTAL_JOBS: ${TOTAL_JOBS}"
echo "Model Variant: ${MODEL_VARIANT}"
echo "Config: ${CONFIG}"
echo "System: ${SYSTEM}"
echo "Seed: ${SEED}"
echo "ENV_DT: ${ENV_DT}"
echo "NUM_STEPS: ${NUM_STEPS}"
echo "SPARSITY_COEFF: ${SPARSITY_COEFF}"
if [[ "${MODEL_VARIANT}" == "lista_blockdiag" ]]; then
  echo "LISTA_ALPHA: ${LISTA_ALPHA}"
  echo "LISTA_NUM_LOOPS: ${LISTA_NUM_LOOPS}"
  echo "K_BLOCK_SIZE: ${K_BLOCK_SIZE}"
fi
if [[ "${SYSTEM}" == "kuramoto" && -n "${KURAMOTO_NUM_OSCILLATORS}" ]]; then
  echo "KURAMOTO_NUM_OSCILLATORS: ${KURAMOTO_NUM_OSCILLATORS}"
fi
if [[ "${SYSTEM}" == "hopfield" && -n "${HOPFIELD_NUM_NEURONS}" ]]; then
  echo "HOPFIELD_NUM_NEURONS: ${HOPFIELD_NUM_NEURONS}"
fi
if [[ "${SYSTEM}" == "hopfield" && -n "${HOPFIELD_NUM_PATTERNS}" ]]; then
  echo "HOPFIELD_NUM_PATTERNS: ${HOPFIELD_NUM_PATTERNS}"
fi
if [[ "${SYSTEM}" == "competitive_lv" && -n "${COMPETITIVE_LV_NUM_SPECIES}" ]]; then
  echo "COMPETITIVE_LV_NUM_SPECIES: ${COMPETITIVE_LV_NUM_SPECIES}"
fi
echo "LOG_DIR: ${LOG_DIR}"
echo "Start Time: $(date)"
echo "============================================="

TRAIN_ARGS=(
  --config "${CONFIG}"
  --env "${SYSTEM}"
  --env_dt "${ENV_DT}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --target_size "${TARGET_SIZE}"
  --res_coeff "${RES_COEFF}"
  --reconst_coeff "${RECONST_COEFF}"
  --pred_coeff "${PRED_COEFF}"
  --sparsity_coeff "${SPARSITY_COEFF}"
  --sequence_length "${SEQUENCE_LENGTH}"
  --eval_profile "${EVAL_PROFILE}"
  --seed "${SEED}"
  --device cuda
  --log_dir "${LOG_DIR}"
)

if [[ "${MODEL_VARIANT}" == "lista_blockdiag" ]]; then
  TRAIN_ARGS+=(
    --k_structure block_diagonal
    --k_block_size "${K_BLOCK_SIZE}"
    --lista_alpha "${LISTA_ALPHA}"
    --lista_num_loops "${LISTA_NUM_LOOPS}"
    --lista_final_op "${LISTA_FINAL_OP}"
  )
fi

if [[ "${SYSTEM}" == "kuramoto" && -n "${KURAMOTO_NUM_OSCILLATORS}" ]]; then
  TRAIN_ARGS+=(--kuramoto_num_oscillators "${KURAMOTO_NUM_OSCILLATORS}")
fi
if [[ "${SYSTEM}" == "hopfield" && -n "${HOPFIELD_NUM_NEURONS}" ]]; then
  TRAIN_ARGS+=(--hopfield_num_neurons "${HOPFIELD_NUM_NEURONS}")
fi
if [[ "${SYSTEM}" == "hopfield" && -n "${HOPFIELD_NUM_PATTERNS}" ]]; then
  TRAIN_ARGS+=(--hopfield_num_patterns "${HOPFIELD_NUM_PATTERNS}")
fi
if [[ "${SYSTEM}" == "competitive_lv" && -n "${COMPETITIVE_LV_NUM_SPECIES}" ]]; then
  TRAIN_ARGS+=(--competitive_lv_num_species "${COMPETITIVE_LV_NUM_SPECIES}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN command: uv run python tools/train.py'
  printf ' %q' "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

uv run python tools/train.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
