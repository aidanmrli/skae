#!/usr/bin/env bash
#SBATCH --job-name=spatial-rd-lista-smoke
#SBATCH --output=logs/slurm/spatial_rd_lista_smoke_%j.out
#SBATCH --error=logs/slurm/spatial_rd_lista_smoke_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs/slurm

echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse --short HEAD || true)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

SOURCE_SYSTEM="${SOURCE_SYSTEM:-cal_square_4}"
SEED="${SEED:-0}"
GRID_SIZE="${GRID_SIZE:-32}"
DIFFUSION="${DIFFUSION:-0.01}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-24}"
TRAIN_TRAJECTORIES="${TRAIN_TRAJECTORIES:-48}"
VAL_TRAJECTORIES="${VAL_TRAJECTORIES:-12}"
TEST_TRAJECTORIES="${TEST_TRAJECTORIES:-12}"
SUBSTEPS_PER_OBSERVATION="${SUBSTEPS_PER_OBSERVATION:-10}"
RK4_DT="${RK4_DT:-0.01}"
LABEL_EXTRA_OBSERVATIONS="${LABEL_EXTRA_OBSERVATIONS:-24}"
LAPLACIAN_SCALING="${LAPLACIAN_SCALING:-continuum}"
SPATIAL_EXTENT="${SPATIAL_EXTENT:-1.0}"
GENERATE_DATASET="${GENERATE_DATASET:-1}"

TRAINER="${TRAINER:-flat}"
MODEL_VARIANT="${MODEL_VARIANT:-conv_lista}"
NUM_STEPS="${NUM_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-4}"
CONFIG="${CONFIG:-lista_parity_generic_sparse}"
MIN_TARGET_SIZE=$((4 * 2 * GRID_SIZE * GRID_SIZE))
TARGET_SIZE="${TARGET_SIZE:-${MIN_TARGET_SIZE}}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS:-2}"
LISTA_ALPHA="${LISTA_ALPHA:-0.1}"
CONV_HIDDEN_CHANNELS="${CONV_HIDDEN_CHANNELS:-64}"
CONV_NUM_BLOCKS="${CONV_NUM_BLOCKS:-3}"
PRED_COEFF="${PRED_COEFF:-1.0}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.01}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-1e-3}"
FAMILY_JACCARD="${FAMILY_JACCARD:-1.0}"
FAMILY_MAX_VALIDATION_REPS="${FAMILY_MAX_VALIDATION_REPS:-256}"
DEEP_THRESHOLD="${DEEP_THRESHOLD:-0.9}"
EVAL_HORIZON="${EVAL_HORIZON:-8}"

RUN_ROOT="${RUN_ROOT:-runs/spatialized_reaction_diffusion/${SOURCE_SYSTEM}_seed${SEED}_grid${GRID_SIZE}_smoke}"
DATASET_PATH="${DATASET_PATH:-${RUN_ROOT}/dataset.pt}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/lista}"
EVAL_PATH="${EVAL_PATH:-${RUN_ROOT}/evaluation.json}"

mkdir -p "${RUN_ROOT}" "${RUN_DIR}"

if [[ "${GENERATE_DATASET}" == "1" ]]; then
  uv run python tools/generate_spatialized_reaction_diffusion.py \
    --output "${DATASET_PATH}" \
    --source_system "${SOURCE_SYSTEM}" \
    --grid_size "${GRID_SIZE}" \
    --diffusion "${DIFFUSION}" \
    --rk4_dt "${RK4_DT}" \
    --substeps_per_observation "${SUBSTEPS_PER_OBSERVATION}" \
    --trajectory_length "${TRAJECTORY_LENGTH}" \
    --label_extra_observations "${LABEL_EXTRA_OBSERVATIONS}" \
    --spatial_extent "${SPATIAL_EXTENT}" \
    --laplacian_scaling "${LAPLACIAN_SCALING}" \
    --train_trajectories "${TRAIN_TRAJECTORIES}" \
    --val_trajectories "${VAL_TRAJECTORIES}" \
    --test_trajectories "${TEST_TRAJECTORIES}" \
    --seed "${SEED}"
else
  echo "Skipping dataset generation; using DATASET_PATH=${DATASET_PATH}"
fi

if [[ "${TRAINER}" == "conv" ]]; then
  uv run python tools/train_spatialized_reaction_diffusion_conv.py \
    --dataset "${DATASET_PATH}" \
    --run_dir "${RUN_DIR}" \
    --model_variant "${MODEL_VARIANT}" \
    --seed "${SEED}" \
    --z_dim "${TARGET_SIZE}" \
    --hidden_channels "${CONV_HIDDEN_CHANNELS}" \
    --num_blocks "${CONV_NUM_BLOCKS}" \
    --num_steps "${NUM_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --sequence_length "${SEQUENCE_LENGTH}" \
    --lista_num_loops "${LISTA_NUM_LOOPS}" \
    --lista_alpha "${LISTA_ALPHA}" \
    --prediction_weight "${PRED_COEFF}" \
    --sparsity_weight "${SPARSITY_COEFF}" \
    --eval_horizon "${EVAL_HORIZON}" \
    --device auto
else
  uv run python tools/train_spatialized_reaction_diffusion_lista.py \
    --dataset "${DATASET_PATH}" \
    --run_dir "${RUN_DIR}" \
    --config "${CONFIG}" \
    --seed "${SEED}" \
    --target_size "${TARGET_SIZE}" \
    --num_steps "${NUM_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --sequence_length "${SEQUENCE_LENGTH}" \
    --lista_num_loops "${LISTA_NUM_LOOPS}" \
    --lista_alpha "${LISTA_ALPHA}" \
    --pred_coeff "${PRED_COEFF}" \
    --sparsity_coeff "${SPARSITY_COEFF}" \
    --eval_horizon "${EVAL_HORIZON}" \
    --device auto
fi

uv run python tools/evaluate_spatialized_reaction_diffusion.py \
  --dataset "${DATASET_PATH}" \
  --checkpoint "${RUN_DIR}/checkpoint.pt" \
  --output "${EVAL_PATH}" \
  --horizons 1 4 8 12 \
  --support_threshold "${SUPPORT_THRESHOLD}" \
  --family_jaccard "${FAMILY_JACCARD}" \
  --max_validation_reps "${FAMILY_MAX_VALIDATION_REPS}" \
  --deep_threshold "${DEEP_THRESHOLD}" \
  --device auto

echo "dataset=${DATASET_PATH}"
echo "run_dir=${RUN_DIR}"
echo "evaluation=${EVAL_PATH}"
echo "date=$(date --iso-8601=seconds)"
