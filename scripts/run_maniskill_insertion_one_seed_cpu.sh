#!/usr/bin/env bash
#SBATCH --job-name=mskill_cpu1
#SBATCH --partition=long
#SBATCH --output=logs/maniskill_insertion_one_seed_cpu_%j.out
#SBATCH --error=logs/maniskill_insertion_one_seed_cpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs runs/maniskill_insertion
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

echo "date=$(date)"
echo "host=$(hostname)"
echo "commit=$(git rev-parse --short HEAD || true)"

DATASET="${DATASET:?Set DATASET to a compact ManiSkill insertion .npz path}"
SEED="${SEED:-0}"
NUM_STEPS="${NUM_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-10}"
Z_DIM="${Z_DIM:-128}"
ENCODER_KIND="${ENCODER_KIND:-lista}"
ACTIVATION="${ACTIVATION:-auto}"
LISTA_LOOPS="${LISTA_LOOPS:-2}"
LISTA_ALPHA="${LISTA_ALPHA:-0.05}"
SPARSITY_WEIGHT="${SPARSITY_WEIGHT:-1e-3}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-1e-3}"
FAMILY_JACCARD="${FAMILY_JACCARD:-0.5}"
RUN_DIR="${RUN_DIR:-runs/maniskill_insertion/controlled_lista_cpu_seed${SEED}}"

uv run python tools/train_maniskill_controlled_lista.py \
  --dataset "${DATASET}" \
  --run_dir "${RUN_DIR}" \
  --seed "${SEED}" \
  --encoder_kind "${ENCODER_KIND}" \
  --activation "${ACTIVATION}" \
  --num_steps "${NUM_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --sequence_length "${SEQUENCE_LENGTH}" \
  --z_dim "${Z_DIM}" \
  --lista_loops "${LISTA_LOOPS}" \
  --lista_alpha "${LISTA_ALPHA}" \
  --sparsity_weight "${SPARSITY_WEIGHT}" \
  --device cpu

uv run python tools/evaluate_maniskill_controlled_lista.py \
  --dataset "${DATASET}" \
  --checkpoint "${RUN_DIR}/checkpoint.pt" \
  --output_dir "${RUN_DIR}/eval_test" \
  --device cpu \
  --split test \
  --horizons 10,25,50,100 \
  --support_threshold "${SUPPORT_THRESHOLD}" \
  --family_jaccard "${FAMILY_JACCARD}"

echo "run_dir=${RUN_DIR}"
echo "summary=${RUN_DIR}/eval_test/metrics_summary.json"
