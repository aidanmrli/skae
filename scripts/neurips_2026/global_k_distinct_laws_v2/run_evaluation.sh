#!/bin/bash
# Evaluate one frozen H/G arm-seed checkpoint on CPU.

#SBATCH --job-name=gkv2_eval
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
export SKAE_GIT_COMMIT="$(git rev-parse HEAD)"
source scripts/common/cluster_env.sh

CARD_PATH="${CARD_PATH:-experiments/neurips_2026/global_k_distinct_laws_v2_card.json}"
SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH:?SOURCE_LOCK_PATH is required}"
EXPECTED_SOURCE_LOCK_SHA="${EXPECTED_SOURCE_LOCK_SHA:?EXPECTED_SOURCE_LOCK_SHA is required}"
TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"
AUDIT_DIR="${AUDIT_DIR:?AUDIT_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
uv run python -m experiments.neurips_2026.global_k_distinct_laws_v2 \
  --card "${CARD_PATH}" \
  --source_lock "${SOURCE_LOCK_PATH}" \
  --expected_source_lock_sha "${EXPECTED_SOURCE_LOCK_SHA}" \
  --task_tsv "${TASK_TSV}" \
  --base_out "${BASE_OUT}" \
  --audit_dir "${AUDIT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --task_index "${SLURM_ARRAY_TASK_ID:?Array task ID is required}"
