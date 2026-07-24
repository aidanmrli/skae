#!/usr/bin/env bash
# Evaluate one frozen GatedLocalLinear checkpoint without fitting dynamics.

#SBATCH --job-name=k_distinct_law
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-2

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
PROJECT_DIR="${ROOT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
ARM="${ARM:?ARM must be sparse or dense}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
CARD_PATH="${CARD_PATH:-experiments/neurips_2026/global_k_distinct_laws_card.json}"

if [[ "${ARM}" != "sparse" && "${ARM}" != "dense" ]]; then
  echo "ARM must be sparse or dense, got ${ARM}." >&2
  exit 2
fi
if [[ "${ARM}" == "dense" ]]; then
  : "${DENSE_TASK_TSV:?DENSE_TASK_TSV is required for ARM=dense}"
  : "${DENSE_BASE_OUT:?DENSE_BASE_OUT is required for ARM=dense}"
fi

cd "${ROOT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export SKAE_GIT_COMMIT
SKAE_GIT_COMMIT="$(git rev-parse HEAD)"

echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "git_commit=${SKAE_GIT_COMMIT}"
echo "arm=${ARM} task=${TASK_ID} cpus=${SLURM_CPUS_PER_TASK} gpu_count=0"
sha256sum \
  "${CARD_PATH}" \
  experiments/neurips_2026/global_k_distinct_laws.py \
  experiments/neurips_2026/global_k_support_invariance.py \
  experiments/neurips_2026/global_k_dense_specificity.py

ARGS=(
  --card "${CARD_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --arm "${ARM}"
  --task_index "${TASK_ID}"
  --device cpu
)
if [[ "${ARM}" == "dense" ]]; then
  ARGS+=(
    --dense_task_tsv "${DENSE_TASK_TSV}"
    --dense_base_out "${DENSE_BASE_OUT}"
  )
fi

/usr/bin/time -v env PYTHONPATH="${ROOT_DIR}" \
  uv run --project "${PROJECT_DIR}" python \
  experiments/neurips_2026/global_k_distinct_laws.py "${ARGS[@]}"

echo "completed_at=$(date --iso-8601=seconds)"
