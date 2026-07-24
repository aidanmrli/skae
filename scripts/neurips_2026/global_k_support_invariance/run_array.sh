#!/usr/bin/env bash
#SBATCH --job-name=k-support-inv
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-44%15

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
PROJECT_DIR="${ROOT_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
SHARD="${OUTPUT_ROOT}/shards/task_$(printf '%03d' "${TASK_ID}").json"

if [[ -e "${SHARD}" ]]; then
  echo "Refusing to overwrite ${SHARD}." >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/shards"
cd "${ROOT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export SKAE_GIT_COMMIT
SKAE_GIT_COMMIT="$(git rev-parse HEAD)"

echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "git_commit=${SKAE_GIT_COMMIT}"
echo "task=${TASK_ID} cpus=${SLURM_CPUS_PER_TASK} gpu_count=0"
sha256sum \
  experiments/neurips_2026/global_k_support_invariance.py \
  experiments/neurips_2026/global_k_support_invariance_card.json \
  experiments/neurips_2026/summarize_global_k_support_invariance.py

/usr/bin/time -v env PYTHONPATH="${ROOT_DIR}" uv run --project "${PROJECT_DIR}" python \
  experiments/neurips_2026/global_k_support_invariance.py \
  --output_dir "${OUTPUT_ROOT}" \
  --task_index "${TASK_ID}" \
  --device cpu

sha256sum "${SHARD}"
echo "completed_at=$(date --iso-8601=seconds)"
