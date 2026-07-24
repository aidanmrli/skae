#!/bin/bash
#SBATCH --job-name=ledmd_poly_repro
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/network/scratch/l/lia/skae/local_edmd_poly_reproduction_20260720/logs/array-%A_%a.out
#SBATCH --error=/network/scratch/l/lia/skae/local_edmd_poly_reproduction_20260720/logs/array-%A_%a.err

set -euo pipefail

repository_root() {
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    printf '%s\n' "${SLURM_SUBMIT_DIR}"
  else
    cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
    pwd
  fi
}

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this worker as a SLURM array with sbatch." >&2
  exit 2
fi

REPOSITORY_ROOT="$(repository_root)"
cd "${REPOSITORY_ROOT}"

RESULT_ROOT="${RESULT_ROOT:-/network/scratch/l/lia/skae/local_edmd_poly_reproduction_20260720}"
TASK_TSV="${TASK_TSV:-${RESULT_ROOT}/inputs/tasks.tsv}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONHASHSEED=0

echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Task index: ${SLURM_ARRAY_TASK_ID}"
echo "CPU allocation: ${SLURM_CPUS_PER_TASK}; GPU allocation: none"
echo "Task TSV: ${TASK_TSV}"

uv run python -m experiments.neurips_2026.local_edmd_reproduction.source_lock
uv run python -m experiments.neurips_2026.local_edmd_reproduction.evaluation \
  --task-tsv "${TASK_TSV}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --result-root "${RESULT_ROOT}"

echo "Completed: $(date --iso-8601=seconds)"

