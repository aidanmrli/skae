#!/bin/bash
#SBATCH --job-name=queue_ledmd_repro
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/network/scratch/l/lia/skae/local_edmd_poly_reproduction_20260720/logs/queue-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/local_edmd_poly_reproduction_20260720/logs/queue-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit this dependency-chain launcher with sbatch." >&2
  exit 2
fi

REPOSITORY_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPOSITORY_ROOT}"
RESULT_ROOT="${RESULT_ROOT:-/network/scratch/l/lia/skae/local_edmd_poly_reproduction_20260720}"
TASK_TSV="${TASK_TSV:-${RESULT_ROOT}/inputs/tasks.tsv}"
mkdir -p "${RESULT_ROOT}/logs"

echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "Git commit: $(git rev-parse HEAD)"
echo "GPU allocation: none"

uv run python -m experiments.neurips_2026.local_edmd_reproduction.source_lock
uv run python -m experiments.neurips_2026.local_edmd_reproduction.tasks \
  --output-root "${RESULT_ROOT}/inputs" --check
for script in scripts/neurips_2026/local_edmd_reproduction/*.sh; do
  bash -n "${script}"
done
uv run python -m compileall -q \
  experiments/neurips_2026/local_edmd_reproduction
uv run pytest tests/test_local_edmd_reproduction.py -q

ARRAY_JOB_ID="$(
  RESULT_ROOT="${RESULT_ROOT}" TASK_TSV="${TASK_TSV}" \
    sbatch --parsable --array=0-74%32 \
      scripts/neurips_2026/local_edmd_reproduction/run_array.sh
)"
ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"
COLLECT_JOB_ID="$(
  RESULT_ROOT="${RESULT_ROOT}" TASK_TSV="${TASK_TSV}" \
    sbatch --parsable --dependency="afterok:${ARRAY_JOB_ID}" \
      scripts/neurips_2026/local_edmd_reproduction/run_collect.sh
)"
COLLECT_JOB_ID="${COLLECT_JOB_ID%%;*}"
CHECK_JOB_ID="$(
  RESULT_ROOT="${RESULT_ROOT}" \
    sbatch --parsable --dependency="afterok:${COLLECT_JOB_ID}" \
      scripts/neurips_2026/local_edmd_reproduction/run_check.sh
)"
CHECK_JOB_ID="${CHECK_JOB_ID%%;*}"

uv run python -m experiments.neurips_2026.local_edmd_reproduction.record_queue \
  --output "${RESULT_ROOT}/queue.json" \
  --queue-job-id "${SLURM_JOB_ID}" \
  --array-job-id "${ARRAY_JOB_ID}" \
  --collect-job-id "${COLLECT_JOB_ID}" \
  --check-job-id "${CHECK_JOB_ID}" \
  --task-tsv "${TASK_TSV}" \
  --result-root "${RESULT_ROOT}"

echo "Array job: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID} (afterok:${ARRAY_JOB_ID})"
echo "Check job: ${CHECK_JOB_ID} (afterok:${COLLECT_JOB_ID})"
