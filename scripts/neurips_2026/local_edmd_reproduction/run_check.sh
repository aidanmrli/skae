#!/bin/bash
#SBATCH --job-name=ledmd_poly_check
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/network/scratch/l/lia/skae/local_edmd_poly_reproduction_20260720/logs/check-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/local_edmd_poly_reproduction_20260720/logs/check-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit this checker with sbatch." >&2
  exit 2
fi

REPOSITORY_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPOSITORY_ROOT}"
RESULT_ROOT="${RESULT_ROOT:-/network/scratch/l/lia/skae/local_edmd_poly_reproduction_20260720}"

echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "Git commit: $(git rev-parse HEAD)"
echo "GPU allocation: none"

uv run python -m experiments.neurips_2026.local_edmd_reproduction.source_lock
uv run python -m experiments.neurips_2026.local_edmd_reproduction.summarize \
  --result-root "${RESULT_ROOT}" \
  --evidence-dir "${RESULT_ROOT}/evidence" \
  --summary-dir "${RESULT_ROOT}/summary"

