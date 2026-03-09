#!/bin/bash
#
# Compare paper-benchmark candidate roots against the generic_sparse anchor.
#
# Required env vars:
#   ROWS_CSV=<forecasting_rows.csv>
#   OUT_DIR=<comparison directory>
#
# Optional:
#   CANDIDATE_ROOTS_CSV=lista_dense,lista_diagonal,lista_blockdiag
#   ANCHOR_ROOT=generic_sparse
#   HORIZON=1000
#
#SBATCH --job-name=compare_paper
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o /network/scratch/l/lia/skae/compare-paper-%A.out
#SBATCH -e /network/scratch/l/lia/skae/compare-paper-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"

ROWS_CSV="${ROWS_CSV:?ROWS_CSV is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
CANDIDATE_ROOTS_CSV="${CANDIDATE_ROOTS_CSV:-lista_dense,lista_diagonal,lista_blockdiag}"
ANCHOR_ROOT="${ANCHOR_ROOT:-generic_sparse}"
HORIZON="${HORIZON:-1000}"

IFS=',' read -r -a CANDIDATES <<< "${CANDIDATE_ROOTS_CSV}"

echo "============================================="
echo "Compare Paper Benchmark"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "ROWS_CSV: ${ROWS_CSV}"
echo "OUT_DIR: ${OUT_DIR}"
echo "ANCHOR_ROOT: ${ANCHOR_ROOT}"
echo "CANDIDATES: ${CANDIDATE_ROOTS_CSV}"
echo "============================================="

for candidate in "${CANDIDATES[@]}"; do
  candidate="$(echo "${candidate}" | xargs)"
  [[ -z "${candidate}" ]] && continue
  uv run python tools/compare_forecasting_roots.py \
    --rows_csv "${ROWS_CSV}" \
    --output_dir "${OUT_DIR}/${candidate}_vs_${ANCHOR_ROOT}" \
    --candidate_root "${candidate}" \
    --anchor_roots "${ANCHOR_ROOT}" \
    --horizon "${HORIZON}"
done
