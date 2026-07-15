#!/bin/bash
#
# Compare controlled candidate roots against the frozen dense anchor.
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
#SBATCH -o slurm-%x-%A.out
#SBATCH -e slurm-%x-%A.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
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
  uv run skae-paper collect compare \
    --rows_csv "${ROWS_CSV}" \
    --output_dir "${OUT_DIR}/${candidate}_vs_${ANCHOR_ROOT}" \
    --candidate_root "${candidate}" \
    --anchor_roots "${ANCHOR_ROOT}" \
    --horizon "${HORIZON}"
done
