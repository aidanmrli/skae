#!/bin/bash
#
# Compare Kuramoto dimension-sweep candidate roots against generic_sparse at each N.
#
# Required env vars:
#   ROWS_CSV=<forecasting_rows.csv>
#   OUT_DIR=<comparison directory>
#
# Optional:
#   DIMENSIONS_CSV=8,16,24,32,64
#   HORIZON=1000
#   ANCHOR_PREFIX=generic_sparse
#   CANDIDATE_PREFIXES_CSV=lista_dense_promoted,lista_blockdiag
#
#SBATCH --job-name=compare_kura_dim
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o /network/scratch/l/lia/skae/compare-kura-dim-%A.out
#SBATCH -e /network/scratch/l/lia/skae/compare-kura-dim-%A.err

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
DIMENSIONS_CSV="${DIMENSIONS_CSV:-8,16,24,32,64}"
HORIZON="${HORIZON:-1000}"
ANCHOR_PREFIX="${ANCHOR_PREFIX:-generic_sparse}"
CANDIDATE_PREFIXES_CSV="${CANDIDATE_PREFIXES_CSV:-lista_dense_promoted,lista_blockdiag}"

IFS=',' read -r -a DIMENSIONS <<< "${DIMENSIONS_CSV}"
IFS=',' read -r -a CANDIDATES <<< "${CANDIDATE_PREFIXES_CSV}"

echo "============================================="
echo "Compare Kuramoto Dimension Sweep"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "ROWS_CSV: ${ROWS_CSV}"
echo "OUT_DIR: ${OUT_DIR}"
echo "DIMENSIONS: ${DIMENSIONS_CSV}"
echo "============================================="

for dimension in "${DIMENSIONS[@]}"; do
  dimension="$(echo "${dimension}" | xargs)"
  [[ -z "${dimension}" ]] && continue
  anchor_root="${ANCHOR_PREFIX}_n${dimension}"
  for candidate in "${CANDIDATES[@]}"; do
    candidate="$(echo "${candidate}" | xargs)"
    [[ -z "${candidate}" ]] && continue
    uv run python tools/compare_forecasting_roots.py \
      --rows_csv "${ROWS_CSV}" \
      --output_dir "${OUT_DIR}/${candidate}_n${dimension}_vs_${anchor_root}" \
      --candidate_root "${candidate}_n${dimension}" \
      --anchor_roots "${anchor_root}" \
      --horizon "${HORIZON}"
  done
done
