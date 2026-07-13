#!/usr/bin/env bash
#SBATCH --job-name=collect-ac-lista
#SBATCH --output=/network/scratch/l/lia/skae/collect-ac-lista-%A.out
#SBATCH --error=/network/scratch/l/lia/skae/collect-ac-lista-%A.err
#SBATCH --time=00:30:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

RESULTS_DIR="${RESULTS_DIR:-results/allen_cahn_lista_stiffness_50k_20260629}"
REFERENCE_DIR="${REFERENCE_DIR:-results/allen_cahn_multistable_pde_h200_periodic_50k_20260629}"

uv run python scripts/summarize_allen_cahn_lista_stiffness.py \
  --results-dir "${RESULTS_DIR}" \
  --reference-dir "${REFERENCE_DIR}"
