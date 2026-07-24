#!/usr/bin/env bash
#SBATCH --job-name=ac-lista-stable-select
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
ROOT_DIR="${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/common/cluster_env.sh"
cd "${PROJECT_DIR}"
uv run python -m experiments.neurips_2026.allen_cahn_lista_refinement_stable.select \
  --root "${SKAE_SCRATCH_ROOT}/allen_cahn_lista_refinement_stable_20260722_v1" \
  --output "${SKAE_SCRATCH_ROOT}/allen_cahn_lista_refinement_stable_20260722_v1/selection.json"
