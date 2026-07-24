#!/usr/bin/env bash
#SBATCH --job-name=ac-lista-v3b48-select
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
sha256sum -c experiments/neurips_2026/allen_cahn_lista_refinement_stable_v2/source_manifest.sha256
uv run python -m experiments.neurips_2026.allen_cahn_lista_refinement_stable.select \
  --root "${SKAE_SCRATCH_ROOT}/allen_cahn_lista_refinement_stable_20260722_v3_b48" \
  --output "${SKAE_SCRATCH_ROOT}/allen_cahn_lista_refinement_stable_20260722_v3_b48/selection.json"
