#!/usr/bin/env bash
#SBATCH --job-name=queue-ac-lista-v3b48
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${PROJECT_DIR}"
sha256sum -c experiments/neurips_2026/allen_cahn_lista_refinement_stable_v2/source_manifest.sha256
SMOKE=$(sbatch --parsable scripts/neurips_2026/allen_cahn_lista_refinement_stable_v2/smoke.sh)
TRAIN=$(sbatch --parsable --dependency="afterok:${SMOKE}" scripts/neurips_2026/allen_cahn_lista_refinement_stable_v2/run_array.sh)
SELECT=$(sbatch --parsable --dependency="afterok:${TRAIN}" scripts/neurips_2026/allen_cahn_lista_refinement_stable_v2/select.sh)
printf 'smoke_job\t%s\ntraining_job\t%s\nselector_job\t%s\n' "${SMOKE}" "${TRAIN}" "${SELECT}"
