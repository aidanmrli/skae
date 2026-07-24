#!/usr/bin/env bash
#SBATCH --job-name=queue-ac-lista-stable
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
ROOT_DIR="${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/common/cluster_env.sh"
cd "${PROJECT_DIR}"
SMOKE_JOB=$(sbatch --parsable scripts/neurips_2026/allen_cahn_lista_refinement_stable/smoke.sh)
TRAIN_JOB=$(sbatch --parsable --dependency="afterok:${SMOKE_JOB}" scripts/neurips_2026/allen_cahn_lista_refinement_stable/run_array.sh)
SELECT_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" scripts/neurips_2026/allen_cahn_lista_refinement_stable/select.sh)
printf 'smoke=%s\ntraining_array=%s\nselection=%s\n' "${SMOKE_JOB}" "${TRAIN_JOB}" "${SELECT_JOB}"
