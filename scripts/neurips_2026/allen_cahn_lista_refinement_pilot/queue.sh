#!/usr/bin/env bash
#SBATCH --job-name=queue-ac-depth
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
OUTPUT_ROOT="${SKAE_SCRATCH_ROOT}/allen_cahn_lista_refinement_pilot_20260722"
mkdir -p "${OUTPUT_ROOT}/logs"
cd "${PROJECT_DIR}"
TRAIN_JOB=$(sbatch --parsable scripts/neurips_2026/allen_cahn_lista_refinement_pilot/run_array.sh)
SELECT_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" scripts/neurips_2026/allen_cahn_lista_refinement_pilot/select.sh)
printf 'training_array=%s\nselection=%s\n' "${TRAIN_JOB}" "${SELECT_JOB}" > "${OUTPUT_ROOT}/job_ids.txt"
echo "Training array: ${TRAIN_JOB}"
echo "Selection: ${SELECT_JOB}"
