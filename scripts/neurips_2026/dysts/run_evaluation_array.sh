#!/bin/bash
#
# Submit one long-horizon Dysts evaluation row per SLURM array element.
# The allocation-free payload is shared with the packed worker.
#
# Required env vars:
#   TASK_TSV=<path>
#
#SBATCH --job-name=dysts_long_eval
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err
#SBATCH --requeue

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

exec bash scripts/neurips_2026/dysts/run_evaluation_task.sh
