#!/bin/bash
#
# Submit one long-horizon Dysts reevaluation row per SLURM array element.
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
#SBATCH -o /network/scratch/l/lia/skae/dysts-long-eval-%A_%a.out
#SBATCH -e /network/scratch/l/lia/skae/dysts-long-eval-%A_%a.err
#SBATCH --requeue

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

exec bash scripts/run_dysts_long_horizon_eval_task.sh
