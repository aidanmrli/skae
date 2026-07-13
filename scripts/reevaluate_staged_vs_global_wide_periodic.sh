#!/bin/bash
#
# Re-evaluate staged local-K and matched global-K LISTA checkpoints using the
# same wide periodic re-encoding period grid.
#
# Required env vars:
#   STAGED_ROOT=<staged run root>
#   GLOBAL_ROOT=<global run root>
#   OUT_DIR=<output directory>
#
# Optional:
#   HORIZONS_CSV=100,500,1000
#   PERIODS_CSV=1,2,5,10,20,25,50,100
#   BATCH_SIZE=100
#   SUPPORT_DEFINITION=absolute:0.001
#   FAMILY_JACCARD_THRESHOLD=0.4
#   FORCE=0

#SBATCH --job-name=wide_periodic_k
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH -o /network/scratch/l/lia/skae/wide-periodic-local-k-%A.out
#SBATCH -e /network/scratch/l/lia/skae/wide-periodic-local-k-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

module load cuda/12.6.0
source .venv/bin/activate

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Start Time: $(date)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

STAGED_ROOT="${STAGED_ROOT:?STAGED_ROOT is required}"
GLOBAL_ROOT="${GLOBAL_ROOT:?GLOBAL_ROOT is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
HORIZONS_CSV="${HORIZONS_CSV:-100,500,1000}"
PERIODS_CSV="${PERIODS_CSV:-1,2,5,10,20,25,50,100}"
BATCH_SIZE="${BATCH_SIZE:-100}"
SUPPORT_DEFINITION="${SUPPORT_DEFINITION:-absolute:0.001}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.4}"
FORCE="${FORCE:-0}"

echo "============================================="
echo "Wide Periodic Staged-vs-Global Re-evaluation"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Staged root: ${STAGED_ROOT}"
echo "Global root: ${GLOBAL_ROOT}"
echo "Output: ${OUT_DIR}"
echo "Horizons: ${HORIZONS_CSV}"
echo "Periods: ${PERIODS_CSV}"
echo "Batch size: ${BATCH_SIZE}"
echo "Support: ${SUPPORT_DEFINITION}"
echo "Family Jaccard: ${FAMILY_JACCARD_THRESHOLD}"
echo "============================================="

ARGS=(
  --staged_root "${STAGED_ROOT}"
  --global_root "${GLOBAL_ROOT}"
  --output_dir "${OUT_DIR}"
  --horizons "${HORIZONS_CSV}"
  --periods "${PERIODS_CSV}"
  --batch_size "${BATCH_SIZE}"
  --support_definition "${SUPPORT_DEFINITION}"
  --family_jaccard_threshold "${FAMILY_JACCARD_THRESHOLD}"
  --device cuda
)

if [[ "${FORCE}" == "1" ]]; then
  ARGS+=(--force)
fi

uv run python tools/reevaluate_staged_vs_global_wide_periodic.py "${ARGS[@]}"

echo "End Time: $(date)"
