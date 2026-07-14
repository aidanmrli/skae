#!/bin/bash
#
# Re-evaluate local-operator and global-operator LISTA checkpoints using the
# same wide periodic re-encoding period grid.
#
# Required env vars:
#   STAGED_ROOT=<staged run root>
#   GLOBAL_ROOT=<global run root>
#   OUT_DIR=<output directory>
#
# Optional:
#   HORIZONS_CSV, PERIODS_CSV, BATCH_SIZE
#   SUPPORT_DEFINITION, FAMILY_JACCARD_THRESHOLD
#   FORCE=0
#
# Unset scientific values resolve from
# experiments.neurips_2026.local_operators.contract.

#SBATCH --job-name=wide_periodic_k
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH -o slurm-%x-%A.out
#SBATCH -e slurm-%x-%A.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

source scripts/common/gpu_guard.sh
trap gpu_guard_stop_sampler EXIT
module load cuda/12.6.0
source .venv/bin/activate

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Start Time: $(date)"
gpu_guard_assert_cuda_visible "staged/global wide-periodic re-evaluation"
gpu_guard_print_context "Staged vs Global Wide-Periodic Re-evaluation"

STAGED_ROOT="${STAGED_ROOT:?STAGED_ROOT is required}"
GLOBAL_ROOT="${GLOBAL_ROOT:?GLOBAL_ROOT is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
HORIZONS_CSV="${HORIZONS_CSV:-}"
PERIODS_CSV="${PERIODS_CSV:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
SUPPORT_DEFINITION="${SUPPORT_DEFINITION:-}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-}"
FORCE="${FORCE:-0}"

echo "============================================="
echo "Wide Periodic Staged-vs-Global Re-evaluation"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Staged root: ${STAGED_ROOT}"
echo "Global root: ${GLOBAL_ROOT}"
echo "Output: ${OUT_DIR}"
echo "Protocol: experiments.neurips_2026.local_operators.contract"
echo "============================================="

ARGS=(
  --staged_root "${STAGED_ROOT}"
  --global_root "${GLOBAL_ROOT}"
  --output_dir "${OUT_DIR}"
  --device cuda
)

append_override() {
  local flag="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    ARGS+=("${flag}" "${value}")
  fi
}
append_override --horizons "${HORIZONS_CSV}"
append_override --periods "${PERIODS_CSV}"
append_override --batch_size "${BATCH_SIZE}"
append_override --support_definition "${SUPPORT_DEFINITION}"
append_override --family_jaccard_threshold "${FAMILY_JACCARD_THRESHOLD}"

if [[ "${FORCE}" == "1" ]]; then
  ARGS+=(--force)
fi

gpu_guard_start_sampler \
  "${OUT_DIR}/gpu_utilization_${SLURM_JOB_ID:-local}.csv" \
  "${GPU_TELEMETRY_INTERVAL:-30}"
gpu_guard_phase "wide-periodic reevaluation start"
uv run skae-paper evaluate local-operators "${ARGS[@]}"
gpu_guard_phase "wide-periodic reevaluation end"
gpu_guard_stop_sampler

echo "End Time: $(date)"
