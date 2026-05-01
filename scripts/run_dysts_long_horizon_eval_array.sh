#!/bin/bash
#
# Array runner for long-horizon Dysts reevaluation tasks.
#
# Required env vars:
#   TASK_TSV=<path>
#
# Optional env vars:
#   OUTPUT_TAG=dysts_long_horizon_h5k_to_h60k_seq10
#   CHECKPOINT_NAME=checkpoint
#   EVAL_DEVICE=cpu
#   DYSTS_CACHE_PROFILE=long60
#   DYSTS_CACHE_SPLIT=test
#   DYSTS_CACHE_DIR=/network/scratch/l/lia/skae/dysts_native_cache
#   DYSTS_CACHE_NUM_WORKERS=2
#   BATCH_SIZE=100
#   HORIZONS="5000 10000 20000 30000 40000 50000 60000"
#   DYSTS_PERIODIC_REENCODE_PERIODS="50 75 100 200 400 600 1000"
#   ARRAY_OFFSET=0
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

source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Date: $(date)"

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
OUTPUT_TAG="${OUTPUT_TAG:-dysts_long_horizon_h5k_to_h60k_seq10}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE:-long60}"
DYSTS_CACHE_SPLIT="${DYSTS_CACHE_SPLIT:-test}"
DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-100}"
HORIZONS="${HORIZONS:-}"
DYSTS_PERIODIC_REENCODE_PERIODS="${DYSTS_PERIODIC_REENCODE_PERIODS:-}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
LINE_NO=$((TASK_ID + ARRAY_OFFSET + 2))
TASK_LINE="$(sed -n "${LINE_NO}p" "${TASK_TSV}" || true)"

if [[ -z "${TASK_LINE}" ]]; then
  echo "No task row for array index ${TASK_ID} (line ${LINE_NO}) in ${TASK_TSV}. Exiting."
  exit 0
fi

TASK_EXPORTS="$(
  uv run python - "${TASK_TSV}" "${LINE_NO}" <<'PY'
import csv
import shlex
import sys

path = sys.argv[1]
line_no = int(sys.argv[2])

with open(path, newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for current_line_no, row in enumerate(reader, start=2):
        if current_line_no != line_no:
            continue
        for key, value in row.items():
            print(f"{key}={shlex.quote(value or '')}")
        break
    else:
        sys.exit(3)
PY
)"
eval "${TASK_EXPORTS}"

echo "============================================="
echo "Dysts Long-Horizon Reevaluation"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "Task Row: ${task_id}"
echo "Root: ${root_label} (${root_display_name})"
echo "Family: ${model_family}"
echo "System: ${system_key}"
echo "Seed: ${seed}"
echo "Run Dir: ${run_dir}"
echo "Output Tag: ${OUTPUT_TAG}"
if [[ -n "${HORIZONS}" ]]; then
  echo "Horizons: ${HORIZONS}"
fi
if [[ -n "${DYSTS_PERIODIC_REENCODE_PERIODS}" ]]; then
  echo "Dysts periodic reencode periods: ${DYSTS_PERIODIC_REENCODE_PERIODS}"
fi
echo "Start Time: $(date)"
echo "============================================="

CMD=(
  uv run python tools/evaluate_dysts_long_horizon_run.py
  --run-dir "${run_dir}"
  --system "${system_key}"
  --checkpoint-name "${CHECKPOINT_NAME}"
  --device "${EVAL_DEVICE}"
  --output-tag "${OUTPUT_TAG}"
  --batch-size "${BATCH_SIZE}"
  --skip-if-complete
  --dysts-cache-profile "${DYSTS_CACHE_PROFILE}"
  --dysts-cache-split "${DYSTS_CACHE_SPLIT}"
  --dysts-cache-num-workers "${DYSTS_CACHE_NUM_WORKERS}"
)

if [[ -n "${HORIZONS}" ]]; then
  HORIZONS_NORMALIZED="${HORIZONS//,/ }"
  read -r -a HORIZON_ARGS <<< "${HORIZONS_NORMALIZED}"
  CMD+=(--horizons "${HORIZON_ARGS[@]}")
fi
if [[ -n "${DYSTS_PERIODIC_REENCODE_PERIODS}" ]]; then
  PERIODS_NORMALIZED="${DYSTS_PERIODIC_REENCODE_PERIODS//,/ }"
  read -r -a PERIOD_ARGS <<< "${PERIODS_NORMALIZED}"
  CMD+=(--dysts-periodic-reencode-periods "${PERIOD_ARGS[@]}")
fi
if [[ -n "${DYSTS_CACHE_DIR}" ]]; then
  CMD+=(--dysts-cache-dir "${DYSTS_CACHE_DIR}")
fi
if [[ "${KEEP_FULL_ROLLOUTS:-0}" == "1" ]]; then
  CMD+=(--keep-full-rollouts)
fi
if [[ "${SAVE_PLOTS:-0}" == "1" ]]; then
  CMD+=(--save-plots)
fi

"${CMD[@]}"

echo "============================================="
echo "End Time: $(date)"
echo "============================================="
