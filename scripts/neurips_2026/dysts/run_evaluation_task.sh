#!/bin/bash
#
# Execute one long-horizon Dysts evaluation row inside an allocation.
#
# Required env vars:
#   TASK_TSV=<path>
#
# Optional env vars:
#   OUTPUT_TAG=dysts_dt30_h100_to_h5000_paper
#   CHECKPOINT_NAME=checkpoint
#   EVAL_DEVICE=cpu
#   DYSTS_CACHE_PROFILE=full
#   DYSTS_CACHE_SPLIT=test
#   DYSTS_CACHE_DIR=<shared scratch>/skae/dysts_native_cache
#   DYSTS_CACHE_NUM_WORKERS=2
#   BATCH_SIZE=100
#   HORIZONS="100 500 1000 1500 2000 3000 4000 5000"
#   DYSTS_PERIODIC_REENCODE_PERIODS="10 25 50 100 150 200"
#   SAVE_SELECTED_ROLLOUTS=0
#   ARRAY_OFFSET=0
#
# This payload has no SBATCH directives. Submit
# run_dysts_long_horizon_eval_array.sh for one row per array element, or the
# packed wrapper for several rows per allocation.

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source scripts/common/cluster_env.sh

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
OUTPUT_TAG="${OUTPUT_TAG:-dysts_dt30_h100_to_h5000_paper}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE:-full}"
DYSTS_CACHE_SPLIT="${DYSTS_CACHE_SPLIT:-test}"
DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-${SKAE_SCRATCH_ROOT}/dysts_native_cache}"
DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-100}"
HORIZONS="${HORIZONS:-100 500 1000 1500 2000 3000 4000 5000}"
DYSTS_PERIODIC_REENCODE_PERIODS="${DYSTS_PERIODIC_REENCODE_PERIODS:-10 25 50 100 150 200}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-}"
TASK_TSV_SHA256="${TASK_TSV_SHA256:-}"
REQUIRE_TRAINING_RECEIPT="${REQUIRE_TRAINING_RECEIPT:-0}"

if [[ -n "${SOURCE_MANIFEST}" ]]; then
  sha256sum -c "${SOURCE_MANIFEST}"
fi
if [[ -n "${TASK_TSV_SHA256}" ]]; then
  printf '%s  %s\n' "${TASK_TSV_SHA256}" "${TASK_TSV}" | sha256sum -c -
fi

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

if [[ "${REQUIRE_TRAINING_RECEIPT}" == "1" ]]; then
  uv run python - "${run_dir}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
receipt = json.loads((run_dir / "training_success.json").read_text())
checkpoint = run_dir / "checkpoint.pt"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if receipt.get("status") != "training_complete":
    raise SystemExit("invalid training receipt status")
if digest != receipt.get("best_checkpoint_sha256"):
    raise SystemExit("best checkpoint hash does not match training receipt")
PY
fi

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
  uv run skae-paper evaluate dysts
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
if [[ "${SAVE_SELECTED_ROLLOUTS:-0}" == "1" ]]; then
  CMD+=(--save-selected-rollouts)
fi
if [[ "${SAVE_PLOTS:-0}" == "1" ]]; then
  CMD+=(--save-plots)
fi

"${CMD[@]}"

echo "============================================="
echo "End Time: $(date)"
echo "============================================="
