#!/bin/bash
#
# SLURM array runner for standalone paper baseline task TSVs.
#
# Required env vars:
#   TASK_TSV=<path>
#   BASE_OUT=<output root>
#
# Optional:
#   ARRAY_OFFSET=0
#
#SBATCH --job-name=paper_base
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH -o /network/scratch/l/lia/skae/paper-baseline-%A_%a.out
#SBATCH -e /network/scratch/l/lia/skae/paper-baseline-%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Start Time: $(date)"

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"

tagify() {
  local raw="$1"
  raw="${raw//:/_}"
  raw="${raw//\//_}"
  raw="${raw//./p}"
  raw="${raw//-/_}"
  echo "${raw}"
}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
LINE_NO=$((TASK_ID + ARRAY_OFFSET + 2))
TASK_LINE="$(sed -n "${LINE_NO}p" "${TASK_TSV}" || true)"

if [[ -z "${TASK_LINE}" ]]; then
  echo "No task row for array index ${TASK_ID} (line ${LINE_NO}) in ${TASK_TSV}. Exiting."
  exit 1
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

SYSTEM_SLUG="$(tagify "${system}")"
OUT_DIR="${BASE_OUT}/${baseline_family}/${SYSTEM_SLUG}/seed_${seed}"
mkdir -p "${OUT_DIR}"

echo "============================================="
echo "Paper Baseline Suite"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "Task Row: ${task_id}"
echo "Family: ${baseline_family}"
echo "System: ${system}"
echo "Seed: ${seed}"
echo "Methods: ${methods}"
echo "Output: ${OUT_DIR}"
echo "============================================="

COMMON_ARGS=(
  --systems "${system}"
  --seeds "${seed}"
  --methods "${methods}"
  --horizons "${horizons}"
  --num_trajectories "${num_trajectories}"
  --trajectory_length "${trajectory_length}"
  --train_fraction "${train_fraction}"
  --ridge_lambda "${ridge_lambda}"
  --output_dir "${OUT_DIR}"
  --env_dt "${env_dt}"
  --dysts_dt_multiplier "${dysts_dt_multiplier:-0.0}"
  --dysts_standardize "${dysts_standardize:-0}"
)

if [[ "${baseline_family}" == "classical_koopman" ]]; then
  uv run python tools/evaluate_classical_koopman_baselines.py \
    "${COMMON_ARGS[@]}" \
    --allow_non_2d \
    --edmd_degree "${edmd_degree}" \
    --kernel_centers "${kernel_centers}" \
    --kernel_gamma "${kernel_gamma}" \
    --max_train_pairs "${max_train_pairs}" \
    --config_name "${config_name}" \
    --torch_threads "${torch_threads}"
elif [[ "${baseline_family}" == "mixture_local_linear" ]]; then
  uv run python tools/evaluate_mixture_local_linear_baselines.py \
    "${COMMON_ARGS[@]}" \
    --num_components "${num_components}" \
    --component_mode "${component_mode}" \
    --config "${config_name}" \
    --torch_num_threads "${torch_threads}"
elif [[ "${baseline_family}" == "local_edmd_koopman" ]]; then
  uv run python tools/evaluate_local_edmd_koopman_baselines.py \
    "${COMMON_ARGS[@]}" \
    --allow_non_2d \
    --edmd_degree "${edmd_degree}" \
    --kernel_centers "${kernel_centers}" \
    --kernel_gamma "${kernel_gamma}" \
    --max_train_pairs "${max_train_pairs}" \
    --num_components_grid "${local_num_components_grid}" \
    --validation_fraction "${local_validation_fraction}" \
    --selection_horizons "${local_selection_horizons}" \
    --min_component_transitions "${local_min_component_transitions}" \
    --max_abs_state_for_fit "${local_max_abs_state_for_fit}" \
    --config_name "${config_name}" \
    --torch_threads "${torch_threads}"
else
  echo "Unknown baseline_family: ${baseline_family}" >&2
  exit 2
fi

echo "End Time: $(date)"
