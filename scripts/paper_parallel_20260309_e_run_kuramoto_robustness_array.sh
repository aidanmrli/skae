#!/bin/bash
#
# SLURM array runner for the Subagent E Kuramoto robustness follow-up.
#
# Required env vars:
#   TASK_TSV=<path>
#   BASE_OUT=<output root>
#
# Optional:
#   ARRAY_OFFSET=0
#   DRY_RUN=0
#
#SBATCH --job-name=e_kuramoto
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/e-kuramoto-%A_%a.out
#SBATCH -e /network/scratch/l/lia/skae/e-kuramoto-%A_%a.err
#SBATCH --requeue

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

module load cuda/12.6.0
module load cuda/12.6.0/cudnn/9.3
source .venv/bin/activate

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
DRY_RUN="${DRY_RUN:-0}"

tagify() {
  local raw="$1"
  raw="${raw//-/m}"
  raw="${raw//./p}"
  echo "${raw}"
}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
LINE_NO=$((TASK_ID + ARRAY_OFFSET + 2))
TASK_LINE="$(sed -n "${LINE_NO}p" "${TASK_TSV}" || true)"

if [[ -z "${TASK_LINE}" ]]; then
  echo "No task row for array index ${TASK_ID} (line ${LINE_NO}) in ${TASK_TSV}. Exiting."
  exit 0
fi

TASK_EXPORTS="$(
  python - "${TASK_TSV}" "${LINE_NO}" <<'PY'
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

DT_TAG="$(tagify "${env_dt}")"
SPREAD_TAG=""
if [[ -n "${kuramoto_omega_spread:-}" ]]; then
  SPREAD_TAG="$(tagify "${kuramoto_omega_spread}")"
fi

LOG_ROOT="${BASE_OUT}/${phase}/${model_variant}"
if [[ -n "${kuramoto_num_oscillators:-}" ]]; then
  LOG_ROOT="${LOG_ROOT}/n_${kuramoto_num_oscillators}"
fi
if [[ -n "${kuramoto_topology:-}" ]]; then
  LOG_ROOT="${LOG_ROOT}/topo_${kuramoto_topology}"
fi
if [[ -n "${kuramoto_omega_mode:-}" ]]; then
  LOG_ROOT="${LOG_ROOT}/omega_${kuramoto_omega_mode}"
fi
if [[ -n "${SPREAD_TAG}" ]]; then
  LOG_ROOT="${LOG_ROOT}/spread_${SPREAD_TAG}"
fi
LOG_DIR="${LOG_ROOT}/${system_slug}/dt_${DT_TAG}/seed_${seed}"
mkdir -p "${LOG_DIR}"

echo "============================================="
echo "Subagent E Kuramoto Robustness Array Runner"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "Task Row: ${task_id}"
echo "Phase: ${phase}"
echo "Model Variant: ${model_variant}"
echo "System: ${system_key}"
echo "Seed: ${seed}"
echo "DT: ${env_dt}"
echo "KURAMOTO_NUM_OSCILLATORS: ${kuramoto_num_oscillators:-<default>}"
echo "KURAMOTO_TOPOLOGY: ${kuramoto_topology:-<default>}"
echo "KURAMOTO_OMEGA_MODE: ${kuramoto_omega_mode:-<default>}"
echo "KURAMOTO_OMEGA_SPREAD: ${kuramoto_omega_spread:-<default>}"
echo "LOG_DIR: ${LOG_DIR}"
echo "Start Time: $(date)"
echo "============================================="

TRAIN_ARGS=(
  --config "${config_name}"
  --env "${env_name}"
  --env_dt "${env_dt}"
  --num_steps "${num_steps}"
  --batch_size "${batch_size}"
  --target_size "${target_size}"
  --res_coeff "${res_coeff}"
  --reconst_coeff "${reconst_coeff}"
  --pred_coeff "${pred_coeff}"
  --sparsity_coeff "${sparsity_coeff}"
  --sequence_length "${sequence_length}"
  --eval_profile "${eval_profile}"
  --seed "${seed}"
  --device cuda
  --log_dir "${LOG_DIR}"
)

if [[ -n "${lista_alpha}" ]]; then
  TRAIN_ARGS+=(--lista_alpha "${lista_alpha}")
fi
if [[ -n "${lista_num_loops}" ]]; then
  TRAIN_ARGS+=(--lista_num_loops "${lista_num_loops}")
fi
if [[ -n "${lista_final_op}" ]]; then
  TRAIN_ARGS+=(--lista_final_op "${lista_final_op}")
fi
if [[ -n "${k_structure}" ]]; then
  TRAIN_ARGS+=(--k_structure "${k_structure}")
fi
if [[ -n "${k_block_size}" ]]; then
  TRAIN_ARGS+=(--k_block_size "${k_block_size}")
fi
if [[ -n "${kuramoto_num_oscillators:-}" ]]; then
  TRAIN_ARGS+=(--kuramoto_num_oscillators "${kuramoto_num_oscillators}")
fi
if [[ -n "${kuramoto_topology:-}" ]]; then
  TRAIN_ARGS+=(--kuramoto_topology "${kuramoto_topology}")
fi
if [[ -n "${kuramoto_omega_mode:-}" ]]; then
  TRAIN_ARGS+=(--kuramoto_omega_mode "${kuramoto_omega_mode}")
fi
if [[ -n "${kuramoto_omega_spread:-}" ]]; then
  TRAIN_ARGS+=(--kuramoto_omega_spread "${kuramoto_omega_spread}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN command: UV_NO_SYNC=1 uv run python tools/train.py'
  printf ' %q' "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

UV_NO_SYNC=1 uv run python tools/train.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
