#!/bin/bash
#
# Generic SLURM array runner for the research-paper benchmark task TSVs.
#
# Required env vars:
#   TASK_TSV=<path>
#   BASE_OUT=<output root>
#
# Optional:
#   ARRAY_OFFSET=0
#
#SBATCH --job-name=paper_bench
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/paper-bench-%A_%a.out
#SBATCH -e /network/scratch/l/lia/skae/paper-bench-%A_%a.err
#SBATCH --requeue

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
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"

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

DT_TAG="$(tagify "${env_dt}")"
LOG_ROOT="${BASE_OUT}/${phase}/${model_variant}"
if [[ -n "${kuramoto_num_oscillators:-}" ]]; then
  LOG_ROOT="${LOG_ROOT}/n_${kuramoto_num_oscillators}"
fi
if [[ -n "${hopfield_num_neurons:-}" ]]; then
  LOG_ROOT="${LOG_ROOT}/n_${hopfield_num_neurons}"
fi
if [[ -n "${competitive_lv_num_species:-}" ]]; then
  LOG_ROOT="${LOG_ROOT}/n_${competitive_lv_num_species}"
fi
LOG_DIR="${LOG_ROOT}/${system_slug}/dt_${DT_TAG}/seed_${seed}"
mkdir -p "${LOG_DIR}"

echo "============================================="
echo "Paper Benchmark Array Runner"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "Task Row: ${task_id}"
echo "Phase: ${phase}"
echo "Model Variant: ${model_variant}"
echo "System: ${system_key}"
echo "Env: ${env_name}"
echo "Seed: ${seed}"
echo "DT: ${env_dt}"
if [[ -n "${kuramoto_num_oscillators:-}" ]]; then
  echo "KURAMOTO_NUM_OSCILLATORS: ${kuramoto_num_oscillators}"
fi
if [[ -n "${hopfield_num_neurons:-}" ]]; then
  echo "HOPFIELD_NUM_NEURONS: ${hopfield_num_neurons}"
fi
if [[ -n "${hopfield_num_patterns:-}" ]]; then
  echo "HOPFIELD_NUM_PATTERNS: ${hopfield_num_patterns}"
fi
if [[ -n "${competitive_lv_num_species:-}" ]]; then
  echo "COMPETITIVE_LV_NUM_SPECIES: ${competitive_lv_num_species}"
fi
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
if [[ -n "${k_num_blocks:-}" ]]; then
  TRAIN_ARGS+=(--k_num_blocks "${k_num_blocks}")
fi
if [[ -n "${lr:-}" ]]; then
  TRAIN_ARGS+=(--lr "${lr}")
fi
if [[ -n "${k_matrix_lr:-}" ]]; then
  TRAIN_ARGS+=(--k_matrix_lr "${k_matrix_lr}")
fi
if [[ -n "${weight_decay:-}" ]]; then
  TRAIN_ARGS+=(--weight_decay "${weight_decay}")
fi
if [[ -n "${kuramoto_num_oscillators:-}" ]]; then
  TRAIN_ARGS+=(--kuramoto_num_oscillators "${kuramoto_num_oscillators}")
fi
if [[ -n "${hopfield_num_neurons:-}" ]]; then
  TRAIN_ARGS+=(--hopfield_num_neurons "${hopfield_num_neurons}")
fi
if [[ -n "${hopfield_num_patterns:-}" ]]; then
  TRAIN_ARGS+=(--hopfield_num_patterns "${hopfield_num_patterns}")
fi
if [[ -n "${competitive_lv_num_species:-}" ]]; then
  TRAIN_ARGS+=(--competitive_lv_num_species "${competitive_lv_num_species}")
fi
if [[ "${standardize:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--standardize)
fi
if [[ "${dysts_native_cache:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--dysts_native_cache)
fi
if [[ -n "${dysts_cache_profile:-}" ]]; then
  TRAIN_ARGS+=(--dysts_cache_profile "${dysts_cache_profile}")
fi
if [[ "${dysts_cache_reuse:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--dysts_cache_reuse)
fi
if [[ -n "${dysts_ic_noise_scale:-}" ]]; then
  TRAIN_ARGS+=(--dysts_ic_noise_scale "${dysts_ic_noise_scale}")
fi

uv run python tools/train.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
