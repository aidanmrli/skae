#!/usr/bin/env bash
#
# CPU array runner for state-observation control world-model tasks.
#
# Required env vars:
#   TASK_TSV=<path to headered TSV from tools/build_control_world_model_tasks.py>
#
# Optional:
#   ARRAY_OFFSET=0
#   DMC_UV_EXTRA=control     # set empty to use DMC_UV_WITH or current env
#   DMC_UV_WITH=             # optional direct package spec fallback
#
#SBATCH --job-name=ctrl-wm
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH -o /network/scratch/l/lia/skae/control-wm-%A_%a.out
#SBATCH -e /network/scratch/l/lia/skae/control-wm-%A_%a.err
#SBATCH --requeue

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
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

EXPECTED_FIELDS = {
    "task_id",
    "task",
    "variant",
    "seed",
    "data_fraction",
    "dataset",
    "dataset_summary",
    "run_dir",
    "dataset_seed",
    "num_episodes",
    "episode_length",
    "train_fraction",
    "val_fraction",
    "num_steps",
    "batch_size",
    "sequence_length",
    "eval_horizons",
    "z_dim",
    "hidden_dim",
    "lr",
    "weight_decay",
    "eval_every",
    "planning_candidates",
}

path = sys.argv[1]
line_no = int(sys.argv[2])
with open(path, newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for current_line_no, row in enumerate(reader, start=2):
        if current_line_no == line_no:
            unexpected = set(row) - EXPECTED_FIELDS
            missing = EXPECTED_FIELDS - set(row)
            if unexpected or missing:
                raise SystemExit(
                    f"Unexpected TSV fields={sorted(unexpected)} missing={sorted(missing)}"
                )
            for key, value in row.items():
                print(f"{key}={shlex.quote(value or '')}")
            break
    else:
        sys.exit(3)
PY
)"
eval "${TASK_EXPORTS}"

echo "============================================="
echo "Control World Model Runner"
echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array task: ${TASK_ID}"
echo "Task row: ${task_id}"
echo "DMC task: ${task}"
echo "Variant: ${variant}"
echo "Seed: ${seed}"
echo "Data fraction: ${data_fraction}"
echo "Dataset: ${dataset}"
echo "Run dir: ${run_dir}"
echo "Start time: $(date)"
echo "============================================="

if [[ -f "${run_dir}/final_metrics.json" && "${SKIP_COMPLETED:-1}" == "1" ]]; then
  echo "Completed run already exists: ${run_dir}/final_metrics.json"
  exit 0
fi

if [[ ! -f "${run_dir}/final_metrics.json" && -e "${run_dir}/metrics_history.jsonl" ]]; then
  BASE_RUN_DIR="${run_dir}"
  run_dir="${BASE_RUN_DIR}/attempt_${SLURM_JOB_ID:-local}_${TASK_ID}_${SLURM_RESTART_COUNT:-0}"
  echo "Existing partial run detected; writing this attempt to ${run_dir}"
fi

mkdir -p "$(dirname "${dataset}")" "$(dirname "${dataset_summary}")" "${run_dir}"

if [[ ! -f "${dataset}" ]]; then
  LOCK_PATH="${dataset}.lock"
  exec 9>"${LOCK_PATH}"
  flock 9
  if [[ ! -f "${dataset}" ]]; then
    echo "Generating dataset ${dataset}"
    GEN_CMD=(uv run)
    if [[ -n "${DMC_UV_EXTRA:-control}" ]]; then
      GEN_CMD+=(--extra "${DMC_UV_EXTRA:-control}")
    elif [[ -n "${DMC_UV_WITH:-}" ]]; then
      GEN_CMD+=(--with "${DMC_UV_WITH}")
    fi
    GEN_CMD+=(
      python tools/generate_dm_control_state_dataset.py
      --task "${task}"
      --output "${dataset}"
      --summary_json "${dataset_summary}"
      --num_episodes "${num_episodes}"
      --episode_length "${episode_length}"
      --seed "${dataset_seed}"
      --train_fraction "${train_fraction}"
      --val_fraction "${val_fraction}"
    )
    "${GEN_CMD[@]}"
  else
    echo "Dataset appeared while waiting for lock: ${dataset}"
  fi
  flock -u 9
else
  echo "Using existing dataset ${dataset}"
fi

uv run python tools/train_control_world_model.py \
  --dataset "${dataset}" \
  --run_dir "${run_dir}" \
  --variant "${variant}" \
  --seed "${seed}" \
  --device cpu \
  --data_fraction "${data_fraction}" \
  --num_steps "${num_steps}" \
  --batch_size "${batch_size}" \
  --sequence_length "${sequence_length}" \
  --eval_horizons "${eval_horizons}" \
  --z_dim "${z_dim}" \
  --hidden_dim "${hidden_dim}" \
  --lr "${lr}" \
  --weight_decay "${weight_decay}" \
  --eval_every "${eval_every}" \
  --planning_candidates "${planning_candidates}"

echo "End time: $(date)"
