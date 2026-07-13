#!/usr/bin/env bash
#SBATCH --job-name=spatial-rd-array
#SBATCH --output=/network/scratch/l/lia/skae/spatial-rd-array-%A_%a.out
#SBATCH --error=/network/scratch/l/lia/skae/spatial-rd-array-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
EVAL_EVERY="${EVAL_EVERY:-500}"
RESUME_FROM_LATEST="${RESUME_FROM_LATEST:-1}"
K_STABILITY_WEIGHT="${K_STABILITY_WEIGHT:-1e-4}"
PERIODIC_REENCODE_PERIODS="${PERIODIC_REENCODE_PERIODS:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LINE_NO=$((TASK_ID + ARRAY_OFFSET + 2))

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
        if current_line_no == line_no:
            for key, value in row.items():
                print(f"{key}={shlex.quote(value or '')}")
            break
    else:
        sys.exit(3)
PY
)"
eval "${TASK_EXPORTS}"

mkdir -p "$(dirname "${dataset_path}")" "${run_dir}" "$(dirname "${eval_path}")"

echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse --short HEAD || true)"
echo "task_id=${task_id}"
echo "source_system=${source_system}"
echo "seed=${seed}"
echo "model_variant=${model_variant}"
echo "trainer=${trainer}"
echo "dataset_path=${dataset_path}"
echo "run_dir=${run_dir}"
echo "eval_path=${eval_path}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

if [[ "${FORCE_REGENERATE}" == "1" || ! -f "${dataset_path}" ]]; then
  dataset_lock="${dataset_path}.lock"
  if mkdir "${dataset_lock}" 2>/dev/null; then
    trap 'rm -rf "${dataset_lock}"' EXIT
    uv run python tools/generate_spatialized_reaction_diffusion.py \
      --output "${dataset_path}" \
      --source_system "${source_system}" \
      --grid_size "${grid_size}" \
      --diffusion "${diffusion}" \
      --rk4_dt "${rk4_dt}" \
      --substeps_per_observation "${substeps_per_observation}" \
      --trajectory_length "${trajectory_length}" \
      --label_extra_observations "${label_extra_observations}" \
      --train_trajectories "${train_trajectories}" \
      --val_trajectories "${val_trajectories}" \
      --test_trajectories "${test_trajectories}" \
      --laplacian_scaling "${laplacian_scaling}" \
      --seed "${seed}"
    rm -rf "${dataset_lock}"
    trap - EXIT
  else
    echo "Waiting for dataset generation lock: ${dataset_lock}"
    for _ in $(seq 1 "${DATASET_LOCK_WAIT_SECONDS:-1800}"); do
      if [[ -f "${dataset_path}" ]]; then
        break
      fi
      sleep 1
    done
    if [[ ! -f "${dataset_path}" ]]; then
      echo "Timed out waiting for dataset: ${dataset_path}" >&2
      exit 2
    fi
  fi
else
  echo "Using existing dataset: ${dataset_path}"
fi

training_summary="${run_dir}/training_summary.json"
if [[ -f "${training_summary}" && -f "${run_dir}/checkpoint.pt" ]]; then
  completed_status="$(
    uv run python - "${training_summary}" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    print(json.load(open(path)).get("status", ""))
except Exception:
    print("")
PY
  )"
else
  completed_status=""
fi

if [[ "${completed_status}" == "completed" ]]; then
  echo "Skipping completed training run: ${run_dir}"
elif [[ "${trainer}" == "conv" ]]; then
  TRAIN_EXTRA_ARGS=()
  if [[ "${RESUME_FROM_LATEST}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--resume_from_latest)
  fi
  uv run python tools/train_spatialized_reaction_diffusion_conv.py \
    --dataset "${dataset_path}" \
    --run_dir "${run_dir}" \
    --model_variant "${model_variant}" \
    --seed "${seed}" \
    --z_dim "${target_size}" \
    --hidden_channels "${hidden_channels}" \
    --num_blocks "${num_blocks}" \
    --conv_activation "${conv_activation:-}" \
    --num_steps "${num_steps}" \
    --batch_size "${batch_size}" \
    --sequence_length "${sequence_length}" \
    --train_observation_limit "${train_observation_limit:-0}" \
    --lista_num_loops "${lista_num_loops}" \
    --lista_alpha "${lista_alpha}" \
    --sparsity_weight "${sparsity_coeff}" \
    --k_stability_weight "${K_STABILITY_WEIGHT}" \
    --eval_every "${EVAL_EVERY}" \
    --eval_horizon "${eval_horizon}" \
    --device auto \
    "${TRAIN_EXTRA_ARGS[@]}"
else
  uv run python tools/train_spatialized_reaction_diffusion_lista.py \
    --dataset "${dataset_path}" \
    --run_dir "${run_dir}" \
    --config "${config_name}" \
    --seed "${seed}" \
    --target_size "${target_size}" \
    --num_steps "${num_steps}" \
    --batch_size "${batch_size}" \
    --sequence_length "${sequence_length}" \
    --lista_num_loops "${lista_num_loops}" \
    --lista_alpha "${lista_alpha}" \
    --sparsity_coeff "${sparsity_coeff}" \
    --eval_every "${EVAL_EVERY}" \
    --eval_horizon "${eval_horizon}" \
    --device auto
fi

read -r -a horizon_array <<< "${eval_horizons//,/ }"
EVAL_EXTRA_ARGS=()
if [[ -n "${PERIODIC_REENCODE_PERIODS}" ]]; then
  read -r -a reencode_period_array <<< "${PERIODIC_REENCODE_PERIODS//,/ }"
  EVAL_EXTRA_ARGS+=(--periodic_reencode_periods "${reencode_period_array[@]}")
fi
uv run python tools/evaluate_spatialized_reaction_diffusion.py \
  --dataset "${dataset_path}" \
  --checkpoint "${run_dir}/checkpoint.pt" \
  --output "${eval_path}" \
  --horizons "${horizon_array[@]}" \
  --support_threshold "${support_threshold}" \
  --family_jaccard "${family_jaccard}" \
  --max_validation_reps "${max_validation_reps}" \
  --deep_threshold "${deep_threshold}" \
  --device auto \
  "${EVAL_EXTRA_ARGS[@]}"

echo "completed_at=$(date --iso-8601=seconds)"
