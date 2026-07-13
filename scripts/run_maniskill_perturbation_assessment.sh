#!/usr/bin/env bash
#SBATCH --job-name=mskill_perturb_eval
#SBATCH --partition=long
#SBATCH --output=logs/maniskill_perturbation_assessment_%j.out
#SBATCH --error=logs/maniskill_perturbation_assessment_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs data/maniskill runs/maniskill_insertion/perturbation_assessment_seed0
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

echo "date=$(date)"
echo "host=$(hostname)"
echo "commit=$(git rev-parse --short HEAD || true)"
nvidia-smi || true

SEED="${SEED:-0}"
DEMO_H5="${DEMO_H5:-${HOME}/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5}"
DEMO_JSON="${DEMO_JSON:-${HOME}/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.json}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"
EPISODE_COUNT="${EPISODE_COUNT:-1}"
MAX_STEPS="${MAX_STEPS:-100}"
SETTLE_STEPS="${SETTLE_STEPS:-20}"
ENV_MAX_EPISODE_STEPS="${ENV_MAX_EPISODE_STEPS:-}"
SETUPS="${SETUPS:-success,jam,miss,drop,partial}"
DATA_ROOT="${DATA_ROOT:-data/maniskill/perturbation_assessment_seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-runs/maniskill_insertion/perturbation_assessment_seed${SEED}}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.70}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-0.2}"
FAMILY_JACCARD="${FAMILY_JACCARD:-0.4}"
EVALUATE="${EVALUATE:-1}"

PYTHON_WITH_MANISKILL=(uv run --with mani_skill --with h5py python)

generate_args=(
  tools/maniskill_generate_perturbed_rollouts.py
  --demo_h5 "${DEMO_H5}" \
  --demo_json "${DEMO_JSON}" \
  --output_dir "${DATA_ROOT}" \
  --episode_index "${EPISODE_INDEX}" \
  --episode_count "${EPISODE_COUNT}" \
  --seed "${SEED}" \
  --max_steps "${MAX_STEPS}" \
  --settle_steps "${SETTLE_STEPS}" \
  --train_fraction "${TRAIN_FRACTION}" \
  --val_fraction "${VAL_FRACTION}" \
  --setups "${SETUPS}"
)
if [[ -n "${ENV_MAX_EPISODE_STEPS}" ]]; then
  generate_args+=(--env_max_episode_steps "${ENV_MAX_EPISODE_STEPS}")
fi
"${PYTHON_WITH_MANISKILL[@]}" "${generate_args[@]}"

if [[ "${EVALUATE}" != "1" ]]; then
  echo "Skipping checkpoint evaluation because EVALUATE=${EVALUATE}"
  echo "data_root=${DATA_ROOT}"
  exit 0
fi

declare -A CHECKPOINTS
CHECKPOINTS[lista]="runs/maniskill_insertion/tuning_alpha0p2_sp0p03_thr0p1_j0p7/checkpoint.pt"
CHECKPOINTS[dense]="runs/maniskill_insertion/dense_controlled_seed0/checkpoint.pt"
CHECKPOINTS[sparse_mlp]="runs/maniskill_insertion/sparse_mlp_controlled_seed0/checkpoint.pt"

IFS=',' read -r -a SETUP_ARRAY <<< "${SETUPS}"
for model_name in lista dense sparse_mlp; do
  checkpoint="${CHECKPOINTS[${model_name}]}"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint for ${model_name}: ${checkpoint}" >&2
    exit 2
  fi
  for setup in "${SETUP_ARRAY[@]}"; do
    dataset="${DATA_ROOT}/${setup}.npz"
    output_dir="${RESULT_ROOT}/${model_name}/${setup}"
    uv run python tools/evaluate_maniskill_controlled_lista.py \
      --dataset "${dataset}" \
      --checkpoint "${checkpoint}" \
      --output_dir "${output_dir}" \
      --device cuda \
      --split test \
      --horizons 10,25,50,100 \
      --support_threshold "${SUPPORT_THRESHOLD}" \
      --family_jaccard "${FAMILY_JACCARD}"
  done
done

echo "data_root=${DATA_ROOT}"
echo "result_root=${RESULT_ROOT}"
