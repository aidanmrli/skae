#!/usr/bin/env bash
# CPU-only data preparation for the ManiSkill-10 default-task forecasting suite.
# This downloads official demos and compacts state/action trajectories; it does
# not train models and should not consume GPUs.
#SBATCH --job-name=mskill10_prep
#SBATCH --partition=long
#SBATCH --output=logs/maniskill10_prepare_%A_%a.out
#SBATCH --error=logs/maniskill10_prepare_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9%4

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs data/maniskill/default_tasks

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"
export CUDA_VISIBLE_DEVICES=""

MANIFEST="${MANIFEST:-experiments/maniskill10_default_tasks.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/maniskill/default_tasks}"
TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
SPLIT_SEED="${SPLIT_SEED:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
OBS_KEY="${OBS_KEY:-env_states}"
MIN_STEPS="${MIN_STEPS:-2}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 2
fi

TASK_ROW="$(awk -F '\t' 'BEGIN {i=0} /^[[:space:]]*#/ {next} NF >= 3 {if (i == idx) {print; exit} i++}' idx="${TASK_INDEX}" "${MANIFEST}")"
if [[ -z "${TASK_ROW}" ]]; then
  echo "No task row for TASK_INDEX=${TASK_INDEX} in ${MANIFEST}" >&2
  exit 2
fi

TASK_ID="$(printf '%s\n' "${TASK_ROW}" | awk -F '\t' '{print $1}')"
MAX_STEPS="$(printf '%s\n' "${TASK_ROW}" | awk -F '\t' '{print $2}')"
PRIMARY_HORIZONS="$(printf '%s\n' "${TASK_ROW}" | awk -F '\t' '{print $3}')"
TASK_DIR="${OUTPUT_ROOT}/${TASK_ID}"
OUTPUT="${TASK_DIR}/${TASK_ID}_state_compact_seed${SPLIT_SEED}.npz"
SUMMARY="${OUTPUT}.summary.json"

PYTHON_WITH_MANISKILL=(uv run --with mani_skill --with h5py python)

echo "date=$(date)"
echo "host=$(hostname)"
echo "commit=$(git rev-parse --short HEAD || true)"
echo "slurm_job_id=${SLURM_JOB_ID:-}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-}"
echo "task_index=${TASK_INDEX}"
echo "task_id=${TASK_ID}"
echo "max_steps=${MAX_STEPS}"
echo "primary_horizons=${PRIMARY_HORIZONS}"
echo "obs_key=${OBS_KEY}"
echo "output=${OUTPUT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

"${PYTHON_WITH_MANISKILL[@]}" -m mani_skill.utils.download_demo "${TASK_ID}"

RAW_ROOT="${HOME}/.maniskill/demos/${TASK_ID}"
RAW_TRAJ="${RAW_TRAJ:-}"
if [[ -z "${RAW_TRAJ}" ]]; then
  candidate="${RAW_ROOT}/motionplanning/trajectory.h5"
  if [[ -f "${candidate}" ]]; then
    RAW_TRAJ="${candidate}"
  else
    RAW_TRAJ="$(
      {
        find "${RAW_ROOT}" -type f -name 'trajectory*.h5' -path '*pd_ee_delta_pose*' 2>/dev/null
        find "${RAW_ROOT}" -type f -name 'trajectory*.h5' 2>/dev/null
      } | sort -u | head -n 1 || true
    )"
  fi
fi
if [[ -z "${RAW_TRAJ}" || ! -f "${RAW_TRAJ}" ]]; then
  echo "Could not find trajectory*.h5 for ${TASK_ID} under ${RAW_ROOT}" >&2
  exit 2
fi
echo "raw_traj=${RAW_TRAJ}"

mkdir -p "${TASK_DIR}"
compact_args=(
  tools/maniskill_prepare_insertion_dataset.py
  --traj_path "${RAW_TRAJ}"
  --output "${OUTPUT}"
  --obs_key "${OBS_KEY}"
  --max_steps "${MAX_STEPS}"
  --min_steps "${MIN_STEPS}"
  --split_seed "${SPLIT_SEED}"
  --summary "${SUMMARY}"
)
if [[ -n "${MAX_EPISODES}" ]]; then
  compact_args+=(--max_episodes "${MAX_EPISODES}")
fi

"${PYTHON_WITH_MANISKILL[@]}" "${compact_args[@]}"

echo "compact_dataset=${OUTPUT}"
echo "summary=${SUMMARY}"
