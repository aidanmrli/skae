#!/usr/bin/env bash
#SBATCH --job-name=mskill_prep
#SBATCH --partition=long
#SBATCH --output=logs/maniskill_insertion_prepare_%j.out
#SBATCH --error=logs/maniskill_insertion_prepare_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs data/maniskill
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

echo "date=$(date)"
echo "host=$(hostname)"
echo "commit=$(git rev-parse --short HEAD || true)"

ENV_ID="${ENV_ID:-PegInsertionSide-v1}"
RAW_TRAJ="${RAW_TRAJ:-${HOME}/.maniskill/demos/${ENV_ID}/motionplanning/trajectory.h5}"
REPLAY_TRAJ="${REPLAY_TRAJ:-}"
USE_REPLAY="${USE_REPLAY:-0}"
OBS_KEY="${OBS_KEY:-env_states}"
OUTPUT="${OUTPUT:-data/maniskill/${ENV_ID}_state_compact_seed0.npz}"
CONTROL_MODE="${CONTROL_MODE:-}"
COUNT="${COUNT:-64}"
MAX_STEPS="${MAX_STEPS:-150}"
SPLIT_SEED="${SPLIT_SEED:-0}"
PYTHON_WITH_MANISKILL=(uv run --with mani_skill --with h5py python)

"${PYTHON_WITH_MANISKILL[@]}" -m mani_skill.utils.download_demo "${ENV_ID}"

if [[ -z "${REPLAY_TRAJ}" && "${USE_REPLAY}" == "1" ]]; then
  replay_args=(
    -m mani_skill.trajectory.replay_trajectory
    --traj-path "${RAW_TRAJ}" \
    -o state \
    --use-env-states \
    --record-rewards \
    --save-traj \
    --count "${COUNT}"
  )
  if [[ -n "${CONTROL_MODE}" ]]; then
    replay_args+=(-c "${CONTROL_MODE}")
  fi
  "${PYTHON_WITH_MANISKILL[@]}" "${replay_args[@]}"

  echo "Set REPLAY_TRAJ to the replayed state .h5 path printed by ManiSkill, then rerun compaction if auto-discovery fails."
  REPLAY_TRAJ="$(find "$(dirname "${RAW_TRAJ}")" -maxdepth 1 -type f -name 'trajectory.state.*.h5' | sort | tail -n 1 || true)"
fi

COMPACT_TRAJ="${REPLAY_TRAJ:-${RAW_TRAJ}}"
if [[ -z "${COMPACT_TRAJ}" || ! -f "${COMPACT_TRAJ}" ]]; then
  echo "Could not find trajectory for compaction. Expected COMPACT_TRAJ=${COMPACT_TRAJ}" >&2
  exit 2
fi

"${PYTHON_WITH_MANISKILL[@]}" tools/maniskill_prepare_insertion_dataset.py \
  --traj_path "${COMPACT_TRAJ}" \
  --output "${OUTPUT}" \
  --obs_key "${OBS_KEY}" \
  --max_steps "${MAX_STEPS}" \
  --split_seed "${SPLIT_SEED}"

echo "compact_dataset=${OUTPUT}"
