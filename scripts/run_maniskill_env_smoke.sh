#!/usr/bin/env bash
#SBATCH --job-name=mskill_env_smoke
#SBATCH --partition=long
#SBATCH --output=logs/maniskill_env_smoke_%j.out
#SBATCH --error=logs/maniskill_env_smoke_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

echo "date=$(date)"
echo "host=$(hostname)"
echo "commit=$(git rev-parse --short HEAD || true)"
nvidia-smi || true

uv run --with mani_skill python - <<'PY'
import json
import gymnasium as gym
import mani_skill.envs  # noqa: F401

env_id = "PegInsertionSide-v1"
kwargs = {
    "obs_mode": "state",
    "control_mode": "pd_ee_delta_pose",
    "render_mode": None,
    "num_envs": 1,
}
env = gym.make(env_id, **kwargs)
obs, info = env.reset(seed=0)
action = env.action_space.sample()
step_obs, reward, terminated, truncated, step_info = env.step(action)
summary = {
    "env_id": env_id,
    "kwargs": kwargs,
    "obs_type": str(type(obs)),
    "action_shape": getattr(action, "shape", None),
    "reward": float(reward[0] if hasattr(reward, "__len__") else reward),
    "terminated": str(terminated),
    "truncated": str(truncated),
    "info_keys": sorted(list(info.keys())) if hasattr(info, "keys") else [],
    "step_info_keys": sorted(list(step_info.keys())) if hasattr(step_info, "keys") else [],
}
print(json.dumps(summary, indent=2, sort_keys=True))
env.close()
PY
