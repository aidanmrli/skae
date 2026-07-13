#!/usr/bin/env bash
# Summarize a completed ManiSkill 5k LISTA tuning pilot.
#SBATCH --job-name=mskill5k_sum
#SBATCH --partition=long
#SBATCH --output=logs/maniskill_5k_sum_%j.out
#SBATCH --error=logs/maniskill_5k_sum_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/summary}"
HORIZONS="${HORIZONS:-10,20,30,40,50,75,100,125}"
PRIMARY_HORIZONS="${PRIMARY_HORIZONS:-10,20,30,40,50}"
ROLLOUT_KEY="${ROLLOUT_KEY:-best_periodic_rollout}"
EVAL_DIR_NAME="${EVAL_DIR_NAME:-eval_test_periodic}"
BASELINE_SETTING="${BASELINE_SETTING:-dense_tanh_sp0}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

echo "date=$(date)"
echo "host=$(hostname)"
echo "commit=$(git rev-parse --short HEAD || true)"
echo "run_root=${RUN_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "rollout_key=${ROLLOUT_KEY}"
echo "eval_dir_name=${EVAL_DIR_NAME}"
echo "horizons=${HORIZONS}"
echo "primary_horizons=${PRIMARY_HORIZONS}"
echo "baseline_setting=${BASELINE_SETTING}"

uv run python tools/summarize_maniskill_5k_tuning.py \
  --run-root "${RUN_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --rollout-key "${ROLLOUT_KEY}" \
  --eval-dir-name "${EVAL_DIR_NAME}" \
  --horizons "${HORIZONS}" \
  --primary-horizons "${PRIMARY_HORIZONS}" \
  --baseline-setting "${BASELINE_SETTING}"
