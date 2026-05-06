#!/bin/bash
#
# Sweep evaluation-time support-family Jaccard thresholds on existing runs.
#
# Required env vars:
#   ROWS_CSVS=<comma-separated forecasting_rows.csv files>
#   OUT_DIR=<output directory>
#
# Optional env vars:
#   ROOT_LABELS_CSV=<comma-separated root labels>
#   SYSTEMS_CSV=<comma-separated system keys/names>
#   SEEDS_CSV=<comma-separated seeds>
#   EXCLUDE_SYSTEMS_CSV=<comma-separated system keys/names to exclude>
#   NUM_TRAJECTORIES=128
#   TRAJECTORY_LENGTH=128
#   EVAL_SEED=42
#   ENDPOINT_ROLLOUT_STEPS=5000
#   DEVICE=cpu
#   SUPPORT_DEFINITIONS=absolute:0.001,topk:8
#   FAMILY_JACCARD_THRESHOLDS=0.2,0.32,0.4,0.45,0.5,0.6,0.7,0.8,0.9
#   DEPTH_SLICE_MODE=global
#   SUBSETS=all,deep,boundary
#   FAMILY_FIT_SCOPE=all
#   PROGRESS_EVERY_RUNS=5
#   FLUSH_EVERY_RUNS=25
#
#SBATCH --job-name=sf_jaccard_sweep
#SBATCH --ntasks=1
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -o /network/scratch/l/lia/skae/support-family-jaccard-%A.out
#SBATCH -e /network/scratch/l/lia/skae/support-family-jaccard-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

ROWS_CSVS="${ROWS_CSVS:?ROWS_CSVS is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
ROOT_LABELS_CSV="${ROOT_LABELS_CSV:-}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
EXCLUDE_SYSTEMS_CSV="${EXCLUDE_SYSTEMS_CSV:-multiwell_strong_transition,claude_checkerboard_potential,claude:checkerboard_potential}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-128}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-128}"
EVAL_SEED="${EVAL_SEED:-42}"
ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS:-5000}"
DEVICE="${DEVICE:-cpu}"
SUPPORT_DEFINITIONS="${SUPPORT_DEFINITIONS:-absolute:0.001,topk:8}"
FAMILY_JACCARD_THRESHOLDS="${FAMILY_JACCARD_THRESHOLDS:-0.2,0.32,0.4,0.45,0.5,0.6,0.7,0.8,0.9}"
DEPTH_SLICE_MODE="${DEPTH_SLICE_MODE:-global}"
SUBSETS="${SUBSETS:-all,deep,boundary}"
FAMILY_FIT_SCOPE="${FAMILY_FIT_SCOPE:-all}"
PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS:-5}"
FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS:-25}"

mkdir -p "${OUT_DIR}"

echo "============================================="
echo "Support-Family Jaccard Threshold Sweep"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "ROWS_CSVS: ${ROWS_CSVS}"
echo "OUT_DIR: ${OUT_DIR}"
echo "ROOT_LABELS_CSV: ${ROOT_LABELS_CSV:-<all>}"
echo "SYSTEMS_CSV: ${SYSTEMS_CSV:-<all>}"
echo "SEEDS_CSV: ${SEEDS_CSV:-<all>}"
echo "EXCLUDE_SYSTEMS_CSV: ${EXCLUDE_SYSTEMS_CSV}"
echo "NUM_TRAJECTORIES: ${NUM_TRAJECTORIES}"
echo "TRAJECTORY_LENGTH: ${TRAJECTORY_LENGTH}"
echo "EVAL_SEED: ${EVAL_SEED}"
echo "ENDPOINT_ROLLOUT_STEPS: ${ENDPOINT_ROLLOUT_STEPS}"
echo "DEVICE: ${DEVICE}"
echo "SUPPORT_DEFINITIONS: ${SUPPORT_DEFINITIONS}"
echo "FAMILY_JACCARD_THRESHOLDS: ${FAMILY_JACCARD_THRESHOLDS}"
echo "DEPTH_SLICE_MODE: ${DEPTH_SLICE_MODE}"
echo "SUBSETS: ${SUBSETS}"
echo "FAMILY_FIT_SCOPE: ${FAMILY_FIT_SCOPE}"
echo "PROGRESS_EVERY_RUNS: ${PROGRESS_EVERY_RUNS}"
echo "FLUSH_EVERY_RUNS: ${FLUSH_EVERY_RUNS}"
echo "============================================="

ARGS=(
  --rows_csvs "${ROWS_CSVS}"
  --output_dir "${OUT_DIR}"
  --exclude_systems "${EXCLUDE_SYSTEMS_CSV}"
  --num_trajectories "${NUM_TRAJECTORIES}"
  --trajectory_length "${TRAJECTORY_LENGTH}"
  --eval_seed "${EVAL_SEED}"
  --endpoint_rollout_steps "${ENDPOINT_ROLLOUT_STEPS}"
  --device "${DEVICE}"
  --support_definitions "${SUPPORT_DEFINITIONS}"
  --family_jaccard_thresholds "${FAMILY_JACCARD_THRESHOLDS}"
  --depth_slice_mode "${DEPTH_SLICE_MODE}"
  --subsets "${SUBSETS}"
  --family_fit_scope "${FAMILY_FIT_SCOPE}"
  --progress_every_runs "${PROGRESS_EVERY_RUNS}"
  --flush_every_runs "${FLUSH_EVERY_RUNS}"
)

if [[ -n "${ROOT_LABELS_CSV}" ]]; then
  ARGS+=(--root_labels "${ROOT_LABELS_CSV}")
fi
if [[ -n "${SYSTEMS_CSV}" ]]; then
  ARGS+=(--systems "${SYSTEMS_CSV}")
fi
if [[ -n "${SEEDS_CSV}" ]]; then
  ARGS+=(--seeds "${SEEDS_CSV}")
fi

uv run python tools/sweep_support_family_jaccard.py "${ARGS[@]}"
