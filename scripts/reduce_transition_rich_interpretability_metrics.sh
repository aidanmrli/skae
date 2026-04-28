#!/bin/bash
#
# Reduce study-plan interpretability metrics from a forecasting_rows.csv file.
#
# Required env vars:
#   ROWS_CSV=<forecasting_rows.csv>
#   OUT_DIR=<output directory>
#
# Optional:
#   ROOT_LABELS_CSV=<comma-separated root labels>
#   ROOT_LABELS_FILE=<text file with MODEL_VARIANT=PATH per line; used if ROOT_LABELS_CSV is empty>
#   SYSTEMS_CSV=<comma-separated system keys>
#   SEEDS_CSV=<comma-separated seeds>
#   NUM_TRAJECTORIES=128
#   TRAJECTORY_LENGTH=128
#   ENDPOINT_ROLLOUT_STEPS=5000
#   DEVICE=cpu
#   MIN_OPERATOR_TRANSITIONS=64
#   FAMILY_JACCARD_THRESHOLD=0.5
#   FREEZE_SUPPORT_HORIZONS=1,5,10,20
#   MAX_FREEZE_STATES=2048
#   MAX_JACOBIAN_STATES=128
#   MIN_JACOBIAN_STATES=16
#   SAVE_VISUALS=0
#   VISUAL_SUPPORTS=absolute:0.001
#   VISUAL_MAX_POINTS=5000
#   VISUAL_MAX_SWITCH_TRAJECTORIES=64
#   PROGRESS_EVERY_RUNS=1
#   FLUSH_EVERY_RUNS=0
#
#SBATCH --job-name=tr_interp_reduce
#SBATCH --ntasks=1
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH -o /network/scratch/l/lia/skae/tr-interpretability-reduce-%A.out
#SBATCH -e /network/scratch/l/lia/skae/tr-interpretability-reduce-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"

ROWS_CSV="${ROWS_CSV:?ROWS_CSV is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
ROOT_LABELS_CSV="${ROOT_LABELS_CSV:-}"
ROOT_LABELS_FILE="${ROOT_LABELS_FILE:-}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-128}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-128}"
ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS:-5000}"
DEVICE="${DEVICE:-cpu}"
MIN_OPERATOR_TRANSITIONS="${MIN_OPERATOR_TRANSITIONS:-64}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.5}"
FREEZE_SUPPORT_HORIZONS="${FREEZE_SUPPORT_HORIZONS:-1,5,10,20}"
MAX_FREEZE_STATES="${MAX_FREEZE_STATES:-2048}"
MAX_JACOBIAN_STATES="${MAX_JACOBIAN_STATES:-128}"
MIN_JACOBIAN_STATES="${MIN_JACOBIAN_STATES:-16}"
SAVE_VISUALS="${SAVE_VISUALS:-0}"
VISUAL_SUPPORTS="${VISUAL_SUPPORTS:-}"
VISUAL_MAX_POINTS="${VISUAL_MAX_POINTS:-5000}"
VISUAL_MAX_SWITCH_TRAJECTORIES="${VISUAL_MAX_SWITCH_TRAJECTORIES:-64}"
PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS:-1}"
FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS:-0}"
DEPTH_SLICE_MODE="${DEPTH_SLICE_MODE:-global}"

if [[ -z "${ROOT_LABELS_CSV}" ]]; then
  if [[ -n "${ROOT_LABELS_FILE}" ]]; then
    mapfile -t ROOT_LABEL_ROWS < "${ROOT_LABELS_FILE}"
    ROOT_LABELS=()
    for row in "${ROOT_LABEL_ROWS[@]}"; do
      [[ -z "${row}" ]] && continue
      ROOT_LABELS+=("${row%%=*}")
    done
    ROOT_LABELS_CSV="$(IFS=,; echo "${ROOT_LABELS[*]}")"
  else
    ROOT_LABELS_CSV="lista_dense_basin_partition,lista_blockdiag_basin_partition"
  fi
fi

echo "============================================="
echo "Reduce Transition-Rich Interpretability Metrics"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "ROWS_CSV: ${ROWS_CSV}"
echo "OUT_DIR: ${OUT_DIR}"
echo "ROOT_LABELS_CSV: ${ROOT_LABELS_CSV}"
echo "ROOT_LABELS_FILE: ${ROOT_LABELS_FILE:-<none>}"
echo "SYSTEMS_CSV: ${SYSTEMS_CSV:-<all>}"
echo "SEEDS_CSV: ${SEEDS_CSV:-<all>}"
echo "NUM_TRAJECTORIES: ${NUM_TRAJECTORIES}"
echo "TRAJECTORY_LENGTH: ${TRAJECTORY_LENGTH}"
echo "ENDPOINT_ROLLOUT_STEPS: ${ENDPOINT_ROLLOUT_STEPS}"
echo "DEVICE: ${DEVICE}"
echo "MIN_OPERATOR_TRANSITIONS: ${MIN_OPERATOR_TRANSITIONS}"
echo "FAMILY_JACCARD_THRESHOLD: ${FAMILY_JACCARD_THRESHOLD}"
echo "FREEZE_SUPPORT_HORIZONS: ${FREEZE_SUPPORT_HORIZONS}"
echo "MAX_FREEZE_STATES: ${MAX_FREEZE_STATES}"
echo "MAX_JACOBIAN_STATES: ${MAX_JACOBIAN_STATES}"
echo "MIN_JACOBIAN_STATES: ${MIN_JACOBIAN_STATES}"
echo "SAVE_VISUALS: ${SAVE_VISUALS}"
echo "VISUAL_SUPPORTS: ${VISUAL_SUPPORTS:-<default>}"
echo "VISUAL_MAX_POINTS: ${VISUAL_MAX_POINTS}"
echo "VISUAL_MAX_SWITCH_TRAJECTORIES: ${VISUAL_MAX_SWITCH_TRAJECTORIES}"
echo "PROGRESS_EVERY_RUNS: ${PROGRESS_EVERY_RUNS}"
echo "FLUSH_EVERY_RUNS: ${FLUSH_EVERY_RUNS}"
echo "DEPTH_SLICE_MODE: ${DEPTH_SLICE_MODE}"
echo "============================================="

REDUCE_ARGS=(
  --rows_csv "${ROWS_CSV}"
  --output_dir "${OUT_DIR}"
  --root_labels "${ROOT_LABELS_CSV}"
  --num_trajectories "${NUM_TRAJECTORIES}"
  --trajectory_length "${TRAJECTORY_LENGTH}"
  --endpoint_rollout_steps "${ENDPOINT_ROLLOUT_STEPS}"
  --device "${DEVICE}"
  --min_operator_transitions "${MIN_OPERATOR_TRANSITIONS}"
  --family_jaccard_threshold "${FAMILY_JACCARD_THRESHOLD}"
  --freeze_support_horizons "${FREEZE_SUPPORT_HORIZONS}"
  --max_freeze_states "${MAX_FREEZE_STATES}"
  --max_jacobian_states "${MAX_JACOBIAN_STATES}"
  --min_jacobian_states "${MIN_JACOBIAN_STATES}"
  --visual_max_points "${VISUAL_MAX_POINTS}"
  --visual_max_switch_trajectories "${VISUAL_MAX_SWITCH_TRAJECTORIES}"
  --progress_every_runs "${PROGRESS_EVERY_RUNS}"
  --flush_every_runs "${FLUSH_EVERY_RUNS}"
  --depth_slice_mode "${DEPTH_SLICE_MODE}"
)

if [[ -n "${SYSTEMS_CSV}" ]]; then
  REDUCE_ARGS+=(--systems "${SYSTEMS_CSV}")
fi
if [[ -n "${SEEDS_CSV}" ]]; then
  REDUCE_ARGS+=(--seeds "${SEEDS_CSV}")
fi
if [[ "${SAVE_VISUALS}" != "0" ]]; then
  REDUCE_ARGS+=(--save_visuals)
  if [[ -n "${VISUAL_SUPPORTS}" ]]; then
    REDUCE_ARGS+=(--visual_supports "${VISUAL_SUPPORTS}")
  fi
fi

uv run python tools/reduce_transition_rich_interpretability_metrics.py "${REDUCE_ARGS[@]}"
