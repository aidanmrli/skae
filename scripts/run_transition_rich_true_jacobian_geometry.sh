#!/bin/bash
#
# Run the true-Jacobian geometry evaluator on existing fixed-17 checkpoints.
#
# Required env vars:
#   ROWS_CSVS=<comma-separated forecasting_rows.csv files>
#   OUT_DIR=<output directory>
#   ROOT_LABELS_CSV=<comma-separated root labels>
#
# Optional env vars mirror the evaluator CLI.
#
#SBATCH --job-name=tr_true_jac_geo
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH -o /network/scratch/l/lia/skae/tr-true-jac-geo-%A.out
#SBATCH -e /network/scratch/l/lia/skae/tr-true-jac-geo-%A.err

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
ROOT_LABELS_CSV="${ROOT_LABELS_CSV:?ROOT_LABELS_CSV is required}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
SUPPORT_DEFINITIONS="${SUPPORT_DEFINITIONS:-absolute:0.001,topk:8,relative:0.1}"
PARTITION_KINDS="${PARTITION_KINDS:-attractor,basin,family,support}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-128}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-128}"
EVAL_SEED="${EVAL_SEED:-42}"
ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS:-2000}"
FIXED_POINT_REFINE_STEPS="${FIXED_POINT_REFINE_STEPS:-2000}"
FIXED_POINT_RESIDUAL_TOL="${FIXED_POINT_RESIDUAL_TOL:-1e-4}"
FIXED_POINT_DEDUP_TOL="${FIXED_POINT_DEDUP_TOL:-1e-3}"
ATTRACTOR_RADIUS="${ATTRACTOR_RADIUS:-0.75}"
ATTRACTOR_RADII="${ATTRACTOR_RADII:-0.25,0.5,0.75}"
MIN_OPERATOR_TRANSITIONS="${MIN_OPERATOR_TRANSITIONS:-32}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-4}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.5}"
NUM_RANDOM_CONTROLS="${NUM_RANDOM_CONTROLS:-4}"
MAX_PARTITION_CLASSES="${MAX_PARTITION_CLASSES:-128}"
DEVICE="${DEVICE:-cpu}"
LABEL_MODE="${LABEL_MODE:-auto}"
MAX_RUNS="${MAX_RUNS:-0}"
PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS:-1}"
FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS:-0}"
SMOKE="${SMOKE:-0}"

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "ROWS_CSVS=${ROWS_CSVS}"
echo "OUT_DIR=${OUT_DIR}"
echo "ROOT_LABELS_CSV=${ROOT_LABELS_CSV}"
echo "SYSTEMS_CSV=${SYSTEMS_CSV:-<all>}"
echo "SEEDS_CSV=${SEEDS_CSV:-<all>}"
echo "SUPPORT_DEFINITIONS=${SUPPORT_DEFINITIONS}"
echo "PARTITION_KINDS=${PARTITION_KINDS}"
echo "ATTRACTOR_RADII=${ATTRACTOR_RADII}"
echo "SMOKE=${SMOKE}"

RUN_ARGS=(
  --rows_csvs "${ROWS_CSVS}"
  --output_dir "${OUT_DIR}"
  --root_labels "${ROOT_LABELS_CSV}"
  --systems "${SYSTEMS_CSV}"
  --seeds "${SEEDS_CSV}"
  --support_definitions "${SUPPORT_DEFINITIONS}"
  --partition_kinds "${PARTITION_KINDS}"
  --num_trajectories "${NUM_TRAJECTORIES}"
  --trajectory_length "${TRAJECTORY_LENGTH}"
  --eval_seed "${EVAL_SEED}"
  --endpoint_rollout_steps "${ENDPOINT_ROLLOUT_STEPS}"
  --fixed_point_refine_steps "${FIXED_POINT_REFINE_STEPS}"
  --fixed_point_residual_tol "${FIXED_POINT_RESIDUAL_TOL}"
  --fixed_point_dedup_tol "${FIXED_POINT_DEDUP_TOL}"
  --attractor_radius "${ATTRACTOR_RADIUS}"
  --attractor_radii "${ATTRACTOR_RADII}"
  --min_operator_transitions "${MIN_OPERATOR_TRANSITIONS}"
  --ridge_lambda "${RIDGE_LAMBDA}"
  --family_jaccard_threshold "${FAMILY_JACCARD_THRESHOLD}"
  --num_random_controls "${NUM_RANDOM_CONTROLS}"
  --max_partition_classes "${MAX_PARTITION_CLASSES}"
  --device "${DEVICE}"
  --label_mode "${LABEL_MODE}"
  --max_runs "${MAX_RUNS}"
  --progress_every_runs "${PROGRESS_EVERY_RUNS}"
  --flush_every_runs "${FLUSH_EVERY_RUNS}"
)

if [[ "${SMOKE}" == "1" ]]; then
  RUN_ARGS+=(--smoke)
fi

uv run python tools/evaluate_transition_rich_true_jacobian_geometry.py "${RUN_ARGS[@]}"
