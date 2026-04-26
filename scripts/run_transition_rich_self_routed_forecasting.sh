#!/bin/bash
#
# Run the self-routed local-forecasting study on existing checkpoints.
#
# Required env vars:
#   ROWS_CSVS=<comma-separated forecasting_rows.csv files>
#   OUT_DIR=<output directory>
#   ROOT_LABELS_CSV=<comma-separated root labels>
#
# Optional env vars:
#   SYSTEMS_CSV=<comma-separated system keys>
#   SEEDS_CSV=<comma-separated seeds>
#   SUPPORT_DEFINITIONS=relative:0.1,topk:8
#   DEPTH_STRATA=all,q1,q2,q3,q4
#   ROLLOUT_MODES=global_k,support_gated_k,support_block_gated_k,support_local_centered,family_local_centered
#   HORIZONS=100,500,1000
#   FIT_NUM_TRAJECTORIES=256
#   FIT_TRAJECTORY_LENGTH=256
#   FIT_EVAL_SEED=42
#   FORECAST_NUM_TRAJECTORIES=128
#   FORECAST_EVAL_SEED=314
#   ENDPOINT_ROLLOUT_STEPS=5000
#   DEVICE=cpu
#   LABEL_MODE=auto
#   RIDGE_LAMBDA=1e-4
#   MIN_OPERATOR_TRANSITIONS=128
#   FAMILY_JACCARD_THRESHOLD=0.5
#   MAX_PARTITION_CLASSES=256
#   PROGRESS_EVERY_RUNS=1
#   FLUSH_EVERY_RUNS=1
#
#SBATCH --job-name=tr_self_routed_fc
#SBATCH --ntasks=1
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH -o /network/scratch/l/lia/skae/tr-self-routed-fc-%A.out
#SBATCH -e /network/scratch/l/lia/skae/tr-self-routed-fc-%A.err

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

ROWS_CSVS="${ROWS_CSVS:?ROWS_CSVS is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
ROOT_LABELS_CSV="${ROOT_LABELS_CSV:?ROOT_LABELS_CSV is required}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
SUPPORT_DEFINITIONS="${SUPPORT_DEFINITIONS:-relative:0.1,topk:8}"
DEPTH_STRATA="${DEPTH_STRATA:-all,q1,q2,q3,q4}"
ROLLOUT_MODES="${ROLLOUT_MODES:-global_k,support_gated_k,support_block_gated_k,support_local_centered,family_local_centered}"
HORIZONS="${HORIZONS:-100,500,1000}"
FIT_NUM_TRAJECTORIES="${FIT_NUM_TRAJECTORIES:-256}"
FIT_TRAJECTORY_LENGTH="${FIT_TRAJECTORY_LENGTH:-256}"
FIT_EVAL_SEED="${FIT_EVAL_SEED:-42}"
FORECAST_NUM_TRAJECTORIES="${FORECAST_NUM_TRAJECTORIES:-128}"
FORECAST_EVAL_SEED="${FORECAST_EVAL_SEED:-314}"
ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS:-5000}"
DEVICE="${DEVICE:-cpu}"
LABEL_MODE="${LABEL_MODE:-auto}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-4}"
MIN_OPERATOR_TRANSITIONS="${MIN_OPERATOR_TRANSITIONS:-128}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.5}"
MAX_PARTITION_CLASSES="${MAX_PARTITION_CLASSES:-256}"
PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS:-1}"
FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS:-1}"

echo "============================================="
echo "Transition-Rich Self-Routed Local Forecasting"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "ROWS_CSVS: ${ROWS_CSVS}"
echo "OUT_DIR: ${OUT_DIR}"
echo "ROOT_LABELS_CSV: ${ROOT_LABELS_CSV}"
echo "SYSTEMS_CSV: ${SYSTEMS_CSV:-<all>}"
echo "SEEDS_CSV: ${SEEDS_CSV:-<all>}"
echo "SUPPORT_DEFINITIONS: ${SUPPORT_DEFINITIONS}"
echo "DEPTH_STRATA: ${DEPTH_STRATA}"
echo "ROLLOUT_MODES: ${ROLLOUT_MODES}"
echo "HORIZONS: ${HORIZONS}"
echo "FIT_NUM_TRAJECTORIES: ${FIT_NUM_TRAJECTORIES}"
echo "FIT_TRAJECTORY_LENGTH: ${FIT_TRAJECTORY_LENGTH}"
echo "FORECAST_NUM_TRAJECTORIES: ${FORECAST_NUM_TRAJECTORIES}"
echo "ENDPOINT_ROLLOUT_STEPS: ${ENDPOINT_ROLLOUT_STEPS}"
echo "DEVICE: ${DEVICE}"
echo "LABEL_MODE: ${LABEL_MODE}"
echo "RIDGE_LAMBDA: ${RIDGE_LAMBDA}"
echo "MIN_OPERATOR_TRANSITIONS: ${MIN_OPERATOR_TRANSITIONS}"
echo "FAMILY_JACCARD_THRESHOLD: ${FAMILY_JACCARD_THRESHOLD}"
echo "MAX_PARTITION_CLASSES: ${MAX_PARTITION_CLASSES}"
echo "============================================="

uv run python tools/evaluate_transition_rich_self_routed_forecasting.py \
  --rows_csvs "${ROWS_CSVS}" \
  --output_dir "${OUT_DIR}" \
  --root_labels "${ROOT_LABELS_CSV}" \
  --systems "${SYSTEMS_CSV}" \
  --seeds "${SEEDS_CSV}" \
  --support_definitions "${SUPPORT_DEFINITIONS}" \
  --depth_strata "${DEPTH_STRATA}" \
  --rollout_modes "${ROLLOUT_MODES}" \
  --horizons "${HORIZONS}" \
  --fit_num_trajectories "${FIT_NUM_TRAJECTORIES}" \
  --fit_trajectory_length "${FIT_TRAJECTORY_LENGTH}" \
  --fit_eval_seed "${FIT_EVAL_SEED}" \
  --forecast_num_trajectories "${FORECAST_NUM_TRAJECTORIES}" \
  --forecast_eval_seed "${FORECAST_EVAL_SEED}" \
  --endpoint_rollout_steps "${ENDPOINT_ROLLOUT_STEPS}" \
  --device "${DEVICE}" \
  --label_mode "${LABEL_MODE}" \
  --ridge_lambda "${RIDGE_LAMBDA}" \
  --min_operator_transitions "${MIN_OPERATOR_TRANSITIONS}" \
  --family_jaccard_threshold "${FAMILY_JACCARD_THRESHOLD}" \
  --max_partition_classes "${MAX_PARTITION_CLASSES}" \
  --progress_every_runs "${PROGRESS_EVERY_RUNS}" \
  --flush_every_runs "${FLUSH_EVERY_RUNS}"
