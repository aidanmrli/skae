#!/bin/bash
#
# Run the explicit regime-discovery local-Koopman comparison on existing checkpoints.
#
# Required env vars:
#   ROWS_CSVS=<comma-separated forecasting_rows.csv files>
#   OUT_DIR=<output directory>
#   ROOT_LABELS_CSV=<comma-separated root labels>
#
# Optional env vars:
#   SYSTEMS_CSV=<comma-separated system keys>
#   SEEDS_CSV=<comma-separated seeds>
#   SUPPORT_DEFINITION=topk:8
#   FEATURE_VIEWS=raw_state,dense_latent,sparse_latent_values,support_binary
#   CLUSTER_METHODS=kmeans,gmm_diag,spectral
#   CLUSTER_COUNT_MODES=basin_count,support_family_count
#   NUM_TRAJECTORIES=256
#   TRAJECTORY_LENGTH=256
#   EVAL_SEED=42
#   ENDPOINT_ROLLOUT_STEPS=5000
#   DEVICE=cpu
#   LABEL_MODE=auto
#   RIDGE_LAMBDA=1e-4
#   MIN_OPERATOR_TRANSITIONS=128
#   FAMILY_JACCARD_THRESHOLD=0.5
#   TRAIN_FRACTION=0.5
#   CLUSTER_FIT_MAX_SAMPLES=4096
#   SPECTRAL_NEIGHBORS=20
#   DECODE_BATCH_SIZE=4096
#   PROGRESS_EVERY_RUNS=1
#   FLUSH_EVERY_RUNS=1
#   MAX_RUNTIME_SECONDS=0
#
#SBATCH --job-name=tr_regime_local_k
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/tr-regime-local-k-%A.out
#SBATCH -e /network/scratch/l/lia/skae/tr-regime-local-k-%A.err

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
SUPPORT_DEFINITION="${SUPPORT_DEFINITION:-topk:8}"
FEATURE_VIEWS="${FEATURE_VIEWS:-raw_state,dense_latent,sparse_latent_values,support_binary}"
CLUSTER_METHODS="${CLUSTER_METHODS:-kmeans,gmm_diag,spectral}"
CLUSTER_COUNT_MODES="${CLUSTER_COUNT_MODES:-basin_count,support_family_count}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-256}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-256}"
EVAL_SEED="${EVAL_SEED:-42}"
ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS:-5000}"
DEVICE="${DEVICE:-cpu}"
LABEL_MODE="${LABEL_MODE:-auto}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-4}"
MIN_OPERATOR_TRANSITIONS="${MIN_OPERATOR_TRANSITIONS:-128}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.5}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.5}"
CLUSTER_FIT_MAX_SAMPLES="${CLUSTER_FIT_MAX_SAMPLES:-4096}"
SPECTRAL_NEIGHBORS="${SPECTRAL_NEIGHBORS:-20}"
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-4096}"
PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS:-1}"
FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS:-1}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-0}"

echo "============================================="
echo "Regime-Discovery Local Koopman Comparison"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "ROWS_CSVS: ${ROWS_CSVS}"
echo "OUT_DIR: ${OUT_DIR}"
echo "ROOT_LABELS_CSV: ${ROOT_LABELS_CSV}"
echo "SYSTEMS_CSV: ${SYSTEMS_CSV:-<all>}"
echo "SEEDS_CSV: ${SEEDS_CSV:-<all>}"
echo "SUPPORT_DEFINITION: ${SUPPORT_DEFINITION}"
echo "FEATURE_VIEWS: ${FEATURE_VIEWS}"
echo "CLUSTER_METHODS: ${CLUSTER_METHODS}"
echo "CLUSTER_COUNT_MODES: ${CLUSTER_COUNT_MODES}"
echo "NUM_TRAJECTORIES: ${NUM_TRAJECTORIES}"
echo "TRAJECTORY_LENGTH: ${TRAJECTORY_LENGTH}"
echo "DEVICE: ${DEVICE}"
echo "LABEL_MODE: ${LABEL_MODE}"
echo "TRAIN_FRACTION: ${TRAIN_FRACTION}"
echo "CLUSTER_FIT_MAX_SAMPLES: ${CLUSTER_FIT_MAX_SAMPLES}"
echo "MAX_RUNTIME_SECONDS: ${MAX_RUNTIME_SECONDS}"
echo "============================================="

uv run python tools/evaluate_regime_discovery_local_koopman.py \
  --rows_csvs "${ROWS_CSVS}" \
  --output_dir "${OUT_DIR}" \
  --root_labels "${ROOT_LABELS_CSV}" \
  --systems "${SYSTEMS_CSV}" \
  --seeds "${SEEDS_CSV}" \
  --support_definition "${SUPPORT_DEFINITION}" \
  --feature_views "${FEATURE_VIEWS}" \
  --cluster_methods "${CLUSTER_METHODS}" \
  --cluster_count_modes "${CLUSTER_COUNT_MODES}" \
  --num_trajectories "${NUM_TRAJECTORIES}" \
  --trajectory_length "${TRAJECTORY_LENGTH}" \
  --eval_seed "${EVAL_SEED}" \
  --endpoint_rollout_steps "${ENDPOINT_ROLLOUT_STEPS}" \
  --device "${DEVICE}" \
  --label_mode "${LABEL_MODE}" \
  --ridge_lambda "${RIDGE_LAMBDA}" \
  --min_operator_transitions "${MIN_OPERATOR_TRANSITIONS}" \
  --family_jaccard_threshold "${FAMILY_JACCARD_THRESHOLD}" \
  --train_fraction "${TRAIN_FRACTION}" \
  --cluster_fit_max_samples "${CLUSTER_FIT_MAX_SAMPLES}" \
  --spectral_neighbors "${SPECTRAL_NEIGHBORS}" \
  --decode_batch_size "${DECODE_BATCH_SIZE}" \
  --progress_every_runs "${PROGRESS_EVERY_RUNS}" \
  --flush_every_runs "${FLUSH_EVERY_RUNS}" \
  --max_runtime_seconds "${MAX_RUNTIME_SECONDS}"
