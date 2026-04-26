#!/bin/bash
#
# Run the transition-rich operator-selection study on existing checkpoints.
#
# Required env vars:
#   ROWS_CSVS=<comma-separated forecasting_rows.csv files>
#   OUT_DIR=<output directory>
#   ROOT_LABELS_CSV=<comma-separated root labels>
#
# Optional env vars:
#   SYSTEMS_CSV=<comma-separated system keys>
#   SEEDS_CSV=<comma-separated seeds>
#   SUPPORT_DEFINITIONS=absolute:0.001
#   SUBSETS=all,deep,boundary
#   PARTITION_KINDS=support,family,group
#   NUM_TRAJECTORIES=256
#   TRAJECTORY_LENGTH=256
#   ENDPOINT_ROLLOUT_STEPS=5000
#   DEVICE=cpu
#   OPERATOR_FIT_KIND=linear
#   LABEL_MODE=auto
#   MIN_OPERATOR_TRANSITIONS=128
#   FAMILY_JACCARD_THRESHOLD=0.5
#   TRAIN_FRACTION=0.5
#   NUM_RANDOM_PARTITIONS=8
#   LATENT_KMEANS_MAX_CLASSES=16
#   MAX_PARTITION_CLASSES=256
#   PROGRESS_EVERY_RUNS=1
#   FLUSH_EVERY_RUNS=0
#
#SBATCH --job-name=tr_op_select
#SBATCH --ntasks=1
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH -o /network/scratch/l/lia/skae/tr-operator-selection-%A.out
#SBATCH -e /network/scratch/l/lia/skae/tr-operator-selection-%A.err

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
SUPPORT_DEFINITIONS="${SUPPORT_DEFINITIONS:-absolute:0.001}"
SUBSETS="${SUBSETS:-all,deep,boundary}"
PARTITION_KINDS="${PARTITION_KINDS:-support,family,group}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-256}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-256}"
ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS:-5000}"
DEVICE="${DEVICE:-cpu}"
OPERATOR_FIT_KIND="${OPERATOR_FIT_KIND:-linear}"
LABEL_MODE="${LABEL_MODE:-auto}"
MIN_OPERATOR_TRANSITIONS="${MIN_OPERATOR_TRANSITIONS:-128}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.5}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.5}"
NUM_RANDOM_PARTITIONS="${NUM_RANDOM_PARTITIONS:-8}"
LATENT_KMEANS_MAX_CLASSES="${LATENT_KMEANS_MAX_CLASSES:-16}"
MAX_PARTITION_CLASSES="${MAX_PARTITION_CLASSES:-256}"
PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS:-1}"
FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS:-0}"

echo "============================================="
echo "Transition-Rich Operator Selection Study"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "ROWS_CSVS: ${ROWS_CSVS}"
echo "OUT_DIR: ${OUT_DIR}"
echo "ROOT_LABELS_CSV: ${ROOT_LABELS_CSV}"
echo "SYSTEMS_CSV: ${SYSTEMS_CSV:-<all>}"
echo "SEEDS_CSV: ${SEEDS_CSV:-<all>}"
echo "SUPPORT_DEFINITIONS: ${SUPPORT_DEFINITIONS}"
echo "SUBSETS: ${SUBSETS}"
echo "PARTITION_KINDS: ${PARTITION_KINDS}"
echo "NUM_TRAJECTORIES: ${NUM_TRAJECTORIES}"
echo "TRAJECTORY_LENGTH: ${TRAJECTORY_LENGTH}"
echo "ENDPOINT_ROLLOUT_STEPS: ${ENDPOINT_ROLLOUT_STEPS}"
echo "DEVICE: ${DEVICE}"
echo "OPERATOR_FIT_KIND: ${OPERATOR_FIT_KIND}"
echo "LABEL_MODE: ${LABEL_MODE}"
echo "MIN_OPERATOR_TRANSITIONS: ${MIN_OPERATOR_TRANSITIONS}"
echo "FAMILY_JACCARD_THRESHOLD: ${FAMILY_JACCARD_THRESHOLD}"
echo "TRAIN_FRACTION: ${TRAIN_FRACTION}"
echo "NUM_RANDOM_PARTITIONS: ${NUM_RANDOM_PARTITIONS}"
echo "LATENT_KMEANS_MAX_CLASSES: ${LATENT_KMEANS_MAX_CLASSES}"
echo "MAX_PARTITION_CLASSES: ${MAX_PARTITION_CLASSES}"
echo "PROGRESS_EVERY_RUNS: ${PROGRESS_EVERY_RUNS}"
echo "FLUSH_EVERY_RUNS: ${FLUSH_EVERY_RUNS}"
echo "============================================="

RUN_ARGS=(
  --rows_csvs "${ROWS_CSVS}"
  --output_dir "${OUT_DIR}"
  --root_labels "${ROOT_LABELS_CSV}"
  --support_definitions "${SUPPORT_DEFINITIONS}"
  --subsets "${SUBSETS}"
  --partition_kinds "${PARTITION_KINDS}"
  --num_trajectories "${NUM_TRAJECTORIES}"
  --trajectory_length "${TRAJECTORY_LENGTH}"
  --endpoint_rollout_steps "${ENDPOINT_ROLLOUT_STEPS}"
  --device "${DEVICE}"
  --operator_fit_kind "${OPERATOR_FIT_KIND}"
  --label_mode "${LABEL_MODE}"
  --min_operator_transitions "${MIN_OPERATOR_TRANSITIONS}"
  --family_jaccard_threshold "${FAMILY_JACCARD_THRESHOLD}"
  --train_fraction "${TRAIN_FRACTION}"
  --num_random_partitions "${NUM_RANDOM_PARTITIONS}"
  --latent_kmeans_max_classes "${LATENT_KMEANS_MAX_CLASSES}"
  --max_partition_classes "${MAX_PARTITION_CLASSES}"
  --progress_every_runs "${PROGRESS_EVERY_RUNS}"
  --flush_every_runs "${FLUSH_EVERY_RUNS}"
)

if [[ -n "${SYSTEMS_CSV}" ]]; then
  RUN_ARGS+=(--systems "${SYSTEMS_CSV}")
fi
if [[ -n "${SEEDS_CSV}" ]]; then
  RUN_ARGS+=(--seeds "${SEEDS_CSV}")
fi

uv run python tools/evaluate_transition_rich_operator_selection.py "${RUN_ARGS[@]}"
