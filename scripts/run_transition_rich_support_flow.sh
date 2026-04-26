#!/bin/bash
#SBATCH --job-name=tr_support_flow
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/mila/l/lia/skae}"
cd "${REPO_DIR}"

echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "Git: $(git rev-parse HEAD)"

ROWS_CSVS="${ROWS_CSVS:-results/transition_rich_basin_partition_final_seed10_20260409/collect_pass1/forecasting_rows.csv}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
ROOT_LABELS="${ROOT_LABELS:?Set ROOT_LABELS}"
SYSTEMS="${SYSTEMS:-}"
SEEDS="${SEEDS:-}"
SUPPORT_DEFINITIONS="${SUPPORT_DEFINITIONS:-relative:0.1,topk:8}"
SUBSETS="${SUBSETS:-deep,boundary}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-256}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-256}"
EVAL_SEED="${EVAL_SEED:-42}"
ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS:-5000}"
DEVICE="${DEVICE:-cpu}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.5}"
LABEL_MODES="${LABEL_MODES:-env_points,estimated_centers}"
PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS:-1}"
FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS:-0}"

echo "Rows CSVs: ${ROWS_CSVS}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Roots: ${ROOT_LABELS}"
echo "Systems: ${SYSTEMS:-<all>}"
echo "Seeds: ${SEEDS:-<all>}"
echo "Support defs: ${SUPPORT_DEFINITIONS}"
echo "Subsets: ${SUBSETS}"
echo "Label modes: ${LABEL_MODES}"

uv run python tools/diagnose_transition_rich_support_flow.py \
  --rows_csvs "${ROWS_CSVS}" \
  --output_dir "${OUTPUT_DIR}" \
  --root_labels "${ROOT_LABELS}" \
  --systems "${SYSTEMS}" \
  --seeds "${SEEDS}" \
  --support_definitions "${SUPPORT_DEFINITIONS}" \
  --subsets "${SUBSETS}" \
  --num_trajectories "${NUM_TRAJECTORIES}" \
  --trajectory_length "${TRAJECTORY_LENGTH}" \
  --eval_seed "${EVAL_SEED}" \
  --endpoint_rollout_steps "${ENDPOINT_ROLLOUT_STEPS}" \
  --device "${DEVICE}" \
  --family_jaccard_threshold "${FAMILY_JACCARD_THRESHOLD}" \
  --label_modes "${LABEL_MODES}" \
  --progress_every_runs "${PROGRESS_EVERY_RUNS}" \
  --flush_every_runs "${FLUSH_EVERY_RUNS}"
