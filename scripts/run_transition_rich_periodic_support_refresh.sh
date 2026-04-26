#!/bin/bash
#
# Run post-entry periodic support-refresh and routed-continuation evaluation.
#
# Required env vars:
#   ROWS_CSVS=<comma-separated forecasting_rows.csv files>
#   OUT_DIR=<output directory>
#   ROOT_LABELS_CSV=<comma-separated root labels>
#
# Optional env vars:
#   SYSTEMS_CSV=<comma-separated system keys>
#   SEEDS_CSV=<comma-separated seeds>
#   SUPPORT_DEFINITIONS=absolute:0.001,topk:8
#   NUM_TRANSFERS_PER_PAIR=2
#   MAX_PAIRS_PER_SYSTEM=0
#   PRE_STEPS=32
#   BRIDGE_STEPS=32
#   POST_STEPS=128
#   CONTINUATION_HORIZON=64
#   REENCODE_PERIODS=1,8
#   START_MODES=target_entry,post_start
#   ENDPOINT_ROLLOUT_STEPS=5000
#   DEVICE=cpu
#   USE_DYNAMICS_PRIOR=0
#   SMOKE=0
#
#SBATCH --job-name=tr_refresh
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH -o /network/scratch/l/lia/skae/tr-refresh-%A.out
#SBATCH -e /network/scratch/l/lia/skae/tr-refresh-%A.err

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
SUPPORT_DEFINITIONS="${SUPPORT_DEFINITIONS:-absolute:0.001,topk:8}"
NUM_TRANSFERS_PER_PAIR="${NUM_TRANSFERS_PER_PAIR:-2}"
MAX_PAIRS_PER_SYSTEM="${MAX_PAIRS_PER_SYSTEM:-0}"
PRE_STEPS="${PRE_STEPS:-32}"
BRIDGE_STEPS="${BRIDGE_STEPS:-32}"
POST_STEPS="${POST_STEPS:-128}"
CONTINUATION_HORIZON="${CONTINUATION_HORIZON:-64}"
REFERENCE_TAIL_STEPS="${REFERENCE_TAIL_STEPS:-32}"
REENCODE_PERIODS="${REENCODE_PERIODS:-1,8}"
START_MODES="${START_MODES:-target_entry,post_start}"
SOURCE_DEPTH_FRACTION="${SOURCE_DEPTH_FRACTION:-0.12}"
TARGET_DEPTH_FRACTION="${TARGET_DEPTH_FRACTION:-0.12}"
SOURCE_PRE_MIN_FRACTION="${SOURCE_PRE_MIN_FRACTION:-0.80}"
FINAL_TARGET_MIN_FRACTION="${FINAL_TARGET_MIN_FRACTION:-0.80}"
ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS:-5000}"
EVAL_SEED="${EVAL_SEED:-42}"
DEVICE="${DEVICE:-cpu}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.5}"
USE_DYNAMICS_PRIOR="${USE_DYNAMICS_PRIOR:-0}"
MAX_SPECS="${MAX_SPECS:-0}"
PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS:-1}"
FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS:-1}"
SMOKE="${SMOKE:-0}"

PRIOR_FLAG="--no-use_dynamics_prior"
if [[ "${USE_DYNAMICS_PRIOR}" == "1" ]]; then
  PRIOR_FLAG="--use_dynamics_prior"
fi

SMOKE_FLAG=()
if [[ "${SMOKE}" == "1" ]]; then
  SMOKE_FLAG=(--smoke)
fi

echo "============================================="
echo "Transition-Rich Periodic Support Refresh"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "ROWS_CSVS: ${ROWS_CSVS}"
echo "OUT_DIR: ${OUT_DIR}"
echo "ROOT_LABELS_CSV: ${ROOT_LABELS_CSV}"
echo "SYSTEMS_CSV: ${SYSTEMS_CSV:-<all>}"
echo "SEEDS_CSV: ${SEEDS_CSV:-<all>}"
echo "SUPPORT_DEFINITIONS: ${SUPPORT_DEFINITIONS}"
echo "NUM_TRANSFERS_PER_PAIR: ${NUM_TRANSFERS_PER_PAIR}"
echo "MAX_PAIRS_PER_SYSTEM: ${MAX_PAIRS_PER_SYSTEM}"
echo "PRE_STEPS: ${PRE_STEPS}"
echo "BRIDGE_STEPS: ${BRIDGE_STEPS}"
echo "POST_STEPS: ${POST_STEPS}"
echo "CONTINUATION_HORIZON: ${CONTINUATION_HORIZON}"
echo "REENCODE_PERIODS: ${REENCODE_PERIODS}"
echo "START_MODES: ${START_MODES}"
echo "ENDPOINT_ROLLOUT_STEPS: ${ENDPOINT_ROLLOUT_STEPS}"
echo "DEVICE: ${DEVICE}"
echo "USE_DYNAMICS_PRIOR: ${USE_DYNAMICS_PRIOR}"
echo "SMOKE: ${SMOKE}"
echo "============================================="

uv run python tools/evaluate_transition_rich_periodic_support_refresh.py \
  --rows_csvs "${ROWS_CSVS}" \
  --output_dir "${OUT_DIR}" \
  --root_labels "${ROOT_LABELS_CSV}" \
  --systems "${SYSTEMS_CSV}" \
  --seeds "${SEEDS_CSV}" \
  --support_definitions "${SUPPORT_DEFINITIONS}" \
  --num_transfers_per_pair "${NUM_TRANSFERS_PER_PAIR}" \
  --max_pairs_per_system "${MAX_PAIRS_PER_SYSTEM}" \
  --pre_steps "${PRE_STEPS}" \
  --bridge_steps "${BRIDGE_STEPS}" \
  --post_steps "${POST_STEPS}" \
  --continuation_horizon "${CONTINUATION_HORIZON}" \
  --reference_tail_steps "${REFERENCE_TAIL_STEPS}" \
  --reencode_periods "${REENCODE_PERIODS}" \
  --start_modes "${START_MODES}" \
  --source_depth_fraction "${SOURCE_DEPTH_FRACTION}" \
  --target_depth_fraction "${TARGET_DEPTH_FRACTION}" \
  --source_pre_min_fraction "${SOURCE_PRE_MIN_FRACTION}" \
  --final_target_min_fraction "${FINAL_TARGET_MIN_FRACTION}" \
  --endpoint_rollout_steps "${ENDPOINT_ROLLOUT_STEPS}" \
  --eval_seed "${EVAL_SEED}" \
  --device "${DEVICE}" \
  --family_jaccard_threshold "${FAMILY_JACCARD_THRESHOLD}" \
  "${PRIOR_FLAG}" \
  --max_specs "${MAX_SPECS}" \
  --progress_every_runs "${PROGRESS_EVERY_RUNS}" \
  --flush_every_runs "${FLUSH_EVERY_RUNS}" \
  "${SMOKE_FLAG[@]}"
