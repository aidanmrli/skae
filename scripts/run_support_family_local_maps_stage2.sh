#!/bin/bash
#
# Train stage-2 support-family local maps, or a calibrated global-map ablation,
# for existing checkpoints.
#
# Required env vars:
#   ROWS_CSVS=<comma-separated forecasting_rows.csv files>
#   OUT_DIR=<output directory>
#   ROOT_LABELS_CSV=<comma-separated root labels>
#
# Optional env vars:
#   SYSTEMS_CSV=<comma-separated system keys>
#   SEEDS_CSV=0
#   REENCODE_PERIODS=1,2,5,10
#   ROUTE_FREEZE_MODES=reroute_each_step,freeze_within_segment
#   TRAIN_STEPS=20000
#   TRAIN_BATCH_SIZE=256
#   TRAIN_POOL_TRAJECTORIES=4096
#   TRAIN_HORIZON=0
#   DEVICE=cpu  # use cuda when the sbatch allocation includes a GPU
#   SUPPORT_DEFINITION=topk:8
#   MIN_OPERATOR_TRANSITIONS=50
#   FAMILY_JACCARD_THRESHOLD=0.4
#   STAGE2_MAP_MODE=family_local_centered # or global_dense_calibrated
#   HORIZONS=100,500,1000
#   MAX_RUNTIME_SECONDS=0
#   RESUME_FROM_OUTPUT_DIRS=
#
#SBATCH --job-name=sf_local_stage2
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -o /network/scratch/l/lia/skae/sf-local-stage2-%A.out
#SBATCH -e /network/scratch/l/lia/skae/sf-local-stage2-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

ROWS_CSVS="${ROWS_CSVS:?ROWS_CSVS is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
ROOT_LABELS_CSV="${ROOT_LABELS_CSV:?ROOT_LABELS_CSV is required}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-0}"
REENCODE_PERIODS="${REENCODE_PERIODS:-1,2,5,10}"
ROUTE_FREEZE_MODES="${ROUTE_FREEZE_MODES:-reroute_each_step,freeze_within_segment}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
TRAIN_POOL_TRAJECTORIES="${TRAIN_POOL_TRAJECTORIES:-4096}"
TRAIN_POOL_SEED="${TRAIN_POOL_SEED:-20260505}"
TRAIN_HORIZON="${TRAIN_HORIZON:-0}"
FIT_NUM_TRAJECTORIES="${FIT_NUM_TRAJECTORIES:-256}"
FIT_TRAJECTORY_LENGTH="${FIT_TRAJECTORY_LENGTH:-256}"
FIT_EVAL_SEED="${FIT_EVAL_SEED:-42}"
FORECAST_NUM_TRAJECTORIES="${FORECAST_NUM_TRAJECTORIES:-128}"
FORECAST_EVAL_SEED="${FORECAST_EVAL_SEED:-314}"
SUPPORT_DEFINITION="${SUPPORT_DEFINITION:-topk:8}"
MIN_OPERATOR_TRANSITIONS="${MIN_OPERATOR_TRANSITIONS:-50}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.4}"
STAGE2_MAP_MODE="${STAGE2_MAP_MODE:-family_local_centered}"
HORIZONS="${HORIZONS:-100,500,1000}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-cpu}"
LABEL_MODE="${LABEL_MODE:-auto}"
PROGRESS_EVERY_STEPS="${PROGRESS_EVERY_STEPS:-500}"
FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS:-1}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-0}"
RESUME_FROM_OUTPUT_DIRS="${RESUME_FROM_OUTPUT_DIRS:-}"

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "ROWS_CSVS: ${ROWS_CSVS}"
echo "OUT_DIR: ${OUT_DIR}"
echo "ROOT_LABELS_CSV: ${ROOT_LABELS_CSV}"
echo "SYSTEMS_CSV: ${SYSTEMS_CSV:-<all>}"
echo "SEEDS_CSV: ${SEEDS_CSV}"
echo "REENCODE_PERIODS: ${REENCODE_PERIODS}"
echo "ROUTE_FREEZE_MODES: ${ROUTE_FREEZE_MODES}"
echo "TRAIN_STEPS: ${TRAIN_STEPS}"
echo "TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE}"
echo "TRAIN_POOL_TRAJECTORIES: ${TRAIN_POOL_TRAJECTORIES}"
echo "TRAIN_HORIZON: ${TRAIN_HORIZON}"
echo "STAGE2_MAP_MODE: ${STAGE2_MAP_MODE}"
echo "RESUME_FROM_OUTPUT_DIRS: ${RESUME_FROM_OUTPUT_DIRS:-<none>}"
echo "DEVICE: ${DEVICE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

uv run python tools/train_support_family_local_maps.py \
  --rows_csvs "${ROWS_CSVS}" \
  --output_dir "${OUT_DIR}" \
  --root_labels "${ROOT_LABELS_CSV}" \
  --systems "${SYSTEMS_CSV}" \
  --seeds "${SEEDS_CSV}" \
  --support_definition "${SUPPORT_DEFINITION}" \
  --reencode_periods "${REENCODE_PERIODS}" \
  --route_freeze_modes "${ROUTE_FREEZE_MODES}" \
  --train_steps "${TRAIN_STEPS}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --train_pool_trajectories "${TRAIN_POOL_TRAJECTORIES}" \
  --train_pool_seed "${TRAIN_POOL_SEED}" \
  --train_horizon "${TRAIN_HORIZON}" \
  --fit_num_trajectories "${FIT_NUM_TRAJECTORIES}" \
  --fit_trajectory_length "${FIT_TRAJECTORY_LENGTH}" \
  --fit_eval_seed "${FIT_EVAL_SEED}" \
  --forecast_num_trajectories "${FORECAST_NUM_TRAJECTORIES}" \
  --forecast_eval_seed "${FORECAST_EVAL_SEED}" \
  --min_operator_transitions "${MIN_OPERATOR_TRANSITIONS}" \
  --family_jaccard_threshold "${FAMILY_JACCARD_THRESHOLD}" \
  --stage2_map_mode "${STAGE2_MAP_MODE}" \
  --horizons "${HORIZONS}" \
  --lr "${LR}" \
  --device "${DEVICE}" \
  --label_mode "${LABEL_MODE}" \
  --resume_from_output_dirs "${RESUME_FROM_OUTPUT_DIRS}" \
  --progress_every_steps "${PROGRESS_EVERY_STEPS}" \
  --flush_every_runs "${FLUSH_EVERY_RUNS}" \
  --max_runtime_seconds "${MAX_RUNTIME_SECONDS}"
