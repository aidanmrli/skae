#!/bin/bash
#
# Submit the true-Jacobian geometry evaluator as one SLURM job.
#
# Required env vars:
#   ROWS_CSVS=<comma-separated forecasting_rows.csv files>
#   OUT_DIR=<output directory>
#   ROOT_LABELS_CSV=<comma-separated root labels>
#
# Optional:
#   QUEUE_MANIFEST_JSON=<path to write queue submission metadata>
#   LOG_DIR=<log directory, default ${OUT_DIR}/logs>
#   WALLTIME=12:00:00
#   CPUS=4
#   MEM=24G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

ROWS_CSVS="${ROWS_CSVS:?ROWS_CSVS is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
ROOT_LABELS_CSV="${ROOT_LABELS_CSV:?ROOT_LABELS_CSV is required}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
QUEUE_MANIFEST_JSON="${QUEUE_MANIFEST_JSON:-}"
WALLTIME="${WALLTIME:-12:00:00}"
CPUS="${CPUS:-4}"
MEM="${MEM:-24G}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

JOB_ID=$(
  ROWS_CSVS="${ROWS_CSVS}" \
  OUT_DIR="${OUT_DIR}" \
  ROOT_LABELS_CSV="${ROOT_LABELS_CSV}" \
  SYSTEMS_CSV="${SYSTEMS_CSV:-}" \
  SEEDS_CSV="${SEEDS_CSV:-}" \
  SUPPORT_DEFINITIONS="${SUPPORT_DEFINITIONS:-absolute:0.001,topk:8,relative:0.1}" \
  PARTITION_KINDS="${PARTITION_KINDS:-attractor,basin,family,support}" \
  NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-128}" \
  TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-128}" \
  EVAL_SEED="${EVAL_SEED:-42}" \
  ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS:-2000}" \
  FIXED_POINT_REFINE_STEPS="${FIXED_POINT_REFINE_STEPS:-2000}" \
  FIXED_POINT_RESIDUAL_TOL="${FIXED_POINT_RESIDUAL_TOL:-1e-4}" \
  FIXED_POINT_DEDUP_TOL="${FIXED_POINT_DEDUP_TOL:-1e-3}" \
  ATTRACTOR_RADIUS="${ATTRACTOR_RADIUS:-0.75}" \
  ATTRACTOR_RADII="${ATTRACTOR_RADII:-0.25,0.5,0.75}" \
  MIN_OPERATOR_TRANSITIONS="${MIN_OPERATOR_TRANSITIONS:-32}" \
  RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-4}" \
  FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.5}" \
  NUM_RANDOM_CONTROLS="${NUM_RANDOM_CONTROLS:-4}" \
  MAX_PARTITION_CLASSES="${MAX_PARTITION_CLASSES:-128}" \
  DEVICE="${DEVICE:-cpu}" \
  LABEL_MODE="${LABEL_MODE:-auto}" \
  MAX_RUNS="${MAX_RUNS:-0}" \
  PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS:-1}" \
  FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS:-0}" \
  SMOKE="${SMOKE:-0}" \
    sbatch \
      --job-name="tr_true_jac_geo" \
      --partition=long \
      --time="${WALLTIME}" \
      --cpus-per-task="${CPUS}" \
      --mem="${MEM}" \
      --output="${LOG_DIR}/true-jac-geo-%A.out" \
      --error="${LOG_DIR}/true-jac-geo-%A.err" \
      scripts/run_transition_rich_true_jacobian_geometry.sh | awk '{print $4}'
)

if [[ -n "${QUEUE_MANIFEST_JSON}" ]]; then
  mkdir -p "$(dirname "${QUEUE_MANIFEST_JSON}")"
  cat > "${QUEUE_MANIFEST_JSON}" <<EOF
{
  "job_id": "${JOB_ID}",
  "rows_csvs": "${ROWS_CSVS}",
  "out_dir": "${OUT_DIR}",
  "root_labels_csv": "${ROOT_LABELS_CSV}",
  "systems_csv": "${SYSTEMS_CSV:-}",
  "seeds_csv": "${SEEDS_CSV:-}",
  "support_definitions": "${SUPPORT_DEFINITIONS:-absolute:0.001,topk:8,relative:0.1}",
  "partition_kinds": "${PARTITION_KINDS:-attractor,basin,family,support}",
  "attractor_radii": "${ATTRACTOR_RADII:-0.25,0.5,0.75}",
  "log_dir": "${LOG_DIR}",
  "walltime": "${WALLTIME}",
  "cpus": "${CPUS}",
  "mem": "${MEM}"
}
EOF
fi

printf 'TRUE_JACOBIAN_GEOMETRY_JOB_ID=%q\n' "${JOB_ID}"
printf 'LOG_DIR=%q\n' "${LOG_DIR}"
printf 'TRUE_JACOBIAN_GEOMETRY_ROWS_CSV=%q\n' "${OUT_DIR}/true_jacobian_geometry_rows.csv"
