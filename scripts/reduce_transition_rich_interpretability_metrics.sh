#!/bin/bash
#
# Reduce the paper's fixed controlled basin/support alignment protocol.
#
# Required env vars:
#   ROWS_CSV=<forecasting_rows.csv>
#   OUT_DIR=<output directory>
#
# Optional execution-scope vars:
#   ROOT_LABELS_CSV=<comma-separated paper row identifiers>
#   ROOT_LABELS_FILE=<MODEL_VARIANT=PATH rows; used if CSV is empty>
#   SYSTEMS_CSV=<comma-separated controlled systems>
#   SEEDS_CSV=<comma-separated seeds>
#   DEVICE=cpu
#   PROGRESS_EVERY_RUNS=1
#   FLUSH_EVERY_RUNS=0
#
# Scientific settings are not configurable here: evaluation seed 42; 128
# trajectories with 128 transitions; absolute support 1e-3; Jaccard 0.50;
# native labels/centers for the two gated systems; endpoint-estimated proxy
# centers after 5,000 steps for the 13 catalog systems; and the tie-inclusive
# within-label high-center-margin slice scored with natural-log entropy.
#
#SBATCH --job-name=tr_align_reduce
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH -o /network/scratch/l/lia/skae/tr-alignment-reduce-%A.out
#SBATCH -e /network/scratch/l/lia/skae/tr-alignment-reduce-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"
source .venv/bin/activate

ROWS_CSV="${ROWS_CSV:?ROWS_CSV is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
ROOT_LABELS_CSV="${ROOT_LABELS_CSV:-}"
ROOT_LABELS_FILE="${ROOT_LABELS_FILE:-}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
DEVICE="${DEVICE:-cpu}"
PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS:-1}"
FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS:-0}"

if [[ -z "${ROOT_LABELS_CSV}" ]]; then
  if [[ -n "${ROOT_LABELS_FILE}" ]]; then
    mapfile -t ROOT_LABELS < <(
      awk -F= 'NF>=1 && $1!="" && !seen[$1]++ {print $1}' "${ROOT_LABELS_FILE}"
    )
    ROOT_LABELS_CSV="$(IFS=,; echo "${ROOT_LABELS[*]}")"
  else
    ROOT_LABELS_CSV="lista_dense_signsplit_p256_hardinit_basin_partition,lista_blockdiag_signsplit_hardinit_basin_partition,lista_dense_softblock_signsplit_p256_hardinit_basin_partition,mlp_sparse_blockdiag_hardinit_basin_partition_control,mlp_sparse_hardinit_basin_partition_control,mlp_zero_sparse_hardinit_basin_partition_control"
  fi
fi

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Rows: ${ROWS_CSV}"
echo "Output: ${OUT_DIR}"
echo "Roots: ${ROOT_LABELS_CSV}"
echo "Systems: ${SYSTEMS_CSV:-<all paper systems>}"
echo "Seeds: ${SEEDS_CSV:-<all discovered seeds>}"
echo "Protocol: seed=42 trajectories=128 transitions=128 endpoint_steps=5000 support=absolute:0.001 scoring=within-label-margin>=q75-tie-inclusive Jaccard=0.50 entropy=nats"

ARGS=(
  --rows_csv "${ROWS_CSV}"
  --output_dir "${OUT_DIR}"
  --root_labels "${ROOT_LABELS_CSV}"
  --device "${DEVICE}"
  --progress_every_runs "${PROGRESS_EVERY_RUNS}"
  --flush_every_runs "${FLUSH_EVERY_RUNS}"
)
if [[ -n "${SYSTEMS_CSV}" ]]; then
  ARGS+=(--systems "${SYSTEMS_CSV}")
fi
if [[ -n "${SEEDS_CSV}" ]]; then
  ARGS+=(--seeds "${SEEDS_CSV}")
fi

uv run python tools/reduce_transition_rich_interpretability_metrics.py "${ARGS[@]}"
