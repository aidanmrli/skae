#!/bin/bash
#
# Merge explicit regime-discovery local-Koopman shard outputs.
#
# Required env vars:
#   SHARDS_DIR=<directory containing shard outputs>
#   OUT_DIR=<merged output directory>
#
# Optional env vars:
#   ROWS_CSVS=<comma-separated forecasting_rows.csv files>
#   ROOT_LABELS_CSV=<comma-separated root labels>
#   SYSTEMS_CSV=<comma-separated systems>
#   SEEDS_CSV=<comma-separated seeds>
#
#SBATCH --job-name=tr_regime_merge
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH -o /network/scratch/l/lia/skae/tr-regime-merge-%A.out
#SBATCH -e /network/scratch/l/lia/skae/tr-regime-merge-%A.err

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

SHARDS_DIR="${SHARDS_DIR:?SHARDS_DIR is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
ROWS_CSVS="${ROWS_CSVS:-}"
ROOT_LABELS_CSV="${ROOT_LABELS_CSV:-}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-}"

echo "============================================="
echo "Merge Regime-Discovery Local Koopman Shards"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "SHARDS_DIR: ${SHARDS_DIR}"
echo "OUT_DIR: ${OUT_DIR}"
echo "ROWS_CSVS: ${ROWS_CSVS:-<none>}"
echo "ROOT_LABELS_CSV: ${ROOT_LABELS_CSV:-<all>}"
echo "SYSTEMS_CSV: ${SYSTEMS_CSV:-<all>}"
echo "SEEDS_CSV: ${SEEDS_CSV:-<all>}"
echo "============================================="

uv run python tools/merge_regime_discovery_local_koopman_shards.py \
  --shards_dir "${SHARDS_DIR}" \
  --output_dir "${OUT_DIR}" \
  --rows_csvs "${ROWS_CSVS}" \
  --root_labels "${ROOT_LABELS_CSV}" \
  --systems "${SYSTEMS_CSV}" \
  --seeds "${SEEDS_CSV}"
