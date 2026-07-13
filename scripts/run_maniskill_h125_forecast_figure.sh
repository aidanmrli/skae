#!/usr/bin/env bash
# Build the ManiSkill H125 forecasting figure from existing evaluation JSONs.
#SBATCH --job-name=mskill_h125_fig
#SBATCH --partition=long
#SBATCH --output=logs/maniskill_h125_fig_%j.out
#SBATCH --error=logs/maniskill_h125_fig_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

echo "date=$(date)"
echo "host=$(hostname)"
echo "commit=$(git rev-parse --short HEAD || true)"

INPUT_ROOT="${INPUT_ROOT:-runs/maniskill_insertion/perturbation_e20_50k_long_eval_20260603}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/figures/neurips_paper_2026}"
RESULTS_DIR="${RESULTS_DIR:-results/maniskill_h125_forecast_20260603}"
HORIZONS="${HORIZONS:-10,25,50,100,125}"
FIGURE_STEM="${FIGURE_STEM:-fig_maniskill_h125_forecasting}"
ROLLOUT_KEY="${ROLLOUT_KEY:-rollout}"

uv run python tools/make_maniskill_h125_forecast_figure.py \
  --input-root "${INPUT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --horizons "${HORIZONS}" \
  --figure-stem "${FIGURE_STEM}" \
  --rollout-key "${ROLLOUT_KEY}"
