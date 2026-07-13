#!/bin/bash
#
# Regenerate paper-style support-coordinate intervention figures from saved CSVs.
#
# Optional env vars:
#   RESULT_DIR=results/support_coordinate_interventions_20260506/gated_local_linear_lista_seed0
#   OUTPUT_DIR=docs/figures/neurips_paper_2026
#   PLOT_FORMAT=pdf,png
#   FILENAME_PREFIX=fig_support_coordinate_
#
#SBATCH --job-name=plot_support_coord
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH -o /network/scratch/l/lia/skae/plot-support-coordinate-interventions-%A.out
#SBATCH -e /network/scratch/l/lia/skae/plot-support-coordinate-interventions-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

RESULT_DIR="${RESULT_DIR:-results/support_coordinate_interventions_20260506/gated_local_linear_lista_seed0}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/figures/neurips_paper_2026}"
PLOT_FORMAT="${PLOT_FORMAT:-pdf,png}"
FILENAME_PREFIX="${FILENAME_PREFIX:-fig_support_coordinate_}"

uv run python tools/plot_support_coordinate_interventions.py \
  --result_dir "${RESULT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --plot_format "${PLOT_FORMAT}" \
  --filename_prefix "${FILENAME_PREFIX}"
