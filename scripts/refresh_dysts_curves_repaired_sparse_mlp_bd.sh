#!/usr/bin/env bash
#SBATCH --job-name=refresh_dysts_curves
#SBATCH --output=results/dysts_dt30_sparse_mlp_bd_repaired_20260506/queue/refresh-curves-%j.out
#SBATCH --error=results/dysts_dt30_sparse_mlp_bd_repaired_20260506/queue/refresh-curves-%j.err
#SBATCH --partition=long
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/mila/l/lia/skae}"
cd "${ROOT_DIR}"

echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "git=$(git rev-parse --short HEAD || true)"

BASE_ROWS="${BASE_ROWS:-results/dysts_dt30_basinblock_p256_seq10_100k_20260430/long_horizon_eval/collect/forecasting_rows.csv}"
REPAIRED_ROWS="${REPAIRED_ROWS:-results/dysts_dt30_sparse_mlp_bd_repaired_20260506/long_horizon_eval/collect/forecasting_rows.csv}"
FIG_DIR="${FIG_DIR:-docs/figures/neurips_paper_2026}"
TABLE_DIR="${TABLE_DIR:-docs/figures/neurips_paper_2026/_tables}"

uv run python tools/analyze_dysts_dt30_results.py \
  --forecasting-csv "${BASE_ROWS}" \
  --replacement-forecasting-csv "${REPAIRED_ROWS}" \
  --replacement-root-labels sparse_mlp_bd \
  --fig-dir "${FIG_DIR}" \
  --table-dir "${TABLE_DIR}" \
  --horizons 100 500 1000 1500 2000 3000 4000 5000 \
  --table-horizons 100 2000 4000

uv run python tools/make_dysts_iqm_over_iqm_appendix.py \
  --summary-csv "${TABLE_DIR}/dysts_dt30_iqm_summary.csv" \
  --fig-dir "${FIG_DIR}" \
  --table-dir "${TABLE_DIR}" \
  --horizons 100 500 1000 1500 2000 3000 4000 5000

echo "done=$(date -Is)"
