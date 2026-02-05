#!/bin/bash

#SBATCH --job-name=lqr_dec_collect
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH -o /network/scratch/l/lia/skae/lqr_dec_collect-%j.out
#SBATCH --requeue

set -euo pipefail

source .venv/bin/activate

BASE_DIR="${BASE_DIR:-/network/scratch/l/lia/skae/lqr_decision}"
OUT_DIR="${OUT_DIR:-results/lqr_decision}"

uv run python tools/collect_lqr_decision_results.py \
  --base_dir "${BASE_DIR}" \
  --output_dir "${OUT_DIR}" \
  --decision_stage 2 \
  --decision_system lyapunov \
  --arms bd_c2,ah_prag

echo "Aggregation complete: ${OUT_DIR}"
