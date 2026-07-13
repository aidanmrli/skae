#!/bin/bash
#
# Summarize support-family hyperparameter sweep outputs.
#
# Required env vars:
#   SWEEP_DIR=<directory containing per-combo evaluator outputs>
#   OUT_DIR=<summary output directory>
#
# Optional env vars:
#   TOP_K=25
#
#SBATCH --job-name=regime_hparam_sum
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH -o /network/scratch/l/lia/skae/regime-hparam-summary-%A.out
#SBATCH -e /network/scratch/l/lia/skae/regime-hparam-summary-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

SWEEP_DIR="${SWEEP_DIR:?SWEEP_DIR is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
TOP_K="${TOP_K:-25}"

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "SWEEP_DIR: ${SWEEP_DIR}"
echo "OUT_DIR: ${OUT_DIR}"
echo "TOP_K: ${TOP_K}"

uv run python tools/summarize_regime_support_family_hparam_sweep.py \
  --sweep_dir "${SWEEP_DIR}" \
  --output_dir "${OUT_DIR}" \
  --top_k "${TOP_K}"
