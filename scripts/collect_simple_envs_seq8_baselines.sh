#!/bin/bash
#
# Collect forecasting summaries for simple-environment sequence-L8 baselines.
#
# Submit:
#   sbatch scripts/collect_simple_envs_seq8_baselines.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/simple_envs_seq8
#   GENERIC_ROOT=/network/scratch/l/lia/skae/simple_envs_seq8/generic_sparse
#   LISTA_ROOT=/network/scratch/l/lia/skae/simple_envs_seq8/lista_best
#   OUT_DIR=results/simple_envs_seq8_baselines
#   HORIZON=1000 SELECT=latest
#
#SBATCH --job-name=simple_s8_collect
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o /network/scratch/l/lia/skae/simple-s8-collect-%j.out

set -euo pipefail

source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/simple_envs_seq8}"
GENERIC_ROOT="${GENERIC_ROOT:-${BASE_OUT}/generic_sparse}"
LISTA_ROOT="${LISTA_ROOT:-${BASE_OUT}/lista_best}"
OUT_DIR="${OUT_DIR:-results/simple_envs_seq8_baselines}"
HORIZON="${HORIZON:-1000}"
SELECT="${SELECT:-latest}"

echo "============================================="
echo "Collect Simple Envs Sequence-L8 Baselines"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "GENERIC_ROOT: ${GENERIC_ROOT}"
echo "LISTA_ROOT: ${LISTA_ROOT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "============================================="

uv run python tools/collect_forecasting_roots.py \
  --run_roots "generic_sparse=${GENERIC_ROOT}" "lista_best_seq8=${LISTA_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --horizon "${HORIZON}" \
  --select "${SELECT}"

EXIT_CODE=$?
echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
