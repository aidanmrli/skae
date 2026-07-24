#!/bin/bash
# Assess the preregistered utilization smoke without reading model outcomes.

#SBATCH --job-name=dense0wd_util
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source scripts/common/cluster_env.sh

CARD_PATH="${CARD_PATH:-experiments/neurips_2026/global_k_dense_zero_wd_card.json}"
TELEMETRY_CSV="${TELEMETRY_CSV:?TELEMETRY_CSV is required}"
TRAINING_LOG="${TRAINING_LOG:?TRAINING_LOG is required}"
OUTPUT_JSON="${OUTPUT_JSON:?OUTPUT_JSON is required}"

uv run python experiments/neurips_2026/assess_global_k_dense_gpu_smoke.py \
  --card "${CARD_PATH}" \
  --telemetry_csv "${TELEMETRY_CSV}" \
  --training_log "${TRAINING_LOG}" \
  --output_json "${OUTPUT_JSON}"
