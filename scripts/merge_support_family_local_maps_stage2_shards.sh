#!/bin/bash
#
# Merge stage-2 support-family local-map shard outputs.
#
# Required env vars:
#   SHARDS_DIR=<directory containing shard output dirs>
#   OUT_DIR=<merged output directory>
#
#SBATCH --job-name=sf_lm_merge
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH -o /network/scratch/l/lia/skae/sf-local-stage2-merge-%A.out
#SBATCH -e /network/scratch/l/lia/skae/sf-local-stage2-merge-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

SHARDS_DIR="${SHARDS_DIR:?SHARDS_DIR is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"

uv run python tools/merge_support_family_local_maps_stage2_shards.py \
  --shards_dir "${SHARDS_DIR}" \
  --output_dir "${OUT_DIR}"
