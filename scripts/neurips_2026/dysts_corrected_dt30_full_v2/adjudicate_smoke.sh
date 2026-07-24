#!/usr/bin/env bash
#SBATCH --job-name=dysts-v5-smoke-gate
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${PROJECT_DIR}"
sha256sum -c "${SOURCE_MANIFEST:?SOURCE_MANIFEST is required}"
uv run python -m experiments.neurips_2026.dysts_corrected_dt30_full_v2.adjudicate_smoke \
  --base_out "${SMOKE_BASE_OUT:?SMOKE_BASE_OUT is required}" \
  --expected_cache_dir "${DYSTS_CACHE_DIR:?DYSTS_CACHE_DIR is required}" \
  --output "${SMOKE_GATE:?SMOKE_GATE is required}"
