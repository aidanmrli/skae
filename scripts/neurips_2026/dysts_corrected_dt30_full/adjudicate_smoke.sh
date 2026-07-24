#!/usr/bin/env bash
#SBATCH --job-name=dysts-v3-smoke-gate
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${PROJECT_DIR}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:?SOURCE_MANIFEST is required}"
SMOKE_BASE_OUT="${SMOKE_BASE_OUT:?SMOKE_BASE_OUT is required}"
SMOKE_GATE="${SMOKE_GATE:?SMOKE_GATE is required}"
sha256sum -c "${SOURCE_MANIFEST}"
uv run python -m experiments.neurips_2026.dysts_corrected_dt30_full.adjudicate_smoke \
  --base_out "${SMOKE_BASE_OUT}" --output "${SMOKE_GATE}"
