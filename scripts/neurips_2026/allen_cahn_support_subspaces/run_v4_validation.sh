#!/usr/bin/env bash
#SBATCH --job-name=ac-support-v4-check
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/network/scratch/l/lia/skae/ac_support_v4_validation_%j.out
#SBATCH --error=/network/scratch/l/lia/skae/ac_support_v4_validation_%j.err

set -euo pipefail

REPO_ROOT=/home/mila/l/lia/skae
cd "${REPO_ROOT}"
uv run pytest tests/test_allen_cahn_support_subspaces.py \
  tests/test_allen_cahn_support_subspaces_canary.py -q
sha256sum -c experiments/neurips_2026/allen_cahn_support_subspaces/source_manifest.sha256
for worker in scripts/neurips_2026/allen_cahn_support_subspaces/*.sh; do
  bash -n "${worker}"
done
test ! -e /network/scratch/l/lia/skae/allen_cahn_support_subspaces_20260720_v4
