#!/usr/bin/env bash
#SBATCH --job-name=l96-multistability-audit
#SBATCH --output=/network/scratch/l/lia/skae/l96-multistability-audit-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/l96-multistability-audit-%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-results/lorenz96_multistability_audit_20260625}"
DIMENSIONS="${DIMENSIONS:-4,5,6,8,16,32}"
FORCINGS="${FORCINGS:-0.5,0.75,1,1.25,1.5,2,2.5,3,4,5,6,8}"
SEED="${SEED:-0}"
INITIALS_PER_PAIR="${INITIALS_PER_PAIR:-48}"
WARMUP_OBSERVATIONS="${WARMUP_OBSERVATIONS:-600}"
TAIL_OBSERVATIONS="${TAIL_OBSERVATIONS:-256}"
SAMPLE_EVERY="${SAMPLE_EVERY:-10}"
MIN_SILHOUETTE="${MIN_SILHOUETTE:-0.25}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-3}"

mkdir -p "${OUTPUT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

uv run python tools/audit_lorenz96_multistability.py \
  --output_dir "${OUTPUT_DIR}" \
  --dimensions "${DIMENSIONS}" \
  --forcings "${FORCINGS}" \
  --seed "${SEED}" \
  --initials_per_pair "${INITIALS_PER_PAIR}" \
  --warmup_observations "${WARMUP_OBSERVATIONS}" \
  --tail_observations "${TAIL_OBSERVATIONS}" \
  --sample_every "${SAMPLE_EVERY}" \
  --min_silhouette "${MIN_SILHOUETTE}" \
  --min_cluster_size "${MIN_CLUSTER_SIZE}"
