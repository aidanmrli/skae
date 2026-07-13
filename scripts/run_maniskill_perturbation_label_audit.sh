#!/usr/bin/env bash
# CPU-only semantic label audit for the perturbation-balanced ManiSkill packet.
# This job intentionally does not request a GPU; it only reads compact NPZ/JSON
# artifacts and checks whether target perturbation labels are paper-usable
# semantic outcome/contact labels.
#SBATCH --job-name=mskill_label_audit
#SBATCH --partition=long
#SBATCH --output=logs/maniskill_label_audit_%j.out
#SBATCH --error=logs/maniskill_label_audit_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"
export CUDA_VISIBLE_DEVICES=""

DATASET="${DATASET:-data/maniskill/perturbation_assessment_seed0_e20/all_setups.npz}"
PERTURBATION_SUMMARY="${PERTURBATION_SUMMARY:-data/maniskill/perturbation_assessment_seed0_e20/perturbation_summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/maniskill_perturbation_label_audit_20260610}"

echo "date=$(date)"
echo "host=$(hostname)"
echo "commit=$(git rev-parse --short HEAD || true)"
echo "slurm_job_id=${SLURM_JOB_ID:-}"
echo "dataset=${DATASET}"
echo "perturbation_summary=${PERTURBATION_SUMMARY}"
echo "output_dir=${OUTPUT_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

uv run python tools/audit_maniskill_perturbation_labels.py \
  --dataset "${DATASET}" \
  --perturbation_summary "${PERTURBATION_SUMMARY}" \
  --output_dir "${OUTPUT_DIR}"

echo "summary=${OUTPUT_DIR}/label_audit_summary.json"
