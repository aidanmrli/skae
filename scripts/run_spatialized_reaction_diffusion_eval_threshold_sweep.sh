#!/usr/bin/env bash
#SBATCH --job-name=spatial-rd-eval-sweep
#SBATCH --output=logs/slurm/spatial_rd_eval_sweep_%j.out
#SBATCH --error=logs/slurm/spatial_rd_eval_sweep_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs/slurm

DATASET="${DATASET:-runs/spatialized_reaction_diffusion/cal_square_4_seed0_grid32_smoke/dataset.pt}"
CHECKPOINT="${CHECKPOINT:-runs/spatialized_reaction_diffusion/cal_square_4_seed0_grid32_smoke/lista/checkpoint.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/spatialized_reaction_diffusion/cal_square_4_seed0_grid32_smoke/eval_threshold_sweep}"
SUPPORT_THRESHOLDS="${SUPPORT_THRESHOLDS:-1e-6 1e-5 1e-4 1e-3}"
FAMILY_JACCARDS="${FAMILY_JACCARDS:-0.4 0.7 1.0}"
DEEP_THRESHOLDS="${DEEP_THRESHOLDS:-0.5 0.7 0.9}"

mkdir -p "${OUTPUT_ROOT}"

for support_threshold in ${SUPPORT_THRESHOLDS}; do
  for family_jaccard in ${FAMILY_JACCARDS}; do
    for deep_threshold in ${DEEP_THRESHOLDS}; do
      safe_support="${support_threshold//./p}"
      safe_support="${safe_support//-/m}"
      safe_jaccard="${family_jaccard//./p}"
      safe_deep="${deep_threshold//./p}"
      output="${OUTPUT_ROOT}/support_${safe_support}_jaccard_${safe_jaccard}_deep_${safe_deep}.json"
      uv run python tools/evaluate_spatialized_reaction_diffusion.py \
        --dataset "${DATASET}" \
        --checkpoint "${CHECKPOINT}" \
        --output "${output}" \
        --horizons 1 4 8 12 \
        --support_threshold "${support_threshold}" \
        --family_jaccard "${family_jaccard}" \
        --deep_threshold "${deep_threshold}" \
        --device auto
    done
  done
done

echo "output_root=${OUTPUT_ROOT}"
