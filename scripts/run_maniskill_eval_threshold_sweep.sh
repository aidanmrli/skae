#!/usr/bin/env bash
#SBATCH --job-name=mskill-eval-sweep
#SBATCH --partition=long
#SBATCH --output=logs/maniskill_eval_sweep_%j.out
#SBATCH --error=logs/maniskill_eval_sweep_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs

DATASET="${DATASET:-data/maniskill/PegInsertionSide-v1_state_compact_seed0.npz}"
CHECKPOINT="${CHECKPOINT:-runs/maniskill_insertion/controlled_lista_seed0/checkpoint.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/maniskill_insertion/controlled_lista_seed0/eval_threshold_sweep}"
SUPPORT_THRESHOLDS="${SUPPORT_THRESHOLDS:-0.001 0.01 0.05 0.1 0.2}"
FAMILY_JACCARDS="${FAMILY_JACCARDS:-0.5 0.7 0.9}"

mkdir -p "${OUTPUT_ROOT}"

for support_threshold in ${SUPPORT_THRESHOLDS}; do
  for family_jaccard in ${FAMILY_JACCARDS}; do
    safe_support="${support_threshold//./p}"
    safe_jaccard="${family_jaccard//./p}"
    output_dir="${OUTPUT_ROOT}/support_${safe_support}_jaccard_${safe_jaccard}"
    uv run python tools/evaluate_maniskill_controlled_lista.py \
      --dataset "${DATASET}" \
      --checkpoint "${CHECKPOINT}" \
      --output_dir "${output_dir}" \
      --device cuda \
      --split test \
      --horizons 10,25,50,100 \
      --support_threshold "${support_threshold}" \
      --family_jaccard "${family_jaccard}"
  done
done

echo "output_root=${OUTPUT_ROOT}"
