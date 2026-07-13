#!/usr/bin/env bash
#SBATCH --job-name=spatial-rd-paper-support
#SBATCH --output=/network/scratch/l/lia/skae/spatial-rd-paper-support-%A_%a.out
#SBATCH --error=/network/scratch/l/lia/skae/spatial-rd-paper-support-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LINE_NO=$((TASK_ID + ARRAY_OFFSET + 2))

TASK_EXPORTS="$(
  uv run python - "${TASK_TSV}" "${LINE_NO}" <<'PY'
import csv
import shlex
import sys

path = sys.argv[1]
line_no = int(sys.argv[2])
with open(path, newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for current_line_no, row in enumerate(reader, start=2):
        if current_line_no == line_no:
            for key, value in row.items():
                print(f"{key}={shlex.quote(value or '')}")
            break
    else:
        sys.exit(3)
PY
)"
eval "${TASK_EXPORTS}"

mkdir -p "$(dirname "${output_path}")"

echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse --short HEAD || true)"
echo "task_id=${task_id}"
echo "source_system=${source_system}"
echo "seed=${seed}"
echo "model_variant=${model_variant}"
echo "dataset_path=${dataset_path}"
echo "checkpoint_path=${checkpoint_path}"
echo "output_path=${output_path}"

read -r -a horizon_array <<< "${eval_horizons//,/ }"
uv run python tools/evaluate_spatialized_reaction_diffusion.py \
  --dataset "${dataset_path}" \
  --checkpoint "${checkpoint_path}" \
  --output "${output_path}" \
  --horizons "${horizon_array[@]}" \
  --support_threshold 1e-3 \
  --family_jaccard 0.5 \
  --max_validation_reps 512 \
  --deep_threshold "${deep_threshold}" \
  --device auto

echo "completed_at=$(date --iso-8601=seconds)"
