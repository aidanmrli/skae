#!/usr/bin/env bash
#SBATCH --job-name=spatial-rd-support-sweep
#SBATCH --output=/network/scratch/l/lia/skae/spatial-rd-support-sweep-%A_%a.out
#SBATCH --error=/network/scratch/l/lia/skae/spatial-rd-support-sweep-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

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

mkdir -p "$(dirname "${output}")"

echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse --short HEAD || true)"
echo "task_id=${task_id}"
echo "source_system=${source_system}"
echo "seed=${seed}"
echo "model_variant=${model_variant}"
echo "dataset=${dataset}"
echo "checkpoint=${checkpoint}"
echo "output=${output}"
echo "support_thresholds_csv=${support_thresholds_csv}"
echo "family_jaccards_csv=${family_jaccards_csv}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

uv run python tools/sweep_spatialized_reaction_diffusion_support.py \
  --dataset "${dataset}" \
  --checkpoint "${checkpoint}" \
  --output "${output}" \
  --support_thresholds_csv "${support_thresholds_csv}" \
  --family_jaccards_csv "${family_jaccards_csv}" \
  --batch_size "${batch_size}" \
  --deep_threshold "${deep_threshold}" \
  --device auto

echo "completed_at=$(date --iso-8601=seconds)"
