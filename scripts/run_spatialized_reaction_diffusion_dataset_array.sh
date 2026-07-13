#!/usr/bin/env bash
#SBATCH --job-name=spatial-rd-data
#SBATCH --output=/network/scratch/l/lia/skae/spatial-rd-data-%A_%a.out
#SBATCH --error=/network/scratch/l/lia/skae/spatial-rd-data-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

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

mkdir -p "$(dirname "${dataset_path}")"

echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse --short HEAD || true)"
echo "task_id=${task_id}"
echo "source_system=${source_system}"
echo "seed=${seed}"
echo "dataset_path=${dataset_path}"
echo "link_paths_csv=${link_paths_csv:-}"

if [[ "${FORCE_REGENERATE}" == "1" || ! -f "${dataset_path}" ]]; then
  uv run python tools/generate_spatialized_reaction_diffusion.py \
    --output "${dataset_path}" \
    --source_system "${source_system}" \
    --grid_size "${grid_size}" \
    --diffusion "${diffusion}" \
    --rk4_dt "${rk4_dt}" \
    --substeps_per_observation "${substeps_per_observation}" \
    --trajectory_length "${trajectory_length}" \
    --label_extra_observations "${label_extra_observations}" \
    --train_trajectories "${train_trajectories}" \
    --val_trajectories "${val_trajectories}" \
    --test_trajectories "${test_trajectories}" \
    --laplacian_scaling "${laplacian_scaling}" \
    --seed "${seed}"
else
  echo "Using existing dataset: ${dataset_path}"
fi

if [[ -n "${link_paths_csv:-}" ]]; then
  IFS=',' read -r -a link_paths <<< "${link_paths_csv}"
  for link_path in "${link_paths[@]}"; do
    if [[ -z "${link_path}" ]]; then
      continue
    fi
    mkdir -p "$(dirname "${link_path}")"
    ln -sfn "${dataset_path}" "${link_path}"
    echo "Linked ${link_path} -> ${dataset_path}"
  done
fi

echo "completed_at=$(date --iso-8601=seconds)"
