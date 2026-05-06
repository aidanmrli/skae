#!/bin/bash
#
# Combine merged stage-2 batch row files and run routed forecasting aggregation.
#
# Required env vars:
#   INPUT_DIRS_CSV=<comma-separated merged stage-2 output dirs>
#   OUT_DIR=<combined output dir>
#   DATASETS=<multibasin|dysts>
#
# Optional env vars:
#   SUPPORT_DEFINITION=topk:8
#
#SBATCH --job-name=sf_lm_combine_agg
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

INPUT_DIRS_CSV="${INPUT_DIRS_CSV:?INPUT_DIRS_CSV is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
DATASETS="${DATASETS:?DATASETS is required}"
SUPPORT_DEFINITION="${SUPPORT_DEFINITION:-topk:8}"

mkdir -p "${OUT_DIR}"
IFS=',' read -r -a INPUT_DIRS <<< "${INPUT_DIRS_CSV}"

ROWS_OUT="${OUT_DIR}/self_routed_forecasting_rows.csv"
: > "${ROWS_OUT}"
header_written=0
for input_dir in "${INPUT_DIRS[@]}"; do
  rows_file="${input_dir}/self_routed_forecasting_rows.csv"
  if [[ ! -s "${rows_file}" ]]; then
    echo "Missing or empty rows file: ${rows_file}" >&2
    exit 1
  fi
  if [[ "${header_written}" == "0" ]]; then
    cat "${rows_file}" >> "${ROWS_OUT}"
    header_written=1
  else
    tail -n +2 "${rows_file}" >> "${ROWS_OUT}"
  fi
done

uv run python - "${OUT_DIR}" "${INPUT_DIRS[@]}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
failures = []
manifests = []
for raw in sys.argv[2:]:
    base = Path(raw)
    failure_path = base / "failures.json"
    if failure_path.exists():
        failures.extend(json.loads(failure_path.read_text()))
    manifest_path = base / "manifest.json"
    if manifest_path.exists():
        manifests.append(json.loads(manifest_path.read_text()))
(out_dir / "failures.json").write_text(json.dumps(failures, indent=2))
(out_dir / "manifest.json").write_text(
    json.dumps(
        {
            "input_dirs": sys.argv[2:],
            "num_failures": len(failures),
            "input_manifests": manifests,
        },
        indent=2,
    )
)
if failures:
    raise SystemExit(f"Combined inputs contain {len(failures)} failures")
PY

uv run python tools/analyze_routed_forecasting_mse.py \
  --multibasin_routed_csv "${ROWS_OUT}" \
  --dysts_routed_csv "${ROWS_OUT}" \
  --output_dir "${OUT_DIR}/aggregation" \
  --datasets "${DATASETS}" \
  --support_definition "${SUPPORT_DEFINITION}"
