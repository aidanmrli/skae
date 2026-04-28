#!/bin/bash
#
# Queue the matched-dimension LISTA-SB fairness sensitivity:
#   current LISTA-SB recipe, hard-init sampling, but target_size=256.
#
# This is a sensitivity control for the paper-facing p64 LISTA-SB row, not a
# replacement for that row. It generates 17 systems x 15 seeds at 200k steps,
# then queues forecasting collection and self-routed forecasting.
#
# Submit with:
#   sbatch scripts/queue_transition_rich_lista_sb_p256_hardinit_fairness.sh
#
# Optional env vars:
#   EXPERIMENT_TAG=transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428
#   ARRAY_THROTTLE=64
#   MAX_EXISTING_JOBS_BEFORE_SUBMIT=650

#SBATCH --job-name=queue_lsb_p256
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-lista-sb-p256-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-lista-sb-p256-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run on a compute node."
  echo "Submit it with: sbatch scripts/queue_transition_rich_lista_sb_p256_hardinit_fairness.sh"
  exit 2
fi

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-20260428}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-transition_rich_lista_sb_p256_hardinit_fairness_seed15_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-transition_rich_basin_partition}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect_pass0}"
SELF_ROUTED_DIR="${SELF_ROUTED_DIR:-${RESULTS_DIR}/self_routed_forecasting}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-${RESULTS_DIR}/queue_logs}"
AUTOMATION_DIR="${AUTOMATION_DIR:-${RESULTS_DIR}/automation}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-64}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-650}"

TEMPLATE_TSV="${TEMPLATE_TSV:-results/transition_rich_basin_partition_final_seed10_20260409/task_tables/transition_rich_basin_partition.tsv}"
SOURCE_VARIANT="${SOURCE_VARIANT:-lista_dense_softblock_signsplit_p64_hardinit_basin_partition}"
TARGET_VARIANT="${TARGET_VARIANT:-lista_dense_softblock_signsplit_p256_hardinit_basin_partition}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}"

mkdir -p "${TASK_DIR}" "${ROOT_SPEC_DIR}" "${COLLECT_DIR}" "${SELF_ROUTED_DIR}" "${QUEUE_LOG_DIR}" "${AUTOMATION_DIR}"

TASK_TSV="${TASK_DIR}/transition_rich_lista_sb_p256_hardinit_fairness.tsv"
MANIFEST_JSON="${TASK_DIR}/transition_rich_lista_sb_p256_hardinit_fairness_manifest.json"
ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/transition_rich_lista_sb_p256_hardinit_roots.txt"
SELF_ROUTED_MANIFEST_JSON="${AUTOMATION_DIR}/self_routed_forecasting_queue.json"

PHASE_LABEL="${PHASE_LABEL}" \
SOURCE_VARIANT="${SOURCE_VARIANT}" \
TARGET_VARIANT="${TARGET_VARIANT}" \
TARGET_SIZE="${TARGET_SIZE}" \
SEEDS_CSV="${SEEDS_CSV}" \
  uv run python - "${TEMPLATE_TSV}" "${TASK_TSV}" "${MANIFEST_JSON}" <<'PY'
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

template_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])

phase_label = os.environ["PHASE_LABEL"]
source_variant = os.environ["SOURCE_VARIANT"]
target_variant = os.environ["TARGET_VARIANT"]
target_size = os.environ["TARGET_SIZE"]
seeds = [seed.strip() for seed in os.environ["SEEDS_CSV"].split(",") if seed.strip()]

with template_path.open(newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fields = list(reader.fieldnames or [])
    source_rows = [row for row in reader if row.get("model_variant") == source_variant]

templates = {}
for row in source_rows:
    system = row["system_key"]
    if system not in templates or row.get("seed") == "0":
        templates[system] = dict(row)

if len(templates) != 17:
    raise SystemExit(f"Expected 17 source systems for {source_variant}, found {len(templates)}")

out_rows = []
for system in sorted(templates):
    template = templates[system]
    for seed in seeds:
        row = dict(template)
        row["task_id"] = str(len(out_rows))
        row["phase"] = phase_label
        row["model_variant"] = target_variant
        row["seed"] = str(seed)
        row["num_steps"] = "200000"
        row["target_size"] = target_size
        row["eval_profile"] = "full"
        out_rows.append(row)

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(out_rows)

manifest = {
    "source_task_tsv": str(template_path),
    "task_tsv": str(out_path),
    "source_variant": source_variant,
    "target_variant": target_variant,
    "target_size": int(target_size),
    "phase_label": phase_label,
    "seeds": [int(seed) for seed in seeds],
    "num_tasks": len(out_rows),
    "counts_by_system": dict(sorted(Counter(row["system_key"] for row in out_rows).items())),
}
manifest_path.write_text(json.dumps(manifest, indent=2))
print(json.dumps({"task_tsv": str(out_path), "num_tasks": len(out_rows), "target_variant": target_variant}, indent=2))
PY

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi

printf '%s=%s/%s/%s\n' "${TARGET_VARIANT}" "${BASE_OUT}" "${PHASE_LABEL}" "${TARGET_VARIANT}" > "${ROOT_SPECS_FILE}"

echo "Generated ${TASK_COUNT} matched-dimension LISTA-SB tasks."
echo "Task TSV: ${TASK_TSV}"
echo "Manifest: ${MANIFEST_JSON}"
echo "Root specs: ${ROOT_SPECS_FILE}"

while true; do
  CURRENT_JOBS=$(squeue -u "${USER}" -h -r | wc -l)
  if (( CURRENT_JOBS <= MAX_EXISTING_JOBS_BEFORE_SUBMIT )); then
    break
  fi
  echo "Current expanded job count ${CURRENT_JOBS} exceeds ${MAX_EXISTING_JOBS_BEFORE_SUBMIT}; sleeping before array submit."
  sleep 60
done

ARRAY_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" \
    sbatch --array=0-$((TASK_COUNT - 1))%"${ARRAY_THROTTLE}" scripts/run_paper_benchmark_array.sh | awk '{print $4}'
)

COLLECT_JOB_ID=$(
  ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" \
  OUT_DIR="${COLLECT_DIR}" \
  HORIZONS_CSV="100,500,1000" \
  GOOD_THRESHOLD="50" \
    sbatch --dependency=afterany:"${ARRAY_JOB_ID}" scripts/collect_transition_rich_basin_partition.sh | awk '{print $4}'
)

SELF_ROUTE_QUEUE_JOB_ID=$(
  sbatch \
    --dependency=afterok:"${COLLECT_JOB_ID}" \
    --job-name="queue_lsbp256_srf" \
    --partition=long \
    --cpus-per-task=1 \
    --mem=2G \
    --time=00:30:00 \
    --output="${QUEUE_LOG_DIR}/queue-self-routed-%A.out" \
    --error="${QUEUE_LOG_DIR}/queue-self-routed-%A.err" \
    --wrap="cd ${ROOT_DIR} && ROWS_CSVS=${COLLECT_DIR}/forecasting_rows.csv OUT_DIR=${SELF_ROUTED_DIR} ROOT_LABELS_CSV=${TARGET_VARIANT} SEEDS_CSV=${SEEDS_CSV} SEED_SPLITS_SEMICOLON='0,1,2,3,4;5,6,7,8,9;10,11,12,13,14' SUPPORT_DEFINITIONS=topk:8 DEPTH_STRATA=all,q4 ROLLOUT_MODES=global_k,support_gated_k,support_local_centered,family_local_centered HORIZONS=100,500,1000 QUEUE_MANIFEST_JSON=${SELF_ROUTED_MANIFEST_JSON} bash scripts/queue_transition_rich_self_routed_forecasting_shards.sh" \
    | awk '{print $4}'
)

{
  printf 'RESULTS_DIR=%q\n' "${RESULTS_DIR}"
  printf 'BASE_OUT=%q\n' "${BASE_OUT}"
  printf 'TASK_TSV=%q\n' "${TASK_TSV}"
  printf 'MANIFEST_JSON=%q\n' "${MANIFEST_JSON}"
  printf 'ROOT_SPECS_FILE=%q\n' "${ROOT_SPECS_FILE}"
  printf 'TASK_COUNT=%q\n' "${TASK_COUNT}"
  printf 'ARRAY_JOB_ID=%q\n' "${ARRAY_JOB_ID}"
  printf 'COLLECT_JOB_ID=%q\n' "${COLLECT_JOB_ID}"
  printf 'SELF_ROUTE_QUEUE_JOB_ID=%q\n' "${SELF_ROUTE_QUEUE_JOB_ID}"
  printf 'SELF_ROUTED_DIR=%q\n' "${SELF_ROUTED_DIR}"
  printf 'ARRAY_THROTTLE=%q\n' "${ARRAY_THROTTLE}"
  printf 'MAX_EXISTING_JOBS_BEFORE_SUBMIT=%q\n' "${MAX_EXISTING_JOBS_BEFORE_SUBMIT}"
} > "${QUEUE_LOG_DIR}/launch_record.env"

echo "Queued matched-dimension LISTA-SB fairness sensitivity."
echo "Array job: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Self-routed queue job: ${SELF_ROUTE_QUEUE_JOB_ID}"
echo "Results dir: ${RESULTS_DIR}"
echo "Base out: ${BASE_OUT}"
