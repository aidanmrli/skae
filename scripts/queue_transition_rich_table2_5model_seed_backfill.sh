#!/bin/bash
#
# Build and queue the Table 2 five-model fixed-17 seed backfill.
#
# This queues:
#   - seeds 10-14 for the five paper-facing Table 2 roots at 200k steps
#   - known missing seed 0-9 hard-init MLP-control rows from the seed-0-9 packet
#
# Submit with:
#   sbatch scripts/queue_transition_rich_table2_5model_seed_backfill.sh
#
# Optional env vars:
#   EXPERIMENT_TAG=transition_rich_table2_5model_seed15_backfill_20260428
#   PHASE_LABEL=transition_rich_basin_partition
#   ARRAY_THROTTLE=64
#   MAX_EXISTING_JOBS_BEFORE_SUBMIT=550

#SBATCH --job-name=queue_t2_seed15
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-table2-seed15-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-table2-seed15-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run on a compute node."
  echo "Submit it with: sbatch scripts/queue_transition_rich_table2_5model_seed_backfill.sh"
  exit 2
fi

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-20260428}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-transition_rich_table2_5model_seed15_backfill_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-transition_rich_basin_partition}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-${RESULTS_DIR}/queue_logs}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-64}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-550}"

LISTA_TASK_TSV="${LISTA_TASK_TSV:-results/transition_rich_basin_partition_final_seed10_20260409/task_tables/transition_rich_basin_partition.tsv}"
MLP_TASK_TSV="${MLP_TASK_TSV:-results/transition_rich_hardinit_mlp_controls_seed10_20260416/task_tables/transition_rich_basin_partition.tsv}"

mkdir -p "${TASK_DIR}" "${QUEUE_LOG_DIR}"

TASK_TSV="${TASK_DIR}/transition_rich_table2_5model_seed_backfill.tsv"
MANIFEST_JSON="${TASK_DIR}/transition_rich_table2_5model_seed_backfill_manifest.json"

PHASE_LABEL="${PHASE_LABEL}" uv run python - "${LISTA_TASK_TSV}" "${MLP_TASK_TSV}" "${TASK_TSV}" "${MANIFEST_JSON}" <<'PY'
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

lista_path = Path(sys.argv[1])
mlp_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
manifest_path = Path(sys.argv[4])
phase_label = os.environ["PHASE_LABEL"]

seed_expansion_variants = [
    "lista_dense_softblock_signsplit_p64_hardinit_basin_partition",
    "lista_blockdiag_signsplit_hardinit_basin_partition",
    "mlp_zero_sparse_hardinit_basin_partition_control",
    "mlp_sparse_hardinit_basin_partition_control",
    "mlp_sparse_blockdiag_hardinit_basin_partition_control",
]

missing_seed_rows = [
    ("mlp_zero_sparse_hardinit_basin_partition_control", "claude:arrested_spiral", "7"),
    ("mlp_zero_sparse_hardinit_basin_partition_control", "claude:cal_asymmetric_3", "6"),
    ("mlp_zero_sparse_hardinit_basin_partition_control", "claude:cal_hexagon_6", "5"),
    ("mlp_sparse_hardinit_basin_partition_control", "claude:var_depth_gradient_4", "4"),
    ("mlp_sparse_hardinit_basin_partition_control", "claude:var_depth_gradient_4", "9"),
    ("mlp_sparse_hardinit_basin_partition_control", "claude:var_diamond_4", "5"),
    ("mlp_sparse_blockdiag_hardinit_basin_partition_control", "claude:snic_multi", "5"),
    ("mlp_sparse_blockdiag_hardinit_basin_partition_control", "claude:var_depth_gradient_4", "3"),
]


def read_tsv(path: Path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


lista_rows, lista_fields = read_tsv(lista_path)
mlp_rows, mlp_fields = read_tsv(mlp_path)
if lista_fields != mlp_fields:
    raise SystemExit("Task TSV schemas differ; refusing to build mixed backfill table.")

fields = lista_fields
templates = {}
sources = {}
for source_name, rows in (("lista", lista_rows), ("mlp", mlp_rows)):
    for row in rows:
        variant = row["model_variant"]
        system = row["system_key"]
        key = (variant, system)
        if variant not in seed_expansion_variants:
            continue
        if key not in templates or row.get("seed") == "0":
            templates[key] = dict(row)
            sources[key] = source_name

systems_by_variant = defaultdict(set)
for variant, system in templates:
    systems_by_variant[variant].add(system)

missing_templates = []
for variant, system, seed in missing_seed_rows:
    key = (variant, system)
    if key not in templates:
        raise SystemExit(f"Missing template row for {variant} / {system}")
    missing_templates.append((variant, system, seed))

out_rows = []
seen = set()


def append_row(template, seed, reason):
    row = dict(template)
    row["task_id"] = str(len(out_rows))
    row["phase"] = phase_label
    row["seed"] = str(seed)
    row["num_steps"] = "200000"
    row["eval_profile"] = "full"
    key = (row["model_variant"], row["system_key"], row["seed"])
    if key in seen:
        raise SystemExit(f"Duplicate generated target: {key}")
    seen.add(key)
    row["_reason"] = reason
    out_rows.append(row)


for variant in seed_expansion_variants:
    systems = sorted(systems_by_variant[variant])
    if len(systems) != 17:
        raise SystemExit(f"Expected 17 systems for {variant}, found {len(systems)}")
    for system in systems:
        template = templates[(variant, system)]
        for seed in range(10, 15):
            append_row(template, seed, "seed_10_to_14")

for variant, system, seed in missing_templates:
    append_row(templates[(variant, system)], seed, "seed_0_to_9_missing")

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(out_rows)

counts_by_variant = Counter(row["model_variant"] for row in out_rows)
counts_by_reason = Counter(row["_reason"] for row in out_rows)
manifest = {
    "phase_label": phase_label,
    "lista_task_tsv": str(lista_path),
    "mlp_task_tsv": str(mlp_path),
    "task_tsv": str(out_path),
    "num_tasks": len(out_rows),
    "counts_by_variant": dict(sorted(counts_by_variant.items())),
    "counts_by_reason": dict(sorted(counts_by_reason.items())),
    "seed_expansion_variants": seed_expansion_variants,
    "missing_seed_rows": [
        {"model_variant": variant, "system_key": system, "seed": int(seed)}
        for variant, system, seed in missing_seed_rows
    ],
    "generated_keys": [
        {"model_variant": row["model_variant"], "system_key": row["system_key"], "seed": int(row["seed"])}
        for row in out_rows
    ],
}
manifest_path.write_text(json.dumps(manifest, indent=2))
print(json.dumps({"task_tsv": str(out_path), "num_tasks": len(out_rows), "counts_by_variant": manifest["counts_by_variant"]}, indent=2))
PY

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi

echo "Generated ${TASK_COUNT} Table 2 seed-backfill tasks."
echo "Task TSV: ${TASK_TSV}"
echo "Manifest: ${MANIFEST_JSON}"

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

{
  printf 'RESULTS_DIR=%q\n' "${RESULTS_DIR}"
  printf 'BASE_OUT=%q\n' "${BASE_OUT}"
  printf 'TASK_TSV=%q\n' "${TASK_TSV}"
  printf 'MANIFEST_JSON=%q\n' "${MANIFEST_JSON}"
  printf 'TASK_COUNT=%q\n' "${TASK_COUNT}"
  printf 'ARRAY_JOB_ID=%q\n' "${ARRAY_JOB_ID}"
  printf 'ARRAY_THROTTLE=%q\n' "${ARRAY_THROTTLE}"
  printf 'MAX_EXISTING_JOBS_BEFORE_SUBMIT=%q\n' "${MAX_EXISTING_JOBS_BEFORE_SUBMIT}"
} > "${QUEUE_LOG_DIR}/launch_record.env"

echo "Queued Table 2 five-model seed backfill."
echo "Array job: ${ARRAY_JOB_ID}"
echo "Results dir: ${RESULTS_DIR}"
echo "Base out: ${BASE_OUT}"
