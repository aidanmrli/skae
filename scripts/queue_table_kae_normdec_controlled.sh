#!/bin/bash
#
# Queue controlled multibasin Table-1 KAE replacements/ablations with a
# normalized linear decoder flag on every row and a configurable sparsity target.
#
# Submit examples:
#   SPARSITY_TARGET=rollout sbatch scripts/queue_table_kae_normdec_controlled.sh
#   SPARSITY_TARGET=encoded sbatch scripts/queue_table_kae_normdec_controlled.sh
#
# Optional env vars:
#   DATE_TAG=20260514
#   EXPERIMENT_TAG=transition_rich_table_kae_normdec_rollout_20260514
#   ARRAY_THROTTLE=48
#   TRAIN_TIME_LIMIT=03:00:00

#SBATCH --job-name=queue_kae_normctl
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-kae-normctl-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-kae-normctl-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run under SLURM."
  echo "Submit it with: sbatch scripts/queue_table_kae_normdec_controlled.sh"
  exit 2
fi

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
SPARSITY_TARGET="${SPARSITY_TARGET:-rollout}"
if [[ "${SPARSITY_TARGET}" != "rollout" && "${SPARSITY_TARGET}" != "encoded" && "${SPARSITY_TARGET}" != "encoded_rollout" ]]; then
  echo "SPARSITY_TARGET must be rollout, encoded, or encoded_rollout; got ${SPARSITY_TARGET}"
  exit 2
fi

VARIANT_SUFFIX="${VARIANT_SUFFIX:-normdec_${SPARSITY_TARGET}}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-transition_rich_table_kae_${VARIANT_SUFFIX}_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-transition_rich_basin_partition}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect_pass0}"
INTERP_DIR="${INTERP_DIR:-${RESULTS_DIR}/interpretability_per_basin_deep_pass0}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-${RESULTS_DIR}/queue_logs}"
AUTOMATION_DIR="${AUTOMATION_DIR:-${RESULTS_DIR}/automation}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-48}"
TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT:-03:00:00}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-10000}"
SUBMIT_WAIT_SECONDS="${SUBMIT_WAIT_SECONDS:-120}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}"
SYSTEMS_CSV="${SYSTEMS_CSV:-gated_local_linear,gated_transfer_linear,claude:arrested_spiral,claude:cal_asymmetric_3,claude:cal_high_cross_3,claude:cal_hexagon_6,claude:cal_octagon_8,claude:cal_pentagon_5,claude:cal_square_4,claude:duffing_triple_well,claude:snic_multi,claude:transition_routes_4,claude:var_depth_gradient_4,claude:var_diamond_4,claude:var_l_shape_5}"

mkdir -p \
  "${TASK_DIR}" \
  "${ROOT_SPEC_DIR}" \
  "${COLLECT_DIR}" \
  "${INTERP_DIR}/shards" \
  "${INTERP_DIR}/logs" \
  "${QUEUE_LOG_DIR}" \
  "${AUTOMATION_DIR}"

TASK_TSV="${TASK_DIR}/table_kae_${VARIANT_SUFFIX}_controlled_tasks.tsv"
MANIFEST_JSON="${TASK_DIR}/table_kae_${VARIANT_SUFFIX}_controlled_manifest.json"
ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/table_kae_${VARIANT_SUFFIX}_controlled_roots.txt"
QUEUE_RECORD_JSON="${AUTOMATION_DIR}/table_kae_${VARIANT_SUFFIX}_controlled_queue.json"

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Date: $(date)"
echo "EXPERIMENT_TAG: ${EXPERIMENT_TAG}"
echo "SPARSITY_TARGET: ${SPARSITY_TARGET}"
echo "VARIANT_SUFFIX: ${VARIANT_SUFFIX}"
echo "BASE_OUT: ${BASE_OUT}"
echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "SYSTEMS_CSV: ${SYSTEMS_CSV}"
echo "SEEDS_CSV: ${SEEDS_CSV}"

SPARSITY_TARGET="${SPARSITY_TARGET}" \
VARIANT_SUFFIX="${VARIANT_SUFFIX}" \
PHASE_LABEL="${PHASE_LABEL}" \
BASE_OUT="${BASE_OUT}" \
SYSTEMS_CSV="${SYSTEMS_CSV}" \
SEEDS_CSV="${SEEDS_CSV}" \
  uv run python - "${TASK_TSV}" "${MANIFEST_JSON}" "${ROOT_SPECS_FILE}" <<'PY'
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

task_tsv = Path(sys.argv[1])
manifest_json = Path(sys.argv[2])
root_specs_file = Path(sys.argv[3])

sparsity_target = os.environ["SPARSITY_TARGET"]
variant_suffix = os.environ["VARIANT_SUFFIX"]
phase_label = os.environ["PHASE_LABEL"]
base_out = Path(os.environ["BASE_OUT"])
systems = {item.strip() for item in os.environ["SYSTEMS_CSV"].split(",") if item.strip()}
seeds = {item.strip() for item in os.environ["SEEDS_CSV"].split(",") if item.strip()}

source_specs = [
    {
        "path": "results/transition_rich_lista_dense_p256_hardinit_table123_20260430/task_tables/transition_rich_lista_dense_p256_hardinit_table123.tsv",
        "source": "lista_dense_signsplit_p256_hardinit_basin_partition",
        "display": "LISTA",
        "family": "lista",
    },
    {
        "paths": [
            "results/transition_rich_basin_partition_final_seed10_20260409/task_tables/transition_rich_basin_partition.tsv",
            "results/transition_rich_table2_5model_seed15_backfill_20260428/task_tables/transition_rich_table2_5model_seed_backfill.tsv",
        ],
        "source": "lista_blockdiag_signsplit_hardinit_basin_partition",
        "display": "LISTA-BD",
        "family": "lista",
    },
    {
        "path": "results/transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428/task_tables/transition_rich_lista_sb_p256_hardinit_fairness.tsv",
        "source": "lista_dense_softblock_signsplit_p256_hardinit_basin_partition",
        "display": "LISTA-SB",
        "family": "lista",
    },
    {
        "path": "results/transition_rich_sparse_mlp_bd_repaired_table1_20260506/task_tables/sparse_mlp_bd_repaired_table1_tasks.tsv",
        "source": "mlp_sparse_blockdiag_hardinit_basin_partition_control",
        "display": "Sparse MLP-BD",
        "family": "mlp",
    },
    {
        "paths": [
            "results/transition_rich_hardinit_mlp_controls_seed10_20260416/task_tables/transition_rich_basin_partition.tsv",
            "results/transition_rich_table2_5model_seed15_backfill_20260428/task_tables/transition_rich_table2_5model_seed_backfill.tsv",
        ],
        "source": "mlp_sparse_hardinit_basin_partition_control",
        "display": "Sparse MLP",
        "family": "mlp",
    },
    {
        "paths": [
            "results/transition_rich_hardinit_mlp_controls_seed10_20260416/task_tables/transition_rich_basin_partition.tsv",
            "results/transition_rich_table2_5model_seed15_backfill_20260428/task_tables/transition_rich_table2_5model_seed_backfill.tsv",
        ],
        "source": "mlp_zero_sparse_hardinit_basin_partition_control",
        "display": "Dense MLP",
        "family": "mlp",
    },
]

rows = []
fieldnames = []
root_specs = []
for spec in source_specs:
    raw_paths = spec["paths"] if "paths" in spec else [spec["path"]]
    source_paths = [Path(path) for path in raw_paths]
    selected_by_key = {}
    for source_path in source_paths:
        with source_path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not fieldnames:
                fieldnames = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fieldnames:
                raise SystemExit(f"Field mismatch in {source_path}")
            for row in reader:
                if (
                    row.get("model_variant") == spec["source"]
                    and row.get("system_key") in systems
                    and row.get("seed") in seeds
                ):
                    key = (row.get("system_key"), row.get("seed"))
                    selected_by_key.setdefault(key, dict(row))
    selected = list(selected_by_key.values())
    expected = len(systems) * len(seeds)
    if len(selected) != expected:
        raise SystemExit(
            f"{spec['source']}: expected {expected} rows from "
            f"{[str(path) for path in source_paths]}, found {len(selected)}"
        )
    target_label = f"{spec['source']}_{variant_suffix}"
    root_specs.append(
        {
            "label": target_label,
            "display": f"{spec['display']} ({variant_suffix})",
            "family": spec["family"],
            "source": spec["source"],
        }
    )
    for row in selected:
        row["task_id"] = str(len(rows))
        row["phase"] = phase_label
        row["model_variant"] = target_label
        row["num_steps"] = "200000"
        row["eval_profile"] = "full"
        row["normalize_decoder_atoms"] = "true"
        row["sparsity_target"] = sparsity_target
        rows.append(row)

for extra in ("normalize_decoder_atoms", "sparsity_target"):
    if extra not in fieldnames:
        fieldnames.append(extra)

task_tsv.parent.mkdir(parents=True, exist_ok=True)
with task_tsv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

root_specs_file.parent.mkdir(parents=True, exist_ok=True)
with root_specs_file.open("w") as handle:
    for spec in root_specs:
        handle.write(f"{spec['label']}={base_out / phase_label / spec['label']}\n")

manifest = {
    "task_tsv": str(task_tsv),
    "phase_label": phase_label,
    "variant_suffix": variant_suffix,
    "sparsity_target": sparsity_target,
    "normalize_decoder_atoms": True,
    "systems": sorted(systems),
    "seeds": sorted(int(seed) for seed in seeds),
    "task_count": len(rows),
    "source_specs": source_specs,
    "root_specs": root_specs,
    "counts_by_root": dict(Counter(row["model_variant"] for row in rows)),
}
manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"task_tsv": str(task_tsv), "task_count": len(rows)}, indent=2))
PY

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
ROOT_LABELS_CSV=$(cut -d= -f1 "${ROOT_SPECS_FILE}" | paste -sd, -)
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi

while true; do
  CURRENT_JOBS=$(squeue -u "${USER}" -h -r | wc -l)
  if (( CURRENT_JOBS <= MAX_EXISTING_JOBS_BEFORE_SUBMIT )); then
    break
  fi
  echo "Current expanded job count ${CURRENT_JOBS} exceeds ${MAX_EXISTING_JOBS_BEFORE_SUBMIT}; sleeping."
  sleep "${SUBMIT_WAIT_SECONDS}"
done

ARRAY_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" \
    sbatch --parsable --time="${TRAIN_TIME_LIMIT}" \
      --array=0-$((TASK_COUNT - 1))%"${ARRAY_THROTTLE}" \
      scripts/run_paper_benchmark_array.sh
)

COLLECT_JOB_ID=$(
  ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" \
  OUT_DIR="${COLLECT_DIR}" \
  HORIZONS_CSV="100,500,1000" \
  GOOD_THRESHOLD="50" \
    sbatch --parsable --dependency=afterany:"${ARRAY_JOB_ID}" \
      scripts/collect_transition_rich_basin_partition.sh
)

INTERP_JOBS=()
while IFS='=' read -r ROOT_LABEL _; do
  [[ -n "${ROOT_LABEL}" ]] || continue
  INTERP_SHARD_DIR="${INTERP_DIR}/shards/${ROOT_LABEL}"
  mkdir -p "${INTERP_SHARD_DIR}"
  JOB_ID=$(
    ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" \
    OUT_DIR="${INTERP_SHARD_DIR}" \
    ROOT_LABELS_CSV="${ROOT_LABEL}" \
    PROGRESS_EVERY_RUNS="1" \
    FLUSH_EVERY_RUNS="5" \
    DEPTH_SLICE_MODE="per_basin" \
      sbatch --parsable \
        --dependency=afterok:"${COLLECT_JOB_ID}" \
        --job-name="tr_interp_normctl" \
        --time="12:00:00" \
        --cpus-per-task="4" \
        --mem="16G" \
        --output="${INTERP_DIR}/logs/${ROOT_LABEL}-%A.out" \
        --error="${INTERP_DIR}/logs/${ROOT_LABEL}-%A.err" \
        scripts/reduce_transition_rich_interpretability_metrics.sh
  )
  INTERP_JOBS+=("${JOB_ID}")
done < "${ROOT_SPECS_FILE}"

INTERP_DEP=$(IFS=:; echo "${INTERP_JOBS[*]}")
INTERP_MERGE_JOB_ID=$(
  SHARDS_DIR="${INTERP_DIR}/shards" \
  OUT_DIR="${INTERP_DIR}" \
  ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" \
  ROOT_LABELS_CSV="${ROOT_LABELS_CSV}" \
    sbatch --parsable \
      --dependency=afterok:"${INTERP_DEP}" \
      --time="00:30:00" \
      --mem="4G" \
      --output="${INTERP_DIR}/logs/merge-%A.out" \
      --error="${INTERP_DIR}/logs/merge-%A.err" \
      scripts/merge_transition_rich_interpretability_shards.sh
)

uv run python - "${QUEUE_RECORD_JSON}" <<PY
import json
from pathlib import Path

payload = {
    "date_tag": "${DATE_TAG}",
    "experiment_tag": "${EXPERIMENT_TAG}",
    "phase_label": "${PHASE_LABEL}",
    "variant_suffix": "${VARIANT_SUFFIX}",
    "sparsity_target": "${SPARSITY_TARGET}",
    "normalize_decoder_atoms": True,
    "base_out": "${BASE_OUT}",
    "results_dir": "${RESULTS_DIR}",
    "task_tsv": "${TASK_TSV}",
    "manifest_json": "${MANIFEST_JSON}",
    "root_specs_file": "${ROOT_SPECS_FILE}",
    "task_count": ${TASK_COUNT},
    "array_job_id": "${ARRAY_JOB_ID}",
    "collect_job_id": "${COLLECT_JOB_ID}",
    "interpretability_shard_job_ids": ${INTERP_JOBS[@]+"["}$(printf '"%s",' "${INTERP_JOBS[@]}" | sed 's/,$//')${INTERP_JOBS[@]+"]"},
    "interpretability_merge_job_id": "${INTERP_MERGE_JOB_ID}",
    "systems_csv": "${SYSTEMS_CSV}",
    "seeds_csv": "${SEEDS_CSV}",
    "training_device": "cuda via scripts/run_paper_benchmark_array.sh"
}
Path("${QUEUE_RECORD_JSON}").write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "Queued controlled KAE ${VARIANT_SUFFIX} table packet."
echo "Training array: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Support diagnostics shards: ${INTERP_JOBS[*]} -> ${INTERP_MERGE_JOB_ID}"
echo "Results dir: ${RESULTS_DIR}"
