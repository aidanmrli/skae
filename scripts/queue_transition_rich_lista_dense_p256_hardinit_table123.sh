#!/bin/bash
#
# Queue the matched dense-K LISTA Table 1/2/3 add-on:
#   same hard-init, d_z=256, sequence length, 200k steps, and sparsity penalty
#   as the matched LISTA-SB Table 1/2/3 row, but with no K-structure
#   regularizer (`soft_block=0`, dense Koopman matrix).
#
# Submit with:
#   sbatch scripts/queue_transition_rich_lista_dense_p256_hardinit_table123.sh
#
# Optional env vars:
#   EXPERIMENT_TAG=transition_rich_lista_dense_p256_hardinit_table123_20260430
#   ARRAY_THROTTLE=64
#   MAX_EXISTING_JOBS_BEFORE_SUBMIT=650

#SBATCH --job-name=queue_ld_p256
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-lista-dense-p256-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-lista-dense-p256-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run on a compute node."
  echo "Submit it with: sbatch scripts/queue_transition_rich_lista_dense_p256_hardinit_table123.sh"
  exit 2
fi

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-20260430}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-transition_rich_lista_dense_p256_hardinit_table123_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-transition_rich_basin_partition}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect_pass0}"
INTERP_DIR="${INTERP_DIR:-${RESULTS_DIR}/interpretability_pass0}"
SELF_ROUTED_DIR="${SELF_ROUTED_DIR:-${RESULTS_DIR}/self_routed_forecasting}"
REFRESH_DIR="${REFRESH_DIR:-${RESULTS_DIR}/periodic_support_refresh}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-${RESULTS_DIR}/queue_logs}"
AUTOMATION_DIR="${AUTOMATION_DIR:-${RESULTS_DIR}/automation}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-64}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-650}"

SOURCE_TSV="${SOURCE_TSV:-results/transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428/task_tables/transition_rich_lista_sb_p256_hardinit_fairness.tsv}"
SOURCE_VARIANT="${SOURCE_VARIANT:-lista_dense_softblock_signsplit_p256_hardinit_basin_partition}"
TARGET_VARIANT="${TARGET_VARIANT:-lista_dense_signsplit_p256_hardinit_basin_partition}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}"
SEED_SPLITS_SEMICOLON="${SEED_SPLITS_SEMICOLON:-0,1,2,3,4;5,6,7,8,9;10,11,12,13,14}"

mkdir -p \
  "${TASK_DIR}" \
  "${ROOT_SPEC_DIR}" \
  "${COLLECT_DIR}" \
  "${INTERP_DIR}/shards" \
  "${SELF_ROUTED_DIR}/shards" \
  "${REFRESH_DIR}/shards" \
  "${QUEUE_LOG_DIR}" \
  "${AUTOMATION_DIR}"

TASK_TSV="${TASK_DIR}/transition_rich_lista_dense_p256_hardinit_table123.tsv"
MANIFEST_JSON="${TASK_DIR}/transition_rich_lista_dense_p256_hardinit_table123_manifest.json"
ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/transition_rich_lista_dense_p256_hardinit_roots.txt"

PHASE_LABEL="${PHASE_LABEL}" \
SOURCE_VARIANT="${SOURCE_VARIANT}" \
TARGET_VARIANT="${TARGET_VARIANT}" \
SEEDS_CSV="${SEEDS_CSV}" \
  uv run python - "${SOURCE_TSV}" "${TASK_TSV}" "${MANIFEST_JSON}" <<'PY'
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

source_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])

phase_label = os.environ["PHASE_LABEL"]
source_variant = os.environ["SOURCE_VARIANT"]
target_variant = os.environ["TARGET_VARIANT"]
requested_seeds = [seed.strip() for seed in os.environ["SEEDS_CSV"].split(",") if seed.strip()]

with source_path.open(newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fields = list(reader.fieldnames or [])
    source_rows = [row for row in reader if row.get("model_variant") == source_variant]

if not source_rows:
    raise SystemExit(f"No rows found for source variant {source_variant} in {source_path}")

templates = {}
for row in source_rows:
    system = row["system_key"]
    seed = str(row["seed"])
    if system not in templates or seed == "0":
        templates[system] = dict(row)

if len(templates) != 17:
    raise SystemExit(f"Expected 17 source systems for {source_variant}, found {len(templates)}")

out_rows = []
for system in sorted(templates):
    template = templates[system]
    for seed in requested_seeds:
        row = dict(template)
        row["task_id"] = str(len(out_rows))
        row["phase"] = phase_label
        row["model_variant"] = target_variant
        row["seed"] = str(seed)
        row["num_steps"] = "200000"
        row["batch_size"] = "256"
        row["target_size"] = "256"
        row["sequence_length"] = "8"
        row["sparsity_coeff"] = "0.003"
        row["k_structure"] = "dense"
        row["k_block_size"] = ""
        row["k_num_blocks"] = ""
        row["soft_block"] = "0"
        row["soft_block_num_blocks"] = ""
        row["soft_block_weight"] = ""
        row["soft_block_norm"] = ""
        row["eval_profile"] = "full"
        out_rows.append(row)

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(out_rows)

settings = {
    "num_steps": 200000,
    "batch_size": 256,
    "target_size": 256,
    "sequence_length": 8,
    "sparsity_coeff": 0.003,
    "k_structure": "dense",
    "soft_block": False,
    "lista_alpha": 0.15,
    "lista_num_loops": 2,
    "lista_final_op": "sign_split",
}
manifest = {
    "experiment": "transition_rich_lista_dense_p256_hardinit_table123",
    "source_task_tsv": str(source_path),
    "source_variant": source_variant,
    "target_variant": target_variant,
    "task_tsv": str(out_path),
    "phase_label": phase_label,
    "seeds": [int(seed) for seed in requested_seeds],
    "settings": settings,
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

echo "Generated ${TASK_COUNT} matched dense-K LISTA tasks."
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

INTERP_SHARD_DIR="${INTERP_DIR}/shards/${TARGET_VARIANT}"
mkdir -p "${INTERP_SHARD_DIR}" "${INTERP_DIR}/logs"
INTERP_SHARD_JOB_ID=$(
  ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" \
  OUT_DIR="${INTERP_SHARD_DIR}" \
  ROOT_LABELS_CSV="${TARGET_VARIANT}" \
  PROGRESS_EVERY_RUNS="1" \
  FLUSH_EVERY_RUNS="5" \
  DEPTH_SLICE_MODE="global" \
    sbatch \
      --dependency=afterok:"${COLLECT_JOB_ID}" \
      --job-name="tr_interp_${TARGET_VARIANT}" \
      --time="12:00:00" \
      --cpus-per-task="4" \
      --mem="16G" \
      --output="${INTERP_DIR}/logs/${TARGET_VARIANT}-%A.out" \
      --error="${INTERP_DIR}/logs/${TARGET_VARIANT}-%A.err" \
      scripts/reduce_transition_rich_interpretability_metrics.sh | awk '{print $4}'
)
INTERP_MERGE_JOB_ID=$(
  SHARDS_DIR="${INTERP_DIR}/shards" \
  OUT_DIR="${INTERP_DIR}" \
  ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" \
  ROOT_LABELS_CSV="${TARGET_VARIANT}" \
    sbatch \
      --dependency=afterok:"${INTERP_SHARD_JOB_ID}" \
      --time="00:30:00" \
      --mem="4G" \
      --output="${INTERP_DIR}/logs/merge-%A.out" \
      --error="${INTERP_DIR}/logs/merge-%A.err" \
      scripts/merge_transition_rich_interpretability_shards.sh | awk '{print $4}'
)

SELF_ROUTED_JOB_IDS=()
IFS=';' read -r -a SEED_SPLITS <<< "${SEED_SPLITS_SEMICOLON}"
mkdir -p "${SELF_ROUTED_DIR}/logs"
for seed_split in "${SEED_SPLITS[@]}"; do
  split_slug="seeds_${seed_split//,/_}"
  shard_out_dir="${SELF_ROUTED_DIR}/shards/${TARGET_VARIANT}__${split_slug}"
  mkdir -p "${shard_out_dir}"
  shard_job_id=$(
    ROWS_CSVS="${COLLECT_DIR}/forecasting_rows.csv" \
    OUT_DIR="${shard_out_dir}" \
    ROOT_LABELS_CSV="${TARGET_VARIANT}" \
    SEEDS_CSV="${seed_split}" \
    SUPPORT_DEFINITIONS="topk:8" \
    DEPTH_STRATA="all,q4" \
    ROLLOUT_MODES="global_k,support_gated_k,support_local_centered,family_local_centered" \
    HORIZONS="100,500,1000" \
    PROGRESS_EVERY_RUNS="1" \
    FLUSH_EVERY_RUNS="1" \
      sbatch \
        --dependency=afterok:"${COLLECT_JOB_ID}" \
        --job-name="tr_srf_${split_slug}" \
        --time="12:00:00" \
        --cpus-per-task="4" \
        --mem="24G" \
        --output="${SELF_ROUTED_DIR}/logs/${TARGET_VARIANT}__${split_slug}-%A.out" \
        --error="${SELF_ROUTED_DIR}/logs/${TARGET_VARIANT}__${split_slug}-%A.err" \
        scripts/run_transition_rich_self_routed_forecasting.sh | awk '{print $4}'
  )
  SELF_ROUTED_JOB_IDS+=("${shard_job_id}")
done
SELF_ROUTED_DEPENDENCY="$(IFS=:; echo "${SELF_ROUTED_JOB_IDS[*]}")"
SELF_ROUTED_MERGE_JOB_ID=$(
  SHARDS_DIR="${SELF_ROUTED_DIR}/shards" \
  OUT_DIR="${SELF_ROUTED_DIR}" \
  ROWS_CSVS="${COLLECT_DIR}/forecasting_rows.csv" \
  ROOT_LABELS_CSV="${TARGET_VARIANT}" \
  SEEDS_CSV="${SEEDS_CSV}" \
    sbatch \
      --dependency=afterok:"${SELF_ROUTED_DEPENDENCY}" \
      --time="00:30:00" \
      --mem="4G" \
      --output="${SELF_ROUTED_DIR}/logs/merge-%A.out" \
      --error="${SELF_ROUTED_DIR}/logs/merge-%A.err" \
      scripts/merge_transition_rich_self_routed_forecasting_shards.sh | awk '{print $4}'
)

REFRESH_JOB_IDS=()
mkdir -p "${REFRESH_DIR}/logs"
for seed_split in "${SEED_SPLITS[@]}"; do
  split_slug="seeds_${seed_split//,/_}"
  shard_out_dir="${REFRESH_DIR}/shards/${TARGET_VARIANT}__${split_slug}"
  mkdir -p "${shard_out_dir}"
  refresh_job_id=$(
    ROWS_CSVS="${COLLECT_DIR}/forecasting_rows.csv" \
    OUT_DIR="${shard_out_dir}" \
    ROOT_LABELS_CSV="${TARGET_VARIANT}" \
    SEEDS_CSV="${seed_split}" \
    SUPPORT_DEFINITIONS="topk:8" \
    REENCODE_PERIODS="1,10" \
    PROGRESS_EVERY_RUNS="1" \
    FLUSH_EVERY_RUNS="1" \
      sbatch \
        --dependency=afterok:"${COLLECT_JOB_ID}" \
        --job-name="tr_refresh_${split_slug}" \
        --time="12:00:00" \
        --cpus-per-task="4" \
        --mem="24G" \
        --output="${REFRESH_DIR}/logs/${TARGET_VARIANT}__${split_slug}-%A.out" \
        --error="${REFRESH_DIR}/logs/${TARGET_VARIANT}__${split_slug}-%A.err" \
        scripts/run_transition_rich_periodic_support_refresh.sh | awk '{print $4}'
  )
  REFRESH_JOB_IDS+=("${refresh_job_id}")
done
REFRESH_DEPENDENCY="$(IFS=:; echo "${REFRESH_JOB_IDS[*]}")"
REFRESH_MERGE_JOB_ID=$(
  SHARDS_DIR="${REFRESH_DIR}/shards" \
  OUT_DIR="${REFRESH_DIR}/merged" \
  ROWS_CSVS="${COLLECT_DIR}/forecasting_rows.csv" \
  ROOT_LABELS_CSV="${TARGET_VARIANT}" \
  SEEDS_CSV="${SEEDS_CSV}" \
    sbatch \
      --dependency=afterok:"${REFRESH_DEPENDENCY}" \
      --time="00:30:00" \
      --mem="4G" \
      --output="${REFRESH_DIR}/logs/merge-%A.out" \
      --error="${REFRESH_DIR}/logs/merge-%A.err" \
      scripts/merge_transition_rich_periodic_support_refresh_shards.sh | awk '{print $4}'
)

cat > "${AUTOMATION_DIR}/table123_queue.json" <<EOF
{
  "target_variant": "${TARGET_VARIANT}",
  "results_dir": "${RESULTS_DIR}",
  "base_out": "${BASE_OUT}",
  "task_tsv": "${TASK_TSV}",
  "manifest_json": "${MANIFEST_JSON}",
  "root_specs_file": "${ROOT_SPECS_FILE}",
  "task_count": ${TASK_COUNT},
  "array_job_id": "${ARRAY_JOB_ID}",
  "collect_job_id": "${COLLECT_JOB_ID}",
  "interpretability_shard_job_id": "${INTERP_SHARD_JOB_ID}",
  "interpretability_merge_job_id": "${INTERP_MERGE_JOB_ID}",
  "self_routed_shard_job_ids": "$(IFS=,; echo "${SELF_ROUTED_JOB_IDS[*]}")",
  "self_routed_merge_job_id": "${SELF_ROUTED_MERGE_JOB_ID}",
  "refresh_shard_job_ids": "$(IFS=,; echo "${REFRESH_JOB_IDS[*]}")",
  "refresh_merge_job_id": "${REFRESH_MERGE_JOB_ID}"
}
EOF

{
  printf 'RESULTS_DIR=%q\n' "${RESULTS_DIR}"
  printf 'BASE_OUT=%q\n' "${BASE_OUT}"
  printf 'TASK_TSV=%q\n' "${TASK_TSV}"
  printf 'MANIFEST_JSON=%q\n' "${MANIFEST_JSON}"
  printf 'ROOT_SPECS_FILE=%q\n' "${ROOT_SPECS_FILE}"
  printf 'TASK_COUNT=%q\n' "${TASK_COUNT}"
  printf 'ARRAY_JOB_ID=%q\n' "${ARRAY_JOB_ID}"
  printf 'COLLECT_JOB_ID=%q\n' "${COLLECT_JOB_ID}"
  printf 'INTERP_SHARD_JOB_ID=%q\n' "${INTERP_SHARD_JOB_ID}"
  printf 'INTERP_MERGE_JOB_ID=%q\n' "${INTERP_MERGE_JOB_ID}"
  printf 'SELF_ROUTED_JOB_IDS_CSV=%q\n' "$(IFS=,; echo "${SELF_ROUTED_JOB_IDS[*]}")"
  printf 'SELF_ROUTED_MERGE_JOB_ID=%q\n' "${SELF_ROUTED_MERGE_JOB_ID}"
  printf 'REFRESH_JOB_IDS_CSV=%q\n' "$(IFS=,; echo "${REFRESH_JOB_IDS[*]}")"
  printf 'REFRESH_MERGE_JOB_ID=%q\n' "${REFRESH_MERGE_JOB_ID}"
  printf 'ARRAY_THROTTLE=%q\n' "${ARRAY_THROTTLE}"
  printf 'MAX_EXISTING_JOBS_BEFORE_SUBMIT=%q\n' "${MAX_EXISTING_JOBS_BEFORE_SUBMIT}"
} > "${QUEUE_LOG_DIR}/launch_record.env"

echo "Queued matched dense-K LISTA Table 1/2/3 add-on."
echo "Array job: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Interpretability: ${INTERP_SHARD_JOB_ID} -> ${INTERP_MERGE_JOB_ID}"
echo "Self-routed shards: $(IFS=,; echo "${SELF_ROUTED_JOB_IDS[*]}") -> ${SELF_ROUTED_MERGE_JOB_ID}"
echo "Refresh shards: $(IFS=,; echo "${REFRESH_JOB_IDS[*]}") -> ${REFRESH_MERGE_JOB_ID}"
echo "Results dir: ${RESULTS_DIR}"
echo "Base out: ${BASE_OUT}"
