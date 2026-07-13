#!/bin/bash
#
# Queue the staged C_stab-routed learned-intercept local-K LISTA experiment for
# the retained 10-system Dysts dt-x30 roster.
#
# Submit:
#   sbatch scripts/queue_staged_cstab_local_k_dysts_dt30.sh
#
# Optional env vars:
#   DATE_TAG=20260516
#   EXPERIMENT_TAG=staged_cstab_learned_intercept_k_lista_dysts_dt30_${DATE_TAG}
#   ARRAY_THROTTLE=16
#   SYSTEMS_CSV=dysts:Chua,dysts:Dadras
#   SEEDS_CSV=0,1
#   QUEUE_DYSTS_LONG_REEVAL=1
#   DYSTS_REEVAL_HORIZONS="100 500 1000 1500 2000 3000 4000 5000"
#   DYSTS_REEVAL_PERIODS="10 25 50 100 150 200"
#
# Each training row uses the existing Table 1 Dysts LISTA recipe:
#   d_z=256, seq_len=10, num_steps=100000, dt_multiplier=30, sparse coeff=0.006.

#SBATCH --job-name=queue_dysts_cstab
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-dysts-cstab-local-k-%j.out
#SBATCH -e /network/scratch/l/lia/skae/queue-dysts-cstab-local-k-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run on a compute node."
  echo "Submit it with: sbatch scripts/queue_staged_cstab_local_k_dysts_dt30.sh"
  exit 2
fi

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-20260516}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-staged_cstab_learned_intercept_k_lista_dysts_dt30_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-dysts_dt30_cstab_learned_intercept_k_p256_seq10_100k}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-${RESULTS_DIR}/queue_logs}"
AUTOMATION_DIR="${AUTOMATION_DIR:-${RESULTS_DIR}/automation}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-16}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-650}"
SUBMIT_WAIT_SECONDS="${SUBMIT_WAIT_SECONDS:-300}"

SOURCE_TSV="${SOURCE_TSV:-results/dysts_dt30_basinblock_p256_seq10_100k_20260430/task_tables/dysts_dt30_basinblock_tasks.tsv}"
SOURCE_VARIANT="${SOURCE_VARIANT:-lista}"
TARGET_VARIANT="${TARGET_VARIANT:-lista_cstab_learned_intercept_k_staged_p256_seq10_dt30}"
BASELINE_ROOT_LABEL="${BASELINE_ROOT_LABEL:-lista_global_k_p256_seq10_dt30}"
BASELINE_ROOT="${BASELINE_ROOT:-/network/scratch/l/lia/skae/dysts_dt30_basinblock_p256_seq10_100k_20260430/dysts_dt30_basinblock_p256_seq10_100k/lista}"
SYSTEMS_CSV="${SYSTEMS_CSV:-dysts:Chua,dysts:Dadras,dysts:DequanLi,dysts:Hadley,dysts:LuChenCheng,dysts:QiChen,dysts:Sakarya,dysts:SanUmSrisuchinwong,dysts:ShimizuMorioka,dysts:WangSun}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}"
EXPECTED_TASK_COUNT="${EXPECTED_TASK_COUNT:-150}"

SUPPORT_DEFINITION="${SUPPORT_DEFINITION:-absolute:0.001}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.4}"
SUPPORT_FIT_BATCHES="${SUPPORT_FIT_BATCHES:-16}"
MIN_FAMILY_TRANSITIONS="${MIN_FAMILY_TRANSITIONS:-1}"
ROUTING_OBJECT="${ROUTING_OBJECT:-stable_support_component}"
STABLE_BASE_OBJECT="${STABLE_BASE_OBJECT:-family}"
STABLE_BASE_FAMILY_JACCARD="${STABLE_BASE_FAMILY_JACCARD:-0.8}"
STABLE_TAIL_WINDOW="${STABLE_TAIL_WINDOW:-32}"
STABLE_MIN_EDGE_COUNT="${STABLE_MIN_EDGE_COUNT:-2}"
STABLE_MIN_EDGE_PROBABILITY="${STABLE_MIN_EDGE_PROBABILITY:-0.02}"
STABLE_MAX_RECURRENT_OUT_PROBABILITY="${STABLE_MAX_RECURRENT_OUT_PROBABILITY:-0.05}"
STABLE_MIN_TAIL_COUNT="${STABLE_MIN_TAIL_COUNT:-8}"
STABLE_MIN_ABSORPTION_OBSERVATIONS="${STABLE_MIN_ABSORPTION_OBSERVATIONS:-8}"
STABLE_MIN_ABSORPTION_CONFIDENCE="${STABLE_MIN_ABSORPTION_CONFIDENCE:-0.80}"
STABLE_FIT_TRAJECTORIES="${STABLE_FIT_TRAJECTORIES:-512}"
STABLE_FIT_TRAJECTORY_LENGTH="${STABLE_FIT_TRAJECTORY_LENGTH:-192}"
STABLE_FIT_SEED_OFFSET="${STABLE_FIT_SEED_OFFSET:-271828}"
LOCAL_MAP_PARAMETERIZATION="${LOCAL_MAP_PARAMETERIZATION:-source_target_affine_learned_intercept}"
LOCAL_LR="${LOCAL_LR:-}"
STAGE2_SELECTION_METRIC="${STAGE2_SELECTION_METRIC:-best_periodic_horizon_mse}"
STAGE2_SELECTION_PERIODS="${STAGE2_SELECTION_PERIODS:-10,25,50,100,150,200}"
STAGE2_SELECTION_HORIZONS="${STAGE2_SELECTION_HORIZONS:-100,500,1000,1500,2000,3000,4000,5000}"
STAGE2_SELECTION_BATCH_SIZE="${STAGE2_SELECTION_BATCH_SIZE:-32}"
STAGE2_SELECTION_SEED_OFFSET="${STAGE2_SELECTION_SEED_OFFSET:-12345}"
EVAL_PERIODIC_PERIODS_OVERRIDE="${EVAL_PERIODIC_PERIODS_OVERRIDE:-10,25,50,100,150,200}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

QUEUE_DYSTS_LONG_REEVAL="${QUEUE_DYSTS_LONG_REEVAL:-1}"
DYSTS_REEVAL_DIR="${DYSTS_REEVAL_DIR:-${RESULTS_DIR}/long_horizon_eval}"
DYSTS_REEVAL_OUTPUT_TAG="${DYSTS_REEVAL_OUTPUT_TAG:-dysts_dt30_cstab_vs_lista_h100_to_h5000_p6periods}"
DYSTS_REEVAL_HORIZONS="${DYSTS_REEVAL_HORIZONS:-100 500 1000 1500 2000 3000 4000 5000}"
DYSTS_REEVAL_PERIODS="${DYSTS_REEVAL_PERIODS:-10 25 50 100 150 200}"
DYSTS_REEVAL_ARRAY_PARALLEL="${DYSTS_REEVAL_ARRAY_PARALLEL:-32}"
DYSTS_REEVAL_PACK_SIZE="${DYSTS_REEVAL_PACK_SIZE:-8}"
DYSTS_REEVAL_TIME_LIMIT="${DYSTS_REEVAL_TIME_LIMIT:-12:00:00}"
DYSTS_REEVAL_DEVICE="${DYSTS_REEVAL_DEVICE:-cpu}"
DYSTS_REEVAL_CACHE_PROFILE="${DYSTS_REEVAL_CACHE_PROFILE:-full}"
DYSTS_REEVAL_CACHE_SPLIT="${DYSTS_REEVAL_CACHE_SPLIT:-test}"
DYSTS_REEVAL_BATCH_SIZE="${DYSTS_REEVAL_BATCH_SIZE:-100}"
DYSTS_DT_MULTIPLIER="${DYSTS_DT_MULTIPLIER:-30}"

mkdir -p "${TASK_DIR}" "${ROOT_SPEC_DIR}" "${QUEUE_LOG_DIR}" "${AUTOMATION_DIR}"

TASK_TSV="${TASK_DIR}/staged_cstab_learned_intercept_k_dysts_dt30_tasks.tsv"
MANIFEST_JSON="${TASK_DIR}/staged_cstab_learned_intercept_k_dysts_dt30_manifest.json"
ROOT_SPECS_TSV="${ROOT_SPEC_DIR}/staged_cstab_learned_intercept_k_dysts_dt30_roots.tsv"

PHASE_LABEL="${PHASE_LABEL}" \
SOURCE_VARIANT="${SOURCE_VARIANT}" \
TARGET_VARIANT="${TARGET_VARIANT}" \
SYSTEMS_CSV="${SYSTEMS_CSV}" \
SEEDS_CSV="${SEEDS_CSV}" \
SUPPORT_DEFINITION="${SUPPORT_DEFINITION}" \
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD}" \
SUPPORT_FIT_BATCHES="${SUPPORT_FIT_BATCHES}" \
MIN_FAMILY_TRANSITIONS="${MIN_FAMILY_TRANSITIONS}" \
ROUTING_OBJECT="${ROUTING_OBJECT}" \
STABLE_BASE_OBJECT="${STABLE_BASE_OBJECT}" \
STABLE_BASE_FAMILY_JACCARD="${STABLE_BASE_FAMILY_JACCARD}" \
STABLE_FIT_TRAJECTORIES="${STABLE_FIT_TRAJECTORIES}" \
STABLE_FIT_TRAJECTORY_LENGTH="${STABLE_FIT_TRAJECTORY_LENGTH}" \
LOCAL_MAP_PARAMETERIZATION="${LOCAL_MAP_PARAMETERIZATION}" \
STAGE2_SELECTION_METRIC="${STAGE2_SELECTION_METRIC}" \
STAGE2_SELECTION_PERIODS="${STAGE2_SELECTION_PERIODS}" \
STAGE2_SELECTION_HORIZONS="${STAGE2_SELECTION_HORIZONS}" \
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
systems = [item.strip() for item in os.environ["SYSTEMS_CSV"].split(",") if item.strip()]
requested_systems = set(systems)
requested_seeds = {item.strip() for item in os.environ["SEEDS_CSV"].split(",") if item.strip()}

with source_path.open(newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fields = list(reader.fieldnames or [])
    source_rows = [
        dict(row)
        for row in reader
        if row.get("model_variant") == source_variant
        and row.get("system_key") in requested_systems
        and str(row.get("seed", "")).strip() in requested_seeds
    ]

if not source_rows:
    raise SystemExit(f"No rows found for {source_variant=} in {source_path}")

system_order = {system: idx for idx, system in enumerate(systems)}
source_rows.sort(key=lambda row: (system_order[row["system_key"]], int(row["seed"])))
out_rows = []
for row in source_rows:
    out = dict(row)
    out["task_id"] = str(len(out_rows))
    out["phase"] = phase_label
    out["model_variant"] = target_variant
    out["eval_profile"] = "full"
    out_rows.append(out)

num_steps_values = sorted({str(row.get("num_steps", "")).strip() for row in out_rows})
target_sizes = sorted({str(row.get("target_size", "")).strip() for row in out_rows})
sequence_lengths = sorted({str(row.get("sequence_length", "")).strip() for row in out_rows})
source_variants = sorted({str(row.get("config_name", "")).strip() for row in out_rows})
if num_steps_values != ["100000"]:
    raise SystemExit(f"Expected Dysts Table 1 num_steps=100000, found {num_steps_values}")
if target_sizes != ["256"]:
    raise SystemExit(f"Expected target_size=256, found {target_sizes}")
if sequence_lengths != ["10"]:
    raise SystemExit(f"Expected sequence_length=10, found {sequence_lengths}")
if source_variants != ["lista_parity_generic_sparse"]:
    raise SystemExit(f"Expected LISTA config, found {source_variants}")

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(out_rows)

counts_by_system = Counter(row["system_key"] for row in out_rows)
manifest = {
    "experiment": "staged_cstab_learned_intercept_k_dysts_dt30",
    "source_task_tsv": str(source_path),
    "source_variant": source_variant,
    "target_variant": target_variant,
    "task_tsv": str(out_path),
    "phase_label": phase_label,
    "systems": systems,
    "seeds": sorted(int(seed) for seed in requested_seeds),
    "num_tasks": len(out_rows),
    "counts_by_system": dict(counts_by_system),
    "settings": {
        "total_steps": 100000,
        "stage1_joint_steps": 50000,
        "stage2_local_steps": 50000,
        "batch_size": 256,
        "target_size": 256,
        "sequence_length": 10,
        "sparsity_coeff": 0.006,
        "k_structure": "dense",
        "lista_alpha": 0.15,
        "lista_num_loops": 1,
        "lista_final_op": "relu",
        "dt_multiplier": 30,
        "support_definition": os.environ["SUPPORT_DEFINITION"],
        "family_jaccard_threshold": float(os.environ["FAMILY_JACCARD_THRESHOLD"]),
        "support_fit_batches": int(os.environ["SUPPORT_FIT_BATCHES"]),
        "min_family_transitions": int(os.environ["MIN_FAMILY_TRANSITIONS"]),
        "routing_object": os.environ["ROUTING_OBJECT"],
        "stable_base_object": os.environ["STABLE_BASE_OBJECT"],
        "stable_base_family_jaccard": float(os.environ["STABLE_BASE_FAMILY_JACCARD"]),
        "stable_fit_trajectories": int(os.environ["STABLE_FIT_TRAJECTORIES"]),
        "stable_fit_trajectory_length": int(os.environ["STABLE_FIT_TRAJECTORY_LENGTH"]),
        "local_map_parameterization": os.environ["LOCAL_MAP_PARAMETERIZATION"],
        "stage2_selection_metric": os.environ["STAGE2_SELECTION_METRIC"],
        "stage2_selection_periods": os.environ["STAGE2_SELECTION_PERIODS"],
        "stage2_selection_horizons": os.environ["STAGE2_SELECTION_HORIZONS"],
    },
    "notes": [
        "Training-time routing is label-free and does not use Dysts basin/scroll counts.",
        "Rows are filtered to the retained 10-system Dysts roster used by the paper-facing dt-x30 table.",
    ],
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"task_tsv": str(out_path), "num_tasks": len(out_rows)}, indent=2))
PY

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi
if [[ "${EXPECTED_TASK_COUNT}" != "0" ]] && (( TASK_COUNT != EXPECTED_TASK_COUNT )); then
  echo "Expected ${EXPECTED_TASK_COUNT} tasks but generated ${TASK_COUNT}"
  exit 1
fi

{
  printf 'label\tdisplay_name\tmodel_family\troot_dir\n'
  printf '%s\t%s\t%s\t%s\n' \
    "${TARGET_VARIANT}" \
    "Staged C_stab learned-intercept LISTA" \
    "lista" \
    "${BASE_OUT}/${PHASE_LABEL}/${TARGET_VARIANT}"
  printf '%s\t%s\t%s\t%s\n' \
    "${BASELINE_ROOT_LABEL}" \
    "Global-K LISTA" \
    "lista" \
    "${BASELINE_ROOT}"
} > "${ROOT_SPECS_TSV}"

echo "Generated ${TASK_COUNT} staged C_stab Dysts tasks."
echo "Task TSV: ${TASK_TSV}"
echo "Manifest: ${MANIFEST_JSON}"
echo "Root specs: ${ROOT_SPECS_TSV}"

while true; do
  CURRENT_JOBS=$(squeue -u "${USER}" -h -r | wc -l)
  if (( CURRENT_JOBS <= MAX_EXISTING_JOBS_BEFORE_SUBMIT )); then
    break
  fi
  echo "Current expanded job count ${CURRENT_JOBS} exceeds ${MAX_EXISTING_JOBS_BEFORE_SUBMIT}; sleeping."
  sleep "${SUBMIT_WAIT_SECONDS}"
done

ARRAY_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" \
  BASE_OUT="${BASE_OUT}" \
  SUPPORT_DEFINITION="${SUPPORT_DEFINITION}" \
  FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD}" \
  SUPPORT_FIT_BATCHES="${SUPPORT_FIT_BATCHES}" \
  MIN_FAMILY_TRANSITIONS="${MIN_FAMILY_TRANSITIONS}" \
  ROUTING_OBJECT="${ROUTING_OBJECT}" \
  STABLE_BASE_OBJECT="${STABLE_BASE_OBJECT}" \
  STABLE_BASE_FAMILY_JACCARD="${STABLE_BASE_FAMILY_JACCARD}" \
  STABLE_TAIL_WINDOW="${STABLE_TAIL_WINDOW}" \
  STABLE_MIN_EDGE_COUNT="${STABLE_MIN_EDGE_COUNT}" \
  STABLE_MIN_EDGE_PROBABILITY="${STABLE_MIN_EDGE_PROBABILITY}" \
  STABLE_MAX_RECURRENT_OUT_PROBABILITY="${STABLE_MAX_RECURRENT_OUT_PROBABILITY}" \
  STABLE_MIN_TAIL_COUNT="${STABLE_MIN_TAIL_COUNT}" \
  STABLE_MIN_ABSORPTION_OBSERVATIONS="${STABLE_MIN_ABSORPTION_OBSERVATIONS}" \
  STABLE_MIN_ABSORPTION_CONFIDENCE="${STABLE_MIN_ABSORPTION_CONFIDENCE}" \
  STABLE_FIT_TRAJECTORIES="${STABLE_FIT_TRAJECTORIES}" \
  STABLE_FIT_TRAJECTORY_LENGTH="${STABLE_FIT_TRAJECTORY_LENGTH}" \
  STABLE_FIT_SEED_OFFSET="${STABLE_FIT_SEED_OFFSET}" \
  LOCAL_MAP_PARAMETERIZATION="${LOCAL_MAP_PARAMETERIZATION}" \
  LOCAL_LR="${LOCAL_LR}" \
  STAGE2_SELECTION_METRIC="${STAGE2_SELECTION_METRIC}" \
  STAGE2_SELECTION_PERIODS="${STAGE2_SELECTION_PERIODS}" \
  STAGE2_SELECTION_HORIZONS="${STAGE2_SELECTION_HORIZONS}" \
  STAGE2_SELECTION_BATCH_SIZE="${STAGE2_SELECTION_BATCH_SIZE}" \
  STAGE2_SELECTION_SEED_OFFSET="${STAGE2_SELECTION_SEED_OFFSET}" \
  EVAL_PERIODIC_PERIODS_OVERRIDE="${EVAL_PERIODIC_PERIODS_OVERRIDE}" \
  EVAL_PROFILE="${EVAL_PROFILE}" \
    sbatch --parsable --array=0-$((TASK_COUNT - 1))%"${ARRAY_THROTTLE}" \
      scripts/run_staged_support_family_local_k_array.sh
)

DYSTS_REEVAL_QUEUE_JOB_ID=""
if [[ "${QUEUE_DYSTS_LONG_REEVAL}" == "1" ]]; then
  DYSTS_REEVAL_QUEUE_JOB_ID=$(
    RESULTS_DIR="${DYSTS_REEVAL_DIR}" \
    INPUT_ROOT_SPECS_TSV="${ROOT_SPECS_TSV}" \
    SYSTEMS_CSV="${SYSTEMS_CSV}" \
    SEEDS_CSV="${SEEDS_CSV}" \
    DYSTS_CACHE_PROFILE="${DYSTS_REEVAL_CACHE_PROFILE}" \
    DYSTS_CACHE_SPLIT="${DYSTS_REEVAL_CACHE_SPLIT}" \
    DYSTS_DT_MULTIPLIER="${DYSTS_DT_MULTIPLIER}" \
    DYSTS_PERIODIC_REENCODE_PERIODS="${DYSTS_REEVAL_PERIODS}" \
    HORIZONS="${DYSTS_REEVAL_HORIZONS}" \
    OUTPUT_TAG="${DYSTS_REEVAL_OUTPUT_TAG}" \
    ARRAY_PARALLEL="${DYSTS_REEVAL_ARRAY_PARALLEL}" \
    EVAL_PACK_SIZE="${DYSTS_REEVAL_PACK_SIZE}" \
    EVAL_TIME_LIMIT="${DYSTS_REEVAL_TIME_LIMIT}" \
    EVAL_DEVICE="${DYSTS_REEVAL_DEVICE}" \
    BATCH_SIZE="${DYSTS_REEVAL_BATCH_SIZE}" \
    STAGED_SUPPORT_DEFINITION="${SUPPORT_DEFINITION}" \
    STAGED_FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD}" \
      sbatch --parsable -p long --dependency=afterany:"${ARRAY_JOB_ID}" \
        scripts/queue_dysts_long_horizon_eval.sh
  )
fi

cat > "${AUTOMATION_DIR}/staged_cstab_learned_intercept_k_dysts_dt30_queue.json" <<EOF
{
  "target_variant": "${TARGET_VARIANT}",
  "baseline_root_label": "${BASELINE_ROOT_LABEL}",
  "baseline_root": "${BASELINE_ROOT}",
  "results_dir": "${RESULTS_DIR}",
  "base_out": "${BASE_OUT}",
  "task_tsv": "${TASK_TSV}",
  "manifest_json": "${MANIFEST_JSON}",
  "root_specs_tsv": "${ROOT_SPECS_TSV}",
  "task_count": ${TASK_COUNT},
  "array_job_id": "${ARRAY_JOB_ID}",
  "dysts_reeval_queue_job_id": "${DYSTS_REEVAL_QUEUE_JOB_ID}",
  "systems_csv": "${SYSTEMS_CSV}",
  "seeds_csv": "${SEEDS_CSV}",
  "support_definition": "${SUPPORT_DEFINITION}",
  "family_jaccard_threshold": ${FAMILY_JACCARD_THRESHOLD},
  "routing_object": "${ROUTING_OBJECT}",
  "stable_base_object": "${STABLE_BASE_OBJECT}",
  "stable_base_family_jaccard": "${STABLE_BASE_FAMILY_JACCARD}",
  "stable_fit_trajectories": "${STABLE_FIT_TRAJECTORIES}",
  "stable_fit_trajectory_length": "${STABLE_FIT_TRAJECTORY_LENGTH}",
  "local_map_parameterization": "${LOCAL_MAP_PARAMETERIZATION}",
  "stage2_selection_metric": "${STAGE2_SELECTION_METRIC}",
  "stage2_selection_periods": "${STAGE2_SELECTION_PERIODS}",
  "stage2_selection_horizons": "${STAGE2_SELECTION_HORIZONS}",
  "eval_periodic_periods_override": "${EVAL_PERIODIC_PERIODS_OVERRIDE}",
  "queue_dysts_long_reeval": "${QUEUE_DYSTS_LONG_REEVAL}",
  "dysts_reeval_dir": "${DYSTS_REEVAL_DIR}",
  "dysts_reeval_output_tag": "${DYSTS_REEVAL_OUTPUT_TAG}",
  "dysts_reeval_horizons": "${DYSTS_REEVAL_HORIZONS}",
  "dysts_reeval_periods": "${DYSTS_REEVAL_PERIODS}"
}
EOF

{
  printf 'RESULTS_DIR=%q\n' "${RESULTS_DIR}"
  printf 'BASE_OUT=%q\n' "${BASE_OUT}"
  printf 'TASK_TSV=%q\n' "${TASK_TSV}"
  printf 'MANIFEST_JSON=%q\n' "${MANIFEST_JSON}"
  printf 'ROOT_SPECS_TSV=%q\n' "${ROOT_SPECS_TSV}"
  printf 'TASK_COUNT=%q\n' "${TASK_COUNT}"
  printf 'ARRAY_JOB_ID=%q\n' "${ARRAY_JOB_ID}"
  printf 'DYSTS_REEVAL_QUEUE_JOB_ID=%q\n' "${DYSTS_REEVAL_QUEUE_JOB_ID}"
  printf 'TARGET_VARIANT=%q\n' "${TARGET_VARIANT}"
  printf 'BASELINE_ROOT_LABEL=%q\n' "${BASELINE_ROOT_LABEL}"
  printf 'SYSTEMS_CSV=%q\n' "${SYSTEMS_CSV}"
  printf 'SEEDS_CSV=%q\n' "${SEEDS_CSV}"
  printf 'SUPPORT_DEFINITION=%q\n' "${SUPPORT_DEFINITION}"
  printf 'FAMILY_JACCARD_THRESHOLD=%q\n' "${FAMILY_JACCARD_THRESHOLD}"
  printf 'ROUTING_OBJECT=%q\n' "${ROUTING_OBJECT}"
  printf 'LOCAL_MAP_PARAMETERIZATION=%q\n' "${LOCAL_MAP_PARAMETERIZATION}"
} > "${QUEUE_LOG_DIR}/launch_record.env"

echo "Queued staged C_stab learned-intercept local-K LISTA Dysts dt-x30 experiment."
echo "Array job: ${ARRAY_JOB_ID}"
if [[ -n "${DYSTS_REEVAL_QUEUE_JOB_ID}" ]]; then
  echo "Dysts long-horizon evaluation queue launcher: ${DYSTS_REEVAL_QUEUE_JOB_ID}"
fi
echo "Results dir: ${RESULTS_DIR}"
echo "Base out: ${BASE_OUT}"
