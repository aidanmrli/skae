#!/bin/bash
#
# Queue staged source-target affine routed local-K LISTA experiments for the
# multibasin Table 1 roster. The default is the historical support-family
# route; callers can override ROUTING_OBJECT for C_stab and matched controls.
#   - first half of the 200k Table 1 budget trains encoder, decoder, and global K
#   - second half freezes encoder/decoder/global K and trains all F_abs K_c maps
#     as source-target affine charts initialized to match global K exactly
#   - support families are label-free, using F_abs absolute:0.001 and Jaccard 0.4
#   - comparison is collected against the existing dense global-K LISTA root
#
# Submit with:
#   sbatch scripts/queue_staged_support_family_local_k_table1.sh
#
# Optional env vars:
#   EXPERIMENT_TAG=staged_fabs_local_affine_k_lista_table1_20260512
#   ARRAY_THROTTLE=32
#   ARRAY_PARTITION=long
#   ARRAY_GRES=              # optional override, e.g. gpu:1 or gpu:a100:1
#   ARRAY_JOB_TIME=03:00:00
#   DOWNSTREAM_DEPENDENCY_TYPE=afterok
#   SUPPORT_DEFINITION=absolute:0.001
#   FAMILY_JACCARD_THRESHOLD=0.4
#   ROUTING_OBJECT=support_family
#   SUPPORT_FAMILY_FIT_SOURCE=stage1_buffer
#   SYSTEMS_CSV=
#   STAGE2_SELECTION_METRIC=
#   STAGE2_SELECTION_PERIODS=
#   STAGE2_SELECTION_HORIZONS=
#   STAGE2_SELECTION_BATCH_SIZE=
#   STAGE2_SELECTION_SEED_OFFSET=
#   BASELINE_ROUTE_SEED_OFFSET=314159
#   BASELINE_LATENT_CLUSTER_COUNT=0
#   BASELINE_KMEANS_N_INIT=10
#   LATENT_FATE_TAIL_WINDOW=16
#   LATENT_FATE_MAX_CLUSTERS=12
#   LATENT_FATE_MIN_SILHOUETTE=0.05
#   LATENT_FATE_PCA_COMPONENTS=16
#   EVAL_PERIODIC_PERIODS_OVERRIDE=
#   SKIP_COMPLETED=1
#   RESUME_FROM_LATEST=1
#   SAVE_LAST_CHECKPOINT=0
#   QUEUE_WIDE_PERIODIC_REEVAL=0
#   WIDE_REEVAL_HORIZONS_CSV=100,500,1000
#   WIDE_REEVAL_PERIODS_CSV=1,2,5,10,20,25,50,100

#SBATCH --job-name=queue_fabs_k
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-staged-fabs-local-k-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-staged-fabs-local-k-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run on a compute node."
  echo "Submit it with: sbatch scripts/queue_staged_support_family_local_k_table1.sh"
  exit 2
fi

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-20260512}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-staged_fabs_local_affine_k_lista_table1_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-transition_rich_basin_partition}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect_pass0}"
COMPARE_DIR="${COMPARE_DIR:-${RESULTS_DIR}/compare_vs_global_k}"
WIDE_REEVAL_DIR="${WIDE_REEVAL_DIR:-${RESULTS_DIR}/wide_periodic_reeval}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-${RESULTS_DIR}/queue_logs}"
AUTOMATION_DIR="${AUTOMATION_DIR:-${RESULTS_DIR}/automation}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-32}"
ARRAY_PARTITION="${ARRAY_PARTITION:-long}"
ARRAY_GRES="${ARRAY_GRES:-}"
ARRAY_JOB_TIME="${ARRAY_JOB_TIME:-03:00:00}"
DOWNSTREAM_DEPENDENCY_TYPE="${DOWNSTREAM_DEPENDENCY_TYPE:-afterok}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-650}"

SOURCE_TSV="${SOURCE_TSV:-results/transition_rich_lista_dense_p256_hardinit_table123_20260430/task_tables/transition_rich_lista_dense_p256_hardinit_table123.tsv}"
SOURCE_VARIANT="${SOURCE_VARIANT:-lista_dense_signsplit_p256_hardinit_basin_partition}"
TARGET_VARIANT="${TARGET_VARIANT:-lista_fabs_local_affine_k_staged_p256_hardinit_basin_partition}"
BASELINE_ROOT_LABEL="${BASELINE_ROOT_LABEL:-lista_dense_signsplit_p256_hardinit_basin_partition}"
BASELINE_ROOT="${BASELINE_ROOT:-/network/scratch/l/lia/skae/transition_rich_lista_dense_p256_hardinit_table123_20260430/transition_rich_basin_partition/lista_dense_signsplit_p256_hardinit_basin_partition}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
EXCLUDED_SYSTEMS_CSV="${EXCLUDED_SYSTEMS_CSV:-multiwell_strong_transition,claude:checkerboard_potential}"
EXPECTED_TASK_COUNT="${EXPECTED_TASK_COUNT:-225}"

SUPPORT_DEFINITION="${SUPPORT_DEFINITION:-absolute:0.001}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.4}"
SUPPORT_FIT_BATCHES="${SUPPORT_FIT_BATCHES:-16}"
MIN_FAMILY_TRANSITIONS="${MIN_FAMILY_TRANSITIONS:-1}"
SUPPORT_FAMILY_FIT_SOURCE="${SUPPORT_FAMILY_FIT_SOURCE:-stage1_buffer}"
ROUTING_OBJECT="${ROUTING_OBJECT:-support_family}"
STABLE_BASE_OBJECT="${STABLE_BASE_OBJECT:-family}"
STABLE_BASE_FAMILY_JACCARD="${STABLE_BASE_FAMILY_JACCARD:-0.8}"
STABLE_TAIL_WINDOW="${STABLE_TAIL_WINDOW:-32}"
STABLE_MIN_EDGE_COUNT="${STABLE_MIN_EDGE_COUNT:-2}"
STABLE_MIN_EDGE_PROBABILITY="${STABLE_MIN_EDGE_PROBABILITY:-0.02}"
STABLE_MAX_RECURRENT_OUT_PROBABILITY="${STABLE_MAX_RECURRENT_OUT_PROBABILITY:-0.05}"
STABLE_MIN_TAIL_COUNT="${STABLE_MIN_TAIL_COUNT:-8}"
STABLE_MIN_ABSORPTION_OBSERVATIONS="${STABLE_MIN_ABSORPTION_OBSERVATIONS:-8}"
STABLE_MIN_ABSORPTION_CONFIDENCE="${STABLE_MIN_ABSORPTION_CONFIDENCE:-0.80}"
STABLE_FIT_TRAJECTORIES="${STABLE_FIT_TRAJECTORIES:-256}"
STABLE_FIT_TRAJECTORY_LENGTH="${STABLE_FIT_TRAJECTORY_LENGTH:-192}"
STABLE_FIT_SEED_OFFSET="${STABLE_FIT_SEED_OFFSET:-271828}"
LOCAL_MAP_PARAMETERIZATION="${LOCAL_MAP_PARAMETERIZATION:-source_target_affine_global_init}"
LOCAL_LR="${LOCAL_LR:-}"
STAGE2_SELECTION_METRIC="${STAGE2_SELECTION_METRIC:-}"
STAGE2_SELECTION_PERIODS="${STAGE2_SELECTION_PERIODS:-}"
STAGE2_SELECTION_HORIZONS="${STAGE2_SELECTION_HORIZONS:-}"
STAGE2_SELECTION_BATCH_SIZE="${STAGE2_SELECTION_BATCH_SIZE:-}"
STAGE2_SELECTION_SEED_OFFSET="${STAGE2_SELECTION_SEED_OFFSET:-}"
BASELINE_ROUTE_SEED_OFFSET="${BASELINE_ROUTE_SEED_OFFSET:-314159}"
BASELINE_LATENT_CLUSTER_COUNT="${BASELINE_LATENT_CLUSTER_COUNT:-0}"
BASELINE_KMEANS_N_INIT="${BASELINE_KMEANS_N_INIT:-10}"
LATENT_FATE_TAIL_WINDOW="${LATENT_FATE_TAIL_WINDOW:-16}"
LATENT_FATE_MAX_CLUSTERS="${LATENT_FATE_MAX_CLUSTERS:-12}"
LATENT_FATE_MIN_SILHOUETTE="${LATENT_FATE_MIN_SILHOUETTE:-0.05}"
LATENT_FATE_PCA_COMPONENTS="${LATENT_FATE_PCA_COMPONENTS:-16}"
EVAL_PERIODIC_PERIODS_OVERRIDE="${EVAL_PERIODIC_PERIODS_OVERRIDE:-}"
NUM_STEPS_OVERRIDE="${NUM_STEPS_OVERRIDE:-}"
STAGE1_STEPS_OVERRIDE="${STAGE1_STEPS_OVERRIDE:-}"
EVAL_EVERY_OVERRIDE="${EVAL_EVERY_OVERRIDE:-}"
EVAL_NUM_STEPS_OVERRIDE="${EVAL_NUM_STEPS_OVERRIDE:-}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESUME_FROM_LATEST="${RESUME_FROM_LATEST:-1}"
SAVE_LAST_CHECKPOINT="${SAVE_LAST_CHECKPOINT:-0}"
QUEUE_WIDE_PERIODIC_REEVAL="${QUEUE_WIDE_PERIODIC_REEVAL:-0}"
WIDE_REEVAL_HORIZONS_CSV="${WIDE_REEVAL_HORIZONS_CSV:-100,500,1000}"
WIDE_REEVAL_PERIODS_CSV="${WIDE_REEVAL_PERIODS_CSV:-1,2,5,10,20,25,50,100}"
WIDE_REEVAL_BATCH_SIZE="${WIDE_REEVAL_BATCH_SIZE:-100}"

case "${DOWNSTREAM_DEPENDENCY_TYPE}" in
  afterok|afterany) ;;
  *)
    echo "DOWNSTREAM_DEPENDENCY_TYPE must be afterok or afterany, got '${DOWNSTREAM_DEPENDENCY_TYPE}'" >&2
    exit 2
    ;;
esac

mkdir -p \
  "${TASK_DIR}" \
  "${ROOT_SPEC_DIR}" \
  "${COLLECT_DIR}" \
  "${COMPARE_DIR}" \
  "${WIDE_REEVAL_DIR}" \
  "${QUEUE_LOG_DIR}" \
  "${AUTOMATION_DIR}"

TASK_TSV="${TASK_DIR}/staged_fabs_local_affine_k_lista_table1.tsv"
MANIFEST_JSON="${TASK_DIR}/staged_fabs_local_affine_k_lista_table1_manifest.json"
ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/staged_fabs_local_affine_k_lista_table1_roots.txt"

PHASE_LABEL="${PHASE_LABEL}" \
EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
SOURCE_VARIANT="${SOURCE_VARIANT}" \
TARGET_VARIANT="${TARGET_VARIANT}" \
SEEDS_CSV="${SEEDS_CSV}" \
SYSTEMS_CSV="${SYSTEMS_CSV}" \
EXCLUDED_SYSTEMS_CSV="${EXCLUDED_SYSTEMS_CSV}" \
SUPPORT_DEFINITION="${SUPPORT_DEFINITION}" \
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD}" \
SUPPORT_FIT_BATCHES="${SUPPORT_FIT_BATCHES}" \
MIN_FAMILY_TRANSITIONS="${MIN_FAMILY_TRANSITIONS}" \
SUPPORT_FAMILY_FIT_SOURCE="${SUPPORT_FAMILY_FIT_SOURCE}" \
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
BASELINE_ROUTE_SEED_OFFSET="${BASELINE_ROUTE_SEED_OFFSET}" \
BASELINE_LATENT_CLUSTER_COUNT="${BASELINE_LATENT_CLUSTER_COUNT}" \
BASELINE_KMEANS_N_INIT="${BASELINE_KMEANS_N_INIT}" \
LATENT_FATE_TAIL_WINDOW="${LATENT_FATE_TAIL_WINDOW}" \
LATENT_FATE_MAX_CLUSTERS="${LATENT_FATE_MAX_CLUSTERS}" \
LATENT_FATE_MIN_SILHOUETTE="${LATENT_FATE_MIN_SILHOUETTE}" \
LATENT_FATE_PCA_COMPONENTS="${LATENT_FATE_PCA_COMPONENTS}" \
LOCAL_MAP_PARAMETERIZATION="${LOCAL_MAP_PARAMETERIZATION}" \
LOCAL_LR="${LOCAL_LR}" \
STAGE2_SELECTION_METRIC="${STAGE2_SELECTION_METRIC}" \
STAGE2_SELECTION_PERIODS="${STAGE2_SELECTION_PERIODS}" \
STAGE2_SELECTION_HORIZONS="${STAGE2_SELECTION_HORIZONS}" \
STAGE2_SELECTION_BATCH_SIZE="${STAGE2_SELECTION_BATCH_SIZE}" \
STAGE2_SELECTION_SEED_OFFSET="${STAGE2_SELECTION_SEED_OFFSET}" \
EVAL_PERIODIC_PERIODS_OVERRIDE="${EVAL_PERIODIC_PERIODS_OVERRIDE}" \
NUM_STEPS_OVERRIDE="${NUM_STEPS_OVERRIDE}" \
STAGE1_STEPS_OVERRIDE="${STAGE1_STEPS_OVERRIDE}" \
EVAL_EVERY_OVERRIDE="${EVAL_EVERY_OVERRIDE}" \
EVAL_NUM_STEPS_OVERRIDE="${EVAL_NUM_STEPS_OVERRIDE}" \
EVAL_PROFILE="${EVAL_PROFILE}" \
SKIP_COMPLETED="${SKIP_COMPLETED}" \
RESUME_FROM_LATEST="${RESUME_FROM_LATEST}" \
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
base_out = Path(os.environ["BASE_OUT"])
requested_seeds = {seed.strip() for seed in os.environ["SEEDS_CSV"].split(",") if seed.strip()}
requested_systems = {
    item.strip()
    for item in os.environ.get("SYSTEMS_CSV", "").split(",")
    if item.strip()
}
excluded_systems = {
    item.strip()
    for item in os.environ["EXCLUDED_SYSTEMS_CSV"].split(",")
    if item.strip()
}
skip_completed = os.environ["SKIP_COMPLETED"] == "1"

def tagify(value):
    return str(value).replace("-", "m").replace(".", "p")

def completed_run_for(row):
    system_slug = (row.get("system_slug") or row.get("system_key") or "").replace(":", "_")
    raw_dt = row.get("env_dt") or "default"
    seed = int(float(row.get("seed", "0")))
    seed_dir = (
        base_out
        / phase_label
        / target_variant
        / system_slug
        / f"dt_{tagify(raw_dt)}"
        / f"seed_{seed}"
    )
    if not seed_dir.is_dir():
        return None
    candidates = [
        path
        for path in seed_dir.glob("20*")
        if path.is_dir() and (path / "evaluation_results_best.json").is_file()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (path.name, str(path)))[-1]

def optional_int(name):
    raw = os.environ.get(name, "").strip()
    return None if not raw else int(raw)

def optional_float(name):
    raw = os.environ.get(name, "").strip()
    return None if not raw else float(raw)

def optional_csv_ints(name):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    return [int(item.strip()) for item in raw.split(",") if item.strip()]

with source_path.open(newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fields = list(reader.fieldnames or [])
    source_rows = [
        dict(row)
        for row in reader
        if row.get("model_variant") == source_variant
        and str(row.get("seed", "")).strip() in requested_seeds
        and (not requested_systems or row.get("system_key") in requested_systems)
        and row.get("system_key") not in excluded_systems
    ]

if not source_rows:
    raise SystemExit(f"No rows found for source variant {source_variant} in {source_path}")

source_rows.sort(key=lambda row: (row.get("system_key", ""), int(row.get("seed", "0"))))
out_rows = []
skipped_completed_rows = []
for row in source_rows:
    out = dict(row)
    out["phase"] = phase_label
    out["model_variant"] = target_variant
    out["eval_profile"] = os.environ["EVAL_PROFILE"]
    completed_run = completed_run_for(out) if skip_completed else None
    if completed_run is not None:
        skipped_completed_rows.append(
            {
                "source_task_id": row.get("task_id"),
                "system_key": row.get("system_key"),
                "seed": row.get("seed"),
                "env_dt": row.get("env_dt"),
                "completed_run": str(completed_run),
            }
        )
        continue
    out["task_id"] = str(len(out_rows))
    out_rows.append(out)

num_steps_values = sorted({str(row.get("num_steps", "")).strip() for row in out_rows})
if num_steps_values != ["200000"]:
    raise SystemExit(f"Expected Table 1 num_steps=200000, found {num_steps_values}")

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(out_rows)

counts_by_system = Counter(row["system_key"] for row in out_rows)
first_row = out_rows[0] if out_rows else source_rows[0]

def row_int(name, default=0):
    raw = str(first_row.get(name, "")).strip()
    return default if not raw else int(float(raw))

def row_float(name, default=0.0):
    raw = str(first_row.get(name, "")).strip()
    return default if not raw else float(raw)

def row_str(name, default=""):
    raw = str(first_row.get(name, "")).strip()
    return raw if raw else default

settings = {
    "total_steps": row_int("num_steps", 200000),
    "stage1_joint_steps": row_int("num_steps", 200000) // 2,
    "stage2_local_steps": row_int("num_steps", 200000) - (row_int("num_steps", 200000) // 2),
    "batch_size": row_int("batch_size", 256),
    "target_size": row_int("target_size", 256),
    "sequence_length": row_int("sequence_length", 8),
    "sparsity_coeff": row_float("sparsity_coeff", 0.0),
    "k_structure": row_str("k_structure", "dense"),
    "lista_alpha": row_float("lista_alpha", 0.0),
    "lista_num_loops": row_int("lista_num_loops", 0),
    "lista_final_op": row_str("lista_final_op", ""),
    "support_definition": os.environ["SUPPORT_DEFINITION"],
    "family_jaccard_threshold": float(os.environ["FAMILY_JACCARD_THRESHOLD"]),
    "support_fit_batches": int(os.environ["SUPPORT_FIT_BATCHES"]),
    "min_family_transitions": int(os.environ["MIN_FAMILY_TRANSITIONS"]),
    "support_family_fit_source": os.environ["SUPPORT_FAMILY_FIT_SOURCE"],
    "routing_object": os.environ["ROUTING_OBJECT"],
    "stable_base_object": os.environ["STABLE_BASE_OBJECT"],
    "stable_base_family_jaccard": float(os.environ["STABLE_BASE_FAMILY_JACCARD"]),
    "stable_tail_window": int(os.environ["STABLE_TAIL_WINDOW"]),
    "stable_min_edge_count": int(os.environ["STABLE_MIN_EDGE_COUNT"]),
    "stable_min_edge_probability": float(os.environ["STABLE_MIN_EDGE_PROBABILITY"]),
    "stable_max_recurrent_out_probability": float(os.environ["STABLE_MAX_RECURRENT_OUT_PROBABILITY"]),
    "stable_min_tail_count": int(os.environ["STABLE_MIN_TAIL_COUNT"]),
    "stable_min_absorption_observations": int(os.environ["STABLE_MIN_ABSORPTION_OBSERVATIONS"]),
    "stable_min_absorption_confidence": float(os.environ["STABLE_MIN_ABSORPTION_CONFIDENCE"]),
    "stable_fit_trajectories": int(os.environ["STABLE_FIT_TRAJECTORIES"]),
    "stable_fit_trajectory_length": int(os.environ["STABLE_FIT_TRAJECTORY_LENGTH"]),
    "stable_fit_seed_offset": int(os.environ["STABLE_FIT_SEED_OFFSET"]),
    "baseline_route_seed_offset": int(os.environ["BASELINE_ROUTE_SEED_OFFSET"]),
    "baseline_latent_cluster_count": int(os.environ["BASELINE_LATENT_CLUSTER_COUNT"]),
    "baseline_kmeans_n_init": int(os.environ["BASELINE_KMEANS_N_INIT"]),
    "latent_fate_tail_window": int(os.environ["LATENT_FATE_TAIL_WINDOW"]),
    "latent_fate_max_clusters": int(os.environ["LATENT_FATE_MAX_CLUSTERS"]),
    "latent_fate_min_silhouette": float(os.environ["LATENT_FATE_MIN_SILHOUETTE"]),
    "latent_fate_pca_components": int(os.environ["LATENT_FATE_PCA_COMPONENTS"]),
    "local_map_parameterization": os.environ["LOCAL_MAP_PARAMETERIZATION"],
    "local_lr": optional_float("LOCAL_LR"),
    "stage2_selection_metric": os.environ["STAGE2_SELECTION_METRIC"],
    "stage2_selection_periods": optional_csv_ints("STAGE2_SELECTION_PERIODS"),
    "stage2_selection_horizons": optional_csv_ints("STAGE2_SELECTION_HORIZONS"),
    "stage2_selection_batch_size": optional_int("STAGE2_SELECTION_BATCH_SIZE"),
    "stage2_selection_seed_offset": optional_int("STAGE2_SELECTION_SEED_OFFSET"),
    "eval_periodic_periods_override": optional_csv_ints("EVAL_PERIODIC_PERIODS_OVERRIDE"),
    "num_steps_override": optional_int("NUM_STEPS_OVERRIDE"),
    "stage1_steps_override": optional_int("STAGE1_STEPS_OVERRIDE"),
    "eval_every_override": optional_int("EVAL_EVERY_OVERRIDE"),
    "eval_num_steps_override": optional_int("EVAL_NUM_STEPS_OVERRIDE"),
    "eval_profile": os.environ["EVAL_PROFILE"],
    "skip_completed": os.environ["SKIP_COMPLETED"],
    "resume_from_latest": os.environ["RESUME_FROM_LATEST"],
}
manifest = {
    "experiment": os.environ["EXPERIMENT_TAG"],
    "experiment_family": "staged_support_family_local_k_table1",
    "source_task_tsv": str(source_path),
    "source_variant": source_variant,
    "target_variant": target_variant,
    "task_tsv": str(out_path),
    "phase_label": phase_label,
    "excluded_systems": sorted(excluded_systems),
    "included_systems": sorted(requested_systems),
    "seeds": sorted(int(seed) for seed in requested_seeds),
    "settings": settings,
    "num_tasks": len(out_rows),
    "counts_by_system": dict(sorted(counts_by_system.items())),
    "skipped_completed_count": len(skipped_completed_rows),
    "skipped_completed_rows": skipped_completed_rows,
}
manifest_path.write_text(json.dumps(manifest, indent=2))
print(json.dumps({
    "task_tsv": str(out_path),
    "num_tasks": len(out_rows),
    "skipped_completed": len(skipped_completed_rows),
}, indent=2))
PY

TASK_COUNT=$(awk 'END { print NR > 0 ? NR - 1 : 0 }' "${TASK_TSV}")
NO_ARRAY=0
if (( TASK_COUNT <= 0 )); then
  if [[ "${SKIP_COMPLETED}" == "1" ]]; then
    echo "No unfinished tasks generated in ${TASK_TSV}; skipping GPU array submission."
    NO_ARRAY=1
  else
    echo "No tasks generated in ${TASK_TSV}"
    exit 1
  fi
fi
if [[ "${EXPECTED_TASK_COUNT}" != "0" && "${SKIP_COMPLETED}" != "1" ]] && (( TASK_COUNT != EXPECTED_TASK_COUNT )); then
  echo "Expected ${EXPECTED_TASK_COUNT} tasks but generated ${TASK_COUNT}"
  exit 1
fi

{
  printf '%s=%s/%s/%s\n' "${TARGET_VARIANT}" "${BASE_OUT}" "${PHASE_LABEL}" "${TARGET_VARIANT}"
  printf '%s=%s\n' "${BASELINE_ROOT_LABEL}" "${BASELINE_ROOT}"
} > "${ROOT_SPECS_FILE}"

echo "Generated ${TASK_COUNT} staged local-K LISTA tasks."
echo "Task TSV: ${TASK_TSV}"
echo "Manifest: ${MANIFEST_JSON}"
echo "Root specs: ${ROOT_SPECS_FILE}"

ARRAY_JOB_ID=""
if [[ "${NO_ARRAY}" == "0" ]]; then
  while true; do
    CURRENT_JOBS=$(squeue -u "${USER}" -h -r | wc -l)
    if (( CURRENT_JOBS <= MAX_EXISTING_JOBS_BEFORE_SUBMIT )); then
      break
    fi
    echo "Current expanded job count ${CURRENT_JOBS} exceeds ${MAX_EXISTING_JOBS_BEFORE_SUBMIT}; sleeping before array submit."
    sleep 60
  done

  ARRAY_SBATCH_ARGS=(
    --parsable
    --array=0-$((TASK_COUNT - 1))%"${ARRAY_THROTTLE}"
    --partition="${ARRAY_PARTITION}"
    --time="${ARRAY_JOB_TIME}"
  )
  if [[ -n "${ARRAY_GRES}" ]]; then
    ARRAY_SBATCH_ARGS+=(--gres="${ARRAY_GRES}")
  fi

  ARRAY_JOB_ID=$(
    TASK_TSV="${TASK_TSV}" \
    BASE_OUT="${BASE_OUT}" \
    SUPPORT_DEFINITION="${SUPPORT_DEFINITION}" \
    FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD}" \
    SUPPORT_FIT_BATCHES="${SUPPORT_FIT_BATCHES}" \
    MIN_FAMILY_TRANSITIONS="${MIN_FAMILY_TRANSITIONS}" \
    SUPPORT_FAMILY_FIT_SOURCE="${SUPPORT_FAMILY_FIT_SOURCE}" \
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
    BASELINE_ROUTE_SEED_OFFSET="${BASELINE_ROUTE_SEED_OFFSET}" \
    BASELINE_LATENT_CLUSTER_COUNT="${BASELINE_LATENT_CLUSTER_COUNT}" \
    BASELINE_KMEANS_N_INIT="${BASELINE_KMEANS_N_INIT}" \
    LATENT_FATE_TAIL_WINDOW="${LATENT_FATE_TAIL_WINDOW}" \
    LATENT_FATE_MAX_CLUSTERS="${LATENT_FATE_MAX_CLUSTERS}" \
    LATENT_FATE_MIN_SILHOUETTE="${LATENT_FATE_MIN_SILHOUETTE}" \
    LATENT_FATE_PCA_COMPONENTS="${LATENT_FATE_PCA_COMPONENTS}" \
    LOCAL_MAP_PARAMETERIZATION="${LOCAL_MAP_PARAMETERIZATION}" \
    LOCAL_LR="${LOCAL_LR}" \
    STAGE2_SELECTION_METRIC="${STAGE2_SELECTION_METRIC}" \
    STAGE2_SELECTION_PERIODS="${STAGE2_SELECTION_PERIODS}" \
    STAGE2_SELECTION_HORIZONS="${STAGE2_SELECTION_HORIZONS}" \
    STAGE2_SELECTION_BATCH_SIZE="${STAGE2_SELECTION_BATCH_SIZE}" \
    STAGE2_SELECTION_SEED_OFFSET="${STAGE2_SELECTION_SEED_OFFSET}" \
    EVAL_PERIODIC_PERIODS_OVERRIDE="${EVAL_PERIODIC_PERIODS_OVERRIDE}" \
    NUM_STEPS_OVERRIDE="${NUM_STEPS_OVERRIDE}" \
    STAGE1_STEPS_OVERRIDE="${STAGE1_STEPS_OVERRIDE}" \
    EVAL_EVERY_OVERRIDE="${EVAL_EVERY_OVERRIDE}" \
    EVAL_NUM_STEPS_OVERRIDE="${EVAL_NUM_STEPS_OVERRIDE}" \
    EVAL_PROFILE="${EVAL_PROFILE}" \
    SKIP_COMPLETED="${SKIP_COMPLETED}" \
    RESUME_FROM_LATEST="${RESUME_FROM_LATEST}" \
    SAVE_LAST_CHECKPOINT="${SAVE_LAST_CHECKPOINT}" \
    GPU_TELEMETRY="${GPU_TELEMETRY:-1}" \
    GPU_TELEMETRY_INTERVAL="${GPU_TELEMETRY_INTERVAL:-30}" \
      sbatch "${ARRAY_SBATCH_ARGS[@]}" \
        scripts/run_staged_support_family_local_k_array.sh
  )
fi

if [[ -n "${ARRAY_JOB_ID}" ]]; then
  COLLECT_JOB_ID=$(
    ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" \
    OUT_DIR="${COLLECT_DIR}" \
    HORIZONS_CSV="100,500,1000" \
    GOOD_THRESHOLD="50" \
      sbatch --parsable --dependency="${DOWNSTREAM_DEPENDENCY_TYPE}:${ARRAY_JOB_ID}" \
        scripts/collect_transition_rich_basin_partition.sh
  )
else
  COLLECT_JOB_ID=$(
    ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" \
    OUT_DIR="${COLLECT_DIR}" \
    HORIZONS_CSV="100,500,1000" \
    GOOD_THRESHOLD="50" \
      sbatch --parsable scripts/collect_transition_rich_basin_partition.sh
  )
fi

COMPARE_JOB_ID=$(
  ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" \
  OUT_DIR="${COMPARE_DIR}" \
  CANDIDATE_ROOTS_CSV="${TARGET_VARIANT}" \
  ANCHOR_ROOT="${BASELINE_ROOT_LABEL}" \
  HORIZON="1000" \
    sbatch --parsable --dependency=afterok:"${COLLECT_JOB_ID}" \
      scripts/compare_paper_benchmark.sh
)

WIDE_REEVAL_JOB_ID=""
if [[ "${QUEUE_WIDE_PERIODIC_REEVAL}" == "1" ]]; then
  if [[ -n "${ARRAY_JOB_ID}" ]]; then
    WIDE_REEVAL_JOB_ID=$(
      STAGED_ROOT="${BASE_OUT}/${PHASE_LABEL}/${TARGET_VARIANT}" \
      GLOBAL_ROOT="${BASELINE_ROOT}" \
      OUT_DIR="${WIDE_REEVAL_DIR}" \
      HORIZONS_CSV="${WIDE_REEVAL_HORIZONS_CSV}" \
      PERIODS_CSV="${WIDE_REEVAL_PERIODS_CSV}" \
      BATCH_SIZE="${WIDE_REEVAL_BATCH_SIZE}" \
      SUPPORT_DEFINITION="${SUPPORT_DEFINITION}" \
      FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD}" \
      FORCE="1" \
        sbatch --parsable --dependency="${DOWNSTREAM_DEPENDENCY_TYPE}:${ARRAY_JOB_ID}" \
          scripts/reevaluate_staged_vs_global_wide_periodic.sh
    )
  else
    WIDE_REEVAL_JOB_ID=$(
      STAGED_ROOT="${BASE_OUT}/${PHASE_LABEL}/${TARGET_VARIANT}" \
      GLOBAL_ROOT="${BASELINE_ROOT}" \
      OUT_DIR="${WIDE_REEVAL_DIR}" \
      HORIZONS_CSV="${WIDE_REEVAL_HORIZONS_CSV}" \
      PERIODS_CSV="${WIDE_REEVAL_PERIODS_CSV}" \
      BATCH_SIZE="${WIDE_REEVAL_BATCH_SIZE}" \
      SUPPORT_DEFINITION="${SUPPORT_DEFINITION}" \
      FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD}" \
      FORCE="1" \
        sbatch --parsable scripts/reevaluate_staged_vs_global_wide_periodic.sh
    )
  fi
fi

QUEUE_JSON_PATH="${AUTOMATION_DIR}/staged_fabs_local_affine_k_table1_queue.json" \
EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
TARGET_VARIANT="${TARGET_VARIANT}" \
BASELINE_ROOT_LABEL="${BASELINE_ROOT_LABEL}" \
BASELINE_ROOT="${BASELINE_ROOT}" \
RESULTS_DIR="${RESULTS_DIR}" \
BASE_OUT="${BASE_OUT}" \
TASK_TSV="${TASK_TSV}" \
MANIFEST_JSON="${MANIFEST_JSON}" \
ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" \
TASK_COUNT="${TASK_COUNT}" \
ARRAY_JOB_ID="${ARRAY_JOB_ID}" \
ARRAY_PARTITION="${ARRAY_PARTITION}" \
ARRAY_GRES="${ARRAY_GRES}" \
ARRAY_JOB_TIME="${ARRAY_JOB_TIME}" \
DOWNSTREAM_DEPENDENCY_TYPE="${DOWNSTREAM_DEPENDENCY_TYPE}" \
COLLECT_JOB_ID="${COLLECT_JOB_ID}" \
COMPARE_JOB_ID="${COMPARE_JOB_ID}" \
WIDE_REEVAL_JOB_ID="${WIDE_REEVAL_JOB_ID}" \
SUPPORT_DEFINITION="${SUPPORT_DEFINITION}" \
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD}" \
SUPPORT_FIT_BATCHES="${SUPPORT_FIT_BATCHES}" \
MIN_FAMILY_TRANSITIONS="${MIN_FAMILY_TRANSITIONS}" \
SUPPORT_FAMILY_FIT_SOURCE="${SUPPORT_FAMILY_FIT_SOURCE}" \
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
BASELINE_ROUTE_SEED_OFFSET="${BASELINE_ROUTE_SEED_OFFSET}" \
BASELINE_LATENT_CLUSTER_COUNT="${BASELINE_LATENT_CLUSTER_COUNT}" \
BASELINE_KMEANS_N_INIT="${BASELINE_KMEANS_N_INIT}" \
LATENT_FATE_TAIL_WINDOW="${LATENT_FATE_TAIL_WINDOW}" \
LATENT_FATE_MAX_CLUSTERS="${LATENT_FATE_MAX_CLUSTERS}" \
LATENT_FATE_MIN_SILHOUETTE="${LATENT_FATE_MIN_SILHOUETTE}" \
LATENT_FATE_PCA_COMPONENTS="${LATENT_FATE_PCA_COMPONENTS}" \
LOCAL_MAP_PARAMETERIZATION="${LOCAL_MAP_PARAMETERIZATION}" \
LOCAL_LR="${LOCAL_LR}" \
STAGE2_SELECTION_METRIC="${STAGE2_SELECTION_METRIC}" \
STAGE2_SELECTION_PERIODS="${STAGE2_SELECTION_PERIODS}" \
STAGE2_SELECTION_HORIZONS="${STAGE2_SELECTION_HORIZONS}" \
STAGE2_SELECTION_BATCH_SIZE="${STAGE2_SELECTION_BATCH_SIZE}" \
STAGE2_SELECTION_SEED_OFFSET="${STAGE2_SELECTION_SEED_OFFSET}" \
EVAL_PERIODIC_PERIODS_OVERRIDE="${EVAL_PERIODIC_PERIODS_OVERRIDE}" \
NUM_STEPS_OVERRIDE="${NUM_STEPS_OVERRIDE}" \
STAGE1_STEPS_OVERRIDE="${STAGE1_STEPS_OVERRIDE}" \
EVAL_EVERY_OVERRIDE="${EVAL_EVERY_OVERRIDE}" \
EVAL_NUM_STEPS_OVERRIDE="${EVAL_NUM_STEPS_OVERRIDE}" \
EVAL_PROFILE="${EVAL_PROFILE}" \
SKIP_COMPLETED="${SKIP_COMPLETED}" \
RESUME_FROM_LATEST="${RESUME_FROM_LATEST}" \
SAVE_LAST_CHECKPOINT="${SAVE_LAST_CHECKPOINT}" \
QUEUE_WIDE_PERIODIC_REEVAL="${QUEUE_WIDE_PERIODIC_REEVAL}" \
WIDE_REEVAL_DIR="${WIDE_REEVAL_DIR}" \
WIDE_REEVAL_HORIZONS_CSV="${WIDE_REEVAL_HORIZONS_CSV}" \
WIDE_REEVAL_PERIODS_CSV="${WIDE_REEVAL_PERIODS_CSV}" \
WIDE_REEVAL_BATCH_SIZE="${WIDE_REEVAL_BATCH_SIZE}" \
  uv run python - <<'PY'
import json
import os
from pathlib import Path

def get(name):
    return os.environ[name]

def maybe_int(name):
    raw = os.environ.get(name, "").strip()
    return None if not raw else int(raw)

def maybe_float(name):
    raw = os.environ.get(name, "").strip()
    return None if not raw else float(raw)

payload = {
    "experiment_tag": get("EXPERIMENT_TAG"),
    "target_variant": get("TARGET_VARIANT"),
    "baseline_root_label": get("BASELINE_ROOT_LABEL"),
    "baseline_root": get("BASELINE_ROOT"),
    "results_dir": get("RESULTS_DIR"),
    "base_out": get("BASE_OUT"),
    "task_tsv": get("TASK_TSV"),
    "manifest_json": get("MANIFEST_JSON"),
    "root_specs_file": get("ROOT_SPECS_FILE"),
    "task_count": int(get("TASK_COUNT")),
    "array_job_id": get("ARRAY_JOB_ID"),
    "array_partition": get("ARRAY_PARTITION"),
    "array_gres": get("ARRAY_GRES"),
    "array_job_time": get("ARRAY_JOB_TIME"),
    "downstream_dependency_type": get("DOWNSTREAM_DEPENDENCY_TYPE"),
    "collect_job_id": get("COLLECT_JOB_ID"),
    "compare_job_id": get("COMPARE_JOB_ID"),
    "wide_periodic_reeval_job_id": get("WIDE_REEVAL_JOB_ID"),
    "support_definition": get("SUPPORT_DEFINITION"),
    "family_jaccard_threshold": float(get("FAMILY_JACCARD_THRESHOLD")),
    "support_fit_batches": int(get("SUPPORT_FIT_BATCHES")),
    "min_family_transitions": int(get("MIN_FAMILY_TRANSITIONS")),
    "support_family_fit_source": get("SUPPORT_FAMILY_FIT_SOURCE"),
    "routing_object": get("ROUTING_OBJECT"),
    "stable_base_object": get("STABLE_BASE_OBJECT"),
    "stable_base_family_jaccard": float(get("STABLE_BASE_FAMILY_JACCARD")),
    "stable_tail_window": int(get("STABLE_TAIL_WINDOW")),
    "stable_min_edge_count": int(get("STABLE_MIN_EDGE_COUNT")),
    "stable_min_edge_probability": float(get("STABLE_MIN_EDGE_PROBABILITY")),
    "stable_max_recurrent_out_probability": float(get("STABLE_MAX_RECURRENT_OUT_PROBABILITY")),
    "stable_min_tail_count": int(get("STABLE_MIN_TAIL_COUNT")),
    "stable_min_absorption_observations": int(get("STABLE_MIN_ABSORPTION_OBSERVATIONS")),
    "stable_min_absorption_confidence": float(get("STABLE_MIN_ABSORPTION_CONFIDENCE")),
    "stable_fit_trajectories": int(get("STABLE_FIT_TRAJECTORIES")),
    "stable_fit_trajectory_length": int(get("STABLE_FIT_TRAJECTORY_LENGTH")),
    "stable_fit_seed_offset": int(get("STABLE_FIT_SEED_OFFSET")),
    "baseline_route_seed_offset": int(get("BASELINE_ROUTE_SEED_OFFSET")),
    "baseline_latent_cluster_count": int(get("BASELINE_LATENT_CLUSTER_COUNT")),
    "baseline_kmeans_n_init": int(get("BASELINE_KMEANS_N_INIT")),
    "latent_fate_tail_window": int(get("LATENT_FATE_TAIL_WINDOW")),
    "latent_fate_max_clusters": int(get("LATENT_FATE_MAX_CLUSTERS")),
    "latent_fate_min_silhouette": float(get("LATENT_FATE_MIN_SILHOUETTE")),
    "latent_fate_pca_components": int(get("LATENT_FATE_PCA_COMPONENTS")),
    "local_map_parameterization": get("LOCAL_MAP_PARAMETERIZATION"),
    "local_lr": maybe_float("LOCAL_LR"),
    "stage2_selection_metric": get("STAGE2_SELECTION_METRIC"),
    "stage2_selection_periods": get("STAGE2_SELECTION_PERIODS"),
    "stage2_selection_horizons": get("STAGE2_SELECTION_HORIZONS"),
    "stage2_selection_batch_size": maybe_int("STAGE2_SELECTION_BATCH_SIZE"),
    "stage2_selection_seed_offset": maybe_int("STAGE2_SELECTION_SEED_OFFSET"),
    "eval_periodic_periods_override": get("EVAL_PERIODIC_PERIODS_OVERRIDE"),
    "num_steps_override": maybe_int("NUM_STEPS_OVERRIDE"),
    "stage1_steps_override": maybe_int("STAGE1_STEPS_OVERRIDE"),
    "eval_every_override": maybe_int("EVAL_EVERY_OVERRIDE"),
    "eval_num_steps_override": maybe_int("EVAL_NUM_STEPS_OVERRIDE"),
    "eval_profile": get("EVAL_PROFILE"),
    "skip_completed": get("SKIP_COMPLETED"),
    "resume_from_latest": get("RESUME_FROM_LATEST"),
    "save_last_checkpoint": get("SAVE_LAST_CHECKPOINT"),
    "queue_wide_periodic_reeval": get("QUEUE_WIDE_PERIODIC_REEVAL"),
    "wide_reeval_dir": get("WIDE_REEVAL_DIR"),
    "wide_reeval_horizons_csv": get("WIDE_REEVAL_HORIZONS_CSV"),
    "wide_reeval_periods_csv": get("WIDE_REEVAL_PERIODS_CSV"),
    "wide_reeval_batch_size": int(get("WIDE_REEVAL_BATCH_SIZE")),
}
Path(get("QUEUE_JSON_PATH")).write_text(json.dumps(payload, indent=2) + "\n")
PY

{
  printf 'RESULTS_DIR=%q\n' "${RESULTS_DIR}"
  printf 'BASE_OUT=%q\n' "${BASE_OUT}"
  printf 'TASK_TSV=%q\n' "${TASK_TSV}"
  printf 'MANIFEST_JSON=%q\n' "${MANIFEST_JSON}"
  printf 'ROOT_SPECS_FILE=%q\n' "${ROOT_SPECS_FILE}"
  printf 'TASK_COUNT=%q\n' "${TASK_COUNT}"
  printf 'ARRAY_JOB_ID=%q\n' "${ARRAY_JOB_ID}"
  printf 'COLLECT_JOB_ID=%q\n' "${COLLECT_JOB_ID}"
  printf 'COMPARE_JOB_ID=%q\n' "${COMPARE_JOB_ID}"
  printf 'WIDE_REEVAL_JOB_ID=%q\n' "${WIDE_REEVAL_JOB_ID}"
  printf 'TARGET_VARIANT=%q\n' "${TARGET_VARIANT}"
  printf 'BASELINE_ROOT_LABEL=%q\n' "${BASELINE_ROOT_LABEL}"
  printf 'SUPPORT_DEFINITION=%q\n' "${SUPPORT_DEFINITION}"
  printf 'FAMILY_JACCARD_THRESHOLD=%q\n' "${FAMILY_JACCARD_THRESHOLD}"
  printf 'SUPPORT_FAMILY_FIT_SOURCE=%q\n' "${SUPPORT_FAMILY_FIT_SOURCE}"
  printf 'ROUTING_OBJECT=%q\n' "${ROUTING_OBJECT}"
  printf 'STABLE_BASE_OBJECT=%q\n' "${STABLE_BASE_OBJECT}"
  printf 'STABLE_BASE_FAMILY_JACCARD=%q\n' "${STABLE_BASE_FAMILY_JACCARD}"
  printf 'STABLE_TAIL_WINDOW=%q\n' "${STABLE_TAIL_WINDOW}"
  printf 'STABLE_MIN_EDGE_COUNT=%q\n' "${STABLE_MIN_EDGE_COUNT}"
  printf 'STABLE_MIN_EDGE_PROBABILITY=%q\n' "${STABLE_MIN_EDGE_PROBABILITY}"
  printf 'STABLE_MAX_RECURRENT_OUT_PROBABILITY=%q\n' "${STABLE_MAX_RECURRENT_OUT_PROBABILITY}"
  printf 'STABLE_MIN_TAIL_COUNT=%q\n' "${STABLE_MIN_TAIL_COUNT}"
  printf 'STABLE_MIN_ABSORPTION_OBSERVATIONS=%q\n' "${STABLE_MIN_ABSORPTION_OBSERVATIONS}"
  printf 'STABLE_MIN_ABSORPTION_CONFIDENCE=%q\n' "${STABLE_MIN_ABSORPTION_CONFIDENCE}"
  printf 'STABLE_FIT_TRAJECTORIES=%q\n' "${STABLE_FIT_TRAJECTORIES}"
  printf 'STABLE_FIT_TRAJECTORY_LENGTH=%q\n' "${STABLE_FIT_TRAJECTORY_LENGTH}"
  printf 'STABLE_FIT_SEED_OFFSET=%q\n' "${STABLE_FIT_SEED_OFFSET}"
  printf 'BASELINE_ROUTE_SEED_OFFSET=%q\n' "${BASELINE_ROUTE_SEED_OFFSET}"
  printf 'BASELINE_LATENT_CLUSTER_COUNT=%q\n' "${BASELINE_LATENT_CLUSTER_COUNT}"
  printf 'BASELINE_KMEANS_N_INIT=%q\n' "${BASELINE_KMEANS_N_INIT}"
  printf 'LATENT_FATE_TAIL_WINDOW=%q\n' "${LATENT_FATE_TAIL_WINDOW}"
  printf 'LATENT_FATE_MAX_CLUSTERS=%q\n' "${LATENT_FATE_MAX_CLUSTERS}"
  printf 'LATENT_FATE_MIN_SILHOUETTE=%q\n' "${LATENT_FATE_MIN_SILHOUETTE}"
  printf 'LATENT_FATE_PCA_COMPONENTS=%q\n' "${LATENT_FATE_PCA_COMPONENTS}"
  printf 'LOCAL_MAP_PARAMETERIZATION=%q\n' "${LOCAL_MAP_PARAMETERIZATION}"
  printf 'STAGE2_SELECTION_METRIC=%q\n' "${STAGE2_SELECTION_METRIC}"
  printf 'STAGE2_SELECTION_PERIODS=%q\n' "${STAGE2_SELECTION_PERIODS}"
  printf 'STAGE2_SELECTION_HORIZONS=%q\n' "${STAGE2_SELECTION_HORIZONS}"
  printf 'STAGE2_SELECTION_BATCH_SIZE=%q\n' "${STAGE2_SELECTION_BATCH_SIZE}"
  printf 'STAGE2_SELECTION_SEED_OFFSET=%q\n' "${STAGE2_SELECTION_SEED_OFFSET}"
  printf 'EVAL_PERIODIC_PERIODS_OVERRIDE=%q\n' "${EVAL_PERIODIC_PERIODS_OVERRIDE}"
  printf 'NUM_STEPS_OVERRIDE=%q\n' "${NUM_STEPS_OVERRIDE}"
  printf 'STAGE1_STEPS_OVERRIDE=%q\n' "${STAGE1_STEPS_OVERRIDE}"
  printf 'EVAL_EVERY_OVERRIDE=%q\n' "${EVAL_EVERY_OVERRIDE}"
  printf 'EVAL_NUM_STEPS_OVERRIDE=%q\n' "${EVAL_NUM_STEPS_OVERRIDE}"
  printf 'EVAL_PROFILE=%q\n' "${EVAL_PROFILE}"
  printf 'SKIP_COMPLETED=%q\n' "${SKIP_COMPLETED}"
  printf 'RESUME_FROM_LATEST=%q\n' "${RESUME_FROM_LATEST}"
  printf 'SAVE_LAST_CHECKPOINT=%q\n' "${SAVE_LAST_CHECKPOINT}"
  printf 'ARRAY_THROTTLE=%q\n' "${ARRAY_THROTTLE}"
  printf 'ARRAY_PARTITION=%q\n' "${ARRAY_PARTITION}"
  printf 'ARRAY_GRES=%q\n' "${ARRAY_GRES}"
  printf 'ARRAY_JOB_TIME=%q\n' "${ARRAY_JOB_TIME}"
  printf 'DOWNSTREAM_DEPENDENCY_TYPE=%q\n' "${DOWNSTREAM_DEPENDENCY_TYPE}"
  printf 'MAX_EXISTING_JOBS_BEFORE_SUBMIT=%q\n' "${MAX_EXISTING_JOBS_BEFORE_SUBMIT}"
} > "${QUEUE_LOG_DIR}/launch_record.env"

echo "Queued staged source-target affine routed local-K LISTA Table 1 experiment."
echo "Array job: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Compare job: ${COMPARE_JOB_ID}"
if [[ -n "${WIDE_REEVAL_JOB_ID}" ]]; then
  echo "Wide periodic re-eval job: ${WIDE_REEVAL_JOB_ID}"
fi
echo "Results dir: ${RESULTS_DIR}"
echo "Base out: ${BASE_OUT}"
