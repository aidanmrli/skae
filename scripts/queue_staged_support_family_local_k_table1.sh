#!/bin/bash
#
# Queue the paper's staged F_abs-routed local affine LISTA experiment.
# The scientific protocol is fixed in code; environment variables below only
# select execution scope, storage, and artifact retention.
#
# Submit with:
#   sbatch scripts/queue_staged_support_family_local_k_table1.sh

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
  echo "This launcher must run under SLURM." >&2
  echo "Submit it with: sbatch scripts/queue_staged_support_family_local_k_table1.sh" >&2
  exit 2
fi
source .venv/bin/activate

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
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

SOURCE_VARIANT="lista_dense_signsplit_p256_hardinit_basin_partition"
TARGET_VARIANT="lista_fabs_local_affine_k_staged_p256_hardinit_basin_partition"
BASELINE_ROOT_LABEL="${BASELINE_ROOT_LABEL:-${SOURCE_VARIANT}}"
BASELINE_ROOT="${BASELINE_ROOT:-/network/scratch/l/lia/skae/transition_rich_lista_dense_p256_hardinit_table123_20260430/transition_rich_basin_partition/${SOURCE_VARIANT}}"

SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
EXPECTED_TASK_COUNT="${EXPECTED_TASK_COUNT:-225}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-32}"
ARRAY_JOB_TIME="${ARRAY_JOB_TIME:-03:00:00}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-650}"
DOWNSTREAM_DEPENDENCY_TYPE="${DOWNSTREAM_DEPENDENCY_TYPE:-afterok}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESUME_FROM_LATEST="${RESUME_FROM_LATEST:-1}"
SAVE_LAST_CHECKPOINT="${SAVE_LAST_CHECKPOINT:-1}"
QUEUE_WIDE_PERIODIC_REEVAL="${QUEUE_WIDE_PERIODIC_REEVAL:-1}"

case "${DOWNSTREAM_DEPENDENCY_TYPE}" in
  afterok|afterany) ;;
  *)
    echo "DOWNSTREAM_DEPENDENCY_TYPE must be afterok or afterany." >&2
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

BASE_TASK_TSV="${TASK_DIR}/.global_lista_paper_recipe.tsv"
BASE_MANIFEST_JSON="${TASK_DIR}/.global_lista_paper_recipe_manifest.json"
TASK_TSV="${TASK_DIR}/staged_fabs_local_affine_k_lista_table1.tsv"
MANIFEST_JSON="${TASK_DIR}/staged_fabs_local_affine_k_lista_table1_manifest.json"
ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/staged_fabs_local_affine_k_lista_table1_roots.txt"

BUILD_ARGS=(
  --paper_protocol
  --phase_label "${PHASE_LABEL}"
  --output_tsv "${BASE_TASK_TSV}"
  --output_manifest_json "${BASE_MANIFEST_JSON}"
  --model_variants_csv "${SOURCE_VARIANT}"
  --seeds_csv "${SEEDS_CSV}"
  --eval_profile full
)
if [[ -n "${SYSTEMS_CSV}" ]]; then
  BUILD_ARGS+=(--systems_csv "${SYSTEMS_CSV}")
fi
uv run python tools/build_transition_rich_basin_partition_tasks.py "${BUILD_ARGS[@]}"

PHASE_LABEL="${PHASE_LABEL}" \
TARGET_VARIANT="${TARGET_VARIANT}" \
BASE_OUT="${BASE_OUT}" \
SKIP_COMPLETED="${SKIP_COMPLETED}" \
  uv run python - \
    "${BASE_TASK_TSV}" \
    "${BASE_MANIFEST_JSON}" \
    "${TASK_TSV}" \
    "${MANIFEST_JSON}" <<'PY'
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

base_tsv, base_manifest, output_tsv, output_manifest = map(Path, sys.argv[1:])
target_variant = os.environ["TARGET_VARIANT"]
phase = os.environ["PHASE_LABEL"]
base_out = Path(os.environ["BASE_OUT"])
skip_completed = os.environ["SKIP_COMPLETED"] == "1"

def tagify(value):
    return str(value).replace("-", "m").replace(".", "p")

def completed_run(row):
    seed_dir = (
        base_out
        / phase
        / target_variant
        / row["system_slug"]
        / f"dt_{tagify(row['env_dt'])}"
        / f"seed_{int(row['seed'])}"
    )
    candidates = [
        path
        for path in seed_dir.glob("20*")
        if path.is_dir() and (path / "evaluation_results_best.json").is_file()
    ]
    return sorted(candidates)[-1] if candidates else None

with base_tsv.open(newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fields = list(reader.fieldnames or [])
    source_rows = [dict(row) for row in reader]

rows = []
skipped = []
for source in source_rows:
    row = dict(source)
    row["phase"] = phase
    row["model_variant"] = target_variant
    prior = completed_run(row) if skip_completed else None
    if prior is not None:
        skipped.append(
            {
                "system_key": row["system_key"],
                "seed": int(row["seed"]),
                "completed_run": str(prior),
            }
        )
        continue
    if int(row["num_steps"]) != 200_000:
        raise SystemExit(f"Expected 200000 steps, got {row['num_steps']}")
    row["task_id"] = str(len(rows))
    rows.append(row)

with output_tsv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

manifest = json.loads(base_manifest.read_text())
manifest.update(
    {
        "experiment_family": "staged_fabs_route_source_v3",
        "source_variant": "lista_dense_signsplit_p256_hardinit_basin_partition",
        "target_variant": target_variant,
        "task_tsv": str(output_tsv),
        "task_count": len(rows),
        "counts_by_system": dict(Counter(row["system_key"] for row in rows)),
        "skipped_completed_count": len(skipped),
        "skipped_completed_rows": skipped,
        "staged_protocol": {
            "route_schema_version": 3,
            "total_steps": 200_000,
            "stage1_joint_steps": 100_000,
            "stage2_local_steps": 100_000,
            "support_definition": "absolute:0.001",
            "family_jaccard_threshold": 0.40,
            "fit_source": "training_distribution_trajectories",
            "fit_construction": "two_bitwise_identical_copies_of_one_256_row_batch",
            "fit_configured_rows": 512,
            "fit_unique_trajectories": 256,
            "fit_duplication_factor": 2,
            "fit_transitions": 192,
            "fit_states": 193,
            "fit_supports_considered": 98_816,
            "fit_source_transitions": 98_304,
            "fit_unique_source_transitions": 49_152,
            "fit_seed_offset": 271_828,
            "family_clustering": "all_193_states_then_fit_on_first_192_sources",
            "family_representative": "modal_source_support",
            "min_family_transitions": 1,
            "routing_cadence": "every_latent_transition_step",
            "reencoding_role": "periodic_decode_encode_refreshes_latent_before_next_route",
            "local_map": "source_target_affine_learned_intercept",
            "checkpoint_selection": {
                "candidate_count": 200,
                "first_regular_step": 100_500,
                "last_regular_step": 199_500,
                "final_step": 199_999,
                "batch_size": 32,
                "seed_offset": 12_345,
                "horizons": [100, 500, 1000],
                "periods": [1, 2, 5, 10, 20, 25, 50, 100],
                "metric": "finite_prefix_state_summed_squared_error",
                "improvement": "strict_less_than"
            },
            "final_evaluation": {
                "batch_size": 100,
                "seed_offset": 12_345,
                "selector_overlap_count": 32,
                "selector_overlap_fraction": 0.32
            },
        },
    }
)
output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
PY
rm -f "${BASE_TASK_TSV}" "${BASE_MANIFEST_JSON}"

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT < 0 )); then
  echo "Malformed task table: ${TASK_TSV}" >&2
  exit 1
fi
if [[ "${SKIP_COMPLETED}" != "1" && "${EXPECTED_TASK_COUNT}" != "0" ]] \
  && (( TASK_COUNT != EXPECTED_TASK_COUNT )); then
  echo "Expected ${EXPECTED_TASK_COUNT} tasks, generated ${TASK_COUNT}." >&2
  exit 1
fi

{
  printf '%s=%s/%s/%s\n' "${TARGET_VARIANT}" "${BASE_OUT}" "${PHASE_LABEL}" "${TARGET_VARIANT}"
  printf '%s=%s\n' "${BASELINE_ROOT_LABEL}" "${BASELINE_ROOT}"
} > "${ROOT_SPECS_FILE}"

echo "Generated ${TASK_COUNT} unfinished staged F_abs tasks."
echo "Task table: ${TASK_TSV}"
echo "Manifest: ${MANIFEST_JSON}"

ARRAY_JOB_ID=""
if (( TASK_COUNT > 0 )); then
  while true; do
    CURRENT_JOBS=$(squeue -u "${USER}" -h -r | wc -l)
    if (( CURRENT_JOBS <= MAX_EXISTING_JOBS_BEFORE_SUBMIT )); then
      break
    fi
    echo "Expanded job count ${CURRENT_JOBS} exceeds ${MAX_EXISTING_JOBS_BEFORE_SUBMIT}; waiting."
    sleep 60
  done
  ARRAY_JOB_ID="$(
    TASK_TSV="${TASK_TSV}" \
    BASE_OUT="${BASE_OUT}" \
    SKIP_COMPLETED="${SKIP_COMPLETED}" \
    RESUME_FROM_LATEST="${RESUME_FROM_LATEST}" \
    SAVE_LAST_CHECKPOINT="${SAVE_LAST_CHECKPOINT}" \
    GPU_TELEMETRY="${GPU_TELEMETRY:-1}" \
    GPU_TELEMETRY_INTERVAL="${GPU_TELEMETRY_INTERVAL:-30}" \
      sbatch --parsable \
        --array=0-$((TASK_COUNT - 1))%"${ARRAY_THROTTLE}" \
        --partition=long \
        --time="${ARRAY_JOB_TIME}" \
        scripts/run_staged_support_family_local_k_array.sh
  )"
fi

COLLECT_DEPENDENCY=()
if [[ -n "${ARRAY_JOB_ID}" ]]; then
  COLLECT_DEPENDENCY=(--dependency="${DOWNSTREAM_DEPENDENCY_TYPE}:${ARRAY_JOB_ID}")
fi
COLLECT_JOB_ID="$(
  ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" \
  OUT_DIR="${COLLECT_DIR}" \
  HORIZONS_CSV="100,500,1000" \
  GOOD_THRESHOLD="50" \
    sbatch --parsable "${COLLECT_DEPENDENCY[@]}" \
      scripts/collect_transition_rich_basin_partition.sh
)"

COMPARE_JOB_ID="$(
  ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" \
  OUT_DIR="${COMPARE_DIR}" \
  CANDIDATE_ROOTS_CSV="${TARGET_VARIANT}" \
  ANCHOR_ROOT="${BASELINE_ROOT_LABEL}" \
  HORIZON="1000" \
    sbatch --parsable --dependency=afterok:"${COLLECT_JOB_ID}" \
      scripts/compare_paper_benchmark.sh
)"

WIDE_REEVAL_JOB_ID=""
if [[ "${QUEUE_WIDE_PERIODIC_REEVAL}" == "1" ]]; then
  WIDE_DEPENDENCY=()
  if [[ -n "${ARRAY_JOB_ID}" ]]; then
    WIDE_DEPENDENCY=(--dependency="${DOWNSTREAM_DEPENDENCY_TYPE}:${ARRAY_JOB_ID}")
  fi
  WIDE_REEVAL_JOB_ID="$(
    STAGED_ROOT="${BASE_OUT}/${PHASE_LABEL}/${TARGET_VARIANT}" \
    GLOBAL_ROOT="${BASELINE_ROOT}" \
    OUT_DIR="${WIDE_REEVAL_DIR}" \
    HORIZONS_CSV="100,500,1000" \
    PERIODS_CSV="1,2,5,10,20,25,50,100" \
    BATCH_SIZE="100" \
    SUPPORT_DEFINITION="absolute:0.001" \
    FAMILY_JACCARD_THRESHOLD="0.4" \
    FORCE="1" \
      sbatch --parsable "${WIDE_DEPENDENCY[@]}" \
        scripts/reevaluate_staged_vs_global_wide_periodic.sh
  )"
fi

QUEUE_JSON_PATH="${AUTOMATION_DIR}/staged_fabs_local_affine_k_table1_queue.json" \
EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
TARGET_VARIANT="${TARGET_VARIANT}" \
BASE_OUT="${BASE_OUT}" \
TASK_TSV="${TASK_TSV}" \
MANIFEST_JSON="${MANIFEST_JSON}" \
TASK_COUNT="${TASK_COUNT}" \
ARRAY_JOB_ID="${ARRAY_JOB_ID}" \
COLLECT_JOB_ID="${COLLECT_JOB_ID}" \
COMPARE_JOB_ID="${COMPARE_JOB_ID}" \
WIDE_REEVAL_JOB_ID="${WIDE_REEVAL_JOB_ID}" \
  uv run python - <<'PY'
import json
import os
from pathlib import Path

keys = (
    "EXPERIMENT_TAG",
    "TARGET_VARIANT",
    "BASE_OUT",
    "TASK_TSV",
    "MANIFEST_JSON",
    "ARRAY_JOB_ID",
    "COLLECT_JOB_ID",
    "COMPARE_JOB_ID",
    "WIDE_REEVAL_JOB_ID",
)
payload = {key.lower(): os.environ[key] for key in keys}
payload["task_count"] = int(os.environ["TASK_COUNT"])
payload["protocol"] = "staged_fabs_route_source_v3"
Path(os.environ["QUEUE_JSON_PATH"]).write_text(json.dumps(payload, indent=2) + "\n")
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
  printf 'PROTOCOL=%q\n' 'staged_fabs_route_source_v3'
} > "${QUEUE_LOG_DIR}/launch_record.env"

echo "Queued staged F_abs local affine LISTA experiment."
echo "Array job: ${ARRAY_JOB_ID:-none (all tasks complete)}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Compare job: ${COMPARE_JOB_ID}"
echo "Wide periodic re-evaluation job: ${WIDE_REEVAL_JOB_ID:-disabled}"
