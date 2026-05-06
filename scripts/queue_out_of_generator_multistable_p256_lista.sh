#!/bin/bash
#
# Queue supplemental out-of-generator multistable systems with the p256 dense-K LISTA recipe.
#
# Submit with:
#   sbatch scripts/queue_out_of_generator_multistable_p256_lista.sh
#
# Optional env vars:
#   EXPERIMENT_TAG=out_of_generator_multistable_p256_lista_20260506
#   SEEDS_CSV=0,1,2
#   ARRAY_THROTTLE=16
#   NUM_STEPS=200000
#
#SBATCH --job-name=queue_ood_p256
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-ood-p256-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-ood-p256-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run on a compute node."
  echo "Submit it with: sbatch scripts/queue_out_of_generator_multistable_p256_lista.sh"
  exit 2
fi

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-20260506}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-out_of_generator_multistable_p256_lista_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-out_of_generator_multistable}"
TARGET_VARIANT="${TARGET_VARIANT:-lista_dense_signsplit_p256_hardinit_out_of_generator}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect_pass0}"
INTERP_DIR="${INTERP_DIR:-${RESULTS_DIR}/interpretability_pass0}"
ORACLE_DIR="${ORACLE_DIR:-${RESULTS_DIR}/oracle_vs_learned_local_koopman}"
REGIME_DIR="${REGIME_DIR:-${RESULTS_DIR}/regime_discovery_local_koopman}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
AUTOMATION_DIR="${AUTOMATION_DIR:-${RESULTS_DIR}/automation}"
SOURCE_TSV="${SOURCE_TSV:-results/transition_rich_lista_dense_p256_hardinit_table123_20260430/task_tables/transition_rich_lista_dense_p256_hardinit_table123.tsv}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
NUM_STEPS="${NUM_STEPS:-200000}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-16}"
CLUSTER_FIT_MAX_SAMPLES="${CLUSTER_FIT_MAX_SAMPLES:-2048}"

mkdir -p \
  "${TASK_DIR}" \
  "${ROOT_SPEC_DIR}" \
  "${COLLECT_DIR}" \
  "${INTERP_DIR}/shards" \
  "${ORACLE_DIR}/shards" \
  "${REGIME_DIR}/shards" \
  "${LOG_DIR}" \
  "${AUTOMATION_DIR}"

TASK_TSV="${TASK_DIR}/out_of_generator_multistable_p256_lista.tsv"
MANIFEST_JSON="${TASK_DIR}/out_of_generator_multistable_p256_lista_manifest.json"
ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/out_of_generator_multistable_p256_lista_roots.txt"

PHASE_LABEL="${PHASE_LABEL}" \
TARGET_VARIANT="${TARGET_VARIANT}" \
SEEDS_CSV="${SEEDS_CSV}" \
SYSTEMS_CSV="${SYSTEMS_CSV}" \
NUM_STEPS="${NUM_STEPS}" \
  uv run python - "${SOURCE_TSV}" "${TASK_TSV}" "${MANIFEST_JSON}" <<'PY'
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

from skae.benchmarks.transition_rich_basin_partition_manifest import (
    resolve_transition_rich_default_dt,
    transition_rich_out_of_generator_multistable_systems,
)

source_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])

phase_label = os.environ["PHASE_LABEL"]
target_variant = os.environ["TARGET_VARIANT"]
requested_seeds = [seed.strip() for seed in os.environ["SEEDS_CSV"].split(",") if seed.strip()]
requested_systems = {item.strip() for item in os.environ["SYSTEMS_CSV"].split(",") if item.strip()}
num_steps = os.environ["NUM_STEPS"]

with source_path.open(newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fields = list(reader.fieldnames or [])

systems = transition_rich_out_of_generator_multistable_systems()
if requested_systems:
    systems = [spec for spec in systems if spec.system_key in requested_systems]
if not systems:
    raise SystemExit("No supplemental out-of-generator systems selected")

rows = []
for spec in systems:
    for seed in requested_seeds:
        row = {field: "" for field in fields}
        row.update(
            {
                "task_id": str(len(rows)),
                "phase": phase_label,
                "model_variant": target_variant,
                "config_name": "lista_parity_generic_sparse",
                "system_key": spec.system_key,
                "system_slug": spec.system_slug,
                "system_group": spec.system_group,
                "paper_role": spec.paper_role,
                "env_name": spec.env_name,
                "basin_count": str(spec.basin_count),
                "seed": str(seed),
                "num_steps": num_steps,
                "batch_size": "256",
                "target_size": "256",
                "sequence_length": "8",
                "hard_init_oversample": "true",
                "hard_init_fraction": "0.5",
                "hard_init_pool_size": "1024",
                "hard_init_num_candidates": "4096",
                "hard_init_probe_steps": "32",
                "hard_init_num_perturbations": "4",
                "hard_init_perturb_scale": "0.04",
                "hard_init_transient_window": "8",
                "hard_init_transient_weight": "0.5",
                "hard_init_jitter_scale": "0.25",
                "res_coeff": "1.0",
                "reconst_coeff": "0.03",
                "pred_coeff": "1.0",
                "sparsity_coeff": "0.003",
                "lista_alpha": "0.15",
                "lista_num_loops": "2",
                "lista_final_op": "sign_split",
                "k_structure": "dense",
                "block_loss": "0",
                "eval_use_dynamics_prior": "false",
                "eval_event_trigger_min_dwell": "0",
                "eval_event_trigger_max_interval": "0",
                "structured": "0",
                "soft_block": "0",
                "lr": "5e-05",
                "k_matrix_lr": "5e-06",
                "weight_decay": "0.0001",
                "env_dt": f"{resolve_transition_rich_default_dt(spec.system_key):.12g}",
                "eval_profile": "full",
                "standardize": "0",
                "dysts_native_cache": "0",
                "dysts_cache_reuse": "0",
            }
        )
        rows.append(row)

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

manifest = {
    "experiment": "out_of_generator_multistable_p256_lista",
    "target_variant": target_variant,
    "task_tsv": str(out_path),
    "phase_label": phase_label,
    "seeds": [int(seed) for seed in requested_seeds],
    "systems": [spec.system_key for spec in systems],
    "settings": {
        "num_steps": int(num_steps),
        "batch_size": 256,
        "target_size": 256,
        "sequence_length": 8,
        "sparsity_coeff": 0.003,
        "k_structure": "dense",
        "lista_alpha": 0.15,
        "lista_num_loops": 2,
        "lista_final_op": "sign_split",
    },
    "num_tasks": len(rows),
    "counts_by_system": dict(sorted(Counter(row["system_key"] for row in rows).items())),
}
manifest_path.write_text(json.dumps(manifest, indent=2))
print(json.dumps({"task_tsv": str(out_path), "num_tasks": len(rows), "target_variant": target_variant}, indent=2))
PY

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi

printf '%s=%s/%s/%s\n' "${TARGET_VARIANT}" "${BASE_OUT}" "${PHASE_LABEL}" "${TARGET_VARIANT}" > "${ROOT_SPECS_FILE}"
SYSTEMS_FOR_EVAL="$(uv run python - "${MANIFEST_JSON}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
print(",".join(payload["systems"]))
PY
)"

ARRAY_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" \
    sbatch \
      --array=0-$((TASK_COUNT - 1))%"${ARRAY_THROTTLE}" \
      --partition=long \
      scripts/run_paper_benchmark_array.sh | awk '{print $4}'
)

COLLECT_JOB_ID=$(
  ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" \
  OUT_DIR="${COLLECT_DIR}" \
  HORIZONS_CSV="100,500,1000" \
  GOOD_THRESHOLD="50" \
    sbatch \
      --dependency=afterany:"${ARRAY_JOB_ID}" \
      --partition=long \
      scripts/collect_transition_rich_basin_partition.sh | awk '{print $4}'
)

INTERP_SHARD_DIR="${INTERP_DIR}/shards/${TARGET_VARIANT}"
mkdir -p "${INTERP_SHARD_DIR}"
INTERP_JOB_ID=$(
  ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" \
  OUT_DIR="${INTERP_SHARD_DIR}" \
  ROOT_LABELS_CSV="${TARGET_VARIANT}" \
  SYSTEMS_CSV="${SYSTEMS_FOR_EVAL}" \
  SEEDS_CSV="${SEEDS_CSV}" \
  DEPTH_SLICE_MODE="global" \
  PROGRESS_EVERY_RUNS="1" \
  FLUSH_EVERY_RUNS="5" \
    sbatch \
      --dependency=afterok:"${COLLECT_JOB_ID}" \
      --partition=long \
      --time="08:00:00" \
      --cpus-per-task="4" \
      --mem="16G" \
      --output="${INTERP_DIR}/interp-%A.out" \
      --error="${INTERP_DIR}/interp-%A.err" \
      scripts/reduce_transition_rich_interpretability_metrics.sh | awk '{print $4}'
)

INTERP_MERGE_JOB_ID=$(
  SHARDS_DIR="${INTERP_DIR}/shards" \
  OUT_DIR="${INTERP_DIR}/merged" \
  ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" \
  ROOT_LABELS_CSV="${TARGET_VARIANT}" \
  SYSTEMS_CSV="${SYSTEMS_FOR_EVAL}" \
  SEEDS_CSV="${SEEDS_CSV}" \
    sbatch \
      --dependency=afterany:"${INTERP_JOB_ID}" \
      --partition=long \
      --time="00:30:00" \
      --mem="4G" \
      --output="${INTERP_DIR}/merge-%A.out" \
      --error="${INTERP_DIR}/merge-%A.err" \
      scripts/merge_transition_rich_interpretability_shards.sh | awk '{print $4}'
)

ORACLE_JOB_ID=$(
  ROWS_CSVS="${COLLECT_DIR}/forecasting_rows.csv" \
  OUT_DIR="${ORACLE_DIR}/shards/${TARGET_VARIANT}" \
  ROOT_LABELS_CSV="${TARGET_VARIANT}" \
  SYSTEMS_CSV="${SYSTEMS_FOR_EVAL}" \
  SEEDS_CSV="${SEEDS_CSV}" \
  SUPPORT_DEFINITIONS="topk:8" \
  DEPTH_STRATA="all,deep" \
  ROLLOUT_MODES="global_k,support_local_centered,family_local_centered,latent_kmeans_local_centered,oracle_basin_local_centered" \
  REENCODE_PERIODS="0" \
  HORIZONS="100,500,1000" \
  LABEL_MODE="auto" \
  PROGRESS_EVERY_RUNS="1" \
  FLUSH_EVERY_RUNS="1" \
    sbatch \
      --dependency=afterok:"${COLLECT_JOB_ID}" \
      --partition=long \
      --time="12:00:00" \
      --cpus-per-task="4" \
      --mem="24G" \
      --output="${ORACLE_DIR}/oracle-%A.out" \
      --error="${ORACLE_DIR}/oracle-%A.err" \
      scripts/run_transition_rich_self_routed_forecasting.sh | awk '{print $4}'
)

ORACLE_MERGE_JOB_ID=$(
  SHARDS_DIR="${ORACLE_DIR}/shards" \
  OUT_DIR="${ORACLE_DIR}/merged" \
  ROWS_CSVS="${COLLECT_DIR}/forecasting_rows.csv" \
  ROOT_LABELS_CSV="${TARGET_VARIANT}" \
  SYSTEMS_CSV="${SYSTEMS_FOR_EVAL}" \
  SEEDS_CSV="${SEEDS_CSV}" \
    sbatch \
      --dependency=afterany:"${ORACLE_JOB_ID}" \
      --partition=long \
      --time="00:30:00" \
      --mem="4G" \
      --output="${ORACLE_DIR}/merge-%A.out" \
      --error="${ORACLE_DIR}/merge-%A.err" \
      scripts/merge_transition_rich_self_routed_forecasting_shards.sh | awk '{print $4}'
)

REGIME_JOB_ID=$(
  ROWS_CSVS="${COLLECT_DIR}/forecasting_rows.csv" \
  OUT_DIR="${REGIME_DIR}/shards/${TARGET_VARIANT}" \
  ROOT_LABELS_CSV="${TARGET_VARIANT}" \
  SYSTEMS_CSV="${SYSTEMS_FOR_EVAL}" \
  SEEDS_CSV="${SEEDS_CSV}" \
  SUPPORT_DEFINITION="topk:8" \
  FEATURE_VIEWS="raw_state,dense_latent,sparse_latent_values,support_binary" \
  CLUSTER_METHODS="kmeans,gmm_diag,spectral" \
  CLUSTER_COUNT_MODES="basin_count,support_family_count" \
  CLUSTER_FIT_MAX_SAMPLES="${CLUSTER_FIT_MAX_SAMPLES}" \
  LABEL_MODE="auto" \
  PROGRESS_EVERY_RUNS="1" \
  FLUSH_EVERY_RUNS="1" \
    sbatch \
      --dependency=afterok:"${COLLECT_JOB_ID}" \
      --partition=long \
      --time="24:00:00" \
      --cpus-per-task="4" \
      --mem="24G" \
      --output="${REGIME_DIR}/regime-%A.out" \
      --error="${REGIME_DIR}/regime-%A.err" \
      scripts/run_regime_discovery_local_koopman.sh | awk '{print $4}'
)

REGIME_MERGE_JOB_ID=$(
  SHARDS_DIR="${REGIME_DIR}/shards" \
  OUT_DIR="${REGIME_DIR}/merged" \
  ROWS_CSVS="${COLLECT_DIR}/forecasting_rows.csv" \
  ROOT_LABELS_CSV="${TARGET_VARIANT}" \
  SYSTEMS_CSV="${SYSTEMS_FOR_EVAL}" \
  SEEDS_CSV="${SEEDS_CSV}" \
    sbatch \
      --dependency=afterany:"${REGIME_JOB_ID}" \
      --partition=long \
      --time="00:30:00" \
      --mem="4G" \
      --output="${REGIME_DIR}/merge-%A.out" \
      --error="${REGIME_DIR}/merge-%A.err" \
      scripts/merge_regime_discovery_local_koopman_shards.sh | awk '{print $4}'
)

cat > "${AUTOMATION_DIR}/queue.json" <<EOF
{
  "experiment_tag": "${EXPERIMENT_TAG}",
  "target_variant": "${TARGET_VARIANT}",
  "results_dir": "${RESULTS_DIR}",
  "base_out": "${BASE_OUT}",
  "task_tsv": "${TASK_TSV}",
  "manifest_json": "${MANIFEST_JSON}",
  "root_specs_file": "${ROOT_SPECS_FILE}",
  "task_count": ${TASK_COUNT},
  "systems_csv": "${SYSTEMS_FOR_EVAL}",
  "seeds_csv": "${SEEDS_CSV}",
  "array_job_id": "${ARRAY_JOB_ID}",
  "collect_job_id": "${COLLECT_JOB_ID}",
  "interpretability_job_id": "${INTERP_JOB_ID}",
  "interpretability_merge_job_id": "${INTERP_MERGE_JOB_ID}",
  "oracle_job_id": "${ORACLE_JOB_ID}",
  "oracle_merge_job_id": "${ORACLE_MERGE_JOB_ID}",
  "regime_job_id": "${REGIME_JOB_ID}",
  "regime_merge_job_id": "${REGIME_MERGE_JOB_ID}"
}
EOF

echo "Queued out-of-generator multistable p256 LISTA packet."
echo "Array job: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Interpretability: ${INTERP_JOB_ID} -> ${INTERP_MERGE_JOB_ID}"
echo "Oracle/local-K: ${ORACLE_JOB_ID} -> ${ORACLE_MERGE_JOB_ID}"
echo "Regime discovery: ${REGIME_JOB_ID} -> ${REGIME_MERGE_JOB_ID}"
echo "Results dir: ${RESULTS_DIR}"
