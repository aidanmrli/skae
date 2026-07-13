#!/bin/bash
#
# Queue a fair d_z=256 sequence-8 HyperLISTA substitution panel found by the
# 10k-step interactive search, plus the matched LISTA dense baseline.
#
# Submit:
#   sbatch scripts/queue_hyperlista_dz256_10k_all_systems.sh
#
# Optional env vars:
#   DATE_TAG=20260512
#   EXPERIMENT_TAG=hyperlista_dz256_seq8_fair_10k_all_systems_${DATE_TAG}
#   SEEDS_CSV=0,1,2
#   ARRAY_THROTTLE=45
#   MAX_EXISTING_JOBS_BEFORE_SUBMIT=650

#SBATCH --job-name=queue_hlista10k
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH -o /network/scratch/l/lia/skae/queue-hyperlista-dz256-10k-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-hyperlista-dz256-10k-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run under SLURM."
  echo "Submit it with: sbatch scripts/queue_hyperlista_dz256_10k_all_systems.sh"
  exit 2
fi

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-20260512}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-hyperlista_dz256_seq8_fair_10k_all_systems_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-transition_rich_hyperlista_dz256_seq8_fair_10k}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect_pass0}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-${RESULTS_DIR}/queue_logs}"
AUTOMATION_DIR="${AUTOMATION_DIR:-${RESULTS_DIR}/automation}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-45}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-650}"
SUBMIT_WAIT_SECONDS="${SUBMIT_WAIT_SECONDS:-60}"

SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
SYSTEMS_CSV="${SYSTEMS_CSV:-gated_local_linear,gated_transfer_linear,claude:arrested_spiral,claude:cal_asymmetric_3,claude:cal_high_cross_3,claude:cal_hexagon_6,claude:cal_octagon_8,claude:cal_pentagon_5,claude:cal_square_4,claude:duffing_triple_well,claude:snic_multi,claude:transition_routes_4,claude:var_depth_gradient_4,claude:var_diamond_4,claude:var_l_shape_5}"

mkdir -p \
  "${TASK_DIR}" \
  "${ROOT_SPEC_DIR}" \
  "${COLLECT_DIR}" \
  "${QUEUE_LOG_DIR}" \
  "${AUTOMATION_DIR}"

TASK_TSV="${TASK_DIR}/hyperlista_dz256_seq8_fair_10k_all_systems_tasks.tsv"
MANIFEST_JSON="${TASK_DIR}/hyperlista_dz256_seq8_fair_10k_all_systems_manifest.json"
ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/hyperlista_dz256_seq8_fair_10k_all_systems_roots.txt"

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Date: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "SYSTEMS_CSV: ${SYSTEMS_CSV}"
echo "SEEDS_CSV: ${SEEDS_CSV}"
echo "Optimizer steps: 10000 for every queued row"

SYSTEMS_CSV="${SYSTEMS_CSV}" \
SEEDS_CSV="${SEEDS_CSV}" \
PHASE_LABEL="${PHASE_LABEL}" \
TASK_TSV="${TASK_TSV}" \
MANIFEST_JSON="${MANIFEST_JSON}" \
uv run python - <<'PY'
import csv
import json
import os
from pathlib import Path

from skae.benchmarks.transition_rich_basin_partition_manifest import (
    get_transition_rich_basin_partition_system,
    resolve_transition_rich_default_dt,
)


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


systems = [get_transition_rich_basin_partition_system(key) for key in parse_csv(os.environ["SYSTEMS_CSV"])]
seeds = [int(seed) for seed in parse_csv(os.environ["SEEDS_CSV"])]
phase = os.environ["PHASE_LABEL"]

base_common = {
    "num_steps": 10000,
    "batch_size": 256,
    "target_size": 256,
    "res_coeff": 1.0,
    "reconst_coeff": 0.03,
    "pred_coeff": 1.0,
    "sparsity_coeff": 0.003,
    "k_structure": "dense",
    "weight_decay": 1e-4,
    "eval_profile": "smoke",
    "eval_every": 10000,
    "eval_num_steps": 100,
    "skip_basin_eval": 1,
}

variants = [
    {
        **base_common,
        "model_variant": "lista_dense_dz256_seq8_10k",
        "config_name": "lista_parity_generic_sparse",
        "sequence_length": 8,
        "lr": 5e-5,
        "k_matrix_lr": 5e-6,
        "lista_alpha": 0.15,
        "lista_num_loops": 1,
        "lista_final_op": "relu",
    },
    {
        **base_common,
        "model_variant": "hyperlista_dense_dz256_seq8_ctheta0p1_noss_nomom_10k",
        "config_name": "hyperlista_parity_generic_sparse",
        "sequence_length": 8,
        "lr": 5e-5,
        "k_matrix_lr": 5e-6,
        "hyperlista_c_theta": 0.1,
        "hyperlista_c_beta": 0.0,
        "hyperlista_c_ss": 0.0,
        "hyperlista_use_ss": "false",
        "hyperlista_use_momentum": "false",
    },
    {
        **base_common,
        "model_variant": "hyperlista_dense_dz256_seq8_ctheta0p2_noss_nomom_10k",
        "config_name": "hyperlista_parity_generic_sparse",
        "sequence_length": 8,
        "lr": 5e-5,
        "k_matrix_lr": 5e-6,
        "hyperlista_c_theta": 0.2,
        "hyperlista_c_beta": 0.0,
        "hyperlista_c_ss": 0.0,
        "hyperlista_use_ss": "false",
        "hyperlista_use_momentum": "false",
    },
]

optional_fields = [
    "lista_alpha",
    "lista_num_loops",
    "lista_use_momentum",
    "lista_momentum_beta",
    "lista_linear_encoder",
    "lista_final_op",
    "hyperlista_c_theta",
    "hyperlista_c_beta",
    "hyperlista_c_ss",
    "hyperlista_use_ss",
    "hyperlista_use_momentum",
]

fieldnames = [
    "task_id",
    "phase",
    "model_variant",
    "config_name",
    "system_key",
    "system_slug",
    "system_group",
    "paper_role",
    "env_name",
    "basin_count",
    "seed",
    "num_steps",
    "batch_size",
    "target_size",
    "sequence_length",
    "res_coeff",
    "reconst_coeff",
    "pred_coeff",
    "sparsity_coeff",
    *optional_fields,
    "k_structure",
    "lr",
    "k_matrix_lr",
    "weight_decay",
    "env_dt",
    "eval_profile",
    "eval_every",
    "eval_num_steps",
    "skip_basin_eval",
]

rows = []
task_id = 0
for variant in variants:
    for system in systems:
        env_dt = resolve_transition_rich_default_dt(system.system_key)
        for seed in seeds:
            row = {field: "" for field in fieldnames}
            row.update(variant)
            row.update(
                {
                    "task_id": task_id,
                    "phase": phase,
                    "system_key": system.system_key,
                    "system_slug": system.system_slug,
                    "system_group": system.system_group,
                    "paper_role": system.paper_role,
                    "env_name": system.env_name,
                    "basin_count": system.basin_count,
                    "seed": seed,
                    "env_dt": env_dt,
                }
            )
            rows.append(row)
            task_id += 1

task_path = Path(os.environ["TASK_TSV"])
task_path.parent.mkdir(parents=True, exist_ok=True)
with task_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

manifest = {
    "experiment": "hyperlista_dz256_seq8_fair_10k_all_systems",
    "phase_label": phase,
    "task_count": len(rows),
    "systems": [system.system_key for system in systems],
    "seeds": seeds,
    "variants": variants,
    "all_rows_num_steps": 10000,
    "interactive_success_context": {
        "autonomous_no_reencode": True,
        "note": "Sequence-8 comparisons are fair against the matched sequence-8 LISTA baseline. The later sequence-32 transition_routes_4 HyperLISTA run is excluded because a matching sequence-32 LISTA baseline beat it.",
    },
}
manifest_path = Path(os.environ["MANIFEST_JSON"])
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

print(f"Wrote {len(rows)} tasks to {task_path}")
PY

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi

{
  printf '%s=%s/%s/%s\n' \
    "lista_dense_dz256_seq8_10k" \
    "${BASE_OUT}" "${PHASE_LABEL}" "lista_dense_dz256_seq8_10k"
  printf '%s=%s/%s/%s\n' \
    "hyperlista_dense_dz256_seq8_ctheta0p1_noss_nomom_10k" \
    "${BASE_OUT}" "${PHASE_LABEL}" "hyperlista_dense_dz256_seq8_ctheta0p1_noss_nomom_10k"
  printf '%s=%s/%s/%s\n' \
    "hyperlista_dense_dz256_seq8_ctheta0p2_noss_nomom_10k" \
    "${BASE_OUT}" "${PHASE_LABEL}" "hyperlista_dense_dz256_seq8_ctheta0p2_noss_nomom_10k"
} > "${ROOT_SPECS_FILE}"

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
    sbatch --array=0-$((TASK_COUNT - 1))%"${ARRAY_THROTTLE}" scripts/run_paper_benchmark_array.sh | awk '{print $4}'
)

COLLECT_JOB_ID=$(
  ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" \
  OUT_DIR="${COLLECT_DIR}" \
  HORIZONS_CSV="100,500,1000" \
  GOOD_THRESHOLD="50" \
    sbatch --dependency=afterany:"${ARRAY_JOB_ID}" scripts/collect_transition_rich_basin_partition.sh | awk '{print $4}'
)

cat > "${AUTOMATION_DIR}/hyperlista_dz256_seq8_fair_10k_all_systems_queue.json" <<EOF
{
  "results_dir": "${RESULTS_DIR}",
  "base_out": "${BASE_OUT}",
  "task_tsv": "${TASK_TSV}",
  "manifest_json": "${MANIFEST_JSON}",
  "root_specs_file": "${ROOT_SPECS_FILE}",
  "task_count": ${TASK_COUNT},
  "array_job_id": "${ARRAY_JOB_ID}",
  "collect_job_id": "${COLLECT_JOB_ID}",
  "systems_csv": "${SYSTEMS_CSV}",
  "seeds_csv": "${SEEDS_CSV}",
  "num_steps": 10000,
  "array_throttle": "${ARRAY_THROTTLE}"
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
} > "${QUEUE_LOG_DIR}/launch_record.env"

echo "Queued fair sequence-8 HyperLISTA d_z=256 10k all-system panel."
echo "Training array: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Task count: ${TASK_COUNT}"
echo "Results dir: ${RESULTS_DIR}"
