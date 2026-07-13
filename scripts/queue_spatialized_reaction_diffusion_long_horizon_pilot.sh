#!/usr/bin/env bash
#
# Queue the spatialized PDE long-horizon pilot.
#
# This stores long trajectories for H512 scoring while limiting training
# windows to the first 128 observation intervals by default.
#
# Submit with:
#   sbatch scripts/queue_spatialized_reaction_diffusion_long_horizon_pilot.sh
#
# Useful overrides:
#   SYSTEMS_CSV=cal_square_4,transition_routes_4 SEEDS_CSV=0,1 \
#   sbatch scripts/queue_spatialized_reaction_diffusion_long_horizon_pilot.sh
#
#SBATCH --job-name=queue-spatial-rd-long
#SBATCH --output=/network/scratch/l/lia/skae/queue-spatial-rd-long-%A.out
#SBATCH --error=/network/scratch/l/lia/skae/queue-spatial-rd-long-%A.err
#SBATCH --time=00:30:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

EXPERIMENT_TAG="${EXPERIMENT_TAG:-spatial_rd_long_horizon_pilot_20260526}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
TASK_TSV="${TASK_TSV:-${RESULTS_DIR}/spatialized_rd_tasks.tsv}"
MANIFEST_JSON="${MANIFEST_JSON:-${RESULTS_DIR}/spatialized_rd_manifest.json}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"

SYSTEMS_CSV="${SYSTEMS_CSV:-cal_square_4,transition_routes_4}"
SEEDS_CSV="${SEEDS_CSV:-0,1}"
MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV:-conv_lista,conv_dense,conv_sparse_mlp}"
GRID_SIZE="${GRID_SIZE:-16}"
DIFFUSION="${DIFFUSION:-0.01}"
RK4_DT="${RK4_DT:-0.005}"
SUBSTEPS_PER_OBSERVATION="${SUBSTEPS_PER_OBSERVATION:-5}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-512}"
TRAIN_OBSERVATION_LIMIT="${TRAIN_OBSERVATION_LIMIT:-128}"
LABEL_EXTRA_OBSERVATIONS="${LABEL_EXTRA_OBSERVATIONS:-512}"
TRAIN_TRAJECTORIES="${TRAIN_TRAJECTORIES:-48}"
VAL_TRAJECTORIES="${VAL_TRAJECTORIES:-16}"
TEST_TRAJECTORIES="${TEST_TRAJECTORIES:-16}"
LAPLACIAN_SCALING="${LAPLACIAN_SCALING:-continuum}"
TARGET_SIZE="${TARGET_SIZE:-2048}"
MIN_LATENT_STATE_RATIO="${MIN_LATENT_STATE_RATIO:-4.0}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-32}"
NUM_BLOCKS="${NUM_BLOCKS:-2}"
CONV_ACTIVATION="${CONV_ACTIVATION:-}"
NUM_STEPS="${NUM_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS:-3}"
LISTA_ALPHA="${LISTA_ALPHA:-0.2}"
LISTA_ALPHA_CSV="${LISTA_ALPHA_CSV:-}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.05}"
SPARSITY_COEFF_CSV="${SPARSITY_COEFF_CSV:-}"
DENSE_SPARSITY_COEFF="${DENSE_SPARSITY_COEFF:-0.0}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-0.2}"
FAMILY_JACCARD="${FAMILY_JACCARD:-0.2}"
MAX_VALIDATION_REPS="${MAX_VALIDATION_REPS:-1024}"
DEEP_THRESHOLD="${DEEP_THRESHOLD:-0.7}"
EVAL_HORIZONS="${EVAL_HORIZONS:-1,4,8,16,32,64,128,256,512}"
EVAL_HORIZON="${EVAL_HORIZON:-32}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
RESUME_FROM_LATEST="${RESUME_FROM_LATEST:-1}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-4}"
RUNNER_PARTITION="${RUNNER_PARTITION:-long}"
RUNNER_GRES="${RUNNER_GRES-gpu:1}"
RUNNER_TIME="${RUNNER_TIME:-02:50:00}"
RUNNER_MEM="${RUNNER_MEM:-24G}"
RUNNER_CPUS="${RUNNER_CPUS:-4}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-scripts/run_spatialized_reaction_diffusion_array.sh}"

CHAIN_SUPPORT_SWEEP="${CHAIN_SUPPORT_SWEEP:-1}"
SUPPORT_EXPERIMENT_TAG="${SUPPORT_EXPERIMENT_TAG:-${EXPERIMENT_TAG}_support_sweep}"
SUPPORT_RESULTS_DIR="${SUPPORT_RESULTS_DIR:-results/${SUPPORT_EXPERIMENT_TAG}}"
SUPPORT_OUTPUT_ROOT="${SUPPORT_OUTPUT_ROOT:-/network/scratch/l/lia/skae/${SUPPORT_EXPERIMENT_TAG}}"
SUPPORT_THRESHOLDS_CSV="${SUPPORT_THRESHOLDS_CSV:-0.05,0.1,0.2,0.3,0.4,0.5}"
SUPPORT_FAMILY_JACCARDS_CSV="${SUPPORT_FAMILY_JACCARDS_CSV:-0.05,0.1,0.2,0.25,0.3,0.4}"
SUPPORT_RUNNER_TIME="${SUPPORT_RUNNER_TIME:-02:00:00}"
SUPPORT_RUNNER_MEM="${SUPPORT_RUNNER_MEM:-12G}"
SUPPORT_RUNNER_CPUS="${SUPPORT_RUNNER_CPUS:-2}"
SUPPORT_ARRAY_THROTTLE="${SUPPORT_ARRAY_THROTTLE:-4}"

mkdir -p "${RESULTS_DIR}" "${BASE_OUT}" "${LOG_DIR}"

echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse --short HEAD || true)"
echo "experiment_tag=${EXPERIMENT_TAG}"
echo "systems_csv=${SYSTEMS_CSV}"
echo "seeds_csv=${SEEDS_CSV}"
echo "model_variants_csv=${MODEL_VARIANTS_CSV}"
echo "trajectory_length=${TRAJECTORY_LENGTH}"
echo "train_observation_limit=${TRAIN_OBSERVATION_LIMIT}"
echo "label_extra_observations=${LABEL_EXTRA_OBSERVATIONS}"
echo "eval_horizons=${EVAL_HORIZONS}"
echo "lista_alpha_csv=${LISTA_ALPHA_CSV}"
echo "sparsity_coeff_csv=${SPARSITY_COEFF_CSV}"
echo "conv_activation=${CONV_ACTIVATION}"

export BASE_OUT SYSTEMS_CSV SEEDS_CSV MODEL_VARIANTS_CSV GRID_SIZE DIFFUSION RK4_DT
export SUBSTEPS_PER_OBSERVATION TRAJECTORY_LENGTH TRAIN_OBSERVATION_LIMIT
export LABEL_EXTRA_OBSERVATIONS TRAIN_TRAJECTORIES VAL_TRAJECTORIES TEST_TRAJECTORIES
export LAPLACIAN_SCALING TARGET_SIZE MIN_LATENT_STATE_RATIO HIDDEN_CHANNELS NUM_BLOCKS
export CONV_ACTIVATION NUM_STEPS BATCH_SIZE SEQUENCE_LENGTH LISTA_NUM_LOOPS LISTA_ALPHA LISTA_ALPHA_CSV
export SPARSITY_COEFF SPARSITY_COEFF_CSV DENSE_SPARSITY_COEFF SUPPORT_THRESHOLD FAMILY_JACCARD
export MAX_VALIDATION_REPS DEEP_THRESHOLD EVAL_HORIZONS EVAL_HORIZON

uv run python - "${TASK_TSV}" "${MANIFEST_JSON}" <<'PY'
import csv
import json
import math
import os
import sys
from pathlib import Path


def parse_csv(raw):
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def tagify(raw):
    return raw.replace(":", "_").replace("/", "_").replace(".", "p").replace("-", "_")


task_tsv = Path(sys.argv[1])
manifest_json = Path(sys.argv[2])
systems = parse_csv(os.environ["SYSTEMS_CSV"])
seeds = [int(seed) for seed in parse_csv(os.environ["SEEDS_CSV"])]
eval_horizons = [int(horizon) for horizon in parse_csv(os.environ["EVAL_HORIZONS"])]
lista_alphas = [float(value) for value in parse_csv(os.environ.get("LISTA_ALPHA_CSV", ""))]
if not lista_alphas:
    lista_alphas = [float(os.environ["LISTA_ALPHA"])]
sparsity_coeffs = [float(value) for value in parse_csv(os.environ.get("SPARSITY_COEFF_CSV", ""))]
if not sparsity_coeffs:
    sparsity_coeffs = [float(os.environ["SPARSITY_COEFF"])]
trajectory_length = int(os.environ["TRAJECTORY_LENGTH"])
too_long = [horizon for horizon in eval_horizons if horizon > trajectory_length]
if too_long:
    raise SystemExit(
        "Requested eval_horizons exceed trajectory_length: "
        f"trajectory_length={trajectory_length}, too_long={too_long}."
    )

grid_size = int(os.environ["GRID_SIZE"])
state_dim = 2 * grid_size * grid_size
min_target_size = math.ceil(float(os.environ["MIN_LATENT_STATE_RATIO"]) * state_dim)
target_size = int(os.environ["TARGET_SIZE"])
if target_size < min_target_size:
    raise SystemExit(
        f"TARGET_SIZE={target_size} is below required overcomplete minimum {min_target_size}."
    )

requested_variants = parse_csv(os.environ["MODEL_VARIANTS_CSV"])
allowed_variants = {"conv_lista", "conv_dense", "conv_sparse_mlp"}
unknown_variants = sorted(set(requested_variants) - allowed_variants)
if unknown_variants:
    raise SystemExit(f"Unknown model variants: {unknown_variants}")
is_hparam_sweep = len(lista_alphas) * len(sparsity_coeffs) > 1
base_out = Path(os.environ["BASE_OUT"])
rows = []
task_id = 0
for system in systems:
    system_slug = tagify(system)
    for seed in seeds:
        for model_variant in requested_variants:
            model_slug = tagify(model_variant)
            dataset_root = base_out / "runs" / model_slug / system_slug / f"grid{grid_size}" / f"seed_{seed}"
            for lista_alpha in lista_alphas:
                for sparse_coeff in sparsity_coeffs:
                    sparsity_coeff = (
                        float(os.environ["DENSE_SPARSITY_COEFF"])
                        if model_variant == "conv_dense"
                        else sparse_coeff
                    )
                    setting_slug = (
                        f"alpha_{tagify(f'{lista_alpha:g}')}_"
                        f"sp_{tagify(f'{sparsity_coeff:g}')}_"
                        f"act_{tagify(os.environ['CONV_ACTIVATION'] or 'default')}"
                    )
                    task_root = dataset_root / setting_slug if is_hparam_sweep else dataset_root
                    rows.append(
                        {
                            "task_id": task_id,
                            "source_system": system,
                            "seed": seed,
                            "model_variant": model_variant,
                            "trainer": "conv",
                            "config_name": "lista_parity_generic_sparse",
                            "setting_slug": setting_slug if is_hparam_sweep else "",
                            "grid_size": grid_size,
                            "diffusion": os.environ["DIFFUSION"],
                            "rk4_dt": os.environ["RK4_DT"],
                            "substeps_per_observation": os.environ["SUBSTEPS_PER_OBSERVATION"],
                            "trajectory_length": trajectory_length,
                            "label_extra_observations": os.environ["LABEL_EXTRA_OBSERVATIONS"],
                            "train_trajectories": os.environ["TRAIN_TRAJECTORIES"],
                            "val_trajectories": os.environ["VAL_TRAJECTORIES"],
                            "test_trajectories": os.environ["TEST_TRAJECTORIES"],
                            "laplacian_scaling": os.environ["LAPLACIAN_SCALING"],
                            "state_dim": state_dim,
                            "target_size": target_size,
                            "min_target_size": min_target_size,
                            "latent_state_ratio": target_size / state_dim,
                            "hidden_channels": os.environ["HIDDEN_CHANNELS"],
                            "num_blocks": os.environ["NUM_BLOCKS"],
                            "conv_activation": os.environ["CONV_ACTIVATION"],
                            "num_steps": os.environ["NUM_STEPS"],
                            "batch_size": os.environ["BATCH_SIZE"],
                            "sequence_length": os.environ["SEQUENCE_LENGTH"],
                            "train_observation_limit": os.environ["TRAIN_OBSERVATION_LIMIT"],
                            "lista_num_loops": os.environ["LISTA_NUM_LOOPS"],
                            "lista_alpha": lista_alpha,
                            "sparsity_coeff": sparsity_coeff,
                            "support_threshold": os.environ["SUPPORT_THRESHOLD"],
                            "family_jaccard": os.environ["FAMILY_JACCARD"],
                            "max_validation_reps": os.environ["MAX_VALIDATION_REPS"],
                            "deep_threshold": os.environ["DEEP_THRESHOLD"],
                            "eval_horizons": os.environ["EVAL_HORIZONS"],
                            "eval_horizon": os.environ["EVAL_HORIZON"],
                            "dataset_path": str(dataset_root / "dataset.pt"),
                            "run_dir": str(task_root / "model"),
                            "eval_path": str(task_root / "evaluation.json"),
                        }
                    )
                    task_id += 1

task_tsv.parent.mkdir(parents=True, exist_ok=True)
with task_tsv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

manifest_json.parent.mkdir(parents=True, exist_ok=True)
manifest_json.write_text(
    json.dumps(
        {
            "num_tasks": len(rows),
            "systems": systems,
            "seeds": seeds,
            "model_variants": requested_variants,
            "lista_alphas": lista_alphas,
            "sparsity_coeffs": sparsity_coeffs,
            "conv_activation": os.environ["CONV_ACTIVATION"],
            "trajectory_length": trajectory_length,
            "train_observation_limit": int(os.environ["TRAIN_OBSERVATION_LIMIT"]),
            "label_extra_observations": int(os.environ["LABEL_EXTRA_OBSERVATIONS"]),
            "eval_horizons": eval_horizons,
            "sequence_length": int(os.environ["SEQUENCE_LENGTH"]),
            "state_dim": state_dim,
            "target_size": target_size,
            "latent_state_ratio": target_size / state_dim,
            "label_policy": "Basin labels and attractor centers are stored for evaluation only.",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(f"Wrote {len(rows)} long-horizon tasks to {task_tsv}")
PY

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if [[ "${TASK_COUNT}" -le 0 ]]; then
  echo "No tasks generated in ${TASK_TSV}." >&2
  exit 1
fi

ARRAY_SPEC="0-$((TASK_COUNT - 1))%${ARRAY_THROTTLE}"
SBATCH_RUNNER_ARGS=(
  --parsable
  --array="${ARRAY_SPEC}"
  --partition="${RUNNER_PARTITION}"
  --time="${RUNNER_TIME}"
  --mem="${RUNNER_MEM}"
  --cpus-per-task="${RUNNER_CPUS}"
  --output="${LOG_DIR}/spatial-rd-long-%A_%a.out"
  --error="${LOG_DIR}/spatial-rd-long-%A_%a.err"
)
if [[ -n "${RUNNER_GRES}" ]]; then
  SBATCH_RUNNER_ARGS+=(--gres="${RUNNER_GRES}")
fi

ARRAY_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" \
  EVAL_EVERY="${EVAL_EVERY}" \
  RESUME_FROM_LATEST="${RESUME_FROM_LATEST}" \
    sbatch \
      "${SBATCH_RUNNER_ARGS[@]}" \
      "${RUNNER_SCRIPT}"
)
ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"

SUPPORT_PARENT_JOB_ID=""
if [[ "${CHAIN_SUPPORT_SWEEP}" == "1" ]]; then
  SUPPORT_PARENT_JOB_ID=$(
    EXPERIMENT_TAG="${SUPPORT_EXPERIMENT_TAG}" \
    RESULTS_DIR="${SUPPORT_RESULTS_DIR}" \
    OUTPUT_ROOT="${SUPPORT_OUTPUT_ROOT}" \
    INPUT_ROOTS_CSV="${BASE_OUT}" \
    SUPPORT_THRESHOLDS_CSV="${SUPPORT_THRESHOLDS_CSV}" \
    FAMILY_JACCARDS_CSV="${SUPPORT_FAMILY_JACCARDS_CSV}" \
    DEEP_THRESHOLD="${DEEP_THRESHOLD}" \
    ARRAY_THROTTLE="${SUPPORT_ARRAY_THROTTLE}" \
    RUNNER_PARTITION="${RUNNER_PARTITION}" \
    RUNNER_GRES="${RUNNER_GRES}" \
    RUNNER_TIME="${SUPPORT_RUNNER_TIME}" \
    RUNNER_MEM="${SUPPORT_RUNNER_MEM}" \
    RUNNER_CPUS="${SUPPORT_RUNNER_CPUS}" \
      sbatch \
        --parsable \
        --dependency="afterok:${ARRAY_JOB_ID}" \
        scripts/queue_spatialized_reaction_diffusion_support_sweep_existing.sh
  )
  SUPPORT_PARENT_JOB_ID="${SUPPORT_PARENT_JOB_ID%%;*}"
fi

cat > "${RESULTS_DIR}/queue.json" <<EOF
{
  "experiment_tag": "${EXPERIMENT_TAG}",
  "results_dir": "${RESULTS_DIR}",
  "base_out": "${BASE_OUT}",
  "task_tsv": "${TASK_TSV}",
  "manifest_json": "${MANIFEST_JSON}",
  "log_dir": "${LOG_DIR}",
  "task_count": ${TASK_COUNT},
  "array_spec": "${ARRAY_SPEC}",
  "array_job_id": "${ARRAY_JOB_ID}",
  "support_parent_job_id": "${SUPPORT_PARENT_JOB_ID}",
  "support_results_dir": "${SUPPORT_RESULTS_DIR}",
  "support_output_root": "${SUPPORT_OUTPUT_ROOT}",
  "runner_partition": "${RUNNER_PARTITION}",
  "runner_gres": "${RUNNER_GRES}",
  "runner_time": "${RUNNER_TIME}",
  "runner_mem": "${RUNNER_MEM}",
  "runner_cpus": "${RUNNER_CPUS}",
  "conv_activation": "${CONV_ACTIVATION}",
  "lista_alpha_csv": "${LISTA_ALPHA_CSV}",
  "sparsity_coeff_csv": "${SPARSITY_COEFF_CSV}"
}
EOF

echo "Queued spatialized PDE long-horizon pilot."
echo "Task count: ${TASK_COUNT}"
echo "Training array job: ${ARRAY_JOB_ID}"
if [[ -n "${SUPPORT_PARENT_JOB_ID}" ]]; then
  echo "Support sweep parent job: ${SUPPORT_PARENT_JOB_ID}"
fi
echo "Results dir: ${RESULTS_DIR}"
echo "Base output: ${BASE_OUT}"
