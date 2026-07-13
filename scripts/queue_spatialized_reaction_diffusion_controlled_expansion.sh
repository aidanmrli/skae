#!/usr/bin/env bash
#
# Queue the controlled spatialized multibasin PDE expansion.
#
# This launch avoids GPU wastage by:
#   1. generating/linking datasets in a CPU-only dependency,
#   2. running PACK_SIZE independent training rows on each allocated GPU, and
#   3. running support-threshold sweeps on CPU after training.
#
# Submit with:
#   sbatch scripts/queue_spatialized_reaction_diffusion_controlled_expansion.sh
#
# Default design:
#   5 systems x 5 seeds x 5 exact model settings = 125 training tasks.
#
#SBATCH --job-name=queue-spatial-rd-ctl
#SBATCH --output=/network/scratch/l/lia/skae/queue-spatial-rd-ctl-%A.out
#SBATCH --error=/network/scratch/l/lia/skae/queue-spatial-rd-ctl-%A.err
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

EXPERIMENT_TAG="${EXPERIMENT_TAG:-spatial_rd_controlled_expansion_20260602}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
DATA_TASK_TSV="${DATA_TASK_TSV:-${RESULTS_DIR}/spatialized_rd_dataset_tasks.tsv}"
TRAIN_TASK_TSV="${TRAIN_TASK_TSV:-${RESULTS_DIR}/spatialized_rd_training_tasks.tsv}"
MANIFEST_JSON="${MANIFEST_JSON:-${RESULTS_DIR}/spatialized_rd_manifest.json}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
GPU_MONITOR_DIR="${GPU_MONITOR_DIR:-${RESULTS_DIR}/gpu_monitor}"

SYSTEMS_CSV="${SYSTEMS_CSV:-cal_square_4,transition_routes_4,cal_high_cross_3,var_l_shape_5,cal_pentagon_5}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
RECIPES_CSV="${RECIPES_CSV:-conv_lista_relu_a0p03_sp0p05|conv_lista|relu|0.03|0.05,conv_lista_relu_a0p03_sp0p01|conv_lista|relu|0.03|0.01,conv_sparse_mlp_relu_a0p03_sp0p1|conv_sparse_mlp|relu|0.03|0.1,conv_dense_relu_nosparse|conv_dense|relu|0.03|0.0,conv_dense_tanh_nosparse|conv_dense||0.03|0.0}"

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
NUM_STEPS="${NUM_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS:-3}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-0.05}"
FAMILY_JACCARD="${FAMILY_JACCARD:-0.2}"
MAX_VALIDATION_REPS="${MAX_VALIDATION_REPS:-1024}"
DEEP_THRESHOLD="${DEEP_THRESHOLD:-0.7}"
EVAL_HORIZONS="${EVAL_HORIZONS:-1,4,8,16,32,64,128,256,512}"
EVAL_HORIZON="${EVAL_HORIZON:-32}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
RESUME_FROM_LATEST="${RESUME_FROM_LATEST:-1}"

DATA_ARRAY_THROTTLE="${DATA_ARRAY_THROTTLE:-8}"
DATA_RUNNER_PARTITION="${DATA_RUNNER_PARTITION:-long}"
DATA_RUNNER_TIME="${DATA_RUNNER_TIME:-01:00:00}"
DATA_RUNNER_MEM="${DATA_RUNNER_MEM:-12G}"
DATA_RUNNER_CPUS="${DATA_RUNNER_CPUS:-4}"
DATA_RUNNER_SCRIPT="${DATA_RUNNER_SCRIPT:-scripts/run_spatialized_reaction_diffusion_dataset_array.sh}"

PACK_SIZE="${PACK_SIZE:-8}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-4}"
RUNNER_PARTITION="${RUNNER_PARTITION:-long}"
RUNNER_GRES="${RUNNER_GRES-gpu:1}"
RUNNER_TIME="${RUNNER_TIME:-02:50:00}"
RUNNER_MEM="${RUNNER_MEM:-48G}"
RUNNER_CPUS="${RUNNER_CPUS:-12}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-scripts/run_spatialized_reaction_diffusion_packed_array.sh}"
GPU_MONITOR="${GPU_MONITOR:-1}"

CHAIN_SUPPORT_SWEEP="${CHAIN_SUPPORT_SWEEP:-1}"
SUPPORT_EXPERIMENT_TAG="${SUPPORT_EXPERIMENT_TAG:-${EXPERIMENT_TAG}_support_sweep}"
SUPPORT_RESULTS_DIR="${SUPPORT_RESULTS_DIR:-results/${SUPPORT_EXPERIMENT_TAG}}"
SUPPORT_OUTPUT_ROOT="${SUPPORT_OUTPUT_ROOT:-/network/scratch/l/lia/skae/${SUPPORT_EXPERIMENT_TAG}}"
SUPPORT_THRESHOLDS_CSV="${SUPPORT_THRESHOLDS_CSV:-0.01,0.03,0.05,0.1,0.2}"
SUPPORT_FAMILY_JACCARDS_CSV="${SUPPORT_FAMILY_JACCARDS_CSV:-0.2,0.25,0.3,0.4}"
SUPPORT_RUNNER_PARTITION="${SUPPORT_RUNNER_PARTITION:-long}"
SUPPORT_RUNNER_GRES="${SUPPORT_RUNNER_GRES:-}"
SUPPORT_RUNNER_TIME="${SUPPORT_RUNNER_TIME:-02:00:00}"
SUPPORT_RUNNER_MEM="${SUPPORT_RUNNER_MEM:-12G}"
SUPPORT_RUNNER_CPUS="${SUPPORT_RUNNER_CPUS:-4}"
SUPPORT_ARRAY_THROTTLE="${SUPPORT_ARRAY_THROTTLE:-12}"
SUPPORT_RUNNER_SCRIPT="${SUPPORT_RUNNER_SCRIPT:-scripts/run_spatialized_reaction_diffusion_support_sweep_array_cpu.sh}"

mkdir -p "${RESULTS_DIR}" "${BASE_OUT}" "${LOG_DIR}" "${GPU_MONITOR_DIR}"

echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse --short HEAD || true)"
echo "experiment_tag=${EXPERIMENT_TAG}"
echo "systems_csv=${SYSTEMS_CSV}"
echo "seeds_csv=${SEEDS_CSV}"
echo "recipes_csv=${RECIPES_CSV}"
echo "pack_size=${PACK_SIZE}"
echo "runner_gres=${RUNNER_GRES}"
echo "gpu_monitor_dir=${GPU_MONITOR_DIR}"

export BASE_OUT SYSTEMS_CSV SEEDS_CSV RECIPES_CSV GRID_SIZE DIFFUSION RK4_DT
export SUBSTEPS_PER_OBSERVATION TRAJECTORY_LENGTH TRAIN_OBSERVATION_LIMIT
export LABEL_EXTRA_OBSERVATIONS TRAIN_TRAJECTORIES VAL_TRAJECTORIES TEST_TRAJECTORIES
export LAPLACIAN_SCALING TARGET_SIZE MIN_LATENT_STATE_RATIO HIDDEN_CHANNELS NUM_BLOCKS
export NUM_STEPS BATCH_SIZE SEQUENCE_LENGTH LISTA_NUM_LOOPS SUPPORT_THRESHOLD FAMILY_JACCARD
export MAX_VALIDATION_REPS DEEP_THRESHOLD EVAL_HORIZONS EVAL_HORIZON

uv run python - "${DATA_TASK_TSV}" "${TRAIN_TASK_TSV}" "${MANIFEST_JSON}" <<'PY'
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


def parse_recipes(raw):
    recipes = []
    for item in parse_csv(raw):
        parts = item.split("|")
        if len(parts) != 5:
            raise SystemExit(
                "Each recipe must be label|model_variant|conv_activation|lista_alpha|sparsity_coeff; "
                f"got {item!r}."
            )
        label, model_variant, conv_activation, lista_alpha, sparsity_coeff = parts
        if model_variant not in {"conv_lista", "conv_sparse_mlp", "conv_dense"}:
            raise SystemExit(f"Unknown model variant in recipe {item!r}")
        recipes.append(
            {
                "label": tagify(label),
                "model_variant": model_variant,
                "conv_activation": conv_activation,
                "lista_alpha": float(lista_alpha),
                "sparsity_coeff": float(sparsity_coeff),
            }
        )
    return recipes


data_task_tsv = Path(sys.argv[1])
train_task_tsv = Path(sys.argv[2])
manifest_json = Path(sys.argv[3])

systems = parse_csv(os.environ["SYSTEMS_CSV"])
seeds = [int(seed) for seed in parse_csv(os.environ["SEEDS_CSV"])]
recipes = parse_recipes(os.environ["RECIPES_CSV"])
eval_horizons = [int(horizon) for horizon in parse_csv(os.environ["EVAL_HORIZONS"])]
trajectory_length = int(os.environ["TRAJECTORY_LENGTH"])
too_long = [horizon for horizon in eval_horizons if horizon > trajectory_length]
if too_long:
    raise SystemExit(
        f"Requested eval_horizons exceed trajectory_length={trajectory_length}: {too_long}"
    )

grid_size = int(os.environ["GRID_SIZE"])
state_dim = 2 * grid_size * grid_size
min_target_size = math.ceil(float(os.environ["MIN_LATENT_STATE_RATIO"]) * state_dim)
target_size = int(os.environ["TARGET_SIZE"])
if target_size < min_target_size:
    raise SystemExit(
        f"TARGET_SIZE={target_size} is below required overcomplete minimum {min_target_size} "
        f"for state_dim={state_dim}."
    )

base_out = Path(os.environ["BASE_OUT"])
grid_slug = f"grid{grid_size}"
model_variants = sorted({recipe["model_variant"] for recipe in recipes})

data_rows = []
for task_id, (system, seed) in enumerate(
    (system, seed) for system in systems for seed in seeds
):
    system_slug = tagify(system)
    central_dataset = (
        base_out / "datasets" / system_slug / grid_slug / f"seed_{seed}" / "dataset.pt"
    )
    link_paths = [
        base_out / "runs" / model_variant / system_slug / grid_slug / f"seed_{seed}" / "dataset.pt"
        for model_variant in model_variants
    ]
    data_rows.append(
        {
            "task_id": task_id,
            "source_system": system,
            "seed": seed,
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
            "dataset_path": str(central_dataset),
            "link_paths_csv": ",".join(str(path) for path in link_paths),
        }
    )

train_rows = []
task_id = 0
for recipe in recipes:
    recipe_slug = recipe["label"]
    for system in systems:
        system_slug = tagify(system)
        for seed in seeds:
            model_variant = recipe["model_variant"]
            dataset_root = base_out / "runs" / model_variant / system_slug / grid_slug / f"seed_{seed}"
            task_root = dataset_root / recipe_slug
            train_rows.append(
                {
                    "task_id": task_id,
                    "source_system": system,
                    "seed": seed,
                    "model_variant": model_variant,
                    "trainer": "conv",
                    "config_name": "lista_parity_generic_sparse",
                    "setting_slug": recipe_slug,
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
                    "conv_activation": recipe["conv_activation"],
                    "num_steps": os.environ["NUM_STEPS"],
                    "batch_size": os.environ["BATCH_SIZE"],
                    "sequence_length": os.environ["SEQUENCE_LENGTH"],
                    "train_observation_limit": os.environ["TRAIN_OBSERVATION_LIMIT"],
                    "lista_num_loops": os.environ["LISTA_NUM_LOOPS"],
                    "lista_alpha": recipe["lista_alpha"],
                    "sparsity_coeff": recipe["sparsity_coeff"],
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

for path, rows in ((data_task_tsv, data_rows), (train_task_tsv, train_rows)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

manifest_json.parent.mkdir(parents=True, exist_ok=True)
manifest_json.write_text(
    json.dumps(
        {
            "num_dataset_tasks": len(data_rows),
            "num_training_tasks": len(train_rows),
            "systems": systems,
            "seeds": seeds,
            "recipes": recipes,
            "grid_size": grid_size,
            "state_dim": state_dim,
            "target_size": target_size,
            "latent_state_ratio": target_size / state_dim,
            "min_latent_state_ratio": float(os.environ["MIN_LATENT_STATE_RATIO"]),
            "trajectory_length": trajectory_length,
            "train_observation_limit": int(os.environ["TRAIN_OBSERVATION_LIMIT"]),
            "label_extra_observations": int(os.environ["LABEL_EXTRA_OBSERVATIONS"]),
            "sequence_length": int(os.environ["SEQUENCE_LENGTH"]),
            "eval_horizons": eval_horizons,
            "gpu_wastage_controls": [
                "datasets are generated before GPU allocation",
                "training array runs multiple rows per allocated GPU",
                "support sweeps run on CPU",
                "packed GPU jobs log nvidia-smi utilization",
            ],
            "label_policy": "Basin labels and attractor centers are stored for evaluation only.",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(f"Wrote {len(data_rows)} dataset tasks to {data_task_tsv}")
print(f"Wrote {len(train_rows)} training tasks to {train_task_tsv}")
PY

DATA_TASK_COUNT=$(( $(wc -l < "${DATA_TASK_TSV}") - 1 ))
TRAIN_TASK_COUNT=$(( $(wc -l < "${TRAIN_TASK_TSV}") - 1 ))
if [[ "${DATA_TASK_COUNT}" -le 0 || "${TRAIN_TASK_COUNT}" -le 0 ]]; then
  echo "No tasks generated." >&2
  exit 1
fi

PACKED_TASK_COUNT=$(( (TRAIN_TASK_COUNT + PACK_SIZE - 1) / PACK_SIZE ))
DATA_ARRAY_SPEC="0-$((DATA_TASK_COUNT - 1))%${DATA_ARRAY_THROTTLE}"
TRAIN_ARRAY_SPEC="0-$((PACKED_TASK_COUNT - 1))%${ARRAY_THROTTLE}"

DATA_JOB_ID=$(
  TASK_TSV="${DATA_TASK_TSV}" \
    sbatch \
      --parsable \
      --array="${DATA_ARRAY_SPEC}" \
      --partition="${DATA_RUNNER_PARTITION}" \
      --time="${DATA_RUNNER_TIME}" \
      --mem="${DATA_RUNNER_MEM}" \
      --cpus-per-task="${DATA_RUNNER_CPUS}" \
      --output="${LOG_DIR}/spatial-rd-data-%A_%a.out" \
      --error="${LOG_DIR}/spatial-rd-data-%A_%a.err" \
      "${DATA_RUNNER_SCRIPT}"
)
DATA_JOB_ID="${DATA_JOB_ID%%;*}"

SBATCH_RUNNER_ARGS=(
  --parsable
  --dependency="afterok:${DATA_JOB_ID}"
  --array="${TRAIN_ARRAY_SPEC}"
  --partition="${RUNNER_PARTITION}"
  --time="${RUNNER_TIME}"
  --mem="${RUNNER_MEM}"
  --cpus-per-task="${RUNNER_CPUS}"
  --output="${LOG_DIR}/spatial-rd-pack-%A_%a.out"
  --error="${LOG_DIR}/spatial-rd-pack-%A_%a.err"
)
if [[ -n "${RUNNER_GRES}" ]]; then
  SBATCH_RUNNER_ARGS+=(--gres="${RUNNER_GRES}")
fi

TRAIN_JOB_ID=$(
  TASK_TSV="${TRAIN_TASK_TSV}" \
  PACK_SIZE="${PACK_SIZE}" \
  EVAL_EVERY="${EVAL_EVERY}" \
  RESUME_FROM_LATEST="${RESUME_FROM_LATEST}" \
  GPU_MONITOR="${GPU_MONITOR}" \
  GPU_MONITOR_DIR="${GPU_MONITOR_DIR}" \
    sbatch \
      "${SBATCH_RUNNER_ARGS[@]}" \
      "${RUNNER_SCRIPT}"
)
TRAIN_JOB_ID="${TRAIN_JOB_ID%%;*}"

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
    RUNNER_PARTITION="${SUPPORT_RUNNER_PARTITION}" \
    RUNNER_GRES="${SUPPORT_RUNNER_GRES}" \
    RUNNER_TIME="${SUPPORT_RUNNER_TIME}" \
    RUNNER_MEM="${SUPPORT_RUNNER_MEM}" \
    RUNNER_CPUS="${SUPPORT_RUNNER_CPUS}" \
    RUNNER_SCRIPT="${SUPPORT_RUNNER_SCRIPT}" \
      sbatch \
        --parsable \
        --dependency="afterok:${TRAIN_JOB_ID}" \
        scripts/queue_spatialized_reaction_diffusion_support_sweep_existing.sh
  )
  SUPPORT_PARENT_JOB_ID="${SUPPORT_PARENT_JOB_ID%%;*}"
fi

cat > "${RESULTS_DIR}/queue.json" <<EOF
{
  "experiment_tag": "${EXPERIMENT_TAG}",
  "results_dir": "${RESULTS_DIR}",
  "base_out": "${BASE_OUT}",
  "data_task_tsv": "${DATA_TASK_TSV}",
  "train_task_tsv": "${TRAIN_TASK_TSV}",
  "manifest_json": "${MANIFEST_JSON}",
  "log_dir": "${LOG_DIR}",
  "gpu_monitor_dir": "${GPU_MONITOR_DIR}",
  "data_task_count": ${DATA_TASK_COUNT},
  "train_task_count": ${TRAIN_TASK_COUNT},
  "packed_task_count": ${PACKED_TASK_COUNT},
  "pack_size": ${PACK_SIZE},
  "data_array_spec": "${DATA_ARRAY_SPEC}",
  "train_array_spec": "${TRAIN_ARRAY_SPEC}",
  "data_job_id": "${DATA_JOB_ID}",
  "train_job_id": "${TRAIN_JOB_ID}",
  "support_parent_job_id": "${SUPPORT_PARENT_JOB_ID}",
  "support_results_dir": "${SUPPORT_RESULTS_DIR}",
  "support_output_root": "${SUPPORT_OUTPUT_ROOT}",
  "runner_partition": "${RUNNER_PARTITION}",
  "runner_gres": "${RUNNER_GRES}",
  "runner_time": "${RUNNER_TIME}",
  "runner_mem": "${RUNNER_MEM}",
  "runner_cpus": "${RUNNER_CPUS}",
  "support_runner_script": "${SUPPORT_RUNNER_SCRIPT}",
  "support_runner_gres": "${SUPPORT_RUNNER_GRES}"
}
EOF

echo "Queued controlled spatialized PDE expansion."
echo "Dataset tasks: ${DATA_TASK_COUNT}; job ${DATA_JOB_ID}; array ${DATA_ARRAY_SPEC}"
echo "Training tasks: ${TRAIN_TASK_COUNT}; packed jobs ${PACKED_TASK_COUNT}; job ${TRAIN_JOB_ID}; array ${TRAIN_ARRAY_SPEC}"
echo "Pack size: ${PACK_SIZE}; GPU monitor dir: ${GPU_MONITOR_DIR}"
if [[ -n "${SUPPORT_PARENT_JOB_ID}" ]]; then
  echo "Support sweep parent job: ${SUPPORT_PARENT_JOB_ID}"
fi
echo "Results dir: ${RESULTS_DIR}"
echo "Base output: ${BASE_OUT}"
