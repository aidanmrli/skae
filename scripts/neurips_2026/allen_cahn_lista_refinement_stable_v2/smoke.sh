#!/usr/bin/env bash
#SBATCH --job-name=smoke-ac-lista-v3b48
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
ROOT_DIR="${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/common/cluster_env.sh"
PINNED_SOURCE_DIR="/network/scratch/l/lia/skae-rebuttal"
SOURCE_DIR="${SKAE_REBUTTAL_SOURCE:-${SKAE_SCRATCH_ROOT}-rebuttal}"
SOURCE_DIR="$(readlink -f "${SOURCE_DIR}")"
[[ "${SOURCE_DIR}" == "${PINNED_SOURCE_DIR}" ]] || {
  echo "Resolved source ${SOURCE_DIR} does not match manifest-pinned ${PINNED_SOURCE_DIR}" >&2
  exit 1
}
OUTPUT_ROOT="${SKAE_SCRATCH_ROOT}/allen_cahn_lista_refinement_stable_smoke_20260722_v3_b48"
DATASET="${SKAE_SCRATCH_ROOT}/allen_cahn_rebuttal_v2_20260719/data/allen_cahn_4_grid16_dt0p1_t20_dev.pt"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing to overwrite ${OUTPUT_ROOT}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_DIR}"
sha256sum -c experiments/neurips_2026/allen_cahn_lista_refinement_stable_v2/source_manifest.sha256
PYTHONPATH="${SOURCE_DIR}:${PROJECT_DIR}" uv run --project "${PROJECT_DIR}" pytest \
  "${SOURCE_DIR}/tests/test_spatialized_conv_koopman_refined.py" \
  "${SOURCE_DIR}/tests/test_spatialized_conv_koopman_refined_stable.py" \
  tests/test_allen_cahn_lista_refinement_stable.py \
  tests/test_allen_cahn_lista_refinement_stable_v2.py -q

COMMON=(--dataset "${DATASET}" --model_variant conv_lista --seed 64002 --device cuda \
  --z_dim 2048 --hidden_channels 32 --num_blocks 2 --conv_activation tanh \
  --padding_mode circular --lista_alpha 0.15 --sequence_length 200 \
  --train_trajectory_limit 512 --lr 0.0003 --k_matrix_lr 0.000001 \
  --weight_decay 0 --k_init_scale 0 --prediction_weight 1 \
  --forecast_weighting uniform --reconstruction_weight 0.25 --latent_weight 0.1 \
  --sparsity_weight 0.01 --temporal_group_sparsity_weight 0 \
  --k_stability_weight 0 --gradient_weight 0.05 --eval_every 250 \
  --eval_horizon 200 --checkpoint_metric joint_endpoints \
  --checkpoint_horizons 160 200 --validation_partition even --log_every 100 \
  --spatial_augmentation)

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit --format=csv --loop=10 > "${OUTPUT_ROOT}/gpu_telemetry.csv" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT
SHARED="${OUTPUT_ROOT}/shared_pretrain/model"
mkdir -p "${SHARED}"
LISTA_S_LR=0.00001 FREEZE_LISTA_S_PRETRAIN=1 \
PYTHONPATH="${SOURCE_DIR}:${PROJECT_DIR}" uv run --project "${PROJECT_DIR}" python \
  "${SOURCE_DIR}/tools/train_spatialized_reaction_diffusion_conv_refined_stable.py" \
  "${COMMON[@]}" --run_dir "${SHARED}" --lista_num_loops 0 \
  --num_steps 0 --pretrain_steps 2000 --batch_size 32 \
  > "${OUTPUT_ROOT}/shared_pretrain.log" 2>&1

train_branch() {
  local depth="$1"; local run_dir="${OUTPUT_ROOT}/refinements_${depth}/model"
  mkdir -p "${run_dir}"
  LISTA_S_LR=0.00001 FREEZE_LISTA_S_PRETRAIN=1 \
  PYTHONPATH="${SOURCE_DIR}:${PROJECT_DIR}" uv run --project "${PROJECT_DIR}" python \
    "${SOURCE_DIR}/tools/train_spatialized_reaction_diffusion_conv_refined_stable.py" \
    "${COMMON[@]}" --run_dir "${run_dir}" --lista_num_loops "${depth}" \
    --num_steps 500 --pretrain_steps 0 --batch_size 48 \
    --warm_start_pretrain_checkpoint "${SHARED}/checkpoint.pt" \
    --warm_start_pretrain_steps 2000
}
PIDS=()
for DEPTH in 0 2 3; do train_branch "${DEPTH}" > "${OUTPUT_ROOT}/refinements_${DEPTH}.log" 2>&1 & PIDS+=("$!"); done
FAILED=0; for PID in "${PIDS[@]}"; do wait "${PID}" || FAILED=1; done
(( FAILED == 0 )) || { echo "Shared-pretrain smoke branch failed" >&2; exit 1; }
kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; trap - EXIT

awk -F, 'NR>2 {gsub(/[^0-9.]/,"",$3); u=$3+0; s+=u; n++; if(u>0){a+=u; an++}; if(u>p)p=u} END{if(n<6 || s/n<70 || (an?a/an:0)<80 || p<95) exit 1; printf "samples\t%d\nmean_all_gpu_utilization_percent\t%.3f\nmean_active_gpu_utilization_percent\t%.3f\npeak_gpu_utilization_percent\t%.1f\n",n,s/n,an?a/an:0,p}' \
  "${OUTPUT_ROOT}/gpu_telemetry.csv" > "${OUTPUT_ROOT}/gpu_utilization_gate.tsv"
uv run python -m experiments.neurips_2026.allen_cahn_lista_refinement_stable.validate_smoke \
  --run_dir "${OUTPUT_ROOT}/refinements_0/model" \
  --run_dir "${OUTPUT_ROOT}/refinements_2/model" \
  --run_dir "${OUTPUT_ROOT}/refinements_3/model" \
  --output "${OUTPUT_ROOT}/stability_gate.json"
