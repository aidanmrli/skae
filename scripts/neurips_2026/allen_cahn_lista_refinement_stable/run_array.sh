#!/usr/bin/env bash
#SBATCH --job-name=ac-lista-stable
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-1%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
PROJECT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
ROOT_DIR="${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/common/cluster_env.sh"
SOURCE_DIR="${SKAE_REBUTTAL_SOURCE:-${SKAE_SCRATCH_ROOT}-rebuttal}"
OUTPUT_ROOT="${SKAE_SCRATCH_ROOT}/allen_cahn_lista_refinement_stable_20260722_v1"
SMOKE_GATE="${SKAE_SCRATCH_ROOT}/allen_cahn_lista_refinement_stable_smoke_20260722_v1/stability_gate.json"
TRAIN_DATASET="${SKAE_SCRATCH_ROOT}/allen_cahn_rebuttal_v2_20260719/data/allen_cahn_4_grid16_dt0p1_t20_dev.pt"
VALIDATION_ROOT="${SKAE_SCRATCH_ROOT}/allen_cahn_periodic_reencoding_confirmation_20260721_v5/data"
REFINEMENTS=$((2 + SLURM_ARRAY_TASK_ID))
DEPTH_ROOT="${OUTPUT_ROOT}/refinements_${REFINEMENTS}"
[[ -f "${SMOKE_GATE}" ]] || { echo "Missing passed smoke gate" >&2; exit 1; }
[[ ! -e "${DEPTH_ROOT}" ]] || { echo "Refusing to overwrite ${DEPTH_ROOT}" >&2; exit 1; }
mkdir -p "${DEPTH_ROOT}"
cd "${PROJECT_DIR}"
sha256sum -c experiments/neurips_2026/allen_cahn_lista_refinement_stable/source_manifest.sha256
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit --format=csv --loop=60 > "${DEPTH_ROOT}/gpu_telemetry.csv" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

train_seed() {
  local seed="$1"
  local run_dir="${DEPTH_ROOT}/seed_${seed}/model"
  mkdir -p "${run_dir}"
  LISTA_S_LR=0.00001 FREEZE_LISTA_S_PRETRAIN=1 \
  PYTHONPATH="${SOURCE_DIR}:${PROJECT_DIR}" uv run --project "${PROJECT_DIR}" python \
    "${SOURCE_DIR}/tools/train_spatialized_reaction_diffusion_conv_refined_stable.py" \
    --dataset "${TRAIN_DATASET}" --run_dir "${run_dir}" \
    --model_variant conv_lista --seed "${seed}" --device cuda \
    --z_dim 2048 --hidden_channels 32 --num_blocks 2 \
    --conv_activation tanh --padding_mode circular \
    --lista_num_loops "${REFINEMENTS}" --lista_alpha 0.15 \
    --num_steps 3500 --pretrain_steps 2000 --batch_size 8 --sequence_length 200 \
    --train_trajectory_limit 512 --lr 0.0003 --k_matrix_lr 0.000001 \
    --weight_decay 0 --k_init_scale 0 --prediction_weight 1 \
    --forecast_weighting uniform --reconstruction_weight 0.25 --latent_weight 0.1 \
    --sparsity_weight 0.01 --temporal_group_sparsity_weight 0 \
    --k_stability_weight 0 --gradient_weight 0.05 \
    --eval_every 250 --eval_horizon 200 --checkpoint_metric joint_endpoints \
    --checkpoint_horizons 160 200 --validation_partition even \
    --log_every 100 --spatial_augmentation
}

PIDS=()
for SEED in 64 65; do
  train_seed "${SEED}" > "${DEPTH_ROOT}/seed_${SEED}.train.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for PID in "${PIDS[@]}"; do wait "${PID}" || FAILED=1; done
(( FAILED == 0 )) || { echo "At least one stable training run failed" >&2; exit 1; }

for SEED in 64 65; do
  RUN_ROOT="${DEPTH_ROOT}/seed_${SEED}"
  PYTHONPATH="${PROJECT_DIR}:${SOURCE_DIR}" uv run --project "${PROJECT_DIR}" python \
    -m experiments.neurips_2026.allen_cahn_lista_refinement_pilot.evaluate \
    --checkpoint "${RUN_ROOT}/model/checkpoint.pt" \
    --output "${RUN_ROOT}/validation.json" --batch_size 768 \
    --periods 1 2 5 10 20 25 50 100 \
    --field 1089785385 "${VALIDATION_ROOT}/validation_seed1089785385_fields.pt" 9006f6537025e071f97a532bc7d81969d5c937770451a87a3448f29e1eca528a \
    --field 114962715 "${VALIDATION_ROOT}/validation_seed114962715_fields.pt" 3c85c388c22d8d50debdceaae73acd889e1dd8313ea0d18dd275588d876995ce \
    --field 61202961 "${VALIDATION_ROOT}/validation_seed61202961_fields.pt" 05a74ff1e1a781dd3dc870d3a5c1bd000b412a72a915a459ef720b50a9030f08
done
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT
awk -F, 'NR>1 {gsub(/[^0-9.]/,"",$3); u=$3+0; s+=u; n++; if(u>0){a+=u; an++}; if(u>p)p=u} END{printf "samples\t%d\nmean_all_gpu_utilization_percent\t%.3f\nmean_active_gpu_utilization_percent\t%.3f\npeak_gpu_utilization_percent\t%.1f\n",n,n?s/n:0,an?a/an:0,p}' "${DEPTH_ROOT}/gpu_telemetry.csv" > "${DEPTH_ROOT}/gpu_utilization_audit.tsv"
