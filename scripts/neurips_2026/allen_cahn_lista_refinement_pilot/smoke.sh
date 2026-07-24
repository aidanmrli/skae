#!/usr/bin/env bash
#SBATCH --job-name=smoke-ac-depth
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
ROOT_DIR="${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/common/cluster_env.sh"
SOURCE_DIR="${SKAE_REBUTTAL_SOURCE:-${SKAE_SCRATCH_ROOT}-rebuttal}"
OUTPUT_ROOT="${SKAE_SCRATCH_ROOT}/allen_cahn_lista_refinement_smoke_20260722"
RUN_DIR="${OUTPUT_ROOT}/model"
TELEMETRY="${OUTPUT_ROOT}/gpu_telemetry.csv"
mkdir -p "${OUTPUT_ROOT}"
cd "${SOURCE_DIR}"

PYTHONPATH="${SOURCE_DIR}:${PROJECT_DIR}" uv run --project "${PROJECT_DIR}" pytest \
  tests/test_spatialized_conv_koopman_refined.py -q
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit --format=csv --loop=2 > "${TELEMETRY}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

PYTHONPATH="${SOURCE_DIR}:${PROJECT_DIR}" uv run --project "${PROJECT_DIR}" python \
  tools/train_spatialized_reaction_diffusion_conv_refined.py \
  --dataset "${SKAE_SCRATCH_ROOT}/allen_cahn_rebuttal_v2_20260719/data/allen_cahn_4_grid16_dt0p1_t20_dev.pt" \
  --run_dir "${RUN_DIR}" \
  --model_variant conv_lista --seed 64002 --device cuda \
  --z_dim 2048 --hidden_channels 32 --num_blocks 2 \
  --conv_activation tanh --padding_mode circular \
  --lista_num_loops 2 --lista_alpha 0.15 \
  --num_steps 50 --pretrain_steps 50 --batch_size 8 --sequence_length 200 \
  --train_trajectory_limit 512 --lr 0.0003 --k_matrix_lr 0.000001 \
  --weight_decay 0 --k_init_scale 0 --prediction_weight 1 \
  --forecast_weighting uniform --reconstruction_weight 0.25 --latent_weight 0.1 \
  --sparsity_weight 0.01 --temporal_group_sparsity_weight 0 \
  --k_stability_weight 0 --gradient_weight 0.05 \
  --eval_every 50 --eval_horizon 200 --checkpoint_metric joint_endpoints \
  --checkpoint_horizons 160 200 --validation_partition even \
  --log_every 25 --spatial_augmentation

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT
awk -F, 'NR > 1 {gsub(/[^0-9.]/, "", $3); u=$3+0; sum+=u; n++; if(u>0){active+=u; an++}; if(u>peak)peak=u} END {printf "samples\t%d\nmean_all_gpu_utilization_percent\t%.3f\nmean_active_gpu_utilization_percent\t%.3f\npeak_gpu_utilization_percent\t%.1f\n", n, n?sum/n:0, an?active/an:0, peak}' "${TELEMETRY}" > "${OUTPUT_ROOT}/gpu_utilization_audit.tsv"
cat "${OUTPUT_ROOT}/gpu_utilization_audit.tsv"
