#!/bin/bash
#
# Generic SLURM array runner for the research-paper benchmark task TSVs.
#
# Required env vars:
#   TASK_TSV=<path>
#   BASE_OUT=<output root>
#
# Optional:
#   ARRAY_OFFSET=0
#
#SBATCH --job-name=paper_bench
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/paper-bench-%A_%a.out
#SBATCH -e /network/scratch/l/lia/skae/paper-bench-%A_%a.err
#SBATCH --requeue

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source scripts/slurm_gpu_guard.sh
trap gpu_guard_stop_sampler EXIT
source .venv/bin/activate

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"

tagify() {
  local raw="$1"
  raw="${raw//-/m}"
  raw="${raw//./p}"
  echo "${raw}"
}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
LINE_NO=$((TASK_ID + ARRAY_OFFSET + 2))
TASK_LINE="$(sed -n "${LINE_NO}p" "${TASK_TSV}" || true)"

if [[ -z "${TASK_LINE}" ]]; then
  echo "No task row for array index ${TASK_ID} (line ${LINE_NO}) in ${TASK_TSV}. Exiting."
  exit 0
fi

TASK_EXPORTS="$(
  uv run python - "${TASK_TSV}" "${LINE_NO}" <<'PY'
import csv
import shlex
import sys

path = sys.argv[1]
line_no = int(sys.argv[2])

with open(path, newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for current_line_no, row in enumerate(reader, start=2):
        if current_line_no != line_no:
            continue
        for key, value in row.items():
            print(f"{key}={shlex.quote(value or '')}")
        break
    else:
        sys.exit(3)
PY
)"
eval "${TASK_EXPORTS}"

DT_TAG="$(tagify "${env_dt}")"
LOG_ROOT="${BASE_OUT}/${phase}/${model_variant}"
if [[ -n "${kuramoto_num_oscillators:-}" ]]; then
  LOG_ROOT="${LOG_ROOT}/n_${kuramoto_num_oscillators}"
fi
if [[ -n "${hopfield_num_neurons:-}" ]]; then
  LOG_ROOT="${LOG_ROOT}/n_${hopfield_num_neurons}"
fi
if [[ -n "${competitive_lv_num_species:-}" ]]; then
  LOG_ROOT="${LOG_ROOT}/n_${competitive_lv_num_species}"
fi
LOG_DIR="${LOG_ROOT}/${system_slug}/dt_${DT_TAG}/seed_${seed}"
mkdir -p "${LOG_DIR}"

COMPLETED_RUN=""
if [[ "${SKIP_COMPLETED:-1}" == "1" ]]; then
  COMPLETED_RUN="$(
    find "${LOG_DIR}" -mindepth 1 -maxdepth 1 -type d \
      -name '20*' -exec test -f '{}/evaluation_summary.json' ';' -print \
      | sort | tail -n 1
  )"
fi

RESUME_CHECKPOINT=""
if [[ -z "${COMPLETED_RUN}" && "${RESUME_FROM_LATEST:-1}" == "1" ]]; then
  RESUME_CHECKPOINT="$(
    find "${LOG_DIR}" -mindepth 2 -maxdepth 2 -type f -name 'last.pt' \
      -printf '%T@ %p\n' 2>/dev/null \
      | sort -n | tail -n 1 | cut -d' ' -f2-
  )"
fi

echo "============================================="
echo "Paper Benchmark Array Runner"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "Task Row: ${task_id}"
echo "Phase: ${phase}"
echo "Model Variant: ${model_variant}"
echo "System: ${system_key}"
echo "Env: ${env_name}"
echo "Seed: ${seed}"
echo "DT: ${env_dt}"
if [[ -n "${kuramoto_num_oscillators:-}" ]]; then
  echo "KURAMOTO_NUM_OSCILLATORS: ${kuramoto_num_oscillators}"
fi
if [[ -n "${kuramoto_topology:-}" ]]; then
  echo "KURAMOTO_TOPOLOGY: ${kuramoto_topology}"
fi
if [[ -n "${kuramoto_omega_mode:-}" ]]; then
  echo "KURAMOTO_OMEGA_MODE: ${kuramoto_omega_mode}"
fi
if [[ -n "${kuramoto_omega_spread:-}" ]]; then
  echo "KURAMOTO_OMEGA_SPREAD: ${kuramoto_omega_spread}"
fi
if [[ -n "${hopfield_num_neurons:-}" ]]; then
  echo "HOPFIELD_NUM_NEURONS: ${hopfield_num_neurons}"
fi
if [[ -n "${hopfield_num_patterns:-}" ]]; then
  echo "HOPFIELD_NUM_PATTERNS: ${hopfield_num_patterns}"
fi
if [[ -n "${competitive_lv_num_species:-}" ]]; then
  echo "COMPETITIVE_LV_NUM_SPECIES: ${competitive_lv_num_species}"
fi
if [[ -n "${competitive_lv_interaction_scale:-}" ]]; then
  echo "COMPETITIVE_LV_INTERACTION_SCALE: ${competitive_lv_interaction_scale}"
fi
if [[ -n "${competitive_lv_system_seed:-}" ]]; then
  echo "COMPETITIVE_LV_SYSTEM_SEED: ${competitive_lv_system_seed}"
fi
echo "LOG_DIR: ${LOG_DIR}"
if [[ -n "${COMPLETED_RUN}" ]]; then
  echo "Completed run already exists: ${COMPLETED_RUN}"
  echo "Skipping completed task."
  exit 0
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  echo "Resuming from latest checkpoint: ${RESUME_CHECKPOINT}"
fi
module load cuda/12.6.0
gpu_guard_assert_cuda_visible "paper benchmark task ${task_id}"
gpu_guard_print_context "Paper Benchmark Array Runner"
echo "Start Time: $(date)"
echo "============================================="

TRAIN_ARGS=(
  --config "${config_name}"
  --env "${env_name}"
  --env_dt "${env_dt}"
  --num_steps "${num_steps}"
  --batch_size "${batch_size}"
  --target_size "${target_size}"
  --res_coeff "${res_coeff}"
  --reconst_coeff "${reconst_coeff}"
  --pred_coeff "${pred_coeff}"
  --sparsity_coeff "${sparsity_coeff}"
  --sequence_length "${sequence_length}"
  --eval_profile "${eval_profile}"
  --seed "${seed}"
  --device cuda
  --log_dir "${LOG_DIR}"
)

if [[ -n "${sparsity_target:-}" ]]; then
  TRAIN_ARGS+=(--sparsity_target "${sparsity_target}")
fi
if [[ -n "${eval_every:-}" ]]; then
  TRAIN_ARGS+=(--eval_every "${eval_every}")
fi
if [[ -n "${eval_num_steps:-}" ]]; then
  TRAIN_ARGS+=(--eval_num_steps "${eval_num_steps}")
fi
if [[ "${skip_basin_eval:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--skip_basin_eval)
fi
if [[ -n "${lista_alpha:-}" ]]; then
  TRAIN_ARGS+=(--lista_alpha "${lista_alpha}")
fi
if [[ -n "${hard_init_oversample:-}" ]]; then
  TRAIN_ARGS+=(--hard_init_oversample "${hard_init_oversample}")
fi
if [[ -n "${hard_init_fraction:-}" ]]; then
  TRAIN_ARGS+=(--hard_init_fraction "${hard_init_fraction}")
fi
if [[ -n "${hard_init_pool_size:-}" ]]; then
  TRAIN_ARGS+=(--hard_init_pool_size "${hard_init_pool_size}")
fi
if [[ -n "${hard_init_num_candidates:-}" ]]; then
  TRAIN_ARGS+=(--hard_init_num_candidates "${hard_init_num_candidates}")
fi
if [[ -n "${hard_init_probe_steps:-}" ]]; then
  TRAIN_ARGS+=(--hard_init_probe_steps "${hard_init_probe_steps}")
fi
if [[ -n "${hard_init_num_perturbations:-}" ]]; then
  TRAIN_ARGS+=(--hard_init_num_perturbations "${hard_init_num_perturbations}")
fi
if [[ -n "${hard_init_perturb_scale:-}" ]]; then
  TRAIN_ARGS+=(--hard_init_perturb_scale "${hard_init_perturb_scale}")
fi
if [[ -n "${hard_init_transient_window:-}" ]]; then
  TRAIN_ARGS+=(--hard_init_transient_window "${hard_init_transient_window}")
fi
if [[ -n "${hard_init_transient_weight:-}" ]]; then
  TRAIN_ARGS+=(--hard_init_transient_weight "${hard_init_transient_weight}")
fi
if [[ -n "${hard_init_jitter_scale:-}" ]]; then
  TRAIN_ARGS+=(--hard_init_jitter_scale "${hard_init_jitter_scale}")
fi
if [[ -n "${lista_num_loops:-}" ]]; then
  TRAIN_ARGS+=(--lista_num_loops "${lista_num_loops}")
fi
if [[ -n "${lista_use_momentum:-}" ]]; then
  TRAIN_ARGS+=(--lista_use_momentum "${lista_use_momentum}")
fi
if [[ -n "${lista_momentum_beta:-}" ]]; then
  TRAIN_ARGS+=(--lista_momentum_beta "${lista_momentum_beta}")
fi
if [[ -n "${lista_linear_encoder:-}" ]]; then
  TRAIN_ARGS+=(--lista_linear_encoder "${lista_linear_encoder}")
fi
if [[ -n "${lista_final_op:-}" ]]; then
  TRAIN_ARGS+=(--lista_final_op "${lista_final_op}")
fi
if [[ -n "${lista_precode_mode:-}" ]]; then
  TRAIN_ARGS+=(--lista_precode_mode "${lista_precode_mode}")
fi
if [[ -n "${lista_precode_residual_scale:-}" ]]; then
  TRAIN_ARGS+=(--lista_precode_residual_scale "${lista_precode_residual_scale}")
fi
if [[ -n "${lista_adaptive_thresholds:-}" ]]; then
  TRAIN_ARGS+=(--lista_adaptive_thresholds "${lista_adaptive_thresholds}")
fi
if [[ -n "${lista_alpha_residual_coeff:-}" ]]; then
  TRAIN_ARGS+=(--lista_alpha_residual_coeff "${lista_alpha_residual_coeff}")
fi
if [[ -n "${lista_alpha_prior_coeff:-}" ]]; then
  TRAIN_ARGS+=(--lista_alpha_prior_coeff "${lista_alpha_prior_coeff}")
fi
if [[ -n "${lista_groupwise_thresholds:-}" ]]; then
  TRAIN_ARGS+=(--lista_groupwise_thresholds "${lista_groupwise_thresholds}")
fi
if [[ -n "${encoder_group_shrinkage:-}" ]]; then
  TRAIN_ARGS+=(--encoder_group_shrinkage "${encoder_group_shrinkage}")
fi
if [[ -n "${encoder_group_threshold_scale:-}" ]]; then
  TRAIN_ARGS+=(--encoder_group_threshold_scale "${encoder_group_threshold_scale}")
fi
if [[ -n "${encoder_topk_groups:-}" ]]; then
  TRAIN_ARGS+=(--encoder_topk_groups "${encoder_topk_groups}")
fi
if [[ -n "${decoder_coherence_weight:-}" ]]; then
  TRAIN_ARGS+=(--decoder_coherence_weight "${decoder_coherence_weight}")
fi
if [[ -n "${normalize_decoder_atoms:-}" ]]; then
  TRAIN_ARGS+=(--normalize_decoder_atoms "${normalize_decoder_atoms}")
fi
if [[ -n "${k_structure:-}" ]]; then
  TRAIN_ARGS+=(--k_structure "${k_structure}")
fi
if [[ -n "${k_block_size:-}" ]]; then
  TRAIN_ARGS+=(--k_block_size "${k_block_size}")
fi
if [[ -n "${k_num_blocks:-}" ]]; then
  TRAIN_ARGS+=(--k_num_blocks "${k_num_blocks}")
fi
if [[ "${block_loss:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--block_loss)
fi
if [[ -n "${block_one_block_loss:-}" ]]; then
  TRAIN_ARGS+=(--block_one_block_loss "${block_one_block_loss}")
fi
if [[ -n "${block_one_block_weight:-}" ]]; then
  TRAIN_ARGS+=(--block_one_block_weight "${block_one_block_weight}")
fi
if [[ -n "${block_top1_margin:-}" ]]; then
  TRAIN_ARGS+=(--block_top1_margin "${block_top1_margin}")
fi
if [[ -n "${block_balance_loss:-}" ]]; then
  TRAIN_ARGS+=(--block_balance_loss "${block_balance_loss}")
fi
if [[ -n "${block_balance_weight:-}" ]]; then
  TRAIN_ARGS+=(--block_balance_weight "${block_balance_weight}")
fi
if [[ -n "${block_energy_norm:-}" ]]; then
  TRAIN_ARGS+=(--block_energy_norm "${block_energy_norm}")
fi
if [[ -n "${hyperlista_c_theta:-}" ]]; then
  TRAIN_ARGS+=(--hyperlista_c_theta "${hyperlista_c_theta}")
fi
if [[ -n "${hyperlista_c_beta:-}" ]]; then
  TRAIN_ARGS+=(--hyperlista_c_beta "${hyperlista_c_beta}")
fi
if [[ -n "${hyperlista_c_ss:-}" ]]; then
  TRAIN_ARGS+=(--hyperlista_c_ss "${hyperlista_c_ss}")
fi
if [[ -n "${hyperlista_step_scale:-}" ]]; then
  TRAIN_ARGS+=(--hyperlista_step_scale "${hyperlista_step_scale}")
fi
if [[ -n "${hyperlista_use_ss:-}" ]]; then
  TRAIN_ARGS+=(--hyperlista_use_ss "${hyperlista_use_ss}")
fi
if [[ -n "${hyperlista_use_momentum:-}" ]]; then
  TRAIN_ARGS+=(--hyperlista_use_momentum "${hyperlista_use_momentum}")
fi
if [[ -n "${eval_use_dynamics_prior:-}" ]]; then
  TRAIN_ARGS+=(--eval_use_dynamics_prior "${eval_use_dynamics_prior}")
fi
if [[ -n "${eval_event_trigger_proj_threshold:-}" ]]; then
  TRAIN_ARGS+=(--eval_event_trigger_proj_threshold "${eval_event_trigger_proj_threshold}")
fi
if [[ -n "${eval_event_trigger_ambiguity_threshold:-}" ]]; then
  TRAIN_ARGS+=(--eval_event_trigger_ambiguity_threshold "${eval_event_trigger_ambiguity_threshold}")
fi
if [[ -n "${eval_event_trigger_spillover_threshold:-}" ]]; then
  TRAIN_ARGS+=(--eval_event_trigger_spillover_threshold "${eval_event_trigger_spillover_threshold}")
fi
if [[ -n "${eval_event_trigger_support_margin_min_ratio:-}" ]]; then
  TRAIN_ARGS+=(--eval_event_trigger_support_margin_min_ratio "${eval_event_trigger_support_margin_min_ratio}")
fi
if [[ -n "${eval_event_trigger_support_threshold:-}" ]]; then
  TRAIN_ARGS+=(--eval_event_trigger_support_threshold "${eval_event_trigger_support_threshold}")
fi
if [[ -n "${eval_event_trigger_min_dwell:-}" ]]; then
  TRAIN_ARGS+=(--eval_event_trigger_min_dwell "${eval_event_trigger_min_dwell}")
fi
if [[ -n "${eval_event_trigger_max_interval:-}" ]]; then
  TRAIN_ARGS+=(--eval_event_trigger_max_interval "${eval_event_trigger_max_interval}")
fi
if [[ "${structured:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--structured)
fi
if [[ -n "${structured_d_global:-}" ]]; then
  TRAIN_ARGS+=(--d_global "${structured_d_global}")
fi
if [[ -n "${structured_num_basins:-}" ]]; then
  TRAIN_ARGS+=(--num_basins "${structured_num_basins}")
fi
if [[ -n "${structured_d_basin:-}" ]]; then
  TRAIN_ARGS+=(--d_basin "${structured_d_basin}")
fi
if [[ -n "${lambda_global:-}" ]]; then
  TRAIN_ARGS+=(--lambda_global "${lambda_global}")
fi
if [[ -n "${lambda_local:-}" ]]; then
  TRAIN_ARGS+=(--lambda_local "${lambda_local}")
fi
if [[ -n "${lambda_exclusivity:-}" ]]; then
  TRAIN_ARGS+=(--lambda_exclusivity "${lambda_exclusivity}")
fi
if [[ -n "${lambda_sparsity:-}" ]]; then
  TRAIN_ARGS+=(--lambda_sparsity "${lambda_sparsity}")
fi
if [[ -n "${lambda_entropy:-}" ]]; then
  TRAIN_ARGS+=(--lambda_entropy "${lambda_entropy}")
fi
if [[ -n "${lambda_dominance:-}" ]]; then
  TRAIN_ARGS+=(--lambda_dominance "${lambda_dominance}")
fi
if [[ -n "${lambda_temporal:-}" ]]; then
  TRAIN_ARGS+=(--lambda_temporal "${lambda_temporal}")
fi
if [[ -n "${excl_warmup_steps:-}" ]]; then
  TRAIN_ARGS+=(--excl_warmup_steps "${excl_warmup_steps}")
fi
if [[ "${soft_block:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--soft_block)
fi
if [[ -n "${soft_block_num_blocks:-}" ]]; then
  TRAIN_ARGS+=(--soft_block_num_blocks "${soft_block_num_blocks}")
fi
if [[ -n "${soft_block_weight:-}" ]]; then
  TRAIN_ARGS+=(--soft_block_weight "${soft_block_weight}")
fi
if [[ -n "${soft_block_norm:-}" ]]; then
  TRAIN_ARGS+=(--soft_block_norm "${soft_block_norm}")
fi
if [[ -n "${lr:-}" ]]; then
  TRAIN_ARGS+=(--lr "${lr}")
fi
if [[ -n "${k_matrix_lr:-}" ]]; then
  TRAIN_ARGS+=(--k_matrix_lr "${k_matrix_lr}")
fi
if [[ -n "${weight_decay:-}" ]]; then
  TRAIN_ARGS+=(--weight_decay "${weight_decay}")
fi
if [[ -n "${kuramoto_num_oscillators:-}" ]]; then
  TRAIN_ARGS+=(--kuramoto_num_oscillators "${kuramoto_num_oscillators}")
fi
if [[ -n "${kuramoto_topology:-}" ]]; then
  TRAIN_ARGS+=(--kuramoto_topology "${kuramoto_topology}")
fi
if [[ -n "${kuramoto_omega_mode:-}" ]]; then
  TRAIN_ARGS+=(--kuramoto_omega_mode "${kuramoto_omega_mode}")
fi
if [[ -n "${kuramoto_omega_spread:-}" ]]; then
  TRAIN_ARGS+=(--kuramoto_omega_spread "${kuramoto_omega_spread}")
fi
if [[ -n "${hopfield_num_neurons:-}" ]]; then
  TRAIN_ARGS+=(--hopfield_num_neurons "${hopfield_num_neurons}")
fi
if [[ -n "${hopfield_num_patterns:-}" ]]; then
  TRAIN_ARGS+=(--hopfield_num_patterns "${hopfield_num_patterns}")
fi
if [[ -n "${competitive_lv_num_species:-}" ]]; then
  TRAIN_ARGS+=(--competitive_lv_num_species "${competitive_lv_num_species}")
fi
if [[ -n "${competitive_lv_interaction_scale:-}" ]]; then
  TRAIN_ARGS+=(--competitive_lv_interaction_scale "${competitive_lv_interaction_scale}")
fi
if [[ -n "${competitive_lv_system_seed:-}" ]]; then
  TRAIN_ARGS+=(--competitive_lv_system_seed "${competitive_lv_system_seed}")
fi
if [[ "${standardize:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--standardize)
fi
if [[ "${dysts_native_cache:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--dysts_native_cache)
fi
if [[ -n "${dysts_cache_profile:-}" ]]; then
  TRAIN_ARGS+=(--dysts_cache_profile "${dysts_cache_profile}")
fi
if [[ "${dysts_cache_reuse:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--dysts_cache_reuse)
fi
if [[ -n "${dysts_ic_noise_scale:-}" ]]; then
  TRAIN_ARGS+=(--dysts_ic_noise_scale "${dysts_ic_noise_scale}")
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--checkpoint "${RESUME_CHECKPOINT}")
fi

gpu_guard_start_sampler \
  "${LOG_DIR}/gpu_utilization_${SLURM_JOB_ID:-local}_${TASK_ID}.csv" \
  "${GPU_TELEMETRY_INTERVAL:-30}"
gpu_guard_phase "paper benchmark training start task_id=${task_id}"
set +e
uv run python tools/train.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?
set -e
gpu_guard_phase "paper benchmark training end task_id=${task_id} exit_code=${EXIT_CODE}"
gpu_guard_stop_sampler

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
