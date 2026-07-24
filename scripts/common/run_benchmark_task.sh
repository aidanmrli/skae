#!/bin/bash
#
# Execute one paper benchmark task-table row inside an allocation.
#
# Required env vars:
#   TASK_TSV=<path>
#   BASE_OUT=<output root>
#
# Optional:
#   ARRAY_OFFSET=0
#
# This payload has no SBATCH directives. Submit run_benchmark_array.sh for one
# row per array element, or run_benchmark_packed_array.sh for several rows per
# allocated GPU.

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

source scripts/common/gpu_guard.sh
trap gpu_guard_stop_sampler EXIT
source .venv/bin/activate

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
TRAIN_SKIP_EVAL="${TRAIN_SKIP_EVAL:-0}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-}"
TASK_TSV_SHA256="${TASK_TSV_SHA256:-}"

if [[ -n "${SOURCE_MANIFEST}" ]]; then
  sha256sum -c "${SOURCE_MANIFEST}"
fi
if [[ -n "${TASK_TSV_SHA256}" ]]; then
  printf '%s  %s\n' "${TASK_TSV_SHA256}" "${TASK_TSV}" | sha256sum -c -
fi

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
LOG_DIR="${LOG_ROOT}/${system_slug}/dt_${DT_TAG}/seed_${seed}"
mkdir -p "${LOG_DIR}"

COMPLETED_RUN=""
if [[ "${SKIP_COMPLETED:-1}" == "1" ]]; then
  if [[ "${TRAIN_SKIP_EVAL}" == "1" ]]; then
    COMPLETED_RUN="$(
      find "${LOG_DIR}" -mindepth 1 -maxdepth 1 -type d \
        -name '20*' -exec test -f '{}/training_success.json' ';' -print \
        | sort | tail -n 1
    )"
  else
    COMPLETED_RUN="$(
      find "${LOG_DIR}" -mindepth 1 -maxdepth 1 -type d \
        -name '20*' -exec test -f '{}/evaluation_summary.json' ';' -print \
        | sort | tail -n 1
    )"
  fi
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
if [[ "${standardize:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--standardize)
fi
if [[ "${dysts_native_cache:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--dysts_native_cache)
fi
if [[ -n "${dysts_cache_profile:-}" ]]; then
  TRAIN_ARGS+=(--dysts_cache_profile "${dysts_cache_profile}")
fi
if [[ -n "${DYSTS_CACHE_DIR:-}" ]]; then
  TRAIN_ARGS+=(--dysts_cache_dir "${DYSTS_CACHE_DIR}")
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
if [[ "${TRAIN_SKIP_EVAL}" == "1" ]]; then
  TRAIN_ARGS+=(--skip_eval --save_last_checkpoint)
fi

if [[ "${PACK_LEVEL_GPU_GUARD:-0}" != "1" ]]; then
  gpu_guard_start_sampler \
    "${LOG_DIR}/gpu_utilization_${SLURM_JOB_ID:-local}_${TASK_ID}.csv" \
    "${GPU_TELEMETRY_INTERVAL:-60}"
fi
gpu_guard_phase "paper benchmark training start task_id=${task_id}"
set +e
uv run skae-train "${TRAIN_ARGS[@]}"
EXIT_CODE=$?
set -e
gpu_guard_phase "paper benchmark training end task_id=${task_id} exit_code=${EXIT_CODE}"
if [[ "${PACK_LEVEL_GPU_GUARD:-0}" != "1" ]]; then
  gpu_guard_stop_sampler
fi

if (( EXIT_CODE == 0 )) && [[ "${TRAIN_SKIP_EVAL}" == "1" ]]; then
  RUN_DIR="$(
    find "${LOG_DIR}" -mindepth 1 -maxdepth 1 -type d -name '20*' \
      -exec test -f '{}/last.pt' ';' -print | sort | tail -n 1
  )"
  [[ -n "${RUN_DIR}" ]] || { echo "Missing completed training run" >&2; exit 1; }
  uv run python - "${RUN_DIR}" "${TASK_TSV}" "${task_id}" "${num_steps}" \
    "${env_name}" "${env_dt}" "${model_variant}" "${lista_num_loops:-}" <<'PY'
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch

run_dir = Path(sys.argv[1])
task_path = Path(sys.argv[2])
task_id = int(sys.argv[3])
expected_steps = int(sys.argv[4])
env_name = sys.argv[5]
expected_dt = float(sys.argv[6])
model_variant = sys.argv[7]
expected_lista_loops = sys.argv[8]
checkpoint = run_dir / "last.pt"
best_checkpoint = run_dir / "checkpoint.pt"
metrics = run_dir / "final_metrics.json"
if not checkpoint.is_file() or not best_checkpoint.is_file() or not metrics.is_file():
    raise SystemExit("missing best/last checkpoint or final metrics")

last_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
if int(last_payload.get("step", -1)) != expected_steps - 1:
    raise SystemExit(
        f"last checkpoint step {last_payload.get('step')} != {expected_steps - 1}"
    )
if env_name.lower().startswith("dysts:"):
    config = last_payload["config"]
    dysts = config["ENV"]["DYSTS"]
    expected_cache = str(Path(os.environ["DYSTS_CACHE_DIR"]).resolve())
    actual_cache = str(Path(dysts["CACHE_DIR"]).resolve())
    if actual_cache != expected_cache:
        raise SystemExit(f"wrong Dysts cache root: {actual_cache} != {expected_cache}")
    if dysts["CACHE_SPLIT"] != "train" or not dysts["USE_NATIVE_CACHE"]:
        raise SystemExit("Dysts training did not use the native train cache")
    if not math.isclose(float(dysts["DT_OVERRIDE"]), expected_dt, rel_tol=0, abs_tol=1e-15):
        raise SystemExit("Dysts checkpoint dt does not match the task")
    required_selection = {
        "checkpoint_selection_rollout": "direct",
        "checkpoint_selection_metric": "direct_strict_full_horizon_cumulative_state_summed_mse",
        "checkpoint_selection_horizon": 200,
        "checkpoint_selection_batch_size": 16,
        "checkpoint_selection_split": "val",
    }
    for key, expected in required_selection.items():
        if best_payload.get(key) != expected:
            raise SystemExit(f"Dysts checkpoint {key}={best_payload.get(key)!r} != {expected!r}")
    if not math.isfinite(float(best_payload.get("checkpoint_selection_score", math.inf))):
        raise SystemExit("Dysts best checkpoint has no finite strict direct score")
    if float(best_payload.get("checkpoint_selection_full_horizon_finite_fraction", 0.0)) != 1.0:
        raise SystemExit("Dysts best checkpoint is not fully finite on validation")
    if model_variant.startswith("lista"):
        actual_loops = int(config["MODEL"]["ENCODER"]["LISTA"]["NUM_LOOPS"])
        if actual_loops != int(expected_lista_loops):
            raise SystemExit(f"LISTA loops {actual_loops} != {expected_lista_loops}")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "schema_version": 1,
    "status": "training_complete",
    "task_id": task_id,
    "expected_optimizer_steps": expected_steps,
    "task_table_sha256": sha256(task_path),
    "last_checkpoint_sha256": sha256(checkpoint),
    "best_checkpoint_sha256": sha256(best_checkpoint),
    "final_metrics_sha256": sha256(metrics),
    "last_checkpoint_step": int(last_payload["step"]),
    "checkpoint_selection_rollout": best_payload.get("checkpoint_selection_rollout"),
    "checkpoint_selection_metric": best_payload.get("checkpoint_selection_metric"),
    "checkpoint_selection_score": best_payload.get("checkpoint_selection_score"),
}
(run_dir / "training_success.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
fi

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
