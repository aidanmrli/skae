#!/usr/bin/env bash
# Reusable GPU-utilization guards for SLURM jobs. Intended to be sourced.

# shellcheck shell=bash

SLURM_GPU_GUARD_SAMPLER_PID="${SLURM_GPU_GUARD_SAMPLER_PID:-}"
SLURM_GPU_GUARD_SAMPLER_LOG="${SLURM_GPU_GUARD_SAMPLER_LOG:-}"
SLURM_GPU_GUARD_TRAP_INSTALLED="${SLURM_GPU_GUARD_TRAP_INSTALLED:-0}"

slurm_gpu_guard_gpu_expected() {
  local value

  for value in "${SLURM_JOB_GPUS:-}" "${SLURM_STEP_GPUS:-}"; do
    if [[ -n "${value}" ]]; then
      return 0
    fi
  done

  for value in \
    "${SLURM_GPUS:-}" \
    "${SLURM_GPUS_ON_NODE:-}" \
    "${SLURM_GPUS_PER_NODE:-}" \
    "${SLURM_GPUS_PER_TASK:-}"; do
    if [[ -n "${value}" && "${value}" != "0" ]]; then
      return 0
    fi
  done

  value="${CUDA_VISIBLE_DEVICES:-}"
  case "${value}" in
    "" | "-1" | "NoDevFiles" | "none" | "void")
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

slurm_gpu_guard_log_context() {
  local log_path="${1:-}"

  if [[ -n "${log_path}" ]]; then
    mkdir -p "$(dirname "${log_path}")"
    _slurm_gpu_guard_print_context >>"${log_path}"
  else
    _slurm_gpu_guard_print_context
  fi
}

slurm_gpu_guard_assert_nvidia_smi() {
  local expected="${1:-auto}"

  case "${expected}" in
    auto | "")
      if ! slurm_gpu_guard_gpu_expected; then
        return 0
      fi
      ;;
    1 | true | yes)
      ;;
    0 | false | no)
      return 0
      ;;
    *)
      echo "Invalid GPU expectation '${expected}'; use auto, yes, or no." >&2
      return 2
      ;;
  esac

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU allocation expected, but nvidia-smi is not on PATH." >&2
    return 1
  fi

  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "GPU allocation expected, but nvidia-smi cannot see a GPU." >&2
    return 1
  fi
}

slurm_gpu_guard_start_sampler() {
  local log_path="${1:-}"
  local interval_seconds="${2:-30}"
  local log_dir

  if [[ -z "${log_path}" ]]; then
    echo "Usage: slurm_gpu_guard_start_sampler <csv_log_path> [interval_seconds]" >&2
    return 2
  fi
  case "${interval_seconds}" in
    "" | *[!0-9]*)
      echo "Sampler interval must be a positive integer number of seconds." >&2
      return 2
      ;;
  esac
  if [[ "${interval_seconds}" -lt 1 ]]; then
    echo "Sampler interval must be a positive integer number of seconds." >&2
    return 2
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Cannot start GPU sampler: nvidia-smi is not on PATH." >&2
    return 1
  fi
  if [[ -n "${SLURM_GPU_GUARD_SAMPLER_PID:-}" ]] \
    && kill -0 "${SLURM_GPU_GUARD_SAMPLER_PID}" >/dev/null 2>&1; then
    echo "GPU sampler already running with pid ${SLURM_GPU_GUARD_SAMPLER_PID}." >&2
    return 0
  fi

  log_dir="$(dirname "${log_path}")"
  mkdir -p "${log_dir}"
  printf '%s\n' \
    "timestamp,index,uuid,name,gpu_utilization_percent,memory_utilization_percent,memory_used_mib,memory_total_mib,power_draw_w" \
    >"${log_path}"

  (
    while :; do
      if ! nvidia-smi \
        --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits >>"${log_path}" 2>/dev/null; then
        printf '%s,,,,,,,,\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" >>"${log_path}"
      fi
      sleep "${interval_seconds}" || break
    done
  ) &

  SLURM_GPU_GUARD_SAMPLER_PID="$!"
  SLURM_GPU_GUARD_SAMPLER_LOG="${log_path}"
  slurm_gpu_guard_stop_sampler_on_exit
}

slurm_gpu_guard_stop_sampler() {
  local sampler_pid="${SLURM_GPU_GUARD_SAMPLER_PID:-}"

  if [[ -z "${sampler_pid}" ]]; then
    return 0
  fi

  if kill -0 "${sampler_pid}" >/dev/null 2>&1; then
    kill "${sampler_pid}" >/dev/null 2>&1 || true
    wait "${sampler_pid}" 2>/dev/null || true
  fi

  SLURM_GPU_GUARD_SAMPLER_PID=""
}

slurm_gpu_guard_stop_sampler_on_exit() {
  local old_trap
  local old_command

  if [[ "${SLURM_GPU_GUARD_TRAP_INSTALLED:-0}" == "1" ]]; then
    return 0
  fi

  old_trap="$(trap -p EXIT || true)"
  if [[ -n "${old_trap}" ]]; then
    old_command="${old_trap#trap -- \'}"
    old_command="${old_command%\' EXIT}"
    if [[ "${old_command}" == "${old_trap}" ]]; then
      trap -- 'slurm_gpu_guard_stop_sampler' EXIT
    else
      trap -- "slurm_gpu_guard_stop_sampler; ${old_command}" EXIT
    fi
  else
    trap -- 'slurm_gpu_guard_stop_sampler' EXIT
  fi

  SLURM_GPU_GUARD_TRAP_INSTALLED="1"
}

_slurm_gpu_guard_print_context() {
  echo "date=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "host=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-}"
  echo "slurm_nodelist=${SLURM_NODELIST:-}"
  echo "slurm_job_gpus=${SLURM_JOB_GPUS:-}"
  echo "slurm_step_gpus=${SLURM_STEP_GPUS:-}"
  echo "slurm_gpus=${SLURM_GPUS:-}"
  echo "slurm_gpus_on_node=${SLURM_GPUS_ON_NODE:-}"
  echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "cuda_device_order=${CUDA_DEVICE_ORDER:-}"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
    nvidia-smi \
      --query-gpu=index,uuid,name,driver_version,memory.total \
      --format=csv || true
  else
    echo "nvidia_smi=not_found"
  fi
}


# Compatibility aliases used by the current benchmark runners.
gpu_guard_print_context() {
  local label="${1:-GPU job}"
  echo "============================================="
  echo "${label} GPU context"
  slurm_gpu_guard_log_context
  echo "============================================="
}

gpu_guard_assert_cuda_visible() {
  local label="${1:-GPU job}"
  if ! slurm_gpu_guard_assert_nvidia_smi yes; then
    echo "${label}: refusing to run CPU fallback inside a GPU job." >&2
    return 2
  fi
}

gpu_guard_start_sampler() {
  local log_path="${1:?GPU telemetry log path is required}"
  local interval_seconds="${2:-30}"
  if [[ "${GPU_TELEMETRY:-1}" != "1" ]]; then
    echo "GPU telemetry disabled (GPU_TELEMETRY=${GPU_TELEMETRY})."
    return 0
  fi
  slurm_gpu_guard_start_sampler "${log_path}" "${interval_seconds}"
}

gpu_guard_stop_sampler() {
  slurm_gpu_guard_stop_sampler
}

gpu_guard_phase() {
  echo "[gpu-guard] $(date '+%Y-%m-%dT%H:%M:%S%z') :: $*"
}
