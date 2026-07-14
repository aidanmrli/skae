#!/usr/bin/env bash
# Portable storage defaults shared by the paper's SLURM launchers.
# Source this file after setting ROOT_DIR and changing into the repository root.

# shellcheck shell=bash

if [[ -z "${ROOT_DIR:-}" ]]; then
  echo "ROOT_DIR must be set before sourcing scripts/common/cluster_env.sh" >&2
  return 2
fi

SKAE_CLUSTER_USER="${USER:-}"
SKAE_CLUSTER_USER_ROOT="/network/scratch/${SKAE_CLUSTER_USER:0:1}/${SKAE_CLUSTER_USER}"

if [[ -z "${SKAE_SCRATCH_ROOT:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    SKAE_SCRATCH_ROOT="${SCRATCH%/}/skae"
  elif [[ -n "${SKAE_CLUSTER_USER}" && -d "${SKAE_CLUSTER_USER_ROOT}" ]]; then
    SKAE_SCRATCH_ROOT="${SKAE_CLUSTER_USER_ROOT}/skae"
  else
    SKAE_SCRATCH_ROOT="${ROOT_DIR}/runs"
  fi
fi

export SKAE_SCRATCH_ROOT
export DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-${SKAE_SCRATCH_ROOT}/dysts_native_cache}"
unset SKAE_CLUSTER_USER SKAE_CLUSTER_USER_ROOT
