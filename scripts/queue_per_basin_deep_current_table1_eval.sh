#!/bin/bash
#
# Queue per-basin deep-slice interpretability re-evaluation for the current
# manuscript Table 1 roster. This is the matched-dimension retained-15 Table 1
# source set used by scripts/build_per_system_stats_and_forest.py, not the older
# five-root p64 soft-block packet.
#
# The run is evaluation-only: it reuses saved checkpoints and changes only the
# deep-slice selection rule to DEPTH_SLICE_MODE=per_basin.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PACKET_BACKFILL="results/transition_rich_table2_5model_seed15_backfill_20260428"
PACKET_SB_P256="results/transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428"
PACKET_LISTA_P256="results/transition_rich_lista_dense_p256_hardinit_table123_20260430"

BACKFILL_ROOTS_CSV="lista_blockdiag_signsplit_hardinit_basin_partition,mlp_sparse_blockdiag_hardinit_basin_partition_control,mlp_sparse_hardinit_basin_partition_control,mlp_zero_sparse_hardinit_basin_partition_control"
SB_P256_ROOTS_CSV="lista_dense_softblock_signsplit_p256_hardinit_basin_partition"
LISTA_P256_ROOTS_CSV="lista_dense_signsplit_p256_hardinit_basin_partition"

OUT_DIR_NAME="interpretability_per_basin_deep_current_table1_pass0"

echo "==============================================="
echo "Current Table 1 per-basin deep-slice re-eval"
echo "==============================================="
echo "Repo: ${REPO_ROOT}"
echo "Output dir name: ${OUT_DIR_NAME}"
echo "Depth slice mode: per_basin"
echo "==============================================="

ROWS_CSV="${PACKET_BACKFILL}/collect_pass0/forecasting_rows.csv" \
OUT_DIR="${PACKET_BACKFILL}/${OUT_DIR_NAME}" \
ROOT_LABELS_CSV="${BACKFILL_ROOTS_CSV}" \
DEPTH_SLICE_MODE="per_basin" \
LOG_DIR="${PACKET_BACKFILL}/${OUT_DIR_NAME}/logs" \
QUEUE_MANIFEST_JSON="${PACKET_BACKFILL}/${OUT_DIR_NAME}/queue_manifest.json" \
  bash scripts/queue_transition_rich_interpretability_shards.sh

ROWS_CSV="${PACKET_SB_P256}/collect_pass0/forecasting_rows.csv" \
OUT_DIR="${PACKET_SB_P256}/${OUT_DIR_NAME}" \
ROOT_LABELS_CSV="${SB_P256_ROOTS_CSV}" \
DEPTH_SLICE_MODE="per_basin" \
LOG_DIR="${PACKET_SB_P256}/${OUT_DIR_NAME}/logs" \
QUEUE_MANIFEST_JSON="${PACKET_SB_P256}/${OUT_DIR_NAME}/queue_manifest.json" \
  bash scripts/queue_transition_rich_interpretability_shards.sh

ROWS_CSV="${PACKET_LISTA_P256}/collect_pass0/forecasting_rows.csv" \
OUT_DIR="${PACKET_LISTA_P256}/${OUT_DIR_NAME}" \
ROOT_LABELS_CSV="${LISTA_P256_ROOTS_CSV}" \
DEPTH_SLICE_MODE="per_basin" \
LOG_DIR="${PACKET_LISTA_P256}/${OUT_DIR_NAME}/logs" \
QUEUE_MANIFEST_JSON="${PACKET_LISTA_P256}/${OUT_DIR_NAME}/queue_manifest.json" \
  bash scripts/queue_transition_rich_interpretability_shards.sh

echo "Submitted current Table 1 per-basin deep-slice jobs."
echo "Manifests:"
echo "  ${PACKET_BACKFILL}/${OUT_DIR_NAME}/queue_manifest.json"
echo "  ${PACKET_SB_P256}/${OUT_DIR_NAME}/queue_manifest.json"
echo "  ${PACKET_LISTA_P256}/${OUT_DIR_NAME}/queue_manifest.json"
