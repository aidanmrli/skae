#!/bin/bash
#
# Queue the per-basin deep-slice interpretability re-evaluation for the five
# boundary-emphasized rows of Table 1 (SLK-BD, SLK-SB, Sparse MLP BD,
# Sparse MLP, Dense MLP no-shrink).
#
# Motivation: under the global top-quartile depth criterion, four of the 17
# benchmark systems have a deep slice concentrated in a single basin, which
# makes the wrong-support-freeze diagnostic undefined on those systems
# (no "wrong" canonical mask exists). The per-basin top-quartile criterion
# admits every basin's relative-deep states, restoring 17/17 coverage for
# the freeze diagnostic.
#
# This script does NOT retrain; it re-runs the interpretability evaluator on
# the saved checkpoints with `--depth_slice_mode per_basin`.
#
# All outputs are written to a NEW sibling directory under each existing
# packet so the global-slice results in `interpretability_final_pass1/` are
# preserved untouched. See docs/PER_BASIN_DEEP_SLICE_PLAN.md for the full
# protocol.
#
# Usage:
#   sbatch scripts/queue_per_basin_deep_eval.sh   # runs the launcher itself
# OR (recommended) launch the launcher inline so each shard gets its own
# sbatch:
#   bash scripts/queue_per_basin_deep_eval.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PACKET_FINAL="results/transition_rich_basin_partition_final_seed10_20260409"
PACKET_HARD="results/transition_rich_hardinit_mlp_controls_seed10_20260416"

# Boundary-emphasized roots only. The first two live in the "final" packet;
# the latter three live in the "hardinit MLP controls" packet.
LISTA_ROOTS_CSV="lista_blockdiag_signsplit_hardinit_basin_partition,lista_dense_softblock_signsplit_p64_hardinit_basin_partition"
MLP_ROOTS_CSV="mlp_sparse_blockdiag_hardinit_basin_partition_control,mlp_sparse_hardinit_basin_partition_control,mlp_zero_sparse_hardinit_basin_partition_control"

# The forecasting_rows.csv files that drive run discovery
ROWS_CSV_FINAL="${PACKET_FINAL}/collect_pass1/forecasting_rows.csv"
ROWS_CSV_HARD="${PACKET_HARD}/collect_pass1/forecasting_rows.csv"

# New output directories; the existing global-slice runs are untouched.
OUT_DIR_FINAL="${PACKET_FINAL}/interpretability_per_basin_deep_pass1"
OUT_DIR_HARD="${PACKET_HARD}/interpretability_per_basin_deep_pass1"

mkdir -p "${OUT_DIR_FINAL}" "${OUT_DIR_HARD}"

echo "==============================================="
echo "Per-basin deep-slice interpretability re-eval"
echo "==============================================="
echo "Repo: ${REPO_ROOT}"
echo "Packet (final, LISTA boundary):  ${PACKET_FINAL}"
echo "  rows_csv: ${ROWS_CSV_FINAL}"
echo "  out_dir:  ${OUT_DIR_FINAL}"
echo "  roots:    ${LISTA_ROOTS_CSV}"
echo "Packet (hardinit MLP controls):  ${PACKET_HARD}"
echo "  rows_csv: ${ROWS_CSV_HARD}"
echo "  out_dir:  ${OUT_DIR_HARD}"
echo "  roots:    ${MLP_ROOTS_CSV}"
echo "All shards: DEPTH_SLICE_MODE=per_basin (boundary-emphasized hard-init sampling)"
echo "==============================================="

# Submit packet A (LISTA finalists)
ROWS_CSV="${ROWS_CSV_FINAL}" \
OUT_DIR="${OUT_DIR_FINAL}" \
ROOT_LABELS_CSV="${LISTA_ROOTS_CSV}" \
DEPTH_SLICE_MODE="per_basin" \
LOG_DIR="${OUT_DIR_FINAL}/logs" \
QUEUE_MANIFEST_JSON="${OUT_DIR_FINAL}/queue_manifest.json" \
  bash scripts/queue_transition_rich_interpretability_shards.sh

# Submit packet B (boundary MLP controls)
ROWS_CSV="${ROWS_CSV_HARD}" \
OUT_DIR="${OUT_DIR_HARD}" \
ROOT_LABELS_CSV="${MLP_ROOTS_CSV}" \
DEPTH_SLICE_MODE="per_basin" \
LOG_DIR="${OUT_DIR_HARD}/logs" \
QUEUE_MANIFEST_JSON="${OUT_DIR_HARD}/queue_manifest.json" \
  bash scripts/queue_transition_rich_interpretability_shards.sh

echo "Submitted. See queue_manifest.json files for shard/merge job IDs."
