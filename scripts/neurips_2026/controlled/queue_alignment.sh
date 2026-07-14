#!/bin/bash
#
# Queue fixed support-alignment evaluation for all six controlled rows.
#
# The run is evaluation-only and reuses saved checkpoints. The reducer fixes
# the protocol to absolute support 1e-3, Jaccard 0.50, and every state at or
# above the within-label center-margin q75 (ties included).

set -euo pipefail

REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
ROOT_DIR="${REPO_ROOT}"
source scripts/common/cluster_env.sh

RESULTS_ROOT="${RESULTS_ROOT:-${SKAE_SCRATCH_ROOT}/results}"
PACKET_BACKFILL="${RESULTS_ROOT}/transition_rich_table2_5model_seed15_backfill_20260428"
PACKET_REPAIRED_MLP_BD="${RESULTS_ROOT}/transition_rich_sparse_mlp_bd_repaired_table1_20260506"
PACKET_SB_P256="${RESULTS_ROOT}/transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428"
PACKET_LISTA_P256="${RESULTS_ROOT}/transition_rich_lista_dense_p256_hardinit_table123_20260430"

BACKFILL_ROOTS_CSV="lista_blockdiag_signsplit_hardinit_basin_partition,mlp_sparse_hardinit_basin_partition_control,mlp_zero_sparse_hardinit_basin_partition_control"
REPAIRED_MLP_BD_ROOTS_CSV="mlp_sparse_blockdiag_hardinit_basin_partition_control"
SB_P256_ROOTS_CSV="lista_dense_softblock_signsplit_p256_hardinit_basin_partition"
LISTA_P256_ROOTS_CSV="lista_dense_signsplit_p256_hardinit_basin_partition"

OUT_DIR_NAME="controlled_support_alignment_pass0"

echo "==============================================="
echo "Controlled support-alignment evaluation"
echo "==============================================="
echo "Repo: ${REPO_ROOT}"
echo "Results root: ${RESULTS_ROOT}"
echo "Output dir name: ${OUT_DIR_NAME}"
echo "Protocol: support=absolute:0.001 scoring=within-label-margin>=q75-tie-inclusive Jaccard=0.50 entropy=nats"
echo "==============================================="

ROWS_CSV="${PACKET_BACKFILL}/collect_pass0/forecasting_rows.csv" \
OUT_DIR="${PACKET_BACKFILL}/${OUT_DIR_NAME}" \
ROOT_LABELS_CSV="${BACKFILL_ROOTS_CSV}" \
LOG_DIR="${PACKET_BACKFILL}/${OUT_DIR_NAME}/logs" \
QUEUE_MANIFEST_JSON="${PACKET_BACKFILL}/${OUT_DIR_NAME}/queue_manifest.json" \
  bash scripts/neurips_2026/controlled/queue_alignment_shards.sh

ROWS_CSV="${PACKET_REPAIRED_MLP_BD}/collect_pass0/forecasting_rows.csv" \
OUT_DIR="${PACKET_REPAIRED_MLP_BD}/${OUT_DIR_NAME}" \
ROOT_LABELS_CSV="${REPAIRED_MLP_BD_ROOTS_CSV}" \
LOG_DIR="${PACKET_REPAIRED_MLP_BD}/${OUT_DIR_NAME}/logs" \
QUEUE_MANIFEST_JSON="${PACKET_REPAIRED_MLP_BD}/${OUT_DIR_NAME}/queue_manifest.json" \
  bash scripts/neurips_2026/controlled/queue_alignment_shards.sh

ROWS_CSV="${PACKET_SB_P256}/collect_pass0/forecasting_rows.csv" \
OUT_DIR="${PACKET_SB_P256}/${OUT_DIR_NAME}" \
ROOT_LABELS_CSV="${SB_P256_ROOTS_CSV}" \
LOG_DIR="${PACKET_SB_P256}/${OUT_DIR_NAME}/logs" \
QUEUE_MANIFEST_JSON="${PACKET_SB_P256}/${OUT_DIR_NAME}/queue_manifest.json" \
  bash scripts/neurips_2026/controlled/queue_alignment_shards.sh

ROWS_CSV="${PACKET_LISTA_P256}/collect_pass0/forecasting_rows.csv" \
OUT_DIR="${PACKET_LISTA_P256}/${OUT_DIR_NAME}" \
ROOT_LABELS_CSV="${LISTA_P256_ROOTS_CSV}" \
LOG_DIR="${PACKET_LISTA_P256}/${OUT_DIR_NAME}/logs" \
QUEUE_MANIFEST_JSON="${PACKET_LISTA_P256}/${OUT_DIR_NAME}/queue_manifest.json" \
  bash scripts/neurips_2026/controlled/queue_alignment_shards.sh

echo "Submitted controlled support-alignment jobs."
echo "Manifests:"
echo "  ${PACKET_BACKFILL}/${OUT_DIR_NAME}/queue_manifest.json"
echo "  ${PACKET_REPAIRED_MLP_BD}/${OUT_DIR_NAME}/queue_manifest.json"
echo "  ${PACKET_SB_P256}/${OUT_DIR_NAME}/queue_manifest.json"
echo "  ${PACKET_LISTA_P256}/${OUT_DIR_NAME}/queue_manifest.json"
