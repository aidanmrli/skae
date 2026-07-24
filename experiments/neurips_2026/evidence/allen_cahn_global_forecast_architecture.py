"""Fail-closed validation of the Allen--Cahn architecture/capacity audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from experiments.neurips_2026.paths import PAPER_DATA_DIR


AUDIT = (
    PAPER_DATA_DIR
    / "allen_cahn_global_k_forecast_optimized_architecture_audit.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_architecture_audit(
    protocol: Mapping[str, object],
    artifacts: pd.DataFrame,
    audit_path: Path = AUDIT,
) -> dict[str, object]:
    """Verify that the audited checkpoints and the matched treatment stay fixed."""

    frozen = protocol["frozen_compact_evidence"]
    if audit_path.name != str(frozen["architecture_audit_path"]):
        raise ValueError("Allen--Cahn architecture-audit path drifted")
    if sha256(audit_path) != str(frozen["architecture_audit_sha256"]):
        raise ValueError("Allen--Cahn architecture-audit hash mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("audit_id")
        != "allen_cahn_global_k_forecast_optimized_architecture_audit"
        or audit.get("packet_id") != "allen_cahn_global_k_forecast_optimized"
        or audit.get("status")
        != "passed_capacity_and_forward_path_parity_with_joint_sparse_treatment"
    ):
        raise ValueError("Allen--Cahn architecture audit is not canonical")

    expected = {
        (str(row.arm), int(row.seed)): str(row.checkpoint_sha256)
        for row in artifacts.itertuples(index=False)
    }
    observed = {
        (str(run["arm"]), int(run["seed"])): str(run["checkpoint_sha256"])
        for run in audit["runs"]
    }
    if observed != expected or len(observed) != 20:
        raise ValueError("Architecture audit does not match the checkpoint roster")

    common = audit["common_checkpoint_structure"]
    if (
        int(common["trainable_parameter_count_from_source_and_state_elements"])
        != 12_698_690
        or int(common["effective_forward_parameter_count_excluding_inert_lista_s"])
        != 8_504_386
        or int(common["model_state_tensor_count"]) != 40
        or list(common["lista_s_shape"]) != [2048, 2048]
    ):
        raise ValueError("Allen--Cahn architecture/capacity parity drifted")
    forward = audit["forward_path_audit"]
    if (
        forward["dense_encode"] != "code"
        or forward["sparse_encode"] != "softshrink(code, lambda=0.15)"
        or bool(forward["capacity_difference"])
        or bool(forward["decoder_normalization_difference"])
    ):
        raise ValueError("Allen--Cahn forward-path parity drifted")
    if any(
        int(run["lista_s_nonzero_count"]) != 0
        or float(run["lista_s_max_abs"]) != 0.0
        for run in audit["runs"]
    ):
        raise ValueError("The nominal LISTA matrix is not inert in every checkpoint")
    configuration = audit["configuration_audit"]
    if not bool(configuration["no_other_paired_scientific_configuration_differences"]):
        raise ValueError("Unreported paired Allen--Cahn configuration differences")
    return audit
