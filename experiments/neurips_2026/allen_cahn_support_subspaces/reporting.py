"""Compact row-level reporting for the Allen--Cahn support-subspace audit."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_rows(path: Path, shards: list[dict[str, Any]]) -> None:
    rows = []
    for item in shards:
        row = {
            "seed": item["seed"],
            "sparse_active_density": item["mask_summary"]["sparse_initial_active_density"],
            "sparse_source_capture": item["initial_projection"]["sparse"]["source_capture_rms"],
            "dense_source_capture": item["initial_projection"]["dense"]["source_capture_rms"],
            "family_eligible": item["family"]["sparse"]["eligible"],
            "family_train_fit_count": item["family"]["sparse"]["train_fit_family_count"],
            "family_qualified_count": item["family"]["sparse"]["qualified_family_count"],
            "family_coverage": item["family"]["sparse"]["all_routed_score_coverage"],
            "qualified_family_coverage": item["family"]["sparse"]
            ["qualified_family_score_coverage"],
            "fit_frozen_top_two": ";".join(
                str(value) for value in item["family"]["sparse"]
                ["fit_frozen_top_two_family_indices"]
            ),
        }
        family_payload = item["sparse_family"]
        family_projection = family_payload.get("initial_projection")
        row["family_source_capture"] = (
            None if family_projection is None else family_projection["source_capture_rms"]
        )
        row["family_initial_reconstruction_ratio"] = (
            None if family_projection is None else
            float(family_projection["projected_initial_reconstruction_mse"])
            / max(float(family_projection["full_initial_reconstruction_mse"]), 1e-20)
        )
        row["signature_observed_over_null"] = (
            family_payload.get("signature_differentiation", {}).get("observed_over_null")
        )
        row["coordinate_distance_observed_over_null_descriptive"] = (
            family_payload.get("coordinate_chart_distance_descriptive_only", {})
            .get("observed_over_null")
        )
        signatures = family_payload.get("signature_differentiation", {}).get(
            "family_signatures", []
        )
        for signature_index in range(2):
            signature = signatures[signature_index] if len(signatures) > signature_index else {}
            components = signature.get("components_mu_s_r", [None, None, None])
            row[f"signature_{signature_index + 1}_family_index"] = signature.get(
                "family_index"
            )
            row[f"signature_{signature_index + 1}_cardinality"] = signature.get(
                "cardinality"
            )
            for component_name, value in zip(("mu", "s", "r"), components):
                row[f"signature_{signature_index + 1}_{component_name}"] = value
        geometry = family_payload.get("fit_frozen_pair_support_geometry", {})
        row["fit_frozen_pair_cardinalities"] = ";".join(
            str(value) for value in geometry.get("cardinalities", [])
        )
        row["fit_frozen_pair_intersection"] = geometry.get("intersection")
        row["fit_frozen_pair_jaccard"] = geometry.get("jaccard")
        for horizon in (160, 200):
            key = str(horizon)
            for arm in ("sparse", "dense"):
                cell = item["closure"][arm]["horizons"][key]
                row[f"h{horizon}_{arm}_k_leakage"] = cell["true"]["activity_k_leakage_rms"]
                row[f"h{horizon}_{arm}_k_null"] = cell["null_median"]["activity_k_leakage_rms"]
                row[f"h{horizon}_{arm}_kminusI_leakage"] = cell["true"][
                    "activity_kminusI_leakage_rms"
                ]
                row[f"h{horizon}_{arm}_rho"] = item["forecast"][arm]["ratios"][key][
                    "mean_restricted_over_mask_once"
                ]
                for mode in ("full", "mask_once", "restricted"):
                    row[f"h{horizon}_{arm}_{mode}_mse"] = item["forecast"][arm][key][mode][
                        "field_mse"
                    ]
            derangement = family_payload.get("top_two_family_derangement")
            row[f"h{horizon}_correct_family_rho"] = None if derangement is None else (
                derangement["correct"]["ratios"][key]["mean_restricted_over_mask_once"]
            )
            row[f"h{horizon}_wrong_family_rho"] = None if derangement is None else (
                derangement["wrong_swap"]["ratios"][key]["mean_restricted_over_mask_once"]
            )
        for arm in ("sparse", "dense"):
            row[f"{arm}_matrix_k_leakage"] = item["closure"][arm]["matrix_true"][
                "matrix_k_leakage_fro"
            ]
            row[f"{arm}_matrix_k_null"] = item["closure"][arm]["matrix_null_median"][
                "matrix_k_leakage_fro"
            ]
            row[f"{arm}_matrix_kminusI_leakage"] = item["closure"][arm]["matrix_true"][
                "matrix_kminusI_leakage_fro"
            ]
        rows.append(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

