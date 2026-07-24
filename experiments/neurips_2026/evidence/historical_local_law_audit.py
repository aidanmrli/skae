"""Authenticate and recompute the April 2026 local-law evidence packets.

This is deliberately a historical audit, not an active paper workflow.  The
source packets live on scratch; a compact deterministic JSON records exactly
what can and cannot be defended from them.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from experiments.neurips_2026.paths import PAPER_DATA_DIR
from experiments.neurips_2026.evidence.historical_local_law_sources import (
    CENTERED_DIR,
    GEOMETRY_DIR,
    SELF_ROUTED_DIR,
    verify_sources,
)


OUTPUT = PAPER_DATA_DIR / "historical_centered_local_law_audit.json"

BLOCK = "lista_blockdiag_signsplit_hardinit_basin_partition"
FULL_SPARSE = "lista_dense_softblock_signsplit_p64_hardinit_basin_partition"
DENSE = "mlp_zero_sparse_basin_partition_control"
ROOTS = (BLOCK, FULL_SPARSE, DENSE)
ALIASES = {BLOCK: "block_sparse", FULL_SPARSE: "full_k_sparse", DENSE: "dense_mlp"}

def _finite(values: Iterable[Any]) -> np.ndarray:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    return array[np.isfinite(array)]


def ratio_summary(values: Iterable[Any]) -> dict[str, Any]:
    """Summarize ratios while exposing non-finite selection explicitly."""
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    present = ~np.isnan(array)
    finite = np.isfinite(array)
    clean = array[finite]
    return {
        "ratio_row_count": int(present.sum()),
        "finite_ratio_count": int(finite.sum()),
        "infinite_ratio_count": int(np.isinf(array).sum()),
        "wins_below_one": int(np.sum(present & (array < 1.0))),
        "exact_ties_at_one": int(np.sum(present & (array == 1.0))),
        "finite_median": float(np.median(clean)) if clean.size else None,
        "finite_mean": float(np.mean(clean)) if clean.size else None,
    }


def scalar_summary(values: Iterable[Any]) -> dict[str, Any]:
    clean = _finite(values)
    return {
        "finite_count": int(clean.size),
        "median": float(np.median(clean)) if clean.size else None,
        "mean": float(np.mean(clean)) if clean.size else None,
    }


def _direct_ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> np.ndarray:
    left = pd.to_numeric(frame[numerator], errors="coerce").to_numpy(float)
    right = pd.to_numeric(frame[denominator], errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return left / right


def _roster(frame: pd.DataFrame) -> set[tuple[str, str, int]]:
    return {
        (str(row.root_label), str(row.system_key), int(row.seed))
        for row in frame[["root_label", "system_key", "seed"]]
        .drop_duplicates()
        .itertuples(index=False)
    }


def _paired_control(obs: pd.DataFrame, control: pd.DataFrame) -> dict[str, Any]:
    keys = ["root_label", "system_key", "seed"]
    paired = obs[keys + ["partition_centered_test_mse"]].merge(
        control[keys + ["partition_centered_test_mse"]],
        on=keys,
        suffixes=("_observed", "_control"),
        validate="one_to_one",
    )
    ratios = (
        paired["partition_centered_test_mse_observed"]
        / paired["partition_centered_test_mse_control"]
    )
    return ratio_summary(ratios)


def centered_results(frame: pd.DataFrame) -> dict[str, Any]:
    condition = (
        (frame.support_definition == "relative:0.1")
        & (frame.depth_stratum == "q4")
        & (frame.transition_regime == "persistent_current")
        & (frame.partition_kind == "support")
    )
    observed = frame.loc[condition & (frame.control_kind == "none")]
    random = frame.loc[condition & (frame.control_kind == "random_count_matched")]
    cluster = frame.loc[condition & (frame.control_kind == "latent_kmeans")]
    results: dict[str, Any] = {}
    for root in ROOTS:
        subset = observed.loc[observed.root_label == root]
        results[ALIASES[root]] = {
            "candidate_observed_rows_in_condition": int(len(subset)),
            "local_refit_over_learned_k": ratio_summary(
                _direct_ratio(
                    subset, "partition_centered_test_mse", "global_k_test_mse"
                )
            ),
            "local_refit_over_separately_refit_global_centered": ratio_summary(
                _direct_ratio(
                    subset,
                    "partition_centered_test_mse",
                    "global_centered_test_mse",
                )
            ),
            "same_learned_k_input_mask_over_learned_k": ratio_summary(
                _direct_ratio(subset, "input_gated_k_test_mse", "global_k_test_mse")
            ),
            "same_learned_k_centered_p_k_p_over_learned_k": ratio_summary(
                _direct_ratio(
                    subset, "submatrix_gated_k_test_mse", "global_k_test_mse"
                )
            ),
            "configured_block_union_p_k_p_over_learned_k": ratio_summary(
                _direct_ratio(
                    subset, "block_submatrix_k_test_mse", "global_k_test_mse"
                )
            ),
            "test_coverage": scalar_summary(subset.test_coverage_fraction),
            "observed_local_refit_over_count_matched_random_refit": _paired_control(
                subset, random.loc[random.root_label == root]
            ),
            "observed_local_refit_over_latent_kmeans_refit": _paired_control(
                subset, cluster.loc[cluster.root_label == root]
            ),
        }

    topk = frame.loc[
        (frame.support_definition == "topk:8")
        & (frame.depth_stratum == "q4")
        & (frame.transition_regime == "persistent_current")
        & (frame.partition_kind == "support")
        & (frame.control_kind == "none")
    ]
    sensitivity = {}
    for root in ROOTS:
        subset = topk.loc[topk.root_label == root]
        sensitivity[ALIASES[root]] = {
            "same_learned_k_input_mask_over_learned_k": ratio_summary(
                _direct_ratio(subset, "input_gated_k_test_mse", "global_k_test_mse")
            ),
            "same_learned_k_centered_p_k_p_over_learned_k": ratio_summary(
                _direct_ratio(
                    subset, "submatrix_gated_k_test_mse", "global_k_test_mse"
                )
            ),
            "configured_block_union_p_k_p_over_learned_k": ratio_summary(
                _direct_ratio(
                    subset, "block_submatrix_k_test_mse", "global_k_test_mse"
                )
            ),
            "test_coverage": scalar_summary(subset.test_coverage_fraction),
        }
    return {
        "condition": {
            "support_definition": "relative:0.1",
            "depth_stratum": "q4 (deepest evaluation-label margin quartile)",
            "transition_regime": "persistent_current",
            "partition_kind": "support",
        },
        "relative_threshold_results": results,
        "topk8_sensitivity": sensitivity,
    }


def self_routed_results(frame: pd.DataFrame) -> dict[str, Any]:
    selected = frame.loc[
        (frame.support_definition == "topk:8") & (frame.depth_stratum == "all")
    ]
    modes = (
        "global_k",
        "support_gated_k",
        "support_block_gated_k",
        "support_local_centered",
        "family_local_centered",
    )
    results = {}
    for root in ROOTS:
        root_frame = selected.loc[selected.root_label == root]
        by_mode = {}
        comparable_finite_sets = []
        for mode in modes:
            subset = root_frame.loc[root_frame.rollout_mode == mode]
            if mode in {
                "support_gated_k",
                "support_local_centered",
                "family_local_centered",
            }:
                comparable_finite_sets.append(
                    set(
                        zip(
                            subset.loc[
                                np.isfinite(subset.h1000_over_global), "system_key"
                            ],
                            subset.loc[
                                np.isfinite(subset.h1000_over_global), "seed"
                            ],
                        )
                    )
                )
            valid = pd.to_numeric(subset.valid_step_fraction, errors="coerce")
            by_mode[mode] = {
                "row_count": int(len(subset)),
                "route_coverage": scalar_summary(subset.route_coverage_fraction),
                "h1000_over_global": ratio_summary(subset.h1000_over_global),
                "valid_step_fraction_full_rows": int(np.sum(valid == 1.0)),
                "valid_step_fraction_partial_rows": int(np.sum((valid >= 0) & (valid < 1))),
            }
        results[ALIASES[root]] = {
            "modes": by_mode,
            "common_finite_h1000_ratio_rows_across_support_gated_local_family": int(
                len(set.intersection(*comparable_finite_sets))
            ),
        }
    return {
        "condition": {
            "support_definition": "topk:8 (foregrounded after two rules existed)",
            "depth_stratum": "all",
            "forecast_trajectories": "128 fresh trajectories, eval seed 314",
            "operator_fit_trajectories": "256 disjoint trajectories, eval seed 42",
            "reencoding": "none (period 0)",
        },
        "results": results,
        "survival_warning": (
            "The evaluator nan-means each finite per-IC prefix and then averages only "
            "surviving ICs. Ratio availability and survivor sets differ by mode; means "
            "therefore need not compare a common set and can be dominated by overflow tails."
        ),
    }


def checkpoint_audit(frame: pd.DataFrame) -> dict[str, Any]:
    runs = frame[["root_label", "system_key", "seed", "run_dir"]].drop_duplicates()
    records = []
    for row in runs.itertuples(index=False):
        config = json.loads((Path(row.run_dir) / "config.json").read_text())
        model = config["MODEL"]
        encoder = model["ENCODER"]
        train = config["TRAIN"]
        soft = model.get("SOFT_BLOCK", {})
        records.append(
            {
                "root": row.root_label,
                "system": row.system_key,
                "seed": int(row.seed),
                "model": model["MODEL_NAME"],
                "target": int(model["TARGET_SIZE"]),
                "sparsity": float(model["SPARSITY_COEFF"]),
                "k_structure": model["K_STRUCTURE"],
                "k_num_blocks": int(model.get("K_NUM_BLOCKS", 0)),
                "soft_enabled": bool(soft.get("ENABLED", False)),
                "soft_num_blocks": int(soft.get("NUM_BLOCKS", 0)),
                "soft_weight": float(soft.get("WEIGHT", 0.0)),
                "activation": encoder["ACTIVATION"],
                "last_relu": bool(encoder["LAST_RELU"]),
                "weight_decay": float(train["WEIGHT_DECAY"]),
            }
        )
    recipes = {}
    for root in ROOTS:
        subset = [record for record in records if record["root"] == root]
        recipes[ALIASES[root]] = {
            "checkpoint_count": len(subset),
            "model_names": sorted({record["model"] for record in subset}),
            "latent_sizes": sorted({record["target"] for record in subset}),
            "sparsity_coefficients": sorted({record["sparsity"] for record in subset}),
            "k_structures": sorted({record["k_structure"] for record in subset}),
            "weight_decays": sorted({record["weight_decay"] for record in subset}),
            "activation_last_relu_counts": dict(
                sorted(
                    Counter(
                        f"{record['activation']}|last_relu={str(record['last_relu']).lower()}"
                        for record in subset
                    ).items()
                )
            ),
            "soft_block_enabled_count": sum(record["soft_enabled"] for record in subset),
            "soft_block_weights": sorted({record["soft_weight"] for record in subset}),
        }
    hard_counts = {
        record["system"]: record["k_num_blocks"]
        for record in records
        if record["root"] == BLOCK and record["seed"] == 0
    }
    soft_counts = {
        record["system"]: record["soft_num_blocks"]
        for record in records
        if record["root"] == FULL_SPARSE and record["seed"] == 0
    }
    assert len(records) == 510
    assert len(hard_counts) == len(soft_counts) == 17
    return {
        "recipes": recipes,
        "preconfigured_block_counts_by_system": hard_counts,
        "hard_and_soft_sparse_block_counts_match_all_systems": hard_counts == soft_counts,
        "dense_baseline_is_zero_sparsity_but_not_zero_regularization": True,
        "dense_baseline_architecture_capacity_matched": False,
        "dense_anomaly": (
            "One multiwell_strong_transition seed-8 dense checkpoint is relu with "
            "LAST_RELU=true; the other 169 are tanh with LAST_RELU=false."
        ),
        "structural_confound": (
            "The hard-block sparse K and the full-K sparse soft-block penalty both use "
            "the same preconfigured system-specific block count. Block-union results "
            "therefore cannot establish label-free discovery of local invariant subspaces."
        ),
    }


def geometry_results(frame: pd.DataFrame) -> dict[str, Any]:
    base = frame.loc[
        (frame.system_name == "gated_local_linear")
        & (frame.attractor_radius == 0.3)
        & (frame.support_definition == "topk:8")
        & (frame.projection_status == "ok")
    ]
    choices = ((FULL_SPARSE, "support"), (BLOCK, "family"), (DENSE, "support"))
    rows = {}
    for root, partition in choices:
        subset = base.loc[(base.root_label == root) & (base.partition_kind == partition)]
        observed = subset.loc[subset.control_kind == "observed", "state_fro_rel_error"]
        random = subset.loc[
            subset.control_kind == "random_count_matched", "state_fro_rel_error"
        ]
        rows[ALIASES[root]] = {
            "partition_kind": partition,
            "observed": scalar_summary(observed),
            "dependent_random_control_rows": scalar_summary(random),
        }
    return {
        "condition": "gated_local_linear, seed 0, radius 0.3, topk:8",
        "results": rows,
        "interpretation": (
            "Prospective motivation only. The evaluator fits new post-hoc centered slopes "
            "and projects them through encoder/decoder Jacobians; it does not restrict the "
            "checkpoint's learned K. Controls are dependent class/control rows from one seed."
        ),
    }


def build_payload() -> dict[str, Any]:
    authentication = verify_sources()
    centered = pd.read_csv(CENTERED_DIR / "centered_chart_mechanism_rows.csv")
    self_routed = pd.read_csv(SELF_ROUTED_DIR / "self_routed_forecasting_rows.csv")
    geometry = pd.read_csv(
        GEOMETRY_DIR / "true_jacobian_geometry_rows.csv", low_memory=False
    )
    assert len(centered) == 74369 and len(self_routed) == 24600 and len(geometry) == 198302
    centered_roster, self_roster = _roster(centered), _roster(self_routed)
    assert centered_roster == self_roster
    assert len(centered_roster) == 510
    return {
        "schema_version": 1,
        "audit_date": "2026-07-20",
        "status": "authenticated_historical_descriptive_evidence_only",
        "paper_promotion_authorized": False,
        "strongest_defensible_claim": (
            "Support-conditioned centered charts and autonomous support routing can reduce "
            "error relative to the originally learned global K in these historical artifacts."
        ),
        "claim_not_established": (
            "The packets do not establish that sparsity uniquely causes the effect or that "
            "one unmodified learned K contains multiple recovered local laws."
        ),
        "authentication": authentication,
        "roster": {
            "checkpoint_specs": len(centered_roster),
            "roots": list(ROOTS),
            "systems_per_root": 17,
            "seeds_per_system": 10,
            "centered_and_self_routed_rosters_identical": True,
        },
        "protocol_audit": {
            "centered": (
                "Support families/prototypes are built on the entire evaluation trajectory "
                "collection before a transition-level random 50/50 split. The construction "
                "is transductive, and transitions from each trajectory can enter both halves."
            ),
            "centered_q4": (
                "q4 uses benchmark-center distance margins and is evaluation-only label geometry."
            ),
            "same_k_gating": (
                "Input gating uses c+((z-c)P)K; submatrix gating uses c+(z-c)PKP. "
                "Support-specific post-hoc centers make these affine charts even though K is shared."
            ),
            "controls": (
                "Random controls are eight count-matched label permutations. K-means is fit only "
                "on the fit half and is omitted when the requested class count exceeds 16."
            ),
            "self_routed": (
                "Post-hoc operators fit on seed-42 trajectories and direct forecasts use fresh "
                "seed-314 trajectories. Routing is label-free and recomputed from predicted latent state."
            ),
            "selection": (
                "Two support rules and five depth strata were evaluated without a prediction card; "
                "foregrounding topk:8 is post-hoc."
            ),
        },
        "checkpoint_recipe_audit": checkpoint_audit(centered),
        "centered_chart": centered_results(centered),
        "self_routed_forecasting": self_routed_results(self_routed),
        "true_geometry_gated_local_linear": geometry_results(geometry),
        "rebuttal_decision": {
            "use_now": (
                "Use only as transparent historical motivation and as evidence that the concern "
                "was investigated; disclose post-hoc/transductive selection and survival filtering."
            ),
            "do_not_say": (
                "Do not call this a sparsity-only causal comparison, a clean all-tanh dense baseline, "
                "or proof that learned supports select invariant subspaces of one learned K."
            ),
            "prospective_experiment": (
                "Test held-out support-restricted learned-K closure in a basis-invariant true-geometry "
                "benchmark with no known-count block structure, a zero-penalty matched tanh dense control, "
                "a preregistered support rule, trajectory-disjoint splits, and common-survivor rollouts."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build_payload()
    serialized = json.dumps(
        payload, sort_keys=True, allow_nan=False, separators=(",", ":")
    ) + "\n"
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(serialized)
        print(f"wrote {OUTPUT}")
        return
    if not OUTPUT.is_file():
        raise FileNotFoundError(OUTPUT)
    frozen = json.loads(OUTPUT.read_text())
    if payload != frozen:
        raise ValueError(f"Historical local-law audit mismatch: {OUTPUT}")
    print(f"verified {OUTPUT}")


if __name__ == "__main__":
    main()
