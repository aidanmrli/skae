"""Render the fixed-seed Allen--Cahn finite-time fate/support display."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.neurips_2026.evidence.highdimensional_rendering import (
    PDF_METADATA,
    PNG_METADATA,
)


DEEP_THRESHOLD = 0.9


def outcome_independent_family_order(records: pd.DataFrame) -> List[int]:
    """Return frozen codebook IDs in numeric order, with ``unknown`` last.

    ``transferred_family`` is the raw index of a representative in the
    training-fitted codebook.  Numeric codebook order therefore does not use
    evaluation-only fate labels or any forecasting outcome.
    """

    if "transferred_family" not in records:
        raise ValueError("Records are missing transferred_family")
    raw = records["transferred_family"].to_numpy()
    if raw.size == 0 or not np.isfinite(raw.astype(np.float64)).all():
        raise ValueError("Transferred family IDs must be finite and nonempty")
    families = raw.astype(np.int64)
    if not np.array_equal(raw.astype(np.float64), families.astype(np.float64)):
        raise ValueError("Transferred family IDs must be integers")
    if bool(np.any(families < -1)):
        raise ValueError("Only -1 may encode an unknown transferred family")
    known = sorted(family for family in np.unique(families).tolist() if family >= 0)
    return known + ([-1] if bool(np.any(families == -1)) else [])


def ordered_contingency(
    records: pd.DataFrame,
    *,
    family_order: List[int] | None = None,
) -> Tuple[np.ndarray, List[int], List[int]]:
    labels = records["global_basin_label"].to_numpy(dtype=np.int64)
    families = records["transferred_family"].to_numpy(dtype=np.int64)
    basin_ids = sorted(np.unique(labels).tolist())
    ordered_families = (
        outcome_independent_family_order(records)
        if family_order is None
        else [int(family) for family in family_order]
    )
    if len(ordered_families) != len(set(ordered_families)):
        raise ValueError("Family order contains duplicate IDs")
    if any(family < -1 for family in ordered_families):
        raise ValueError("Only -1 may encode an unknown transferred family")
    missing = set(np.unique(families).tolist()) - set(ordered_families)
    if missing:
        raise ValueError(
            f"Family order omits observed transferred IDs: {sorted(missing)}"
        )
    counts = np.zeros((len(basin_ids), len(ordered_families)), dtype=np.float64)
    for row, basin in enumerate(basin_ids):
        for column, family in enumerate(ordered_families):
            counts[row, column] = np.sum((labels == basin) & (families == family))
    row_totals = counts.sum(axis=1, keepdims=True)
    if bool(np.any(row_totals == 0)):
        raise ValueError("Every benchmark basin must have at least one trajectory.")
    return counts / row_totals, basin_ids, ordered_families


def fate_row_labels(records: pd.DataFrame, basin_ids: List[int]) -> List[str]:
    """Label every evaluation fate with its panel-specific trajectory count."""

    labels = records["global_basin_label"].to_numpy(dtype=np.int64)
    return [
        f"fate {basin + 1} (n={int(np.sum(labels == basin))})"
        for basin in basin_ids
    ]


def render_contingency(
    records: pd.DataFrame,
    *,
    output_pdf: Path,
    output_png: Path,
) -> None:
    required = {
        "model",
        "seed",
        "trajectory_index",
        "global_basin_label",
        "transferred_family",
        "majority_fraction",
    }
    if not required.issubset(records.columns):
        raise ValueError(f"Missing fixed-seed columns: {sorted(required - set(records))}")
    if set(records["model"]) != {"dense", "sparse"} or set(records["seed"]) != {21}:
        raise ValueError("The display packet must contain dense/sparse seed 21 only.")
    dense = records.loc[records["model"] == "dense"]
    sparse = records.loc[records["model"] == "sparse"]
    deep = sparse.loc[sparse["majority_fraction"] >= DEEP_THRESHOLD]
    if deep.shape[0] != 130:
        raise ValueError(
            "The frozen single-well-dominated slice must contain 130 trajectories."
        )
    dense_order = outcome_independent_family_order(dense)
    sparse_order = outcome_independent_family_order(sparse)
    panel_specs = (
        ("Exact-dense tanh KAE", dense, dense_order, "all T=20 final states", "D"),
        ("Temporal sparse KAE", sparse, sparse_order, "all T=20 final states", "S"),
        (
            "Temporal sparse KAE",
            deep,
            sparse_order,
            r"$\geq90\%$ final modal-well occupancy",
            "S",
        ),
    )
    panels = tuple(
        (
            model_label,
            ordered_contingency(selected, family_order=family_order),
            slice_label,
            prefix,
            selected,
        )
        for model_label, selected, family_order, slice_label, prefix in panel_specs
    )
    basin_ids = panels[0][1][1]
    if any(panel[1][1] != basin_ids for panel in panels[1:]):
        raise ValueError("Dense and sparse records do not contain the same basins.")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    widths = [max(1.6, 0.42 * len(panel[1][2])) for panel in panels]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(11.6, 3.0),
        constrained_layout=True,
        gridspec_kw={"width_ratios": widths},
        sharey=False,
    )
    image = None
    for axis, (model_label, contingency, slice_label, prefix, selected) in zip(
        axes, panels
    ):
        matrix, _basins, family_ids = contingency
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
        family_labels = [
            "unknown" if family < 0 else f"{prefix}{family}"
            for family in family_ids
        ]
        axis.set_xticks(np.arange(len(family_ids)), labels=family_labels, rotation=45)
        axis.set_yticks(
            np.arange(len(basin_ids)), labels=fate_row_labels(selected, basin_ids)
        )
        axis.tick_params(axis="y", labelsize=7.5)
        axis.set_xlabel("Transferred support family (raw codebook ID)")
        observed_ids = set(selected["transferred_family"].astype(int).tolist())
        known_count = sum(family >= 0 for family in observed_ids)
        family_word = "family" if known_count == 1 else "families"
        unknown_suffix = " + unknown" if -1 in observed_ids else ""
        active_prefix = (
            "active "
            if len(family_ids) > len(observed_ids)
            else ""
        )
        axis.set_title(
            f"{model_label}, {slice_label}\n"
            f"({known_count} {active_prefix}{family_word}{unknown_suffix})"
        )
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = float(matrix[row, column])
                if value >= 0.045:
                    axis.text(
                        column,
                        row,
                        f"{100.0 * value:.0f}%",
                        ha="center",
                        va="center",
                        color="white" if value >= 0.55 else "#111111",
                        fontsize=7,
                    )
    axes[0].set_ylabel("Evaluation-only finite-time modal-well fate")
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
    colorbar.set_label("Fraction within fate")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight", metadata=PDF_METADATA)
    figure.savefig(output_png, dpi=300, bbox_inches="tight", metadata=PNG_METADATA)
    plt.close(figure)
