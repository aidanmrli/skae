"""Render the fixed-seed Allen--Cahn basin/support contingency display."""

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


def ordered_contingency(
    records: pd.DataFrame,
) -> Tuple[np.ndarray, List[int], List[int]]:
    labels = records["global_basin_label"].to_numpy(dtype=np.int64)
    families = records["transferred_family"].to_numpy(dtype=np.int64)
    basin_ids = sorted(np.unique(labels).tolist())
    family_ids = sorted(family for family in np.unique(families).tolist() if family >= 0)

    def family_key(family: int) -> Tuple[int, int, int]:
        selected = labels[families == family]
        counts = {basin: int(np.sum(selected == basin)) for basin in basin_ids}
        majority = min(basin_ids, key=lambda basin: (-counts[basin], basin))
        return majority, -int(selected.size), family

    ordered_families = sorted(family_ids, key=family_key)
    if bool(np.any(families < 0)):
        ordered_families.append(-1)
    counts = np.zeros((len(basin_ids), len(ordered_families)), dtype=np.float64)
    for row, basin in enumerate(basin_ids):
        for column, family in enumerate(ordered_families):
            counts[row, column] = np.sum((labels == basin) & (families == family))
    row_totals = counts.sum(axis=1, keepdims=True)
    if bool(np.any(row_totals == 0)):
        raise ValueError("Every benchmark basin must have at least one trajectory.")
    return counts / row_totals, basin_ids, ordered_families


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
        raise ValueError("The frozen deep-interior slice must contain 130 trajectories.")
    panels = (
        ("Exact-dense tanh KAE", ordered_contingency(dense), "all trajectories"),
        ("Temporal sparse KAE", ordered_contingency(sparse), "all trajectories"),
        (
            "Temporal sparse KAE",
            ordered_contingency(deep),
            r"$\geq90\%$ modal-well interior",
        ),
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
        sharey=True,
    )
    image = None
    for axis, (model_label, contingency, slice_label) in zip(axes, panels):
        matrix, _basins, family_ids = contingency
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
        family_labels = [
            "unknown" if family < 0 else f"F{index + 1}"
            for index, family in enumerate(family_ids)
        ]
        axis.set_xticks(np.arange(len(family_ids)), labels=family_labels, rotation=45)
        axis.set_yticks(
            np.arange(len(basin_ids)), labels=[f"fate {basin + 1}" for basin in basin_ids]
        )
        axis.set_xlabel("Transferred support family")
        known_count = sum(family >= 0 for family in family_ids)
        family_word = "family" if known_count == 1 else "families"
        unknown_suffix = " + unknown" if -1 in family_ids else ""
        axis.set_title(
            f"{model_label}, {slice_label}\n"
            f"({known_count} {family_word}{unknown_suffix})"
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
    axes[0].set_ylabel("Evaluation-only final basin fate")
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
    colorbar.set_label("Fraction within fate")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight", metadata=PDF_METADATA)
    figure.savefig(output_png, dpi=300, bbox_inches="tight", metadata=PNG_METADATA)
    plt.close(figure)
