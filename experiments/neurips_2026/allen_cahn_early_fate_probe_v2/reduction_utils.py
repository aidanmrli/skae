"""Small outcome-reduction helpers fixed before semantic access."""

from __future__ import annotations

import numpy as np
import torch

from .features import nearest_well_maps


def labels_and_eligibility(
    flat_fields: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    maps = nearest_well_maps(flat_fields)
    counts = torch.stack([(maps == index).sum((1, 2)) for index in range(4)], dim=1)
    sorted_counts = counts.sort(dim=1, descending=True).values
    eligible = sorted_counts[:, 0] != sorted_counts[:, 1]
    return counts.argmax(1).long(), eligible.bool()


def occupancy_diagnostics(flat_fields: torch.Tensor) -> dict[str, object]:
    maps = nearest_well_maps(flat_fields)
    counts = torch.stack([(maps == index).sum((1, 2)) for index in range(4)], dim=1)
    sorted_counts = counts.sort(dim=1, descending=True).values
    ties = sorted_counts[:, 0] == sorted_counts[:, 1]
    labels = counts.argmax(1).cpu().numpy()
    margins = (sorted_counts[:, 0] - sorted_counts[:, 1]).double() / 256.0
    eligible = ~ties
    eligible_top = sorted_counts[eligible, 0].double() / 256.0
    return {
        "row_count": int(counts.shape[0]),
        "exact_top_count_ties": int(ties.sum()),
        "eligible_count": int((~ties).sum()),
        "eligible_fraction": float((~ties).double().mean()),
        "fraction_top_occupancy_at_least_half": float(
            (sorted_counts[:, 0].double() / 256.0 >= 0.5).double().mean()
        ),
        "eligible_fraction_top_occupancy_at_least_half": float(
            (eligible_top >= 0.5).double().mean() if eligible_top.numel() else 0.0
        ),
        "mean_top_occupancy": float(sorted_counts[:, 0].double().mean() / 256.0),
        "mean_top1_minus_top2_margin": float(
            margins.mean()
        ),
        "top1_minus_top2_margin_quantiles": [
            float(value) for value in torch.quantile(margins, torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0], dtype=torch.float64))
        ],
        "class_counts": np.bincount(labels, minlength=4).tolist(),
    }


def split_time_matrix(
    matrix: np.ndarray, *, time_index: int, layout: dict[str, object]
) -> tuple[np.ndarray, list[np.ndarray]]:
    rows = int(layout["rows_per_time"])
    block = matrix[time_index * rows : (time_index + 1) * rows]
    train_start, train_stop = layout["train_slice"]
    tests = [block[start:stop] for start, stop in layout["test_slices"]]
    return block[train_start:train_stop], tests


def relative_pass(summary: dict[str, object], adjusted_p: float) -> bool:
    return bool(
        summary["mean_difference"] >= 0.05
        and summary["model_seed_wins"] >= 9
        and summary["dataset_seed_wins"] >= 3
        and summary["two_way_bootstrap_interval"][0] > 0.0
        and adjusted_p <= 0.05
    )
