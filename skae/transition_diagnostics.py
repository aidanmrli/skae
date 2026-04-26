"""Transition-rich rollout diagnostics for basin and region label paths."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch


def _labels_to_int_list(labels: Sequence[int] | torch.Tensor) -> List[int]:
    """Convert a label sequence to a plain Python list of ints."""
    if isinstance(labels, torch.Tensor):
        return [int(item) for item in labels.detach().cpu().view(-1).tolist()]
    return [int(item) for item in labels]


def compress_label_path(
    labels: Sequence[int] | torch.Tensor,
    *,
    ignore_invalid: bool = True,
    invalid_label: int = -1,
) -> List[int]:
    """Collapse repeated labels into a compressed path."""
    compressed: List[int] = []
    for label in _labels_to_int_list(labels):
        if ignore_invalid and label == invalid_label:
            continue
        if compressed and label == compressed[-1]:
            continue
        compressed.append(label)
    return compressed


def transition_count(
    labels: Sequence[int] | torch.Tensor,
    *,
    ignore_invalid: bool = True,
    invalid_label: int = -1,
) -> int:
    """Count distinct label transitions along a trajectory."""
    path = compress_label_path(labels, ignore_invalid=ignore_invalid, invalid_label=invalid_label)
    return max(0, len(path) - 1)


def first_exit_step(
    labels: Sequence[int] | torch.Tensor,
    *,
    ignore_invalid: bool = True,
    invalid_label: int = -1,
) -> Optional[int]:
    """Return the first step index where the trajectory leaves its initial label."""
    label_list = _labels_to_int_list(labels)
    initial_label: Optional[int] = None
    for label in label_list:
        if ignore_invalid and label == invalid_label:
            continue
        initial_label = label
        break
    if initial_label is None:
        return None

    for step_index, label in enumerate(label_list):
        if ignore_invalid and label == invalid_label:
            continue
        if label != initial_label:
            return step_index
    return None


@dataclass
class TrajectoryTransitionSummary:
    """Per-trajectory path summary."""

    initial_label: Optional[int]
    final_label: Optional[int]
    compressed_path: List[int]
    transition_count: int
    has_transition: bool
    first_exit_step: Optional[int]


@dataclass
class TransitionLabelSummary:
    """Aggregate transition summary for a collection of trajectories."""

    num_trajectories: int
    crossing_fraction: float
    first_exit_rate: float
    mean_first_exit_step: Optional[float]
    mean_transition_count: float
    transition_count_histogram: Dict[str, int] = field(default_factory=dict)
    endpoint_histogram: Dict[str, int] = field(default_factory=dict)
    trajectory_summaries: List[TrajectoryTransitionSummary] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class TransitionComparisonSummary:
    """Comparison between true and predicted label sequences."""

    num_trajectories: int
    endpoint_accuracy: float
    path_exact_match_fraction: float
    transition_count_mae: float
    true_summary: TransitionLabelSummary
    pred_summary: TransitionLabelSummary

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_trajectories": self.num_trajectories,
            "endpoint_accuracy": self.endpoint_accuracy,
            "path_exact_match_fraction": self.path_exact_match_fraction,
            "transition_count_mae": self.transition_count_mae,
            "true_summary": self.true_summary.to_dict(),
            "pred_summary": self.pred_summary.to_dict(),
        }


def summarize_label_sequences(
    sequences: Iterable[Sequence[int] | torch.Tensor],
    *,
    ignore_invalid: bool = True,
    invalid_label: int = -1,
) -> TransitionLabelSummary:
    """Summarize transition behavior over many trajectories."""
    trajectory_summaries: List[TrajectoryTransitionSummary] = []
    transition_hist = Counter()
    endpoint_hist = Counter()
    num_crossing = 0
    num_exit = 0
    total_transition_count = 0
    exit_steps: List[int] = []

    for labels in sequences:
        path = compress_label_path(labels, ignore_invalid=ignore_invalid, invalid_label=invalid_label)
        transitions = max(0, len(path) - 1)
        exit_step = first_exit_step(labels, ignore_invalid=ignore_invalid, invalid_label=invalid_label)
        initial_label = path[0] if path else None
        final_label = path[-1] if path else None
        has_transition = transitions > 0

        if has_transition:
            num_crossing += 1
        if exit_step is not None:
            num_exit += 1
            exit_steps.append(int(exit_step))
        total_transition_count += transitions
        transition_hist[str(transitions)] += 1
        if final_label is not None:
            endpoint_hist[str(final_label)] += 1

        trajectory_summaries.append(
            TrajectoryTransitionSummary(
                initial_label=initial_label,
                final_label=final_label,
                compressed_path=path,
                transition_count=transitions,
                has_transition=has_transition,
                first_exit_step=exit_step,
            )
        )

    num_trajectories = len(trajectory_summaries)
    if num_trajectories == 0:
        return TransitionLabelSummary(
            num_trajectories=0,
            crossing_fraction=0.0,
            first_exit_rate=0.0,
            mean_first_exit_step=None,
            mean_transition_count=0.0,
        )

    return TransitionLabelSummary(
        num_trajectories=num_trajectories,
        crossing_fraction=float(num_crossing) / float(num_trajectories),
        first_exit_rate=float(num_exit) / float(num_trajectories),
        mean_first_exit_step=float(sum(exit_steps)) / float(len(exit_steps)) if exit_steps else None,
        mean_transition_count=float(total_transition_count) / float(num_trajectories),
        transition_count_histogram=dict(sorted(transition_hist.items(), key=lambda item: int(item[0]))),
        endpoint_histogram=dict(sorted(endpoint_hist.items(), key=lambda item: int(item[0]))),
        trajectory_summaries=trajectory_summaries,
    )


def compare_label_sequences(
    true_sequences: Sequence[Sequence[int] | torch.Tensor],
    pred_sequences: Sequence[Sequence[int] | torch.Tensor],
    *,
    ignore_invalid: bool = True,
    invalid_label: int = -1,
) -> TransitionComparisonSummary:
    """Compare true and predicted label-path behavior."""
    if len(true_sequences) != len(pred_sequences):
        raise ValueError("true_sequences and pred_sequences must have the same length.")

    true_summary = summarize_label_sequences(
        true_sequences,
        ignore_invalid=ignore_invalid,
        invalid_label=invalid_label,
    )
    pred_summary = summarize_label_sequences(
        pred_sequences,
        ignore_invalid=ignore_invalid,
        invalid_label=invalid_label,
    )

    endpoint_matches = 0
    path_matches = 0
    transition_count_error = 0.0
    for true_traj, pred_traj in zip(true_summary.trajectory_summaries, pred_summary.trajectory_summaries):
        if true_traj.final_label == pred_traj.final_label:
            endpoint_matches += 1
        if true_traj.compressed_path == pred_traj.compressed_path:
            path_matches += 1
        transition_count_error += abs(true_traj.transition_count - pred_traj.transition_count)

    num_trajectories = len(true_summary.trajectory_summaries)
    denom = float(max(num_trajectories, 1))
    return TransitionComparisonSummary(
        num_trajectories=num_trajectories,
        endpoint_accuracy=float(endpoint_matches) / denom,
        path_exact_match_fraction=float(path_matches) / denom,
        transition_count_mae=transition_count_error / denom,
        true_summary=true_summary,
        pred_summary=pred_summary,
    )


__all__ = [
    "TrajectoryTransitionSummary",
    "TransitionComparisonSummary",
    "TransitionLabelSummary",
    "compare_label_sequences",
    "compress_label_path",
    "first_exit_step",
    "summarize_label_sequences",
    "transition_count",
]
