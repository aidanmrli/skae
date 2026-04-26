"""Schema checks for transition-rich diagnostic summaries."""

from __future__ import annotations

from skae.transition_diagnostics import compare_label_sequences, summarize_label_sequences


def test_transition_label_summary_emits_required_fields():
    summary = summarize_label_sequences(
        [
            [0, 0, 1, 1],
            [1, 1, 1, 1],
        ]
    ).to_dict()

    assert "num_trajectories" in summary
    assert "crossing_fraction" in summary
    assert "first_exit_rate" in summary
    assert "mean_first_exit_step" in summary
    assert "mean_transition_count" in summary
    assert "trajectory_summaries" in summary
    assert "compressed_path" in summary["trajectory_summaries"][0]


def test_transition_comparison_summary_emits_required_fields():
    comparison = compare_label_sequences(
        [[0, 0, 1], [1, 1, 1]],
        [[0, 1, 1], [1, 1, 1]],
    ).to_dict()

    assert "endpoint_accuracy" in comparison
    assert "path_exact_match_fraction" in comparison
    assert "transition_count_mae" in comparison
    assert "true_summary" in comparison
    assert "pred_summary" in comparison
