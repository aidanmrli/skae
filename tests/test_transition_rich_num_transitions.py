"""Checks for transition-count aggregates used in transition-rich diagnostics."""

from __future__ import annotations

from skae.transition_diagnostics import summarize_label_sequences


def test_crossing_fraction_matches_number_of_crossing_trajectories():
    summary = summarize_label_sequences(
        [
            [0, 0, 0, 0],  # no transition
            [0, 1, 1, 1],  # one transition
            [2, 2, 1, 1],  # one transition
            [1, 1, 1, 1],  # no transition
            [2, 0, 2, 2],  # two transitions
        ]
    )

    assert summary.num_trajectories == 5
    assert summary.crossing_fraction == 3 / 5
    assert summary.transition_count_histogram == {"0": 2, "1": 2, "2": 1}
