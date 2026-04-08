"""Unit tests for transition-rich label-path crossing metrics."""

from __future__ import annotations

from skae.transition_diagnostics import compress_label_path, first_exit_step, transition_count


def test_compress_label_path_removes_consecutive_repeats():
    labels = [0, 0, 1, 1, 1, 2, 2, 0]
    assert compress_label_path(labels) == [0, 1, 2, 0]


def test_transition_count_counts_distinct_path_changes():
    labels = [0, 0, 1, 1, 2, 2, 2, 1]
    assert transition_count(labels) == 3


def test_first_exit_step_finds_first_change_from_initial_label():
    labels = [2, 2, 2, 1, 1, 0]
    assert first_exit_step(labels) == 3
