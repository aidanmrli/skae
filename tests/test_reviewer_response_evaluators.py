from __future__ import annotations

import numpy as np

from tools import evaluate_transition_rich_controlled_transfer_switching as ctrl
from tools import evaluate_transition_rich_true_jacobian_geometry as true_geo


def test_true_geometry_emits_class_fixed_point_pairs_when_class_spans_attractors():
    fixed_points = [
        true_geo.FixedPointRecord(
            basin_id=0,
            point=np.asarray([0.0, 0.0], dtype=np.float32),
            source="test",
            step_residual=0.0,
            continuous_residual=None,
        ),
        true_geo.FixedPointRecord(
            basin_id=1,
            point=np.asarray([1.0, 0.0], dtype=np.float32),
            source="test",
            step_residual=0.0,
            continuous_residual=None,
        ),
    ]
    rows = true_geo._evaluate_partition_rows(
        common={
            "root_label": "root",
            "support_definition": "topk:1",
            "partition_kind": "family",
            "control_kind": "observed",
            "max_partition_classes": 8,
        },
        model=None,
        device="cpu",
        true_jacobians={},
        x_all=np.zeros((4, 2), dtype=np.float64),
        y_all=np.zeros((4, 2), dtype=np.float64),
        labels_all=np.asarray(["shared", "shared", "shared", "shared"], dtype=object),
        near_fixed_idx=np.asarray([0, 0, 1, 1], dtype=np.int64),
        near_fixed_dist=np.asarray([0.1, 0.2, 0.1, 0.2], dtype=np.float32),
        support_sizes={"shared": 1.0},
        fixed_points=fixed_points,
        projection_cache={},
        attractor_radius=0.5,
        min_operator_transitions=3,
        ridge_lambda=1e-4,
    )

    assert len(rows) == 2
    assert {row["fixed_point_basin_id"] for row in rows} == {0, 1}
    assert all(row["transition_count"] == 2.0 for row in rows)
    assert all(row["skip_reason"] == "transition_count<3" for row in rows)


def test_controlled_transfer_summary_uses_switch_rows_for_pre_source(tmp_path):
    rows = [
        {
            "root_label": "root",
            "support_definition": "topk:1",
            "object_kind": "support",
            "control_kind": "transfer",
            "status": "ok",
            "source_target_same_object": True,
            "pre_source_dominance": 1.0,
            "post_target_dominance": 1.0,
            "post_phase_target_dominance": 1.0,
            "crossing_lag_steps": 0,
            "post_phase_crossing_lag_steps": 0,
            "source_exit_to_target_entry_steps": 1,
            "premature_switch_rate": 0.0,
            "post_chatter_switch_rate": 0.0,
        },
        {
            "root_label": "root",
            "support_definition": "topk:1",
            "object_kind": "support",
            "control_kind": "transfer",
            "status": "ok",
            "source_target_same_object": False,
            "pre_source_dominance": 0.25,
            "post_target_dominance": 0.5,
            "post_phase_target_dominance": 0.75,
            "crossing_lag_steps": 2,
            "post_phase_crossing_lag_steps": 3,
            "source_exit_to_target_entry_steps": 4,
            "premature_switch_rate": 0.1,
            "post_chatter_switch_rate": 0.2,
        },
    ]
    summary_path = tmp_path / "summary.md"
    ctrl._write_summary(summary_path, rows)
    summary = summary_path.read_text()

    assert "| `root` | `topk:1` | `support` | `transfer` | 2 | 1 | 0.2500 |" in summary
