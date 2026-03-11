"""Tests for Kuramoto mode-support audit task generation."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from tools.build_kuramoto_mode_support_audit_tasks import _build_rows


def _write_checkpoint(root: Path, variant: str, seed: int, run_id: str) -> None:
    checkpoint = (
        root
        / variant
        / "kuramoto"
        / "dt_0p00625"
        / f"seed_{seed}"
        / run_id
        / "checkpoint.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("checkpoint")


def test_kuramoto_mode_support_audit_builds_expected_grid(tmp_path):
    source_root = tmp_path / "source"
    scratch_root = tmp_path / "scratch"
    for variant in ("generic_sparse", "lista_dense", "lista_blockdiag"):
        for seed in (0, 1):
            _write_checkpoint(source_root, variant, seed, "20260308-000000")

    args = Namespace(
        phase_label="kuramoto_mode_support_audit",
        source_root=str(source_root),
        scratch_root=str(scratch_root),
        model_variants_csv="generic_sparse,lista_dense,lista_blockdiag",
        seeds_csv="0,1",
        sampling_strategies_csv="random,balanced",
        env_dt=0.00625,
        random_num_trajectories=256,
        balanced_trajectories_per_basin=16,
        balanced_target_raw_labels_csv="-2,-1,0,1,2",
        trajectory_length=256,
        long_rollout_steps=5000,
        support_threshold=1e-3,
        support_modes_csv="mean,majority,modal",
        threshold_sweep_modes_csv="mean,modal",
        thresholds_csv="1e-4,1e-3",
        max_attempts=20000,
        device="cuda",
    )

    rows = _build_rows(args)

    assert len(rows) == 3 * 2 * 2
    assert {row["family"] for row in rows} == {"generic", "dense_lista", "blockdiag_lista"}
    assert {row["sampling_strategy"] for row in rows} == {"random", "balanced"}

    random_rows = [row for row in rows if row["sampling_strategy"] == "random"]
    balanced_rows = [row for row in rows if row["sampling_strategy"] == "balanced"]

    assert {row["num_trajectories"] for row in random_rows} == {256}
    assert {row["target_raw_labels_csv"] for row in random_rows} == {""}
    assert {row["trajectories_per_basin"] for row in balanced_rows} == {16}
    assert {row["target_raw_labels_csv"] for row in balanced_rows} == {"-2,-1,0,1,2"}
    assert len({row["output_dir"] for row in rows}) == len(rows)


def test_kuramoto_mode_support_audit_prefers_latest_checkpoint(tmp_path):
    source_root = tmp_path / "source"
    scratch_root = tmp_path / "scratch"
    _write_checkpoint(source_root, "generic_sparse", 0, "20260308-000000")
    _write_checkpoint(source_root, "generic_sparse", 0, "20260309-120000")

    args = Namespace(
        phase_label="kuramoto_mode_support_audit",
        source_root=str(source_root),
        scratch_root=str(scratch_root),
        model_variants_csv="generic_sparse",
        seeds_csv="0",
        sampling_strategies_csv="random",
        env_dt=0.00625,
        random_num_trajectories=256,
        balanced_trajectories_per_basin=16,
        balanced_target_raw_labels_csv="-2,-1,0,1,2",
        trajectory_length=256,
        long_rollout_steps=5000,
        support_threshold=1e-3,
        support_modes_csv="mean,majority,modal",
        threshold_sweep_modes_csv="mean,modal",
        thresholds_csv="1e-4,1e-3",
        max_attempts=20000,
        device="cuda",
    )

    row = _build_rows(args)[0]

    assert row["checkpoint"].endswith("20260309-120000/checkpoint.pt")
