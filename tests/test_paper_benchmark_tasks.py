"""Tests for paper benchmark task generation."""

from __future__ import annotations

from argparse import Namespace

from tools.build_paper_benchmark_tasks import _build_rows


def _base_args(dt_table: str | None) -> Namespace:
    return Namespace(
        phase="rescue",
        phase_label="rescue_pass1",
        output_tsv="unused.tsv",
        output_manifest_json=None,
        systems_csv=None,
        model_variants_csv=None,
        seeds_csv=None,
        num_steps=None,
        eval_profile=None,
        dt_table=dt_table,
    )


def test_rescue_task_build_uses_requested_dt_subset(tmp_path):
    dt_table = tmp_path / "dt_request.tsv"
    dt_table.write_text(
        "\n".join(
            [
                "system_key\tenv_name\trequested_dt\tpass_index",
                "lotka_volterra\tlotka_volterra\t0.005\t1",
                "kuramoto\tkuramoto\t0.025\t1",
            ]
        )
        + "\n"
    )

    rows = _build_rows(_base_args(str(dt_table)))

    assert len(rows) == 2 * 3
    assert {row["model_variant"] for row in rows} == {"generic_sparse"}
    assert {row["system_key"] for row in rows} == {"lotka_volterra", "kuramoto"}
    assert {row["env_dt"] for row in rows if row["system_key"] == "lotka_volterra"} == {0.005}
    assert {row["env_dt"] for row in rows if row["system_key"] == "kuramoto"} == {0.025}


def test_rescue_task_build_with_empty_request_table_is_empty(tmp_path):
    dt_table = tmp_path / "dt_request_empty.tsv"
    dt_table.write_text("system_key\tenv_name\trequested_dt\tpass_index\n")

    rows = _build_rows(_base_args(str(dt_table)))

    assert rows == []
