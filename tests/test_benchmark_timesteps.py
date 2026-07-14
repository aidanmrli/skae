"""Tests for benchmark timestep lookup independent of retired manifests."""

from skae.benchmarks.timesteps import resolve_system_default_dt


def test_resolve_configured_controlled_timestep():
    assert resolve_system_default_dt("gated_local_linear") == 0.04


def test_resolve_dysts_native_timestep(monkeypatch):
    monkeypatch.setattr(
        "skae.benchmarks.dysts_adapter.get_dysts_system_metadata",
        lambda name: {"dt": 0.125, "name": name},
    )

    assert resolve_system_default_dt("dysts:ExampleFlow") == 0.125
