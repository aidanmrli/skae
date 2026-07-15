"""Tests for the documented Python/shell scratch precedence."""

from pathlib import Path

from skae.dysts_cache_profiles import default_dysts_cache_dir
from skae.runtime_paths import resolve_scratch_root


def test_explicit_skae_root_wins(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("SKAE_SCRATCH_ROOT", str(explicit))
    monkeypatch.setenv("SCRATCH", str(tmp_path / "cluster"))

    assert resolve_scratch_root(fallback=tmp_path / "fallback") == explicit


def test_cluster_scratch_matches_shell_precedence(monkeypatch, tmp_path):
    cluster = tmp_path / "cluster"
    monkeypatch.delenv("SKAE_SCRATCH_ROOT", raising=False)
    monkeypatch.setenv("SCRATCH", str(cluster))

    assert resolve_scratch_root(fallback=tmp_path / "fallback") == cluster / "skae"
    assert default_dysts_cache_dir() == str(cluster / "skae/dysts_native_cache")


def test_missing_user_falls_back_locally(monkeypatch, tmp_path):
    fallback = tmp_path / "fallback"
    monkeypatch.delenv("SKAE_SCRATCH_ROOT", raising=False)
    monkeypatch.delenv("SCRATCH", raising=False)
    monkeypatch.setenv("USER", "")

    assert resolve_scratch_root(fallback=fallback) == Path(fallback)
