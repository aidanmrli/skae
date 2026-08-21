"""Contracts and parsers for the exact-shape utilization pilot.

This module intentionally has no training implementation.  The scientific
workload is delegated to the repository's ``skae-train`` entry point with a
fixed argument list.  The surrounding pilot records identity, progress,
telemetry, and profiler evidence without changing B, z, H, seed, or any
scientific loss setting.
"""

from __future__ import annotations

import csv
import hashlib
import json
import datetime as _datetime
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


NCU_SMO_METRIC = "sm__warps_active.avg.pct_of_peak_sustained_active"
SCHEMA_VERSION = 1

# This is the historical GenericKM gated-transfer-linear task identity.  The
# pilot's warmup/timed/profile step counts are deliberately separate from the
# historical 200k-step workload count and are never allowed to alter shape.
TASK_IDENTITY: dict[str, Any] = {
    "config_name": "generic_sparse",
    "model_name": "GenericKM",
    "environment": "gated_transfer_linear",
    "env_dt": 0.04,
    "seed": 4,
    "batch_size": 256,
    "target_size": 256,
    "sequence_length": 8,
    "observation_size": 2,
    "k_structure": "dense",
    "res_coeff": 1.0,
    "reconst_coeff": 0.03,
    "pred_coeff": 1.0,
    "sparsity_coeff": 0.0,
    "lr": 5e-5,
    "k_matrix_lr": 5e-6,
    "weight_decay": 1e-4,
    "hard_init_oversample": True,
    "hard_init_fraction": 0.5,
    "hard_init_pool_size": 1024,
    "hard_init_num_candidates": 4096,
    "hard_init_probe_steps": 32,
    "hard_init_num_perturbations": 4,
    "hard_init_perturb_scale": 0.04,
    "hard_init_transient_window": 8,
    "hard_init_transient_weight": 0.5,
    "hard_init_jitter_scale": 0.25,
    "historical_num_steps": 200000,
}


class MetricUnavailable(RuntimeError):
    """Raised when a profiler output has no usable SM occupancy samples."""


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for identity hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write a text artifact atomically on the same filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a canonical, fsynced JSON receipt with an atomic rename."""

    atomic_write_text(
        path,
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
    )


def exact_task_identity(label: str = "base", variant: str = "base") -> dict[str, Any]:
    """Return the fixed scientific identity plus a non-scientific comparison label."""

    if not label or any(character.isspace() for character in label):
        raise ValueError("pilot label must be non-empty and contain no whitespace")
    if not variant or any(character.isspace() for character in variant):
        raise ValueError("pilot variant must be non-empty and contain no whitespace")
    identity = dict(TASK_IDENTITY)
    identity["comparison"] = {"label": label, "variant": variant}
    identity["task_identity_sha256"] = sha256_bytes(canonical_json(identity).encode())
    return identity


def validate_task_identity(identity: Mapping[str, Any]) -> None:
    """Fail closed if any scientific task field differs from the frozen task."""

    for key, expected in TASK_IDENTITY.items():
        actual = identity.get(key)
        if actual != expected:
            raise ValueError(f"task identity mismatch for {key}: {actual!r} != {expected!r}")
    comparison = identity.get("comparison")
    if not isinstance(comparison, Mapping):
        raise ValueError("task identity is missing the comparison label/variant")
    for key in ("label", "variant"):
        value = comparison.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"comparison.{key} must be a non-empty string")
    without_hash = dict(identity)
    supplied_hash = without_hash.pop("task_identity_sha256", None)
    expected_hash = sha256_bytes(canonical_json(without_hash).encode())
    if supplied_hash != expected_hash:
        raise ValueError("task identity hash does not match its fields")


def build_source_manifest(repo_root: Path, output_path: Path) -> dict[str, Any]:
    """Hash committed source blobs and refuse dirty or untracked code."""

    root = repo_root.resolve()
    for command, message in (
        (["git", "-C", str(root), "diff", "--quiet"], "unstaged changes"),
        (["git", "-C", str(root), "diff", "--cached", "--quiet"], "staged changes"),
    ):
        result = subprocess.run(command)
        if result.returncode != 0:
            raise RuntimeError(f"refusing pilot with {message}")
    untracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    if untracked:
        raise RuntimeError(f"refusing pilot with untracked files: {untracked[:5]}")
    result = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-r", "-z", "--full-tree", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
    )
    lines = []
    for entry in result.stdout.decode().split("\0"):
        if not entry or "\t" not in entry:
            continue
        metadata, relative = entry.split("\t", 1)
        fields = metadata.split()
        if len(fields) >= 3 and fields[1] == "blob":
            lines.append(f"{fields[2]}  {relative}\n")
    manifest_text = "".join(lines)
    atomic_write_text(output_path, manifest_text)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    return {
        "path": str(output_path),
        "sha256": sha256_bytes(manifest_text.encode()),
        "file_count": len(lines),
        "git_commit": commit,
        "content_source": "committed HEAD git blobs",
        "working_tree_clean": True,
    }


def _header_and_rows(path: Path) -> tuple[list[str], Iterable[list[str]]]:
    """Find the first Nsight CSV header, ignoring ``==PROF==`` preamble lines."""

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.reader(handle, skipinitialspace=True))
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if "Metric Name" in row and "Metric Value" in row
        ),
        None,
    )
    if header_index is None:
        raise MetricUnavailable(f"Nsight CSV has no metric header: {path}")
    return rows[header_index], rows[header_index + 1 :]


def _parse_metric_value(value: str) -> float | None:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] == '"':
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.replace(",", "")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()
    if not cleaned or cleaned.upper() in {"N/A", "NA", "NOT_SUPPORTED", "-"}:
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def ncu_counter_permission_error(path: Path) -> str | None:
    try:
        raw_output = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if "ERR_NVGPUCTRPERM" in raw_output:
        return "Nsight Compute counter permission denied: ERR_NVGPUCTRPERM"
    return None


def parse_ncu_smo_csv(path: Path, metric: str = NCU_SMO_METRIC) -> dict[str, Any]:
    """Parse actual Nsight metric rows; never substitute GPU-Util.

    Nsight versions differ in their preamble and in whether the metric value
    includes a percent sign.  The parser accepts both forms but requires a
    row whose ``Metric Name`` exactly matches the requested SM occupancy
    metric and whose value is finite.  Missing or malformed rows are
    fail-closed via :class:`MetricUnavailable`.
    """

    try:
        raw_output = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise MetricUnavailable(f"Nsight CSV cannot be read: {path}: {exc}") from exc
    if "ERR_NVGPUCTRPERM" in raw_output:
        raise MetricUnavailable(
            "Nsight Compute counter permission denied: ERR_NVGPUCTRPERM"
        )
    header, rows = _header_and_rows(path)
    columns = {name.strip(): index for index, name in enumerate(header)}
    name_index = columns["Metric Name"]
    value_index = columns["Metric Value"]
    unit_index = columns.get("Metric Unit")
    if unit_index is None:
        raise MetricUnavailable("Nsight CSV is missing the Metric Unit column")
    values: list[float] = []
    units: set[str] = set()
    for row in rows:
        if max(name_index, value_index) >= len(row):
            continue
        if row[name_index].strip() != metric:
            continue
        if unit_index >= len(row):
            continue
        unit = row[unit_index].strip().lower()
        if unit not in {"%", "percent", "percentage", "pct"}:
            continue
        parsed = _parse_metric_value(row[value_index])
        if parsed is not None and 0.0 <= parsed <= 100.0:
            values.append(parsed)
            units.add(unit)
    if not values:
        raise MetricUnavailable(
            f"Nsight CSV has no finite rows for {metric}; GPU-Util is not SMO"
        )
    return {
        "metric": metric,
        "unit": "percent_of_peak_sustained_active",
        "observed_units": sorted(units),
        "sample_count": len(values),
        "values_percent": values,
        "mean_percent": sum(values) / len(values),
        "min_percent": min(values),
        "max_percent": max(values),
        "source": "Nsight Compute",
        "aggregation": "unweighted_per_kernel_arithmetic_mean",
        "duration_weighted": False,
        "gpu_utilization_as_smo": False,
    }


# Descriptive aliases keep downstream adjudication code independent of the
# filename chosen for the raw Nsight artifact.
parse_ncu_metrics = parse_ncu_smo_csv
require_ncu_smo_metrics = parse_ncu_smo_csv


def _sample_unix(timestamp: str) -> float | None:
    value = timestamp.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return _datetime.datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue
    return None


def parse_nvidia_smi_csv(
    path: Path,
    *,
    start_unix: float | None = None,
    end_unix: float | None = None,
) -> dict[str, Any]:
    """Summarize raw 1-second nvidia-smi samples without relabeling SMO."""

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    utilization: list[float] = []
    memory: list[float] = []
    power: list[float] = []
    selected_rows = []
    parsed_timestamps = 0
    for row in rows:
        timestamp = row.get("timestamp", "")
        sample_time = _sample_unix(timestamp)
        if sample_time is not None:
            parsed_timestamps += 1
        elif start_unix is not None or end_unix is not None:
            continue
        if start_unix is not None and sample_time is not None and sample_time < start_unix:
            continue
        if end_unix is not None and sample_time is not None and sample_time > end_unix:
            continue
        selected_rows.append(row)
    if rows and (start_unix is not None or end_unix is not None) and parsed_timestamps == 0:
        raise MetricUnavailable(f"nvidia-smi timestamps could not be filtered for {path}")
    for row in selected_rows:
        def number(key: str) -> float | None:
            value = row.get(key, "").strip().replace("%", "")
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(parsed):
                return None
            if key == "utilization.gpu [%]" and not 0.0 <= parsed <= 100.0:
                return None
            return parsed

        for target, key in (
            (utilization, "utilization.gpu [%]"),
            (memory, "memory.used [MiB]"),
            (power, "power.draw [W]"),
        ):
            parsed = number(key)
            if parsed is not None:
                target.append(parsed)
    if not utilization:
        raise MetricUnavailable(
            f"nvidia-smi has no finite measured GPUUtil samples: {path}"
        )
    return {
        "sample_count": len(selected_rows),
        "window_start_unix": start_unix,
        "window_end_unix": end_unix,
        "window_filter_applied": start_unix is not None or end_unix is not None,
        "gpu_utilization_mean_percent": (
            sum(utilization) / len(utilization) if utilization else None
        ),
        "memory_used_mean_mib": sum(memory) / len(memory) if memory else None,
        "memory_used_peak_mib": max(memory) if memory else None,
        "power_draw_mean_w": sum(power) / len(power) if power else None,
        "raw_source": "nvidia-smi 1-second samples",
        "gpu_utilization_is_smo": False,
    }


def phase_telemetry(
    path: Path,
    *,
    phase: str,
    child_return_code: int,
    start_unix: float | None = None,
    end_unix: float | None = None,
) -> dict[str, Any]:
    """Parse telemetry, tolerating only profile-window sampling gaps."""

    try:
        return parse_nvidia_smi_csv(
            path, start_unix=start_unix, end_unix=end_unix
        )
    except (MetricUnavailable, OSError) as exc:
        if phase != "profile":
            if child_return_code != 0:
                raise RuntimeError(
                    f"{phase} child exited with status {child_return_code}; "
                    f"telemetry was unavailable: {exc}"
                ) from exc
            raise
        return {
            "status": "missing",
            "availability": "best_effort_profile_window",
            "error": f"{type(exc).__name__}: {exc}",
            "path": str(path),
            "sample_count": 0,
            "window_start_unix": start_unix,
            "window_end_unix": end_unix,
            "window_filter_applied": start_unix is not None or end_unix is not None,
            "gpu_utilization_mean_percent": None,
            "memory_used_mean_mib": None,
            "memory_used_peak_mib": None,
            "power_draw_mean_w": None,
            "raw_source": "nvidia-smi 1-second samples",
            "gpu_utilization_is_smo": False,
        }


def rgu_accounting(
    *, gpu_count: int, measured_elapsed_seconds: float, allocation_elapsed_seconds: float
) -> dict[str, Any]:
    """Keep measured timing separate from allocated wall time for RGU accounting."""

    measured_hours = float(measured_elapsed_seconds) / 3600.0
    allocation_hours = float(allocation_elapsed_seconds) / 3600.0
    return {
        "gpu_count": int(gpu_count),
        "measured_elapsed_hours": measured_hours,
        "allocation_wall_elapsed_hours": allocation_hours,
        "rgu_per_gpu": None,
        "absolute_rgu_hours": None,
        "formula": "rgu_per_gpu * gpu_count * allocation_wall_elapsed_hours",
        "same_gpu_elapsed_ratio_formula": (
            "candidate_allocation_wall_seconds / baseline_allocation_wall_seconds"
        ),
        "mapping_status": "unknown_absolute_rgu_coefficient",
    }
