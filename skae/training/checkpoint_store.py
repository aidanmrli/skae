"""Durable generation storage for complete training checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from skae.training.checkpoint_validation import (
    CHECKPOINT_SCHEMA_VERSION,
    valid_complete_payload,
)


_MANIFEST_PATTERN = re.compile(r"^checkpoint-(\d{8})\.manifest\.json$")
_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d{8})\.pt$")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is invalid, incomplete, or unsafe to resume."""


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as source_handle, temporary.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(path, map_location=map_location)


def _is_cpu_map_location(map_location: Any) -> bool:
    return map_location == "cpu" or (
        isinstance(map_location, torch.device) and map_location.type == "cpu"
    )


def _manifest_generation(path: Path) -> Optional[int]:
    match = _MANIFEST_PATTERN.fullmatch(path.name)
    return None if match is None else int(match.group(1))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_manifest_payload(
    path: Path, root: Path, map_location: Any = "cpu"
) -> Optional[Dict[str, Any]]:
    """Validate the manifest, checksum, and complete state, failing closed."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return None
        generation_from_name = _manifest_generation(path)
        if generation_from_name is None:
            return None
        if (
            not _is_int(manifest.get("schema_version"))
            or manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION
        ):
            return None
        if (
            not _is_int(manifest.get("generation"))
            or manifest["generation"] != generation_from_name
        ):
            return None
        if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
            return None
        checkpoint_name = manifest.get("checkpoint_file")
        checkpoint_match = (
            _CHECKPOINT_PATTERN.fullmatch(checkpoint_name)
            if isinstance(checkpoint_name, str)
            else None
        )
        if checkpoint_match is None or int(checkpoint_match.group(1)) != generation_from_name:
            return None
        digest = manifest.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            return None
        if not _is_int(manifest.get("size_bytes")) or manifest["size_bytes"] < 0:
            return None
        if not _is_int(manifest.get("next_step")) or manifest["next_step"] < 0:
            return None
        checkpoint = root / checkpoint_name
        if not checkpoint.is_file() or checkpoint.stat().st_size != manifest["size_bytes"]:
            return None
        if _sha256(checkpoint) != digest:
            return None
        payload = _torch_load(checkpoint, map_location=map_location)
        if not valid_complete_payload(payload):
            return None
        if payload["run_id"] != manifest["run_id"]:
            return None
        if payload["generation"] != generation_from_name:
            return None
        if payload["next_step"] != manifest["next_step"]:
            return None
        return {"manifest": manifest, "payload": payload, "path": checkpoint}
    except Exception:
        # Includes malformed JSON, bad UTF-8, EOF, pickle/unpickling errors,
        # non-dict torch payloads, missing keys, and filesystem races.
        return None


class CheckpointManager:
    """Generation store with valid-generation retention and permanent fallback."""

    def __init__(
        self,
        root: Path | str,
        *,
        retention: int = 3,
        permanent_root: Optional[Path | str] = None,
        run_id: Optional[str] = None,
        min_free_bytes: int = 64 * 1024 * 1024,
        checkpoint_interval: int = 100,
    ) -> None:
        if int(retention) < 2:
            raise ValueError("checkpoint retention must keep at least two generations")
        if int(checkpoint_interval) < 1:
            raise ValueError("checkpoint interval must be >= 1")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.retention = int(retention)
        self.min_free_bytes = max(0, int(min_free_bytes))
        self.checkpoint_interval = int(checkpoint_interval)
        self.permanent_root = (
            None if permanent_root is None else Path(permanent_root).expanduser().resolve()
        )
        if self.permanent_root is not None:
            self.permanent_root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex
        self._best_generation: Optional[int] = self._read_pointer("best.json")

    def _read_pointer(self, name: str) -> Optional[int]:
        try:
            payload = json.loads((self.root / name).read_text(encoding="utf-8"))
            generation = payload.get("generation") if isinstance(payload, dict) else None
            return generation if _is_int(generation) and generation > 0 else None
        except Exception:
            return None

    def _manifest_paths(self) -> Iterable[Path]:
        return sorted(
            (path for path in self.root.glob("checkpoint-*.manifest.json")
             if _manifest_generation(path) is not None),
            key=lambda path: _manifest_generation(path) or 0,
            reverse=True,
        )

    def _next_generation(self) -> int:
        generations = [self._generation(path) for path in self._manifest_paths()]
        return max(generations, default=0) + 1

    @staticmethod
    def _generation(path: Path) -> int:
        generation = _manifest_generation(path)
        if generation is None:
            raise CheckpointError(f"invalid checkpoint manifest name: {path.name}")
        return generation

    def _check_free_space(self, path: Path) -> int:
        free_bytes = int(shutil.disk_usage(path).free)
        if free_bytes < self.min_free_bytes:
            raise CheckpointError(
                f"insufficient checkpoint storage: {free_bytes} free bytes, "
                f"need at least {self.min_free_bytes}"
            )
        return free_bytes

    def save(self, state: Dict[str, Any], *, next_step: int, is_best: bool = False) -> Dict[str, Any]:
        """Atomically save one complete state and update recovery pointers."""
        if not isinstance(state, dict):
            raise CheckpointError("checkpoint state must be a dictionary")
        generation = self._next_generation()
        payload = dict(state)
        payload.setdefault("schema_version", CHECKPOINT_SCHEMA_VERSION)
        payload.setdefault("run_id", self.run_id)
        payload["next_step"] = int(next_step)
        payload["generation"] = generation
        if not valid_complete_payload(payload):
            raise CheckpointError("refusing to write incomplete checkpoint state")
        free_before = self._check_free_space(self.root)
        permanent_free_before = None
        if self.permanent_root is not None:
            permanent_free_before = self._check_free_space(self.permanent_root)
        started = time.monotonic()
        checkpoint = self.root / f"checkpoint-{generation:08d}.pt"
        temporary = self.root / f".{checkpoint.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, checkpoint)
            _fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)
        free_after = int(shutil.disk_usage(self.root).free)
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": payload["run_id"],
            "generation": generation,
            "next_step": int(next_step),
            "checkpoint_file": checkpoint.name,
            "sha256": _sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "save_duration_seconds": 0.0,
            "storage_free_bytes_before": free_before,
            "storage_free_bytes_after": free_after,
            "permanent_free_bytes_before": permanent_free_before,
        }
        _atomic_json_write(self.root / f"checkpoint-{generation:08d}.manifest.json", manifest)
        _atomic_json_write(self.root / "latest.json", {**manifest, "pointer": "latest"})
        if is_best:
            self._best_generation = generation
            _atomic_json_write(self.root / "best.json", {**manifest, "pointer": "best"})
        self._copy_permanent("latest", manifest)
        if is_best:
            self._copy_permanent("best", manifest)
        self._apply_retention()
        duration = time.monotonic() - started
        manifest["save_duration_seconds"] = duration
        manifest_path = self.root / f"checkpoint-{generation:08d}.manifest.json"
        _atomic_json_write(manifest_path, manifest)
        _atomic_json_write(self.root / "latest.json", {**manifest, "pointer": "latest"})
        if is_best:
            _atomic_json_write(self.root / "best.json", {**manifest, "pointer": "best"})
        self._write_permanent_metadata("latest", manifest)
        if is_best:
            self._write_permanent_metadata("best", manifest)
        _atomic_json_write(
            self.root / "checkpoint_receipt.json",
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "generation": generation,
                "next_step": int(next_step),
                "checkpoint_interval": self.checkpoint_interval,
                "save_duration_seconds": duration,
                "storage_free_bytes_before": free_before,
                "storage_free_bytes_after": free_after,
                "permanent_free_bytes_before": permanent_free_before,
                "is_best": bool(is_best),
            },
        )
        return manifest

    def _copy_permanent(self, label: str, manifest: Dict[str, Any]) -> None:
        if self.permanent_root is None:
            return
        source = self.root / manifest["checkpoint_file"]
        target = self.permanent_root / f"{label}.pt"
        _atomic_copy(source, target)
        target_digest = _sha256(target)
        if target_digest != manifest["sha256"]:
            raise CheckpointError(f"permanent {label} checksum differs from source manifest")
        self._write_permanent_metadata(label, manifest, target_digest)

    def _write_permanent_metadata(
        self, label: str, manifest: Dict[str, Any], digest: Optional[str] = None
    ) -> None:
        if self.permanent_root is None:
            return
        target = self.permanent_root / f"{label}.pt"
        target_digest = digest or _sha256(target)
        _atomic_json_write(
            self.permanent_root / f"{label}.manifest.json",
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "label": label,
                "source_manifest": manifest,
                "permanent_file": target.name,
                "sha256": target_digest,
                "size_bytes": target.stat().st_size,
            },
        )

    def _apply_retention(self) -> None:
        valid = []
        invalid = []
        for path in self._manifest_paths():
            entry = _valid_manifest_payload(path, self.root)
            if entry is None:
                invalid.append(path)
            else:
                valid.append(entry)
        valid.sort(key=lambda entry: entry["manifest"]["generation"], reverse=True)
        keep = {entry["manifest"]["generation"] for entry in valid[: self.retention]}
        best = next((entry for entry in valid if entry["manifest"]["generation"] == self._best_generation), None)
        if best is not None:
            keep.add(self._best_generation)
        for entry in valid:
            generation = entry["manifest"]["generation"]
            if generation not in keep:
                entry["path"].unlink(missing_ok=True)
                (self.root / f"checkpoint-{generation:08d}.manifest.json").unlink(missing_ok=True)
        for path in invalid:
            generation = _manifest_generation(path)
            if generation is None:
                continue
            # Remove only the invalid manifest and its matching basename.  A
            # malformed path can never authorize deletion of another file.
            path.unlink(missing_ok=True)
            checkpoint = self.root / f"checkpoint-{generation:08d}.pt"
            checkpoint.unlink(missing_ok=True)

    def load_permanent_label(self, label: str, map_location: Any = "cpu") -> Optional[Dict[str, Any]]:
        if self.permanent_root is None or label not in {"latest", "best"}:
            return None
        try:
            metadata = json.loads((self.permanent_root / f"{label}.manifest.json").read_text(encoding="utf-8"))
            if (
                not isinstance(metadata, dict)
                or not _is_int(metadata.get("schema_version"))
                or metadata["schema_version"] != CHECKPOINT_SCHEMA_VERSION
            ):
                return None
            if metadata.get("label") != label or metadata.get("permanent_file") != f"{label}.pt":
                return None
            source = metadata.get("source_manifest")
            source_generation = source.get("generation") if isinstance(source, dict) else None
            source_file = source.get("checkpoint_file") if isinstance(source, dict) else None
            source_digest = source.get("sha256") if isinstance(source, dict) else None
            source_size = source.get("size_bytes") if isinstance(source, dict) else None
            source_step = source.get("next_step") if isinstance(source, dict) else None
            source_run_id = source.get("run_id") if isinstance(source, dict) else None
            if (
                not isinstance(source, dict)
                or not _is_int(source.get("schema_version"))
                or source["schema_version"] != CHECKPOINT_SCHEMA_VERSION
                or not _is_int(source_generation)
                or source_generation < 1
                or not isinstance(source_file, str)
                or source_file != f"checkpoint-{source_generation:08d}.pt"
                or not isinstance(source_digest, str)
                or source_digest != metadata.get("sha256")
                or not _is_int(source_size)
                or source_size < 0
                or not _is_int(source_step)
                or source_step < 0
                or not isinstance(source_run_id, str)
                or not source_run_id
            ):
                return None
            checkpoint = self.permanent_root / f"{label}.pt"
            if not checkpoint.is_file() or _sha256(checkpoint) != metadata["sha256"]:
                return None
            if (
                not _is_int(metadata.get("size_bytes"))
                or checkpoint.stat().st_size != metadata["size_bytes"]
                or checkpoint.stat().st_size != source_size
            ):
                return None
            payload = _torch_load(checkpoint, map_location=map_location)
            if not valid_complete_payload(payload):
                return None
            if payload.get("run_id") != source_run_id or payload.get("generation") != source_generation:
                return None
            if payload.get("next_step") != source_step:
                return None
            return {"manifest": source, "payload": payload, "path": checkpoint, "permanent": metadata}
        except Exception:
            return None

    def load_newest_valid(self, map_location: Any = "cpu") -> Optional[Dict[str, Any]]:
        """Recover the highest-progress valid state without loading all to GPU."""
        candidates = []
        for path in self._manifest_paths():
            # Discovery validates every generation on CPU.  Loading every
            # optimizer/model state onto an accelerator during discovery can
            # transiently OOM before the highest-progress candidate is known.
            entry = _valid_manifest_payload(path, self.root, map_location="cpu")
            if entry is not None:
                candidates.append(("scratch", path, entry))
        permanent = self.load_permanent_label("latest", map_location="cpu")
        if permanent is not None:
            candidates.append(("permanent", None, permanent))
        if not candidates:
            return None
        _, selected_path, selected = max(
            candidates,
            key=lambda entry: (
                int(entry[2]["payload"]["next_step"]),
                int(entry[2]["payload"]["generation"]),
            ),
        )
        if _is_cpu_map_location(map_location):
            return selected
        # Revalidate and deserialize only the selected state on the requested
        # device.  A race/corruption between discovery and this load fails
        # closed instead of silently returning an unvalidated payload.
        if selected_path is None:
            return self.load_permanent_label("latest", map_location=map_location)
        return _valid_manifest_payload(
            selected_path, self.root, map_location=map_location
        )

    def load_path(self, path: Path | str, map_location: Any = "cpu") -> Dict[str, Any]:
        checkpoint = Path(path)
        if checkpoint.parent.resolve() != self.root:
            raise CheckpointError("explicit checkpoint must live in checkpoint_dir")
        manifest_path = self.root / f"{checkpoint.stem}.manifest.json"
        valid = _valid_manifest_payload(manifest_path, self.root, map_location=map_location)
        if valid is None or valid["manifest"].get("checkpoint_file") != checkpoint.name:
            raise CheckpointError(f"invalid checkpoint: {checkpoint}")
        return valid

    def materialize_legacy_aliases(self, run_dir: Optional[Path] = None) -> None:
        destination = Path(run_dir or self.root)
        latest = self.load_newest_valid()
        if latest is None:
            return
        _atomic_copy(latest["path"], destination / "last.pt")
        best = None
        if self._best_generation is not None:
            best_manifest = self.root / f"checkpoint-{self._best_generation:08d}.manifest.json"
            best = _valid_manifest_payload(best_manifest, self.root)
        if best is None:
            best = self.load_permanent_label("best") or latest
        _atomic_copy(best["path"], destination / "checkpoint.pt")
