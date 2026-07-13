"""Utilities for compact state-only ManiSkill insertion datasets.

The compact format is intentionally simple:

``observations``
    Float32 array with shape ``[episode, time + 1, obs_dim]``.
``actions``
    Float32 array with shape ``[episode, time, action_dim]``.
``valid``
    Boolean transition mask with shape ``[episode, time]``.
``outcome``
    Integer trajectory labels used only by evaluation. ``-1`` means unavailable.
``contact_phase``
    Optional integer state labels used only by evaluation. ``-1`` means unavailable.

Training code should consume only observations, actions, valid masks, and splits.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


OUTCOME_NAMES = ("unknown", "success", "failure_flag", "timeout_or_partial")
CONTACT_PHASE_NAMES = ("free_space_or_no_contact", "contact_or_near", "inserted_or_success")
DEFAULT_SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class CompactManiSkillDataset:
    """In-memory view of the compact insertion benchmark dataset."""

    observations: np.ndarray
    actions: np.ndarray
    valid: np.ndarray
    split: np.ndarray
    outcome: np.ndarray
    contact_phase: Optional[np.ndarray]
    episode_ids: np.ndarray
    feature_names: Tuple[str, ...]
    action_names: Tuple[str, ...]
    metadata: Dict[str, Any]

    @property
    def num_episodes(self) -> int:
        return int(self.observations.shape[0])

    @property
    def max_transitions(self) -> int:
        return int(self.actions.shape[1])

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[-1])

    def indices_for_split(self, split_name: str) -> np.ndarray:
        return np.nonzero(self.split.astype(str) == split_name)[0].astype(np.int64)


def load_compact_dataset(path: str | Path) -> CompactManiSkillDataset:
    """Load a compact ManiSkill insertion dataset from ``.npz``."""

    dataset_path = Path(path)
    with np.load(dataset_path, allow_pickle=False) as data:
        metadata_json = _scalar_string(data["metadata_json"])
        contact_phase = data["contact_phase"] if "contact_phase" in data.files else None
        if contact_phase is not None and np.all(contact_phase < 0):
            contact_phase = None
        metadata = json.loads(metadata_json)
        return CompactManiSkillDataset(
            observations=np.asarray(data["observations"], dtype=np.float32),
            actions=np.asarray(data["actions"], dtype=np.float32),
            valid=np.asarray(data["valid"], dtype=bool),
            split=np.asarray(data["split"]).astype(str),
            outcome=np.asarray(data["outcome"], dtype=np.int64),
            contact_phase=None if contact_phase is None else np.asarray(contact_phase, dtype=np.int64),
            episode_ids=np.asarray(data["episode_ids"], dtype=np.int64),
            feature_names=tuple(np.asarray(data["feature_names"]).astype(str).tolist()),
            action_names=tuple(np.asarray(data["action_names"]).astype(str).tolist()),
            metadata=metadata,
        )


def save_compact_dataset(path: str | Path, dataset: CompactManiSkillDataset) -> None:
    """Save a compact ManiSkill insertion dataset as compressed ``.npz``."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contact_phase = dataset.contact_phase
    if contact_phase is None:
        contact_phase = np.full(
            (dataset.num_episodes, dataset.max_transitions + 1),
            -1,
            dtype=np.int64,
        )

    np.savez_compressed(
        output_path,
        observations=np.asarray(dataset.observations, dtype=np.float32),
        actions=np.asarray(dataset.actions, dtype=np.float32),
        valid=np.asarray(dataset.valid, dtype=bool),
        split=np.asarray(dataset.split, dtype=str),
        outcome=np.asarray(dataset.outcome, dtype=np.int64),
        contact_phase=np.asarray(contact_phase, dtype=np.int64),
        episode_ids=np.asarray(dataset.episode_ids, dtype=np.int64),
        feature_names=np.asarray(dataset.feature_names, dtype=str),
        action_names=np.asarray(dataset.action_names, dtype=str),
        metadata_json=np.asarray(json.dumps(dataset.metadata, sort_keys=True), dtype=str),
    )


def build_compact_dataset_from_h5(
    traj_path: str | Path,
    output_path: str | Path,
    *,
    obs_key: str = "obs",
    append_prev_action: bool = True,
    max_episodes: Optional[int] = None,
    max_steps: Optional[int] = None,
    min_steps: int = 2,
    split_seed: int = 0,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> CompactManiSkillDataset:
    """Build the compact state/action dataset from a ManiSkill HDF5 trajectory.

    The HDF5 input should usually be produced by ManiSkill replay with
    ``-o state --save-traj``. If ``obs`` is absent, this function can fall back
    to ``env_states`` by passing ``obs_key="env_states"``.
    """

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on optional ManiSkill stack
        raise RuntimeError(
            "h5py is required to build compact ManiSkill datasets. It is usually "
            "available after installing ManiSkill."
        ) from exc

    traj_path = Path(traj_path)
    json_path = traj_path.with_suffix(".json")
    json_metadata: Dict[str, Any] = {}
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as handle:
            json_metadata = json.load(handle)
    episode_metadata = _episode_metadata_by_id(json_metadata)

    records: List[Dict[str, Any]] = []
    feature_names: Optional[Tuple[str, ...]] = None
    action_dim: Optional[int] = None

    with h5py.File(traj_path, "r") as handle:
        traj_keys = _sorted_trajectory_keys(handle.keys())
        if max_episodes is not None and max_episodes >= 0:
            traj_keys = traj_keys[: int(max_episodes)]

        for traj_key in traj_keys:
            group = handle[traj_key]
            if "actions" not in group:
                continue
            actions = np.asarray(group["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[:, None]
            if actions.ndim != 2:
                raise ValueError(f"{traj_key}/actions must be rank 1 or 2, got {actions.shape}")
            if action_dim is None:
                action_dim = int(actions.shape[-1])
            if int(actions.shape[-1]) != action_dim:
                raise ValueError(
                    f"Action dimension changed from {action_dim} to {actions.shape[-1]} in {traj_key}"
                )

            transition_count = int(actions.shape[0])
            if max_steps is not None and max_steps > 0:
                transition_count = min(transition_count, int(max_steps))
                actions = actions[:transition_count]
            if transition_count < int(min_steps):
                continue

            source_key = obs_key if obs_key in group else "env_states"
            if source_key not in group:
                raise ValueError(
                    f"{traj_key} contains neither '{obs_key}' nor 'env_states'. "
                    "Replay the raw demo with `-o state --save-traj` first."
                )

            source_tree = _read_h5_tree(group[source_key])
            obs, names = flatten_time_series_tree(
                source_tree,
                expected_length=transition_count + 1,
                root_name=source_key,
            )
            if append_prev_action:
                prev_action = np.zeros((transition_count + 1, action_dim), dtype=np.float32)
                prev_action[1:] = actions[:transition_count]
                obs = np.concatenate([obs, prev_action], axis=-1)
                names = tuple(names) + tuple(f"prev_action/{index}" for index in range(action_dim))
            if obs.shape[0] != transition_count + 1:
                raise ValueError(
                    f"{traj_key}/{source_key} flattened to length {obs.shape[0]}, "
                    f"expected {transition_count + 1}"
                )
            if feature_names is None:
                feature_names = tuple(names)
            elif tuple(names) != feature_names:
                raise ValueError(
                    f"Feature names changed in {traj_key}; compact one-seed path expects fixed state shape."
                )

            episode_id = _episode_id_from_key(traj_key)
            success = _read_optional_bool_series(group, "success", max_len=transition_count)
            fail = _read_optional_bool_series(group, "fail", max_len=transition_count)
            terminated = _read_optional_bool_series(group, "terminated", max_len=transition_count)
            truncated = _read_optional_bool_series(group, "truncated", max_len=transition_count)
            outcome = derive_outcome_label(
                success=success,
                fail=fail,
                terminated=terminated,
                truncated=truncated,
                episode_info=episode_metadata.get(episode_id, {}),
            )
            contact_phase = derive_contact_phase_labels(
                observations=obs,
                feature_names=names,
                success=success,
            )

            records.append(
                {
                    "episode_id": episode_id,
                    "observations": obs,
                    "actions": actions,
                    "outcome": outcome,
                    "contact_phase": contact_phase,
                    "transition_count": transition_count,
                }
            )

    if not records:
        raise ValueError(f"No usable trajectories found in {traj_path}")
    if feature_names is None or action_dim is None:
        raise ValueError(f"Could not infer feature/action dimensions from {traj_path}")

    max_transitions = max(int(record["transition_count"]) for record in records)
    obs_dim = int(records[0]["observations"].shape[-1])
    observations = np.zeros((len(records), max_transitions + 1, obs_dim), dtype=np.float32)
    actions = np.zeros((len(records), max_transitions, action_dim), dtype=np.float32)
    valid = np.zeros((len(records), max_transitions), dtype=bool)
    outcomes = np.full((len(records),), -1, dtype=np.int64)
    contact_phase = np.full((len(records), max_transitions + 1), -1, dtype=np.int64)
    episode_ids = np.zeros((len(records),), dtype=np.int64)

    for index, record in enumerate(records):
        length = int(record["transition_count"])
        observations[index, : length + 1] = record["observations"]
        actions[index, :length] = record["actions"]
        valid[index, :length] = True
        outcomes[index] = int(record["outcome"])
        episode_ids[index] = int(record["episode_id"])
        if record["contact_phase"] is not None:
            contact_phase[index, : length + 1] = record["contact_phase"][: length + 1]

    split = make_episode_splits(
        num_episodes=len(records),
        seed=split_seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    action_names = tuple(f"action/{index}" for index in range(action_dim))
    metadata = {
        "source": "maniskill_insertion_compact_v1",
        "source_traj_path": str(traj_path),
        "source_json_path": str(json_path) if json_path.exists() else None,
        "obs_key": obs_key,
        "append_prev_action": bool(append_prev_action),
        "env_id": json_metadata.get("env_info", {}).get("env_id"),
        "env_kwargs": json_metadata.get("env_info", {}).get("env_kwargs", {}),
        "outcome_names": list(OUTCOME_NAMES),
        "contact_phase_names": list(CONTACT_PHASE_NAMES),
        "labels_are_evaluation_only": True,
        "split_seed": int(split_seed),
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
        "max_steps": None if max_steps is None else int(max_steps),
        "min_steps": int(min_steps),
        "num_episodes": int(len(records)),
        "max_transitions": int(max_transitions),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "contact_phase_available": bool(np.any(contact_phase >= 0)),
        "outcome_available": bool(np.any(outcomes >= 0)),
    }

    dataset = CompactManiSkillDataset(
        observations=observations,
        actions=actions,
        valid=valid,
        split=split,
        outcome=outcomes,
        contact_phase=contact_phase if np.any(contact_phase >= 0) else None,
        episode_ids=episode_ids,
        feature_names=tuple(feature_names),
        action_names=action_names,
        metadata=metadata,
    )
    save_compact_dataset(output_path, dataset)
    return dataset


def flatten_time_series_tree(
    tree: Any,
    *,
    expected_length: int,
    root_name: str,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Flatten a ManiSkill observation/state tree into ``[T, D]`` features."""

    leaves: List[Tuple[str, np.ndarray]] = []
    for path, array in _iter_numeric_leaves(tree, prefix=root_name):
        arr = np.asarray(array)
        if arr.ndim == 0 or arr.shape[0] < int(expected_length):
            continue
        if arr.dtype.kind not in "biuf":
            continue
        flat = np.asarray(arr[: int(expected_length)], dtype=np.float32).reshape(int(expected_length), -1)
        for offset in range(flat.shape[-1]):
            leaves.append((f"{path}/{offset}", flat[:, offset]))

    if not leaves:
        raise ValueError(
            f"No numeric leaves with leading length {expected_length} were found under '{root_name}'"
        )

    names = tuple(name for name, _values in leaves)
    values = np.stack([values for _name, values in leaves], axis=-1).astype(np.float32, copy=False)
    return values, names


def make_episode_splits(
    *,
    num_episodes: int,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> np.ndarray:
    """Create trajectory-level train/val/test splits."""

    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0, 1)")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be below 1")

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(num_episodes))
    train_count = max(1, int(round(num_episodes * train_fraction)))
    val_count = int(round(num_episodes * val_fraction))
    if num_episodes >= 3:
        val_count = max(1, val_count)
        train_count = min(train_count, num_episodes - val_count - 1)
    else:
        val_count = min(val_count, max(0, num_episodes - train_count))

    split = np.full((num_episodes,), "test", dtype="<U8")
    split[order[:train_count]] = "train"
    split[order[train_count : train_count + val_count]] = "val"
    return split


def derive_outcome_label(
    *,
    success: Optional[np.ndarray],
    fail: Optional[np.ndarray],
    terminated: Optional[np.ndarray],
    truncated: Optional[np.ndarray],
    episode_info: Mapping[str, Any],
) -> int:
    """Derive a coarse evaluation-only trajectory outcome label."""

    info = episode_info.get("info", {}) if isinstance(episode_info, Mapping) else {}
    info_success = _coerce_optional_bool(_nested_lookup(info, ("success", "is_success")))
    info_fail = _coerce_optional_bool(_nested_lookup(info, ("fail", "failure", "is_failure")))

    if _last_true(success) or info_success is True:
        return int(OUTCOME_NAMES.index("success"))
    if _last_true(fail) or info_fail is True:
        return int(OUTCOME_NAMES.index("failure_flag"))
    if _last_true(terminated) or _last_true(truncated):
        return int(OUTCOME_NAMES.index("timeout_or_partial"))
    return -1


def derive_contact_phase_labels(
    *,
    observations: np.ndarray,
    feature_names: Sequence[str],
    success: Optional[np.ndarray],
    contact_threshold: float = 0.5,
    depth_threshold: float = 1e-3,
    distance_threshold: float = 0.03,
) -> Optional[np.ndarray]:
    """Derive coarse evaluation-only contact phases when state names expose them.

    This is deliberately conservative. If no contact-like, depth-like, or
    distance-like state feature exists, it returns ``None`` rather than inventing
    labels.
    """

    lower_names = [name.lower() for name in feature_names]
    contact_cols = [
        idx
        for idx, name in enumerate(lower_names)
        if "contact" in name or "collision" in name or "touch" in name
    ]
    depth_cols = [
        idx
        for idx, name in enumerate(lower_names)
        if "depth" in name or "insert" in name or "seated" in name
    ]
    distance_cols = [
        idx
        for idx, name in enumerate(lower_names)
        if ("dist" in name or "distance" in name) and ("peg" in name or "hole" in name or "goal" in name)
    ]
    if not contact_cols and not depth_cols and not distance_cols and success is None:
        return None

    phase = np.zeros((observations.shape[0],), dtype=np.int64)
    if contact_cols:
        contact_score = np.nanmax(np.abs(observations[:, contact_cols]), axis=1)
        phase[contact_score > float(contact_threshold)] = int(CONTACT_PHASE_NAMES.index("contact_or_near"))
    if distance_cols:
        distance_score = np.nanmin(np.abs(observations[:, distance_cols]), axis=1)
        phase[distance_score < float(distance_threshold)] = int(CONTACT_PHASE_NAMES.index("contact_or_near"))
    if depth_cols:
        depth_score = np.nanmax(observations[:, depth_cols], axis=1)
        phase[depth_score > float(depth_threshold)] = int(CONTACT_PHASE_NAMES.index("inserted_or_success"))
    if success is not None and len(success) > 0:
        success_state = np.zeros_like(phase, dtype=bool)
        success_state[1 : len(success) + 1] = np.asarray(success, dtype=bool)[: len(phase) - 1]
        phase[success_state] = int(CONTACT_PHASE_NAMES.index("inserted_or_success"))
    return phase


def feature_group_indices(feature_names: Sequence[str]) -> Dict[str, np.ndarray]:
    """Infer optional metric groups from flattened ManiSkill state feature names."""

    groups: Dict[str, List[int]] = {
        "peg_or_object_pose": [],
        "relative_peg_hole": [],
        "insertion_depth": [],
        "peg_hole_distance": [],
        "contact_signal": [],
    }
    for index, raw_name in enumerate(feature_names):
        name = raw_name.lower()
        if any(token in name for token in ("peg", "obj", "object")) and any(
            token in name for token in ("pose", "pos", "quat", "p ", "qpos")
        ):
            groups["peg_or_object_pose"].append(index)
        if ("rel" in name or "relative" in name) and ("peg" in name or "hole" in name or "goal" in name):
            groups["relative_peg_hole"].append(index)
        if "depth" in name or "insert" in name or "seated" in name:
            groups["insertion_depth"].append(index)
        if ("dist" in name or "distance" in name) and ("peg" in name or "hole" in name or "goal" in name):
            groups["peg_hole_distance"].append(index)
        if "contact" in name or "collision" in name or "touch" in name:
            groups["contact_signal"].append(index)
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in groups.items() if indices}


def _read_h5_tree(node: Any) -> Any:
    if hasattr(node, "keys"):
        return {str(key): _read_h5_tree(node[key]) for key in node.keys()}
    return np.asarray(node)


def _iter_numeric_leaves(tree: Any, *, prefix: str) -> Iterable[Tuple[str, np.ndarray]]:
    if isinstance(tree, Mapping):
        for key in sorted(tree.keys()):
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            yield from _iter_numeric_leaves(tree[key], prefix=child_prefix)
        return
    yield prefix, np.asarray(tree)


def _sorted_trajectory_keys(keys: Iterable[str]) -> List[str]:
    traj_keys = [key for key in keys if re.fullmatch(r"traj_\d+", str(key))]
    return sorted(traj_keys, key=_episode_id_from_key)


def _episode_id_from_key(key: str) -> int:
    match = re.search(r"(\d+)$", str(key))
    if match is None:
        return -1
    return int(match.group(1))


def _episode_metadata_by_id(metadata: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    episodes = metadata.get("episodes", []) if isinstance(metadata, Mapping) else []
    out: Dict[int, Mapping[str, Any]] = {}
    for item in episodes:
        if not isinstance(item, Mapping) or "episode_id" not in item:
            continue
        out[int(item["episode_id"])] = item
    return out


def _read_optional_bool_series(group: Any, key: str, *, max_len: int) -> Optional[np.ndarray]:
    if key not in group:
        return None
    values = np.asarray(group[key]).astype(bool)
    if values.ndim == 0:
        values = values.reshape(1)
    return values.reshape(-1)[: int(max_len)]


def _last_true(values: Optional[np.ndarray]) -> bool:
    if values is None or len(values) == 0:
        return False
    return bool(np.asarray(values, dtype=bool).reshape(-1)[-1])


def _nested_lookup(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if isinstance(mapping, Mapping) and key in mapping:
            return mapping[key]
    return None


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return None


def _scalar_string(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(array.reshape(-1)[0])
