"""Generate one-seed perturbed ManiSkill insertion rollout packets.

This tool is intentionally small and assessment-oriented. It replays one
downloaded demonstration seed under simple action-space perturbations and saves
compact state/action ``.npz`` files compatible with the controlled SKAE
evaluator. The labels are evaluation metadata only; no training code consumes
them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from skae.benchmarks.maniskill_insertion_dataset import (
    CompactManiSkillDataset,
    make_episode_splits,
    save_compact_dataset,
)


SETUP_NAMES = ("success", "jam", "miss", "drop", "partial")
OUTCOME_NAMES = ("unknown", *SETUP_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo_h5", type=Path, required=True)
    parser.add_argument("--demo_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--episode_count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--settle_steps", type=int, default=20)
    parser.add_argument("--env_max_episode_steps", type=int, default=None)
    parser.add_argument("--setups", default="success,jam,miss,drop,partial")
    parser.add_argument("--combined_output", type=Path, default=None)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setups = [item.strip() for item in args.setups.split(",") if item.strip()]
    for setup in setups:
        if setup not in SETUP_NAMES:
            raise ValueError(f"Unknown setup '{setup}'. Expected one of {SETUP_NAMES}")

    metadata = json.loads(args.demo_json.read_text(encoding="utf-8"))
    episodes = list(metadata.get("episodes", []))
    if not episodes:
        raise ValueError(f"No episodes found in {args.demo_json}")
    first_episode = int(args.episode_index)
    episode_count = max(1, int(args.episode_count))
    selected_episodes = episodes[first_episode : first_episode + episode_count]
    if not selected_episodes:
        raise ValueError(f"No episodes selected from index {first_episode}")
    env_max_episode_steps = args.env_max_episode_steps
    if env_max_episode_steps is None:
        env_max_episode_steps = int(args.max_steps) + int(args.settle_steps)
    env, env_summary = make_env(metadata, max_episode_steps=env_max_episode_steps)

    datasets: List[CompactManiSkillDataset] = []
    summaries: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed))
    try:
        by_setup: Dict[str, List[CompactManiSkillDataset]] = {setup: [] for setup in setups}
        for local_episode_index, episode in enumerate(selected_episodes):
            episode_id = int(episode.get("episode_id", first_episode + local_episode_index))
            actions = read_demo_actions(args.demo_h5, episode_id)
            for setup in setups:
                rollout = generate_rollout(
                    env,
                    episode,
                    actions,
                    setup=setup,
                    rng=rng,
                    max_steps=int(args.max_steps),
                    settle_steps=int(args.settle_steps),
                )
                dataset = rollout_to_dataset(
                    rollout,
                    setup=setup,
                    episode_id=episode_id,
                    source_episode=episode,
                    env_summary=env_summary,
                    split="test",
                )
                datasets.append(dataset)
                by_setup[setup].append(dataset)
                summary = {
                    "setup": setup,
                    "episode_id": episode_id,
                    "steps": int(dataset.valid.sum()),
                    "obs_dim": int(dataset.obs_dim),
                    "action_dim": int(dataset.action_dim),
                    "target_outcome_label": int(dataset.outcome[0]),
                    "actual_success_any": bool(np.any(rollout["success"])),
                    "actual_success_final": bool(rollout["success"][-1]) if len(rollout["success"]) else False,
                }
                summaries.append(summary)

        setup_outputs: Dict[str, str] = {}
        for setup, setup_datasets in by_setup.items():
            output = args.output_dir / f"{setup}.npz"
            save_compact_dataset(
                output,
                combine_datasets(
                    setup_datasets,
                    setups=[setup] * len(setup_datasets),
                    env_summary=env_summary,
                    split_seed=int(args.seed),
                    train_fraction=0.0,
                    val_fraction=0.0,
                    all_test=True,
                ),
            )
            setup_outputs[setup] = str(output)

        combined_output = args.combined_output or (args.output_dir / "all_setups.npz")
        save_compact_dataset(
            combined_output,
            combine_datasets(
                datasets,
                setups=[dataset.metadata.get("setup", "unknown") for dataset in datasets],
                env_summary=env_summary,
                split_seed=int(args.seed),
                train_fraction=float(args.train_fraction),
                val_fraction=float(args.val_fraction),
                all_test=False,
            ),
        )
        payload = {
            "demo_h5": str(args.demo_h5),
            "demo_json": str(args.demo_json),
            "episode_index": int(args.episode_index),
            "episode_count": int(len(selected_episodes)),
            "seed": int(args.seed),
            "combined_output": str(combined_output),
            "setup_outputs": setup_outputs,
            "setups": summaries,
            "labels_used_for_training": False,
        }
        summary_path = args.output_dir / "perturbation_summary.json"
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        env.close()


def read_demo_actions(path: Path, episode_id: int) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("h5py is required to read ManiSkill demonstrations") from exc

    with h5py.File(path, "r") as handle:
        key = f"traj_{int(episode_id)}"
        if key not in handle:
            keys = sorted(k for k in handle.keys() if k.startswith("traj_"))
            if not keys:
                raise ValueError(f"No traj_* groups found in {path}")
            key = keys[0]
        actions = np.asarray(handle[key]["actions"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[:, None]
    return actions


def make_env(metadata: Mapping[str, Any], *, max_episode_steps: int):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    env_info = dict(metadata.get("env_info", {}))
    env_id = str(env_info.get("env_id", "PegInsertionSide-v1"))
    env_kwargs = dict(env_info.get("env_kwargs", {}))
    env_kwargs["obs_mode"] = "state"
    env_kwargs["render_mode"] = None
    env_kwargs.setdefault("control_mode", "pd_joint_pos")
    env_kwargs.setdefault("reward_mode", "dense")
    env_kwargs.setdefault("num_envs", 1)
    env = gym.make(env_id, max_episode_steps=int(max_episode_steps), **env_kwargs)
    summary = {"env_id": env_id, "env_kwargs": env_kwargs, "max_episode_steps": int(max_episode_steps)}
    return env, summary


def generate_rollout(
    env: Any,
    episode: Mapping[str, Any],
    demo_actions: np.ndarray,
    *,
    setup: str,
    rng: np.random.Generator,
    max_steps: int,
    settle_steps: int,
) -> Dict[str, Any]:
    reset_kwargs = dict(episode.get("reset_kwargs", {}))
    obs, info = env.reset(**reset_kwargs)
    obs_vec = flatten_observation(obs)

    low, high = action_bounds(env, demo_actions.shape[-1])
    actions = perturb_actions(
        demo_actions,
        setup=setup,
        rng=rng,
        low=low,
        high=high,
        max_steps=max_steps,
        settle_steps=settle_steps,
    )

    observations = [obs_vec]
    executed_actions = []
    success = []
    terminated_values = []
    truncated_values = []
    for action in actions:
        step_obs, _reward, terminated, truncated, step_info = env.step(action.astype(np.float32, copy=False))
        observations.append(flatten_observation(step_obs))
        executed_actions.append(action.astype(np.float32, copy=False))
        success.append(bool_tensor_value(extract_info_value(step_info, "success")))
        terminated_values.append(bool_tensor_value(terminated))
        truncated_values.append(bool_tensor_value(truncated))
        if terminated_values[-1] or truncated_values[-1]:
            break

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(executed_actions, dtype=np.float32),
        "success": np.asarray(success, dtype=bool),
        "terminated": np.asarray(terminated_values, dtype=bool),
        "truncated": np.asarray(truncated_values, dtype=bool),
        "initial_info_keys": sorted(list(info.keys())) if hasattr(info, "keys") else [],
    }


def perturb_actions(
    demo_actions: np.ndarray,
    *,
    setup: str,
    rng: np.random.Generator,
    low: np.ndarray,
    high: np.ndarray,
    max_steps: int,
    settle_steps: int,
) -> np.ndarray:
    actions = np.asarray(demo_actions[: max(1, int(max_steps))], dtype=np.float32).copy()
    if setup == "success":
        actions = actions + rng.normal(0.0, 0.002, size=actions.shape).astype(np.float32)
    elif setup == "jam":
        start = max(1, int(0.35 * len(actions)))
        dims = min(3, actions.shape[-1])
        bias = np.zeros((actions.shape[-1],), dtype=np.float32)
        bias[:dims] = np.asarray([0.04, -0.04, 0.03][:dims], dtype=np.float32)
        actions[start:] += bias
        actions += rng.normal(0.0, 0.01, size=actions.shape).astype(np.float32)
    elif setup == "miss":
        dims = min(4, actions.shape[-1])
        bias = np.zeros((actions.shape[-1],), dtype=np.float32)
        bias[:dims] = np.asarray([0.15, -0.12, 0.10, -0.08][:dims], dtype=np.float32)
        actions += bias
    elif setup == "drop":
        if actions.shape[-1] > 0:
            start = max(1, int(0.20 * len(actions)))
            actions[start:, -1] = low[-1]
        actions += rng.normal(0.0, 0.004, size=actions.shape).astype(np.float32)
    elif setup == "partial":
        hold_index = min(len(actions) - 1, max(1, int(0.45 * len(actions))))
        actions[hold_index:] = actions[hold_index]
    else:  # pragma: no cover
        raise ValueError(setup)

    if settle_steps > 0 and len(actions) > 0:
        settle_action = actions[-1:]
        actions = np.concatenate([actions, np.repeat(settle_action, int(settle_steps), axis=0)], axis=0)
    actions = actions[: int(max_steps)]
    return np.clip(actions, low, high).astype(np.float32, copy=False)


def rollout_to_dataset(
    rollout: Mapping[str, Any],
    *,
    setup: str,
    episode_id: int,
    source_episode: Mapping[str, Any],
    env_summary: Mapping[str, Any],
    split: str,
) -> CompactManiSkillDataset:
    actions = np.asarray(rollout["actions"], dtype=np.float32)
    observations = np.asarray(rollout["observations"], dtype=np.float32)
    if actions.ndim != 2 or observations.ndim != 2:
        raise ValueError("Expected rank-2 actions and observations")
    if observations.shape[0] != actions.shape[0] + 1:
        raise ValueError("Expected observations length to equal actions length + 1")

    previous_action = np.zeros((observations.shape[0], actions.shape[-1]), dtype=np.float32)
    previous_action[1:] = actions
    observations = np.concatenate([observations, previous_action], axis=-1)

    valid = np.ones((1, actions.shape[0]), dtype=bool)
    contact_phase = np.full((1, observations.shape[0]), -1, dtype=np.int64)
    success = np.asarray(rollout["success"], dtype=bool)
    if success.size:
        contact_phase[0, 1 : success.size + 1] = np.where(success, 2, 0)

    feature_names = tuple(f"state/{index}" for index in range(observations.shape[-1] - actions.shape[-1])) + tuple(
        f"prev_action/{index}" for index in range(actions.shape[-1])
    )
    action_names = tuple(f"action/{index}" for index in range(actions.shape[-1]))
    metadata = {
        "source": "maniskill_perturbation_assessment_v1",
        "setup": setup,
        "source_episode_id": int(episode_id),
        "source_episode_seed": source_episode.get("episode_seed"),
        "env_summary": dict(env_summary),
        "outcome_names": list(OUTCOME_NAMES),
        "contact_phase_names": ["free_or_not_success", "reserved", "success"],
        "labels_are_evaluation_only": True,
        "num_episodes": 1,
        "max_transitions": int(actions.shape[0]),
        "obs_dim": int(observations.shape[-1]),
        "action_dim": int(actions.shape[-1]),
    }
    return CompactManiSkillDataset(
        observations=observations[None],
        actions=actions[None],
        valid=valid,
        split=np.asarray([split], dtype="<U8"),
        outcome=np.asarray([OUTCOME_NAMES.index(setup)], dtype=np.int64),
        contact_phase=contact_phase,
        episode_ids=np.asarray([int(episode_id)], dtype=np.int64),
        feature_names=feature_names,
        action_names=action_names,
        metadata=metadata,
    )


def combine_datasets(
    datasets: Sequence[CompactManiSkillDataset],
    *,
    setups: Sequence[str],
    env_summary: Mapping[str, Any],
    split_seed: int,
    train_fraction: float,
    val_fraction: float,
    all_test: bool,
) -> CompactManiSkillDataset:
    if not datasets:
        raise ValueError("No datasets to combine")
    obs_dim = datasets[0].obs_dim
    action_dim = datasets[0].action_dim
    max_transitions = max(dataset.max_transitions for dataset in datasets)
    observations = np.zeros((len(datasets), max_transitions + 1, obs_dim), dtype=np.float32)
    actions = np.zeros((len(datasets), max_transitions, action_dim), dtype=np.float32)
    valid = np.zeros((len(datasets), max_transitions), dtype=bool)
    contact_phase = np.full((len(datasets), max_transitions + 1), -1, dtype=np.int64)
    outcomes = np.zeros((len(datasets),), dtype=np.int64)
    episode_ids = np.zeros((len(datasets),), dtype=np.int64)
    for index, dataset in enumerate(datasets):
        length = dataset.max_transitions
        observations[index, : length + 1] = dataset.observations[0, : length + 1]
        actions[index, :length] = dataset.actions[0, :length]
        valid[index, :length] = dataset.valid[0, :length]
        if dataset.contact_phase is not None:
            contact_phase[index, : length + 1] = dataset.contact_phase[0, : length + 1]
        outcomes[index] = int(dataset.outcome[0])
        episode_ids[index] = int(dataset.episode_ids[0])
    if all_test:
        split = np.asarray(["test"] * len(datasets), dtype="<U8")
    else:
        split = make_group_splits(
            groups=episode_ids,
            seed=int(split_seed),
            train_fraction=float(train_fraction),
            val_fraction=float(val_fraction),
        )
    metadata = {
        "source": "maniskill_perturbation_assessment_v1",
        "setups": list(setups),
        "env_summary": dict(env_summary),
        "outcome_names": list(OUTCOME_NAMES),
        "contact_phase_names": ["free_or_not_success", "reserved", "success"],
        "labels_are_evaluation_only": True,
        "num_episodes": int(len(datasets)),
        "max_transitions": int(max_transitions),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
        "split_seed": int(split_seed),
        "split_group": "source_episode_id",
    }
    return CompactManiSkillDataset(
        observations=observations,
        actions=actions,
        valid=valid,
        split=split,
        outcome=outcomes,
        contact_phase=contact_phase,
        episode_ids=episode_ids,
        feature_names=datasets[0].feature_names,
        action_names=datasets[0].action_names,
        metadata=metadata,
    )


def make_group_splits(
    *,
    groups: np.ndarray,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> np.ndarray:
    """Assign splits by source episode so perturbations of one reset stay together."""

    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    group_splits = make_episode_splits(
        num_episodes=int(unique_groups.size),
        seed=int(seed),
        train_fraction=float(train_fraction),
        val_fraction=float(val_fraction),
    )
    split_by_group = {group: group_splits[index] for index, group in enumerate(unique_groups.tolist())}
    return np.asarray([split_by_group[group] for group in groups.tolist()], dtype="<U8")


def flatten_observation(obs: Any) -> np.ndarray:
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None  # type: ignore

    if torch is not None and isinstance(obs, torch.Tensor):
        arr = obs.detach().cpu().numpy()
        return squeeze_env_axis(arr).astype(np.float32, copy=False).reshape(-1)
    if isinstance(obs, np.ndarray):
        return squeeze_env_axis(obs).astype(np.float32, copy=False).reshape(-1)
    leaves = []
    for _name, arr in iter_numeric_leaves(obs, prefix="obs"):
        leaves.append(squeeze_env_axis(arr).astype(np.float32, copy=False).reshape(-1))
    if not leaves:
        raise ValueError(f"Could not flatten observation of type {type(obs)}")
    return np.concatenate(leaves, axis=0).astype(np.float32, copy=False)


def iter_numeric_leaves(value: Any, *, prefix: str) -> Iterable[Tuple[str, np.ndarray]]:
    if isinstance(value, Mapping):
        for key in sorted(value.keys()):
            yield from iter_numeric_leaves(value[key], prefix=f"{prefix}/{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from iter_numeric_leaves(item, prefix=f"{prefix}/{index}")
        return
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None  # type: ignore
    if torch is not None and isinstance(value, torch.Tensor):
        yield prefix, value.detach().cpu().numpy()
        return
    arr = np.asarray(value)
    if arr.dtype.kind in "biuf":
        yield prefix, arr


def squeeze_env_axis(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def action_bounds(env: Any, action_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    if low.size != int(action_dim):
        low = np.full((int(action_dim),), -np.inf, dtype=np.float32)
    if high.size != int(action_dim):
        high = np.full((int(action_dim),), np.inf, dtype=np.float32)
    low = np.where(np.isfinite(low), low, -1e6).astype(np.float32)
    high = np.where(np.isfinite(high), high, 1e6).astype(np.float32)
    return low, high


def extract_info_value(info: Any, key: str) -> Any:
    if isinstance(info, Mapping):
        return info.get(key, False)
    return False


def bool_tensor_value(value: Any) -> bool:
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None  # type: ignore
    if torch is not None and isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        return bool(np.asarray(arr).reshape(-1)[-1])
    arr = np.asarray(value)
    if arr.size == 0:
        return False
    return bool(arr.reshape(-1)[-1])


if __name__ == "__main__":
    main()
