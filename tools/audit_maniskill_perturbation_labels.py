"""Audit perturbation-target labels for compact ManiSkill rollout packets.

This audit is intentionally conservative. It checks whether the current
``success/jam/miss/drop/partial`` labels are validated physical outcomes or
only target perturbation labels, and it writes row-level and aggregate artifacts
for deciding whether model-support diagnostics are paper-usable.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from skae.benchmarks.maniskill_insertion_dataset import load_compact_dataset


GEOMETRY_GROUPS: Mapping[str, Tuple[str, ...]] = {
    "contact": ("contact", "collision", "touch", "force"),
    "insertion_depth": ("depth", "insert", "seated"),
    "peg_hole_distance": ("dist", "distance", "hole", "goal"),
    "grasp_drop": ("grasp", "gripper", "held", "drop"),
    "rim_alignment": ("rim", "align", "angle", "yaw"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Compact perturbation .npz packet")
    parser.add_argument(
        "--perturbation_summary",
        type=Path,
        default=None,
        help="perturbation_summary.json emitted by tools/maniskill_generate_perturbed_rollouts.py",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_compact_dataset(args.dataset)
    perturbation_summary = _load_json(args.perturbation_summary)
    summary_entries = list(perturbation_summary.get("setups", []))
    by_key = _summary_by_key(summary_entries)

    rows = []
    mismatch_count = 0
    missing_summary_count = 0
    for index in range(dataset.num_episodes):
        target_label = int(dataset.outcome[index]) if index < dataset.outcome.shape[0] else -1
        target_name = _outcome_name(dataset.metadata, target_label)
        episode_id = int(dataset.episode_ids[index])
        entry = by_key.get((episode_id, target_name))
        if entry is None:
            missing_summary_count += 1
        actual_success_any = _optional_bool(entry, "actual_success_any")
        actual_success_final = _optional_bool(entry, "actual_success_final")
        semantic_label = _semantic_label_from_available_signals(actual_success_final)
        if target_name == "success" and actual_success_final is False:
            mismatch_count += 1
        if target_name != "success" and actual_success_final is True:
            mismatch_count += 1
        rows.append(
            {
                "dataset_row": index,
                "episode_id": episode_id,
                "split": str(dataset.split[index]),
                "valid_steps": int(dataset.valid[index].sum()),
                "target_label": target_label,
                "target_name": target_name,
                "actual_success_any": _csv_bool(actual_success_any),
                "actual_success_final": _csv_bool(actual_success_final),
                "semantic_label_from_available_signals": semantic_label,
                "semantic_label_status": _semantic_label_status(entry, actual_success_final),
            }
        )

    feature_audit = _feature_audit(dataset.feature_names)
    aggregate = _aggregate(rows, feature_audit)
    aggregate.update(
        {
            "dataset": str(args.dataset),
            "perturbation_summary": None if args.perturbation_summary is None else str(args.perturbation_summary),
            "num_dataset_rows": int(dataset.num_episodes),
            "num_summary_entries": int(len(summary_entries)),
            "missing_summary_rows": int(missing_summary_count),
            "target_success_mismatch_rows": int(mismatch_count),
            "labels_used_for_training": False,
            "paper_claim_status": _paper_claim_status(feature_audit, rows),
            "recommended_next_step": (
                "Use this packet for pipeline/debugging only. For paper-facing "
                "outcome/contact support claims, regenerate or augment rollouts "
                "with explicit success, insertion depth, peg-hole distance, "
                "grasp/drop, sustained contact, and rim/alignment signals."
            ),
        }
    )

    _write_csv(args.output_dir / "label_audit_rows.csv", rows)
    _write_json(args.output_dir / "label_audit_summary.json", aggregate)
    _write_markdown(args.output_dir / "label_audit_summary.md", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


def _load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary_by_key(entries: Iterable[Mapping[str, Any]]) -> Dict[Tuple[int, str], Mapping[str, Any]]:
    out: Dict[Tuple[int, str], Mapping[str, Any]] = {}
    for entry in entries:
        try:
            episode_id = int(entry["episode_id"])
            setup = str(entry["setup"])
        except (KeyError, TypeError, ValueError):
            continue
        out[(episode_id, setup)] = entry
    return out


def _outcome_name(metadata: Mapping[str, Any], label: int) -> str:
    names = metadata.get("outcome_names", [])
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)) and 0 <= int(label) < len(names):
        return str(names[int(label)])
    return f"label_{int(label)}"


def _optional_bool(entry: Optional[Mapping[str, Any]], key: str) -> Optional[bool]:
    if entry is None or key not in entry:
        return None
    return _coerce_optional_bool(entry[key])


def _coerce_optional_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        if int(value) == 0:
            return False
        if int(value) == 1:
            return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0"}:
            return False
        if normalized in {"", "none", "null", "nan"}:
            return None
    raise ValueError(f"Cannot parse boolean value {value!r}")


def _csv_bool(value: Optional[bool]) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def _semantic_label_from_available_signals(actual_success_final: Optional[bool]) -> str:
    if actual_success_final is None:
        return "unknown"
    if actual_success_final:
        return "success"
    return "non_success_untyped"


def _semantic_label_status(entry: Optional[Mapping[str, Any]], actual_success_final: Optional[bool]) -> str:
    if entry is None:
        return "missing_summary"
    if actual_success_final is None:
        return "missing_success_signal"
    return "binary_success_only"


def _feature_audit(feature_names: Sequence[str]) -> Dict[str, object]:
    lower_names = [str(name).lower() for name in feature_names]
    groups: Dict[str, Dict[str, object]] = {}
    for group, tokens in GEOMETRY_GROUPS.items():
        matches = [name for name in lower_names if any(token in name for token in tokens)]
        groups[group] = {
            "available": bool(matches),
            "match_count": int(len(matches)),
            "examples": matches[:8],
        }
    has_geometry_for_five_class = all(
        bool(groups[group]["available"])
        for group in ("contact", "insertion_depth", "peg_hole_distance", "grasp_drop")
    )
    generic_state_names = sum(name.startswith("state/") for name in lower_names)
    return {
        "obs_dim": int(len(feature_names)),
        "generic_state_name_fraction": float(generic_state_names / max(1, len(feature_names))),
        "groups": groups,
        "has_required_geometry_for_five_class_labels": bool(has_geometry_for_five_class),
    }


def _aggregate(rows: Sequence[Mapping[str, object]], feature_audit: Mapping[str, object]) -> Dict[str, object]:
    target_counts = Counter(str(row["target_name"]) for row in rows)
    split_counts = Counter(str(row["split"]) for row in rows)
    semantic_counts = Counter(str(row["semantic_label_from_available_signals"]) for row in rows)
    success_by_target: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        target = str(row["target_name"])
        final = str(row["actual_success_final"])
        success_by_target[target][final or "missing"] += 1
    return {
        "target_label_counts": dict(sorted(target_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "semantic_label_counts_from_available_signals": dict(sorted(semantic_counts.items())),
        "actual_success_final_by_target": {
            target: dict(sorted(counter.items())) for target, counter in sorted(success_by_target.items())
        },
        "feature_audit": feature_audit,
    }


def _paper_claim_status(feature_audit: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    semantic_counts = Counter(str(row["semantic_label_from_available_signals"]) for row in rows)
    has_five_class_geometry = bool(feature_audit.get("has_required_geometry_for_five_class_labels", False))
    has_only_binary = set(semantic_counts.keys()).issubset({"success", "non_success_untyped", "unknown"})
    if not has_five_class_geometry or has_only_binary:
        return "not_ready_target_labels_only_or_binary_success_only"
    return "candidate_semantic_labels_available"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: Mapping[str, object]) -> None:
    lines = [
        "# ManiSkill Perturbation Label Audit",
        "",
        f"Dataset: `{payload['dataset']}`",
        f"Paper claim status: `{payload['paper_claim_status']}`",
        "",
        "## Target Labels",
        "",
    ]
    for label, count in payload["target_label_counts"].items():  # type: ignore[index]
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Actual Final Success By Target", ""])
    for label, counts in payload["actual_success_final_by_target"].items():  # type: ignore[index]
        lines.append(f"- `{label}`: `{counts}`")
    lines.extend(["", "## Semantic Labels From Available Signals", ""])
    for label, count in payload["semantic_label_counts"].items():  # type: ignore[index]
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Feature Audit", ""])
    feature_audit = payload["feature_audit"]  # type: ignore[index]
    lines.append(
        "Required geometry for five-class labels: "
        f"`{feature_audit['has_required_geometry_for_five_class_labels']}`"
    )
    lines.append("")
    for group, info in feature_audit["groups"].items():
        lines.append(
            f"- `{group}`: available=`{info['available']}`, "
            f"matches=`{info['match_count']}`"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            str(payload["recommended_next_step"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
