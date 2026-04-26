#!/usr/bin/env python3
"""Compare support-conditioned learned local laws to true local geometry.

This evaluator is intentionally conservative. It uses existing fixed-17
checkpoints and only claims a geometry comparison when all required pieces are
available:

* a recovered fixed point / attractor center with small residual,
* a differentiable true one-step state-space Jacobian at that point,
* a fitted support/family/basin centered latent operator near that point,
* local encoder and decoder Jacobians for projecting the latent operator back
  to state space.

Rows that cannot satisfy those gates are emitted with explicit N/A fields and a
skip reason instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

EPS = 1e-12


@dataclass(frozen=True)
class RunSpec:
    root_label: str
    system_key: str
    system_name: str
    seed: int
    run_dir: str


@dataclass(frozen=True)
class FixedPointRecord:
    basin_id: int
    point: np.ndarray
    source: str
    step_residual: float
    continuous_residual: Optional[float]


def _load_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


REDUCER = _load_module(
    "reduce_transition_rich_interpretability_metrics.py",
    "reduce_transition_rich_interpretability_metrics_true_jacobian_geometry",
)
OPSEL = _load_module(
    "evaluate_transition_rich_operator_selection.py",
    "evaluate_transition_rich_operator_selection_true_jacobian_geometry",
)
CCM = _load_module(
    "evaluate_transition_rich_centered_chart_mechanism.py",
    "evaluate_transition_rich_centered_chart_mechanism_true_jacobian_geometry",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csvs", required=True, help="comma-separated forecasting_rows.csv files")
    parser.add_argument("--output_dir", required=True, help="directory for evaluator artifacts")
    parser.add_argument("--root_labels", required=True, help="comma-separated root labels to include")
    parser.add_argument("--systems", default="", help="optional comma-separated system_key filter")
    parser.add_argument("--seeds", default="", help="optional comma-separated integer seed filter")
    parser.add_argument(
        "--support_definitions",
        default="absolute:0.001,topk:8,relative:0.1",
        help="comma-separated support definitions formatted as scheme:value",
    )
    parser.add_argument("--partition_kinds", default="attractor,basin,family,support")
    parser.add_argument("--num_trajectories", type=int, default=128)
    parser.add_argument("--trajectory_length", type=int, default=128)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--endpoint_rollout_steps", type=int, default=2000)
    parser.add_argument("--fixed_point_refine_steps", type=int, default=2000)
    parser.add_argument("--fixed_point_residual_tol", type=float, default=1e-4)
    parser.add_argument("--fixed_point_dedup_tol", type=float, default=1e-3)
    parser.add_argument("--finite_difference_eps", type=float, default=1e-4)
    parser.add_argument("--attractor_radius", type=float, default=0.75, help="legacy single-radius fallback")
    parser.add_argument(
        "--attractor_radii",
        default="0.25,0.5,0.75",
        help="comma-separated local-neighborhood radii around recovered fixed points",
    )
    parser.add_argument("--min_operator_transitions", type=int, default=32)
    parser.add_argument("--ridge_lambda", type=float, default=1e-4)
    parser.add_argument("--family_jaccard_threshold", type=float, default=0.5)
    parser.add_argument("--num_random_controls", type=int, default=4)
    parser.add_argument("--max_partition_classes", type=int, default=128)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--label_mode",
        default="auto",
        choices=["auto", "native", "env_points", "estimated_centers"],
    )
    parser.add_argument("--max_runs", type=int, default=0, help="0 means no limit")
    parser.add_argument("--progress_every_runs", type=int, default=1)
    parser.add_argument("--flush_every_runs", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="small subset for protocol validation")
    return parser.parse_args()


def _parse_csv_strings(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_support_definitions(raw: str) -> List[Tuple[str, float]]:
    definitions: List[Tuple[str, float]] = []
    for item in _parse_csv_strings(raw):
        if ":" not in item:
            raise ValueError(f"Support definition must be scheme:value, got '{item}'")
        scheme, raw_value = item.split(":", 1)
        scheme = scheme.strip()
        if scheme == "topk":
            definitions.append((scheme, float(int(raw_value))))
        else:
            definitions.append((scheme, float(raw_value)))
    return definitions


def _support_name(scheme: str, value: float) -> str:
    if scheme == "topk":
        return f"{scheme}:{int(value)}"
    return f"{scheme}:{value:.6g}"


def _apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.num_trajectories = min(args.num_trajectories, 16)
    args.trajectory_length = min(args.trajectory_length, 32)
    args.endpoint_rollout_steps = min(args.endpoint_rollout_steps, 128)
    args.fixed_point_refine_steps = min(args.fixed_point_refine_steps, 128)
    args.min_operator_transitions = min(args.min_operator_transitions, 8)
    args.num_random_controls = min(args.num_random_controls, 1)
    args.max_runs = args.max_runs if args.max_runs > 0 else 1
    if not args.attractor_radii:
        args.attractor_radii = str(args.attractor_radius)


def _load_latest_specs(
    rows_csvs: Sequence[Path],
    *,
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
    max_runs: int,
) -> List[RunSpec]:
    raw_specs = OPSEL._load_latest_specs(
        rows_csvs,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
    )
    specs = [
        RunSpec(
            root_label=spec.root_label,
            system_key=spec.system_key,
            system_name=spec.system_name,
            seed=int(spec.seed),
            run_dir=spec.run_dir,
        )
        for spec in raw_specs
    ]
    if max_runs > 0:
        return specs[:max_runs]
    return specs


def _as_numpy_2d(points: object) -> List[np.ndarray]:
    if points is None:
        return []
    if isinstance(points, torch.Tensor):
        arr = points.detach().cpu().numpy()
    else:
        arr = np.asarray(points)
    if arr.ndim != 2:
        return []
    return [np.asarray(row, dtype=np.float32) for row in arr]


def _candidate_points_from_env(env) -> List[Tuple[str, np.ndarray]]:
    base = env.unwrapped
    candidates: List[Tuple[str, np.ndarray]] = []
    for attr_name in ("points", "points_2d", "centers", "well_centers"):
        for point in _as_numpy_2d(getattr(base, attr_name, None)):
            candidates.append((f"env.{attr_name}", point))

    system = getattr(base, "system", None)
    if system is not None:
        for attr_name in ("points", "points_2d", "centers", "well_centers"):
            for point in _as_numpy_2d(getattr(system, attr_name, None)):
                candidates.append((f"system.{attr_name}", point))
        for attr_name in ("wells", "_wells"):
            wells = getattr(system, attr_name, None)
            if wells is None:
                continue
            for well in wells:
                if len(well) >= 2:
                    candidates.append((f"system.{attr_name}", np.asarray(well[:2], dtype=np.float32)))

    # Some systems have an origin attractor but no explicit center list.
    dim = int(base.observation_size)
    candidates.append(("origin_fallback", np.zeros(dim, dtype=np.float32)))
    return candidates


def _rollout_point(env, point: torch.Tensor, steps: int) -> torch.Tensor:
    current = point.detach().clone().to(dtype=torch.float32)
    with torch.no_grad():
        for _ in range(max(0, int(steps))):
            current = env.step(current)
    return current.detach().to(dtype=torch.float32)


def _continuous_residual(env, point: torch.Tensor) -> Optional[float]:
    base = env.unwrapped
    dynamics = getattr(base, "dynamics", None)
    if dynamics is None and getattr(base, "system", None) is not None:
        dynamics = getattr(base.system, "dynamics", None)
    if dynamics is None:
        return None
    try:
        with torch.no_grad():
            value = dynamics(point.to(dtype=torch.float64)).detach().to(dtype=torch.float32)
        return float(torch.linalg.vector_norm(value).item())
    except Exception:
        return None


def _basin_id_for_point(env, point: torch.Tensor, fallback_index: int) -> int:
    if hasattr(env.unwrapped, "basin_label"):
        try:
            label = env.unwrapped.basin_label(point.unsqueeze(0))[0]
            return int(label.item())
        except Exception:
            pass
    centers = getattr(env.unwrapped, "points", None)
    if isinstance(centers, torch.Tensor) and centers.ndim == 2:
        try:
            dists = torch.cdist(point.unsqueeze(0), centers.to(dtype=point.dtype))
            return int(dists.argmin(dim=1).item())
        except Exception:
            pass
    return int(fallback_index)


def _recover_fixed_points(
    env,
    *,
    refine_steps: int,
    residual_tol: float,
    dedup_tol: float,
) -> Tuple[List[FixedPointRecord], List[str]]:
    records: List[FixedPointRecord] = []
    notes: List[str] = []
    for source, candidate_np in _candidate_points_from_env(env):
        point = torch.from_numpy(candidate_np).to(dtype=torch.float32)
        if point.shape[-1] != int(env.unwrapped.observation_size):
            notes.append(f"{source}:dimension_mismatch")
            continue
        refined = _rollout_point(env, point, refine_steps)
        try:
            next_point = env.step(refined)
            step_residual = float(torch.linalg.vector_norm(next_point - refined).item())
        except Exception as exc:
            notes.append(f"{source}:step_failed:{repr(exc)}")
            continue
        continuous = _continuous_residual(env, refined)
        if step_residual > float(residual_tol):
            notes.append(f"{source}:residual>{residual_tol:g}:{step_residual:.3g}")
            continue
        refined_np = refined.detach().cpu().numpy().astype(np.float32, copy=False)
        duplicate = any(float(np.linalg.norm(refined_np - item.point)) <= dedup_tol for item in records)
        if duplicate:
            continue
        basin_id = _basin_id_for_point(env, refined, len(records))
        records.append(
            FixedPointRecord(
                basin_id=basin_id,
                point=refined_np,
                source=source,
                step_residual=step_residual,
                continuous_residual=continuous,
            )
        )
    if not records and not notes:
        notes.append("no_candidate_points")
    return sorted(records, key=lambda item: item.basin_id), notes


def _true_step_jacobian(env, point_np: np.ndarray, *, device: str) -> np.ndarray:
    point = torch.from_numpy(point_np).to(device=device, dtype=torch.float32)
    with torch.enable_grad():
        x = point.detach().clone().requires_grad_(True)

        def step_fn(inp: torch.Tensor) -> torch.Tensor:
            return env.step(inp).to(dtype=torch.float32)

        jac = torch.autograd.functional.jacobian(step_fn, x)
    return jac.detach().cpu().numpy().astype(np.float64, copy=False)


def _continuous_jacobian(env, point_np: np.ndarray, *, device: str) -> Optional[np.ndarray]:
    base = env.unwrapped
    dynamics = getattr(base, "dynamics", None)
    if dynamics is None and getattr(base, "system", None) is not None:
        dynamics = getattr(base.system, "dynamics", None)
    if dynamics is None:
        return None
    point = torch.from_numpy(point_np).to(device=device, dtype=torch.float32)
    try:
        with torch.enable_grad():
            x = point.detach().clone().requires_grad_(True)

            def dynamics_fn(inp: torch.Tensor) -> torch.Tensor:
                return dynamics(inp).to(dtype=torch.float32)

            jac = torch.autograd.functional.jacobian(dynamics_fn, x)
        return jac.detach().cpu().numpy().astype(np.float64, copy=False)
    except Exception:
        return None


def _encoder_jacobian(model, point_np: np.ndarray, *, device: str) -> np.ndarray:
    point = torch.from_numpy(point_np).to(device=device, dtype=torch.float32)
    with torch.enable_grad():
        x = point.detach().clone().requires_grad_(True)

        def encode_fn(inp: torch.Tensor) -> torch.Tensor:
            return model.encode(inp.unsqueeze(0)).squeeze(0)

        jac = torch.autograd.functional.jacobian(encode_fn, x)
    return jac.detach().cpu().numpy().astype(np.float64, copy=False)


def _decoder_jacobian(model, latent_np: np.ndarray, *, device: str) -> np.ndarray:
    latent = torch.from_numpy(latent_np).to(device=device, dtype=torch.float32)
    with torch.enable_grad():
        z = latent.detach().clone().requires_grad_(True)

        def decode_fn(inp: torch.Tensor) -> torch.Tensor:
            return model.decode(inp.unsqueeze(0)).squeeze(0)

        jac = torch.autograd.functional.jacobian(decode_fn, z)
    return jac.detach().cpu().numpy().astype(np.float64, copy=False)


def _encode_point(model, point_np: np.ndarray, *, device: str) -> np.ndarray:
    point = torch.from_numpy(point_np).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        return model.encode(point.unsqueeze(0)).squeeze(0).detach().cpu().numpy().astype(np.float64)


def _decode_point(model, latent_np: np.ndarray, *, device: str) -> np.ndarray:
    latent = torch.from_numpy(latent_np).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        return model.decode(latent.unsqueeze(0)).squeeze(0).detach().cpu().numpy().astype(np.float64)


def _fit_centered_operator(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge_lambda: float,
    center: np.ndarray,
    target_center: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if x.shape[0] == 0 or x.shape != y.shape:
        return None, None
    target = center if target_center is None else target_center
    operator = REDUCER._fit_linear_operator(x - center, y - target, ridge_lambda)
    return operator, center


def _match_eigenvalues(true_eig: np.ndarray, learned_eig: np.ndarray) -> List[Tuple[int, int]]:
    dim = int(true_eig.shape[0])
    if dim <= 6:
        best_perm: Optional[Tuple[int, ...]] = None
        best_cost = float("inf")
        for perm in itertools.permutations(range(dim)):
            cost = float(sum(abs(true_eig[i] - learned_eig[perm[i]]) for i in range(dim)))
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
        if best_perm is not None:
            return [(i, best_perm[i]) for i in range(dim)]
    true_order = np.lexsort((true_eig.imag, true_eig.real))
    learned_order = np.lexsort((learned_eig.imag, learned_eig.real))
    return list(zip(true_order.tolist(), learned_order.tolist()))


def _min_pairwise_eig_gap(eigvals: np.ndarray) -> float:
    if eigvals.shape[0] <= 1:
        return float("inf")
    gaps = []
    for i in range(eigvals.shape[0]):
        for j in range(i + 1, eigvals.shape[0]):
            gaps.append(abs(eigvals[i] - eigvals[j]))
    return float(np.min(gaps)) if gaps else float("inf")


def _matrix_condition_number(matrix: np.ndarray) -> Optional[float]:
    try:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    if singular_values.size == 0:
        return None
    smallest = float(np.min(singular_values))
    largest = float(np.max(singular_values))
    if smallest <= EPS:
        return float("inf")
    return largest / smallest


def _chart_metrics(
    *,
    model,
    fixed_point: np.ndarray,
    z_anchor: np.ndarray,
    encoder_jac: np.ndarray,
    decoder_jac: np.ndarray,
    device: str,
) -> Dict[str, object]:
    state_dim = int(fixed_point.shape[0])
    chart_identity = decoder_jac @ encoder_jac
    identity = np.eye(state_dim, dtype=np.float64)
    identity_error = float(np.linalg.norm(chart_identity - identity, ord="fro"))
    recon = _decode_point(model, z_anchor, device=device)
    recon_error = float(np.linalg.norm(recon - fixed_point.astype(np.float64)))
    anchor_norm = float(np.linalg.norm(fixed_point.astype(np.float64)))
    return {
        "anchor_reconstruction_error": recon_error,
        "anchor_reconstruction_rel_error": recon_error / max(anchor_norm, 1.0),
        "chart_identity_fro_error": identity_error,
        "chart_identity_fro_rel_error": identity_error / max(float(np.linalg.norm(identity, ord="fro")), EPS),
        "encoder_jacobian_rank": int(np.linalg.matrix_rank(encoder_jac)),
        "decoder_jacobian_rank": int(np.linalg.matrix_rank(decoder_jac)),
        "encoder_jacobian_condition": _matrix_condition_number(encoder_jac),
        "decoder_jacobian_condition": _matrix_condition_number(decoder_jac),
    }


def _geometry_metrics(
    true_jac: np.ndarray,
    learned_jac: np.ndarray,
) -> Dict[str, object]:
    true_norm = float(np.linalg.norm(true_jac, ord="fro"))
    diff_norm = float(np.linalg.norm(learned_jac - true_jac, ord="fro"))
    metrics: Dict[str, object] = {
        "state_fro_error": diff_norm,
        "state_fro_rel_error": None if true_norm <= EPS else diff_norm / true_norm,
    }
    try:
        true_eigvals, true_eigvecs = np.linalg.eig(true_jac)
        learned_eigvals, learned_eigvecs = np.linalg.eig(learned_jac)
        pairs = _match_eigenvalues(true_eigvals, learned_eigvals)
        eig_errors = [abs(true_eigvals[i] - learned_eigvals[j]) for i, j in pairs]
        true_gap = _min_pairwise_eig_gap(true_eigvals)
        learned_gap = _min_pairwise_eig_gap(learned_eigvals)
        metrics.update(
            {
                "eigval_mean_abs_error": float(np.mean(eig_errors)) if eig_errors else None,
                "eigval_max_abs_error": float(np.max(eig_errors)) if eig_errors else None,
                "true_min_pairwise_eig_gap": true_gap,
                "learned_min_pairwise_eig_gap": learned_gap,
                "true_spectral_radius": float(np.max(np.abs(true_eigvals))),
                "learned_spectral_radius": float(np.max(np.abs(learned_eigvals))),
                "spectral_radius_abs_error": float(
                    abs(np.max(np.abs(true_eigvals)) - np.max(np.abs(learned_eigvals)))
                ),
            }
        )
        cosines = []
        eigendir_status = "N/A:near_repeated_eigenvalues"
        if min(true_gap, learned_gap) > 1e-6:
            for true_idx, learned_idx in pairs:
                if abs(true_eigvals[true_idx].imag) > 1e-8 or abs(learned_eigvals[learned_idx].imag) > 1e-8:
                    continue
                true_vec = np.real(true_eigvecs[:, true_idx])
                learned_vec = np.real(learned_eigvecs[:, learned_idx])
                true_vec_norm = float(np.linalg.norm(true_vec))
                learned_vec_norm = float(np.linalg.norm(learned_vec))
                if true_vec_norm <= EPS or learned_vec_norm <= EPS:
                    continue
                cosines.append(abs(float(np.dot(true_vec, learned_vec) / (true_vec_norm * learned_vec_norm))))
            eigendir_status = "ok" if cosines else "N/A:no_real_simple_eigendirections"
        metrics["real_eigendirection_abs_cos_mean"] = float(np.mean(cosines)) if cosines else None
        metrics["eigendirection_status"] = eigendir_status
    except Exception as exc:
        metrics.update(
            {
                "eigval_mean_abs_error": None,
                "eigval_max_abs_error": None,
                "true_min_pairwise_eig_gap": None,
                "learned_min_pairwise_eig_gap": None,
                "true_spectral_radius": None,
                "learned_spectral_radius": None,
                "spectral_radius_abs_error": None,
                "real_eigendirection_abs_cos_mean": None,
                "eigendirection_status": f"N/A:eigendecomposition_failed:{repr(exc)}",
            }
        )
    return metrics


def _support_key_to_text(value: object) -> str:
    if isinstance(value, bytes):
        return "bytes:" + value.hex()[:32]
    return str(value)


def _near_attractor_assignments(states: np.ndarray, fixed_points: Sequence[FixedPointRecord]) -> Tuple[np.ndarray, np.ndarray]:
    points = np.stack([item.point for item in fixed_points], axis=0).astype(np.float32)
    deltas = states[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(deltas, axis=-1)
    nearest = dists.argmin(axis=1)
    nearest_dist = dists[np.arange(states.shape[0]), nearest]
    return nearest.astype(np.int64), nearest_dist.astype(np.float32)


def _prototype_support_size(labels: np.ndarray, support_masks: np.ndarray) -> Dict[object, float]:
    counters: Dict[object, List[float]] = defaultdict(list)
    for label, mask in zip(labels.tolist(), support_masks):
        counters[label].append(float(mask.sum()))
    return {key: float(np.mean(values)) for key, values in counters.items()}


def _skip_row(common: Dict[str, object], *, reason: str) -> Dict[str, object]:
    defaults = {
        "transition_count": 0.0,
        "support_size_mean": None,
        "latent_anchor_distance": None,
        "state_fro_error": None,
        "state_fro_rel_error": None,
        "eigval_mean_abs_error": None,
        "eigval_max_abs_error": None,
        "true_min_pairwise_eig_gap": None,
        "learned_min_pairwise_eig_gap": None,
        "true_spectral_radius": None,
        "learned_spectral_radius": None,
        "spectral_radius_abs_error": None,
        "real_eigendirection_abs_cos_mean": None,
        "anchor_reconstruction_error": None,
        "anchor_reconstruction_rel_error": None,
        "chart_identity_fro_error": None,
        "chart_identity_fro_rel_error": None,
        "encoder_jacobian_rank": None,
        "decoder_jacobian_rank": None,
        "encoder_jacobian_condition": None,
        "decoder_jacobian_condition": None,
    }
    return {
        **defaults,
        **common,
        "projection_status": "skipped",
        "skip_reason": reason,
        "eigendirection_status": "N/A:skipped",
    }


def _evaluate_partition_rows(
    *,
    common: Dict[str, object],
    model,
    device: str,
    true_jacobians: Dict[int, np.ndarray],
    x_all: np.ndarray,
    y_all: np.ndarray,
    labels_all: np.ndarray,
    near_fixed_idx: np.ndarray,
    near_fixed_dist: np.ndarray,
    support_sizes: Dict[object, float],
    fixed_points: Sequence[FixedPointRecord],
    projection_cache: Dict[int, Dict[str, object]],
    attractor_radius: float,
    min_operator_transitions: int,
    ridge_lambda: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    class_counts = Counter(labels_all.tolist())
    if len(class_counts) > int(common["max_partition_classes"]):
        return [_skip_row(common, reason=f"class_count>{common['max_partition_classes']}")]

    for class_id, _class_count in sorted(class_counts.items(), key=lambda item: str(item[0])):
        class_mask = labels_all == class_id
        if not bool(np.any(class_mask)):
            continue
        class_count = int(_class_count)
        near_radius_mask = class_mask & (near_fixed_dist <= float(attractor_radius))
        local_fixed_indices = sorted({int(idx) for idx in near_fixed_idx[near_radius_mask].tolist()})
        if not local_fixed_indices:
            fixed_counter = Counter(near_fixed_idx[class_mask].tolist())
            local_fixed_indices = [int(fixed_counter.most_common(1)[0][0])]

        near_radius_count = int(near_radius_mask.sum())
        for fixed_local_idx in local_fixed_indices:
            near_mask = (
                class_mask
                & (near_fixed_idx == int(fixed_local_idx))
                & (near_fixed_dist <= float(attractor_radius))
            )
            transition_count = int(near_mask.sum())
            fixed_point = fixed_points[int(fixed_local_idx)]
            row_common = {
                **common,
                "class_id": _support_key_to_text(class_id),
                "class_total_count": float(class_count),
                "class_near_radius_count": float(near_radius_count),
                "class_fixed_point_fraction": float(transition_count) / max(float(class_count), 1.0),
                "near_radius_fixed_point_fraction": (
                    None
                    if near_radius_count <= 0
                    else float(transition_count) / max(float(near_radius_count), 1.0)
                ),
                "fixed_point_basin_id": fixed_point.basin_id,
                "fixed_point_source": fixed_point.source,
                "fixed_point_step_residual": fixed_point.step_residual,
                "continuous_residual": fixed_point.continuous_residual,
                "transition_count": float(transition_count),
                "support_size_mean": support_sizes.get(class_id),
            }
            if transition_count < min_operator_transitions:
                rows.append(
                    _skip_row(
                        row_common,
                        reason=f"transition_count<{min_operator_transitions}",
                    )
                )
                continue

            try:
                cache_key = int(fixed_local_idx)
                if cache_key not in projection_cache:
                    z_cached = _encode_point(model, fixed_point.point, device=device)
                    enc_cached = _encoder_jacobian(model, fixed_point.point, device=device)
                    dec_cached = _decoder_jacobian(model, z_cached, device=device)
                    projection_cache[cache_key] = {
                        "z_anchor": z_cached,
                        "encoder_jac": enc_cached,
                        "decoder_jac": dec_cached,
                        "chart_metrics": _chart_metrics(
                            model=model,
                            fixed_point=fixed_point.point,
                            z_anchor=z_cached,
                            encoder_jac=enc_cached,
                            decoder_jac=dec_cached,
                            device=device,
                        ),
                    }
                cached = projection_cache[cache_key]
                z_anchor = cached["z_anchor"]
                encoder_jac = cached["encoder_jac"]
                decoder_jac = cached["decoder_jac"]
                operator, latent_center = _fit_centered_operator(
                    x_all[near_mask],
                    y_all[near_mask],
                    ridge_lambda=ridge_lambda,
                    center=z_anchor,
                )
                if operator is None or latent_center is None:
                    rows.append(_skip_row(row_common, reason="operator_fit_failed"))
                    continue

                # _fit_linear_operator uses row-vector convention: (z_t - c) @ A.
                # Autograd Jacobians use column-vector convention, so project A.T.
                learned_state_jac = decoder_jac @ operator.astype(np.float64).T @ encoder_jac
                true_jac = true_jacobians[int(fixed_local_idx)]
                sample_center = x_all[near_mask].mean(axis=0)
                latent_anchor_distance = float(np.linalg.norm(sample_center - z_anchor))
                rows.append(
                    {
                        **row_common,
                        "projection_status": "ok",
                        "skip_reason": "",
                        "latent_anchor_distance": latent_anchor_distance,
                        "operator_center_source": "encoded_fixed_point",
                        **cached["chart_metrics"],
                        **_geometry_metrics(true_jac, learned_state_jac),
                    }
                )
            except Exception as exc:
                rows.append(_skip_row(row_common, reason=f"projection_failed:{repr(exc)}"))
    return rows


def evaluate_run(
    spec: RunSpec,
    *,
    args: argparse.Namespace,
    support_definitions: Sequence[Tuple[str, float]],
    partition_kinds: Sequence[str],
) -> List[Dict[str, object]]:
    checkpoint_path = Path(spec.run_dir) / "checkpoint.pt"
    _cfg, env, model = REDUCER._load_checkpoint_model(checkpoint_path, spec.system_key, args.device)
    fixed_points, fixed_point_notes = _recover_fixed_points(
        env,
        refine_steps=args.fixed_point_refine_steps,
        residual_tol=args.fixed_point_residual_tol,
        dedup_tol=args.fixed_point_dedup_tol,
    )

    run_common = {
        "root_label": spec.root_label,
        "system_key": spec.system_key,
        "system_name": spec.system_name,
        "seed": spec.seed,
        "run_dir": spec.run_dir,
        "device": args.device,
        "label_mode": args.label_mode,
        "ridge_lambda": args.ridge_lambda,
        "min_operator_transitions": args.min_operator_transitions,
        "max_partition_classes": args.max_partition_classes,
    }
    if not fixed_points:
        return [
            _skip_row(
                {
                    **run_common,
                    "support_definition": "N/A",
                    "partition_kind": "N/A",
                    "control_kind": "N/A",
                    "class_id": "N/A",
                    "attractor_radius": None,
                    "fixed_point_basin_id": None,
                    "fixed_point_source": None,
                    "fixed_point_step_residual": None,
                    "continuous_residual": None,
                    "fixed_point_notes": ";".join(fixed_point_notes),
                },
                reason="no_reliable_fixed_points",
            )
        ]

    true_jacobians: Dict[int, np.ndarray] = {}
    continuous_jacobian_available = 0
    for idx, fixed_point in enumerate(fixed_points):
        true_jacobians[idx] = _true_step_jacobian(env, fixed_point.point, device=args.device)
        if _continuous_jacobian(env, fixed_point.point, device=args.device) is not None:
            continuous_jacobian_available += 1

    trajectories = REDUCER._generate_observation_trajectories(
        env,
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        eval_seed=args.eval_seed,
    )
    basin_labels, _centers, label_source = OPSEL._label_sequences_for_mode(
        env,
        trajectories,
        system_key=spec.system_key,
        endpoint_rollout_steps=args.endpoint_rollout_steps,
        label_mode=args.label_mode,
    )
    basin_np = basin_labels.cpu().numpy() if isinstance(basin_labels, torch.Tensor) else np.asarray(basin_labels)
    latents = REDUCER._encode_trajectories(model, trajectories, args.device)
    x_all = latents[:, :-1, :].reshape(-1, latents.shape[-1]).astype(np.float64, copy=False)
    y_all = latents[:, 1:, :].reshape(-1, latents.shape[-1]).astype(np.float64, copy=False)
    states_cur = trajectories[:, :-1, :].reshape(-1, trajectories.shape[-1]).cpu().numpy().astype(np.float32)
    near_fixed_idx, near_fixed_dist = _near_attractor_assignments(states_cur, fixed_points)
    basin_cur = basin_np[:, :-1].reshape(-1).astype(object)
    attractor_cur = near_fixed_idx.astype(object)
    attractor_radii = _parse_csv_floats(args.attractor_radii) or [float(args.attractor_radius)]

    rows: List[Dict[str, object]] = []
    projection_cache: Dict[int, Dict[str, object]] = {}
    rng = np.random.default_rng(args.eval_seed + 10_000 * int(spec.seed))
    for scheme, value in support_definitions:
        support_definition = _support_name(scheme, value)
        support_mask = REDUCER._support_mask(latents, scheme=scheme, value=value)
        support_keys = REDUCER._support_keys(support_mask)
        family_labels = REDUCER.support_family_labels(
            support_mask,
            min_jaccard=args.family_jaccard_threshold,
        )

        support_cur = support_keys[:, :-1].reshape(-1).astype(object)
        family_cur = family_labels[:, :-1].reshape(-1).astype(object)
        support_source_masks = support_mask[:, :-1, :].reshape(-1, support_mask.shape[-1])

        labels_by_partition = {
            "attractor": attractor_cur,
            "basin": basin_cur,
            "family": family_cur,
            "support": support_cur,
        }
        support_sizes_by_partition = {
            "attractor": _prototype_support_size(attractor_cur, support_source_masks),
            "basin": _prototype_support_size(basin_cur, support_source_masks),
            "family": _prototype_support_size(family_cur, support_source_masks),
            "support": _prototype_support_size(support_cur, support_source_masks),
        }

        for attractor_radius in attractor_radii:
            for partition_kind in partition_kinds:
                if partition_kind not in labels_by_partition:
                    continue
                labels = labels_by_partition[partition_kind]
                common = {
                    **run_common,
                    "support_definition": support_definition,
                    "partition_kind": partition_kind,
                    "control_kind": "observed",
                    "label_source": label_source,
                    "attractor_radius": float(attractor_radius),
                    "fixed_point_count": float(len(fixed_points)),
                    "continuous_jacobian_count": float(continuous_jacobian_available),
                    "fixed_point_notes": ";".join(fixed_point_notes),
                }
                rows.extend(
                    _evaluate_partition_rows(
                        common=common,
                        model=model,
                        device=args.device,
                        true_jacobians=true_jacobians,
                        x_all=x_all,
                        y_all=y_all,
                        labels_all=labels,
                        near_fixed_idx=near_fixed_idx,
                        near_fixed_dist=near_fixed_dist,
                        support_sizes=support_sizes_by_partition[partition_kind],
                        fixed_points=fixed_points,
                        projection_cache=projection_cache,
                        attractor_radius=float(attractor_radius),
                        min_operator_transitions=args.min_operator_transitions,
                        ridge_lambda=args.ridge_lambda,
                    )
                )
                for control_idx in range(max(0, int(args.num_random_controls))):
                    random_labels = labels[rng.permutation(labels.shape[0])]
                    random_common = {
                        **common,
                        "control_kind": "random_count_matched",
                        "random_control_index": control_idx,
                    }
                    rows.extend(
                        _evaluate_partition_rows(
                            common=random_common,
                            model=model,
                            device=args.device,
                            true_jacobians=true_jacobians,
                            x_all=x_all,
                            y_all=y_all,
                            labels_all=random_labels,
                            near_fixed_idx=near_fixed_idx,
                            near_fixed_dist=near_fixed_dist,
                            support_sizes=support_sizes_by_partition[partition_kind],
                            fixed_points=fixed_points,
                            projection_cache=projection_cache,
                            attractor_radius=float(attractor_radius),
                            min_operator_transitions=args.min_operator_transitions,
                            ridge_lambda=args.ridge_lambda,
                        )
                    )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_mean(values: Iterable[object]) -> Optional[float]:
    clean = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    return float(np.mean(clean)) if clean else None


def _fmt(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _write_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("root_label", "")),
            str(row.get("support_definition", "")),
            str(row.get("partition_kind", "")),
            str(row.get("control_kind", "")),
            str(row.get("attractor_radius", "")),
        )
        grouped[key].append(row)

    lines = [
        "# True Jacobian Geometry Summary",
        "",
        "Conservative fixed-point/eigendirection comparison for support-conditioned centered local laws.",
        "",
        "| root | support | partition | control | radius | rows | ok rows | skipped rows | mean rel Fro | mean eig abs | mean eigdir cos | mean chart-id rel |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, group in sorted(grouped.items()):
        root, support, partition, control, radius = key
        ok_count = sum(1 for row in group if row.get("projection_status") == "ok")
        skip_count = len(group) - ok_count
        lines.append(
            f"| `{root}` | `{support}` | `{partition}` | `{control}` | {radius} | "
            f"{len(group)} | {ok_count} | {skip_count} | "
            f"{_fmt(_safe_mean(row.get('state_fro_rel_error') for row in group))} | "
            f"{_fmt(_safe_mean(row.get('eigval_mean_abs_error') for row in group))} | "
            f"{_fmt(_safe_mean(row.get('real_eigendirection_abs_cos_mean') for row in group))} | "
            f"{_fmt(_safe_mean(row.get('chart_identity_fro_rel_error') for row in group))} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    rows_csvs: Sequence[str],
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
    support_definitions: Sequence[Tuple[str, float]],
    partition_kinds: Sequence[str],
    num_specs: int,
    completed_specs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    status: str,
    elapsed_seconds: float,
) -> None:
    payload = {
        "rows_csvs": list(rows_csvs),
        "root_labels": list(root_labels),
        "systems": list(systems),
        "seeds": list(seeds),
        "support_definitions": [{"scheme": scheme, "value": value} for scheme, value in support_definitions],
        "partition_kinds": list(partition_kinds),
        "num_trajectories": args.num_trajectories,
        "trajectory_length": args.trajectory_length,
        "eval_seed": args.eval_seed,
        "endpoint_rollout_steps": args.endpoint_rollout_steps,
        "fixed_point_refine_steps": args.fixed_point_refine_steps,
        "fixed_point_residual_tol": args.fixed_point_residual_tol,
        "attractor_radius": args.attractor_radius,
        "attractor_radii": _parse_csv_floats(args.attractor_radii) or [float(args.attractor_radius)],
        "min_operator_transitions": args.min_operator_transitions,
        "num_random_controls": args.num_random_controls,
        "smoke": bool(args.smoke),
        "num_runs": num_specs,
        "completed_runs": completed_specs,
        "remaining_runs": max(0, num_specs - completed_specs),
        "num_rows": len(rows),
        "ok_rows": sum(1 for row in rows if row.get("projection_status") == "ok"),
        "num_failures": len(failures),
        "elapsed_seconds": elapsed_seconds,
        "status": status,
    }
    path.write_text(json.dumps(payload, indent=2))


def _flush_outputs(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    rows_csvs: Sequence[str],
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
    support_definitions: Sequence[Tuple[str, float]],
    partition_kinds: Sequence[str],
    num_specs: int,
    completed_specs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    status: str,
    elapsed_seconds: float,
) -> None:
    _write_csv(output_dir / "true_jacobian_geometry_rows.csv", rows)
    _write_summary(output_dir / "true_jacobian_geometry_summary.md", rows)
    (output_dir / "failures.json").write_text(json.dumps(list(failures), indent=2))
    (output_dir / "progress.json").write_text(
        json.dumps(
            {
                "completed_runs": completed_specs,
                "num_runs": num_specs,
                "remaining_runs": max(0, num_specs - completed_specs),
                "num_rows": len(rows),
                "ok_rows": sum(1 for row in rows if row.get("projection_status") == "ok"),
                "num_failures": len(failures),
                "elapsed_seconds": elapsed_seconds,
            },
            indent=2,
        )
    )
    _write_manifest(
        output_dir / "manifest.json",
        args=args,
        rows_csvs=rows_csvs,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        support_definitions=support_definitions,
        partition_kinds=partition_kinds,
        num_specs=num_specs,
        completed_specs=completed_specs,
        rows=rows,
        failures=failures,
        status=status,
        elapsed_seconds=elapsed_seconds,
    )


def main() -> None:
    args = _parse_args()
    _apply_smoke_overrides(args)
    rows_csvs = _parse_csv_strings(args.rows_csvs)
    root_labels = _parse_csv_strings(args.root_labels)
    systems = _parse_csv_strings(args.systems)
    seeds = _parse_csv_ints(args.seeds)
    support_definitions = _parse_support_definitions(args.support_definitions)
    partition_kinds = _parse_csv_strings(args.partition_kinds)
    specs = _load_latest_specs(
        [Path(item) for item in rows_csvs],
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        max_runs=int(args.max_runs),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    start_time = time.time()
    progress_every_runs = max(1, int(args.progress_every_runs))
    flush_every_runs = max(0, int(args.flush_every_runs))

    _flush_outputs(
        output_dir,
        args=args,
        rows_csvs=rows_csvs,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        support_definitions=support_definitions,
        partition_kinds=partition_kinds,
        num_specs=len(specs),
        completed_specs=0,
        rows=rows,
        failures=failures,
        status="running",
        elapsed_seconds=0.0,
    )

    for index, spec in enumerate(specs, start=1):
        try:
            run_rows = evaluate_run(
                spec,
                args=args,
                support_definitions=support_definitions,
                partition_kinds=partition_kinds,
            )
            rows.extend(run_rows)
            if index % progress_every_runs == 0:
                print(
                    f"[{index}/{len(specs)}] ok root={spec.root_label} system={spec.system_key} "
                    f"seed={spec.seed} rows={len(rows)}"
                )
        except Exception as exc:  # pragma: no cover - surfaced in artifacts
            failures.append(
                {
                    "root_label": spec.root_label,
                    "system_key": spec.system_key,
                    "seed": spec.seed,
                    "run_dir": spec.run_dir,
                    "error": repr(exc),
                }
            )
            print(
                f"[{index}/{len(specs)}] error root={spec.root_label} system={spec.system_key} "
                f"seed={spec.seed}: {exc}",
                file=sys.stderr,
            )
        if flush_every_runs and index % flush_every_runs == 0:
            _flush_outputs(
                output_dir,
                args=args,
                rows_csvs=rows_csvs,
                root_labels=root_labels,
                systems=systems,
                seeds=seeds,
                support_definitions=support_definitions,
                partition_kinds=partition_kinds,
                num_specs=len(specs),
                completed_specs=index,
                rows=rows,
                failures=failures,
                status="running",
                elapsed_seconds=time.time() - start_time,
            )

    _flush_outputs(
        output_dir,
        args=args,
        rows_csvs=rows_csvs,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        support_definitions=support_definitions,
        partition_kinds=partition_kinds,
        num_specs=len(specs),
        completed_specs=len(specs),
        rows=rows,
        failures=failures,
        status="complete" if not failures else "complete_with_failures",
        elapsed_seconds=time.time() - start_time,
    )


if __name__ == "__main__":
    main()
