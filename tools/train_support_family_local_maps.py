#!/usr/bin/env python3
"""Stage-2 training for support-family local transition maps.

This script keeps an existing sparse Koopman autoencoder checkpoint fixed and
trains either route-local centered latent maps selected by the model's own
``F_top8`` support families, or a single calibrated global latent map as an
ablation. The training loss is decoded multi-step rollout MSE under the same
periodic re-encoding loop used at evaluation time.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import signal
import sys
import time
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

STOP_REQUESTED = False
INVALID_ROUTE = "__invalid__"
FALLBACK_ROUTE = "__global_fallback__"


def _load_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SRF = _load_module(
    "evaluate_transition_rich_self_routed_forecasting.py",
    "evaluate_transition_rich_self_routed_forecasting_stage2",
)
REDUCER = SRF.REDUCER
OPSEL = SRF.OPSEL


class LocalMapBundle(nn.Module):
    """Trainable centered local maps for a fixed route codebook."""

    def __init__(
        self,
        *,
        family_ids: Sequence[object],
        centers: Dict[object, np.ndarray],
        global_k: np.ndarray,
        device: str,
    ) -> None:
        super().__init__()
        self.family_ids = [str(item) for item in family_ids]
        self.family_to_index = {family_id: idx for idx, family_id in enumerate(self.family_ids)}
        center_array = np.stack([centers[item] for item in family_ids], axis=0).astype(np.float32, copy=False)
        self.register_buffer("centers", torch.from_numpy(center_array).to(device=device))
        init_k = torch.from_numpy(global_k.astype(np.float32, copy=False)).to(device=device)
        self.local_maps = nn.Parameter(init_k.unsqueeze(0).repeat(len(self.family_ids), 1, 1))
        self.register_buffer("global_k", init_k)

    def forward(self, z: torch.Tensor, route_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = route_index >= 0
        out = z @ self.global_k
        if bool(valid.any()):
            selected = route_index[valid]
            centers = self.centers[selected]
            maps = self.local_maps[selected]
            z_valid = z[valid]
            out[valid] = centers + torch.bmm((z_valid - centers).unsqueeze(1), maps).squeeze(1)
        return out, valid


class GlobalMapBundle(nn.Module):
    """Trainable single global map initialized from the checkpoint map."""

    def __init__(
        self,
        *,
        family_ids: Sequence[object],
        global_k: np.ndarray,
        device: str,
    ) -> None:
        super().__init__()
        self.family_ids = [str(item) for item in family_ids]
        self.family_to_index = {family_id: idx for idx, family_id in enumerate(self.family_ids)}
        init_k = torch.from_numpy(global_k.astype(np.float32, copy=False)).to(device=device)
        self.global_map = nn.Parameter(init_k.clone())
        self.register_buffer("initial_global_k", init_k)

    def forward(self, z: torch.Tensor, route_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        del route_index
        used_global = torch.ones(z.shape[0], dtype=torch.bool, device=z.device)
        return z @ self.global_map, used_global


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csvs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--root_labels", required=True)
    parser.add_argument("--systems", default="")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--support_definition", default="topk:8")
    parser.add_argument("--reencode_periods", default="1,2,5,10")
    parser.add_argument(
        "--route_freeze_modes",
        default="reroute_each_step,freeze_within_segment",
        help="comma-separated modes from {reroute_each_step,freeze_within_segment}",
    )
    parser.add_argument("--train_steps", type=int, default=20000)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--train_pool_trajectories", type=int, default=4096)
    parser.add_argument("--train_pool_seed", type=int, default=20260505)
    parser.add_argument("--fit_num_trajectories", type=int, default=256)
    parser.add_argument("--fit_trajectory_length", type=int, default=256)
    parser.add_argument("--fit_eval_seed", type=int, default=42)
    parser.add_argument("--forecast_num_trajectories", type=int, default=128)
    parser.add_argument("--forecast_eval_seed", type=int, default=314)
    parser.add_argument("--horizons", default="100,500,1000")
    parser.add_argument("--train_horizon", type=int, default=0, help="0 uses checkpoint TRAIN.SEQUENCE_LENGTH")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_operator_transitions", type=int, default=50)
    parser.add_argument("--family_jaccard_threshold", type=float, default=0.4)
    parser.add_argument(
        "--stage2_map_mode",
        default="family_local_centered",
        choices=["family_local_centered", "global_dense_calibrated"],
        help=(
            "family_local_centered trains one centered K_c per retained support family; "
            "global_dense_calibrated trains one dense global K initialized from the checkpoint."
        ),
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--label_mode", default="auto", choices=["auto", "native", "env_points", "estimated_centers", "none"])
    parser.add_argument("--endpoint_rollout_steps", type=int, default=5000)
    parser.add_argument("--progress_every_steps", type=int, default=500)
    parser.add_argument("--flush_every_runs", type=int, default=1)
    parser.add_argument("--max_runtime_seconds", type=int, default=0)
    parser.add_argument(
        "--resume_from_output_dirs",
        default="",
        help=(
            "comma-separated prior stage-2 output roots or shard dirs. If the "
            "current shard has no train_checkpoint.pt, load the matching "
            "stage2_runs/<run_slug>/train_checkpoint.pt from these roots."
        ),
    )
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument(
        "--save_metrics_history",
        action="store_true",
        help="Write per-run metrics_history.jsonl files. Off by default.",
    )
    parser.add_argument(
        "--save_train_checkpoint",
        action="store_true",
        help="Write train_checkpoint.pt files for resumability. Off by default.",
    )
    parser.add_argument(
        "--save_stage2_artifacts",
        action="store_true",
        help="Write local_maps.pt/global_map.pt tensors. Off by default; CSV/JSON summaries are still written.",
    )
    return parser.parse_args()


def _request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; will stop after the current run.", flush=True)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(content)
    tmp_path.replace(path)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        _atomic_write_text(path, "")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    _atomic_write_text(path, buffer.getvalue())


def _save_train_checkpoint(
    path: Path,
    *,
    bundle: nn.Module,
    optimizer: torch.optim.Optimizer,
    next_step: int,
    rng: np.random.Generator,
    last_metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(
        {
            "next_step": int(next_step),
            "bundle_state": bundle.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": rng.bit_generator.state,
            "last_metrics": dict(last_metrics),
        },
        tmp_path,
    )
    tmp_path.replace(path)


def _load_existing_rows(path: Path) -> Tuple[List[Dict[str, object]], set[Tuple[str, str, int, str, int, str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], set()
    with path.open("r", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    completed = {
        (
            str(row.get("root_label", "")),
            str(row.get("system_key", "")),
            int(row.get("seed", 0)),
            str(row.get("run_dir", "")),
            int(float(row.get("reencode_period", 0))),
            str(row.get("route_freeze_mode", "")),
            str(row.get("stage2_map_mode", "family_local_centered") or "family_local_centered"),
        )
        for row in rows
        if row.get("run_dir")
    }
    return rows, completed


def _parse_route_freeze_modes(raw: str) -> List[str]:
    modes = SRF._parse_csv_strings(raw)
    allowed = {"reroute_each_step", "freeze_within_segment"}
    unknown = [item for item in modes if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown route_freeze_modes: {unknown}")
    return modes


def _stringify_support_key(key: object) -> str:
    if isinstance(key, bytes):
        return key.hex()
    return str(key)


def _make_slug(*parts: object) -> str:
    text = "__".join(str(part) for part in parts)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _find_prior_train_checkpoint(prior_output_dirs: Sequence[str], run_slug: str) -> Optional[Path]:
    """Find a matching prior train checkpoint for a stage-2 run slug."""
    for raw_base in prior_output_dirs:
        if not raw_base:
            continue
        base = Path(raw_base)
        candidates = [
            base / "stage2_runs" / run_slug / "train_checkpoint.pt",
            base / run_slug / "train_checkpoint.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        shards_dir = base / "shards"
        if shards_dir.exists():
            matches = sorted(shards_dir.glob(f"*/stage2_runs/{run_slug}/train_checkpoint.pt"))
            if matches:
                return matches[0]
        if base.exists():
            matches = sorted(base.glob(f"**/stage2_runs/{run_slug}/train_checkpoint.pt"))
            if matches:
                return matches[0]
    return None


def _build_route_codebook(
    fit_latents: np.ndarray,
    *,
    scheme: str,
    value: float,
    min_operator_transitions: int,
    family_jaccard_threshold: float,
) -> Dict[str, object]:
    support_mask = REDUCER._support_mask(fit_latents, scheme=scheme, value=value)
    support_keys = REDUCER._support_keys(support_mask)
    family_labels = REDUCER.support_family_labels(support_mask, min_jaccard=family_jaccard_threshold)
    x_fit = fit_latents[:, :-1, :].reshape(-1, fit_latents.shape[-1]).astype(np.float32, copy=False)
    support_cur = support_keys[:, :-1].reshape(-1).astype(object)
    family_cur = family_labels[:, :-1].reshape(-1).astype(object)
    counts = Counter(family_cur.tolist())
    fitted_family_ids = sorted(
        [family_id for family_id, count in counts.items() if int(count) >= int(min_operator_transitions)],
        key=lambda item: str(item),
    )
    centers: Dict[object, np.ndarray] = {}
    for family_id in fitted_family_ids:
        centers[family_id] = x_fit[family_cur == family_id].mean(axis=0).astype(np.float32, copy=False)
    flat_support_mask = support_mask[:, :-1, :].reshape(-1, support_mask.shape[-1])
    family_prototypes = OPSEL._prototype_masks_from_exact_support(
        family_cur,
        support_cur,
        flat_support_mask,
        class_kind="family",
    )
    support_key_to_family: Dict[object, object] = {}
    for support_key, family_id in zip(support_cur.tolist(), family_cur.tolist()):
        if support_key not in support_key_to_family:
            support_key_to_family[support_key] = family_id
    return {
        "support_mask": support_mask,
        "family_labels": family_labels,
        "family_counts": counts,
        "fitted_family_ids": fitted_family_ids,
        "centers": centers,
        "family_prototypes": family_prototypes,
        "support_key_to_family": support_key_to_family,
    }


def _assign_family_ids_np(
    latents: np.ndarray,
    *,
    scheme: str,
    value: float,
    family_jaccard_threshold: float,
    support_key_to_family: Dict[object, object],
    family_prototypes: Dict[object, np.ndarray],
    family_cache: Dict[object, object],
) -> np.ndarray:
    support_masks = REDUCER._support_mask(latents, scheme=scheme, value=value)
    support_keys = REDUCER._support_keys(support_masks)
    return SRF._assign_family_ids(
        support_masks,
        support_keys,
        support_key_to_family=support_key_to_family,
        family_prototypes=family_prototypes,
        min_jaccard=family_jaccard_threshold,
        cache=family_cache,
    )


def _route_indices_np(
    latents: np.ndarray,
    *,
    scheme: str,
    value: float,
    family_jaccard_threshold: float,
    support_key_to_family: Dict[object, object],
    family_prototypes: Dict[object, np.ndarray],
    family_to_index: Dict[str, int],
    family_cache: Dict[object, object],
) -> np.ndarray:
    family_ids = _assign_family_ids_np(
        latents,
        scheme=scheme,
        value=value,
        family_jaccard_threshold=family_jaccard_threshold,
        support_key_to_family=support_key_to_family,
        family_prototypes=family_prototypes,
        family_cache=family_cache,
    )
    out = np.full(family_ids.shape[0], -1, dtype=np.int64)
    for idx, family_id in enumerate(family_ids.tolist()):
        if family_id is None:
            continue
        mapped = family_to_index.get(str(family_id))
        if mapped is not None:
            out[idx] = int(mapped)
    return out


def _balanced_indices_by_route(route_indices: np.ndarray) -> Dict[int, np.ndarray]:
    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx, route_index in enumerate(route_indices.tolist()):
        if int(route_index) >= 0:
            buckets[int(route_index)].append(idx)
    return {key: np.asarray(values, dtype=np.int64) for key, values in buckets.items() if values}


def _sample_balanced_indices(
    buckets: Dict[int, np.ndarray],
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    route_ids = np.asarray(sorted(buckets.keys()), dtype=np.int64)
    if route_ids.size == 0:
        raise RuntimeError("No trainable route buckets are available for route-balanced sampling")
    sampled_routes = rng.choice(route_ids, size=int(batch_size), replace=True)
    out = np.empty(int(batch_size), dtype=np.int64)
    for idx, route_id in enumerate(sampled_routes.tolist()):
        choices = buckets[int(route_id)]
        out[idx] = int(rng.choice(choices))
    rng.shuffle(out)
    return out


def _step_routes_for_torch(
    z: torch.Tensor,
    *,
    scheme: str,
    value: float,
    family_jaccard_threshold: float,
    support_key_to_family: Dict[object, object],
    family_prototypes: Dict[object, np.ndarray],
    family_to_index: Dict[str, int],
    family_cache: Dict[object, object],
    device: torch.device,
) -> torch.Tensor:
    route_np = _route_indices_np(
        z.detach().cpu().numpy().astype(np.float32, copy=False),
        scheme=scheme,
        value=value,
        family_jaccard_threshold=family_jaccard_threshold,
        support_key_to_family=support_key_to_family,
        family_prototypes=family_prototypes,
        family_to_index=family_to_index,
        family_cache=family_cache,
    )
    return torch.from_numpy(route_np).to(device=device, dtype=torch.long)


def _train_one_local_bundle(
    model,
    bundle: nn.Module,
    train_pool: torch.Tensor,
    initial_route_indices: np.ndarray,
    *,
    scheme: str,
    support_value: float,
    family_jaccard_threshold: float,
    support_key_to_family: Dict[object, object],
    family_prototypes: Dict[object, np.ndarray],
    reencode_period: int,
    route_freeze_mode: str,
    train_steps: int,
    train_batch_size: int,
    lr: float,
    seed: int,
    progress_every_steps: int,
    metrics_path: Optional[Path],
    checkpoint_path: Optional[Path],
    initial_checkpoint_path: Optional[Path],
    resume: bool,
    max_runtime_seconds: int,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(bundle.parameters(), lr=float(lr), weight_decay=0.0)
    buckets = _balanced_indices_by_route(initial_route_indices)
    rng = np.random.default_rng(seed)
    horizon = int(train_pool.shape[1]) - 1
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
    family_cache: Dict[object, object] = {}
    last_metrics: Dict[str, float] = {}
    start_step = 0
    train_start = time.time()

    checkpoint_to_load = (
        checkpoint_path
        if checkpoint_path is not None and checkpoint_path.exists()
        else initial_checkpoint_path
    )
    if resume and checkpoint_to_load is not None and checkpoint_to_load.exists():
        try:
            payload = torch.load(checkpoint_to_load, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_to_load, map_location=device)
        bundle.load_state_dict(payload["bundle_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if "rng_state" in payload:
            rng.bit_generator.state = payload["rng_state"]
        start_step = int(payload.get("next_step", 0))
        last_metrics = dict(payload.get("last_metrics", {}))
        print(
            f"  resumed stage2 training from step {start_step}/{train_steps} "
            f"using {checkpoint_to_load}",
            flush=True,
        )

    if start_step >= int(train_steps):
        return last_metrics

    for step in range(start_step, int(train_steps)):
        batch_indices = _sample_balanced_indices(buckets, batch_size=train_batch_size, rng=rng)
        x_seq = train_pool[batch_indices].to(device)
        x_true = x_seq[:, 1:, :]
        optimizer.zero_grad()
        with torch.no_grad():
            z = model.encode(x_seq[:, 0, :])

        preds: List[torch.Tensor] = []
        used_locals: List[torch.Tensor] = []
        frozen_route: Optional[torch.Tensor] = None
        for offset in range(horizon):
            if (
                route_freeze_mode == "reroute_each_step"
                or frozen_route is None
                or (int(reencode_period) > 0 and offset % int(reencode_period) == 0)
            ):
                frozen_route = _step_routes_for_torch(
                    z,
                    scheme=scheme,
                    value=support_value,
                    family_jaccard_threshold=family_jaccard_threshold,
                    support_key_to_family=support_key_to_family,
                    family_prototypes=family_prototypes,
                    family_to_index=bundle.family_to_index,
                    family_cache=family_cache,
                    device=device,
                )
            z_next, used_local = bundle(z, frozen_route)
            x_pred = model.decode(z_next)
            preds.append(x_pred)
            used_locals.append(used_local.float())
            if int(reencode_period) > 0 and (offset + 1) % int(reencode_period) == 0:
                z = model.encode(x_pred)
            else:
                z = z_next
        x_pred_seq = torch.stack(preds, dim=1)
        loss = torch.mean((x_pred_seq - x_true) ** 2)
        loss.backward()
        optimizer.step()

        if step % max(1, int(progress_every_steps)) == 0 or step == int(train_steps) - 1:
            coverage = torch.stack(used_locals, dim=1).mean().detach().cpu().item()
            last_metrics = {
                "step": float(step),
                "loss": float(loss.detach().cpu().item()),
                "route_coverage": float(coverage),
            }
            if metrics_path is not None:
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(last_metrics) + "\n")
            if checkpoint_path is not None:
                _save_train_checkpoint(
                    checkpoint_path,
                    bundle=bundle,
                    optimizer=optimizer,
                    next_step=step + 1,
                    rng=rng,
                    last_metrics=last_metrics,
                )
            print(
                f"  stage2 step {step}/{train_steps} loss={last_metrics['loss']:.6g} "
                f"coverage={last_metrics['route_coverage']:.3f}",
                flush=True,
            )
        elapsed = time.time() - train_start
        if STOP_REQUESTED or (int(max_runtime_seconds) > 0 and elapsed >= int(max_runtime_seconds)):
            if checkpoint_path is not None:
                _save_train_checkpoint(
                    checkpoint_path,
                    bundle=bundle,
                    optimizer=optimizer,
                    next_step=step + 1,
                    rng=rng,
                    last_metrics=last_metrics,
                )
                checkpoint_msg = f"; checkpoint saved to {checkpoint_path}"
            else:
                checkpoint_msg = "; rerun with --save_train_checkpoint to resume partial progress"
            raise RuntimeError(
                f"Stage-2 training stopped at step {step + 1}/{train_steps}{checkpoint_msg}"
            )
    return last_metrics


def _rollout_stage2_local(
    model,
    x0: torch.Tensor,
    *,
    max_horizon: int,
    device: str,
    local_bundle: LocalMapBundle,
    scheme: str,
    support_value: float,
    family_jaccard_threshold: float,
    support_key_to_family: Dict[object, object],
    family_prototypes: Dict[object, np.ndarray],
    reencode_period: int,
    route_freeze_mode: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    model.eval()
    model_device = next(model.parameters()).device
    x0 = x0.to(model_device)
    with torch.no_grad():
        latent = model.encode(x0).detach().cpu().numpy().astype(np.float32, copy=False)

    global_k = local_bundle.global_k.detach().cpu().numpy().astype(np.float32, copy=False)
    local_maps = local_bundle.local_maps.detach().cpu().numpy().astype(np.float32, copy=False)
    centers = local_bundle.centers.detach().cpu().numpy().astype(np.float32, copy=False)
    family_to_index = dict(local_bundle.family_to_index)
    batch, latent_dim = latent.shape
    obs_dim = int(x0.shape[-1])
    predictions = np.full((int(max_horizon), batch, obs_dim), np.nan, dtype=np.float32)
    used_local = np.zeros((int(max_horizon), batch), dtype=bool)
    route_labels = np.empty((int(max_horizon), batch), dtype=object)
    route_labels[:] = INVALID_ROUTE
    valid = np.ones(batch, dtype=bool)
    family_cache: Dict[object, object] = {}
    frozen_route_index = np.full(batch, -1, dtype=np.int64)

    for step in range(int(max_horizon)):
        if not bool(np.any(valid)):
            break
        next_latent = np.full((batch, latent_dim), np.nan, dtype=np.float32)
        valid_indices = np.flatnonzero(valid)
        current_latent = latent[valid_indices]
        should_select = (
            route_freeze_mode == "reroute_each_step"
            or step == 0
            or (int(reencode_period) > 0 and step % int(reencode_period) == 0)
        )
        if should_select:
            selected_route = _route_indices_np(
                current_latent,
                scheme=scheme,
                value=support_value,
                family_jaccard_threshold=family_jaccard_threshold,
                support_key_to_family=support_key_to_family,
                family_prototypes=family_prototypes,
                family_to_index=family_to_index,
                family_cache=family_cache,
            )
            frozen_route_index[valid_indices] = selected_route
        else:
            selected_route = frozen_route_index[valid_indices]

        next_valid = current_latent @ global_k
        labels = np.full(valid_indices.shape[0], FALLBACK_ROUTE, dtype=object)
        for route_index in sorted({int(item) for item in selected_route.tolist() if int(item) >= 0}):
            select = selected_route == route_index
            next_valid[select] = centers[route_index] + (current_latent[select] - centers[route_index]) @ local_maps[route_index]
            used_local[step, valid_indices[select]] = True
            labels[select] = local_bundle.family_ids[route_index]
        next_latent[valid_indices] = next_valid
        route_labels[step, valid_indices] = labels

        with torch.no_grad():
            pred_state = model.decode(
                torch.from_numpy(next_latent).to(device=model_device, dtype=x0.dtype)
            ).detach().cpu().numpy()
        predictions[step] = pred_state.astype(np.float32, copy=False)

        finite_mask = np.logical_and(
            np.all(np.isfinite(next_latent), axis=1),
            np.all(np.isfinite(pred_state), axis=1),
        )
        latent_for_next = next_latent
        if int(reencode_period) > 0 and (step + 1) % int(reencode_period) == 0:
            refresh_indices = np.flatnonzero(np.logical_and(valid, finite_mask))
            refreshed_latent = np.full_like(next_latent, np.nan)
            if refresh_indices.size:
                with torch.no_grad():
                    refreshed = model.encode(
                        torch.from_numpy(pred_state[refresh_indices]).to(model_device, dtype=x0.dtype)
                    ).detach().cpu().numpy().astype(np.float32, copy=False)
                refreshed_latent[refresh_indices] = refreshed
            finite_mask = np.logical_and(finite_mask, np.all(np.isfinite(refreshed_latent), axis=1))
            latent_for_next = refreshed_latent
        valid = np.logical_and(valid, finite_mask)
        latent = latent_for_next

    return predictions, SRF._summarize_route_metrics(predictions, used_local, route_labels)


def _rollout_stage2_global(
    model,
    x0: torch.Tensor,
    *,
    max_horizon: int,
    global_bundle: GlobalMapBundle,
    reencode_period: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    model.eval()
    model_device = next(model.parameters()).device
    x0 = x0.to(model_device)
    with torch.no_grad():
        latent = model.encode(x0).detach().cpu().numpy().astype(np.float32, copy=False)

    global_k = global_bundle.global_map.detach().cpu().numpy().astype(np.float32, copy=False)
    batch, latent_dim = latent.shape
    obs_dim = int(x0.shape[-1])
    predictions = np.full((int(max_horizon), batch, obs_dim), np.nan, dtype=np.float32)
    used_global = np.zeros((int(max_horizon), batch), dtype=bool)
    route_labels = np.empty((int(max_horizon), batch), dtype=object)
    route_labels[:] = "stage2_global_calibrated"
    valid = np.ones(batch, dtype=bool)

    for step in range(int(max_horizon)):
        if not bool(np.any(valid)):
            break
        next_latent = np.full((batch, latent_dim), np.nan, dtype=np.float32)
        valid_indices = np.flatnonzero(valid)
        current_latent = latent[valid_indices]
        next_valid = current_latent @ global_k
        next_latent[valid_indices] = next_valid
        used_global[step, valid_indices] = True

        with torch.no_grad():
            pred_state = model.decode(
                torch.from_numpy(next_latent).to(device=model_device, dtype=x0.dtype)
            ).detach().cpu().numpy()
        predictions[step] = pred_state.astype(np.float32, copy=False)

        finite_mask = np.logical_and(
            np.all(np.isfinite(next_latent), axis=1),
            np.all(np.isfinite(pred_state), axis=1),
        )
        latent_for_next = next_latent
        if int(reencode_period) > 0 and (step + 1) % int(reencode_period) == 0:
            refresh_indices = np.flatnonzero(np.logical_and(valid, finite_mask))
            refreshed_latent = np.full_like(next_latent, np.nan)
            if refresh_indices.size:
                with torch.no_grad():
                    refreshed = model.encode(
                        torch.from_numpy(pred_state[refresh_indices]).to(model_device, dtype=x0.dtype)
                    ).detach().cpu().numpy().astype(np.float32, copy=False)
                refreshed_latent[refresh_indices] = refreshed
            finite_mask = np.logical_and(finite_mask, np.all(np.isfinite(refreshed_latent), axis=1))
            latent_for_next = refreshed_latent
        valid = np.logical_and(valid, finite_mask)
        latent = latent_for_next

    return predictions, SRF._summarize_route_metrics(predictions, used_global, route_labels)


def _coerce_train_horizon(cfg, requested: int) -> int:
    if int(requested) > 0:
        return int(requested)
    try:
        horizon = int(cfg.TRAIN.SEQUENCE_LENGTH)
    except Exception:
        horizon = 8
    return max(1, horizon)


def _save_local_artifacts(
    run_output_dir: Path,
    *,
    bundle: LocalMapBundle,
    route_codebook: Dict[str, object],
    metadata: Dict[str, object],
    save_weights: bool,
) -> None:
    run_output_dir.mkdir(parents=True, exist_ok=True)
    if save_weights:
        torch.save(
            {
                "family_ids": list(bundle.family_ids),
                "centers": bundle.centers.detach().cpu(),
                "local_maps": bundle.local_maps.detach().cpu(),
                "global_k": bundle.global_k.detach().cpu(),
                "metadata": metadata,
            },
            run_output_dir / "local_maps.pt",
        )
    codebook_json = {
        "family_counts": {str(key): int(value) for key, value in route_codebook["family_counts"].items()},
        "fitted_family_ids": [str(item) for item in route_codebook["fitted_family_ids"]],
        "family_prototypes": {
            str(key): np.asarray(value, dtype=bool).astype(int).tolist()
            for key, value in route_codebook["family_prototypes"].items()
        },
    }
    _atomic_write_text(run_output_dir / "route_codebook.json", json.dumps(codebook_json, indent=2))
    _atomic_write_text(run_output_dir / "stage2_config.json", json.dumps(metadata, indent=2))


def _save_global_artifacts(
    run_output_dir: Path,
    *,
    bundle: GlobalMapBundle,
    route_codebook: Dict[str, object],
    metadata: Dict[str, object],
    save_weights: bool,
) -> None:
    run_output_dir.mkdir(parents=True, exist_ok=True)
    if save_weights:
        torch.save(
            {
                "global_map": bundle.global_map.detach().cpu(),
                "initial_global_k": bundle.initial_global_k.detach().cpu(),
                "metadata": metadata,
            },
            run_output_dir / "global_map.pt",
        )
    codebook_json = {
        "family_counts": {str(key): int(value) for key, value in route_codebook["family_counts"].items()},
        "fitted_family_ids": [str(item) for item in route_codebook["fitted_family_ids"]],
        "family_prototypes": {
            str(key): np.asarray(value, dtype=bool).astype(int).tolist()
            for key, value in route_codebook["family_prototypes"].items()
        },
    }
    _atomic_write_text(run_output_dir / "route_balancing_codebook.json", json.dumps(codebook_json, indent=2))
    _atomic_write_text(run_output_dir / "stage2_config.json", json.dumps(metadata, indent=2))


def _evaluate_one_setting(
    spec,
    *,
    args: argparse.Namespace,
    scheme: str,
    support_value: float,
    reencode_period: int,
    route_freeze_mode: str,
    output_dir: Path,
) -> List[Dict[str, object]]:
    checkpoint_path = Path(spec.run_dir) / "checkpoint.pt"
    cfg, env, model = REDUCER._load_checkpoint_model(checkpoint_path, spec.system_key, args.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model_device = next(model.parameters()).device
    global_k = model.kmatrix().detach().cpu().numpy().astype(np.float32, copy=False)
    train_horizon = _coerce_train_horizon(cfg, args.train_horizon)

    fit_trajectories = REDUCER._generate_observation_trajectories(
        env,
        num_trajectories=int(args.fit_num_trajectories),
        trajectory_length=int(args.fit_trajectory_length),
        eval_seed=int(args.fit_eval_seed),
    )
    if args.label_mode == "none":
        centers = None
        label_source = "none"
    else:
        _labels, centers, label_source = OPSEL._label_sequences_for_mode(
            env,
            fit_trajectories,
            system_key=spec.system_key,
            endpoint_rollout_steps=int(args.endpoint_rollout_steps),
            label_mode=args.label_mode,
        )
    fit_latents = REDUCER._encode_trajectories(model, fit_trajectories, args.device)
    route_codebook = _build_route_codebook(
        fit_latents,
        scheme=scheme,
        value=support_value,
        min_operator_transitions=int(args.min_operator_transitions),
        family_jaccard_threshold=float(args.family_jaccard_threshold),
    )
    fitted_family_ids = route_codebook["fitted_family_ids"]
    if not fitted_family_ids:
        raise RuntimeError("No support families reached the minimum transition threshold")

    if args.stage2_map_mode == "family_local_centered":
        bundle = LocalMapBundle(
            family_ids=fitted_family_ids,
            centers=route_codebook["centers"],
            global_k=global_k,
            device=args.device,
        ).to(model_device)
        rollout_mode = "family_local_centered"
        local_map_source = "stage2_rollout_trained"
    elif args.stage2_map_mode == "global_dense_calibrated":
        bundle = GlobalMapBundle(
            family_ids=fitted_family_ids,
            global_k=global_k,
            device=args.device,
        ).to(model_device)
        rollout_mode = "global_dense_calibrated"
        local_map_source = "stage2_global_rollout_trained"
    else:
        raise ValueError(f"Unsupported stage2_map_mode={args.stage2_map_mode}")

    train_pool = REDUCER._generate_observation_trajectories(
        env,
        num_trajectories=int(args.train_pool_trajectories),
        trajectory_length=train_horizon + 1,
        eval_seed=int(args.train_pool_seed) + int(spec.seed),
    ).float()
    with torch.no_grad():
        z0 = model.encode(train_pool[:, 0, :].to(model_device)).detach().cpu().numpy().astype(np.float32, copy=False)
    initial_routes = _route_indices_np(
        z0,
        scheme=scheme,
        value=support_value,
        family_jaccard_threshold=float(args.family_jaccard_threshold),
        support_key_to_family=route_codebook["support_key_to_family"],
        family_prototypes=route_codebook["family_prototypes"],
        family_to_index=bundle.family_to_index,
        family_cache={},
    )
    route_buckets = _balanced_indices_by_route(initial_routes)
    if not route_buckets:
        raise RuntimeError("Training pool has no examples assigned to fitted support families")

    slug_parts: List[object] = [
        spec.root_label,
        spec.system_key,
        f"seed{spec.seed}",
        f"p{int(reencode_period)}",
        route_freeze_mode,
    ]
    if args.stage2_map_mode != "family_local_centered":
        slug_parts.append(args.stage2_map_mode)
    run_slug = _make_slug(*slug_parts)
    run_output_dir = output_dir / "stage2_runs" / run_slug
    metrics_path = run_output_dir / "metrics_history.jsonl" if args.save_metrics_history else None
    train_checkpoint_path = run_output_dir / "train_checkpoint.pt" if args.save_train_checkpoint else None
    prior_checkpoint_path = _find_prior_train_checkpoint(
        SRF._parse_csv_strings(args.resume_from_output_dirs),
        run_slug,
    ) if not args.no_resume else None
    run_output_dir.mkdir(parents=True, exist_ok=True)
    if (
        metrics_path is not None
        and metrics_path.exists()
        and (
            args.no_resume
            or train_checkpoint_path is None
            or not train_checkpoint_path.exists()
        )
    ):
        metrics_path.unlink()

    print(
        f"Training stage2 {args.stage2_map_mode} root={spec.root_label} system={spec.system_key} "
        f"seed={spec.seed} period={reencode_period} freeze={route_freeze_mode} "
        f"families_for_balancing={len(fitted_family_ids)} horizon={train_horizon}",
        flush=True,
    )
    final_train_metrics = _train_one_local_bundle(
        model,
        bundle,
        train_pool,
        initial_routes,
        scheme=scheme,
        support_value=support_value,
        family_jaccard_threshold=float(args.family_jaccard_threshold),
        support_key_to_family=route_codebook["support_key_to_family"],
        family_prototypes=route_codebook["family_prototypes"],
        reencode_period=int(reencode_period),
        route_freeze_mode=route_freeze_mode,
        train_steps=int(args.train_steps),
        train_batch_size=int(args.train_batch_size),
        lr=float(args.lr),
        seed=int(args.train_pool_seed) + int(spec.seed) + 1009 * int(reencode_period),
        progress_every_steps=int(args.progress_every_steps),
        metrics_path=metrics_path,
        checkpoint_path=train_checkpoint_path,
        initial_checkpoint_path=prior_checkpoint_path,
        resume=not args.no_resume,
        max_runtime_seconds=int(args.max_runtime_seconds),
    )

    max_horizon = max(SRF._parse_horizons(args.horizons))
    forecast_trajectories = REDUCER._generate_observation_trajectories(
        env,
        num_trajectories=int(args.forecast_num_trajectories),
        trajectory_length=max_horizon + 1,
        eval_seed=int(args.forecast_eval_seed),
    )
    initial_states = forecast_trajectories[:, 0, :]
    true_future = (
        forecast_trajectories[:, 1 : max_horizon + 1, :]
        .permute(1, 0, 2)
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    depth_masks = SRF._all_depth_mask(initial_states) if args.label_mode == "none" else SRF._initial_depth_masks(initial_states, centers)
    if args.stage2_map_mode == "family_local_centered":
        predictions, route_metrics = _rollout_stage2_local(
            model,
            initial_states,
            max_horizon=max_horizon,
            device=args.device,
            local_bundle=bundle,
            scheme=scheme,
            support_value=support_value,
            family_jaccard_threshold=float(args.family_jaccard_threshold),
            support_key_to_family=route_codebook["support_key_to_family"],
            family_prototypes=route_codebook["family_prototypes"],
            reencode_period=int(reencode_period),
            route_freeze_mode=route_freeze_mode,
        )
    else:
        predictions, route_metrics = _rollout_stage2_global(
            model,
            initial_states,
            max_horizon=max_horizon,
            global_bundle=bundle,
            reencode_period=int(reencode_period),
        )

    metadata = {
        "source_run_dir": spec.run_dir,
        "root_label": spec.root_label,
        "system_key": spec.system_key,
        "seed": int(spec.seed),
        "support_definition": SRF._stringify_support_definition(scheme, support_value),
        "stage2_map_mode": args.stage2_map_mode,
        "reencode_period": int(reencode_period),
        "route_freeze_mode": route_freeze_mode,
        "train_steps": int(args.train_steps),
        "train_batch_size": int(args.train_batch_size),
        "train_pool_trajectories": int(args.train_pool_trajectories),
        "train_horizon": int(train_horizon),
        "lr": float(args.lr),
        "min_operator_transitions": int(args.min_operator_transitions),
        "family_jaccard_threshold": float(args.family_jaccard_threshold),
        "resume_from_output_dirs": SRF._parse_csv_strings(args.resume_from_output_dirs),
        "resume_from_train_checkpoint": str(prior_checkpoint_path) if prior_checkpoint_path else "",
        "save_metrics_history": bool(args.save_metrics_history),
        "save_train_checkpoint": bool(args.save_train_checkpoint),
        "save_stage2_artifacts": bool(args.save_stage2_artifacts),
        "fit_family_class_count_total": int(len(route_codebook["family_counts"])),
        "fit_family_class_count_fit": int(len(fitted_family_ids)),
        "final_train_metrics": final_train_metrics,
    }
    if args.stage2_map_mode == "family_local_centered":
        _save_local_artifacts(
            run_output_dir,
            bundle=bundle,
            route_codebook=route_codebook,
            metadata=metadata,
            save_weights=bool(args.save_stage2_artifacts),
        )
    else:
        _save_global_artifacts(
            run_output_dir,
            bundle=bundle,
            route_codebook=route_codebook,
            metadata=metadata,
            save_weights=bool(args.save_stage2_artifacts),
        )

    rows: List[Dict[str, object]] = []
    for depth_stratum, subset_mask in depth_masks.items():
        route_summary = SRF._compute_subset_route_summary(route_metrics, subset_mask)
        horizon_stats = SRF._compute_horizon_stats(
            predictions,
            true_future,
            SRF._parse_horizons(args.horizons),
            subset_mask,
        )
        rows.append(
            {
                "root_label": spec.root_label,
                "system_key": spec.system_key,
                "system_name": spec.system_name,
                "seed": int(spec.seed),
                "run_dir": spec.run_dir,
                "stage2_artifact_dir": str(run_output_dir),
                "support_definition": SRF._stringify_support_definition(scheme, support_value),
                "stage2_map_mode": args.stage2_map_mode,
                "depth_stratum": depth_stratum,
                "rollout_mode": rollout_mode,
                "local_map_source": local_map_source,
                "route_freeze_mode": route_freeze_mode,
                "reencode_period": int(reencode_period),
                "label_mode": args.label_mode,
                "label_source": label_source,
                "fit_num_trajectories": float(args.fit_num_trajectories),
                "fit_trajectory_length": float(args.fit_trajectory_length),
                "fit_eval_seed": float(args.fit_eval_seed),
                "forecast_num_trajectories": float(args.forecast_num_trajectories),
                "forecast_eval_seed": float(args.forecast_eval_seed),
                "train_steps": float(args.train_steps),
                "train_horizon": float(train_horizon),
                "train_pool_trajectories": float(args.train_pool_trajectories),
                "fit_family_class_count_total": float(len(route_codebook["family_counts"])),
                "fit_family_class_count_fit": float(len(fitted_family_ids)),
                **route_summary,
                **horizon_stats,
                "skip_reason": "",
            }
        )
    return rows


def _write_summary(path: Path, rows: Sequence[Dict[str, object]], horizons: Sequence[int]) -> None:
    lines = [
        "# Stage-2 Transition-Map Summary",
        "",
        "Rows are stage-2 trained maps under the frozen encoder/decoder rollout objective.",
        "",
        "| root | period | freeze mode | systems | mean coverage | mean fallback | "
        + " | ".join([f"mean H{h}" for h in horizons])
        + " |",
        "|---|---:|---|---:|---:|---:|" + "".join(["---:|" for _ in horizons]),
    ]
    grouped: Dict[Tuple[str, str, str, int, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("depth_stratum")) != "all":
            continue
        grouped[
            (
                str(row["root_label"]),
                str(row.get("stage2_map_mode", "family_local_centered")),
                str(row.get("local_map_source", "")),
                int(row["reencode_period"]),
                str(row["route_freeze_mode"]),
            )
        ].append(row)
    for (root, stage2_mode, source, period, mode), group_rows in sorted(grouped.items()):
        pieces = [
            f"| `{root}/{stage2_mode}/{source}` | {period} | `{mode}` | "
            f"{len({row['system_key'] for row in group_rows})} | "
            f"{_mean(row.get('route_coverage_fraction') for row in group_rows):.4g} | "
            f"{_mean(row.get('fallback_fraction') for row in group_rows):.4g} |"
        ]
        for horizon in horizons:
            pieces.append(f" {_mean(row.get(f'h{horizon}_mean') for row in group_rows):.4g} |")
        lines.append("".join(pieces))
    _atomic_write_text(path, "\n".join(lines) + "\n")


def _mean(values: Iterable[object]) -> float:
    clean = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            clean.append(numeric)
    return float(np.mean(clean)) if clean else float("nan")


def main() -> None:
    args = _parse_args()
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_stop)

    rows_csvs = SRF._parse_csv_strings(args.rows_csvs)
    root_labels = SRF._parse_csv_strings(args.root_labels)
    systems = SRF._parse_csv_strings(args.systems)
    seeds = SRF._parse_csv_ints(args.seeds)
    scheme, support_value = SRF._parse_support_definitions(args.support_definition)[0]
    reencode_periods = SRF._parse_reencode_periods(args.reencode_periods)
    route_freeze_modes = _parse_route_freeze_modes(args.route_freeze_modes)
    horizons = SRF._parse_horizons(args.horizons)
    specs = OPSEL._load_latest_specs(
        [Path(item) for item in rows_csvs],
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    completed: set[Tuple[str, str, int, str, int, str, str]] = set()
    if not args.no_resume:
        rows, completed = _load_existing_rows(output_dir / "self_routed_forecasting_rows.csv")

    failures: List[Dict[str, object]] = []
    start_time = time.time()
    max_runtime_seconds = max(0, int(args.max_runtime_seconds))
    total_runs = len(specs) * len(reencode_periods) * len(route_freeze_modes)
    run_index = 0
    for spec in specs:
        for reencode_period in reencode_periods:
            for route_freeze_mode in route_freeze_modes:
                run_index += 1
                key = (
                    spec.root_label,
                    spec.system_key,
                    int(spec.seed),
                    spec.run_dir,
                    int(reencode_period),
                    route_freeze_mode,
                    args.stage2_map_mode,
                )
                if key in completed:
                    continue
                try:
                    new_rows = _evaluate_one_setting(
                        spec,
                        args=args,
                        scheme=scheme,
                        support_value=support_value,
                        reencode_period=int(reencode_period),
                        route_freeze_mode=route_freeze_mode,
                        output_dir=output_dir,
                    )
                    rows.extend(new_rows)
                    completed.add(key)
                    print(f"[{run_index}/{total_runs}] completed {key}", flush=True)
                except Exception as exc:
                    failures.append(
                        {
                            "root_label": spec.root_label,
                            "system_key": spec.system_key,
                            "seed": int(spec.seed),
                            "run_dir": spec.run_dir,
                            "reencode_period": int(reencode_period),
                            "route_freeze_mode": route_freeze_mode,
                            "stage2_map_mode": args.stage2_map_mode,
                            "error": repr(exc),
                        }
                    )
                    print(f"[{run_index}/{total_runs}] error {key}: {exc}", flush=True)

                if int(args.flush_every_runs) > 0:
                    _write_csv(output_dir / "self_routed_forecasting_rows.csv", rows)
                    _write_summary(output_dir / "self_routed_forecasting_summary.md", rows, horizons)
                    _atomic_write_text(output_dir / "failures.json", json.dumps(failures, indent=2))

                elapsed = time.time() - start_time
                if STOP_REQUESTED or (max_runtime_seconds > 0 and elapsed >= max_runtime_seconds):
                    _write_csv(output_dir / "self_routed_forecasting_rows.csv", rows)
                    _write_summary(output_dir / "self_routed_forecasting_summary.md", rows, horizons)
                    _atomic_write_text(output_dir / "failures.json", json.dumps(failures, indent=2))
                    _atomic_write_text(
                        output_dir / "manifest.json",
                        json.dumps(
                            {
                                "status": "stopped",
                                "completed_runs": len(completed),
                                "total_runs": total_runs,
                                "num_rows": len(rows),
                                "num_failures": len(failures),
                            },
                            indent=2,
                        ),
                    )
                    raise SystemExit(1)

    _write_csv(output_dir / "self_routed_forecasting_rows.csv", rows)
    _write_summary(output_dir / "self_routed_forecasting_summary.md", rows, horizons)
    _atomic_write_text(output_dir / "failures.json", json.dumps(failures, indent=2))
    _atomic_write_text(
        output_dir / "manifest.json",
        json.dumps(
            {
                "rows_csvs": rows_csvs,
                "root_labels": root_labels,
                "systems": systems,
                "seeds": seeds,
                "support_definition": SRF._stringify_support_definition(scheme, support_value),
                "reencode_periods": reencode_periods,
                "route_freeze_modes": route_freeze_modes,
                "stage2_map_mode": args.stage2_map_mode,
                "train_steps": int(args.train_steps),
                "train_batch_size": int(args.train_batch_size),
                "train_pool_trajectories": int(args.train_pool_trajectories),
                "save_metrics_history": bool(args.save_metrics_history),
                "save_train_checkpoint": bool(args.save_train_checkpoint),
                "save_stage2_artifacts": bool(args.save_stage2_artifacts),
                "status": "complete" if not failures else "complete_with_failures",
                "completed_runs": len(completed),
                "total_runs": total_runs,
                "num_rows": len(rows),
                "num_failures": len(failures),
            },
            indent=2,
        ),
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
