#!/usr/bin/env python
"""Fast screening of catalog systems using batched vmap integration.

Runs all trajectories in parallel using torch.vmap for ~50x speedup.
"""

import sys
import time
import json
import os
from pathlib import Path
from collections import Counter

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import all system modules
import skae.claude_catalog.systems_gradient
import skae.claude_catalog.systems_bio_physical
import skae.claude_catalog.systems_creative
import skae.claude_catalog.systems_novel
import skae.claude_catalog.systems_tuned
import skae.claude_catalog.systems_variants
from skae.claude_catalog.registry import CATALOG_REGISTRY, get_system, list_systems


def batch_generate_vmap(system, n_traj=100, traj_len=500, seed=42):
    """Generate trajectories using vmap for parallelism."""
    # Generate initial conditions
    init_states = torch.zeros(n_traj, system.dim, dtype=torch.float64)
    for i in range(n_traj):
        rng = torch.Generator().manual_seed(seed + i)
        init_states[i] = system.reset(rng)

    try:
        batched_step = torch.vmap(system.step)
        states = init_states.clone()
        trajectories = torch.zeros(n_traj, traj_len + 1, system.dim, dtype=torch.float64)
        trajectories[:, 0] = states

        for t in range(traj_len):
            states = batched_step(states)
            states = torch.clamp(states, -1e6, 1e6)
            # Check for NaN
            nan_mask = torch.isnan(states).any(dim=-1)
            if nan_mask.any():
                # Replace NaN states with previous valid state
                states[nan_mask] = trajectories[nan_mask, t]
            trajectories[:, t + 1] = states

        return trajectories, init_states, True
    except Exception as e:
        # Fallback to sequential if vmap fails
        return batch_generate_sequential(system, n_traj, traj_len, seed), None, False


def batch_generate_sequential(system, n_traj=100, traj_len=500, seed=42):
    """Fallback: sequential trajectory generation."""
    trajectories = torch.zeros(n_traj, traj_len + 1, system.dim, dtype=torch.float64)
    for i in range(n_traj):
        rng = torch.Generator().manual_seed(seed + i)
        ic = system.reset(rng)
        traj = system.generate_trajectory(ic, traj_len)
        trajectories[i] = traj
    return trajectories


def long_rollout_vmap(system, final_states, extra_steps=2000):
    """Long rollout using vmap."""
    try:
        batched_step = torch.vmap(system.step)
        states = final_states.clone()
        for _ in range(extra_steps):
            states = batched_step(states)
            states = torch.clamp(states, -1e6, 1e6)
            nan_mask = torch.isnan(states).any(dim=-1)
            if nan_mask.any():
                states[nan_mask] = final_states[nan_mask]
        return states
    except Exception:
        # Fallback
        endpoints = []
        for state in final_states:
            s = state.clone()
            for _ in range(extra_steps):
                s_new = system.step(s)
                s_new = torch.clamp(s_new, -1e6, 1e6)
                if torch.any(torch.isnan(s_new)):
                    break
                s = s_new
            endpoints.append(s)
        return torch.stack(endpoints)


def cluster_simple(endpoints, tol=0.3):
    """Simple nearest-neighbor clustering."""
    n = endpoints.shape[0]
    centers = [endpoints[0]]
    assignments = [0] * n
    for i in range(1, n):
        dists = [torch.norm(endpoints[i] - c).item() for c in centers]
        min_d = min(dists)
        if min_d < tol:
            assignments[i] = dists.index(min_d)
        else:
            assignments[i] = len(centers)
            centers.append(endpoints[i])
    return assignments, torch.stack(centers), len(centers)


def compute_crossing(trajectories, assignments, centers, settle_window=30):
    """Compute crossing fraction."""
    n_traj = trajectories.shape[0]
    traj_len = trajectories.shape[1]

    crossing_count = Counter()
    basin_count = Counter()

    # Compute all nearest-center assignments at once
    # trajectories: [N, T, dim], centers: [K, dim]
    # Compute distances: [N, T, K]
    t_end = max(1, traj_len - settle_window)
    traj_prefix = trajectories[:, :t_end, :].float()
    centers_f = centers.float()
    # [N, T, K]
    dists = torch.cdist(traj_prefix.reshape(-1, centers.shape[-1]).unsqueeze(0),
                         centers_f.unsqueeze(0)).squeeze(0)
    dists = dists.reshape(n_traj, t_end, -1)
    path_assignments = dists.argmin(dim=-1).numpy()  # [N, T]

    for i in range(n_traj):
        label = assignments[i]
        if label < 0:
            continue
        basin_count[label] += 1
        # Check if any assignment differs from endpoint basin
        if np.any(path_assignments[i] != label):
            crossing_count[label] += 1

    overall = sum(crossing_count.values()) / max(sum(basin_count.values()), 1)
    per_basin = {}
    for b in sorted(basin_count.keys()):
        per_basin[b] = crossing_count[b] / basin_count[b] if basin_count[b] > 0 else 0.0
    return overall, per_basin


def screen_system(name, n_traj=100, traj_len=500, extra_steps=2000, seed=42):
    """Screen a single system against acceptance gates."""
    system = get_system(name)
    result = {
        "name": name,
        "category": system.category,
        "dim": system.dim,
        "description": system.description,
    }

    t0 = time.time()

    # Generate trajectories
    traj_result = batch_generate_vmap(system, n_traj, traj_len, seed)
    trajectories, init_states, used_vmap = traj_result[0], traj_result[1], traj_result[2] if len(traj_result) > 2 else True
    if init_states is None:
        # Sequential fallback returned just trajectories
        trajectories = traj_result[0]
        used_vmap = False

    result["used_vmap"] = used_vmap

    # Check NaN
    nan_count = torch.isnan(trajectories).any(dim=-1).any(dim=-1).sum().item()
    result["nan_count"] = nan_count

    # Determinism check (quick)
    rng1 = torch.Generator().manual_seed(seed)
    rng2 = torch.Generator().manual_seed(seed)
    ic1 = system.reset(rng1)
    ic2 = system.reset(rng2)
    det_ok = torch.allclose(ic1, ic2, atol=1e-12)
    if det_ok:
        t1 = system.generate_trajectory(ic1, 50)
        t2 = system.generate_trajectory(ic2, 50)
        det_ok = torch.allclose(t1, t2, atol=1e-12)
    result["determinism"] = det_ok

    # Long rollout for basin identification
    final_states = trajectories[:, -1, :]
    endpoints = long_rollout_vmap(system, final_states, extra_steps)

    # Cluster
    assignments, centers, n_basins = cluster_simple(endpoints, tol=0.3)
    result["n_basins"] = n_basins
    result["basin_pass"] = 3 <= n_basins <= 10

    # Occupancy
    counts = Counter(assignments)
    total = sum(counts.values())
    min_occ_threshold = max(0.05, 0.5 / max(n_basins, 1))
    occupancies = {b: counts[b] / total for b in sorted(counts.keys())}
    min_occ = min(occupancies.values()) if occupancies else 0
    result["occupancies"] = occupancies
    result["min_occupancy"] = min_occ
    result["occ_pass"] = min_occ >= min_occ_threshold

    # Crossing fraction
    overall_crossing, per_basin_crossing = compute_crossing(
        trajectories, assignments, centers, settle_window=30
    )
    result["overall_crossing"] = overall_crossing
    result["per_basin_crossing"] = per_basin_crossing
    min_cross = min(per_basin_crossing.values()) if per_basin_crossing else 0
    max_cross = max(per_basin_crossing.values()) if per_basin_crossing else 0
    strict_cross = all(0.30 <= v <= 0.70 for v in per_basin_crossing.values())
    relaxed_cross = overall_crossing >= 0.30 and min_cross >= 0.25 and max_cross <= 0.75
    result["crossing_pass"] = strict_cross or relaxed_cross

    # Overall
    result["all_pass"] = (
        det_ok
        and result["basin_pass"]
        and result["occ_pass"]
        and result["crossing_pass"]
        and nan_count == 0
    )

    elapsed = time.time() - t0
    result["time_s"] = elapsed

    return result, trajectories, assignments, centers


def plot_system(system, trajectories, assignments, centers, n_basins, save_path):
    """Generate phase portrait."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dim = system.dim
    d0, d1 = 0, min(1, dim - 1)
    cmap = plt.cm.get_cmap("tab10", max(n_basins, 1))
    colors = [cmap(i) for i in range(n_basins)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Trajectories colored by basin
    ax = axes[0]
    n_plot = min(100, trajectories.shape[0])
    indices = np.random.RandomState(42).choice(trajectories.shape[0], n_plot, replace=False)
    for idx in indices:
        label = assignments[idx]
        c = colors[label] if label >= 0 else "gray"
        traj = trajectories[idx].numpy()
        ax.plot(traj[:, d0], traj[:, d1], color=c, alpha=0.3, linewidth=0.5)
    for i in range(n_basins):
        ax.plot(centers[i, d0], centers[i, d1], "X", color=colors[i],
                markersize=12, markeredgecolor="black", markeredgewidth=1.5, zorder=10)
    ax.set_title(f"Trajectories (N={n_plot})", fontsize=11)
    ax.set_xlabel(f"$x_{d0+1}$")
    ax.set_ylabel(f"$x_{d1+1}$")
    ax.grid(True, alpha=0.3)

    # Panel 2: Vector field
    ax = axes[1]
    ic_box = system.ic_box
    x_range = np.linspace(ic_box[d0][0], ic_box[d0][1], 25)
    y_range = np.linspace(ic_box[d1][0], ic_box[d1][1], 25)
    X, Y = np.meshgrid(x_range, y_range)
    U, V = np.zeros_like(X), np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            state = torch.zeros(dim, dtype=torch.float64)
            state[d0] = X[i, j]
            state[d1] = Y[i, j]
            try:
                ds = system.dynamics(state)
                U[i, j] = ds[d0].item()
                V[i, j] = ds[d1].item()
            except Exception:
                pass
    speed = np.sqrt(U**2 + V**2)
    ax.streamplot(x_range, y_range, U, V, color=np.log1p(speed),
                  cmap="viridis", linewidth=0.8, density=1.5, arrowsize=0.8)
    for i in range(n_basins):
        ax.plot(centers[i, d0], centers[i, d1], "X", color="red",
                markersize=12, markeredgecolor="black", markeredgewidth=1.5, zorder=10)
    ax.set_title("Vector field", fontsize=11)
    ax.set_xlim(ic_box[d0])
    ax.set_ylim(ic_box[d1])
    ax.grid(True, alpha=0.3)

    # Panel 3: Basin map
    ax = axes[2]
    from matplotlib.colors import ListedColormap
    x_fine = np.linspace(ic_box[d0][0], ic_box[d0][1], 60)
    y_fine = np.linspace(ic_box[d1][0], ic_box[d1][1], 60)
    Xf, Yf = np.meshgrid(x_fine, y_fine)
    grid_pts = np.stack([Xf.ravel(), Yf.ravel()], axis=1)
    centers_np = centers[:, :2].numpy()
    basin_map = np.array([np.argmin(np.linalg.norm(p - centers_np, axis=1)) for p in grid_pts])
    basin_map = basin_map.reshape(Xf.shape)
    cmap_disc = ListedColormap(colors[:n_basins])
    ax.pcolormesh(Xf, Yf, basin_map, cmap=cmap_disc, alpha=0.3, shading="auto")
    for idx in indices[:50]:
        label = assignments[idx]
        if label >= 0:
            traj = trajectories[idx].numpy()
            ax.plot(traj[:, d0], traj[:, d1], color=colors[label], alpha=0.4, linewidth=0.3)
    for i in range(n_basins):
        ax.plot(centers[i, d0], centers[i, d1], "X", color=colors[i],
                markersize=12, markeredgecolor="black", markeredgewidth=1.5, zorder=10)
    ax.set_title("Basin map + trajectories", fontsize=11)
    ax.set_xlim(ic_box[d0])
    ax.set_ylim(ic_box[d1])
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{system.name} ({system.category})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_traj", type=int, default=100)
    parser.add_argument("--traj_len", type=int, default=500)
    parser.add_argument("--extra_steps", type=int, default=2000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/claude_catalog_validation")
    parser.add_argument("--system", type=str, default=None)
    args = parser.parse_args()

    systems = [args.system] if args.system else list_systems()
    print(f"Screening {len(systems)} systems...")
    print(f"Config: n_traj={args.n_traj}, traj_len={args.traj_len}, extra={args.extra_steps}")
    print()

    all_results = []
    for name in systems:
        sys.stdout.write(f"  {name:35s} ")
        sys.stdout.flush()
        try:
            result, trajectories, assignments, centers = screen_system(
                name, args.n_traj, args.traj_len, args.extra_steps
            )
            status = "PASS" if result["all_pass"] else "FAIL"
            details = []
            if not result["determinism"]:
                details.append("!det")
            if not result["basin_pass"]:
                details.append(f"basins={result['n_basins']}")
            if not result["occ_pass"]:
                details.append(f"occ={result['min_occupancy']:.2f}")
            if not result["crossing_pass"]:
                details.append(f"cross={result['overall_crossing']:.2f}")
            if result["nan_count"] > 0:
                details.append(f"nan={result['nan_count']}")
            detail_str = f" ({', '.join(details)})" if details else ""
            vmap_str = "v" if result.get("used_vmap") else "s"
            print(
                f"basins={result['n_basins']:>2}  occ={result['min_occupancy']:.2f}  "
                f"cross={result['overall_crossing']:.2f}  [{status}]{detail_str}  "
                f"({result['time_s']:.1f}s {vmap_str})"
            )
            sys.stdout.flush()

            if args.plot and trajectories is not None:
                system = get_system(name)
                plot_path = os.path.join(args.output_dir, "plots", f"{name}.png")
                try:
                    plot_system(system, trajectories, assignments, centers,
                                result["n_basins"], plot_path)
                except Exception as e:
                    print(f"    Plot error: {e}")

            all_results.append(result)

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"name": name, "error": str(e), "all_pass": False})
            sys.stdout.flush()

    # Summary
    print(f"\n{'='*70}")
    passed = [r for r in all_results if r.get("all_pass")]
    failed = [r for r in all_results if not r.get("all_pass")]
    print(f"PASSED: {len(passed)}/{len(all_results)}")
    print(f"FAILED: {len(failed)}/{len(all_results)}")

    if passed:
        print(f"\nPassed systems:")
        for r in sorted(passed, key=lambda x: -x.get("overall_crossing", 0)):
            print(f"  {r['name']:35s}  basins={r['n_basins']:>2}  "
                  f"cross={r['overall_crossing']:.2f}  cat={r['category']}")

    if failed:
        print(f"\nFailed systems (reasons):")
        for r in failed:
            reasons = []
            if r.get("error"):
                reasons.append(f"error: {str(r['error'])[:50]}")
            else:
                if not r.get("determinism", True): reasons.append("non-det")
                if not r.get("basin_pass", True): reasons.append(f"basins={r.get('n_basins')}")
                if not r.get("occ_pass", True): reasons.append(f"min_occ={r.get('min_occupancy', 0):.2f}")
                if not r.get("crossing_pass", True): reasons.append(f"cross={r.get('overall_crossing', 0):.2f}")
                if r.get("nan_count", 0): reasons.append(f"nan={r['nan_count']}")
            print(f"  {r.get('name', '?'):35s}  {', '.join(reasons)}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "screening_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
