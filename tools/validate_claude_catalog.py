#!/usr/bin/env python
"""Validate and visualize transition-rich dynamical systems from the Claude catalog.

For each system, this script:
1. Generates N trajectories from diverse initial conditions
2. Identifies endpoint basins by long-rollout clustering
3. Computes acceptance-gate metrics (basin count, occupancy, crossing fraction)
4. Generates phase portraits with basin coloring
5. Reports pass/fail for each acceptance gate

Usage:
    uv run python tools/validate_claude_catalog.py [--system NAME] [--all] [--n_traj 500]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

import torch
import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_trajectories(system, n_traj=500, traj_len=1000, seed=42):
    """Generate trajectories from diverse ICs."""
    rng = torch.Generator().manual_seed(seed)
    trajectories = []
    init_states = []
    for _ in range(n_traj):
        ic = system.reset(rng)
        traj = system.generate_trajectory(ic, traj_len)
        trajectories.append(traj)
        init_states.append(ic)
    return torch.stack(trajectories), torch.stack(init_states)


def identify_basins_long_rollout(system, final_states, extra_steps=5000):
    """Run long rollouts from final states to identify endpoint basins."""
    endpoints = []
    for state in final_states:
        s = state.clone()
        for _ in range(extra_steps):
            s_new = system.step(s)
            s_new = torch.clamp(s_new, -1e6, 1e6)
            if torch.any(torch.isnan(s_new)) or torch.any(torch.isinf(s_new)):
                break
            s = s_new
        endpoints.append(s)
    return torch.stack(endpoints)


def cluster_endpoints(endpoints, method="dbscan", min_samples=5, eps_auto=True):
    """Cluster endpoint states to identify basins.

    Returns:
        labels: integer basin label per trajectory (-1 for noise)
        centers: cluster centers [n_clusters, dim]
        n_clusters: number of clusters found
    """
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    X = endpoints.numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "dbscan":
        # Auto-tune eps
        if eps_auto:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=min_samples)
            nn.fit(X_scaled)
            distances, _ = nn.kneighbors(X_scaled)
            sorted_dists = np.sort(distances[:, -1])
            # Use the knee point (heuristic: 90th percentile of k-NN distances)
            eps = float(sorted_dists[int(0.3 * len(sorted_dists))])
            eps = max(eps, 0.05)  # floor
        else:
            eps = 0.5

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)

        # If DBSCAN gives too many/few clusters, fall back to k-means with silhouette
        unique_labels = set(labels) - {-1}
        n_clusters = len(unique_labels)
        if n_clusters < 2 or n_clusters > 15:
            return cluster_endpoints(
                endpoints, method="kmeans_auto", min_samples=min_samples
            )
    elif method == "kmeans_auto":
        # Try k=2..12, pick best silhouette
        best_score = -1
        best_k = 3
        best_labels = None
        for k in range(2, 13):
            if k >= len(X_scaled):
                break
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            lab = km.fit_predict(X_scaled)
            if len(set(lab)) < 2:
                continue
            score = silhouette_score(X_scaled, lab)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = lab
        labels = best_labels if best_labels is not None else np.zeros(len(X), dtype=int)
        n_clusters = best_k
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute centers in original space
    unique_labels = sorted(set(labels) - {-1})
    n_clusters = len(unique_labels)
    centers = np.zeros((n_clusters, X.shape[1]))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        centers[i] = X[mask].mean(axis=0)

    # Relabel: 0, 1, 2, ... (excluding noise as -1)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    label_map[-1] = -1
    labels = np.array([label_map[l] for l in labels])

    return labels, torch.tensor(centers, dtype=torch.float64), n_clusters


def compute_crossing_fraction(trajectories, basin_labels, centers, settle_window=50):
    """Compute crossing fraction per endpoint basin.

    A trajectory is 'crossing' if it visits at least one region (nearest-center)
    different from its endpoint basin at any point before the settling window.

    Returns:
        overall_crossing: fraction of all trajectories that cross
        per_basin_crossing: dict mapping basin_id -> crossing fraction
    """
    n_traj = trajectories.shape[0]
    traj_len = trajectories.shape[1]

    crossing_count = Counter()
    basin_count = Counter()

    for i in range(n_traj):
        label = basin_labels[i]
        if label == -1:
            continue  # skip noise

        basin_count[label] += 1
        traj = trajectories[i]  # [T+1, dim]

        # Assign each point to nearest center
        # traj: [T+1, dim], centers: [K, dim]
        dists = torch.cdist(
            traj[: traj_len - settle_window].unsqueeze(0).float(),
            centers.unsqueeze(0).float(),
        ).squeeze(
            0
        )  # [T-settle, K]
        assignments = dists.argmin(dim=1).numpy()  # [T-settle]

        # Check if any assignment differs from endpoint basin
        if np.any(assignments != label):
            crossing_count[label] += 1

    overall_crossing = sum(crossing_count.values()) / max(
        sum(basin_count.values()), 1
    )
    per_basin_crossing = {}
    for b in sorted(basin_count.keys()):
        if basin_count[b] > 0:
            per_basin_crossing[b] = crossing_count[b] / basin_count[b]
        else:
            per_basin_crossing[b] = 0.0

    return overall_crossing, per_basin_crossing


def check_determinism(system, n_checks=20, traj_len=100, seed=42):
    """Verify system is deterministic."""
    rng1 = torch.Generator().manual_seed(seed)
    rng2 = torch.Generator().manual_seed(seed)
    for _ in range(n_checks):
        ic1 = system.reset(rng1)
        ic2 = system.reset(rng2)
        if not torch.allclose(ic1, ic2, atol=1e-12):
            return False
        traj1 = system.generate_trajectory(ic1, traj_len)
        traj2 = system.generate_trajectory(ic2, traj_len)
        if not torch.allclose(traj1, traj2, atol=1e-12):
            return False
    return True


def check_acceptance_gates(
    system,
    n_traj=500,
    traj_len=1000,
    extra_steps=5000,
    settle_window=50,
    seed=42,
    verbose=True,
):
    """Run all acceptance gates for a system.

    Returns dict with results and pass/fail.
    """
    results = {"name": system.name, "category": system.category, "dim": system.dim}

    # Gate 1: Determinism
    t0 = time.time()
    is_det = check_determinism(system, seed=seed)
    results["determinism"] = is_det
    results["determinism_pass"] = is_det
    if verbose:
        print(f"  Determinism: {'PASS' if is_det else 'FAIL'} ({time.time()-t0:.1f}s)")

    # Generate trajectories
    t0 = time.time()
    trajectories, init_states = generate_trajectories(
        system, n_traj=n_traj, traj_len=traj_len, seed=seed
    )
    if verbose:
        print(f"  Generated {n_traj} trajectories ({time.time()-t0:.1f}s)")

    # Check for NaN/inf
    nan_count = torch.isnan(trajectories).any(dim=-1).any(dim=-1).sum().item()
    inf_count = torch.isinf(trajectories).any(dim=-1).any(dim=-1).sum().item()
    results["nan_trajectories"] = nan_count
    results["inf_trajectories"] = inf_count
    if nan_count > 0 or inf_count > 0:
        if verbose:
            print(f"  WARNING: {nan_count} NaN, {inf_count} inf trajectories")

    # Identify basins
    t0 = time.time()
    final_states = trajectories[:, -1, :]
    endpoints = identify_basins_long_rollout(system, final_states, extra_steps=extra_steps)
    basin_labels, centers, n_clusters = cluster_endpoints(endpoints)
    results["n_basins"] = n_clusters
    results["basin_count_pass"] = 3 <= n_clusters <= 10
    if verbose:
        print(
            f"  Basin count: {n_clusters} {'PASS' if results['basin_count_pass'] else 'FAIL'} ({time.time()-t0:.1f}s)"
        )

    # Basin occupancy
    label_counts = Counter(basin_labels[basin_labels >= 0])
    total_valid = sum(label_counts.values())
    min_occ_threshold = max(0.05, 0.5 / max(n_clusters, 1))
    occupancies = {}
    min_occ = 1.0
    for b in sorted(label_counts.keys()):
        occ = label_counts[b] / max(total_valid, 1)
        occupancies[b] = occ
        min_occ = min(min_occ, occ)
    results["occupancies"] = occupancies
    results["min_occupancy"] = min_occ
    results["occupancy_threshold"] = min_occ_threshold
    results["occupancy_pass"] = min_occ >= min_occ_threshold
    if verbose:
        print(
            f"  Min occupancy: {min_occ:.3f} (threshold {min_occ_threshold:.3f}) "
            f"{'PASS' if results['occupancy_pass'] else 'FAIL'}"
        )

    # Crossing fraction
    t0 = time.time()
    overall_crossing, per_basin_crossing = compute_crossing_fraction(
        trajectories, basin_labels, centers, settle_window=settle_window
    )
    results["overall_crossing"] = overall_crossing
    results["per_basin_crossing"] = per_basin_crossing
    crossing_pass = True
    min_crossing = 1.0
    max_crossing = 0.0
    for b, cf in per_basin_crossing.items():
        min_crossing = min(min_crossing, cf)
        max_crossing = max(max_crossing, cf)
        if cf < 0.30 or cf > 0.70:
            crossing_pass = False
    results["crossing_pass_strict"] = crossing_pass
    # Relaxed: allow if overall is high and no basin below 0.25
    crossing_pass_relaxed = (
        overall_crossing >= 0.30 and min_crossing >= 0.25 and max_crossing <= 0.75
    )
    results["crossing_pass_relaxed"] = crossing_pass or crossing_pass_relaxed
    if verbose:
        print(
            f"  Crossing: overall={overall_crossing:.3f}, "
            f"per-basin={dict((b, f'{v:.3f}') for b,v in per_basin_crossing.items())} "
            f"{'PASS' if results['crossing_pass_relaxed'] else 'FAIL'} ({time.time()-t0:.1f}s)"
        )

    # Overall pass
    results["all_pass"] = (
        results["determinism_pass"]
        and results["basin_count_pass"]
        and results["occupancy_pass"]
        and results["crossing_pass_relaxed"]
        and nan_count == 0
    )
    if verbose:
        print(f"  OVERALL: {'PASS' if results['all_pass'] else 'FAIL'}")

    # Store trajectories and labels for visualization
    results["_trajectories"] = trajectories
    results["_basin_labels"] = basin_labels
    results["_centers"] = centers
    results["_init_states"] = init_states

    return results


def plot_phase_portrait(system, results, save_path=None, max_traj=200):
    """Generate a publication-quality phase portrait with basin coloring."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    trajectories = results["_trajectories"]
    basin_labels = results["_basin_labels"]
    centers = results["_centers"]
    n_basins = results["n_basins"]
    dim = system.dim

    # Use first 2 dims for plotting
    d0, d1 = 0, min(1, dim - 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Color map
    cmap = plt.cm.get_cmap("tab10", max(n_basins, 1))
    colors = [cmap(i) for i in range(n_basins)]

    # Panel 1: All trajectories colored by endpoint basin
    ax = axes[0]
    n_plot = min(max_traj, trajectories.shape[0])
    indices = np.random.RandomState(42).choice(
        trajectories.shape[0], n_plot, replace=False
    )
    for idx in indices:
        label = basin_labels[idx]
        if label < 0:
            c = "gray"
            alpha = 0.1
        else:
            c = colors[label]
            alpha = 0.3
        traj = trajectories[idx].numpy()
        ax.plot(traj[:, d0], traj[:, d1], color=c, alpha=alpha, linewidth=0.5)

    # Plot basin centers
    for i in range(n_basins):
        ax.plot(
            centers[i, d0],
            centers[i, d1],
            "X",
            color=colors[i],
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=10,
        )

    ax.set_title(f"Trajectories (N={n_plot})", fontsize=11)
    ax.set_xlabel(f"$x_{d0+1}$")
    ax.set_ylabel(f"$x_{d1+1}$")
    ax.grid(True, alpha=0.3)

    # Panel 2: Vector field + basin boundaries
    ax = axes[1]
    ic_box = system.ic_box
    x_range = np.linspace(ic_box[d0][0], ic_box[d0][1], 30)
    y_range = np.linspace(ic_box[d1][0], ic_box[d1][1], 30)
    X, Y = np.meshgrid(x_range, y_range)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            state = torch.zeros(dim, dtype=torch.float64)
            state[d0] = X[i, j]
            state[d1] = Y[i, j]
            try:
                dstate = system.dynamics(state)
                U[i, j] = dstate[d0].item()
                V[i, j] = dstate[d1].item()
            except Exception:
                U[i, j] = 0
                V[i, j] = 0

    speed = np.sqrt(U**2 + V**2)
    speed = np.clip(speed, 1e-10, None)
    ax.streamplot(
        x_range,
        y_range,
        U,
        V,
        color=np.log1p(speed),
        cmap="viridis",
        linewidth=0.8,
        density=1.5,
        arrowsize=0.8,
    )

    # Plot centers
    for i in range(n_basins):
        ax.plot(
            centers[i, d0],
            centers[i, d1],
            "X",
            color="red",
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=10,
        )

    ax.set_title("Vector field + stream", fontsize=11)
    ax.set_xlabel(f"$x_{d0+1}$")
    ax.set_ylabel(f"$x_{d1+1}$")
    ax.set_xlim(ic_box[d0])
    ax.set_ylim(ic_box[d1])
    ax.grid(True, alpha=0.3)

    # Panel 3: Basin classification map
    ax = axes[2]
    x_fine = np.linspace(ic_box[d0][0], ic_box[d0][1], 80)
    y_fine = np.linspace(ic_box[d1][0], ic_box[d1][1], 80)
    Xf, Yf = np.meshgrid(x_fine, y_fine)
    basin_map = np.full(Xf.shape, -1, dtype=int)

    # For each grid point, assign to nearest center
    grid_points = np.stack([Xf.ravel(), Yf.ravel()], axis=1)
    centers_np = centers[:, :2].numpy()
    for k in range(grid_points.shape[0]):
        dists = np.linalg.norm(grid_points[k] - centers_np, axis=1)
        basin_map.ravel()[k] = np.argmin(dists)

    # Use pcolormesh for Voronoi-like coloring
    cmap_discrete = ListedColormap(colors[:n_basins])
    ax.pcolormesh(Xf, Yf, basin_map, cmap=cmap_discrete, alpha=0.3, shading="auto")

    # Overlay some trajectories
    for idx in indices[:50]:
        label = basin_labels[idx]
        if label < 0:
            continue
        traj = trajectories[idx].numpy()
        ax.plot(traj[:, d0], traj[:, d1], color=colors[label], alpha=0.4, linewidth=0.3)
        # Mark start and end
        ax.plot(traj[0, d0], traj[0, d1], "o", color=colors[label], markersize=2, alpha=0.5)

    for i in range(n_basins):
        ax.plot(
            centers[i, d0],
            centers[i, d1],
            "X",
            color=colors[i],
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=10,
        )

    ax.set_title("Basin map (Voronoi) + trajectories", fontsize=11)
    ax.set_xlabel(f"$x_{d0+1}$")
    ax.set_ylabel(f"$x_{d1+1}$")
    ax.set_xlim(ic_box[d0])
    ax.set_ylim(ic_box[d1])
    ax.grid(True, alpha=0.3)

    # Overall title
    crossing = results["overall_crossing"]
    n_basins_found = results["n_basins"]
    pass_str = "PASS" if results["all_pass"] else "FAIL"
    fig.suptitle(
        f"{system.name} ({system.category}) — "
        f"{n_basins_found} basins, {crossing:.1%} crossing, {pass_str}",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    else:
        plt.savefig("/tmp/phase_portrait.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return "/tmp/phase_portrait.png"


def main():
    parser = argparse.ArgumentParser(description="Validate transition-rich systems")
    parser.add_argument("--system", type=str, default=None, help="Validate single system")
    parser.add_argument("--all", action="store_true", help="Validate all systems")
    parser.add_argument("--n_traj", type=int, default=500, help="Number of trajectories")
    parser.add_argument("--traj_len", type=int, default=1000, help="Trajectory length")
    parser.add_argument("--extra_steps", type=int, default=5000, help="Long-rollout steps")
    parser.add_argument("--plot", action="store_true", help="Generate phase portraits")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/claude_catalog_validation",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Import all system modules to populate registry
    try:
        import skae.claude_catalog.systems_gradient
    except ImportError as e:
        print(f"Warning: Could not import systems_gradient: {e}")
    try:
        import skae.claude_catalog.systems_bio_physical
    except ImportError as e:
        print(f"Warning: Could not import systems_bio_physical: {e}")
    try:
        import skae.claude_catalog.systems_creative
    except ImportError as e:
        print(f"Warning: Could not import systems_creative: {e}")
    try:
        import skae.claude_catalog.systems_novel
    except ImportError as e:
        print(f"Warning: Could not import systems_novel: {e}")
    try:
        import skae.claude_catalog.systems_tuned
    except ImportError as e:
        print(f"Warning: Could not import systems_tuned: {e}")

    from skae.claude_catalog.registry import CATALOG_REGISTRY, get_system, list_systems

    if not CATALOG_REGISTRY:
        print("ERROR: No systems registered. Check imports.")
        sys.exit(1)

    print(f"Registered systems: {len(CATALOG_REGISTRY)}")
    print(f"  Names: {list_systems()}")
    print()

    # Select systems to validate
    if args.system:
        systems_to_validate = [args.system]
    elif args.all:
        systems_to_validate = list_systems()
    else:
        systems_to_validate = list_systems()

    all_results = []
    for name in systems_to_validate:
        print(f"\n{'='*60}")
        print(f"Validating: {name}")
        print(f"{'='*60}")

        try:
            system = get_system(name)
            print(f"  Category: {system.category}, Dim: {system.dim}")
            print(f"  Description: {system.description}")
            print(f"  IC box: {system.ic_box}")
            print(f"  dt: {system.dt}")

            results = check_acceptance_gates(
                system,
                n_traj=args.n_traj,
                traj_len=args.traj_len,
                extra_steps=args.extra_steps,
                seed=args.seed,
            )

            if args.plot:
                plot_path = os.path.join(args.output_dir, "plots", f"{name}.png")
                plot_phase_portrait(system, results, save_path=plot_path)
                print(f"  Plot saved: {plot_path}")

            # Store serializable results
            serializable = {
                k: v
                for k, v in results.items()
                if not k.startswith("_")
            }
            # Convert numpy/torch types
            for k, v in serializable.items():
                if isinstance(v, (np.integer, np.floating)):
                    serializable[k] = float(v)
                elif isinstance(v, dict):
                    serializable[k] = {
                        str(kk): float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                        for kk, vv in v.items()
                    }
            all_results.append(serializable)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"name": name, "error": str(e), "all_pass": False})

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = [r for r in all_results if r.get("all_pass", False)]
    failed = [r for r in all_results if not r.get("all_pass", False)]
    print(f"  Passed: {len(passed)}/{len(all_results)}")
    print(f"  Failed: {len(failed)}/{len(all_results)}")

    if passed:
        print("\n  PASSED systems:")
        for r in passed:
            print(
                f"    {r['name']:30s} basins={r.get('n_basins','?'):>2} "
                f"crossing={r.get('overall_crossing',0):.1%}"
            )

    if failed:
        print("\n  FAILED systems:")
        for r in failed:
            reasons = []
            if r.get("error"):
                reasons.append(f"error: {r['error'][:60]}")
            else:
                if not r.get("determinism_pass", True):
                    reasons.append("non-deterministic")
                if not r.get("basin_count_pass", True):
                    reasons.append(f"basins={r.get('n_basins', '?')}")
                if not r.get("occupancy_pass", True):
                    reasons.append(f"min_occ={r.get('min_occupancy', 0):.3f}")
                if not r.get("crossing_pass_relaxed", True):
                    reasons.append(f"crossing={r.get('overall_crossing', 0):.1%}")
                if r.get("nan_trajectories", 0) > 0:
                    reasons.append(f"nan={r['nan_trajectories']}")
            print(f"    {r.get('name','?'):30s} — {', '.join(reasons)}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "validation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    main()
