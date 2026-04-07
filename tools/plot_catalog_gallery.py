#!/usr/bin/env python
"""Generate a publication-quality gallery of catalog systems.

Creates:
1. Individual 3-panel portraits per system
2. A composite gallery figure with mini-portraits for all passing systems
3. A comparison figure showing different crossing fractions
"""

import numpy as np
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
import matplotlib.gridspec as gridspec


def make_well_system(wells, omega, conf=0.03):
    """Create dynamics function for Gaussian wells with independent rotation."""
    wells = np.array(wells)

    def dyn(s):
        x, y = s[:, 0:1], s[:, 1:2]
        dVdx, dVdy = np.zeros_like(x), np.zeros_like(y)
        for i in range(len(wells)):
            cx, cy, amp, sigma = wells[i]
            dx, dy = x - cx, y - cy
            g = amp * np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))
            dVdx += g * dx / (sigma ** 2)
            dVdy += g * dy / (sigma ** 2)
        dVdx += 4 * conf * x ** 3
        dVdy += 4 * conf * y ** 3
        return np.concatenate([-dVdx + omega * y, -dVdy - omega * x], axis=1)

    return dyn


def rk4(dyn, s, dt=0.03):
    k1 = dyn(s)
    k2 = dyn(s + 0.5 * dt * k1)
    k3 = dyn(s + 0.5 * dt * k2)
    k4 = dyn(s + dt * k3)
    return s + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(dyn, n_traj=80, traj_len=400, ic_range=3.3, seed=42):
    """Simulate and return trajectories, endpoints, cluster assignments."""
    rng = np.random.RandomState(seed)
    s = rng.uniform(-ic_range, ic_range, (n_traj, 2))
    trajs = np.zeros((n_traj, traj_len + 1, 2))
    trajs[:, 0] = s
    for t in range(traj_len):
        s = np.clip(rk4(dyn, s), -15, 15)
        trajs[:, t + 1] = s
    for _ in range(1500):
        s = np.clip(rk4(dyn, s), -15, 15)
    clust = [s[0]]
    asn = [0] * n_traj
    for i in range(1, n_traj):
        ds = [np.linalg.norm(s[i] - c) for c in clust]
        if min(ds) < 0.35:
            asn[i] = ds.index(min(ds))
        else:
            asn[i] = len(clust)
            clust.append(s[i])
    ca = np.array(clust)
    # Crossing
    nc = 0
    for i in range(n_traj):
        fb = asn[i]
        tr = trajs[i, : traj_len - 30]
        d = np.linalg.norm(tr[:, None, :] - ca[None, :, :], axis=2)
        if np.any(d.argmin(axis=1) != fb):
            nc += 1
    return trajs, ca, asn, nc / n_traj


def plot_portrait(ax, trajs, ca, asn, dyn, ic_r, title=None, show_labels=True):
    """Plot a compact portrait (trajectories + streamlines on one axis)."""
    nb = len(ca)
    cmap = plt.colormaps.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(nb)]

    # Streamlines
    xr = np.linspace(-ic_r, ic_r, 20)
    yr = np.linspace(-ic_r, ic_r, 20)
    X, Y = np.meshgrid(xr, yr)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    dS = dyn(pts)
    U = dS[:, 0].reshape(X.shape)
    V = dS[:, 1].reshape(X.shape)
    ax.streamplot(
        xr, yr, U, V, color="#cccccc", linewidth=0.3, density=1.0, arrowsize=0.5
    )

    # Trajectories
    n_traj = len(asn)
    for i in range(min(60, n_traj)):
        c = colors[asn[i]]
        ax.plot(trajs[i, :, 0], trajs[i, :, 1], color=c, alpha=0.35, linewidth=0.6)
        # Start marker
        ax.plot(
            trajs[i, 0, 0],
            trajs[i, 0, 1],
            "o",
            color=c,
            ms=1.5,
            alpha=0.5,
        )

    # Basin centers
    for i in range(nb):
        ax.plot(
            ca[i, 0],
            ca[i, 1],
            "X",
            color=colors[i],
            ms=10,
            mec="black",
            mew=1.5,
            zorder=10,
        )

    ax.set_xlim(-ic_r, ic_r)
    ax.set_ylim(-ic_r, ic_r)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    if show_labels:
        ax.set_xlabel("$x_1$", fontsize=9)
        ax.set_ylabel("$x_2$", fontsize=9)
    ax.tick_params(labelsize=7)


def main():
    output_dir = "results/claude_catalog_validation/gallery"
    os.makedirs(output_dir, exist_ok=True)

    # Define systems
    systems = {}

    def poly(n, r=1.8, amp=2.0, omega=1.0, conf=0.03):
        return (
            [(r * math.cos(2 * math.pi * i / n + math.pi / 2),
              r * math.sin(2 * math.pi * i / n + math.pi / 2), amp, 0.5)
             for i in range(n)],
            omega, conf, r + 1.5,
        )

    systems["3-well triangle"] = poly(3, amp=2.0, omega=1.0)
    systems["4-well square"] = poly(4, amp=3.0, omega=1.0)
    systems["5-well pentagon"] = poly(5, amp=2.0, omega=1.0)
    systems["6-well hexagon"] = poly(6, amp=2.0, omega=0.8)
    systems["8-well octagon"] = (
        [(2.2 * math.cos(2 * math.pi * i / 8),
          2.2 * math.sin(2 * math.pi * i / 8), 3.0, 0.5)
         for i in range(8)],
        1.2, 0.02, 4.0,
    )
    systems["asymmetric 3-well"] = (
        [(0, 1.8, 2.5, 0.55), (-1.5, -0.9, 1.5, 0.4), (1.5, -0.9, 2.0, 0.5)],
        1.0, 0.03, 3.0,
    )
    systems["star topology 5"] = (
        [(2 * math.cos(2 * math.pi * i / 5 + math.pi / 10),
          2 * math.sin(2 * math.pi * i / 5 + math.pi / 10), 3.0, 0.55)
         for i in range(5)],
        1.5, 0.03, 3.5,
    )
    systems["depth gradient 4"] = (
        [(-1.3, 1.3, 1.5, 0.5), (1.3, 1.3, 2.0, 0.5),
         (1.3, -1.3, 2.5, 0.5), (-1.3, -1.3, 3.0, 0.5)],
        1.0, 0.03, 3.0,
    )
    systems["L-shape 5"] = (
        [(-1.5, 1.5, 2.5, 0.5), (-1.5, 0, 2.5, 0.5), (-1.5, -1.5, 2.5, 0.5),
         (0, -1.5, 2.5, 0.5), (1.5, -1.5, 2.5, 0.5)],
        1.0, 0.03, 3.5,
    )
    systems["diamond 4"] = (
        [(0, 2.2, 2.5, 0.5), (2.2, 0, 2.5, 0.5),
         (0, -2.2, 2.5, 0.5), (-2.2, 0, 2.5, 0.5)],
        1.0, 0.02, 3.5,
    )
    systems["high crossing 3 (62%)"] = (
        [(1.8 * math.cos(2 * math.pi * i / 3 + math.pi / 2),
          1.8 * math.sin(2 * math.pi * i / 3 + math.pi / 2), 3.0, 0.5)
         for i in range(3)],
        2.0, 0.03, 3.3,
    )
    systems["random placement 5"] = (
        [(-1.8, 1.5, 2, 0.5), (0, 2.2, 2, 0.5), (2, 0.5, 2, 0.5),
         (1, -1.8, 2, 0.5), (-1.5, -1, 2, 0.5)],
        1.0, 0.03, 3.5,
    )

    # Simulate all
    results = {}
    for name, (wells, omega, conf, ic_r) in systems.items():
        print(f"  Simulating: {name}", end="... ", flush=True)
        dyn = make_well_system(wells, omega, conf)
        trajs, ca, asn, cr = simulate(dyn, ic_range=ic_r)
        results[name] = (trajs, ca, asn, cr, dyn, ic_r)
        print(f"{len(ca)} basins, {cr:.0%} crossing")

    # Create composite gallery: 4x3 grid
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
    for idx, (name, (trajs, ca, asn, cr, dyn, ic_r)) in enumerate(results.items()):
        if idx >= nrows * ncols:
            break
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        plot_portrait(ax, trajs, ca, asn, dyn, ic_r,
                      title=f"{name}\n{len(ca)}B, {cr:.0%}C",
                      show_labels=(r == nrows - 1))

    fig.suptitle(
        "Transition-Rich Basin Partitioning Benchmark — System Gallery",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, "system_gallery.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGallery saved: {save_path}")

    # Create crossing-fraction comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (name, cr_target) in enumerate([
        ("3-well triangle", 0.47),  # moderate crossing
        ("high crossing 3 (62%)", 0.62),  # high crossing
        ("5-well pentagon", 0.41),  # multi-basin
    ]):
        trajs, ca, asn, cr, dyn, ic_r = results[name]
        ax = axes[idx]
        plot_portrait(ax, trajs, ca, asn, dyn, ic_r, title=f"{name}")
        ax.text(
            0.02, 0.98,
            f"B={len(ca)}, C={cr:.0%}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    fig.suptitle(
        "Crossing Fraction Comparison",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "crossing_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Crossing comparison saved: {save_path}")


if __name__ == "__main__":
    main()
