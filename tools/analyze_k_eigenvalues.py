"""Per-block eigenvalue analysis of the Koopman matrix K.

Loads a trained checkpoint, extracts eigenvalues from each K block (respecting
the K structure: dense, diagonal, block_diagonal, arrowhead), and optionally
correlates block activations with ground-truth basin labels.

Outputs:
  - eigenvalue_analysis.json   Per-block eigenvalue stats
  - eigenvalues_complex_plane.png   Eigenvalues on the complex plane
  - spectral_radius_by_block.png    Spectral radius bar chart
  - basin_block_heatmap.png         Basin x block activation heatmap (with --correlate_basins)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from skae.config import Config
from skae.data import make_env
from skae.model import make_model, StructuredLISTAKM, LISTAKM


# ---------------------------------------------------------------------------
# Eigenvalue extraction per block
# ---------------------------------------------------------------------------


def _eigvals_of_block(block: torch.Tensor) -> np.ndarray:
    """Compute eigenvalues of a square matrix, returned as complex128 numpy."""
    with torch.no_grad():
        ev = torch.linalg.eigvals(block.cpu())
    return ev.numpy().astype(np.complex128)


def extract_block_eigenvalues(model) -> List[Dict]:
    """Extract eigenvalues from each K block depending on model structure.

    Returns a list of dicts, one per block, each with:
        name:           human-readable block label
        eigenvalues:    complex128 ndarray of eigenvalues
        spectral_radius: max |lambda|
        num_stable:     count of |lambda| < 1
        num_unstable:   count of |lambda| > 1
        size:           block dimension
    """
    blocks: List[Dict] = []

    if isinstance(model, StructuredLISTAKM):
        # Arrowhead: global block + per-basin blocks
        with torch.no_grad():
            ev = _eigvals_of_block(model.K_global)
        blocks.append(_block_stats("global", ev, model.d_global))

        for k in range(model.num_basins):
            with torch.no_grad():
                ev = _eigvals_of_block(model.K_basin[k])
            blocks.append(_block_stats(f"basin_{k}", ev, model.d_basin))

    elif isinstance(model, LISTAKM):
        k_struct = model._k_structure

        if k_struct == "diagonal":
            with torch.no_grad():
                diag_vals = model.kmat_diag.detach().cpu().numpy()
            ev = diag_vals.astype(np.complex128)
            blocks.append(_block_stats("diagonal", ev, len(diag_vals)))

        elif k_struct == "block_diagonal":
            for i, blk in enumerate(model.kmat_blocks):
                ev = _eigvals_of_block(blk)
                blocks.append(_block_stats(f"block_{i}", ev, blk.shape[0]))
            if model._k_remainder > 0:
                ev = _eigvals_of_block(model.kmat_remainder)
                blocks.append(_block_stats("remainder", ev, model.kmat_remainder.shape[0]))

        else:
            # Dense
            ev = _eigvals_of_block(model.kmat)
            blocks.append(_block_stats("dense", ev, model.kmat.shape[0]))
    else:
        # GenericKM or others with .kmat
        kmat = model.kmatrix()
        ev = _eigvals_of_block(kmat)
        blocks.append(_block_stats("dense", ev, kmat.shape[0]))

    return blocks


def _block_stats(name: str, eigenvalues: np.ndarray, size: int) -> Dict:
    mags = np.abs(eigenvalues)
    return {
        "name": name,
        "eigenvalues": eigenvalues,
        "spectral_radius": float(mags.max()) if mags.size > 0 else 0.0,
        "num_stable": int((mags < 1.0).sum()),
        "num_unstable": int((mags > 1.0).sum()),
        "num_marginal": int(np.isclose(mags, 1.0, atol=1e-4).sum()),
        "size": size,
    }


# ---------------------------------------------------------------------------
# Basin-block correlation
# ---------------------------------------------------------------------------


def compute_basin_block_activation(
    model,
    system: str,
    cfg: Config,
    device: str,
    num_trajectories: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Compute mean activation magnitude per latent dim per basin, grouped by K block.

    Returns:
        heatmap:     [num_basins, num_blocks] activation matrix
        basin_names: list of basin label strings
        block_names: list of block label strings
    """
    from skae.basin_utils import BasinLabeledDataset

    dataset = BasinLabeledDataset(
        system=system,
        cfg=cfg,
        num_trajectories=num_trajectories,
        seed=seed,
    )

    # Encode all trajectories, accumulate per-dim magnitude per basin
    model.eval()
    basin_dim_accum: Dict[int, List[np.ndarray]] = {
        b: [] for b in range(dataset.num_basins)
    }

    with torch.no_grad():
        for traj in dataset.trajectories:
            states = traj.states.to(device)
            z = model.encode(states)
            z_mean_abs = z.abs().mean(dim=0).cpu().numpy()  # [zdim]
            basin_dim_accum[traj.final_basin].append(z_mean_abs)

    # Compute per-basin mean activation per dim
    zdim = model.target_size
    basin_act = np.zeros((dataset.num_basins, zdim))
    for b, vecs in basin_dim_accum.items():
        if vecs:
            basin_act[b] = np.mean(vecs, axis=0)

    # Map dims to blocks
    block_ranges, block_names = _get_block_ranges(model)
    num_blocks = len(block_ranges)

    # Aggregate: mean activation per block per basin
    heatmap = np.zeros((dataset.num_basins, num_blocks))
    for bi in range(dataset.num_basins):
        for blk_idx, (start, end) in enumerate(block_ranges):
            heatmap[bi, blk_idx] = basin_act[bi, start:end].mean()

    basin_names = dataset.basin_names
    return heatmap, basin_names, block_names


def _get_block_ranges(model) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Return (start, end) index ranges and names for each K block."""
    if isinstance(model, StructuredLISTAKM):
        ranges = [(0, model.d_global)]
        names = ["global"]
        offset = model.d_global
        for k in range(model.num_basins):
            ranges.append((offset, offset + model.d_basin))
            names.append(f"basin_{k}")
            offset += model.d_basin
        return ranges, names

    if isinstance(model, LISTAKM):
        k_struct = model._k_structure

        if k_struct == "diagonal":
            # Treat each element as its own "block" is too granular;
            # group into chunks matching num_basins for readability
            zdim = model.target_size
            chunk = max(1, zdim // 13)
            ranges = []
            names = []
            idx = 0
            blk_i = 0
            while idx < zdim:
                end = min(idx + chunk, zdim)
                ranges.append((idx, end))
                names.append(f"diag_chunk_{blk_i}")
                idx = end
                blk_i += 1
            return ranges, names

        if k_struct == "block_diagonal":
            bs = model._k_block_size
            ranges = []
            names = []
            for i in range(model._k_num_blocks):
                ranges.append((i * bs, (i + 1) * bs))
                names.append(f"block_{i}")
            if model._k_remainder > 0:
                offset = model._k_num_blocks * bs
                ranges.append((offset, offset + model._k_remainder))
                names.append("remainder")
            return ranges, names

    # Dense / generic: single block
    zdim = model.target_size
    return [(0, zdim)], ["dense"]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _ensure_matplotlib():
    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")


def plot_eigenvalues_complex_plane(
    block_data: List[Dict],
    path: Path,
) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1, alpha=0.4, label="unit circle")

    n_blocks = len(block_data)
    cmap = cm.get_cmap("tab20", max(n_blocks, 1))

    for idx, blk in enumerate(block_data):
        ev = blk["eigenvalues"]
        color = cmap(idx)
        ax.scatter(
            ev.real, ev.imag,
            c=[color], s=20, alpha=0.7,
            label=f"{blk['name']} (r={blk['spectral_radius']:.3f})",
            zorder=3,
        )

    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.set_title("Koopman eigenvalues by block")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)
    if n_blocks <= 20:
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_spectral_radius_bar(
    block_data: List[Dict],
    path: Path,
) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    names = [b["name"] for b in block_data]
    radii = [b["spectral_radius"] for b in block_data]

    fig, ax = plt.subplots(1, 1, figsize=(max(6, len(names) * 0.4), 5))
    bars = ax.bar(range(len(names)), radii, color="#4682B4", alpha=0.85, edgecolor="white")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.6, label="|lambda|=1")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("Spectral radius")
    ax.set_title("Spectral radius per K block")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_basin_block_heatmap(
    heatmap: np.ndarray,
    basin_names: List[str],
    block_names: List[str],
    path: Path,
) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(max(6, len(block_names) * 0.5), max(5, len(basin_names) * 0.4)))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(block_names)))
    ax.set_xticklabels(block_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(basin_names)))
    ax.set_yticklabels(basin_names, fontsize=7)
    ax.set_xlabel("K block")
    ax.set_ylabel("Basin")
    ax.set_title("Mean activation magnitude (basin x block)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------


def _serialize_block_data(block_data: List[Dict]) -> List[Dict]:
    """Convert block data to JSON-serializable form (drop complex arrays)."""
    out = []
    for blk in block_data:
        ev = blk["eigenvalues"]
        entry = {
            "name": blk["name"],
            "size": blk["size"],
            "spectral_radius": blk["spectral_radius"],
            "num_stable": blk["num_stable"],
            "num_unstable": blk["num_unstable"],
            "num_marginal": blk["num_marginal"],
            "eigenvalues_real": ev.real.tolist(),
            "eigenvalues_imag": ev.imag.tolist(),
        }
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Per-block eigenvalue analysis of Koopman matrix")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--system", type=str, default=None,
                        help="System for basin correlation (default: from checkpoint config)")
    parser.add_argument("--correlate_basins", action="store_true",
                        help="Compute basin-to-block activation correlation")
    parser.add_argument("--num_trajectories", type=int, default=100,
                        help="Number of trajectories for basin correlation")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: alongside checkpoint)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Device for inference")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "eigenvalue_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    cfg = Config.from_dict(checkpoint["config"])

    system = args.system if args.system else cfg.ENV.ENV_NAME
    cfg.ENV.ENV_NAME = system

    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    # Determine K structure for reporting
    if isinstance(model, StructuredLISTAKM):
        k_structure = "arrowhead"
    elif isinstance(model, LISTAKM):
        k_structure = model._k_structure
    else:
        k_structure = "dense"

    print(f"Model: {type(model).__name__}, K structure: {k_structure}, "
          f"target_size: {model.target_size}")

    # --- Eigenvalue extraction ---
    print("Extracting per-block eigenvalues ...")
    block_data = extract_block_eigenvalues(model)
    for blk in block_data:
        print(f"  {blk['name']}: size={blk['size']}, "
              f"spectral_radius={blk['spectral_radius']:.4f}, "
              f"stable={blk['num_stable']}, unstable={blk['num_unstable']}, "
              f"marginal={blk['num_marginal']}")

    # --- Plots ---
    plot_eigenvalues_complex_plane(block_data, output_dir / "eigenvalues_complex_plane.png")
    print(f"Saved eigenvalue complex plane plot")
    plot_spectral_radius_bar(block_data, output_dir / "spectral_radius_by_block.png")
    print(f"Saved spectral radius bar chart")

    # --- Basin correlation ---
    heatmap_data = None
    if args.correlate_basins:
        print(f"Computing basin-block activation correlation ({args.num_trajectories} trajectories) ...")
        heatmap, basin_names, block_names = compute_basin_block_activation(
            model=model,
            system=system,
            cfg=cfg,
            device=args.device,
            num_trajectories=args.num_trajectories,
            seed=args.seed,
        )
        plot_basin_block_heatmap(heatmap, basin_names, block_names, output_dir / "basin_block_heatmap.png")
        print(f"Saved basin-block heatmap")
        heatmap_data = {
            "basin_names": basin_names,
            "block_names": block_names,
            "heatmap": heatmap.tolist(),
        }

    # --- Save JSON ---
    results = {
        "checkpoint": str(checkpoint_path),
        "model_class": type(model).__name__,
        "k_structure": k_structure,
        "target_size": model.target_size,
        "system": system,
        "blocks": _serialize_block_data(block_data),
    }
    if heatmap_data is not None:
        results["basin_block_correlation"] = heatmap_data

    results_path = output_dir / "eigenvalue_analysis.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
