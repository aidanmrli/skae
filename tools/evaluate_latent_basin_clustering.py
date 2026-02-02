"""
Latent Space Basin Clustering Analysis for Any Koopman Model.

This script evaluates whether ANY Koopman model (GenericKM, LISTAKM, etc.)
naturally learns latent representations that cluster by basin of attraction,
WITHOUT requiring explicit basin structure in the model architecture.

Key questions:
1. Do trajectories from the same ground-truth basin cluster together in latent space?
2. Is there natural separation between basins in the learned representation?
3. Can a simple classifier predict basin from latent codes?

Usage:
    python evaluate_latent_basin_clustering.py \
        --checkpoint runs/generic_sparse/checkpoint.pt \
        --system duffing \
        --num_trajectories 100 \
        --output_dir results/latent_clustering
"""

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from skae.config import Config
from skae.data import make_env, generate_trajectory, Duffing, LyapunovMultiAttractor
from skae.model import make_model, KoopmanMachine


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

# Try to import dysts adapter for dysts environment support
try:
    from skae.benchmarks.dysts_adapter import DystsEnv, is_dysts_available
    HAS_DYSTS = is_dysts_available()
except ImportError:
    HAS_DYSTS = False
    DystsEnv = None


@dataclass
class BasinLabeledTrajectory:
    """A trajectory with ground-truth basin labels."""
    states: torch.Tensor          # [T, state_dim] - trajectory states
    final_basin: int              # Basin determined by long-term convergence
    system_name: str              # e.g., "duffing", "lyapunov"


@dataclass
class LatentAnalysisResults:
    """Results from latent space basin clustering analysis."""
    system_name: str
    model_name: str
    num_trajectories: int
    num_ground_truth_basins: int
    latent_dim: int

    # Clustering metrics
    silhouette_score: float           # Quality of clustering by GT basin
    adjusted_rand_index: float        # Agreement between k-means and GT basins
    kmeans_purity: float              # Fraction of clusters dominated by single GT basin

    # Separability metrics
    linear_classifier_accuracy: float  # Logistic regression accuracy
    linear_classifier_cv_std: float    # Cross-val std

    # Sparsity metrics
    mean_sparsity: float              # Fraction of near-zero entries
    mean_l1_norm: float               # Average L1 norm of latent codes

    # Variance explained
    pca_variance_ratio_2d: float      # Variance explained by first 2 PCs
    pca_variance_ratio_5d: float      # Variance explained by first 5 PCs

    # Per-basin statistics
    per_basin_centroid_distances: Dict[str, float] = field(default_factory=dict)
    basin_distribution: Dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Basin Identification (copied from evaluate_basin_structure.py)
# ---------------------------------------------------------------------------


def identify_duffing_basin(
    env: Duffing,
    trajectory: torch.Tensor,
    long_rollout_steps: int = 5000,
) -> int:
    """Identify Duffing basin: 0=left well, 1=right well."""
    state = trajectory[-1].clone()
    for _ in range(long_rollout_steps):
        state = env.step(state)
    return 0 if state[0].item() < 0 else 1


def identify_lyapunov_basin(
    env: LyapunovMultiAttractor,
    trajectory: torch.Tensor,
    long_rollout_steps: int = 5000,
) -> int:
    """Identify Lyapunov basin by nearest attractor."""
    state = trajectory[-1].clone()
    for _ in range(long_rollout_steps):
        state = env.step(state)
    distances = torch.norm(state - env.points, dim=-1)
    return distances.argmin().item()


def identify_dysts_duffing_basin(
    env,
    trajectory: torch.Tensor,
    long_rollout_steps: int = 5000,
) -> int:
    """
    Identify basin for dysts Duffing oscillator.

    The dysts Duffing oscillator has the same basin structure as the built-in:
    two stable fixed points. We identify by running a long rollout and checking
    the sign of the first state component (position).

    Returns:
        0 if trajectory converges to negative position (left well)
        1 if trajectory converges to positive position (right well)
    """
    state = trajectory[-1].clone()
    for _ in range(long_rollout_steps):
        state = env.step(state)
    # Duffing first component is position
    return 0 if state[0].item() < 0 else 1


# ---------------------------------------------------------------------------
# Dataset Generation
# ---------------------------------------------------------------------------


class BasinLabeledDataset:
    """Dataset of trajectories with ground-truth basin labels."""

    def __init__(
        self,
        system: str,
        cfg: Config,
        num_trajectories: int = 100,
        trajectory_length: int = 500,
        long_rollout_steps: int = 5000,
        seed: int = 42,
    ):
        self.system = system
        self.cfg = cfg
        self.num_trajectories = num_trajectories
        self.trajectory_length = trajectory_length
        self.long_rollout_steps = long_rollout_steps
        self.seed = seed

        self.env = make_env(cfg)

        # Normalize system name for comparison
        system_lower = system.lower()
        is_dysts_duffing = system_lower == 'dysts:duffing' or (
            system_lower.startswith('dysts:') and 'duffing' in system_lower.lower()
        )

        if system_lower == 'duffing' or is_dysts_duffing:
            self.num_basins = 2
            self.basin_names = ['Left Well (x<0)', 'Right Well (x>0)']
            # Attractor positions depend on the specific Duffing parameterization
            # For visualization purposes, approximate positions
            self.attractor_positions = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
            self.is_dysts_duffing = is_dysts_duffing
        elif system_lower == 'lyapunov':
            self.num_basins = 13
            self.attractor_positions = self.env.points.clone()
            self.basin_names = [f'Attractor {i}' for i in range(13)]
            self.is_dysts_duffing = False
        else:
            raise ValueError(
                f"Unknown system: {system}. Supported: duffing, dysts:Duffing, lyapunov"
            )

        self.trajectories: List[BasinLabeledTrajectory] = []
        self._generate_trajectories()

    def _generate_trajectories(self):
        """Generate trajectories with basin labels."""
        print(f"Generating {self.num_trajectories} trajectories for {self.system}...")

        for i in range(self.num_trajectories):
            env_rng = torch.Generator().manual_seed(self.seed + i)
            init_state = self.env.reset(env_rng)

            traj = generate_trajectory(
                self.env.step,
                init_state,
                length=self.trajectory_length,
            )
            traj = torch.cat([init_state.unsqueeze(0), traj], dim=0)

            # Identify basin based on system type
            system_lower = self.system.lower()
            if system_lower == 'duffing':
                final_basin = identify_duffing_basin(
                    self.env, traj, self.long_rollout_steps
                )
            elif system_lower == 'lyapunov':
                final_basin = identify_lyapunov_basin(
                    self.env, traj, self.long_rollout_steps
                )
            elif self.is_dysts_duffing:
                final_basin = identify_dysts_duffing_basin(
                    self.env, traj, self.long_rollout_steps
                )
            else:
                raise ValueError(f"No basin identification for system: {self.system}")

            self.trajectories.append(BasinLabeledTrajectory(
                states=traj,
                final_basin=final_basin,
                system_name=self.system,
            ))

            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{self.num_trajectories} trajectories")

        basin_counts = {}
        for traj in self.trajectories:
            b = traj.final_basin
            basin_counts[b] = basin_counts.get(b, 0) + 1
        print(f"Basin distribution: {basin_counts}")
        self.basin_counts = basin_counts

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> BasinLabeledTrajectory:
        return self.trajectories[idx]


# ---------------------------------------------------------------------------
# Latent Space Analysis
# ---------------------------------------------------------------------------


def encode_trajectories(
    model: KoopmanMachine,
    dataset: BasinLabeledDataset,
    device: str = 'cpu',
    aggregation: str = 'mean',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode all trajectories and return latent representations.

    Args:
        model: Any KoopmanMachine model
        dataset: Basin-labeled trajectories
        device: Device to run on
        aggregation: How to aggregate over time ('mean', 'last', 'all')

    Returns:
        latents: [N, latent_dim] or [N*T, latent_dim] if aggregation='all'
        labels: [N] or [N*T] ground-truth basin labels
    """
    model.eval()
    latents = []
    labels = []

    print(f"Encoding {len(dataset)} trajectories...")

    with torch.no_grad():
        for i, traj_data in enumerate(dataset.trajectories):
            states = traj_data.states.to(device)
            z = model.encode(states)  # [T, latent_dim]

            if aggregation == 'mean':
                z_agg = z.mean(dim=0)  # [latent_dim]
                latents.append(z_agg.cpu().numpy())
                labels.append(traj_data.final_basin)
            elif aggregation == 'last':
                z_agg = z[-1]  # [latent_dim]
                latents.append(z_agg.cpu().numpy())
                labels.append(traj_data.final_basin)
            elif aggregation == 'all':
                latents.append(z.cpu().numpy())
                labels.extend([traj_data.final_basin] * len(z))
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            if (i + 1) % 50 == 0:
                print(f"  Encoded {i + 1}/{len(dataset)} trajectories")

    if aggregation == 'all':
        latents = np.concatenate(latents, axis=0)
    else:
        latents = np.stack(latents, axis=0)

    labels = np.array(labels)

    return latents, labels


def compute_sparsity_metrics(latents: np.ndarray, threshold: float = 0.01) -> Dict[str, float]:
    """Compute sparsity metrics for latent codes."""
    # Fraction of entries below threshold
    sparsity = (np.abs(latents) < threshold).mean()

    # L1 norm
    l1_norm = np.abs(latents).sum(axis=-1).mean()

    # L0 "norm" (number of non-zero entries)
    l0_norm = (np.abs(latents) >= threshold).sum(axis=-1).mean()

    return {
        'mean_sparsity': float(sparsity),
        'mean_l1_norm': float(l1_norm),
        'mean_l0_norm': float(l0_norm),
    }


def compute_clustering_metrics(
    latents: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
) -> Dict[str, float]:
    """Compute clustering quality metrics."""
    # Silhouette score: how well do points cluster by their GT labels?
    if len(np.unique(labels)) > 1:
        sil_score = silhouette_score(latents, labels)
    else:
        sil_score = 0.0

    # K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(latents)

    # Adjusted Rand Index: agreement between k-means and GT
    ari = adjusted_rand_score(labels, kmeans_labels)

    # Cluster purity: for each cluster, what fraction is the dominant class?
    purity = 0.0
    for cluster_id in range(num_clusters):
        mask = kmeans_labels == cluster_id
        if mask.sum() > 0:
            cluster_labels = labels[mask]
            counts = np.bincount(cluster_labels)
            purity += counts.max()
    purity /= len(labels)

    return {
        'silhouette_score': float(sil_score),
        'adjusted_rand_index': float(ari),
        'kmeans_purity': float(purity),
    }


def compute_separability_metrics(
    latents: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute linear separability of basins."""
    # Logistic regression with cross-validation
    if len(np.unique(labels)) < 2:
        return {
            'linear_classifier_accuracy': 1.0,
            'linear_classifier_cv_std': 0.0,
        }

    clf = LogisticRegression(max_iter=1000, random_state=42)

    # Use fewer folds if we have few samples per class
    min_samples = min(np.bincount(labels))
    n_folds = min(5, min_samples)

    if n_folds < 2:
        # Not enough samples for CV, just fit and score
        clf.fit(latents, labels)
        acc = clf.score(latents, labels)
        return {
            'linear_classifier_accuracy': float(acc),
            'linear_classifier_cv_std': 0.0,
        }

    scores = cross_val_score(clf, latents, labels, cv=n_folds)

    return {
        'linear_classifier_accuracy': float(scores.mean()),
        'linear_classifier_cv_std': float(scores.std()),
    }


def compute_pca_metrics(latents: np.ndarray) -> Tuple[Dict[str, float], PCA]:
    """Compute PCA variance explained."""
    n_components = min(latents.shape[1], latents.shape[0], 10)
    pca = PCA(n_components=n_components)
    pca.fit(latents)

    var_ratio = pca.explained_variance_ratio_

    return {
        'pca_variance_ratio_2d': float(var_ratio[:2].sum()) if len(var_ratio) >= 2 else float(var_ratio.sum()),
        'pca_variance_ratio_5d': float(var_ratio[:5].sum()) if len(var_ratio) >= 5 else float(var_ratio.sum()),
    }, pca


def compute_centroid_distances(
    latents: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute distances between basin centroids."""
    unique_labels = np.unique(labels)
    centroids = {}

    for label in unique_labels:
        mask = labels == label
        centroids[label] = latents[mask].mean(axis=0)

    distances = {}
    for i, l1 in enumerate(unique_labels):
        for l2 in unique_labels[i+1:]:
            dist = np.linalg.norm(centroids[l1] - centroids[l2])
            distances[f'dist_{l1}_{l2}'] = float(dist)

    return distances


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def get_basin_colors(num_basins: int, cmap_name: str = 'tab20') -> List[str]:
    """Get distinct colors for basins."""
    cmap = plt.get_cmap(cmap_name)
    return [mcolors.to_hex(cmap(i / max(num_basins - 1, 1))) for i in range(num_basins)]


def plot_latent_pca(
    latents: np.ndarray,
    labels: np.ndarray,
    basin_names: List[str],
    pca: PCA,
    title: str = 'Latent Space PCA',
    output_path: Optional[Path] = None,
):
    """2D PCA visualization of latent space colored by basin."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Transform to 2D
    latents_2d = pca.transform(latents)[:, :2]

    colors = get_basin_colors(len(basin_names))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        name = basin_names[label] if label < len(basin_names) else f'Basin {label}'
        ax.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            c=colors[label % len(colors)],
            label=name,
            alpha=0.6,
            s=30,
        )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved PCA plot to {output_path}")

    plt.close()


def plot_latent_tsne(
    latents: np.ndarray,
    labels: np.ndarray,
    basin_names: List[str],
    title: str = 'Latent Space t-SNE',
    output_path: Optional[Path] = None,
    perplexity: int = 30,
):
    """t-SNE visualization of latent space colored by basin."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Reduce perplexity if we have few samples
    perplexity = min(perplexity, len(latents) // 4, len(latents) - 1)
    perplexity = max(perplexity, 5)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    latents_2d = tsne.fit_transform(latents)

    colors = get_basin_colors(len(basin_names))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        name = basin_names[label] if label < len(basin_names) else f'Basin {label}'
        ax.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            c=colors[label % len(colors)],
            label=name,
            alpha=0.6,
            s=30,
        )

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved t-SNE plot to {output_path}")

    plt.close()


def plot_latent_activation_heatmap(
    latents: np.ndarray,
    labels: np.ndarray,
    basin_names: List[str],
    title: str = 'Mean Latent Activations by Basin',
    output_path: Optional[Path] = None,
    max_dims: int = 50,
):
    """Heatmap of mean latent activations per basin."""
    unique_labels = np.unique(labels)

    # Compute mean activation per basin
    mean_activations = []
    for label in unique_labels:
        mask = labels == label
        mean_activations.append(latents[mask].mean(axis=0))

    mean_activations = np.stack(mean_activations)

    # Limit dimensions for visibility
    if mean_activations.shape[1] > max_dims:
        # Show dimensions with highest variance across basins
        var = mean_activations.var(axis=0)
        top_dims = np.argsort(-var)[:max_dims]
        mean_activations = mean_activations[:, top_dims]
        dim_labels = [f'd{i}' for i in top_dims]
    else:
        dim_labels = [f'd{i}' for i in range(mean_activations.shape[1])]

    fig, ax = plt.subplots(figsize=(14, 6))

    im = ax.imshow(mean_activations, aspect='auto', cmap='RdBu_r')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Activation')

    ax.set_yticks(range(len(unique_labels)))
    ax.set_yticklabels([basin_names[l] if l < len(basin_names) else f'Basin {l}'
                        for l in unique_labels])

    # Only show some x labels if too many
    if len(dim_labels) > 30:
        step = len(dim_labels) // 20
        ax.set_xticks(range(0, len(dim_labels), step))
        ax.set_xticklabels(dim_labels[::step], rotation=45, ha='right')
    else:
        ax.set_xticks(range(len(dim_labels)))
        ax.set_xticklabels(dim_labels, rotation=45, ha='right')

    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Ground-Truth Basin')
    ax.set_title(title)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved activation heatmap to {output_path}")

    plt.close()


def plot_latent_sparsity_distribution(
    latents: np.ndarray,
    labels: np.ndarray,
    basin_names: List[str],
    title: str = 'Latent Sparsity Distribution',
    output_path: Optional[Path] = None,
):
    """Distribution of sparsity (fraction of near-zero entries) per basin."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Per-sample sparsity
    sparsity_per_sample = (np.abs(latents) < 0.01).mean(axis=1)
    l1_per_sample = np.abs(latents).sum(axis=1)

    colors = get_basin_colors(len(basin_names))
    unique_labels = np.unique(labels)

    # Sparsity histogram
    for label in unique_labels:
        mask = labels == label
        name = basin_names[label] if label < len(basin_names) else f'Basin {label}'
        axes[0].hist(
            sparsity_per_sample[mask],
            bins=20,
            alpha=0.5,
            label=name,
            color=colors[label % len(colors)],
        )

    axes[0].set_xlabel('Sparsity (fraction < 0.01)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Sparsity Distribution by Basin')
    axes[0].legend(fontsize=8)

    # L1 norm histogram
    for label in unique_labels:
        mask = labels == label
        name = basin_names[label] if label < len(basin_names) else f'Basin {label}'
        axes[1].hist(
            l1_per_sample[mask],
            bins=20,
            alpha=0.5,
            label=name,
            color=colors[label % len(colors)],
        )

    axes[1].set_xlabel('L1 Norm')
    axes[1].set_ylabel('Count')
    axes[1].set_title('L1 Norm Distribution by Basin')
    axes[1].legend(fontsize=8)

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved sparsity distribution to {output_path}")

    plt.close()


def plot_phase_portrait_with_latent_coloring(
    dataset: BasinLabeledDataset,
    latents_per_traj: List[np.ndarray],
    pca: PCA,
    output_path: Optional[Path] = None,
    max_trajectories: int = 30,
):
    """
    Phase portrait colored by latent space PC1 value.

    This shows whether different regions of phase space map to different
    latent representations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: colored by ground-truth basin
    # Right: colored by PC1 value

    gt_colors = get_basin_colors(dataset.num_basins)

    num_to_plot = min(max_trajectories, len(dataset))

    all_pc1_values = []
    for latent in latents_per_traj[:num_to_plot]:
        pc_vals = pca.transform(latent)[:, 0]
        all_pc1_values.extend(pc_vals)

    pc1_min, pc1_max = min(all_pc1_values), max(all_pc1_values)

    for i in range(num_to_plot):
        traj_data = dataset.trajectories[i]
        states = traj_data.states.numpy()
        gt_basin = traj_data.final_basin

        # Left: ground-truth coloring
        axes[0].plot(
            states[:, 0], states[:, 1],
            color=gt_colors[gt_basin],
            alpha=0.5,
            linewidth=0.8,
        )

        # Right: PC1 coloring
        latent = latents_per_traj[i]
        pc1_values = pca.transform(latent)[:, 0]

        # Normalize to [0, 1] for colormap
        pc1_norm = (pc1_values - pc1_min) / (pc1_max - pc1_min + 1e-8)

        for t in range(len(states) - 1):
            color = plt.cm.coolwarm(pc1_norm[t])
            axes[1].plot(
                states[t:t+2, 0], states[t:t+2, 1],
                color=color,
                alpha=0.7,
                linewidth=0.8,
            )

    # Add attractors
    if hasattr(dataset, 'attractor_positions'):
        for ax in axes:
            ax.scatter(
                dataset.attractor_positions[:, 0].numpy(),
                dataset.attractor_positions[:, 1].numpy(),
                marker='*', s=200, c='black', zorder=10,
            )

    axes[0].set_title('Ground-Truth Basin Coloring')
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')

    axes[1].set_title('Latent PC1 Coloring')
    axes[1].set_xlabel('x1')
    axes[1].set_ylabel('x2')

    # Add colorbar for PC1
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(pc1_min, pc1_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1])
    cbar.set_label('PC1 Value')

    # Legend for ground-truth
    from matplotlib.lines import Line2D
    gt_legend = [
        Line2D([0], [0], color=gt_colors[i], label=dataset.basin_names[i] if i < len(dataset.basin_names) else f'Basin {i}')
        for i in range(min(dataset.num_basins, 10))
    ]
    axes[0].legend(handles=gt_legend, loc='upper right', fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved phase portrait with latent coloring to {output_path}")

    plt.close()


# ---------------------------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------------------------


def run_full_analysis(
    model: KoopmanMachine,
    dataset: BasinLabeledDataset,
    device: str = 'cpu',
) -> Tuple[LatentAnalysisResults, Dict[str, Any]]:
    """Run complete latent space analysis."""

    # Encode trajectories (mean over time)
    latents_mean, labels = encode_trajectories(model, dataset, device, aggregation='mean')

    # Also get per-trajectory latents for some visualizations
    latents_per_traj = []
    model.eval()
    with torch.no_grad():
        for traj_data in dataset.trajectories:
            states = traj_data.states.to(device)
            z = model.encode(states).cpu().numpy()
            latents_per_traj.append(z)

    print(f"\nLatent space shape: {latents_mean.shape}")
    print(f"Labels shape: {labels.shape}, unique: {np.unique(labels)}")

    # Compute all metrics
    print("\nComputing metrics...")

    sparsity_metrics = compute_sparsity_metrics(latents_mean)
    print(f"  Sparsity: {sparsity_metrics['mean_sparsity']:.4f}")
    print(f"  L1 norm: {sparsity_metrics['mean_l1_norm']:.4f}")

    clustering_metrics = compute_clustering_metrics(latents_mean, labels, dataset.num_basins)
    print(f"  Silhouette score: {clustering_metrics['silhouette_score']:.4f}")
    print(f"  Adjusted Rand Index: {clustering_metrics['adjusted_rand_index']:.4f}")
    print(f"  K-means purity: {clustering_metrics['kmeans_purity']:.4f}")

    separability_metrics = compute_separability_metrics(latents_mean, labels)
    print(f"  Linear classifier accuracy: {separability_metrics['linear_classifier_accuracy']:.4f}")

    pca_metrics, pca = compute_pca_metrics(latents_mean)
    print(f"  PCA variance (2D): {pca_metrics['pca_variance_ratio_2d']:.4f}")
    print(f"  PCA variance (5D): {pca_metrics['pca_variance_ratio_5d']:.4f}")

    centroid_distances = compute_centroid_distances(latents_mean, labels)

    # Build results
    results = LatentAnalysisResults(
        system_name=dataset.system,
        model_name=type(model).__name__,
        num_trajectories=len(dataset),
        num_ground_truth_basins=dataset.num_basins,
        latent_dim=latents_mean.shape[1],
        silhouette_score=clustering_metrics['silhouette_score'],
        adjusted_rand_index=clustering_metrics['adjusted_rand_index'],
        kmeans_purity=clustering_metrics['kmeans_purity'],
        linear_classifier_accuracy=separability_metrics['linear_classifier_accuracy'],
        linear_classifier_cv_std=separability_metrics['linear_classifier_cv_std'],
        mean_sparsity=sparsity_metrics['mean_sparsity'],
        mean_l1_norm=sparsity_metrics['mean_l1_norm'],
        pca_variance_ratio_2d=pca_metrics['pca_variance_ratio_2d'],
        pca_variance_ratio_5d=pca_metrics['pca_variance_ratio_5d'],
        per_basin_centroid_distances=centroid_distances,
        basin_distribution=dataset.basin_counts,
    )

    # Return extra data for visualizations
    extra_data = {
        'latents_mean': latents_mean,
        'latents_per_traj': latents_per_traj,
        'labels': labels,
        'pca': pca,
    }

    return results, extra_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate latent space basin clustering for any Koopman model'
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--system', type=str, default=None,
                        help='Dynamical system to evaluate on. If not specified, uses the system '
                             'from the checkpoint. Supported: duffing, lyapunov, dysts:Duffing')
    parser.add_argument('--num_trajectories', type=int, default=100,
                        help='Number of test trajectories to generate')
    parser.add_argument('--trajectory_length', type=int, default=500,
                        help='Length of each trajectory')
    parser.add_argument('--output_dir', type=str, default='results/latent_clustering',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for trajectory generation')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to run on')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Load config
    cfg = Config.from_dict(checkpoint['config'])

    # Determine system to evaluate on
    if args.system is not None:
        system = args.system
        cfg.ENV.ENV_NAME = system
    else:
        system = cfg.ENV.ENV_NAME

    print(f"Evaluating on system: {system}")

    # Create model
    print("Creating model...")
    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    print(f"Model type: {type(model).__name__}")
    print(f"Latent dim: {cfg.MODEL.TARGET_SIZE}")

    # Generate dataset
    dataset = BasinLabeledDataset(
        system=system,
        cfg=cfg,
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        seed=args.seed,
    )

    # Run analysis
    results, extra_data = run_full_analysis(model, dataset, device=args.device)

    # Save results
    results_path = output_dir / 'analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\nSaved analysis results to {results_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    latents_mean = extra_data['latents_mean']
    labels = extra_data['labels']
    pca = extra_data['pca']
    latents_per_traj = extra_data['latents_per_traj']

    # 1. PCA plot
    plot_latent_pca(
        latents_mean, labels, dataset.basin_names, pca,
        title=f'Latent Space PCA ({type(model).__name__})',
        output_path=output_dir / 'latent_pca.png',
    )

    # 2. t-SNE plot
    plot_latent_tsne(
        latents_mean, labels, dataset.basin_names,
        title=f'Latent Space t-SNE ({type(model).__name__})',
        output_path=output_dir / 'latent_tsne.png',
    )

    # 3. Activation heatmap
    plot_latent_activation_heatmap(
        latents_mean, labels, dataset.basin_names,
        title=f'Mean Latent Activations ({type(model).__name__})',
        output_path=output_dir / 'activation_heatmap.png',
    )

    # 4. Sparsity distribution
    plot_latent_sparsity_distribution(
        latents_mean, labels, dataset.basin_names,
        title=f'Latent Sparsity ({type(model).__name__})',
        output_path=output_dir / 'sparsity_distribution.png',
    )

    # 5. Phase portrait with latent coloring
    plot_phase_portrait_with_latent_coloring(
        dataset, latents_per_traj, pca,
        output_path=output_dir / 'phase_portrait_latent.png',
    )

    # Print summary
    print("\n" + "=" * 60)
    print("LATENT SPACE BASIN CLUSTERING ANALYSIS")
    print("=" * 60)
    print(f"System: {results.system_name}")
    print(f"Model: {results.model_name}")
    print(f"Latent dim: {results.latent_dim}")
    print(f"Trajectories: {results.num_trajectories}")
    print(f"Ground-truth basins: {results.num_ground_truth_basins}")
    print("-" * 60)
    print("CLUSTERING METRICS:")
    print(f"  Silhouette Score: {results.silhouette_score:.4f}  (>0.5 = good separation)")
    print(f"  Adjusted Rand Index: {results.adjusted_rand_index:.4f}  (1.0 = perfect)")
    print(f"  K-means Purity: {results.kmeans_purity:.4f}  (1.0 = perfect)")
    print("-" * 60)
    print("SEPARABILITY METRICS:")
    print(f"  Linear Classifier Acc: {results.linear_classifier_accuracy:.4f} ± {results.linear_classifier_cv_std:.4f}")
    print("-" * 60)
    print("SPARSITY METRICS:")
    print(f"  Mean Sparsity: {results.mean_sparsity:.4f}  (fraction < 0.01)")
    print(f"  Mean L1 Norm: {results.mean_l1_norm:.4f}")
    print("-" * 60)
    print("PCA METRICS:")
    print(f"  Variance (2D): {results.pca_variance_ratio_2d:.4f}")
    print(f"  Variance (5D): {results.pca_variance_ratio_5d:.4f}")
    print("-" * 60)
    print(f"Basin distribution: {results.basin_distribution}")
    print("=" * 60)
    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()
