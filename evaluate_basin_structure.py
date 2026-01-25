"""
Basin Structure Evaluation for StructuredLISTAKM.

This script evaluates whether StructuredLISTAKM learns distinct Koopman dynamics
for each basin of attraction by analyzing the correspondence between:
- Ground-truth basins (determined by long-term convergence)
- Predicted basins (determined by which model basin block is most active)

Usage:
    python evaluate_basin_structure.py \
        --checkpoint runs/structured_lista/checkpoint.pt \
        --system duffing \
        --num_trajectories 100 \
        --output_dir results/basin_analysis
"""

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from config import Config
from data import make_env, generate_trajectory, Duffing, LyapunovMultiAttractor
from model import make_model, StructuredLISTAKM


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class BasinLabeledTrajectory:
    """A trajectory with ground-truth basin labels."""
    states: torch.Tensor          # [T, state_dim] - trajectory states
    basin_labels: torch.Tensor    # [T] - ground-truth basin index at each timestep
    final_basin: int              # Basin determined by long-term convergence
    system_name: str              # e.g., "duffing", "lyapunov"


@dataclass
class BasinActivations:
    """Activations from encoding a trajectory."""
    z: torch.Tensor               # [T, latent_dim] - full latent codes
    z_global: torch.Tensor        # [T, d_global] - global block
    z_basins: torch.Tensor        # [T, B, d_basin] - basin blocks
    basin_norms: torch.Tensor     # [T, B] - L2 norm of each basin block
    active_basin: torch.Tensor    # [T] - argmax basin at each timestep
    activation_entropy: torch.Tensor  # [T] - entropy of normalized basin norms


@dataclass
class AnalysisResults:
    """Results from basin structure analysis."""
    system_name: str
    num_trajectories: int
    num_ground_truth_basins: int
    num_model_basins: int

    # Metrics
    basin_assignment_accuracy: float
    temporal_consistency: float
    mean_activation_entropy: float
    within_basin_similarity: float
    cross_basin_separation: float

    # Confusion matrix: [num_gt_basins, num_model_basins]
    confusion_matrix: List[List[float]] = field(default_factory=list)

    # Per ground-truth basin metrics
    per_basin_accuracy: Dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Basin Identification
# ---------------------------------------------------------------------------


def identify_duffing_basin(
    env: Duffing,
    trajectory: torch.Tensor,
    long_rollout_steps: int = 5000,
) -> int:
    """
    Identify which basin a Duffing trajectory belongs to.

    Duffing has two stable fixed points at (x, x') = (±1, 0).
    Run long rollout and check sign of final x position.

    Args:
        env: Duffing environment
        trajectory: [T, 2] trajectory
        long_rollout_steps: Steps to run for convergence

    Returns:
        0 if trajectory converges to x ≈ -1 (left well)
        1 if trajectory converges to x ≈ +1 (right well)
    """
    # Start from last state and run long rollout
    state = trajectory[-1].clone()
    for _ in range(long_rollout_steps):
        state = env.step(state)

    # Check which well we ended up in
    return 0 if state[0].item() < 0 else 1


def identify_lyapunov_basin(
    env: LyapunovMultiAttractor,
    trajectory: torch.Tensor,
    long_rollout_steps: int = 5000,
) -> int:
    """
    Identify which basin a Lyapunov trajectory belongs to.

    Lyapunov system has 13 fixed-point attractors.
    Identify basin by nearest attractor after convergence.

    Args:
        env: LyapunovMultiAttractor environment
        trajectory: [T, 2] trajectory
        long_rollout_steps: Steps to run for convergence

    Returns:
        Index of nearest attractor (0 to 12)
    """
    # Start from last state and run long rollout
    state = trajectory[-1].clone()
    for _ in range(long_rollout_steps):
        state = env.step(state)

    # Find nearest attractor
    distances = torch.norm(state - env.points, dim=-1)
    return distances.argmin().item()


def get_basin_colors(num_basins: int, cmap_name: str = 'tab20') -> List[str]:
    """Get a list of distinct colors for basins."""
    cmap = plt.get_cmap(cmap_name)
    return [mcolors.to_hex(cmap(i / max(num_basins - 1, 1))) for i in range(num_basins)]


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

        # Create environment
        self.env = make_env(cfg)

        # Get attractor info for the system
        if system.lower() == 'duffing':
            self.num_basins = 2
            self.basin_names = ['Left Well (x→-1)', 'Right Well (x→+1)']
            self.attractor_positions = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        elif system.lower() == 'lyapunov':
            self.num_basins = 13
            self.attractor_positions = self.env.points.clone()
            self.basin_names = [f'Attractor {i}' for i in range(13)]
        else:
            raise ValueError(f"Unknown system: {system}. Supported: duffing, lyapunov")

        # Generate trajectories
        self.trajectories: List[BasinLabeledTrajectory] = []
        self._generate_trajectories()

    def _generate_trajectories(self):
        """Generate trajectories with basin labels."""
        rng = torch.Generator().manual_seed(self.seed)

        print(f"Generating {self.num_trajectories} trajectories for {self.system}...")

        for i in range(self.num_trajectories):
            # Reset to random initial condition
            env_rng = torch.Generator().manual_seed(self.seed + i)
            init_state = self.env.reset(env_rng)

            # Generate trajectory
            traj = generate_trajectory(
                self.env.step,
                init_state,
                length=self.trajectory_length,
            )
            # Prepend initial state
            traj = torch.cat([init_state.unsqueeze(0), traj], dim=0)

            # Identify basin
            if self.system.lower() == 'duffing':
                final_basin = identify_duffing_basin(
                    self.env, traj, self.long_rollout_steps
                )
            elif self.system.lower() == 'lyapunov':
                final_basin = identify_lyapunov_basin(
                    self.env, traj, self.long_rollout_steps
                )

            # For simplicity, label entire trajectory with final basin
            # (assumes no transitions within trajectory)
            basin_labels = torch.full((len(traj),), final_basin, dtype=torch.long)

            self.trajectories.append(BasinLabeledTrajectory(
                states=traj,
                basin_labels=basin_labels,
                final_basin=final_basin,
                system_name=self.system,
            ))

            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{self.num_trajectories} trajectories")

        # Print basin distribution
        basin_counts = {}
        for traj in self.trajectories:
            b = traj.final_basin
            basin_counts[b] = basin_counts.get(b, 0) + 1
        print(f"Basin distribution: {basin_counts}")

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> BasinLabeledTrajectory:
        return self.trajectories[idx]


# ---------------------------------------------------------------------------
# Model Analysis
# ---------------------------------------------------------------------------


def compute_basin_activations(
    model: StructuredLISTAKM,
    trajectory: torch.Tensor,
    device: str = 'cpu',
) -> BasinActivations:
    """
    Encode trajectory and compute basin block activations.

    Args:
        model: StructuredLISTAKM model
        trajectory: [T, state_dim] trajectory states
        device: Device to run on

    Returns:
        BasinActivations with all computed values
    """
    model.eval()
    trajectory = trajectory.to(device)

    with torch.no_grad():
        # Encode
        z = model.encode(trajectory)  # [T, latent_dim]

        # Partition
        z_global, z_basins = model._partition_latent(z)  # z_basins: [T, B, d_basin]

        # Compute norms
        basin_norms = torch.norm(z_basins, p=2, dim=-1)  # [T, B]

        # Active basin (argmax)
        active_basin = basin_norms.argmax(dim=-1)  # [T]

        # Entropy of softmax normalized norms
        basin_probs = F.softmax(basin_norms, dim=-1)
        entropy = -torch.sum(basin_probs * torch.log(basin_probs + 1e-8), dim=-1)

    return BasinActivations(
        z=z.cpu(),
        z_global=z_global.cpu(),
        z_basins=z_basins.cpu(),
        basin_norms=basin_norms.cpu(),
        active_basin=active_basin.cpu(),
        activation_entropy=entropy.cpu(),
    )


def compute_basin_assignment_accuracy(
    activations_list: List[BasinActivations],
    ground_truth_basins: List[int],
    num_ground_truth_basins: int,
    num_model_basins: int,
) -> Tuple[float, torch.Tensor, Dict[int, int]]:
    """
    Compute accuracy of basin assignment after finding optimal mapping.

    Since model basins are unlabeled, we find the best mapping from
    model basin indices to ground-truth basin indices.

    Args:
        activations_list: List of activations for each trajectory
        ground_truth_basins: Ground-truth basin for each trajectory
        num_ground_truth_basins: Number of ground-truth basins
        num_model_basins: Number of model basin blocks

    Returns:
        Tuple of (accuracy, confusion_matrix, best_mapping)
    """
    # Build confusion matrix: [num_gt_basins, num_model_basins]
    confusion = torch.zeros(num_ground_truth_basins, num_model_basins)

    for activations, gt_basin in zip(activations_list, ground_truth_basins):
        # Count how often each model basin is active for this ground-truth basin
        active_basins = activations.active_basin  # [T]
        for model_basin in active_basins:
            confusion[gt_basin, model_basin.item()] += 1

    # Normalize rows to get probabilities
    row_sums = confusion.sum(dim=1, keepdim=True)
    confusion_normalized = confusion / (row_sums + 1e-8)

    # Find best mapping: for each ground-truth basin, which model basin is most common?
    # This is a greedy assignment (not optimal Hungarian, but usually good enough)
    best_mapping = {}
    used_model_basins = set()

    # Sort ground-truth basins by their count (process most common first)
    gt_counts = [(gt, confusion[gt].sum().item()) for gt in range(num_ground_truth_basins)]
    gt_counts.sort(key=lambda x: -x[1])

    for gt_basin, _ in gt_counts:
        # Find best unused model basin for this ground-truth basin
        row = confusion[gt_basin].clone()
        for used in used_model_basins:
            row[used] = -1  # Exclude already used

        best_model_basin = row.argmax().item()
        best_mapping[gt_basin] = best_model_basin
        used_model_basins.add(best_model_basin)

    # Compute accuracy using best mapping
    correct = 0
    total = 0

    for activations, gt_basin in zip(activations_list, ground_truth_basins):
        expected_model_basin = best_mapping[gt_basin]
        correct += (activations.active_basin == expected_model_basin).sum().item()
        total += len(activations.active_basin)

    accuracy = correct / total if total > 0 else 0.0

    return accuracy, confusion_normalized, best_mapping


def compute_temporal_consistency(
    activations_list: List[BasinActivations],
) -> float:
    """
    Compute temporal consistency: fraction of timesteps where active basin = mode.

    For each trajectory, find the mode (most common active basin), then compute
    what fraction of timesteps match that mode.
    """
    consistencies = []

    for activations in activations_list:
        active = activations.active_basin  # [T]

        # Find mode
        counts = torch.bincount(active)
        mode = counts.argmax().item()

        # Fraction matching mode
        consistency = (active == mode).float().mean().item()
        consistencies.append(consistency)

    return sum(consistencies) / len(consistencies) if consistencies else 0.0


def compute_within_basin_similarity(
    activations_list: List[BasinActivations],
    ground_truth_basins: List[int],
) -> float:
    """
    Compute average cosine similarity of basin activations within same ground-truth basin.
    """
    # Group activations by ground-truth basin
    basin_activations = {}
    for activations, gt_basin in zip(activations_list, ground_truth_basins):
        if gt_basin not in basin_activations:
            basin_activations[gt_basin] = []
        # Use mean activation over trajectory
        mean_activation = activations.basin_norms.mean(dim=0)  # [B]
        basin_activations[gt_basin].append(mean_activation)

    # Compute pairwise cosine similarity within each basin
    similarities = []
    for gt_basin, acts in basin_activations.items():
        if len(acts) < 2:
            continue
        acts_tensor = torch.stack(acts)  # [N, B]
        # Normalize
        acts_norm = F.normalize(acts_tensor, dim=-1)
        # Pairwise cosine similarity
        sim_matrix = acts_norm @ acts_norm.T
        # Extract upper triangle (excluding diagonal)
        n = len(acts)
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(sim_matrix[i, j].item())

    return sum(similarities) / len(similarities) if similarities else 0.0


def compute_cross_basin_separation(
    activations_list: List[BasinActivations],
    ground_truth_basins: List[int],
) -> float:
    """
    Compute average L2 distance between basin activations from different ground-truth basins.
    """
    # Group activations by ground-truth basin
    basin_activations = {}
    for activations, gt_basin in zip(activations_list, ground_truth_basins):
        if gt_basin not in basin_activations:
            basin_activations[gt_basin] = []
        mean_activation = activations.basin_norms.mean(dim=0)  # [B]
        basin_activations[gt_basin].append(mean_activation)

    # Compute mean activation per basin
    basin_means = {}
    for gt_basin, acts in basin_activations.items():
        basin_means[gt_basin] = torch.stack(acts).mean(dim=0)

    # Compute pairwise distances between basin means
    distances = []
    basins = list(basin_means.keys())
    for i, b1 in enumerate(basins):
        for b2 in basins[i + 1:]:
            dist = torch.norm(basin_means[b1] - basin_means[b2]).item()
            distances.append(dist)

    return sum(distances) / len(distances) if distances else 0.0


# ---------------------------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------------------------


class BasinStructureAnalyzer:
    """Analyzer for basin structure correspondence."""

    def __init__(
        self,
        model: StructuredLISTAKM,
        dataset: BasinLabeledDataset,
        device: str = 'cpu',
    ):
        self.model = model
        self.dataset = dataset
        self.device = device

        self.activations_list: List[BasinActivations] = []
        self.ground_truth_basins: List[int] = []

    def compute_all_activations(self):
        """Compute activations for all trajectories in dataset."""
        print(f"Computing activations for {len(self.dataset)} trajectories...")

        self.activations_list = []
        self.ground_truth_basins = []

        for i, traj_data in enumerate(self.dataset.trajectories):
            activations = compute_basin_activations(
                self.model, traj_data.states, self.device
            )
            self.activations_list.append(activations)
            self.ground_truth_basins.append(traj_data.final_basin)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(self.dataset)} trajectories")

    def run_full_analysis(self) -> AnalysisResults:
        """Run full analysis and return results."""
        if not self.activations_list:
            self.compute_all_activations()

        num_gt_basins = self.dataset.num_basins
        num_model_basins = self.model.num_basins

        print("\nComputing metrics...")

        # Basin assignment accuracy
        accuracy, confusion_matrix, best_mapping = compute_basin_assignment_accuracy(
            self.activations_list,
            self.ground_truth_basins,
            num_gt_basins,
            num_model_basins,
        )
        print(f"  Basin assignment accuracy: {accuracy:.4f}")
        print(f"  Best mapping (GT -> Model): {best_mapping}")

        # Temporal consistency
        temporal_consistency = compute_temporal_consistency(self.activations_list)
        print(f"  Temporal consistency: {temporal_consistency:.4f}")

        # Mean activation entropy
        all_entropies = torch.cat([a.activation_entropy for a in self.activations_list])
        mean_entropy = all_entropies.mean().item()
        print(f"  Mean activation entropy: {mean_entropy:.4f}")

        # Within-basin similarity
        within_sim = compute_within_basin_similarity(
            self.activations_list, self.ground_truth_basins
        )
        print(f"  Within-basin similarity: {within_sim:.4f}")

        # Cross-basin separation
        cross_sep = compute_cross_basin_separation(
            self.activations_list, self.ground_truth_basins
        )
        print(f"  Cross-basin separation: {cross_sep:.4f}")

        # Per-basin accuracy
        per_basin_accuracy = {}
        for gt_basin in range(num_gt_basins):
            expected_model_basin = best_mapping.get(gt_basin)
            if expected_model_basin is None:
                continue

            correct = 0
            total = 0
            for activations, gt in zip(self.activations_list, self.ground_truth_basins):
                if gt == gt_basin:
                    correct += (activations.active_basin == expected_model_basin).sum().item()
                    total += len(activations.active_basin)

            per_basin_accuracy[gt_basin] = correct / total if total > 0 else 0.0

        return AnalysisResults(
            system_name=self.dataset.system,
            num_trajectories=len(self.dataset),
            num_ground_truth_basins=num_gt_basins,
            num_model_basins=num_model_basins,
            basin_assignment_accuracy=accuracy,
            temporal_consistency=temporal_consistency,
            mean_activation_entropy=mean_entropy,
            within_basin_similarity=within_sim,
            cross_basin_separation=cross_sep,
            confusion_matrix=confusion_matrix.tolist(),
            per_basin_accuracy=per_basin_accuracy,
        )


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def plot_phase_portrait_basin_comparison(
    dataset: BasinLabeledDataset,
    activations_list: List[BasinActivations],
    best_mapping: Dict[int, int],
    output_path: Optional[Path] = None,
    max_trajectories: int = 20,
):
    """
    Create phase portrait showing ground-truth vs predicted basin assignments.

    Side-by-side plots:
    - Left: trajectory colored by ground-truth basin
    - Right: same trajectory colored by predicted basin
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    num_gt_basins = dataset.num_basins
    num_model_basins = max(best_mapping.values()) + 1 if best_mapping else 1

    gt_colors = get_basin_colors(num_gt_basins, 'Set1')
    model_colors = get_basin_colors(num_model_basins, 'Set2')

    # Limit number of trajectories for clarity
    num_to_plot = min(max_trajectories, len(dataset))

    for i in range(num_to_plot):
        traj_data = dataset.trajectories[i]
        activations = activations_list[i]

        states = traj_data.states.numpy()
        gt_basin = traj_data.final_basin
        active_basins = activations.active_basin.numpy()

        # Left plot: ground-truth coloring
        axes[0].plot(
            states[:, 0], states[:, 1],
            color=gt_colors[gt_basin],
            alpha=0.5,
            linewidth=0.8,
        )

        # Right plot: predicted coloring (scatter to show per-timestep)
        # Use segments to color by active basin
        for t in range(len(states) - 1):
            axes[1].plot(
                states[t:t+2, 0], states[t:t+2, 1],
                color=model_colors[active_basins[t] % len(model_colors)],
                alpha=0.5,
                linewidth=0.8,
            )

    # Add attractor positions
    if hasattr(dataset, 'attractor_positions'):
        for ax in axes:
            ax.scatter(
                dataset.attractor_positions[:, 0].numpy(),
                dataset.attractor_positions[:, 1].numpy(),
                marker='*', s=200, c='black', zorder=10,
                label='Attractors'
            )

    axes[0].set_title('Ground-Truth Basin Coloring')
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')

    axes[1].set_title('Predicted (Active Model Basin) Coloring')
    axes[1].set_xlabel('x1')
    axes[1].set_ylabel('x2')

    # Add legends
    from matplotlib.lines import Line2D
    gt_legend = [
        Line2D([0], [0], color=gt_colors[i], label=f'GT Basin {i}')
        for i in range(min(num_gt_basins, 10))
    ]
    axes[0].legend(handles=gt_legend, loc='upper right', fontsize=8)

    model_legend = [
        Line2D([0], [0], color=model_colors[i], label=f'Model Basin {i}')
        for i in range(min(num_model_basins, 10))
    ]
    axes[1].legend(handles=model_legend, loc='upper right', fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved phase portrait to {output_path}")

    plt.close()


def plot_basin_norm_timeseries(
    activations: BasinActivations,
    ground_truth_basin: int,
    title: str = None,
    output_path: Optional[Path] = None,
    max_basins_to_show: int = 10,
):
    """
    Plot L2 norms of all basin blocks over time for a single trajectory.
    """
    basin_norms = activations.basin_norms.numpy()  # [T, B]
    T, B = basin_norms.shape

    # Only show top basins by mean activation
    mean_norms = basin_norms.mean(axis=0)
    top_basins = np.argsort(-mean_norms)[:max_basins_to_show]

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = get_basin_colors(len(top_basins))

    for i, basin_idx in enumerate(top_basins):
        ax.plot(
            basin_norms[:, basin_idx],
            label=f'Basin {basin_idx}',
            color=colors[i],
            alpha=0.8,
        )

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Basin Block L2 Norm')
    ax.set_title(title or f'Basin Norms Over Time (GT Basin: {ground_truth_basin})')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved basin norm timeseries to {output_path}")

    plt.close()


def plot_confusion_matrix(
    confusion_matrix: torch.Tensor,
    ground_truth_labels: List[str],
    num_model_basins: int,
    output_path: Optional[Path] = None,
):
    """
    Heatmap showing correspondence between ground-truth and model basins.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Only show model basins with non-trivial activation
    cm = confusion_matrix.numpy() if isinstance(confusion_matrix, torch.Tensor) else np.array(confusion_matrix)

    # Find which model basins have any activation
    active_model_basins = np.where(cm.sum(axis=0) > 0.01)[0]
    if len(active_model_basins) == 0:
        active_model_basins = np.arange(min(10, cm.shape[1]))

    cm_subset = cm[:, active_model_basins]

    im = ax.imshow(cm_subset, cmap='Blues', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fraction of Timesteps')

    # Labels
    ax.set_xticks(np.arange(len(active_model_basins)))
    ax.set_xticklabels([f'Model {i}' for i in active_model_basins], rotation=45, ha='right')
    ax.set_yticks(np.arange(len(ground_truth_labels)))
    ax.set_yticklabels(ground_truth_labels)

    ax.set_xlabel('Predicted Model Basin')
    ax.set_ylabel('Ground-Truth Basin')
    ax.set_title('Basin Confusion Matrix')

    # Add text annotations
    for i in range(cm_subset.shape[0]):
        for j in range(cm_subset.shape[1]):
            val = cm_subset[i, j]
            if val > 0.01:
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {output_path}")

    plt.close()


def plot_activation_distributions(
    activations_list: List[BasinActivations],
    ground_truth_basins: List[int],
    num_gt_basins: int,
    output_path: Optional[Path] = None,
    max_model_basins: int = 10,
):
    """
    For each ground-truth basin, show distribution of model basin activations.
    """
    # Group by ground-truth basin
    basin_norms_by_gt = {i: [] for i in range(num_gt_basins)}

    for activations, gt_basin in zip(activations_list, ground_truth_basins):
        # Mean over time
        mean_norms = activations.basin_norms.mean(dim=0).numpy()  # [B]
        basin_norms_by_gt[gt_basin].append(mean_norms)

    # Find most active model basins overall
    all_norms = np.concatenate([
        np.stack(norms) for norms in basin_norms_by_gt.values() if norms
    ])
    mean_all = all_norms.mean(axis=0)
    top_model_basins = np.argsort(-mean_all)[:max_model_basins]

    # Create violin plot
    fig, axes = plt.subplots(1, num_gt_basins, figsize=(4 * num_gt_basins, 5), sharey=True)
    if num_gt_basins == 1:
        axes = [axes]

    for gt_basin, ax in enumerate(axes):
        if not basin_norms_by_gt[gt_basin]:
            continue

        data = np.stack(basin_norms_by_gt[gt_basin])[:, top_model_basins]  # [N, top_k]

        positions = np.arange(len(top_model_basins))
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)

        ax.set_xticks(positions)
        ax.set_xticklabels([f'M{i}' for i in top_model_basins], rotation=45)
        ax.set_xlabel('Model Basin')
        ax.set_title(f'GT Basin {gt_basin}')

        if gt_basin == 0:
            ax.set_ylabel('Mean Basin Norm')

    plt.suptitle('Model Basin Activations by Ground-Truth Basin')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved activation distributions to {output_path}")

    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate basin structure correspondence in StructuredLISTAKM'
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained StructuredLISTAKM checkpoint')
    parser.add_argument('--system', type=str, default='duffing',
                        choices=['duffing', 'lyapunov'],
                        help='Dynamical system to evaluate on')
    parser.add_argument('--num_trajectories', type=int, default=100,
                        help='Number of test trajectories to generate')
    parser.add_argument('--trajectory_length', type=int, default=500,
                        help='Length of each trajectory')
    parser.add_argument('--output_dir', type=str, default='results/basin_analysis',
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

    # Override environment
    cfg.ENV.ENV_NAME = args.system

    # Create model
    print("Creating model...")
    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    # Verify it's a StructuredLISTAKM
    if not isinstance(model, StructuredLISTAKM):
        raise ValueError(
            f"Model is {type(model).__name__}, expected StructuredLISTAKM. "
            "This evaluation only works with StructuredLISTAKM models."
        )

    print(f"Model: {model.num_basins} model basins, "
          f"d_global={model.d_global}, d_basin={model.d_basin}")

    # Generate dataset
    dataset = BasinLabeledDataset(
        system=args.system,
        cfg=cfg,
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        seed=args.seed,
    )

    # Run analysis
    analyzer = BasinStructureAnalyzer(model, dataset, device=args.device)
    results = analyzer.run_full_analysis()

    # Get best mapping for visualizations
    _, confusion_matrix, best_mapping = compute_basin_assignment_accuracy(
        analyzer.activations_list,
        analyzer.ground_truth_basins,
        dataset.num_basins,
        model.num_basins,
    )

    # Save results
    results_path = output_dir / 'analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\nSaved analysis results to {results_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Phase portrait comparison
    plot_phase_portrait_basin_comparison(
        dataset,
        analyzer.activations_list,
        best_mapping,
        output_path=output_dir / 'phase_portrait_comparison.png',
    )

    # 2. Confusion matrix
    plot_confusion_matrix(
        confusion_matrix,
        dataset.basin_names,
        model.num_basins,
        output_path=output_dir / 'confusion_matrix.png',
    )

    # 3. Basin norm timeseries (for a few example trajectories)
    for i in range(min(5, len(dataset))):
        plot_basin_norm_timeseries(
            analyzer.activations_list[i],
            dataset.trajectories[i].final_basin,
            title=f'Trajectory {i} (GT Basin: {dataset.trajectories[i].final_basin})',
            output_path=output_dir / f'basin_norms_traj_{i}.png',
        )

    # 4. Activation distributions
    plot_activation_distributions(
        analyzer.activations_list,
        analyzer.ground_truth_basins,
        dataset.num_basins,
        output_path=output_dir / 'activation_distributions.png',
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BASIN STRUCTURE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"System: {results.system_name}")
    print(f"Trajectories: {results.num_trajectories}")
    print(f"Ground-truth basins: {results.num_ground_truth_basins}")
    print(f"Model basins: {results.num_model_basins}")
    print("-" * 60)
    print(f"Basin Assignment Accuracy: {results.basin_assignment_accuracy:.4f}")
    print(f"Temporal Consistency: {results.temporal_consistency:.4f}")
    print(f"Mean Activation Entropy: {results.mean_activation_entropy:.4f}")
    print(f"Within-Basin Similarity: {results.within_basin_similarity:.4f}")
    print(f"Cross-Basin Separation: {results.cross_basin_separation:.4f}")
    print("-" * 60)
    print("Per-Basin Accuracy:")
    for gt_basin, acc in results.per_basin_accuracy.items():
        basin_name = dataset.basin_names[gt_basin] if gt_basin < len(dataset.basin_names) else f"Basin {gt_basin}"
        print(f"  {basin_name}: {acc:.4f}")
    print("=" * 60)
    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()
