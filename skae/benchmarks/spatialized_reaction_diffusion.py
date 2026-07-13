"""Spatialized multibasin reaction-diffusion benchmark utilities.

This module implements the appendix protocol's PDE path: lift a two-dimensional
multibasin source system into a two-channel field ``u(x, y, t)``, add periodic
diffusion, and save evaluation-only basin labels. Training consumers should use
only ``fields`` and ``split_indices``; all basin metadata is for post hoc
evaluation and auditing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class SpatialReactionDiffusionConfig:
    source_system: str = "cal_square_4"
    grid_size: int = 32
    diffusion: float = 0.01
    rk4_dt: float = 0.01
    substeps_per_observation: int = 10
    trajectory_length: int = 24
    label_extra_observations: int = 24
    train_trajectories: int = 48
    val_trajectories: int = 12
    test_trajectories: int = 12
    seed: int = 0
    spatial_extent: float = 1.0
    laplacian_scaling: str = "continuum"
    min_regions: int = 2
    max_regions: int = 3
    mask_temperature: float = 0.65
    low_frequency_cutoff: int = 3
    noise_scale: float = 0.03
    require_min_area_fraction: float = 0.08
    max_initial_condition_attempts: int = 32
    clip_value: float = 8.0
    allen_cahn_beta: float = 8.0
    allen_cahn_reaction_strength: float = 1.0
    allen_cahn_center_radius: float = 1.5


class MultiwellAllenCahnSystem:
    """Vector-valued multiwell Allen-Cahn local reaction system.

    The spatialized PDE is ``u_t = diffusion * Laplacian(u) - grad W(u)``.
    Here ``W`` is a smooth soft-minimum of quadratic wells placed on a circle
    in the two-dimensional order-parameter plane. This keeps the existing
    two-channel field/evaluation layout while giving a canonical multistable
    phase-field benchmark with known wells for post-hoc labeling only.
    """

    def __init__(
        self,
        *,
        num_wells: int,
        beta: float = 8.0,
        reaction_strength: float = 1.0,
        center_radius: float = 1.5,
    ) -> None:
        if int(num_wells) < 2:
            raise ValueError("Multiwell Allen-Cahn requires at least two wells.")
        self.num_wells = int(num_wells)
        self.beta = float(beta)
        self.reaction_strength = float(reaction_strength)
        self.center_radius = float(center_radius)
        angles = torch.linspace(0.0, 2.0 * torch.pi, self.num_wells + 1)[:-1]
        self.centers = self.center_radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        self.name = f"allen_cahn_{self.num_wells}"
        self.description = (
            "Vector-valued multiwell Allen-Cahn reaction with smooth soft-min "
            "quadratic wells on a circle; basin labels are nearest-well labels "
            "computed only for evaluation."
        )

    def dynamics(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim < 2 or state.shape[0] != 2:
            raise ValueError("Allen-Cahn state must have leading channel dimension 2.")
        centers = self.centers.to(device=state.device, dtype=state.dtype)
        view_shape = (self.num_wells, 2) + (1,) * (state.ndim - 1)
        diff = state.unsqueeze(0) - centers.reshape(view_shape)
        dist2 = diff.square().sum(dim=1)
        weights = torch.softmax(-float(self.beta) * dist2, dim=0)
        grad = (weights.unsqueeze(1) * diff).sum(dim=0)
        return -float(self.reaction_strength) * grad


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_allen_cahn_source(source_system: str) -> Optional[int]:
    normalized = source_system.lower().replace("-", "_").replace(":", "_")
    match = re.fullmatch(r"(?:multiwell_)?allen_cahn(?:_?)(\d+)?", normalized)
    if match is None:
        return None
    return int(match.group(1) or 4)


def get_source_system(source_system: str, cfg: SpatialReactionDiffusionConfig | None = None):
    """Instantiate a source system from the Claude transition-rich catalog."""

    allen_cahn_wells = _parse_allen_cahn_source(source_system)
    if allen_cahn_wells is not None:
        return MultiwellAllenCahnSystem(
            num_wells=allen_cahn_wells,
            beta=float(cfg.allen_cahn_beta) if cfg is not None else 8.0,
            reaction_strength=float(cfg.allen_cahn_reaction_strength) if cfg is not None else 1.0,
            center_radius=float(cfg.allen_cahn_center_radius) if cfg is not None else 1.5,
        )

    system_name = source_system.split(":", 1)[1] if source_system.startswith("claude:") else source_system
    from skae.claude_catalog import ensure_catalog_registered, get_system

    ensure_catalog_registered()
    return get_system(system_name)


def _centers_from_sequence(values: Any) -> Optional[torch.Tensor]:
    if isinstance(values, torch.Tensor):
        if values.ndim == 2 and values.shape[1] >= 2:
            return values[:, :2].detach().cpu().to(dtype=torch.float32)
        return None
    if isinstance(values, (list, tuple)) and values:
        centers: List[Tuple[float, float]] = []
        for item in values:
            if isinstance(item, torch.Tensor):
                flat = item.detach().cpu().flatten()
                if flat.numel() < 2:
                    return None
                centers.append((float(flat[0]), float(flat[1])))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                centers.append((float(item[0]), float(item[1])))
            else:
                return None
        return torch.tensor(centers, dtype=torch.float32)
    return None


def extract_attractor_centers(system: Any) -> torch.Tensor:
    """Best-effort extraction of source-system attractor centers.

    The smoke path intentionally supports the Gaussian-well systems listed in
    the appendix first. Labels are evaluation-only and are not needed by the
    training scripts.
    """

    for attr_name in (
        "centers",
        "_wells",
        "wells",
        "well_centers",
        "room_centers",
        "basins",
    ):
        if hasattr(system, attr_name):
            centers = _centers_from_sequence(getattr(system, attr_name))
            if centers is not None and centers.shape[0] >= 2:
                return centers

    raise ValueError(
        f"Could not infer attractor centers for source system '{getattr(system, 'name', system)}'. "
        "Use a catalog system with wells/centers metadata for the first PDE smoke run."
    )


def _low_frequency_noise(
    num_fields: int,
    grid_size: int,
    cutoff: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate smooth low-frequency random fields using Fourier truncation."""

    noise = torch.randn(num_fields, grid_size, grid_size, generator=generator)
    spectrum = torch.fft.fft2(noise)
    freqs = torch.fft.fftfreq(grid_size) * float(grid_size)
    keep = (
        freqs[:, None].square() + freqs[None, :].square()
    ).sqrt() <= float(max(1, cutoff))
    spectrum = spectrum * keep.unsqueeze(0).to(spectrum.device)
    smooth = torch.fft.ifft2(spectrum).real
    smooth = smooth - smooth.mean(dim=(-2, -1), keepdim=True)
    smooth = smooth / smooth.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return smooth.to(dtype=torch.float32)


def _source_dynamics(system: Any, field: torch.Tensor) -> torch.Tensor:
    state = field.movedim(-1, 0).to(dtype=torch.float64)
    rhs = system.dynamics(state).movedim(0, -1)
    return rhs.to(device=field.device, dtype=field.dtype)


def _periodic_laplacian(field: torch.Tensor) -> torch.Tensor:
    return (
        torch.roll(field, shifts=1, dims=0)
        + torch.roll(field, shifts=-1, dims=0)
        + torch.roll(field, shifts=1, dims=1)
        + torch.roll(field, shifts=-1, dims=1)
        - 4.0 * field
    )


def _laplacian_scale(cfg: SpatialReactionDiffusionConfig) -> float:
    """Return the finite-difference scale for the configured Laplacian.

    ``continuum`` treats the grid as a discretization of ``[0, spatial_extent]^2``
    and applies the usual ``1 / dx^2`` factor. ``graph`` keeps the older smoke
    convention where the unscaled periodic graph Laplacian is used.
    """

    mode = str(cfg.laplacian_scaling).lower()
    if mode in {"continuum", "dx", "finite_difference"}:
        dx = float(cfg.spatial_extent) / float(int(cfg.grid_size))
        return 1.0 / max(dx * dx, 1e-12)
    if mode in {"graph", "unscaled", "legacy"}:
        return 1.0
    raise ValueError("laplacian_scaling must be 'continuum' or 'graph'.")


def _pde_rhs(system: Any, field: torch.Tensor, cfg: SpatialReactionDiffusionConfig) -> torch.Tensor:
    return _source_dynamics(system, field) + float(cfg.diffusion) * _laplacian_scale(cfg) * _periodic_laplacian(field)


def _rk4_pde_step(system: Any, field: torch.Tensor, cfg: SpatialReactionDiffusionConfig) -> torch.Tensor:
    dt = float(cfg.rk4_dt)
    k1 = _pde_rhs(system, field, cfg)
    k2 = _pde_rhs(system, field + 0.5 * dt * k1, cfg)
    k3 = _pde_rhs(system, field + 0.5 * dt * k2, cfg)
    k4 = _pde_rhs(system, field + dt * k3, cfg)
    return field + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _sample_initial_condition(
    centers: torch.Tensor,
    cfg: SpatialReactionDiffusionConfig,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_centers = int(centers.shape[0])
    max_regions = min(int(cfg.max_regions), num_centers)
    min_regions = min(max(1, int(cfg.min_regions)), max_regions)

    last_field: Optional[torch.Tensor] = None
    last_indices: Optional[torch.Tensor] = None
    last_area_fractions: Optional[torch.Tensor] = None
    for _ in range(max(1, int(cfg.max_initial_condition_attempts))):
        if min_regions == max_regions:
            num_regions = min_regions
        else:
            num_regions = int(
                torch.randint(
                    low=min_regions,
                    high=max_regions + 1,
                    size=(1,),
                    generator=generator,
                ).item()
            )
        selected = torch.randperm(num_centers, generator=generator)[:num_regions]
        logits = _low_frequency_noise(
            num_regions,
            int(cfg.grid_size),
            int(cfg.low_frequency_cutoff),
            generator,
        )
        weights = torch.softmax(logits / max(float(cfg.mask_temperature), 1e-6), dim=0)
        field = torch.einsum("rxy,rc->xyc", weights, centers[selected])
        if float(cfg.noise_scale) > 0.0:
            eps = _low_frequency_noise(2, int(cfg.grid_size), int(cfg.low_frequency_cutoff) + 1, generator)
            field = field + float(cfg.noise_scale) * eps.movedim(0, -1)

        owner = weights.argmax(dim=0)
        counts = torch.bincount(owner.reshape(-1), minlength=num_regions).float()
        area_fractions = counts / float(cfg.grid_size * cfg.grid_size)
        last_field = field.to(dtype=torch.float32)
        last_indices = selected.to(dtype=torch.int64)
        last_area_fractions = area_fractions.to(dtype=torch.float32)
        if torch.all(area_fractions >= float(cfg.require_min_area_fraction)):
            return last_field, last_indices, last_area_fractions

    assert last_field is not None and last_indices is not None and last_area_fractions is not None
    return last_field, last_indices, last_area_fractions


def basin_map_from_field(field: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Assign each pixel to its nearest attractor center."""

    distances = torch.sum((field.unsqueeze(-2) - centers.view(1, 1, -1, 2)) ** 2, dim=-1)
    return distances.argmin(dim=-1).to(dtype=torch.int64)


def modal_basin_and_fraction(basin_map: torch.Tensor, num_basins: int) -> Tuple[int, float]:
    counts = torch.bincount(basin_map.reshape(-1), minlength=int(num_basins))
    modal = int(counts.argmax().item())
    fraction = float(counts.max().item() / max(1, basin_map.numel()))
    return modal, fraction


def simulate_trajectory(
    system: Any,
    centers: torch.Tensor,
    cfg: SpatialReactionDiffusionConfig,
    generator: torch.Generator,
) -> Dict[str, torch.Tensor]:
    """Simulate one spatialized trajectory and compute evaluation labels."""

    field, selected_indices, initial_area_fractions = _sample_initial_condition(centers, cfg, generator)
    frames = [field.clone()]
    invalid_value_count = 0
    clipped_value_count = 0
    for _ in range(int(cfg.trajectory_length)):
        for _substep in range(int(cfg.substeps_per_observation)):
            field = _rk4_pde_step(system, field, cfg)
            invalid_value_count += int((~torch.isfinite(field)).sum().item())
            field = torch.nan_to_num(field, nan=0.0, posinf=float(cfg.clip_value), neginf=-float(cfg.clip_value))
            clipped_value_count += int((field.abs() > float(cfg.clip_value)).sum().item())
            field = torch.clamp(field, -float(cfg.clip_value), float(cfg.clip_value))
        frames.append(field.clone())

    trajectory = torch.stack(frames, dim=0).to(dtype=torch.float32)
    observed_final_basin_map = basin_map_from_field(trajectory[-1], centers)
    observed_global_basin, observed_majority_fraction = modal_basin_and_fraction(
        observed_final_basin_map,
        centers.shape[0],
    )

    label_field = field.clone()
    for _ in range(max(0, int(cfg.label_extra_observations))):
        for _substep in range(int(cfg.substeps_per_observation)):
            label_field = _rk4_pde_step(system, label_field, cfg)
            invalid_value_count += int((~torch.isfinite(label_field)).sum().item())
            label_field = torch.nan_to_num(
                label_field,
                nan=0.0,
                posinf=float(cfg.clip_value),
                neginf=-float(cfg.clip_value),
            )
            clipped_value_count += int((label_field.abs() > float(cfg.clip_value)).sum().item())
            label_field = torch.clamp(label_field, -float(cfg.clip_value), float(cfg.clip_value))

    label_basin_map = basin_map_from_field(label_field, centers)
    global_basin, majority_fraction = modal_basin_and_fraction(label_basin_map, centers.shape[0])
    return {
        "fields": trajectory,
        "basin_map": label_basin_map,
        "observed_final_basin_map": observed_final_basin_map,
        "global_basin_label": torch.tensor(global_basin, dtype=torch.int64),
        "majority_fraction": torch.tensor(majority_fraction, dtype=torch.float32),
        "observed_global_basin_label": torch.tensor(observed_global_basin, dtype=torch.int64),
        "observed_majority_fraction": torch.tensor(observed_majority_fraction, dtype=torch.float32),
        "selected_center_indices": selected_indices,
        "initial_area_fractions": initial_area_fractions,
        "invalid_value_count": torch.tensor(invalid_value_count, dtype=torch.int64),
        "clipped_value_count": torch.tensor(clipped_value_count, dtype=torch.int64),
    }


def generate_dataset(cfg: SpatialReactionDiffusionConfig) -> Dict[str, Any]:
    """Generate a complete train/val/test smoke dataset."""

    if int(cfg.grid_size) < 4:
        raise ValueError("grid_size must be at least 4.")
    if int(cfg.trajectory_length) < 2:
        raise ValueError("trajectory_length must be at least 2.")
    if int(cfg.substeps_per_observation) < 1:
        raise ValueError("substeps_per_observation must be at least 1.")

    system = get_source_system(cfg.source_system, cfg)
    centers = extract_attractor_centers(system)
    counts = {
        "train": int(cfg.train_trajectories),
        "val": int(cfg.val_trajectories),
        "test": int(cfg.test_trajectories),
    }
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("At least one trajectory is required.")

    fields: List[torch.Tensor] = []
    basin_maps: List[torch.Tensor] = []
    observed_basin_maps: List[torch.Tensor] = []
    global_labels: List[torch.Tensor] = []
    majority_fractions: List[torch.Tensor] = []
    observed_global_labels: List[torch.Tensor] = []
    observed_majority_fractions: List[torch.Tensor] = []
    selected_indices: List[torch.Tensor] = []
    initial_area_fractions: List[torch.Tensor] = []
    invalid_value_counts: List[torch.Tensor] = []
    clipped_value_counts: List[torch.Tensor] = []
    split_indices: Dict[str, List[int]] = {"train": [], "val": [], "test": []}

    trajectory_index = 0
    for split_name, split_count in counts.items():
        for _ in range(split_count):
            generator = torch.Generator().manual_seed(int(cfg.seed) + 10_000 * trajectory_index)
            item = simulate_trajectory(system, centers, cfg, generator)
            fields.append(item["fields"])
            basin_maps.append(item["basin_map"])
            observed_basin_maps.append(item["observed_final_basin_map"])
            global_labels.append(item["global_basin_label"])
            majority_fractions.append(item["majority_fraction"])
            observed_global_labels.append(item["observed_global_basin_label"])
            observed_majority_fractions.append(item["observed_majority_fraction"])
            selected_indices.append(item["selected_center_indices"])
            initial_area_fractions.append(item["initial_area_fractions"])
            invalid_value_counts.append(item["invalid_value_count"])
            clipped_value_counts.append(item["clipped_value_count"])
            split_indices[split_name].append(trajectory_index)
            trajectory_index += 1

    max_selected = max(int(x.numel()) for x in selected_indices)
    selected_padded = torch.full((total, max_selected), -1, dtype=torch.int64)
    area_padded = torch.full((total, max_selected), float("nan"), dtype=torch.float32)
    for idx, selected in enumerate(selected_indices):
        selected_padded[idx, : selected.numel()] = selected
        area_padded[idx, : initial_area_fractions[idx].numel()] = initial_area_fractions[idx]

    metadata = {
        **asdict(cfg),
        "format": "spatialized_reaction_diffusion_v1",
        "source_system_name": getattr(system, "name", cfg.source_system),
        "source_description": getattr(system, "description", ""),
        "state_layout": "[trajectory, time, x, y, channel]",
        "stored_dt": float(cfg.rk4_dt) * int(cfg.substeps_per_observation),
        "label_dt": float(cfg.rk4_dt) * int(cfg.substeps_per_observation),
        "label_extra_time": float(cfg.rk4_dt) * int(cfg.substeps_per_observation) * int(cfg.label_extra_observations),
        "label_horizon_time": float(cfg.rk4_dt)
        * int(cfg.substeps_per_observation)
        * (int(cfg.trajectory_length) + int(cfg.label_extra_observations)),
        "laplacian_scale": _laplacian_scale(cfg),
        "spatial_step": float(cfg.spatial_extent) / float(int(cfg.grid_size)),
        "labels_are_evaluation_only": True,
        "training_label_policy": "No basin labels, basin counts, or selected center indices are used by training.",
    }
    return {
        "fields": torch.stack(fields, dim=0).contiguous(),
        "basin_maps": torch.stack(basin_maps, dim=0).contiguous(),
        "observed_final_basin_maps": torch.stack(observed_basin_maps, dim=0).contiguous(),
        "global_basin_labels": torch.stack(global_labels, dim=0),
        "majority_fractions": torch.stack(majority_fractions, dim=0),
        "observed_global_basin_labels": torch.stack(observed_global_labels, dim=0),
        "observed_majority_fractions": torch.stack(observed_majority_fractions, dim=0),
        "selected_center_indices": selected_padded,
        "initial_area_fractions": area_padded,
        "invalid_value_counts": torch.stack(invalid_value_counts, dim=0),
        "clipped_value_counts": torch.stack(clipped_value_counts, dim=0),
        "attractor_centers": centers.contiguous(),
        "split_indices": {key: torch.tensor(value, dtype=torch.int64) for key, value in split_indices.items()},
        "metadata": metadata,
    }


def save_dataset(bundle: Dict[str, Any], path: str | Path) -> None:
    """Save a dataset bundle.

    ``.pt`` is dependency-free and is the smoke default. ``.h5``/``.hdf5`` is
    supported when h5py is installed, preserving the appendix layout.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "HDF5 output requires h5py, which is not currently a project dependency. "
                "Use a .pt output path for the smoke run or install h5py before requesting HDF5."
            ) from exc
        with h5py.File(path, "w") as handle:
            for key in (
                "fields",
                "basin_maps",
                "observed_final_basin_maps",
                "global_basin_labels",
                "majority_fractions",
                "observed_global_basin_labels",
                "observed_majority_fractions",
                "selected_center_indices",
                "initial_area_fractions",
                "invalid_value_counts",
                "clipped_value_counts",
                "attractor_centers",
            ):
                handle.create_dataset(key, data=bundle[key].cpu().numpy())
            split_group = handle.create_group("split_indices")
            for key, value in bundle["split_indices"].items():
                split_group.create_dataset(key, data=value.cpu().numpy())
            handle.attrs["metadata_json"] = json.dumps(bundle["metadata"], sort_keys=True)
        return

    torch.save(bundle, path)


def load_dataset(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Reading HDF5 datasets requires h5py.") from exc
        with h5py.File(path, "r") as handle:
            bundle = {
                "fields": torch.from_numpy(handle["fields"][()]).float(),
                "basin_maps": torch.from_numpy(handle["basin_maps"][()]).long(),
                "observed_final_basin_maps": torch.from_numpy(handle["observed_final_basin_maps"][()]).long()
                if "observed_final_basin_maps" in handle
                else torch.from_numpy(handle["basin_maps"][()]).long(),
                "global_basin_labels": torch.from_numpy(handle["global_basin_labels"][()]).long(),
                "majority_fractions": torch.from_numpy(handle["majority_fractions"][()]).float(),
                "observed_global_basin_labels": torch.from_numpy(handle["observed_global_basin_labels"][()]).long()
                if "observed_global_basin_labels" in handle
                else torch.from_numpy(handle["global_basin_labels"][()]).long(),
                "observed_majority_fractions": torch.from_numpy(handle["observed_majority_fractions"][()]).float()
                if "observed_majority_fractions" in handle
                else torch.from_numpy(handle["majority_fractions"][()]).float(),
                "selected_center_indices": torch.from_numpy(handle["selected_center_indices"][()]).long(),
                "initial_area_fractions": torch.from_numpy(handle["initial_area_fractions"][()]).float()
                if "initial_area_fractions" in handle
                else torch.empty(0),
                "invalid_value_counts": torch.from_numpy(handle["invalid_value_counts"][()]).long()
                if "invalid_value_counts" in handle
                else torch.zeros(handle["fields"].shape[0], dtype=torch.int64),
                "clipped_value_counts": torch.from_numpy(handle["clipped_value_counts"][()]).long()
                if "clipped_value_counts" in handle
                else torch.zeros(handle["fields"].shape[0], dtype=torch.int64),
                "attractor_centers": torch.from_numpy(handle["attractor_centers"][()]).float(),
                "split_indices": {
                    key: torch.from_numpy(value[()]).long()
                    for key, value in handle["split_indices"].items()
                },
                "metadata": json.loads(handle.attrs["metadata_json"]),
            }
            return bundle
    return _with_backward_compatible_fields(_torch_load(path))


def _with_backward_compatible_fields(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Populate fields added after the first smoke dataset format."""

    if "observed_final_basin_maps" not in bundle and "basin_maps" in bundle:
        bundle["observed_final_basin_maps"] = bundle["basin_maps"]
    if "observed_global_basin_labels" not in bundle and "global_basin_labels" in bundle:
        bundle["observed_global_basin_labels"] = bundle["global_basin_labels"]
    if "observed_majority_fractions" not in bundle and "majority_fractions" in bundle:
        bundle["observed_majority_fractions"] = bundle["majority_fractions"]
    if "initial_area_fractions" not in bundle:
        selected = bundle.get("selected_center_indices")
        if isinstance(selected, torch.Tensor):
            bundle["initial_area_fractions"] = torch.full(selected.shape, float("nan"), dtype=torch.float32)
        else:
            bundle["initial_area_fractions"] = torch.empty(0, dtype=torch.float32)
    if "invalid_value_counts" not in bundle and "fields" in bundle:
        bundle["invalid_value_counts"] = torch.zeros(bundle["fields"].shape[0], dtype=torch.int64)
    if "clipped_value_counts" not in bundle and "fields" in bundle:
        bundle["clipped_value_counts"] = torch.zeros(bundle["fields"].shape[0], dtype=torch.int64)
    return bundle


def split_fields(bundle: Dict[str, Any], split: str) -> torch.Tensor:
    indices = bundle["split_indices"][split]
    return bundle["fields"][indices]


def flatten_fields(fields: torch.Tensor) -> torch.Tensor:
    """Convert [trajectory, time, x, y, channel] fields to [trajectory, time, obs]."""

    if fields.ndim != 5 or fields.shape[-1] != 2:
        raise ValueError("Expected fields with shape [trajectory, time, x, y, 2].")
    return fields.reshape(fields.shape[0], fields.shape[1], -1).contiguous()


def reshape_flat_fields(flat: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Convert [batch, horizon, obs] or [batch, obs] flattened fields back to images."""

    if flat.ndim == 2:
        return flat.reshape(flat.shape[0], int(grid_size), int(grid_size), 2)
    if flat.ndim == 3:
        return flat.reshape(flat.shape[0], flat.shape[1], int(grid_size), int(grid_size), 2)
    raise ValueError("Expected flattened fields with rank 2 or 3.")


def spatial_gradient(field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Periodic first differences for [*, x, y, channel] fields."""

    grad_x = torch.roll(field, shifts=-1, dims=-3) - field
    grad_y = torch.roll(field, shifts=-1, dims=-2) - field
    return grad_x, grad_y


def field_modal_basin_labels(fields: torch.Tensor, centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return modal nearest-center labels and majority fractions for final fields."""

    labels: List[int] = []
    fractions: List[float] = []
    for field in fields:
        basin_map = basin_map_from_field(field, centers)
        label, fraction = modal_basin_and_fraction(basin_map, centers.shape[0])
        labels.append(label)
        fractions.append(fraction)
    return torch.tensor(labels, dtype=torch.int64), torch.tensor(fractions, dtype=torch.float32)
