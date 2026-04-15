"""Configuration system using Python dataclasses for PyTorch implementation.

This module provides a type-safe, native Python configuration system using dataclasses.
No external dependencies required (no ml_collections, no argparse conversion).

## Usage

### 1. Quick Start - Use Predefined Configs

```python
from skae.config import get_config

# Get a named configuration
cfg = get_config("lista")  # Options: "generic", "generic_sparse", "lista", etc.

# Access nested fields with dot notation
print(cfg.MODEL.TARGET_SIZE)  # 2048
print(cfg.TRAIN.LR)  # 0.001
```

### 2. Modify Configs

```python
# Start with a base config and customize
cfg = get_config("generic")
cfg.MODEL.TARGET_SIZE = 128
cfg.TRAIN.BATCH_SIZE = 512
cfg.TRAIN.LR = 5e-4
cfg.ENV.ENV_NAME = "pendulum"
```

### 3. Create Custom Configs

```python
from skae.config import Config, ModelConfig, TrainConfig

# Build from scratch
cfg = Config()
cfg.SEED = 42
cfg.MODEL.TARGET_SIZE = 256
cfg.TRAIN.NUM_STEPS = 5000

# Or use nested dataclasses
cfg = Config(
    SEED=42,
    MODEL=ModelConfig(TARGET_SIZE=256, SPARSITY_COEFF=0.01),
    TRAIN=TrainConfig(NUM_STEPS=5000, LR=1e-3)
)
```

### 4. Save and Load Configs

```python
# Save to JSON for reproducibility
cfg.to_json("experiments/run_001/config.json")

# Load from JSON
cfg = Config.from_json("experiments/run_001/config.json")

# Convert to dict for logging (e.g., wandb)
config_dict = cfg.to_dict()
```

### 5. Use in Training Code

```python
from skae.config import get_config

cfg = get_config("lista")

# Pass config sections directly to your model/trainer
model = build_model(cfg.MODEL)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
dataloader = create_dataloader(cfg.ENV, cfg.TRAIN)

# Use loss coefficients
loss = (cfg.MODEL.RES_COEFF * alignment_loss + 
        cfg.MODEL.RECONST_COEFF * recon_loss +
        cfg.MODEL.SPARSITY_COEFF * sparsity_loss)
```

### 6. Notebook-Friendly

```python
# In Jupyter/IPython notebooks
cfg = get_config("lista")
cfg.TRAIN.NUM_STEPS = 1000  # Quick experiment
cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 5  # Faster iterations

# Configs are just Python objects - easy to inspect
cfg.MODEL  # Shows all model settings
```

## Available Configs

- `"default"`: Base configuration with minimal settings
- `"generic"`: Standard KoopmanAE with MLP encoder (64-dim latent)
- `"generic_sparse"`: KoopmanAE with L1 sparsity regularization
- `"generic_no_shrink"`: KoopmanAE MLP control with no sparsity penalty and no zero-inducing ReLU gate
- `"generic_prediction"`: Prediction-focused (no reconstruction)
- `"lista"`: LISTA-based sparse autoencoder (2048-dim latent)
- `"lista_nonlinear"`: LISTA with nonlinear MLP encoder
- `"hyperlista"`: LISTAKM with HyperLISTA encoder mode
- `"lista_parity_generic_sparse"`: LISTA preset aligned to `generic_sparse` non-depth settings

## Structure

```
Config
├── SEED: int
├── ENV: EnvConfig
│   ├── ENV_NAME: str (system choice)
│   └── <SYSTEM>: SystemConfig (dt, params)
├── MODEL: ModelConfig
│   ├── MODEL_NAME: str
│   ├── TARGET_SIZE: int (latent dim)
│   ├── Loss coefficients (RES_COEFF, RECONST_COEFF, etc.)
│   ├── ENCODER: EncoderConfig
│   │   ├── LAYERS: List[int]
│   │   ├── ACTIVATION, USE_BIAS, etc.
│   │   └── LISTA: ListaConfig (for LISTAKM models)
│   └── DECODER: DecoderConfig
└── TRAIN: TrainConfig
    ├── NUM_STEPS: int (epochs)
    ├── BATCH_SIZE: int
    ├── LR: float
    └── DATA_SIZE: int
```
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class ParabolicConfig:
    """Parabolic attractor system parameters."""
    LAMBDA: float = -1.0
    MU: float = -0.1
    DT: float = 0.1


@dataclass
class DuffingConfig:
    """Duffing oscillator system parameters."""
    DT: float = 0.01


@dataclass
class PendulumConfig:
    """Pendulum system parameters."""
    DT: float = 0.01


@dataclass
class LotkaVolterraConfig:
    """Lotka-Volterra system parameters."""
    DT: float = 0.01


@dataclass
class Lorenz63Config:
    """Lorenz63 system parameters."""
    DT: float = 0.01


@dataclass
class LyapunovConfig:
    """Lyapunov multi-attractor system parameters."""
    DT: float = 0.05
    SIGMA: float = 0.5
    DIM: int = 2  # state dimension
    NUM_BASINS: int = 13  # number of attractor centers
    POINTS_MODE: str = "fixed"  # "fixed" (use canonical 2D grid) or "random"
    CENTER_SCALE: float = 2.0  # range for random centers (uniform in [-scale, scale])
    MIN_SEPARATION: float = 0.6  # minimum separation for random centers
    INIT_RANGE: float = 2.5  # reset range for initial states (uniform in [-r, r])
    EXTEND_MODE: str = "embed"  # "embed" (2D Lyapunov + linear decay dims) or "full"
    EXTRA_DECAY: float = 1.0  # decay for extra dims in embed mode


@dataclass
class BlendedConfig:
    """Three-basin blended linear system parameters."""
    DT: float = 0.05
    SIGMA: float = 1.5


@dataclass
class MultiWellConfig:
    """Multi-well transition system parameters."""
    DT: float = 0.02
    SIGMA: float = 0.7
    DIM: int = 2  # state dimension
    NUM_BASINS: int = 5  # number of attractor centers
    POINTS_MODE: str = "fixed"  # "fixed" (canonical 5-point layout) or "random"
    CENTER_SCALE: float = 2.0  # range for random centers (uniform in [-scale, scale])
    MIN_SEPARATION: float = 0.6  # minimum separation for random centers
    INIT_RANGE: float = 2.5  # reset range for initial states (uniform in [-r, r])
    EXTEND_MODE: str = "embed"  # "embed" (2D core + linear decay dims) or "full"
    EXTRA_DECAY: float = 1.0  # decay for extra dims in embed mode
    DYNAMICS_MODE: str = "gradient"  # ["gradient", "rotational", "energy", "strong_transition"]
    ALPHA: float = 0.6  # rotational coupling for "rotational"
    BETA: float = 1.2  # radial energy injection for "energy"
    STRONG_ALPHA: float = 0.8  # rotational coupling for "strong_transition"
    STRONG_BETA: float = 1.0  # radial coupling for "strong_transition"


@dataclass
class KuramotoConfig:
    """Coupled Kuramoto oscillator system parameters."""
    DT: float = 0.05
    NUM_OSCILLATORS: int = 16
    K_COUPLING: float = 4.0
    TOPOLOGY: str = "ring"  # ["ring", "all_to_all"]
    OMEGA_MODE: str = "identical"  # ["identical", "uniform_spread", "random"]
    OMEGA_SPREAD: float = 0.5


@dataclass
class HopfieldConfig:
    """Continuous Hopfield network parameters."""
    DT: float = 0.05
    NUM_NEURONS: int = 16
    NUM_PATTERNS: int = 4
    BETA: float = 1.0
    TAU: float = 1.0
    # "orthogonal" = QR-derived decorrelated pattern directions, then sign-binarized to +/-1.
    PATTERN_MODE: str = "random"  # ["random", "orthogonal"]
    INIT_SCALE: float = 1.0


@dataclass
class CompetitiveLVConfig:
    """Competitive N-species Lotka-Volterra parameters."""
    DT: float = 0.01
    NUM_SPECIES: int = 10
    INTERACTION_MODE: str = "symmetric"  # ["symmetric", "asymmetric", "block_diagonal"]
    R_MODE: str = "uniform"  # ["uniform", "heterogeneous"]
    INTERACTION_SCALE: float = 0.70
    POSITIVITY_CLIP: bool = True
    SURVIVAL_THRESHOLD: float = 1e-3
    INIT_MIN: float = 0.05
    INIT_MAX: float = 1.5


@dataclass
class GatedLocalLinearConfig:
    """Configuration for the native gated local-linear transition-rich toy."""

    DT: float = 0.04


@dataclass
class GatedTransferLinearConfig:
    """Configuration for the native gated transfer transition-rich toy."""

    DT: float = 0.04


@dataclass
class DystsConfig:
    """Configuration for dysts-based environments.
    
    The dysts library provides 135+ chaotic systems. Use ENV_NAME="dysts:SystemName"
    (e.g., "dysts:Lorenz", "dysts:Chua") to select a dysts system.
    
    See benchmarks/system_catalog.py for curated system lists.
    """
    SYSTEM_NAME: str = "Lorenz"  # Name matching dysts.flows class
    DT_OVERRIDE: float = 0.0  # If > 0, override dysts default dt
    # Scale for perturbation noise around default initial condition.
    # IMPORTANT: this should be relatively small for most dysts systems; large values
    # can throw trajectories far off-attractor and make phase portraits look "wrong".
    # Kept aligned with benchmarks.dysts_adapter.DystsEnv default.
    IC_NOISE_SCALE: float = 0.2
    STANDARDIZE: bool = False  # Whether to standardize trajectories
    # Use dysts's period-based resampling for native trajectories.
    # NOTE: Resampling changes the effective time spacing between points and can
    # make comparisons against one-step RK4 rollouts misleading. Keep this False
    # unless you *really* intend to train/evaluate on the resampled map.
    RESAMPLE: bool = False
    PTS_PER_PERIOD: int = 100  # Points per period if resampling
    USE_NATIVE_CACHE: bool = False  # Precompute native dysts trajectories for training
    CACHE_STEPS: int = 30000  # Length of each cached trajectory
    CACHE_TRAJECTORIES: int = 200  # Number of cached trajectories
    CACHE_WARMUP: int = 200  # Discard initial steps to skip transients
    CACHE_DIR: str = ""  # Optional on-disk cache directory for trajectory reuse
    CACHE_REUSE: bool = False  # Reuse/load/save on-disk cached trajectories
    CACHE_SPLIT: str = "train"  # Cache namespace: train/val/test
    CACHE_NUM_WORKERS: int = 1  # Parallel workers for cache construction


@dataclass
class ClaudeCatalogConfig:
    """Configuration for Claude transition-rich catalog environments.

    Use ENV_NAME="claude:SystemName" to select a registered catalog system.
    DT <= 0 keeps the system's intrinsic default timestep.
    """

    DT: float = 0.0


@dataclass
class EnvConfig:
    """Environment configuration.
    
    For built-in environments, set ENV_NAME to one of:
        [
            "duffing", "parabolic", "pendulum", "lotka_volterra", "lorenz63",
            "lyapunov", "blended", "multiwell", "multiwell:<mode>",
            "gated_local_linear", "gated_transfer_linear",
            "kuramoto", "hopfield", "competitive_lv", "claude:SystemName"
        ]
    
    For Claude catalog systems, set ENV_NAME to
    "claude:SystemName" (e.g., "claude:cal_triangle_3").

    For dysts systems, set ENV_NAME to
    "dysts:SystemName" (e.g., "dysts:Lorenz", "dysts:Chua").
    The DYSTS config section can be used to customize dysts system behavior.
    """
    ENV_NAME: str = "duffing"
    PARABOLIC: ParabolicConfig = field(default_factory=ParabolicConfig)
    DUFFING: DuffingConfig = field(default_factory=DuffingConfig)
    PENDULUM: PendulumConfig = field(default_factory=PendulumConfig)
    LOTKA_VOLTERRA: LotkaVolterraConfig = field(default_factory=LotkaVolterraConfig)
    LORENZ63: Lorenz63Config = field(default_factory=Lorenz63Config)
    LYAPUNOV: LyapunovConfig = field(default_factory=LyapunovConfig)
    BLENDED: BlendedConfig = field(default_factory=BlendedConfig)
    MULTIWELL: MultiWellConfig = field(default_factory=MultiWellConfig)
    GATED_LOCAL_LINEAR: GatedLocalLinearConfig = field(default_factory=GatedLocalLinearConfig)
    GATED_TRANSFER_LINEAR: GatedTransferLinearConfig = field(default_factory=GatedTransferLinearConfig)
    KURAMOTO: KuramotoConfig = field(default_factory=KuramotoConfig)
    HOPFIELD: HopfieldConfig = field(default_factory=HopfieldConfig)
    COMPETITIVE_LV: CompetitiveLVConfig = field(default_factory=CompetitiveLVConfig)
    CLAUDE_CATALOG: ClaudeCatalogConfig = field(default_factory=ClaudeCatalogConfig)
    DYSTS: DystsConfig = field(default_factory=DystsConfig)


@dataclass
class ListaConfig:
    """LISTA encoder-specific configuration."""
    NUM_LOOPS: int = 5  # LISTA iterations
    USE_MOMENTUM: bool = False  # add heavy-ball momentum between LISTA refinement steps
    MOMENTUM_BETA: float = 0.0  # fixed LISTA momentum coefficient when USE_MOMENTUM is enabled
    L: float = 1e3  # Lipschitz constant estimate
    ALPHA: float = 0.1  # sparsity threshold
    LINEAR_ENCODER: bool = False  # use MLP vs linear encoder
    FINAL_OP: str = "relu"  # final nonlinearity in LISTA loop: ["shrink", "relu", "sign_split"]
    GROUP_SHRINKAGE: bool = False  # apply sparse-group shrinkage over inferred latent groups
    GROUP_THRESHOLD_SCALE: float = 1.0  # group threshold multiplier relative to elementwise threshold
    TOPK_GROUPS: int = 0  # keep only the top-k groups before within-group thresholding (0 disables)
    PRECODE_MODE: str = "auto"  # ["auto", "free_mlp", "linear", "dictionary_tied", "hybrid"]
    PRECODE_RESIDUAL_SCALE: float = 0.1  # residual MLP scale for hybrid pre-code
    ADAPTIVE_THRESHOLDS: bool = False  # use per-sample thresholds based on residual/prior mismatch
    ALPHA_RESIDUAL_COEFF: float = 0.0  # coefficient on reconstruction residual term in adaptive threshold
    ALPHA_PRIOR_COEFF: float = 0.0  # coefficient on latent-prior mismatch term in adaptive threshold
    GROUPWISE_THRESHOLDS: bool = False  # learn separate base thresholds per inferred latent group


@dataclass
class HyperListaConfig:
    """HyperLISTA encoder-specific configuration.
    
    HyperLISTA uses analytically-derived weights from the decoder dictionary D,
    reducing learnable parameters from O(n² × L) to just 3 scalar hyperparameters.
    
    See: "Hyperparameter Tuning is All You Need for LISTA" (Chen et al., NeurIPS 2021)
    """
    NUM_LOOPS: int = 5              # Number of unrolled iterations
    C_THETA: float = 5e-3            # Threshold scaling hyperparameter (c₁)
    C_BETA: float = 5e-3             # Momentum hyperparameter (c₂)
    C_SS: float = 0.5                # Support selection hyperparameter (c₃)
    C_THETA_MIN: float = 1e-6        # Minimum threshold scale when constraining c_theta
    CONSTRAIN_C_THETA: bool = True   # Reparameterize c_theta to stay strictly positive
    USE_SUPPORT_SELECTION: bool = True  # Enable adaptive support selection
    USE_MOMENTUM: bool = True        # Enable momentum acceleration
    MAG_RATIO: float = 0.1           # Threshold for support size approximation
    LEARN_HYPERPARAMS: bool = True   # Whether c_theta, c_beta, c_ss are learnable
    PINV_CACHE_MODE: str = "none"    # Pseudo-inverse caching mode; "none" recomputes every forward
    GROUP_SHRINKAGE: bool = False    # apply sparse-group shrinkage over inferred latent groups
    GROUP_THRESHOLD_SCALE: float = 1.0  # group threshold multiplier relative to adaptive theta
    TOPK_GROUPS: int = 0             # keep only the top-k groups before within-group thresholding (0 disables)


@dataclass
class StructuredLatentConfig:
    """Configuration for structured latent space partitioning.

    Enables basin-aware Koopman dynamics with:
    - Global dynamics block (always active, shared physics)
    - B basin blocks (mutually exclusive, local linear dynamics)

    Total latent dim = D_GLOBAL + NUM_BASINS * D_BASIN
    """
    ENABLED: bool = False              # Enable structured latent space
    D_GLOBAL: int = 8                  # Dimension of global dynamics block
    NUM_BASINS: int = 20               # Number of basin slots (B)
    D_BASIN: int = 8                   # Dimension of each basin block
    LAMBDA_GLOBAL: float = 1e-4        # Sparsity weight for global block (near-zero)
    LAMBDA_LOCAL: float = 1e-3         # Sparsity weight for basin blocks
    LAMBDA_EXCLUSIVITY: float = 1e-2   # Final exclusivity penalty weight (pairwise)
    LAMBDA_ENTROPY: float = 0.0        # Entropy-based exclusivity penalty (low entropy = 1 dominant basin)
    LAMBDA_DOMINANCE: float = 0.0      # Top-1 dominance loss (penalize non-max basins)
    LAMBDA_SPARSITY: float = 1e-3      # Explicit L1 sparsity weight on full z
    LAMBDA_TEMPORAL: float = 0.0       # Temporal consistency loss weight (sequence training only)
    EXCL_WARMUP_STEPS: int = 1000      # Steps to ramp exclusivity/sparsity from 0 to final


@dataclass
class BlockLossConfig:
    """Loss configuration for block activation behavior in block_diagonal K.

    These losses encourage per-sample single-block activation and
    balanced block usage across the batch without relying on basin labels.
    """
    ENABLED: bool = False
    ONE_BLOCK_LOSS: str = "none"       # ["none", "low_entropy", "pairwise_overlap", "top1_margin"]
    ONE_BLOCK_WEIGHT: float = 0.0
    TOP1_MARGIN: float = 0.1
    BALANCE_LOSS: str = "none"         # ["none", "usage_entropy", "kl_uniform"]
    BALANCE_WEIGHT: float = 0.0
    ENERGY_NORM: str = "l2"            # ["l1", "l2"]
    EPS: float = 1e-8


@dataclass
class SoftBlockConfig:
    """Penalty configuration for soft block-sparse dense Koopman matrices."""

    ENABLED: bool = False
    NUM_BLOCKS: int = 0                # exact number of near-equal latent blocks
    NORM: str = "l1"                   # ["l1", "fro"]
    WEIGHT: float = 0.0


@dataclass
class EncoderConfig:
    """Encoder architecture configuration."""
    ENCODER_TYPE: str = "lista"  # from ["lista", "hyperlista"]
    LAYERS: List[int] = field(default_factory=lambda: [16, 16])  # hidden layer sizes
    LAST_RELU: bool = False
    USE_BIAS: bool = False
    ACTIVATION: str = "relu"  # from ["relu", "tanh", "gelu"]
    LISTA: ListaConfig = field(default_factory=ListaConfig)
    HYPERLISTA: HyperListaConfig = field(default_factory=HyperListaConfig)


@dataclass
class DecoderConfig:
    """Decoder architecture configuration."""
    LAYERS: List[int] = field(default_factory=list)  # linear decoder by default
    USE_BIAS: bool = False
    ACTIVATION: str = "relu"
    AFFINE_BIAS: bool = False  # Add learnable bias to decoder output (useful for LISTA)


@dataclass
class ModelConfig:
    """Model architecture and loss configuration."""
    MODEL_NAME: str = "SparseKM"  # from ["GenericKM", "SparseKM", "LISTAKM", "StructuredLISTAKM"]
    NORM_FN: str = "id"  # from ["id", "ball"]
    TARGET_SIZE: int = 16  # latent_dim i.e. zdim
    OBS_LOSS_DIM_NORMALIZATION: str = "sqrt_dim"  # ["none", "sqrt_dim", "dim"]
    
    # Loss coefficients
    RES_COEFF: float = 1.0  # alignment loss weight
    RECONST_COEFF: float = 0.02  # reconstruction loss weight
    PRED_COEFF: float = 0.0  # prediction loss weight
    SPARSITY_COEFF: float = 1e-3  # sparsity loss weight (L1 regularization)
    HOMOGENEOUS_COEFF: float = 1.0  # homogeneous coordinate consistency loss weight
    DECODER_COHERENCE_WEIGHT: float = 0.0  # weight on normalized dictionary off-diagonal Gram penalty
    
    # Homogeneous coordinates: append 1 to input, enables implicit bias learning
    USE_HOMOGENEOUS: bool = False

    # Koopman matrix structure: "dense" (full NxN), "diagonal", "block_diagonal"
    K_STRUCTURE: str = "dense"
    K_BLOCK_SIZE: int = 0  # block size for block_diagonal (0 = auto: target_size // 13)
    K_NUM_BLOCKS: int = 0  # exact number of blocks for block_diagonal (0 = disabled)

    # Sub-configs
    ENCODER: EncoderConfig = field(default_factory=EncoderConfig)
    DECODER: DecoderConfig = field(default_factory=DecoderConfig)
    STRUCTURED: StructuredLatentConfig = field(default_factory=StructuredLatentConfig)
    BLOCK_LOSS: BlockLossConfig = field(default_factory=BlockLossConfig)
    SOFT_BLOCK: SoftBlockConfig = field(default_factory=SoftBlockConfig)


@dataclass
class TrainConfig:
    """Training configuration."""
    @dataclass
    class HardInitOversampleConfig:
        """Training-only oversampling of hard initial conditions near separatrices.

        The hardness score is computed without basin labels from two black-box
        signals that are available for every environment in this branch:
        sensitivity to small perturbations and lingering late-time transient
        motion over a short probe rollout.
        """

        ENABLED: bool = False
        FRACTION: float = 0.5
        POOL_SIZE: int = 1024
        NUM_CANDIDATES: int = 4096
        PROBE_STEPS: int = 32
        NUM_PERTURBATIONS: int = 4
        PERTURB_SCALE: float = 0.04
        TRANSIENT_WINDOW: int = 8
        TRANSIENT_WEIGHT: float = 0.5
        JITTER_SCALE: float = 0.25
        BUILD_CHUNK_SIZE: int = 128

    NUM_STEPS: int = 2_000  # total training steps (epochs)
    BATCH_SIZE: int = 256
    DATA_SIZE: int = 256 * 8  # total dataset size
    LR: float = 1e-4  # main learning rate (encoder/decoder)
    WEIGHT_DECAY: float = 1e-4  # weight decay for AdamW optimizer
    K_MATRIX_LR: float = 1e-5  # learning rate for Koopman matrix parameters
    
    # Lightweight evaluation during training
    EVAL_EVERY: int = 500  # evaluation interval in training steps
    EVAL_NUM_STEPS: int = 200  # rollout horizon for the quick eval() helper
    
    # Unified horizon-based training parameter.
    # H=1 matches former pairwise behavior.
    SEQUENCE_LENGTH: int = 1
    HARD_INIT_OVERSAMPLE: HardInitOversampleConfig = field(
        default_factory=HardInitOversampleConfig
    )

@dataclass
class Config:
    """Main configuration container."""
    SEED: int = 0
    ENV: EnvConfig = field(default_factory=EnvConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    TRAIN: TrainConfig = field(default_factory=TrainConfig)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: str) -> None:
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> Config:
        """Create config from dictionary."""
        # Recursively construct nested dataclasses
        env_dict = config_dict.get("ENV", {})
        env = EnvConfig(
            ENV_NAME=env_dict.get("ENV_NAME", "duffing"),
            PARABOLIC=ParabolicConfig(**env_dict.get("PARABOLIC", {})),
            DUFFING=DuffingConfig(**env_dict.get("DUFFING", {})),
            PENDULUM=PendulumConfig(**env_dict.get("PENDULUM", {})),
            LOTKA_VOLTERRA=LotkaVolterraConfig(**env_dict.get("LOTKA_VOLTERRA", {})),
            LORENZ63=Lorenz63Config(**env_dict.get("LORENZ63", {})),
            LYAPUNOV=LyapunovConfig(**env_dict.get("LYAPUNOV", {})),
            BLENDED=BlendedConfig(**env_dict.get("BLENDED", {})),
            MULTIWELL=MultiWellConfig(**env_dict.get("MULTIWELL", {})),
            GATED_LOCAL_LINEAR=GatedLocalLinearConfig(**env_dict.get("GATED_LOCAL_LINEAR", {})),
            GATED_TRANSFER_LINEAR=GatedTransferLinearConfig(**env_dict.get("GATED_TRANSFER_LINEAR", {})),
            KURAMOTO=KuramotoConfig(**env_dict.get("KURAMOTO", {})),
            HOPFIELD=HopfieldConfig(**env_dict.get("HOPFIELD", {})),
            COMPETITIVE_LV=CompetitiveLVConfig(**env_dict.get("COMPETITIVE_LV", {})),
            CLAUDE_CATALOG=ClaudeCatalogConfig(**env_dict.get("CLAUDE_CATALOG", {})),
            DYSTS=DystsConfig(**env_dict.get("DYSTS", {})),
        )
        
        model_dict = config_dict.get("MODEL", {})
        encoder_dict = model_dict.get("ENCODER", {})
        lista = ListaConfig(**encoder_dict.get("LISTA", {}))
        hyperlista = HyperListaConfig(**encoder_dict.get("HYPERLISTA", {}))
        encoder = EncoderConfig(**{k: v for k, v in encoder_dict.items() if k not in ["LISTA", "HYPERLISTA"]})
        encoder.LISTA = lista
        encoder.HYPERLISTA = hyperlista
        decoder = DecoderConfig(**model_dict.get("DECODER", {}))
        
        structured = StructuredLatentConfig(**model_dict.get("STRUCTURED", {}))
        block_loss = BlockLossConfig(**model_dict.get("BLOCK_LOSS", {}))
        soft_block = SoftBlockConfig(**model_dict.get("SOFT_BLOCK", {}))
        model = ModelConfig(**{
            k: v
            for k, v in model_dict.items()
            if k not in ["ENCODER", "DECODER", "STRUCTURED", "BLOCK_LOSS", "SOFT_BLOCK"]
        })
        model.ENCODER = encoder
        model.DECODER = decoder
        model.STRUCTURED = structured
        model.BLOCK_LOSS = block_loss
        model.SOFT_BLOCK = soft_block
        
        train_dict = config_dict.get("TRAIN", {})
        train = TrainConfig(
            **{k: v for k, v in train_dict.items() if k != "HARD_INIT_OVERSAMPLE"}
        )
        train.HARD_INIT_OVERSAMPLE = TrainConfig.HardInitOversampleConfig(
            **train_dict.get("HARD_INIT_OVERSAMPLE", {})
        )
        
        return cls(
            SEED=config_dict.get("SEED", 0),
            ENV=env,
            MODEL=model,
            TRAIN=train
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> Config:
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


_BUILTIN_ENV_DT_NAMES = {
    "parabolic",
    "duffing",
    "pendulum",
    "lotka_volterra",
    "lorenz63",
    "lyapunov",
    "blended",
    "multiwell",
    "gated_local_linear",
    "gated_transfer_linear",
    "kuramoto",
    "hopfield",
    "competitive_lv",
    "claude_catalog",
}


def canonical_env_name(env_name: str) -> str:
    """Normalize an environment name for config-level routing."""
    lowered = env_name.lower()
    if lowered.startswith("claude:"):
        return "claude_catalog"
    if lowered.startswith("dysts:"):
        return "dysts"
    if lowered.startswith("multiwell:"):
        return "multiwell"
    if lowered.startswith("multiwell_"):
        return "multiwell"
    if lowered in _BUILTIN_ENV_DT_NAMES:
        return lowered
    # Unknown names are treated as dysts because make_env() falls back to Dysts.
    return "dysts"


def _get_claude_catalog_default_dt(env_name: str) -> float:
    """Read the intrinsic dt for a Claude catalog environment name."""
    if not env_name.lower().startswith("claude:"):
        raise ValueError(f"Expected 'claude:SystemName', got '{env_name}'.")
    system_name = env_name.split(":", 1)[1]
    from skae.claude_catalog import ensure_catalog_registered, get_system

    ensure_catalog_registered()
    return float(get_system(system_name).dt)


def apply_env_dt_override(cfg: Config, dt: float, env_name: Optional[str] = None) -> None:
    """Apply a timestep override to the active environment config."""
    if dt <= 0.0:
        raise ValueError("Environment dt override must be positive.")

    target_env = canonical_env_name(env_name or cfg.ENV.ENV_NAME)
    dt = float(dt)

    if target_env == "dysts":
        cfg.ENV.DYSTS.DT_OVERRIDE = dt
    elif target_env == "parabolic":
        cfg.ENV.PARABOLIC.DT = dt
    elif target_env == "duffing":
        cfg.ENV.DUFFING.DT = dt
    elif target_env == "pendulum":
        cfg.ENV.PENDULUM.DT = dt
    elif target_env == "lotka_volterra":
        cfg.ENV.LOTKA_VOLTERRA.DT = dt
    elif target_env == "lorenz63":
        cfg.ENV.LORENZ63.DT = dt
    elif target_env == "lyapunov":
        cfg.ENV.LYAPUNOV.DT = dt
    elif target_env == "blended":
        cfg.ENV.BLENDED.DT = dt
    elif target_env == "multiwell":
        cfg.ENV.MULTIWELL.DT = dt
    elif target_env == "gated_local_linear":
        cfg.ENV.GATED_LOCAL_LINEAR.DT = dt
    elif target_env == "gated_transfer_linear":
        cfg.ENV.GATED_TRANSFER_LINEAR.DT = dt
    elif target_env == "kuramoto":
        cfg.ENV.KURAMOTO.DT = dt
    elif target_env == "hopfield":
        cfg.ENV.HOPFIELD.DT = dt
    elif target_env == "competitive_lv":
        cfg.ENV.COMPETITIVE_LV.DT = dt
    elif target_env == "claude_catalog":
        cfg.ENV.CLAUDE_CATALOG.DT = dt
    else:
        raise ValueError(f"Unsupported environment for dt override: '{env_name or cfg.ENV.ENV_NAME}'")


def get_env_dt(cfg: Config, env_name: Optional[str] = None) -> float:
    """Read the configured timestep for the active environment."""
    target_env = canonical_env_name(env_name or cfg.ENV.ENV_NAME)

    if target_env == "dysts":
        if cfg.ENV.DYSTS.DT_OVERRIDE > 0.0:
            return float(cfg.ENV.DYSTS.DT_OVERRIDE)
        return 0.0
    if target_env == "parabolic":
        return float(cfg.ENV.PARABOLIC.DT)
    if target_env == "duffing":
        return float(cfg.ENV.DUFFING.DT)
    if target_env == "pendulum":
        return float(cfg.ENV.PENDULUM.DT)
    if target_env == "lotka_volterra":
        return float(cfg.ENV.LOTKA_VOLTERRA.DT)
    if target_env == "lorenz63":
        return float(cfg.ENV.LORENZ63.DT)
    if target_env == "lyapunov":
        return float(cfg.ENV.LYAPUNOV.DT)
    if target_env == "blended":
        return float(cfg.ENV.BLENDED.DT)
    if target_env == "multiwell":
        return float(cfg.ENV.MULTIWELL.DT)
    if target_env == "gated_local_linear":
        return float(cfg.ENV.GATED_LOCAL_LINEAR.DT)
    if target_env == "gated_transfer_linear":
        return float(cfg.ENV.GATED_TRANSFER_LINEAR.DT)
    if target_env == "kuramoto":
        return float(cfg.ENV.KURAMOTO.DT)
    if target_env == "hopfield":
        return float(cfg.ENV.HOPFIELD.DT)
    if target_env == "competitive_lv":
        return float(cfg.ENV.COMPETITIVE_LV.DT)
    if target_env == "claude_catalog":
        if cfg.ENV.CLAUDE_CATALOG.DT > 0.0:
            return float(cfg.ENV.CLAUDE_CATALOG.DT)
        return _get_claude_catalog_default_dt(env_name or cfg.ENV.ENV_NAME)
    raise ValueError(f"Unsupported environment for dt lookup: '{env_name or cfg.ENV.ENV_NAME}'")


def get_default_config() -> Config:
    """Create default configuration.
    
    Returns:
        Config with default settings.
    """
    return Config()


def get_train_generic_km_config() -> Config:
    """Training configuration for GenericKM (standard Koopman AE with MLP encoder)."""
    cfg = Config()
    cfg.TRAIN.LR = 1e-4
    cfg.MODEL.MODEL_NAME = "GenericKM"
    cfg.MODEL.TARGET_SIZE = 64
    cfg.MODEL.NORM_FN = "id"
    cfg.MODEL.DECODER.LAYERS = []
    cfg.MODEL.ENCODER.LAYERS = [64, 64]
    cfg.MODEL.SPARSITY_COEFF = 0.0
    return cfg


def get_train_generic_sparse_config() -> Config:
    """Training configuration for GenericKM with L1 regularization."""
    cfg = Config()
    cfg.TRAIN.LR = 1e-4
    cfg.MODEL.MODEL_NAME = "GenericKM"
    cfg.MODEL.TARGET_SIZE = 64
    cfg.MODEL.NORM_FN = "id"
    cfg.MODEL.DECODER.LAYERS = []
    cfg.MODEL.ENCODER.LAYERS = [64, 64]
    cfg.MODEL.ENCODER.LAST_RELU = True
    cfg.MODEL.ENCODER.USE_BIAS = True
    cfg.MODEL.RECONST_COEFF = 0.5
    cfg.MODEL.SPARSITY_COEFF = 0.01
    return cfg


def get_train_generic_no_shrink_config() -> Config:
    """Training configuration for GenericKM without architectural sparsification.

    This control removes the latent L1 penalty and also avoids the sparse
    preset's ReLU output gate so zero activity is not induced by architecture.
    """
    cfg = Config()
    cfg.TRAIN.LR = 1e-4
    cfg.MODEL.MODEL_NAME = "GenericKM"
    cfg.MODEL.TARGET_SIZE = 64
    cfg.MODEL.NORM_FN = "id"
    cfg.MODEL.DECODER.LAYERS = []
    cfg.MODEL.ENCODER.LAYERS = [64, 64]
    cfg.MODEL.ENCODER.ACTIVATION = "tanh"
    cfg.MODEL.ENCODER.LAST_RELU = False
    cfg.MODEL.ENCODER.USE_BIAS = True
    cfg.MODEL.RECONST_COEFF = 0.5
    cfg.MODEL.SPARSITY_COEFF = 0.0
    return cfg


def get_train_generic_prediction_config() -> Config:
    """Training configuration for prediction-focused KoopmanAE."""
    cfg = Config()
    cfg.MODEL.MODEL_NAME = "GenericKM"
    cfg.TRAIN.LR = 1e-3
    cfg.MODEL.DECODER.LAYERS = []
    cfg.MODEL.PRED_COEFF = 1.0
    cfg.MODEL.RES_COEFF = 0.0
    cfg.MODEL.RECONST_COEFF = 0.0
    cfg.MODEL.SPARSITY_COEFF = 0.0
    return cfg


def get_train_lista_config() -> Config:
    """Configuration for LISTA-based Sparse KM."""
    cfg = Config()
    cfg.MODEL.MODEL_NAME = "LISTAKM"
    cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = True
    cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 5
    cfg.MODEL.TARGET_SIZE = 1024 * 2
    cfg.MODEL.RES_COEFF = 1.0
    cfg.MODEL.RECONST_COEFF = 1.0
    cfg.MODEL.PRED_COEFF = 0.0
    cfg.MODEL.SPARSITY_COEFF = 1.0
    cfg.MODEL.NORM_FN = "id"
    cfg.MODEL.ENCODER.LISTA.L = 0.1
    cfg.MODEL.ENCODER.LISTA.ALPHA = 5e-3
    cfg.MODEL.ENCODER.USE_BIAS = False
    cfg.MODEL.USE_HOMOGENEOUS = True  # Enable homogeneous coordinates for implicit bias
    cfg.MODEL.HOMOGENEOUS_COEFF = 1.0
    return cfg


def get_train_lista_nonlinear_config() -> Config:
    """Training configuration for LISTA with nonlinear encoder."""
    cfg = Config()
    cfg.MODEL.MODEL_NAME = "LISTAKM"
    cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = False
    cfg.MODEL.ENCODER.LAYERS = [64, 64, 64]
    cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 5
    cfg.MODEL.TARGET_SIZE = 1024 * 2
    cfg.MODEL.RES_COEFF = 1.0
    cfg.MODEL.RECONST_COEFF = 1.0
    cfg.MODEL.PRED_COEFF = 0.0
    cfg.MODEL.SPARSITY_COEFF = 1.0
    cfg.MODEL.NORM_FN = "id"
    cfg.MODEL.ENCODER.LISTA.L = 1e4
    cfg.MODEL.ENCODER.LISTA.ALPHA = 1.0
    cfg.MODEL.ENCODER.LAST_RELU = False
    cfg.MODEL.ENCODER.USE_BIAS = False
    cfg.MODEL.DECODER.AFFINE_BIAS = False
    cfg.MODEL.USE_HOMOGENEOUS = True  # Enable homogeneous coordinates for implicit bias
    cfg.MODEL.HOMOGENEOUS_COEFF = 1.0
    return cfg


def get_train_lista_parity_generic_sparse_config() -> Config:
    """LISTA preset aligned to generic_sparse settings for depth-parity sweeps."""
    cfg = Config()
    cfg.TRAIN.LR = 1e-4
    cfg.MODEL.MODEL_NAME = "LISTAKM"
    cfg.MODEL.TARGET_SIZE = 64
    cfg.MODEL.NORM_FN = "id"
    cfg.MODEL.RES_COEFF = 1.0
    cfg.MODEL.RECONST_COEFF = 0.5
    cfg.MODEL.PRED_COEFF = 0.0
    cfg.MODEL.SPARSITY_COEFF = 0.01
    cfg.MODEL.USE_HOMOGENEOUS = False
    cfg.MODEL.DECODER.LAYERS = []
    cfg.MODEL.ENCODER.LAYERS = [64, 64]
    cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = False
    cfg.MODEL.ENCODER.LISTA.FINAL_OP = "relu"
    cfg.MODEL.ENCODER.LISTA.ALPHA = 0.1
    cfg.MODEL.ENCODER.USE_BIAS = True
    return cfg


def get_train_hyperlista_config() -> Config:
    """Configuration for HyperLISTA-based Sparse KM.
    
    HyperLISTA uses analytically-derived encoder weights from the decoder dictionary,
    with only 3 learnable scalar hyperparameters (c_theta, c_beta, c_ss).
    This enables gradient flow from the Koopman loss back through the encoder to the dictionary.
    """
    cfg = Config()
    cfg.MODEL.MODEL_NAME = "LISTAKM"
    cfg.MODEL.ENCODER.ENCODER_TYPE = "hyperlista"
    cfg.MODEL.TARGET_SIZE = 1024 * 2
    cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = 5
    cfg.MODEL.ENCODER.HYPERLISTA.C_THETA = 1e-2
    cfg.MODEL.ENCODER.HYPERLISTA.C_BETA = 1e-4
    cfg.MODEL.ENCODER.HYPERLISTA.C_SS = 0.5
    cfg.MODEL.ENCODER.HYPERLISTA.USE_SUPPORT_SELECTION = True
    cfg.MODEL.ENCODER.HYPERLISTA.USE_MOMENTUM = True
    cfg.MODEL.ENCODER.HYPERLISTA.LEARN_HYPERPARAMS = True
    cfg.MODEL.RES_COEFF = 1.0
    cfg.MODEL.RECONST_COEFF = 1.0
    cfg.MODEL.PRED_COEFF = 1.0
    cfg.MODEL.SPARSITY_COEFF = 1.0
    cfg.MODEL.USE_HOMOGENEOUS = True
    cfg.MODEL.HOMOGENEOUS_COEFF = 1.0
    return cfg


def get_train_hyperlista_parity_generic_sparse_config() -> Config:
    """HyperLISTA preset aligned to generic_sparse non-depth settings.

    Mirrors the LISTA parity baseline coefficients/structure while using
    ENCODER_TYPE=hyperlista so threshold adaptation can be tested fairly.
    """
    cfg = Config()
    cfg.TRAIN.LR = 1e-4
    cfg.MODEL.MODEL_NAME = "LISTAKM"
    cfg.MODEL.ENCODER.ENCODER_TYPE = "hyperlista"
    cfg.MODEL.TARGET_SIZE = 64
    cfg.MODEL.NORM_FN = "id"
    cfg.MODEL.RES_COEFF = 1.0
    cfg.MODEL.RECONST_COEFF = 0.5
    cfg.MODEL.PRED_COEFF = 0.0
    cfg.MODEL.SPARSITY_COEFF = 0.01
    cfg.MODEL.USE_HOMOGENEOUS = False
    cfg.MODEL.DECODER.LAYERS = []
    cfg.MODEL.ENCODER.LAYERS = [64, 64]
    cfg.MODEL.ENCODER.USE_BIAS = True
    cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = 5
    cfg.MODEL.ENCODER.HYPERLISTA.C_THETA = 1e-2
    cfg.MODEL.ENCODER.HYPERLISTA.C_BETA = 1e-4
    cfg.MODEL.ENCODER.HYPERLISTA.C_SS = 0.5
    cfg.MODEL.ENCODER.HYPERLISTA.USE_SUPPORT_SELECTION = True
    cfg.MODEL.ENCODER.HYPERLISTA.USE_MOMENTUM = True
    cfg.MODEL.ENCODER.HYPERLISTA.LEARN_HYPERPARAMS = True
    return cfg


_TRAIN_CONFIG_REGISTRY = {
    "generic": get_train_generic_km_config,
    "generic_sparse": get_train_generic_sparse_config,
    "generic_no_shrink": get_train_generic_no_shrink_config,
    "generic_prediction": get_train_generic_prediction_config,
    "lista": get_train_lista_config,
    "lista_nonlinear": get_train_lista_nonlinear_config,
    "lista_parity_generic_sparse": get_train_lista_parity_generic_sparse_config,
    "hyperlista": get_train_hyperlista_config,
    "hyperlista_parity_generic_sparse": get_train_hyperlista_parity_generic_sparse_config,
}


def get_config(name: str = "default") -> Config:
    """Get a named configuration.
    
    Args:
        name: Configuration name. Options:
            - "default": Base configuration
            - "generic": Standard KoopmanMachine
            - "generic_sparse": Sparse KoopmanMachine with L1
            - "generic_no_shrink": MLP control without L1 or ReLU-induced sparsification
            - "generic_prediction": Prediction-focused
            - "lista": LISTA-based KoopmanMachine
            - "lista_nonlinear": LISTA with MLP encoder
            - "hyperlista": LISTAKM with HyperLISTA encoder mode
            - "lista_parity_generic_sparse": LISTA parity preset for generic_sparse alignment
            - "hyperlista_parity_generic_sparse": HyperLISTA parity preset for generic_sparse alignment
    
    Returns:
        Config for the specified configuration.
    """
    if name == "default":
        return get_default_config()
    if name not in _TRAIN_CONFIG_REGISTRY:
        raise ValueError(f"Unknown config name '{name}'. Available: {list(_TRAIN_CONFIG_REGISTRY.keys())}")
    return _TRAIN_CONFIG_REGISTRY[name]()
