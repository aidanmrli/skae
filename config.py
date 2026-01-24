"""Configuration system using Python dataclasses for PyTorch implementation.

This module provides a type-safe, native Python configuration system using dataclasses.
No external dependencies required (no ml_collections, no argparse conversion).

## Usage

### 1. Quick Start - Use Predefined Configs

```python
from config import get_config

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
from config import Config, ModelConfig, TrainConfig

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
from config import get_config

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
- `"generic_prediction"`: Prediction-focused (no reconstruction)
- `"lista"`: LISTA-based sparse autoencoder (2048-dim latent)
- `"lista_nonlinear"`: LISTA with nonlinear MLP encoder

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
from typing import List
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


@dataclass
class EnvConfig:
    """Environment configuration.
    
    For built-in environments, set ENV_NAME to one of:
        ["duffing", "parabolic", "pendulum", "lotka_volterra", "lorenz63", "lyapunov"]
    
    For dysts systems, set ENV_NAME to "dysts:SystemName" (e.g., "dysts:Lorenz", "dysts:Chua").
    The DYSTS config section can be used to customize dysts system behavior.
    """
    ENV_NAME: str = "duffing"
    PARABOLIC: ParabolicConfig = field(default_factory=ParabolicConfig)
    DUFFING: DuffingConfig = field(default_factory=DuffingConfig)
    PENDULUM: PendulumConfig = field(default_factory=PendulumConfig)
    LOTKA_VOLTERRA: LotkaVolterraConfig = field(default_factory=LotkaVolterraConfig)
    LORENZ63: Lorenz63Config = field(default_factory=Lorenz63Config)
    LYAPUNOV: LyapunovConfig = field(default_factory=LyapunovConfig)
    DYSTS: DystsConfig = field(default_factory=DystsConfig)


@dataclass
class ListaConfig:
    """LISTA encoder-specific configuration."""
    NUM_LOOPS: int = 5  # LISTA iterations
    L: float = 1e3  # Lipschitz constant estimate
    ALPHA: float = 0.1  # sparsity threshold
    LINEAR_ENCODER: bool = False  # use MLP vs linear encoder


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
    USE_SUPPORT_SELECTION: bool = True  # Enable adaptive support selection
    USE_MOMENTUM: bool = True        # Enable momentum acceleration
    MAG_RATIO: float = 0.1           # Threshold for support size approximation
    LEARN_HYPERPARAMS: bool = True   # Whether c_theta, c_beta, c_ss are learnable


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
    LAMBDA_EXCLUSIVITY: float = 1e-2   # Final exclusivity penalty weight
    EXCL_WARMUP_STEPS: int = 1000      # Steps to ramp exclusivity from 0 to final


@dataclass
class EncoderConfig:
    """Encoder architecture configuration."""
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
    MODEL_NAME: str = "SparseKM"  # from ["GenericKM", "SparseKM", "LISTAKM"]
    NORM_FN: str = "id"  # from ["id", "ball"]
    TARGET_SIZE: int = 16  # latent_dim i.e. zdim
    
    # Loss coefficients
    RES_COEFF: float = 1.0  # alignment loss weight
    RECONST_COEFF: float = 0.02  # reconstruction loss weight
    PRED_COEFF: float = 0.0  # prediction loss weight
    SPARSITY_COEFF: float = 1e-3  # sparsity loss weight (L1 regularization)
    HOMOGENEOUS_COEFF: float = 1.0  # homogeneous coordinate consistency loss weight
    
    # Homogeneous coordinates: append 1 to input, enables implicit bias learning
    USE_HOMOGENEOUS: bool = False
    
    # Sub-configs
    ENCODER: EncoderConfig = field(default_factory=EncoderConfig)
    DECODER: DecoderConfig = field(default_factory=DecoderConfig)
    STRUCTURED: StructuredLatentConfig = field(default_factory=StructuredLatentConfig)


@dataclass
class TrainConfig:
    """Training configuration."""
    NUM_STEPS: int = 2_000  # total training steps (epochs)
    BATCH_SIZE: int = 256
    DATA_SIZE: int = 256 * 8  # total dataset size
    LR: float = 1e-4  # main learning rate (encoder/decoder)
    WEIGHT_DECAY: float = 1e-4  # weight decay for AdamW optimizer
    K_MATRIX_LR: float = 1e-5  # learning rate for Koopman matrix parameters
    
    # Lightweight evaluation during training
    EVAL_EVERY: int = 500  # evaluation interval in training steps
    EVAL_NUM_STEPS: int = 200  # rollout horizon for the quick eval() helper
    
    # Sequence training parameters
    USE_SEQUENCE_LOSS: bool = False  # default to single-step loss for parity with JAX
    SEQUENCE_LENGTH: int = 10  # number of forward steps in each training sequence (T)

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
        model = ModelConfig(**{k: v for k, v in model_dict.items() if k not in ["ENCODER", "DECODER", "STRUCTURED"]})
        model.ENCODER = encoder
        model.DECODER = decoder
        model.STRUCTURED = structured
        
        train = TrainConfig(**config_dict.get("TRAIN", {}))
        
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


def get_train_hyperlista_config() -> Config:
    """Configuration for HyperLISTA-based Sparse KM.
    
    HyperLISTA uses analytically-derived encoder weights from the decoder dictionary,
    with only 3 learnable scalar hyperparameters (c_theta, c_beta, c_ss).
    This enables gradient flow from the Koopman loss back through the encoder to the dictionary.
    """
    cfg = Config()
    cfg.MODEL.MODEL_NAME = "HyperLISTAKM"
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


_TRAIN_CONFIG_REGISTRY = {
    "generic": get_train_generic_km_config,
    "generic_sparse": get_train_generic_sparse_config,
    "generic_prediction": get_train_generic_prediction_config,
    "lista": get_train_lista_config,
    "lista_nonlinear": get_train_lista_nonlinear_config,
    "hyperlista": get_train_hyperlista_config,
}


def get_config(name: str = "default") -> Config:
    """Get a named configuration.
    
    Args:
        name: Configuration name. Options:
            - "default": Base configuration
            - "generic": Standard KoopmanMachine
            - "generic_sparse": Sparse KoopmanMachine with L1
            - "generic_prediction": Prediction-focused
            - "lista": LISTA-based KoopmanMachine
            - "lista_nonlinear": LISTA with MLP encoder
    
    Returns:
        Config for the specified configuration.
    """
    if name == "default":
        return get_default_config()
    if name not in _TRAIN_CONFIG_REGISTRY:
        raise ValueError(f"Unknown config name '{name}'. Available: {list(_TRAIN_CONFIG_REGISTRY.keys())}")
    return _TRAIN_CONFIG_REGISTRY[name]()

