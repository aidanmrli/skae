"""Tests for the config system (config.py).

Tests configuration loading, modification, and registry functionality.
"""

import pytest

from skae.config import (
    Config,
    apply_env_dt_override,
    canonical_env_name,
    get_config,
    get_default_config,
    get_env_dt,
    get_train_generic_km_config,
    get_train_lista_config,
    get_train_lista_parity_generic_sparse_config,
)


def test_get_default_config():
    """Test that default config has expected structure and values."""
    cfg = get_default_config()
    
    # Check top-level keys
    assert hasattr(cfg, "SEED")
    assert hasattr(cfg, "ENV")
    assert hasattr(cfg, "MODEL")
    assert hasattr(cfg, "TRAIN")
    
    # Check ENV structure
    assert hasattr(cfg.ENV, "ENV_NAME")
    assert hasattr(cfg.ENV, "DUFFING")
    assert hasattr(cfg.ENV, "PARABOLIC")
    assert hasattr(cfg.ENV, "MULTIWELL")
    assert hasattr(cfg.ENV, "BLENDED")
    
    # Check MODEL structure
    assert hasattr(cfg.MODEL, "MODEL_NAME")
    assert hasattr(cfg.MODEL, "TARGET_SIZE")
    assert hasattr(cfg.MODEL, "ENCODER")
    assert hasattr(cfg.MODEL, "DECODER")
    
    # Check loss coefficients
    assert hasattr(cfg.MODEL, "RES_COEFF")
    assert hasattr(cfg.MODEL, "RECONST_COEFF")
    assert hasattr(cfg.MODEL, "PRED_COEFF")
    assert hasattr(cfg.MODEL, "SPARSITY_COEFF")
    
    # Check TRAIN structure
    assert hasattr(cfg.TRAIN, "NUM_STEPS")
    assert hasattr(cfg.TRAIN, "BATCH_SIZE")
    assert hasattr(cfg.TRAIN, "LR")


def test_get_named_configs():
    """Test that named configurations load correctly."""
    # Test generic config
    cfg_generic = get_train_generic_km_config()
    assert cfg_generic.MODEL.MODEL_NAME == "GenericKM"
    assert cfg_generic.MODEL.TARGET_SIZE == 64
    
    # Test LISTA config
    cfg_lista = get_train_lista_config()
    assert cfg_lista.MODEL.MODEL_NAME == "LISTAKM"
    assert cfg_lista.MODEL.ENCODER.LISTA.NUM_LOOPS == 5
    assert cfg_lista.MODEL.TARGET_SIZE == 1024 * 2

    cfg_parity = get_train_lista_parity_generic_sparse_config()
    assert cfg_parity.MODEL.MODEL_NAME == "LISTAKM"
    assert cfg_parity.MODEL.TARGET_SIZE == 64
    assert cfg_parity.MODEL.SPARSITY_COEFF == 0.01
    assert cfg_parity.MODEL.ENCODER.LISTA.FINAL_OP == "relu"


def test_config_registry():
    """Test that config registry works."""
    cfg = get_config("default")
    assert cfg is not None
    
    cfg_generic = get_config("generic")
    assert cfg_generic.MODEL.MODEL_NAME == "GenericKM"
    
    cfg_lista = get_config("lista")
    assert cfg_lista.MODEL.MODEL_NAME == "LISTAKM"

    cfg_parity = get_config("lista_parity_generic_sparse")
    assert cfg_parity.MODEL.MODEL_NAME == "LISTAKM"
    assert cfg_parity.MODEL.TARGET_SIZE == 64

    with pytest.raises(ValueError):
        get_config("nonexistent")


def test_config_modification():
    """Test that config can be modified."""
    cfg = get_default_config()
    original_lr = cfg.TRAIN.LR
    
    cfg.TRAIN.LR = 2000
    assert cfg.TRAIN.LR == 2000
    assert cfg.TRAIN.LR != original_lr


def test_config_dt_extraction():
    """Test that dt is correctly set in environment-specific configs."""
    cfg = get_default_config()
    cfg.ENV.ENV_NAME = "duffing"
    cfg.ENV.DUFFING.DT = 0.02
    
    # Check that dt is correctly set
    assert cfg.ENV.DUFFING.DT == 0.02
    assert cfg.ENV.ENV_NAME == "duffing"


def test_unknown_train_key_raises():
    """Unknown TRAIN keys should fail fast (strict config parsing)."""
    cfg_dict = get_default_config().to_dict()
    cfg_dict["TRAIN"]["USE_SEQUENCE_LOSS"] = True

    with pytest.raises(TypeError):
        Config.from_dict(cfg_dict)


def test_hyperlista_preset_uses_listakm():
    """HyperLISTA preset should select LISTAKM with hyperlista encoder mode."""
    cfg = get_config("hyperlista")
    assert cfg.MODEL.MODEL_NAME == "LISTAKM"
    assert cfg.MODEL.ENCODER.ENCODER_TYPE == "hyperlista"


def test_encoder_type_default():
    """Encoder type defaults to standard LISTA mode."""
    cfg = get_default_config()
    assert cfg.MODEL.ENCODER.ENCODER_TYPE == "lista"


def test_encoder_type_roundtrip_json(tmp_path):
    """Encoder type should persist through JSON serialization."""
    cfg = get_default_config()
    cfg.MODEL.ENCODER.ENCODER_TYPE = "hyperlista"
    path = tmp_path / "config.json"
    cfg.to_json(str(path))

    loaded = Config.from_json(str(path))
    assert loaded.MODEL.ENCODER.ENCODER_TYPE == "hyperlista"


def test_high_dim_env_config_from_dict_roundtrip():
    """High-dimensional benchmark env configs should survive dict roundtrip."""
    cfg = get_default_config()
    cfg.ENV.ENV_NAME = "hopfield"
    cfg.MODEL.OBS_LOSS_DIM_NORMALIZATION = "dim"
    cfg.ENV.KURAMOTO.NUM_OSCILLATORS = 32
    cfg.ENV.HOPFIELD.NUM_NEURONS = 20
    cfg.ENV.HOPFIELD.NUM_PATTERNS = 5
    cfg.ENV.COMPETITIVE_LV.NUM_SPECIES = 12

    loaded = Config.from_dict(cfg.to_dict())

    assert loaded.ENV.KURAMOTO.NUM_OSCILLATORS == 32
    assert loaded.ENV.HOPFIELD.NUM_NEURONS == 20
    assert loaded.ENV.HOPFIELD.NUM_PATTERNS == 5
    assert loaded.ENV.COMPETITIVE_LV.NUM_SPECIES == 12
    assert loaded.MODEL.OBS_LOSS_DIM_NORMALIZATION == "dim"


@pytest.mark.parametrize(
    ("env_name", "expected"),
    [
        ("duffing", "duffing"),
        ("claude:cal_triangle_3", "claude_catalog"),
        ("multiwell:energy", "multiwell"),
        ("blended", "blended"),
        ("kuramoto", "kuramoto"),
        ("dysts:LorenzCoupled", "dysts"),
        ("LorenzCoupled", "dysts"),
    ],
)
def test_canonical_env_name(env_name, expected):
    """Environment aliases should normalize to the correct config bucket."""
    assert canonical_env_name(env_name) == expected


@pytest.mark.parametrize(
    "env_name",
    [
        "duffing",
        "claude:cal_triangle_3",
        "multiwell_rotational",
        "multiwell:energy",
        "blended",
        "kuramoto",
        "dysts:LorenzCoupled",
    ],
)
def test_apply_env_dt_override_sets_expected_bucket(env_name):
    """dt overrides should land on the environment config actually used by training."""
    cfg = get_default_config()
    cfg.ENV.ENV_NAME = env_name

    apply_env_dt_override(cfg, dt=0.123, env_name=env_name)

    assert get_env_dt(cfg, env_name=env_name) == pytest.approx(0.123)


def test_get_env_dt_uses_claude_default_when_not_overridden():
    """Claude catalog env dt should fall back to the system's intrinsic default."""
    cfg = get_default_config()
    cfg.ENV.ENV_NAME = "claude:cal_triangle_3"

    assert get_env_dt(cfg) == pytest.approx(0.03)
