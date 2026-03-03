"""Tests for the config system (config.py).

Tests configuration loading, modification, and registry functionality.
"""

import pytest

from skae.config import (
    Config,
    get_config,
    get_default_config,
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
    assert cfg_parity.MODEL.ENCODER.LISTA.FINAL_OP == "shrink"


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
