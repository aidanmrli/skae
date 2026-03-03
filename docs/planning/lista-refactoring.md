# Refactor: Unified LISTAKM with Pluggable Encoder (Clean Break)

## Context

We currently have three LISTA-family model classes: `LISTAKM`, `HyperLISTAKM`, and `StructuredLISTAKM`. `LISTAKM` and `HyperLISTAKM` duplicate most logic (dictionary init, homogeneous coordinates, decode path, sparsity/homogeneous losses). The true variation is only the encoder implementation (`LISTA` vs `HyperLISTA`).

Goal: keep one canonical LISTA model class with a pluggable encoder and remove all legacy class-level branching.

## Hard Constraints

- **No backward compatibility.**
- Do **not** preserve `MODEL_NAME="HyperLISTAKM"` behavior.
- Do **not** add migration shims in config/model loaders.
- Favor the cleanest modular code over compatibility.

## Plan

### 1. Add encoder selector to config (`skae/config.py`)

- Add `ENCODER_TYPE: str = "lista"` to `EncoderConfig`.
- Allowed values: `"lista"` and `"hyperlista"`.
- Update `get_train_hyperlista_config()` to set:
  - `cfg.MODEL.MODEL_NAME = "LISTAKM"`
  - `cfg.MODEL.ENCODER.ENCODER_TYPE = "hyperlista"`
- Remove `HyperLISTAKM` from `MODEL_NAME` documentation/comments.

### 2. Unify encoder construction in `LISTAKM` (`skae/model.py`)

- Add `_init_dictionary(self, zdim)` helper for dictionary initialization.
- Add `_build_encoder(self, cfg, internal_obs_size, wd_init)`:
  - `"lista"`: compute Lipschitz constant from `wd_init` and build `LISTA(..., L_override=...)`.
  - `"hyperlista"`: build `HyperLISTA(cfg, internal_obs_size, self.dict)` so gradients flow through live dictionary parameters.

### 3. Refactor `LISTAKM.__init__`

- Use `_init_dictionary(...)` and `_build_encoder(...)`.
- Replace `self.lista` with `self.encoder`.
- Keep existing Koopman structure, decode behavior, block-loss plumbing, and homogeneous-coordinate behavior unchanged.

### 4. Rename encoder references

- Replace all `self.lista(...)` calls with `self.encoder(...)`.
- Update inline comments/docstrings to refer to generic encoder, not LISTA-only field names.

### 5. Delete `HyperLISTAKM`

- Remove `HyperLISTAKM` class entirely.
- Remove any imports/usages tied to this class.

### 6. Update model factory registry

- Remove `"HyperLISTAKM"` from `_MODEL_REGISTRY`.
- Keep strict failure semantics for unknown model names.

### 7. Align `StructuredLISTAKM`

- Ensure parent/child comments and assumptions reflect `self.encoder` naming.
- No separate structured hyperlista class; it inherits unified behavior from `LISTAKM`.

### 8. Update training script defaults (`tools/train.py`)

- Replace model-name-only log-dir branching with LISTA encoder-aware routing:
  - `LISTAKM + ENCODER_TYPE="hyperlista"` -> `./runs/hyperlista`
  - `LISTAKM + other encoder types` -> `./runs/lista`
  - non-LISTA models -> `./runs/kae`
- Remove `HyperLISTAKM` references from help text/comments.

### 9. Update tests

- `tests/test_hyperlista.py`:
  - Keep `HyperLISTA` unit tests.
  - Replace `HyperLISTAKM` model tests with unified `LISTAKM` (`ENCODER_TYPE="hyperlista"`).
  - Update gradient checks to use `model.encoder` for hyperparameters.
- `tests/test_model.py`:
  - Add/adjust factory tests to verify:
    - `get_config("hyperlista")` builds `LISTAKM`.
    - `MODEL_NAME="HyperLISTAKM"` raises `ValueError`.
- `tests/test_integration.py`:
  - Update any assumptions that imply a separate HyperLISTAKM class.

### 10. Remove legacy HyperLISTA-specific tooling

- Delete `tools/tune_hyperlista.py`.

## Files to Modify

| File | Changes |
|------|---------|
| `skae/model.py` | Add `_init_dictionary` + `_build_encoder`, rename `self.lista` -> `self.encoder`, delete `HyperLISTAKM`, update registry |
| `skae/config.py` | Add `ENCODER_TYPE`, update `hyperlista` preset, clean model-name docs |
| `tools/train.py` | Encoder-aware LISTA log-dir default routing, remove legacy references |
| `tests/test_hyperlista.py` | Move model tests to unified LISTAKM hyperlista mode |
| `tests/test_model.py` | Update factory expectations and strict unknown-model behavior |
| `tests/test_integration.py` | Align integration expectations with unified class model |
| `tools/tune_hyperlista.py` | **Delete** |

## Design Decisions

- **Single LISTA-family model class**: all LISTA variants live in `LISTAKM`.
- **No encoder base class**: both encoders already satisfy `nn.Module.forward(x) -> z`.
- **Factory method inside `LISTAKM`**: encoder construction depends on both `wd_init` and `self.dict`.
- **Strict clean break**: no aliases, no migrations, no compatibility adapters.

## TDD Test Plan (Write First)

Execute in red-green-refactor order. Add these tests before implementation edits.

1. `tests/test_config.py`
- `test_hyperlista_preset_uses_listakm`: `get_config("hyperlista").MODEL.MODEL_NAME == "LISTAKM"`.
- `test_encoder_type_default`: default encoder type is `"lista"`.
- `test_encoder_type_roundtrip_json`: `ENCODER_TYPE` survives `to_json`/`from_json`.

2. `tests/test_model.py` (factory contract + clean break)
- `test_make_model_hyperlista_returns_listakm`.
- `test_make_model_rejects_hyperlistakm_name`: strict `ValueError` on `"HyperLISTAKM"`.
- `test_model_module_has_no_hyperlistakm_symbol`: no `HyperLISTAKM` symbol exported.

3. `tests/test_model.py` or `tests/test_hyperlista.py` (encoder dispatch)
- `test_listakm_builds_lista_encoder_when_encoder_type_lista` (`isinstance(model.encoder, LISTA)`).
- `test_listakm_builds_hyperlista_encoder_when_encoder_type_hyperlista` (`isinstance(model.encoder, HyperLISTA)`).
- `test_invalid_encoder_type_raises_value_error`.

4. `tests/test_hyperlista.py` (unified model behavior)
- `test_unified_hyperlista_encode_decode_shapes`.
- `test_unified_hyperlista_loss_runs`.
- `test_unified_hyperlista_homogeneous_mode_runs`.

5. `tests/test_hyperlista.py` (gradient semantics)
- `test_encode_only_hyperlista_backprops_to_dict`: `model.encode(x).sum().backward()` gives nonzero `model.dict.grad`.
- `test_encode_only_lista_does_not_backprop_to_dict`: encode-only path leaves dictionary gradient absent/zero in `"lista"` mode.

6. `tests/test_model.py` (field rename safety)
- `test_listakm_uses_encoder_field`: model has `encoder` and no `lista` attribute.

7. `tests/test_integration.py`
- `test_hyperlista_config_end_to_end_through_make_model_and_loss`: no direct `HyperLISTAKM` class usage.

8. `tests/test_train.py` (default run-dir routing)
- `test_default_log_dir_lista_encoder_type`.
- `test_default_log_dir_hyperlista_encoder_type`.
- `test_default_log_dir_non_lista_model`.

## Verification

1. `uv run pytest tests/test_model.py tests/test_hyperlista.py tests/test_integration.py -v`
2. Verify model creation behavior:
   - `make_model(get_config("lista"), obs_size)` -> `LISTAKM` with `LISTA` encoder
   - `make_model(get_config("hyperlista"), obs_size)` -> `LISTAKM` with `HyperLISTA` encoder
   - `MODEL_NAME="HyperLISTAKM"` -> raises `ValueError`
3. `uv run pytest tests/ -v`
