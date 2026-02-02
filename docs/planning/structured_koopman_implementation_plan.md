# Structured LISTA Koopman - Implementation Complete

**Status: IMPLEMENTED** (2026-01-26)

This plan has been fully implemented. For current methodology and results, see `notes.tex` Sections 3.3-3.5.

---

## Implementation Summary

### Completed Components

| Component | Location | Status |
|-----------|----------|--------|
| `StructuredLatentConfig` | `config.py:251-271` | ✅ Complete |
| `StructuredLISTAKM` class | `model.py:1400-2100` | ✅ Complete |
| Arrowhead Koopman blocks | `model.py:1458-1470` | ✅ Complete |
| Pairwise exclusivity loss | `model.py:1717-1743` | ✅ Complete |
| Entropy exclusivity loss | `model.py:1745-1771` | ✅ Complete |
| Top-1 dominance loss | `model.py:1773-1800` | ✅ Complete |
| Warmup schedules | `model.py:1824-1884` | ✅ Complete |
| CLI arguments | `train.py:879-898` | ✅ Complete |
| Basin structure evaluation | `evaluate_basin_structure.py` | ✅ Complete |

### CLI Usage

```bash
# Train StructuredLISTAKM with all new loss functions
python train.py \
  --config lista_nonlinear \
  --env lyapunov \
  --structured \
  --d_global 16 \
  --num_basins 20 \
  --d_basin 32 \
  --lambda_exclusivity 0.05 \
  --lambda_entropy 0.05 \
  --lambda_dominance 0.05 \
  --lambda_sparsity 0.5 \
  --excl_warmup_steps 2000 \
  --pairwise \
  --device cuda
```

### Key Hyperparameters

| Parameter | Default | Recommended Range | Description |
|-----------|---------|-------------------|-------------|
| `d_global` | 8 | 16-32 | Global block dimension |
| `d_basin` | 8 | 16-64 | Per-basin block dimension |
| `num_basins` | 20 | 1.5× GT basins | Over-specify for better separation |
| `lambda_exclusivity` | 0.01 | 0.01-0.1 | Pairwise exclusivity weight |
| `lambda_entropy` | 0.0 | 0.01-0.1 | Entropy-based exclusivity weight |
| `lambda_dominance` | 0.0 | 0.01-0.1 | Top-1 dominance weight |
| `excl_warmup_steps` | 1000 | 1000-2000 | Warmup horizon for exclusivity losses |

---

## Key Findings from Initial Experiments

1. **Over-specification helps**: Using B=20 model basins for 13 GT basins improved accuracy from 37.9% → 61.5%

2. **Small d_basin causes collapse**: With d_basin=4, multiple GT basins collapse to single model basins

3. **Exclusivity alone insufficient**: Strong λ_excl doesn't prevent collapse; need combination of losses

4. **Koopman theory constraint**: Finite-dimensional lifting requires sufficient capacity per basin

---

## Current Experimental Status

See `basin_structure_evaluation_plan.md` for running experiments and monitoring commands.

**72 experiments running** testing:
- Large basin dimensions (d_basin: 16, 32, 64)
- Entropy and dominance losses
- Unstructured LISTA baselines
- GenericKM baselines
