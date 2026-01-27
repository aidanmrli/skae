# Basin Structure Evaluation - Status & Quick Reference

This document tracks the experimental status for basin structure correspondence experiments. For detailed methodology and results, see `notes.tex` Section 4.3.

---

## Current Status (2026-01-26)

### Latest Result: 76.6% Basin Assignment Accuracy

**Best configuration found:** d_global=16, d_basin=16, λ_excl=0.01, B=20
- Previous best: 61.5% → New best: **76.6%** (+15.1 percentage points)
- Key insight: Moderate basin dimensions (d_b=16) outperform larger ones (32, 64)

### Critical Finding: Entropy/Dominance Losses Harm Performance

**The entropy and dominance losses (λ_entropy, λ_dominance) cause severe degradation:**
- Baseline with exclusivity only: **61.5% accuracy**
- Adding entropy/dominance (even at 0.01): **~20% accuracy** (worse than random)

The losses over-regularize and cause the model to collapse to a degenerate solution where nearly all points are assigned to the same few basins. These losses have been disabled in subsequent experiments.

### Completed Experiments

| Experiment | Best Result | Key Finding |
|------------|-------------|-------------|
| Baseline (d_b=4, B=13, λ=0.001) | 37.9% accuracy | Basin collapse: 8/13 basins used |
| Exclusivity sweep (λ: 0.01-0.1) | 42.9% accuracy | Stronger λ_excl has diminishing returns |
| Larger d_basin (d_b=8) | 47.4% accuracy | Modest improvement from capacity |
| Over-specified basins (B=20) | 61.5% accuracy | Excess capacity helps |
| Entropy/Dominance sweep | ~20% accuracy | **NEGATIVE RESULT - harms performance** |
| **Large dims sweep (d_b=16)** | **76.6% accuracy** | **BEST - moderate dims optimal** |
| Unstructured LISTA baseline | 72% linear acc | Medium dims (256) best |
| GenericKM baseline | 96% linear acc | Near-perfect basin separability |

### Completed Experiments (Job IDs)

| Job ID | Sweep | Description | Status |
|--------|-------|-------------|--------|
| ~~8559855~~ | ~~Entropy/Dominance~~ | ~~Test entropy/dominance losses~~ | **CANCELLED** (harmful) |
| 8560031 | Structured Large Dims | d_basin ∈ {16,32,64}, d_global ∈ {16,32} | **COMPLETED** ✓ |
| 8559857 | Unstructured LISTA | target_size ∈ {256,512,1024,2048} | **COMPLETED** ✓ |
| 8559858 | GenericKM Baseline | target_size ∈ {64,128,256,512} | **COMPLETED** ✓ |

### Monitoring Commands

```bash
# Check job status
squeue -u $USER

# Watch structured large dims results
watch -n 60 'grep "Basin Assignment Accuracy" /network/scratch/l/lia/skae/struct_large-8560031*.out 2>/dev/null'

# Watch unstructured LISTA results (uses linear classifier)
watch -n 60 'grep "Linear Classifier Accuracy" /network/scratch/l/lia/skae/lista_large*.out 2>/dev/null'

# Collect all results when done
python collect_sweep_results.py --output /network/scratch/l/lia/skae/all_results.csv
```

---

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Basin Assignment Accuracy** | Fraction matching GT basin (with optimal mapping) | >80% |
| **Temporal Consistency** | Fraction where active basin = trajectory mode | >95% |
| **Linear Classifier Accuracy** | Logistic regression on latent codes | >88% (GenericKM baseline) |

---

## Hypothesis Results

1. ~~**Capacity hypothesis**~~: Small d_basin (4) causes collapse due to insufficient capacity
   - **PARTIALLY CONFIRMED**: d_basin=16 improves accuracy to 76.6%, but d_basin=64 performs *worse* (26-46%)
   - **Insight**: Moderate dimensions optimal; very large dimensions hurt due to sparse signal dilution

2. ~~**Entropy hypothesis**~~: ~~Pairwise exclusivity doesn't directly minimize activation entropy~~
   - **REJECTED**: Entropy loss harms performance

3. ~~**Dominance hypothesis**~~: ~~Need explicit penalty on non-dominant basins~~
   - **REJECTED**: Dominance loss harms performance

4. ~~**Unstructured hypothesis**~~: LISTA's sparse supports may cluster by basin without explicit partitioning
   - **REJECTED**: Best unstructured LISTA achieves only 72% linear classifier accuracy vs 96% for GenericKM

---

## Output Directories

```
/network/scratch/l/lia/skae/
├── structured_excl_sweep_lyapunov/       # First exclusivity sweep (complete)
├── structured_entropy_sweep_lyapunov/    # Entropy/dominance sweep (CANCELLED - negative results)
├── structured_large_dims_lyapunov/       # Large dimensions sweep (running - FIXED)
├── unstructured_lista_lyapunov/          # Unstructured LISTA sweep (running)
├── generic_baseline_lyapunov/            # GenericKM baseline (running)
```

---

## Next Steps

Current best: **76.6%** (below 80% target)

### Immediate priorities:
1. ✓ ~~Analyze results~~ - Done
2. ✓ ~~Update notes.tex~~ - Done
3. **Reduce optimization variance**: 30pp gap between seeds indicates unstable training
   - Try multiple seeds with best config (d_g=16, d_b=16, λ_excl=0.01)
   - Consider ensemble approaches

### To reach 80%+ accuracy:
1. **Contrastive losses**: Penalize when different GT basins activate the same model basin
2. **Temperature scheduling**: Start with soft basin assignments, anneal to hard
3. **Curriculum learning**: Train on fewer attractors first, then increase complexity
4. **Initialization strategies**: Pre-assign basin blocks to different state-space regions

### Optional (revisit with better baseline):
- Re-test entropy/dominance losses with 76.6% baseline (previously tested with 61.5%)
- These may help if pairwise exclusivity alone is insufficient at higher accuracy levels
