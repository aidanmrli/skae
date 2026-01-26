# Basin Structure Evaluation - Status & Quick Reference

This document tracks the experimental status for basin structure correspondence experiments. For detailed methodology and results, see `notes.tex` Section 4.3.

---

## Current Status (2026-01-26)

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
| **Over-specified basins (B=20)** | **61.5% accuracy** | **Best result - excess capacity helps** |
| Entropy/Dominance sweep | ~20% accuracy | **NEGATIVE RESULT - harms performance** |

### Running Experiments (Job IDs)

| Job ID | Sweep | Description | Status |
|--------|-------|-------------|--------|
| ~~8559855~~ | ~~Entropy/Dominance~~ | ~~Test entropy/dominance losses~~ | **CANCELLED** (harmful) |
| **8560031** | Structured Large Dims (FIXED) | d_basin ∈ {16,32,64}, d_global ∈ {16,32} | **RUNNING** (24 jobs) |
| 8559857 | Unstructured LISTA | target_size ∈ {256,512,1024,2048} | RUNNING (24 jobs) |
| 8559858 | GenericKM Baseline | target_size ∈ {64,128,256,512} | RUNNING (12 jobs) |

**Note:** Job 8560031 is the corrected version of 8559856 - removed entropy/dominance losses that were contaminating results.

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

## Remaining Hypotheses Under Test

1. **Capacity hypothesis**: Small d_basin (4) causes collapse due to insufficient capacity for local Koopman dynamics
   - *Testing*: d_basin ∈ {16, 32, 64} sweep (Job 8560031)

2. ~~**Entropy hypothesis**~~: ~~Pairwise exclusivity doesn't directly minimize activation entropy~~
   - **REJECTED**: Entropy loss harms performance

3. ~~**Dominance hypothesis**~~: ~~Need explicit penalty on non-dominant basins~~
   - **REJECTED**: Dominance loss harms performance

4. **Unstructured hypothesis**: LISTA's natural sparse supports may cluster by basin without explicit partitioning
   - *Testing*: Large LISTAKM (256-2048 dims) baseline (Job 8559857)

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

## Next Steps (After Current Sweeps)

1. **Analyze results**: Run `collect_sweep_results.py` and identify best configurations
2. **Update notes.tex**: Document findings with tables and interpretation
3. **If accuracy >80%**: Proceed to long-horizon forecasting experiments
4. **If accuracy <80%**: Consider:
   - Contrastive losses between basins (encourage different basins for different inputs)
   - Supervised pre-training with basin labels
   - Soft basin assignment with temperature scheduling
