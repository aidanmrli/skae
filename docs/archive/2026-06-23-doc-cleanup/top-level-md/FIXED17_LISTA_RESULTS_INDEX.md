# Fixed-17 LISTA Results Index

Date: April 18, 2026

Purpose:
- This is the canonical quick-reference page for the fixed-`17` transition-rich LISTA results.
- It consolidates the root name, packet location, and headline saved result in one place so senior coauthors do not need to scan the full experiment log.
- It covers every fixed-`17` LISTA root that currently has a paper-facing saved result or that appears in the fixed-`17` phase-portrait handoff packet.
- Full root definitions still live in [transition_rich_basin_partition_manifest.py](/home/mila/l/lia/skae/skae/benchmarks/transition_rich_basin_partition_manifest.py).

## Locked Fixed-17 Finalists

Packet: [transition_rich_basin_partition_final_seed10_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409)  
Forecasting: [collect_pass1/forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/collect_pass1/forecasting_summary.md)  
Interpretability comparison: [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md)

| Human label | Root | H100 / H500 / H1000 | Headline result |
|---|---|---:|---|
| Block-diagonal sign-split LISTA, hard-init oversampling | `lista_blockdiag_signsplit_hardinit_basin_partition` | `0.0182 / 0.0491 / 0.0516` | Best finalized fixed-`17` forecasting root; use it as the forecast-retaining companion result. |
| Dense soft-block sign-split LISTA, `p=64`, hard-init oversampling | `lista_dense_softblock_signsplit_p64_hardinit_basin_partition` | `0.0196 / 0.0733 / 0.0775` | Locked deep-basin basin-support winner; on the selected slice it reaches `H(S|B)=0.0543`, `U_exact=0.9923`, and `freeze/base@20=0.1691`, with paired wins on `15/17`, `14/17`, and `16/17` systems against the matched sparse MLP control. |

## Hard-Init Seed-0 Quartet

Packet: [transition_rich_basin_partition_hardinit_seed0_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409)  
Forecasting: [collect/forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409/collect/forecasting_summary.md)  
Interpretability: [reduce/interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409/reduce/interpretability_summary.md)

| Human label | Root | H100 / H500 / H1000 | Headline result |
|---|---|---:|---|
| Block-diagonal sign-split LISTA | `lista_blockdiag_signsplit_basin_partition` | `0.0472 / 0.0802 / 0.0800` | Standard-sampling blockdiag anchor for the hard-init comparison. |
| Block-diagonal sign-split LISTA, hard-init oversampling | `lista_blockdiag_signsplit_hardinit_basin_partition` | `0.0467 / 0.0823 / 0.0841` | Hard-init improves the blockdiag interpretability read with nearly neutral forecasting. |
| Dense soft-block sign-split LISTA, `p=64` | `lista_dense_softblock_signsplit_p64_basin_partition` | `0.0816 / 0.1512 / 0.1358` | Standard-sampling dense `p=64` anchor for the hard-init comparison. |
| Dense soft-block sign-split LISTA, `p=64`, hard-init oversampling | `lista_dense_softblock_signsplit_p64_hardinit_basin_partition` | `0.0373 / 0.0827 / 0.0794` | Stronger forecasting / intervention tradeoff than the standard dense `p=64` root. |

Key within-family deltas from this packet:
- Blockdiag hard-init versus standard blockdiag: `H(S|B) 1.4297 -> 1.3493`, `U_exact 0.7181 -> 0.7447`, `H(F|B) 0.1129 -> 0.1018`, own-basin projection ratio `25.5197 -> 7.7018`, and freeze-support ratio `0.7599 -> 0.3034`.
- Dense `p=64` hard-init versus standard dense `p=64`: `H1000 0.1358 -> 0.0794`, own-basin projection ratio `9.9799 -> 3.0431`, and freeze-support ratio `0.8926 -> 0.6768`.

## Working-Budget Default-Sampling Shortlist Provenance

Packets:
- `v5`: [transition_rich_basin_partition_20260409_seed0_smoke_v5](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v5), [forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v5/collect_pass0/forecasting_summary.md), [interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v5/interpretability_pass0/interpretability_summary.md)
- `v6`: [transition_rich_basin_partition_20260409_seed0_smoke_v6](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v6), [forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v6/collect_pass0/forecasting_summary.md), [interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v6/interpretability_reduce_pass0/interpretability_summary.md)
- `v7`: [transition_rich_basin_partition_20260410_seed0_smoke_v7](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260410_seed0_smoke_v7), [forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260410_seed0_smoke_v7/collect_pass0/forecasting_summary.md), [interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260410_seed0_smoke_v7/interpretability_reduce_pass0/interpretability_summary.md)

| Root | Headline result | Why it still matters |
|---|---|---|
| `lista_dense_softblock_signsplit_p64_basin_partition` | `H(S|B)=0.7719`, `U_exact=0.8064`, `H(F|B)=0.0521`, `H1000=0.1819`, `16/17` good systems | Best exact-support shortlist root from the default-sampling working-budget screen. |
| `lista_blockdiag_signsplit_basin_partition` | `H1000=0.0119`, `17/17` good systems | Best forecast-retention root across the working-budget shortlist tiers. |
| `lista_dense_softblock_signsplit_coherence_basin_partition` | `H1000=0.0585`, `17/17` good systems | Best forecast-preserving root from the `v6` identifiability follow-up. |
| `lista_dense_softblock_signsplit_linear_encoder_coherence_basin_partition` | `H(S|B)=1.0575`, `U_exact=0.7837`, `H(F|B)=0.0841`, `H1000=0.9118`, `15/17` good systems | Compression-improving but forecast-costly `v6` root. |
| `lista_blockdiag_sparsegroup_basin_partition` | `H1000=0.0846` | Best new forecasting root from the `v7` screened design-note axes. |
| `lista_dense_softblock_signsplit_p64_softblock5em4_basin_partition` | `H(S|B)=0.6795`, `U_exact=0.8453`, `H(F|B)=0.0634`; missing `1/17` systems after a fast failure | Strongest new deep-basin support-compression read from `v7`. |

## Long-Budget Default-Sampling Reopen Check

Packet: [transition_rich_basin_partition_v6_v7_200k_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_basin_partition_v6_v7_200k_seed10_20260410)  
Forecasting: [collect_pass0/forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_v6_v7_200k_seed10_20260410/collect_pass0/forecasting_summary.md)

| Root | H100 / H500 / H1000 | Read |
|---|---:|---|
| `lista_dense_softblock_signsplit_coherence_basin_partition` | `0.0416 / 0.0761 / 0.0796` | Better of the two long-budget reopen-check roots, but still not enough to reopen the locked shortlist. |
| `lista_blockdiag_sparsegroup_basin_partition` | `0.0437 / 0.1142 / 0.1193` | Forecast-competitive, but still behind the promoted hard-init finalists. |

## Roots Used in the Fixed-17 Phase-Portrait Handoff Packet

Manifest: [fixed17_h1000_h3000_h5000_lista_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/fixed17_lista_phase_portraits_20260414/fixed17_h1000_h3000_h5000_lista_phase_portraits_manifest.json)

Selection rule:
- The handoff packet selects the lowest saved `H1000` best-periodic LISTA run per system from collected fixed-`17` results and then reuses that run's saved `H1000` best-periodic mode at `H1000`, `H3000`, and `H5000`.

| Root | Systems selected |
|---|---:|
| `lista_blockdiag_signsplit_hardinit_basin_partition` | `6` |
| `lista_dense_softblock_signsplit_coherence_basin_partition` | `3` |
| `lista_dense_basin_partition` | `2` |
| `lista_dense_softblock_signsplit_p64_hardinit_basin_partition` | `2` |
| `lista_blockdiag_sparsegroup_basin_partition` | `1` |
| `lista_blockdiag_basin_partition` | `1` |
| `lista_blockdiag_signsplit_basin_partition` | `1` |
| `lista_blockdiag_double_basin_partition` | `1` |

## Scope Note

- If a fixed-`17` LISTA root appears in this file, it already has a paper-relevant saved result or appears in the coauthor handoff figures.
- If a root exists in [transition_rich_basin_partition_manifest.py](/home/mila/l/lia/skae/skae/benchmarks/transition_rich_basin_partition_manifest.py) but not here, it is defined or screened but does not currently carry a promoted paper-facing result.
