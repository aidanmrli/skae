# Paper Seed Statistics Reanalysis

Date: March 31, 2026

This report re-computes paper-facing seed statistics from the lowest-level artifacts still available in the repository.

Provenance rules:
- Forecasting: raw `evaluation_results_best.json` under the original run directories listed by the collector CSVs.
- Support alignment: per-seed `support_alignment.json` files under the collector `per_run/` trees.
- Local linearity: per-seed `metrics.json` files under the collector `per_case/` trees.
- Kuramoto mode-support audit: raw per-seed `analysis_results.json` files referenced by the collector rows.
- Label-free clustering: raw per-seed `analysis_results.json` files under the March 21 scratch evaluation trees, with the collected row CSVs used only as manifests and verification references.

Machine-readable outputs are under `results/paper_seed_statistics_20260331`.

## Raw-Source Verification

Every family with both a raw artifact and a collector row file was checked for exact agreement up to float tolerance before computing the summary tables below.

| Context | Records | Fields | Cells | Reference | Notes |
| --- | --- | --- | --- | --- | --- |
| forecasting::benchmark_fixed_cadence_periodic_100 | 3223 | 0 | 0 | raw only | Raw-only extraction; collector rows do not contain the evaluated mode. |
| forecasting::benchmark_main_best_periodic | 6090 | 2 | 12180 | results/paper_zero_sparse_benchmark_200k_20260321/collect/forecasting_rows.csv | Verified best-periodic means and selected periodic modes against collector rows. |
| forecasting::canonical_kuramoto_dt0p01 | 350 | 2 | 700 | results/kuramoto_dt0p01_200k_canonical_20260323/collect/forecasting_rows.csv | Verified best-periodic means and selected periodic modes against collector rows. |
| forecasting::hard_clv_4basin_dt0p01 | 520 | 2 | 1040 | results/zero_sparse_hard_systems_20260321/collect/clv_4basin/forecasting_rows.csv | Verified best-periodic means and selected periodic modes against collector rows. |
| forecasting::hard_clv_8basin_dt0p0025 | 350 | 2 | 700 | results/zero_sparse_hard_systems_20260321/collect/clv_8bas_dt0p0025/forecasting_rows.csv | Verified best-periodic means and selected periodic modes against collector rows. |
| forecasting::hard_clv_8basin_dt0p005 | 350 | 2 | 700 | results/zero_sparse_hard_systems_20260321/collect/clv_8bas_dt0p005/forecasting_rows.csv | Verified best-periodic means and selected periodic modes against collector rows. |
| forecasting::hard_hopfield_n16_dt0p00625 | 210 | 2 | 420 | results/zero_sparse_hard_systems_20260321/collect/hopfield_n16/forecasting_rows.csv | Verified best-periodic means and selected periodic modes against collector rows. |
| forecasting::hard_hopfield_n64_dt0p0015625 | 338 | 2 | 676 | results/zero_sparse_hard_systems_20260321/collect/hopfield_n64/forecasting_rows.csv | Verified best-periodic means and selected periodic modes against collector rows. |
| forecasting::hard_kuramoto_dimension_sweep_dt0p00625 | 500 | 2 | 1000 | results/zero_sparse_hard_systems_20260321/collect/kuramoto_dimension/forecasting_rows.csv | Verified best-periodic means and selected periodic modes against collector rows. |
| forecasting::hard_kuramoto_n16_identical_dt0p00625 | 350 | 2 | 700 | results/zero_sparse_hard_systems_20260321/collect/kuramoto_n16_identical/forecasting_rows.csv | Verified best-periodic means and selected periodic modes against collector rows. |
| forecasting::hard_kuramoto_uniform_spread_dt0p00625 | 350 | 2 | 700 | results/zero_sparse_hard_systems_20260321/collect/kuramoto_uniform_spread/forecasting_rows.csv | Verified best-periodic means and selected periodic modes against collector rows. |
| kuramoto_mode_support_audit | 40 | 4 | 160 | results/zero_sparse_mechanisms_20260321/kuramoto_mode_support_audit/summary/kuramoto_mode_support_audit_rows.csv |  |
| label_free::label_free_benchmark | 2460 | 11 | 27060 | results/zero_sparse_mechanisms_20260321/label_free_clustering_v2/label_free_clustering_v2_rows.csv |  |
| label_free::label_free_clv_followup | 360 | 11 | 3960 | results/zero_sparse_mechanisms_20260321/competitive_lv_representation_followup/label_free_clustering_v2/label_free_clustering_v2_rows.csv |  |
| local_linearity | 120 | 9 | 1080 | results/zero_sparse_mechanisms_20260321/lista_support_local_linearity/seed_summary.csv |  |
| support_alignment::support_alignment_benchmark | 550 | 4 | 2200 | results/zero_sparse_mechanisms_20260321/support_alignment_benchmark/support_alignment_rows.csv |  |
| support_alignment::support_alignment_clv_followup | 18 | 4 | 72 | results/zero_sparse_mechanisms_20260321/competitive_lv_representation_followup/support_alignment/support_alignment_rows.csv |  |

## Seed Coverage Audit

| Family | Packet/setting | Unit | Min seeds | Max seeds | Rows |
| --- | --- | --- | --- | --- | --- |
| forecasting | benchmark_fixed_cadence_periodic_100 | system/root/horizon | 1 | 10 | 329 |
| forecasting | benchmark_main_best_periodic | system/root/horizon | 10 | 10 | 609 |
| forecasting | canonical_kuramoto_dt0p01 | system/root/horizon | 10 | 10 | 35 |
| forecasting | hard_clv_4basin_dt0p01 | system/root/horizon | 14 | 15 | 35 |
| forecasting | hard_clv_8basin_dt0p0025 | system/root/horizon | 10 | 10 | 35 |
| forecasting | hard_clv_8basin_dt0p005 | system/root/horizon | 10 | 10 | 35 |
| forecasting | hard_hopfield_n16_dt0p00625 | system/root/horizon | 10 | 10 | 21 |
| forecasting | hard_hopfield_n64_dt0p0015625 | system/root/horizon | 7 | 10 | 35 |
| forecasting | hard_kuramoto_dimension_sweep_dt0p00625 | system/root/horizon | 10 | 10 | 50 |
| forecasting | hard_kuramoto_n16_identical_dt0p00625 | system/root/horizon | 10 | 10 | 35 |
| forecasting | hard_kuramoto_uniform_spread_dt0p00625 | system/root/horizon | 10 | 10 | 35 |
| kuramoto_mode_support_audit | kuramoto_mode_support_audit | root/sampling/metric | 5 | 5 | 32 |
| label_free_clustering | label_free_benchmark | system/root/view/metric | 10 | 10 | 1230 |
| label_free_clustering | label_free_clv_followup | system/root/view/metric | 10 | 10 | 180 |
| local_linearity | lista_support_local_linearity | system/root/metric | 10 | 10 | 72 |
| support_alignment | support_alignment_benchmark | system/root/metric | 10 | 10 | 220 |
| support_alignment | support_alignment_clv_followup | system/root/metric | 3 | 3 | 24 |

Coverage note:
- The benchmark best-periodic packet is 10 seeds throughout, but the raw finite-value audit is not uniformly seed-complete on every retained forecasting slice.
- Fixed-cadence `periodic_100` drops below 10 finite seeds on several late-horizon Hopfield and embedded-multiwell rows.
- Corrected 4-basin CLV block-diagonal LISTA has 14 finite seeds at `H1000-H3000`, and repaired Hopfield `N=64` block-diagonal MLP has 7 finite seeds at `H1500-H3000`.
- The corrected competitive-LV support-alignment branch is still only 3 seeds per root, and the direct Kuramoto mode-support audit is still only 5 seeds per root and sampling strategy.

## Cross-System Forecasting

The tables below replace the cross-system seed medians with statistics over the 29 system-level seed means. The detailed per-system replacements for the built-in and Dysts appendix tables are in:

- `results/paper_seed_statistics_20260331/forecasting_builtin_appendix_seed_stats.csv`
- `results/paper_seed_statistics_20260331/forecasting_dysts_appendix_seed_stats.csv`

### Main 29-system benchmark

| Horizon | Model | Systems | Mean | SD | IQM | Mean 95% CI | IQM 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H100 | Dense LISTA | 29 | 22.223 | 111.046 | 0.0037 | [0.105, 64.697] | [2.470e-04, 0.151] |
| H100 | Sparse MLP | 29 | 5.582 | 20.689 | 0.0034 | [0.0235, 13.323] | [2.297e-04, 0.0378] |
| H100 | Zero MLP | 29 | 17.811 | 85.576 | 0.0038 | [0.0334, 53.266] | [2.988e-04, 0.0527] |
| H500 | Dense LISTA | 29 | 7.281e+23 | 3.921e+24 | 0.0548 | [1.054, 2.184e+24] | [0.0048, 1.311] |
| H500 | Sparse MLP | 29 | 2.117e+18 | 1.135e+19 | 0.0585 | [1.072, 6.343e+18] | [0.0046, 1.532] |
| H500 | Zero MLP | 29 | 5.318e+19 | 2.792e+20 | 0.0610 | [0.490, 1.582e+20] | [0.0078, 0.741] |
| H1000 | Dense LISTA | 29 | 5.201e+22 | 2.801e+23 | 0.0910 | [1.023, 1.560e+23] | [0.0142, 1.447] |
| H1000 | Sparse MLP | 29 | 2.472e+31 | 1.318e+32 | 0.243 | [287.054, 7.392e+31] | [0.0164, 430.087] |
| H1000 | Zero MLP | 29 | 2.148e+25 | 1.157e+26 | 0.148 | [0.333, 6.445e+25] | [0.0286, 0.564] |
| H1500 | Dense LISTA | 29 | 1.630e+31 | 8.726e+31 | 0.177 | [12.996, 4.881e+31] | [0.0335, 20.413] |
| H1500 | Sparse MLP | 29 | 1.690e+23 | 8.559e+23 | 0.328 | [5.695e+04, 4.969e+23] | [0.0422, 8.128e+04] |
| H1500 | Zero MLP | 29 | 9.623e+25 | 5.182e+26 | 0.280 | [0.532, 2.887e+26] | [0.0605, 0.910] |
| H2000 | Dense LISTA | 29 | 2.010e+30 | 1.059e+31 | 0.357 | [1772.851, 5.988e+30] | [0.0696, 1969.779] |
| H2000 | Sparse MLP | 29 | 2.842e+22 | 1.531e+23 | 0.311 | [2.886e+06, 8.527e+22] | [0.0847, 4.387e+06] |
| H2000 | Zero MLP | 29 | 1.490e+21 | 8.022e+21 | 0.386 | [3.037, 4.469e+21] | [0.110, 3.096] |
| H2500 | Dense LISTA | 29 | 9.503e+30 | 5.118e+31 | 0.311 | [3994.108, 2.851e+31] | [0.0943, 5990.274] |
| H2500 | Sparse MLP | 29 | 3.596e+22 | 1.936e+23 | 0.409 | [2.293e+08, 1.079e+23] | [0.128, 3.741e+08] |
| H2500 | Zero MLP | 29 | 1.780e+27 | 9.585e+27 | 0.493 | [3.385, 5.340e+27] | [0.161, 4.961] |
| H3000 | Dense LISTA | 29 | 1.731e+31 | 9.323e+31 | 0.573 | [8.217e+06, 5.194e+31] | [0.153, 1.235e+07] |
| H3000 | Sparse MLP | 29 | 3.596e+22 | 1.936e+23 | 0.507 | [4.328e+10, 1.079e+23] | [0.174, 7.029e+10] |
| H3000 | Zero MLP | 29 | 1.256e+30 | 6.762e+30 | 0.590 | [3.456, 3.767e+30] | [0.220, 5.203] |

### Fixed cadence `periodic_100`

| Horizon | Model | Systems | Mean | SD | IQM | Mean 95% CI | IQM 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H1500 | Dense LISTA | 28 | 7.120e+32 | 3.040e+33 | 5.009e+11 | [6.023e+30, 1.977e+33] | [4.274, 7.667e+30] |
| H1500 | Sparse MLP | 28 | 1.780e+33 | 8.362e+33 | 1.888e+06 | [2.193e+17, 5.134e+33] | [0.0763, 3.416e+17] |
| H1500 | Zero MLP | 27 | 2.623e+27 | 1.352e+28 | 4687.962 | [2.123e+11, 7.848e+27] | [0.136, 4.906e+11] |
| H2000 | Dense LISTA | 28 | 7.244e+32 | 3.054e+33 | 6.966e+18 | [4.829e+31, 1.943e+33] | [572.538, 1.068e+32] |
| H2000 | Sparse MLP | 27 | 7.740e+32 | 3.989e+33 | 1.840e+09 | [1.939e+24, 2.316e+33] | [0.163, 9.176e+24] |
| H2000 | Zero MLP | 27 | 7.320e+31 | 3.804e+32 | 1.163e+07 | [6.860e+17, 2.196e+32] | [0.244, 6.163e+19] |
| H2500 | Dense LISTA | 28 | 2.195e+33 | 8.554e+33 | 1.156e+25 | [8.051e+31, 5.936e+33] | [8332.986, 1.288e+32] |
| H2500 | Sparse MLP | 27 | 1.161e+32 | 3.573e+32 | 7.378e+12 | [1.646e+31, 2.638e+32] | [0.270, 4.141e+31] |
| H2500 | Zero MLP | 27 | 5.741e+32 | 2.980e+33 | 4.882e+10 | [4.810e+24, 1.722e+33] | [0.381, 8.262e+25] |
| H3000 | Dense LISTA | 28 | 1.173e+33 | 5.272e+33 | 9.350e+28 | [5.352e+31, 3.243e+33] | [8.487e+04, 1.060e+32] |
| H3000 | Sparse MLP | 27 | 2.485e+33 | 1.210e+34 | 3.004e+16 | [1.976e+31, 7.260e+33] | [0.360, 4.663e+31] |
| H3000 | Zero MLP | 27 | 3.921e+31 | 1.688e+32 | 2.241e+14 | [7.634e+29, 1.089e+32] | [0.520, 1.586e+30] |


## Hard-System Forecasting

Full horizon-by-horizon stats are in `results/paper_seed_statistics_20260331/forecasting_hard_system_seed_stats.csv`.


### Corrected 4-basin CLV dt=0.01

| Model | Horizon | Seeds | Mean | SD | IQM | Mean 95% CI | IQM 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Block-diag. LISTA | H1000 | 14 | 1.330e+35 | 4.869e+35 | 0.244 | [0.263, 3.962e+35] | [0.145, 7.999e+33] |
| Block-diag. MLP | H1000 | 15 | 1.903e+32 | 7.370e+32 | 0.289 | [0.259, 5.709e+32] | [0.173, 0.651] |
| Dense LISTA | H1000 | 15 | 0.158 | 0.0515 | 0.148 | [0.139, 0.187] | [0.135, 0.160] |
| Sparse MLP | H1000 | 15 | 0.191 | 0.125 | 0.152 | [0.146, 0.259] | [0.136, 0.190] |
| Zero MLP | H1000 | 15 | 0.132 | 0.0254 | 0.129 | [0.120, 0.144] | [0.115, 0.145] |
| Block-diag. LISTA | H3000 | 14 | 1.330e+35 | 4.869e+35 | 0.385 | [0.410, 3.962e+35] | [0.197, 8.030e+33] |
| Block-diag. MLP | H3000 | 15 | 5.730e+25 | 2.219e+26 | 0.436 | [0.374, 1.719e+26] | [0.274, 0.842] |
| Dense LISTA | H3000 | 15 | 0.239 | 0.112 | 0.220 | [0.191, 0.302] | [0.173, 0.272] |
| Sparse MLP | H3000 | 15 | 0.304 | 0.243 | 0.244 | [0.217, 0.435] | [0.195, 0.308] |
| Zero MLP | H3000 | 15 | 0.223 | 0.0847 | 0.206 | [0.185, 0.268] | [0.168, 0.262] |

### Fixed 8-basin CLV dt=0.0025

| Model | Horizon | Seeds | Mean | SD | IQM | Mean 95% CI | IQM 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Block-diag. LISTA | H1000 | 10 | 0.667 | 0.495 | 0.532 | [0.474, 0.999] | [0.469, 0.731] |
| Block-diag. MLP | H1000 | 10 | 0.617 | 0.466 | 0.490 | [0.435, 0.926] | [0.426, 0.673] |
| Dense LISTA | H1000 | 10 | 7.838e+10 | 2.479e+11 | 0.523 | [0.454, 2.351e+11] | [0.443, 7.838e+10] |
| Sparse MLP | H1000 | 10 | 0.904 | 1.856 | 0.252 | [0.237, 2.091] | [0.229, 0.991] |
| Zero MLP | H1000 | 10 | 0.284 | 0.0526 | 0.288 | [0.254, 0.314] | [0.244, 0.327] |
| Block-diag. LISTA | H3000 | 10 | 0.911 | 0.698 | 0.710 | [0.644, 1.367] | [0.643, 0.989] |
| Block-diag. MLP | H3000 | 10 | 0.892 | 0.744 | 0.682 | [0.609, 1.378] | [0.590, 0.972] |
| Dense LISTA | H3000 | 10 | 0.685 | 0.148 | 0.683 | [0.598, 0.768] | [0.592, 0.810] |
| Sparse MLP | H3000 | 10 | 1.172 | 2.234 | 0.403 | [0.378, 2.601] | [0.366, 1.268] |
| Zero MLP | H3000 | 10 | 0.396 | 0.0629 | 0.398 | [0.359, 0.434] | [0.351, 0.448] |

### Fixed 8-basin CLV dt=0.005

| Model | Horizon | Seeds | Mean | SD | IQM | Mean 95% CI | IQM 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Block-diag. LISTA | H1000 | 10 | 1.050e+35 | 2.525e+35 | 9.537e+32 | [0.404, 2.634e+35] | [0.322, 1.788e+35] |
| Block-diag. MLP | H1000 | 10 | 2.466 | 4.613 | 0.548 | [0.386, 5.700] | [0.307, 3.920] |
| Dense LISTA | H1000 | 10 | 0.575 | 0.0768 | 0.570 | [0.533, 0.622] | [0.518, 0.616] |
| Sparse MLP | H1000 | 10 | 0.519 | 0.100 | 0.526 | [0.459, 0.575] | [0.456, 0.600] |
| Zero MLP | H1000 | 10 | 0.431 | 0.117 | 0.403 | [0.372, 0.510] | [0.360, 0.478] |
| Block-diag. LISTA | H3000 | 10 | 1.050e+35 | 2.525e+35 | 9.537e+32 | [0.427, 2.639e+35] | [0.379, 1.788e+35] |
| Block-diag. MLP | H3000 | 10 | 2.628 | 4.643 | 0.669 | [0.583, 5.758] | [0.327, 4.088] |
| Dense LISTA | H3000 | 10 | 0.676 | 0.0670 | 0.679 | [0.637, 0.716] | [0.628, 0.719] |
| Sparse MLP | H3000 | 10 | 0.622 | 0.120 | 0.635 | [0.551, 0.688] | [0.559, 0.710] |
| Zero MLP | H3000 | 10 | 0.518 | 0.134 | 0.485 | [0.451, 0.604] | [0.441, 0.577] |

### Hopfield N=16 dt=0.00625

| Model | Horizon | Seeds | Mean | SD | IQM | Mean 95% CI | IQM 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Block-diag. LISTA | H1000 | 10 | 6.842 | 3.284 | 5.906 | [5.091, 8.892] | [4.731, 8.799] |
| Sparse MLP | H1000 | 10 | 3.036 | 1.396 | 2.798 | [2.307, 3.914] | [2.033, 3.751] |
| Zero MLP | H1000 | 10 | 2.717 | 1.005 | 2.679 | [2.123, 3.280] | [1.952, 3.664] |
| Block-diag. LISTA | H3000 | 10 | 13.701 | 3.299 | 13.827 | [11.721, 15.635] | [11.504, 16.190] |
| Sparse MLP | H3000 | 10 | 9.665 | 2.737 | 9.300 | [8.090, 11.313] | [7.824, 11.518] |
| Zero MLP | H3000 | 10 | 9.991 | 9.544 | 7.030 | [6.076, 16.369] | [5.568, 12.092] |

### Hopfield N=64 dt=0.0015625

| Model | Horizon | Seeds | Mean | SD | IQM | Mean 95% CI | IQM 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Block-diag. LISTA | H1000 | 10 | 343.188 | 52.591 | 345.631 | [312.159, 373.566] | [308.600, 382.794] |
| Block-diag. MLP | H1000 | 10 | 222.255 | 126.886 | 218.905 | [150.548, 298.875] | [117.122, 300.825] |
| Dense LISTA | H1000 | 10 | 299.556 | 78.502 | 283.748 | [258.513, 349.147] | [247.630, 343.639] |
| Sparse MLP | H1000 | 10 | 104.137 | 90.842 | 70.894 | [56.600, 161.407] | [47.098, 156.560] |
| Zero MLP | H1000 | 10 | 71.502 | 27.493 | 66.984 | [57.540, 89.236] | [53.862, 84.251] |
| Block-diag. LISTA | H3000 | 10 | 1297.219 | 232.502 | 1301.535 | [1163.619, 1429.653] | [1128.530, 1467.285] |
| Block-diag. MLP | H3000 | 7 | 1329.988 | 1050.410 | 996.285 | [827.426, 2128.000] | [787.067, 2035.571] |
| Dense LISTA | H3000 | 10 | 1152.469 | 327.711 | 1086.492 | [979.268, 1360.134] | [924.736, 1347.857] |
| Sparse MLP | H3000 | 10 | 649.928 | 277.340 | 612.102 | [497.275, 815.635] | [425.460, 866.397] |
| Zero MLP | H3000 | 10 | 554.796 | 151.186 | 526.925 | [472.148, 645.684] | [453.934, 642.472] |

### Kuramoto N=16 identical dt=0.00625

| Model | Horizon | Seeds | Mean | SD | IQM | Mean 95% CI | IQM 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Block-diag. LISTA | H1000 | 10 | 7.070 | 0.235 | 7.032 | [6.941, 7.213] | [6.902, 7.206] |
| Block-diag. MLP | H1000 | 10 | 5.927 | 0.862 | 6.280 | [5.365, 6.357] | [5.449, 6.411] |
| Dense LISTA | H1000 | 10 | 18.948 | 8.924 | 16.819 | [14.503, 24.771] | [13.255, 22.690] |
| Sparse MLP | H1000 | 10 | 31.312 | 6.102 | 30.645 | [27.829, 34.879] | [26.480, 36.150] |
| Zero MLP | H1000 | 10 | 9.205 | 0.821 | 9.115 | [8.743, 9.694] | [8.589, 9.826] |
| Block-diag. LISTA | H3000 | 10 | 22.512 | 3.529 | 21.715 | [20.869, 24.950] | [20.985, 23.301] |
| Block-diag. MLP | H3000 | 10 | 17.651 | 2.112 | 17.072 | [16.523, 18.992] | [16.585, 19.127] |
| Dense LISTA | H3000 | 10 | 2.218e+07 | 6.722e+07 | 5.462e+04 | [1.018e+04, 6.494e+07] | [547.844, 2.310e+07] |
| Sparse MLP | H3000 | 10 | 8.900e+06 | 5.409e+06 | 9.128e+06 | [5.860e+06, 1.219e+07] | [4.550e+06, 1.173e+07] |
| Zero MLP | H3000 | 10 | 5576.619 | 7725.240 | 3212.600 | [2299.922, 1.070e+04] | [2016.828, 7154.371] |

### Kuramoto N=16 uniform-spread dt=0.00625

| Model | Horizon | Seeds | Mean | SD | IQM | Mean 95% CI | IQM 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Block-diag. LISTA | H1000 | 10 | 10.410 | 1.087 | 10.392 | [9.786, 11.051] | [9.523, 11.158] |
| Block-diag. MLP | H1000 | 10 | 7.746 | 0.955 | 8.021 | [7.133, 8.226] | [7.333, 8.337] |
| Dense LISTA | H1000 | 10 | 17.954 | 3.559 | 17.850 | [15.944, 20.174] | [16.049, 19.674] |
| Sparse MLP | H1000 | 10 | 41.668 | 5.618 | 41.075 | [38.370, 44.957] | [37.158, 46.093] |
| Zero MLP | H1000 | 10 | 9.035 | 1.125 | 8.810 | [8.465, 9.778] | [8.334, 9.467] |
| Block-diag. LISTA | H3000 | 10 | 2.556e+04 | 3.714e+04 | 1.007e+04 | [6738.624, 5.039e+04] | [3759.742, 5.146e+04] |
| Block-diag. MLP | H3000 | 10 | 1382.175 | 477.388 | 1433.056 | [1102.120, 1650.012] | [1048.095, 1752.921] |
| Dense LISTA | H3000 | 10 | 1.393e+07 | 4.309e+07 | 1.680e+05 | [1.088e+05, 4.129e+07] | [1763.191, 1.428e+07] |
| Sparse MLP | H3000 | 10 | 1.728e+08 | 4.877e+08 | 1.487e+07 | [1.103e+07, 4.845e+08] | [8.492e+06, 1.882e+08] |
| Zero MLP | H3000 | 10 | 1.319e+04 | 3.588e+04 | 1923.638 | [1170.270, 3.627e+04] | [661.370, 1.445e+04] |

### Kuramoto dimension sweep dt=0.00625


**N=16**

| Model | Horizon | Seeds | Mean | SD | IQM | Mean 95% CI | IQM 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Block-diag. LISTA | H1000 | 10 | 7.104 | 0.245 | 7.102 | [6.953, 7.246] | [6.958, 7.290] |
| Block-diag. MLP | H1000 | 10 | 5.909 | 0.856 | 6.208 | [5.377, 6.372] | [5.397, 6.480] |
| Dense LISTA | H1000 | 10 | 16.346 | 7.574 | 13.429 | [12.425, 21.381] | [11.722, 21.848] |
| Sparse MLP | H1000 | 10 | 33.088 | 8.319 | 31.359 | [28.720, 38.239] | [27.915, 36.827] |
| Zero MLP | H1000 | 10 | 9.521 | 1.449 | 9.078 | [8.747, 10.457] | [8.522, 10.425] |
| Block-diag. LISTA | H3000 | 10 | 24.414 | 7.843 | 22.106 | [20.973, 29.666] | [20.470, 26.012] |
| Block-diag. MLP | H3000 | 10 | 16.685 | 0.849 | 16.885 | [16.151, 17.144] | [16.245, 17.192] |
| Dense LISTA | H3000 | 10 | 5456.331 | 1.357e+04 | 611.725 | [375.868, 1.434e+04] | [210.087, 7082.037] |
| Sparse MLP | H3000 | 10 | 2.753e+07 | 4.275e+07 | 1.004e+07 | [7.789e+06, 5.546e+07] | [6.243e+06, 4.515e+07] |
| Zero MLP | H3000 | 10 | 1.872e+04 | 4.590e+04 | 3695.520 | [2814.108, 4.817e+04] | [2298.920, 2.123e+04] |

**N=24**

| Model | Horizon | Seeds | Mean | SD | IQM | Mean 95% CI | IQM 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Block-diag. LISTA | H1000 | 10 | 6.736 | 0.604 | 6.563 | [6.458, 7.152] | [6.464, 6.880] |
| Block-diag. MLP | H1000 | 10 | 5.542 | 0.340 | 5.612 | [5.327, 5.729] | [5.307, 5.806] |
| Dense LISTA | H1000 | 10 | 26.281 | 20.517 | 19.020 | [15.191, 39.049] | [12.571, 38.022] |
| Sparse MLP | H1000 | 10 | 7.644 | 1.214 | 7.492 | [6.975, 8.377] | [6.680, 8.327] |
| Zero MLP | H1000 | 10 | 7.401 | 0.503 | 7.253 | [7.143, 7.724] | [7.062, 7.692] |
| Block-diag. LISTA | H3000 | 10 | 1559.468 | 4837.312 | 26.170 | [24.955, 4620.434] | [23.943, 1565.173] |
| Block-diag. MLP | H3000 | 10 | 19.102 | 1.175 | 19.099 | [18.412, 19.798] | [18.249, 19.798] |
| Dense LISTA | H3000 | 10 | 7.145e+07 | 2.231e+08 | 1.323e+04 | [3929.576, 2.127e+08] | [481.972, 7.228e+07] |
| Sparse MLP | H3000 | 10 | 5.517e+04 | 1.137e+05 | 7017.227 | [5056.047, 1.245e+05] | [3524.289, 9.991e+04] |
| Zero MLP | H3000 | 10 | 38.754 | 8.140 | 37.917 | [34.260, 43.735] | [32.210, 45.032] |

**N=32**

| Model | Horizon | Seeds | Mean | SD | IQM | Mean 95% CI | IQM 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Block-diag. LISTA | H1000 | 10 | 6.130 | 0.370 | 6.129 | [5.913, 6.344] | [5.827, 6.433] |
| Block-diag. MLP | H1000 | 10 | 5.195 | 0.0753 | 5.207 | [5.150, 5.239] | [5.137, 5.259] |
| Dense LISTA | H1000 | 10 | 75.876 | 35.070 | 91.774 | [51.232, 92.956] | [50.942, 93.982] |
| Sparse MLP | H1000 | 10 | 7.536 | 1.155 | 7.201 | [6.922, 8.253] | [6.728, 8.140] |
| Zero MLP | H1000 | 10 | 7.176 | 0.903 | 6.991 | [6.782, 7.774] | [6.729, 7.374] |
| Block-diag. LISTA | H3000 | 10 | 31.904 | 9.715 | 29.911 | [26.788, 38.108] | [24.608, 37.079] |
| Block-diag. MLP | H3000 | 10 | 20.415 | 0.168 | 20.415 | [20.319, 20.511] | [20.308, 20.537] |
| Dense LISTA | H3000 | 10 | 138.866 | 68.120 | 121.072 | [103.431, 182.067] | [103.691, 180.136] |
| Sparse MLP | H3000 | 10 | 151.387 | 109.308 | 113.327 | [90.010, 221.561] | [77.229, 231.038] |
| Zero MLP | H3000 | 10 | 38.605 | 8.953 | 36.316 | [34.893, 44.642] | [34.482, 40.219] |

**N=64**

| Model | Horizon | Seeds | Mean | SD | IQM | Mean 95% CI | IQM 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Block-diag. LISTA | H1000 | 10 | 89.698 | 102.993 | 71.905 | [29.467, 149.805] | [8.150, 190.277] |
| Block-diag. MLP | H1000 | 10 | 127.641 | 104.787 | 147.834 | [66.714, 188.498] | [26.221, 208.899] |
| Dense LISTA | H1000 | 10 | 208.526 | 0.508 | 208.553 | [208.220, 208.815] | [208.199, 208.879] |
| Sparse MLP | H1000 | 10 | 188.707 | 63.401 | 208.767 | [148.533, 208.941] | [188.358, 208.985] |
| Zero MLP | H1000 | 10 | 8.310 | 0.853 | 8.148 | [7.833, 8.820] | [7.735, 8.880] |
| Block-diag. LISTA | H3000 | 10 | 119.859 | 92.566 | 113.814 | [67.819, 172.325] | [39.606, 210.498] |
| Block-diag. MLP | H3000 | 10 | 144.314 | 102.904 | 164.165 | [84.465, 204.134] | [44.798, 224.121] |
| Dense LISTA | H3000 | 10 | 224.093 | 0.589 | 224.133 | [223.742, 224.425] | [223.722, 224.476] |
| Sparse MLP | H3000 | 10 | 205.794 | 57.648 | 224.007 | [169.251, 224.210] | [205.463, 224.283] |
| Zero MLP | H3000 | 10 | 49.367 | 4.403 | 48.974 | [46.829, 51.954] | [46.013, 52.763] |

**N=8**

| Model | Horizon | Seeds | Mean | SD | IQM | Mean 95% CI | IQM 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Block-diag. LISTA | H1000 | 10 | 10.693 | 3.273 | 9.678 | [8.953, 12.756] | [8.153, 13.148] |
| Block-diag. MLP | H1000 | 10 | 9.492 | 2.405 | 9.930 | [8.055, 10.871] | [7.372, 11.349] |
| Dense LISTA | H1000 | 10 | 405.008 | 330.825 | 323.601 | [229.665, 616.701] | [181.102, 629.471] |
| Sparse MLP | H1000 | 10 | 939.061 | 922.786 | 806.715 | [415.745, 1498.074] | [93.037, 1601.637] |
| Zero MLP | H1000 | 10 | 75.833 | 130.130 | 39.237 | [29.497, 160.402] | [28.393, 83.243] |
| Block-diag. LISTA | H3000 | 10 | 76.777 | 25.445 | 72.222 | [62.431, 92.341] | [55.755, 98.154] |
| Block-diag. MLP | H3000 | 10 | 25.050 | 6.424 | 25.518 | [21.190, 28.777] | [19.587, 29.919] |
| Dense LISTA | H3000 | 10 | 4.891e+11 | 1.072e+12 | 7.644e+10 | [4.277e+10, 1.220e+12] | [6.713e+08, 6.663e+11] |
| Sparse MLP | H3000 | 10 | 1.130e+13 | 2.969e+13 | 1.417e+12 | [7.644e+11, 3.074e+13] | [3.145e+10, 1.331e+13] |
| Zero MLP | H3000 | 10 | 6.336e+08 | 2.003e+09 | 1.666e+05 | [9.197e+04, 1.900e+09] | [3.682e+04, 6.337e+08] |


## Basin-Support And Mechanism

### Support Alignment

Detailed per-system support-alignment stats are in `results/paper_seed_statistics_20260331/support_alignment_seed_stats.csv`.

The benchmark multiwell family below uses the paper-facing sparse / zero / dense / block-diagonal-LISTA packet, with the block-diagonal LISTA root aligned to `sc6em3`. The corrected competitive-LV single-system rows use the `sc3em3` block-diagonal LISTA root because that is the root used in the March 14 support-reuse table.

**Multiwell family aggregate over system-level seed means**

| Metric | Model | Systems | Mean | SD | IQM | Mean 95% CI | Range of system means |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cons | Block-diag. LISTA | 8 | 0.0985 | 0.0393 | 0.101 | [0.0734, 0.123] | 0.0534 .. 0.147 |
| Cons | Dense LISTA | 8 | 0.102 | 0.0445 | 0.102 | [0.0746, 0.131] | 0.0534 .. 0.166 |
| Cons | Sparse MLP | 8 | 0.0874 | 0.0314 | 0.0862 | [0.0681, 0.108] | 0.0534 .. 0.138 |
| Cons | Zero MLP | 8 | 0.131 | 0.0345 | 0.132 | [0.108, 0.153] | 0.0873 .. 0.170 |
| CosSep | Block-diag. LISTA | 8 | 0.499 | 0.139 | 0.520 | [0.406, 0.581] | 0.228 .. 0.656 |
| CosSep | Dense LISTA | 8 | 0.424 | 0.144 | 0.418 | [0.330, 0.517] | 0.176 .. 0.607 |
| CosSep | Sparse MLP | 8 | 0.570 | 0.113 | 0.563 | [0.496, 0.641] | 0.376 .. 0.726 |
| CosSep | Zero MLP | 8 | 0.197 | 0.132 | 0.186 | [0.113, 0.280] | 0.0515 .. 0.389 |
| Uniq | Block-diag. LISTA | 8 | 0.936 | 0.0702 | 0.946 | [0.891, 0.982] | 0.852 .. 1.000 |
| Uniq | Dense LISTA | 8 | 0.940 | 0.0681 | 0.953 | [0.895, 0.982] | 0.848 .. 1.000 |
| Uniq | Sparse MLP | 8 | 0.960 | 0.0451 | 0.966 | [0.931, 0.988] | 0.905 .. 1.000 |
| Uniq | Zero MLP | 8 | 0.891 | 0.0556 | 0.902 | [0.852, 0.924] | 0.815 .. 0.953 |

**Single-system support-alignment rows**

| Case | Metric | Model | Seeds | Mean | SD | IQM | Mean 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Competitive LV | Cons | Block-diag. LISTA | 3 | 0.107 | 0 | 0.107 | [0.107, 0.107] |
| Competitive LV | Cons | Dense LISTA | 3 | 0.107 | 0 | 0.107 | [0.107, 0.107] |
| Competitive LV | Cons | Sparse MLP | 3 | 0.107 | 0 | 0.107 | [0.107, 0.107] |
| Competitive LV | Cons | Zero MLP | 3 | 0.794 | 0.164 | 0.837 | [0.605, 0.898] |
| Competitive LV | CosSep | Block-diag. LISTA | 3 | -0.0680 | 0.0044 | -0.0681 | [-0.0725, -0.0636] |
| Competitive LV | CosSep | Dense LISTA | 3 | -0.0618 | 0.0054 | -0.0615 | [-0.0675, -0.0568] |
| Competitive LV | CosSep | Sparse MLP | 3 | -0.0537 | 0.0046 | -0.0547 | [-0.0571, -0.0485] |
| Competitive LV | CosSep | Zero MLP | 3 | -0.0089 | 4.907e-04 | -0.0090 | [-0.0094, -0.0084] |
| Competitive LV | Uniq | Block-diag. LISTA | 3 | 1.000 | 0 | 1.000 | [1.000, 1.000] |
| Competitive LV | Uniq | Dense LISTA | 3 | 1.000 | 0 | 1.000 | [1.000, 1.000] |
| Competitive LV | Uniq | Sparse MLP | 3 | 1.000 | 0 | 1.000 | [1.000, 1.000] |
| Competitive LV | Uniq | Zero MLP | 3 | 0.0533 | 0.0115 | 0.0567 | [0.0400, 0.0600] |
| Duffing | Cons | Block-diag. LISTA | 10 | 0.0667 | 0.0150 | 0.0647 | [0.0586, 0.0759] |
| Duffing | Cons | Dense LISTA | 10 | 0.0511 | 0.0111 | 0.0542 | [0.0442, 0.0571] |
| Duffing | Cons | Sparse MLP | 10 | 0.0649 | 0.0212 | 0.0610 | [0.0537, 0.0780] |
| Duffing | Cons | Zero MLP | 10 | 0.110 | 0.0162 | 0.111 | [0.100, 0.119] |
| Duffing | CosSep | Block-diag. LISTA | 10 | -0.118 | 0.0063 | -0.118 | [-0.122, -0.115] |
| Duffing | CosSep | Dense LISTA | 10 | -0.112 | 0.0082 | -0.113 | [-0.117, -0.107] |
| Duffing | CosSep | Sparse MLP | 10 | -0.0838 | 0.0073 | -0.0832 | [-0.0881, -0.0796] |
| Duffing | CosSep | Zero MLP | 10 | -0.0451 | 0.0029 | -0.0447 | [-0.0470, -0.0436] |
| Duffing | Uniq | Block-diag. LISTA | 10 | 0.901 | 0.0331 | 0.909 | [0.880, 0.919] |
| Duffing | Uniq | Dense LISTA | 10 | 0.917 | 0.0258 | 0.911 | [0.904, 0.934] |
| Duffing | Uniq | Sparse MLP | 10 | 0.905 | 0.0276 | 0.903 | [0.889, 0.921] |
| Duffing | Uniq | Zero MLP | 10 | 0.622 | 0.0588 | 0.629 | [0.588, 0.656] |
| Hopfield | Cons | Block-diag. LISTA | 10 | 0.143 | 0.0688 | 0.172 | [0.100, 0.186] |
| Hopfield | Cons | Dense LISTA | 10 | 0.143 | 0.0688 | 0.172 | [0.100, 0.172] |
| Hopfield | Cons | Sparse MLP | 10 | 0.147 | 0.0722 | 0.172 | [0.100, 0.189] |
| Hopfield | Cons | Zero MLP | 10 | 0.167 | 0.0593 | 0.168 | [0.132, 0.201] |
| Hopfield | CosSep | Block-diag. LISTA | 10 | 0.617 | 0.0834 | 0.598 | [0.571, 0.667] |
| Hopfield | CosSep | Dense LISTA | 10 | 0.666 | 0.0536 | 0.682 | [0.632, 0.693] |
| Hopfield | CosSep | Sparse MLP | 10 | 0.527 | 0.0768 | 0.538 | [0.480, 0.569] |
| Hopfield | CosSep | Zero MLP | 10 | 0.313 | 0.0326 | 0.320 | [0.294, 0.332] |
| Hopfield | Uniq | Block-diag. LISTA | 10 | 1.000 | 0 | 1.000 | [1.000, 1.000] |
| Hopfield | Uniq | Dense LISTA | 10 | 1.000 | 0 | 1.000 | [1.000, 1.000] |
| Hopfield | Uniq | Sparse MLP | 10 | 0.976 | 0.0759 | 1.000 | [0.928, 1.000] |
| Hopfield | Uniq | Zero MLP | 10 | 0.722 | 0.152 | 0.739 | [0.628, 0.807] |
| Kuramoto | Cons | Block-diag. LISTA | 10 | 0.424 | 0 | 0.424 | [0.424, 0.424] |
| Kuramoto | Cons | Dense LISTA | 10 | 0.424 | 0 | 0.424 | [0.424, 0.424] |
| Kuramoto | Cons | Sparse MLP | 10 | 0.424 | 0 | 0.424 | [0.424, 0.424] |
| Kuramoto | Cons | Zero MLP | 10 | 0.426 | 0.0071 | 0.424 | [0.424, 0.431] |
| Kuramoto | CosSep | Block-diag. LISTA | 10 | -0.295 | 0.0159 | -0.294 | [-0.304, -0.285] |
| Kuramoto | CosSep | Dense LISTA | 10 | -0.278 | 0.0270 | -0.274 | [-0.294, -0.262] |
| Kuramoto | CosSep | Sparse MLP | 10 | -0.298 | 0.0143 | -0.299 | [-0.306, -0.290] |
| Kuramoto | CosSep | Zero MLP | 10 | -0.0728 | 0.0037 | -0.0726 | [-0.0751, -0.0706] |
| Kuramoto | Uniq | Block-diag. LISTA | 10 | 1.000 | 0 | 1.000 | [1.000, 1.000] |
| Kuramoto | Uniq | Dense LISTA | 10 | 1.000 | 0 | 1.000 | [1.000, 1.000] |
| Kuramoto | Uniq | Sparse MLP | 10 | 1.000 | 0 | 1.000 | [1.000, 1.000] |
| Kuramoto | Uniq | Zero MLP | 10 | 0.983 | 0.0469 | 0.999 | [0.953, 1.000] |


### Label-Free Clustering

All per-view stats are in `results/paper_seed_statistics_20260331/label_free_clustering_view_seed_stats.csv`, and the selected-view replacements are in `results/paper_seed_statistics_20260331/label_free_clustering_selected_seed_stats.csv`.

The selected-view rows keep the same historical view-selection rule as the March 14 tables, so only the across-seed summary statistic changes here. The benchmark cases therefore use the best feature view by seed-median ARI, and corrected competitive-LV uses the best support view by seed-median ARI.

**Multiwell family aggregate over selected-view system means**

| Model | Systems | Mean ARI | SD | IQM | Mean 95% CI | Range of system means |
| --- | --- | --- | --- | --- | --- | --- |
| Block-diag. LISTA | 8 | 0.911 | 0.0878 | 0.941 | [0.846, 0.959] | 0.717 .. 0.978 |
| Dense LISTA | 8 | 0.893 | 0.0943 | 0.911 | [0.829, 0.949] | 0.726 .. 0.993 |
| Sparse MLP | 8 | 0.974 | 0.0133 | 0.972 | [0.965, 0.983] | 0.954 .. 0.994 |
| Zero MLP | 8 | 0.891 | 0.145 | 0.931 | [0.790, 0.971] | 0.588 .. 1.000 |

**Duffing, Kuramoto, and corrected competitive-LV selected-view ARI**

| Case | Model | Selected view | Seeds | Mean ARI | SD | IQM | Mean 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Competitive LV | Block-diag. LISTA | last_step_support | 10 | 0.440 | 0.485 | 0.378 | [0.164, 0.726] |
| Competitive LV | Dense LISTA | last_step_support | 10 | 0.453 | 0.477 | 0.401 | [0.174, 0.734] |
| Competitive LV | Sparse MLP | last_step_support | 10 | 0.453 | 0.477 | 0.403 | [0.184, 0.735] |
| Competitive LV | Zero MLP | traj_mean_support | 10 | 0.433 | 0.494 | 0.368 | [0.154, 0.728] |
| Duffing | Block-diag. LISTA | modal_support | 10 | 0.241 | 0.0459 | 0.239 | [0.215, 0.268] |
| Duffing | Dense LISTA | modal_support | 10 | 0.220 | 0.0739 | 0.234 | [0.173, 0.257] |
| Duffing | Sparse MLP | majority_support | 10 | 0.253 | 0.0291 | 0.248 | [0.236, 0.270] |
| Duffing | Zero MLP | majority_support | 10 | 0.225 | 0.0143 | 0.224 | [0.217, 0.233] |
| Kuramoto | Block-diag. LISTA | modal_support | 10 | 0.0020 | 0.0058 | 9.071e-04 | [-0.0013, 0.0055] |
| Kuramoto | Dense LISTA | traj_mean_support | 10 | 0.0019 | 0.0069 | 0.0012 | [-0.0020, 0.0062] |
| Kuramoto | Sparse MLP | modal_support | 10 | 0.0027 | 0.0043 | 0.0024 | [2.884e-04, 0.0054] |
| Kuramoto | Zero MLP | modal_support | 10 | 0.0014 | 0.0042 | 6.615e-04 | [-9.341e-04, 0.0040] |


### Direct Kuramoto Mode-Support Audit

Detailed stats are in `results/paper_seed_statistics_20260331/kuramoto_mode_support_audit_seed_stats.csv`. This packet is still only 5 seeds per root and sampling strategy.

| Sampling | Metric | Model | Seeds | Mean | SD | IQM | Mean 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | Cons | Block-diag. LISTA | 5 | 0.0625 | 0 | 0.0625 | [0.0625, 0.0625] |
| balanced | Cons | Dense LISTA | 5 | 0.0625 | 0 | 0.0625 | [0.0625, 0.0625] |
| balanced | Cons | Sparse MLP | 5 | 0.0625 | 0 | 0.0625 | [0.0625, 0.0625] |
| balanced | Cons | Zero MLP | 5 | 0.0625 | 0 | 0.0625 | [0.0625, 0.0625] |
| balanced | HammingRatio | Block-diag. LISTA | 5 | 1.000 | 0.0035 | 1.001 | [0.997, 1.003] |
| balanced | HammingRatio | Dense LISTA | 5 | 1.001 | 0.0024 | 1.002 | [0.999, 1.003] |
| balanced | HammingRatio | Sparse MLP | 5 | 0.995 | 0.0025 | 0.995 | [0.994, 0.997] |
| balanced | HammingRatio | Zero MLP | 5 | 0.994 | 0.0046 | 0.994 | [0.990, 0.997] |
| balanced | TrajUniq | Block-diag. LISTA | 5 | 1.000 | 0 | 1.000 | [1.000, 1.000] |
| balanced | TrajUniq | Dense LISTA | 5 | 1.000 | 0 | 1.000 | [1.000, 1.000] |
| balanced | TrajUniq | Sparse MLP | 5 | 1.000 | 0 | 1.000 | [1.000, 1.000] |
| balanced | TrajUniq | Zero MLP | 5 | 1.000 | 0 | 1.000 | [1.000, 1.000] |
| balanced | UniqModes | Block-diag. LISTA | 5 | 5.000 | 0 | 5.000 | [5.000, 5.000] |
| balanced | UniqModes | Dense LISTA | 5 | 5.000 | 0 | 5.000 | [5.000, 5.000] |
| balanced | UniqModes | Sparse MLP | 5 | 5.000 | 0 | 5.000 | [5.000, 5.000] |
| balanced | UniqModes | Zero MLP | 5 | 5.000 | 0 | 5.000 | [5.000, 5.000] |
| random | Cons | Block-diag. LISTA | 5 | 0.309 | 0 | 0.309 | [0.309, 0.309] |
| random | Cons | Dense LISTA | 5 | 0.309 | 0 | 0.309 | [0.309, 0.309] |
| random | Cons | Sparse MLP | 5 | 0.309 | 0 | 0.309 | [0.309, 0.309] |
| random | Cons | Zero MLP | 5 | 0.310 | 0.0019 | 0.310 | [0.309, 0.312] |
| random | HammingRatio | Block-diag. LISTA | 5 | 1.002 | 0.0021 | 1.002 | [1.000, 1.004] |
| random | HammingRatio | Dense LISTA | 5 | 1.003 | 0.0020 | 1.003 | [1.002, 1.005] |
| random | HammingRatio | Sparse MLP | 5 | 1.003 | 0.0018 | 1.004 | [1.002, 1.005] |
| random | HammingRatio | Zero MLP | 5 | 0.997 | 0.0081 | 0.995 | [0.991, 1.003] |
| random | TrajUniq | Block-diag. LISTA | 5 | 1.000 | 0 | 1.000 | [1.000, 1.000] |
| random | TrajUniq | Dense LISTA | 5 | 1.000 | 0 | 1.000 | [1.000, 1.000] |
| random | TrajUniq | Sparse MLP | 5 | 1.000 | 0 | 1.000 | [1.000, 1.000] |
| random | TrajUniq | Zero MLP | 5 | 0.997 | 0.0017 | 0.996 | [0.996, 0.998] |
| random | UniqModes | Block-diag. LISTA | 5 | 5.000 | 0 | 5.000 | [5.000, 5.000] |
| random | UniqModes | Dense LISTA | 5 | 5.000 | 0 | 5.000 | [5.000, 5.000] |
| random | UniqModes | Sparse MLP | 5 | 5.000 | 0 | 5.000 | [5.000, 5.000] |
| random | UniqModes | Zero MLP | 5 | 5.000 | 0 | 5.000 | [5.000, 5.000] |


### Recurring-Support Local Linearity

Detailed stats are in `results/paper_seed_statistics_20260331/local_linearity_seed_stats.csv`.

**Multiwell strong transition**

| Model | Ok seeds | Coverage mean | L20 mean | G20 mean | S20 mean | W1 mean |
| --- | --- | --- | --- | --- | --- | --- |
| Sparse MLP | 10/10 | 0.448 | 0.0217 | 0.0370 | 0.106 | 0.111 |
| Zero MLP | 10/10 | 0.513 | 0.0227 | 0.0327 | 0.0859 | -0.236 |
| Block-diag. LISTA | 10/10 | 0.447 | 0.0214 | 0.0397 | 0.0898 | -0.558 |
| Dense LISTA | 10/10 | 0.430 | 0.0221 | 0.0382 | 0.0886 | -0.141 |

**Kuramoto and corrected competitive-LV ok-seed rates**

| Case | Model | Ok seeds | Ok-seed rate mean | Mean 95% CI |
| --- | --- | --- | --- | --- |
| Competitive LV | Block-diag. LISTA | 0/10 | 0 | [0, 0] |
| Competitive LV | Dense LISTA | 0/10 | 0 | [0, 0] |
| Competitive LV | Sparse MLP | 0/10 | 0 | [0, 0] |
| Competitive LV | Zero MLP | 10/10 | 1.000 | [1.000, 1.000] |
| Kuramoto | Block-diag. LISTA | 0/10 | 0 | [0, 0] |
| Kuramoto | Dense LISTA | 0/10 | 0 | [0, 0] |
| Kuramoto | Sparse MLP | 0/10 | 0 | [0, 0] |
| Kuramoto | Zero MLP | 0/10 | 0 | [0, 0] |

