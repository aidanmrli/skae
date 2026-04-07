# Dysts System Performance (2026-03-31)

This note extracts the 15 Dysts systems from the verified benchmark table at [results/paper_seed_statistics_20260331/forecasting_system_seed_stats.csv](../results/paper_seed_statistics_20260331/forecasting_system_seed_stats.csv), restricted to `setting_id = benchmark_main_best_periodic`.

Exact per-system robust statistics are in [results/paper_seed_statistics_20260331/dysts_benchmark_main_best_periodic_seed_stats.csv](../results/paper_seed_statistics_20260331/dysts_benchmark_main_best_periodic_seed_stats.csv).

All retained Dysts rows in this packet have consistent coverage: seed_count=10/n=10.

## IQM winner counts by horizon

| Horizon | Dense LISTA wins | Sparse MLP wins | Zero MLP wins |
| --- | --- | --- | --- |
| 100 | 7 | 3 | 5 |
| 500 | 11 | 2 | 2 |
| 1000 | 12 | 1 | 2 |
| 1500 | 14 | 1 | 0 |
| 2000 | 14 | 1 | 0 |
| 2500 | 13 | 2 | 0 |
| 3000 | 13 | 2 | 0 |

## Mean IQM across the 15 Dysts systems

Lower is better.

| Horizon | Dense LISTA | Sparse MLP | Zero MLP |
| --- | --- | --- | --- |
| 100 | 0.0011 | 0.0012 | 0.0012 |
| 500 | 0.0233 | 0.0239 | 0.0244 |
| 1000 | 0.0630 | 0.0650 | 0.0677 |
| 1500 | 0.0924 | 0.104 | 0.112 |
| 2000 | 0.136 | 0.163 | 0.192 |
| 2500 | 0.198 | 0.246 | 0.289 |
| 3000 | 0.277 | 0.335 | 0.394 |

## Per-system IQM at H1000

| System | Dense LISTA IQM | Sparse MLP IQM | Zero MLP IQM | Best IQM |
| --- | --- | --- | --- | --- |
| Chua | 8.786e-04 | 0.0018 | 0.0017 | Dense LISTA |
| Dadras | 0.0096 | 0.0111 | 0.0102 | Dense LISTA |
| DequanLi | 0.0050 | 0.0048 | 0.0110 | Sparse MLP |
| Duffing | 0.852 | 0.854 | 0.849 | Zero MLP |
| Hadley | 0.0032 | 0.0040 | 0.0035 | Dense LISTA |
| LorenzCoupled | 0.0257 | 0.0272 | 0.0353 | Dense LISTA |
| LuChenCheng | 8.778e-04 | 0.0018 | 0.0024 | Dense LISTA |
| MultiChua | 0.0022 | 0.0034 | 0.0048 | Dense LISTA |
| QiChen | 0.0029 | 0.0042 | 0.0052 | Dense LISTA |
| RikitakeDynamo | 0.0028 | 0.0038 | 0.0075 | Dense LISTA |
| Sakarya | 0.0091 | 0.0128 | 0.0117 | Dense LISTA |
| SanUmSrisuchinwong | 0.0038 | 0.0056 | 0.0040 | Dense LISTA |
| ShimizuMorioka | 0.0016 | 0.0044 | 0.0071 | Dense LISTA |
| SprottTorus | 0.0089 | 0.0085 | 0.0069 | Zero MLP |
| WangSun | 0.0158 | 0.0281 | 0.0557 | Dense LISTA |

## Per-system IQM at H3000

| System | Dense LISTA IQM | Sparse MLP IQM | Zero MLP IQM | Best IQM |
| --- | --- | --- | --- | --- |
| Chua | 0.0071 | 0.0173 | 0.0193 | Dense LISTA |
| Dadras | 0.386 | 0.481 | 0.559 | Dense LISTA |
| DequanLi | 0.0472 | 0.0404 | 0.139 | Sparse MLP |
| Duffing | 1.872 | 2.123 | 1.900 | Dense LISTA |
| Hadley | 0.0230 | 0.0321 | 0.0330 | Dense LISTA |
| LorenzCoupled | 0.792 | 0.981 | 1.201 | Dense LISTA |
| LuChenCheng | 0.0062 | 0.0176 | 0.0214 | Dense LISTA |
| MultiChua | 0.0331 | 0.0386 | 0.0556 | Dense LISTA |
| QiChen | 0.0492 | 0.0770 | 0.175 | Dense LISTA |
| RikitakeDynamo | 0.0402 | 0.0826 | 0.109 | Dense LISTA |
| Sakarya | 0.170 | 0.227 | 0.199 | Dense LISTA |
| SanUmSrisuchinwong | 0.0354 | 0.0774 | 0.0399 | Dense LISTA |
| ShimizuMorioka | 0.0214 | 0.0706 | 0.0661 | Dense LISTA |
| SprottTorus | 0.110 | 0.0914 | 0.148 | Sparse MLP |
| WangSun | 0.557 | 0.672 | 1.239 | Dense LISTA |

## IQM winner by system across horizons

| System | H100 | H500 | H1000 | H1500 | H2000 | H2500 | H3000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chua | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA |
| Dadras | Sparse MLP | Sparse MLP | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA |
| DequanLi | Sparse MLP | Sparse MLP | Sparse MLP | Dense LISTA | Dense LISTA | Sparse MLP | Sparse MLP |
| Duffing | Dense LISTA | Dense LISTA | Zero MLP | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA |
| Hadley | Zero MLP | Zero MLP | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA |
| LorenzCoupled | Zero MLP | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA |
| LuChenCheng | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA |
| MultiChua | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA |
| QiChen | Sparse MLP | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA |
| RikitakeDynamo | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA |
| Sakarya | Zero MLP | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA |
| SanUmSrisuchinwong | Zero MLP | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA |
| ShimizuMorioka | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA |
| SprottTorus | Zero MLP | Zero MLP | Zero MLP | Sparse MLP | Sparse MLP | Sparse MLP | Sparse MLP |
| WangSun | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA | Dense LISTA |
