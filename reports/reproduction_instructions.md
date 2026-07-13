# Reproduction instructions

Run Python workloads on a compute node. Example:

```bash
salloc --mem=8G -c 4 --partition=long --time=01:00:00 \
  srun --cpu-bind=none bash -lc 'uv run python -m experiments.run_suite --config configs/skae_benchmark_suite.yaml'
```

Main tables are regenerated from `results/raw_metrics.parquet` by rerunning the same command.
