# Compatibility command shims

`tools/` no longer owns research implementations. Its small Python files keep
historical commands and imports working while forwarding to maintained package
modules.

Use the installed entry points for all new documentation and automation:

```bash
uv run skae-train ...
uv run skae-evaluate ...
uv run skae-plot-training ...
uv run skae-paper --help
```

## Canonical ownership

| Historical surface | Maintained implementation |
|---|---|
| `tools/train.py` | `skae.training.runner` / `skae-train` |
| `tools/evaluate_checkpoints.py` | `skae.cli.evaluate` / `skae-evaluate` |
| Paper task builders and collectors | `experiments.neurips_2026.workflows` / `skae-paper` |
| Paper tables and figures | `experiments.neurips_2026.evidence` / `skae-paper build ...` |
| Standalone controls | `experiments.neurips_2026.baselines` |
| Staged routed-operator protocol | `experiments.neurips_2026.local_operators` |
| Coordinate interventions | `experiments.neurips_2026.interventions` |
| Reusable support routing | `skae.support` |

The full command and artifact map lives in
[`experiments/neurips_2026/README.md`](../experiments/neurips_2026/README.md).

## Migration policy

- Do not add logic, protocol constants, or new commands here.
- Preserve a shim only while retained scripts, checkpoints, or downstream users
  need its filename/import path.
- Tests should import the maintained module unless they explicitly verify
  compatibility.
- Remove shims in a dedicated cleanup after downstream references have been
  audited; their presence must never create a second source of truth.
