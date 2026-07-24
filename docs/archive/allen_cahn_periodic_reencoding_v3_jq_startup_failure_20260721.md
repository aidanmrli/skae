# Allen--Cahn periodic reencoding V3 startup failure (2026-07-21)

## Concrete result

Scientific SLURM job `10169046` failed with exit code `1:0` after four
seconds on `cn-g020`. Its stderr contains only
`line 48: jq: command not found` followed by the fail-closed smoke-receipt
message. The V3 scientific output root was absent after the job, so no
dataset, checkpoint evaluation, selection, forecast payload, or outcome guard
was created. Dependent summary job `10169047` was canceled by the failed
dependency before allocation. No scientific outcomes were accessed, and the
sealed invalid V1 payload remained unopened.

The V3 stdout and stderr SHA-256 digests are respectively
`b7744f4490ffac1d95ca0efe1c7e0054349b8d73d35fc38f97d0c3f51d705f59`
and `cf86a5871584d0387e4b5bb69b994182aee0b23b82585dc522c2f47d62e84a36`.
The unchanged V3 card and source-manifest digests are
`a8636606f3248135759efe18f82b6dd62c95ad0d731f979611bf2093bdee5f48`
and `6e738758bcb1fc6d3041e0cd46696588a668085df53e722e7a5203af54ad8a68`.

## Experimental context

The prior outcome-free V3 smoke had passed on job `10168965`. The queue-side
receipt checks also passed and submitted the dependency chain. The failure
occurred in the GPU job's shell preflight before source execution, telemetry
startup, root creation, data generation, checkpoint loading, or scientific
evaluation. The cause was an undeclared node-level executable dependency,
not a model, dataset, CUDA, utilization, or numerical failure.

## Interpretation

V3 supplies no scientific evidence and cannot adjudicate periodic versus
direct forecasting or sparse versus dense forecasting. Because execution
stopped before the unique V3 root existed, its failure is outcome-blind and
does not consume the frozen prospective datasets.

## Project implications

V3 is retained as invalid historical execution provenance. V4 preserves its
checkpoints, seeds, cadence grid, horizons, estimands, inference, success
gates, and scientific implementation exactly, while replacing the shell JSON
checks with one strict duplicate-safe Python guard in the locked environment.

## Next step

Freeze V4 under unique smoke and scientific roots, require queue and GPU-job
preflights to invoke the same Python smoke guard, run a fresh outcome-free GPU
smoke, and launch the unchanged scientific computation only after that smoke
issues hash-bound receipts.
