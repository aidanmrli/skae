# Sparse Koopman Autoencoders for Multibasin Dynamics

This repository contains the PyTorch implementation and evidence package for
the paper draft **“Sparse Koopman Supports for Multibasin Forecasting and
Transductive Regime Alignment.”** The
current project goal is to turn the completed multibasin experiments into a
clear, reproducible NeurIPS submission.

The paper-facing question is whether sparse Koopman lifts learn latent support
patterns that align with basins of attraction and provide useful coordinates
for forecasting and local dynamics. The active claim set, experiment order,
and interpretation live in
[`docs/neurips_sparse_koopman_multibasin.tex`](docs/neurips_sparse_koopman_multibasin.tex).

## Modeling and evaluation conventions

These constraints apply to new code, experiments, and documentation:

- At training and deployment time, the number of basins and trajectory-to-basin
  assignments are unknown. Ground-truth basin counts or labels must not be
  required by the proposed method.
- Known basin labels and counts may be used only for benchmark evaluation or an
  explicitly labeled diagnostic ablation.
- The primary representation objective is **basin-support alignment**: each
  basin should map to a distinct sparse support in latent `z`. Basin-block
  alignment is not the default objective.
- The dense no-sparsity baseline uses `tanh` hidden activations. A dense ReLU
  model is a ReLU ablation, not the dense baseline.
- Spatialized multibasin PDE lifts must be overcomplete. Enforce
  `d_z >= 4 * d_x`, where `d_x = channels * grid_size**2`. With two channels,
  grid 16 requires `d_z >= 2048`, and grid 32 requires `d_z >= 8192`.

## Sources of truth

- [`AGENTS.md`](AGENTS.md): repository-wide operating, compute, testing, and
  documentation rules.
- [`docs/README.md`](docs/README.md): compact map of paper documentation and
  evidence ownership.
- [`docs/neurips_sparse_koopman_multibasin.tex`](docs/neurips_sparse_koopman_multibasin.tex):
  current claims, narrative, display order, and interpretation.
- [`docs/appendix/`](docs/appendix/): protocol and result details included by
  the paper.
- [`docs/figures/neurips_paper_2026/`](docs/figures/neurips_paper_2026/): active
  displays, compact source data, and display provenance.
- [`docs/archive/`](docs/archive/): dated, non-current provenance. Archived
  plans and trackers are not active instructions.

Do not create a second experiment-status or paper-status tracker. A result that
changes a paper claim, protocol, display, or priority belongs in the paper or
its single-purpose appendix. Other useful provenance belongs in a dated archive
note.

## Repository layout

```text
skae/       Core models, dynamical systems, configuration, and evaluation code
tools/      Maintained training, evaluation, collection, and plotting CLIs
scripts/    Maintained SLURM launchers and paper experiment orchestration
tests/      Pytest suites
docs/       Active paper, included appendices, evidence artifacts, and archive
runs/       Local training outputs; ignored by Git
results/    Local collected outputs; ignored by Git
```

Keep reusable implementation in `skae/`, user-facing entry points in `tools/`,
and cluster orchestration in `scripts/`. Avoid adding one-off Python files at
the repository root.

## Environment

The lock file and `uv` are the supported environment path:

```bash
uv sync
```

An editable installation without the lock file is available when necessary:

```bash
uv pip install -e .
```

Always invoke Python through `uv run`; do not call `python` or `python3`
directly.

## Cluster execution

Do not run Python programs, tests, training, evaluation, or validation sweeps on
the login node. First obtain a compute allocation:

```bash
# CPU
salloc --mem=8G -c 4 --partition=long

# GPU
salloc --gpus 1 --mem=8G -c 4 --partition=long
```

Inside the allocation, typical entry points are:

```bash
uv run python tools/build_table1_forecasting_support.py --check
uv run python tools/build_dysts_paper_evidence.py --check
uv run pytest tests/test_model.py -v
```

Submit scripts containing `#SBATCH` directives with `sbatch`, using the
repository-standard `long` partition unless a documented resource requirement
dictates otherwise. Before submitting a GPU job, verify that the called code
uses CUDA and that the workload can keep the requested GPU busy.

The canonical training launchers are paper-scoped:

```bash
sbatch scripts/queue_controlled_paper_training.sh
sbatch scripts/queue_dysts_dt30_basinblock_p256_seeds0to14.sh
```

See [`scripts/README.md`](scripts/README.md) before launching either campaign.

LaTeX manuscript builds are permitted on the login node. From `docs/`:

```bash
latexmk -pdf neurips_sparse_koopman_multibasin.tex
```

## Working on the paper evidence

Before changing an experiment or claim:

1. Read `AGENTS.md`, `docs/README.md`, and the relevant paper section.
2. Trace the cited table or figure to its source data and generation path.
3. Keep training-time method design free of benchmark basin labels and fixed
   basin counts.
4. Run focused tests on a compute node.
5. Report new results as concrete values, context, interpretation, project
   implications, and next steps.
6. Update the paper or relevant appendix when the result changes the active
   evidence; otherwise add a dated archive note.

Training artifacts under `runs/` and collected local outputs under `results/`
are not versioned. Preserve only compact, paper-relevant evidence and provenance
under the active paper figure directory.

## License

See [`LICENSE`](LICENSE).
