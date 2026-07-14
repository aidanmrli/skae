# Sparse Koopman Autoencoders for Multibasin Dynamics

This repository contains the reusable PyTorch implementation and the frozen
evidence workflow for **“Sparse Koopman Supports for Multibasin Forecasting and
Transductive Regime Alignment.”** The research question is compact: whether
sparse latent supports align with basins and provide useful coordinates for
forecasting and local dynamics.

The paper-facing source of truth is
[`docs/neurips_sparse_koopman_multibasin.tex`](docs/neurips_sparse_koopman_multibasin.tex).
The end-to-end code and evidence guide is
[`experiments/neurips_2026/README.md`](experiments/neurips_2026/README.md).

## Where to look

| If you want to… | Start here | Responsibility |
|---|---|---|
| Read the claims and results | `docs/neurips_sparse_koopman_multibasin.tex` | Current narrative, claims, display order, and interpretation |
| Reproduce every paper artifact | `experiments/neurips_2026/README.md` | One ordered workflow from environment setup to PDF |
| Inspect the frozen protocol | `experiments/neurips_2026/protocol.py` | System rosters, seeds, budgets, model rows, display labels |
| Trace a table or figure | `docs/figures/neurips_paper_2026/manifest.json` | Compact inputs, generators, outputs, and provenance |
| Change reusable modeling code | `skae/README.md` | Core model, data, dynamics, support, training, and evaluation map |
| Launch paper jobs | `scripts/README.md` | SLURM launchers grouped by experiment family |
| Find a command | `uv run skae-paper --help` | Stable paper command surface |
| Understand old filenames | `tools/README.md` | Compatibility wrappers and migration policy |
| Find historical context | `docs/archive/` | Dated, non-current provenance only |

## Repository architecture

```text
skae/                       reusable research library
  dynamics/analytic/        maintained analytic multibasin systems
  support/                  support families and routed local operators
  training/                 general training runner and metric plotting
  cli/                      general checkpoint evaluation
experiments/neurips_2026/   paper-only protocol, workflows, and evidence code
scripts/
  common/                   shared SLURM workers and cluster helpers
  neurips_2026/             paper launchers, separated by experiment family
docs/                       paper source, appendices, compact evidence, archive
tests/                      reusable-library and frozen-protocol regression tests
tools/                      deprecated filename-compatible Python shims
runs/, results/             unversioned execution outputs
```

The segregation rule is simple: code that would remain useful for another
paper belongs in `skae/`; choices tied to this paper belong in
`experiments/neurips_2026/`; resource scheduling belongs in `scripts/`; claims
and durable evidence belong in `docs/`. Do not add new implementations to
`tools/`.

Each scientific choice has one owner. Protocol/contract modules define rosters,
budgets, model rows, and baseline method sets; workflow modules consume that
contract; builders turn versioned row evidence into displays; the paper cites
those displays. Historical names survive only at compatibility and provenance
boundaries, never as a second maintained implementation.

## Quick start

Install the locked environment:

```bash
uv sync
```

Do not run Python programs on the login node. Obtain a compute allocation first:

```bash
# CPU
salloc --mem=8G -c 4 --partition=long

# GPU
salloc --gpus 1 --mem=8G -c 4 --partition=long
```

Inside the allocation, validate the frozen protocol and all compact evidence:

```bash
uv run skae-paper protocol validate
uv run skae-paper check
uv run pytest
```

Build the review PDF from the login node or a compute node:

```bash
latexmk -cd -pdf docs/neurips_sparse_koopman_multibasin.tex
```

The main campaign launchers are:

```bash
sbatch scripts/neurips_2026/controlled/queue_training.sh
sbatch scripts/neurips_2026/dysts/queue_training.sh
```

Launchers default to the `long` partition. Output storage resolves from
`$SKAE_SCRATCH_ROOT`, then `$SCRATCH/skae`, then the current Mila user's scratch
directory, with a repository-local fallback. No contributor-specific path is
encoded in the maintained workflow.

## Scientific invariants

- Training and deployment do not know the number of basins or trajectory basin
  assignments. Known labels and counts are evaluation-only benchmark metadata.
- The primary representation goal is basin-support alignment, not basin-block
  alignment.
- The dense no-sparsity baseline uses `tanh`; dense ReLU is an ablation.
- Spatialized multibasin PDE lifts satisfy `d_z >= 4 * d_x`.
- The controlled and Dysts paper rosters, shared 15-seed contract, training
  budgets, and model-row mapping are frozen in
  `experiments/neurips_2026/protocol.py`.

See [`AGENTS.md`](AGENTS.md) for the complete development, compute, and
documentation rules.

## Documentation ownership

Do not create another paper-status tracker. A result that changes a claim,
protocol, display, or priority goes into the paper or its single-purpose
appendix. Useful non-current provenance goes into a dated note under
`docs/archive/`. Active generated evidence stays under
`docs/figures/neurips_paper_2026/` with compact row data and provenance.

## License

See [`LICENSE`](LICENSE).
