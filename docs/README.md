# Paper Documentation Map

The documentation tree has one organizing source of truth:
[`neurips_sparse_koopman_multibasin.tex`](neurips_sparse_koopman_multibasin.tex).
It owns the current claims, narrative, experiment order, display plan, and
paper-facing interpretation. The adjacent PDF is the current rendered review
copy; the TeX source takes precedence if they differ.

## Active ownership

- `neurips_sparse_koopman_multibasin.tex`: main paper and current project
  priorities that affect the paper.
- `neurips_sparse_koopman_multibasin.bib`: references cited by the active TeX.
- `appendix/experimental_details.tex`: canonical human-readable training and
  evaluation protocol.
- `appendix/benchmark_inventory.tex` and its included inventory fragments:
  benchmark composition, equations, and evaluation-only labels.
- `appendix/support_definitions.tex`: canonical support and support-family
  definitions.
- `appendix/statistical_testing.tex`: aggregation, uncertainty, and hypothesis
  testing protocol.
- `appendix/controlled_benchmark_results.tex` and `appendix/dysts_full.tex`:
  per-system and robustness displays for the two reported benchmarks.
- `appendix/highdimensional_confirmation.tex`: fixed-data Lorenz--96 and
  physics-based Allen--Cahn confirmation protocols, dense audits, and limits.
- `appendix/support_coordinate_interventions.tex`: the scoped one-checkpoint
  intervention result; it is not a general paper claim.
- `appendix/checklist_details.tex`: submission checklist, limitations, compute,
  assets, and disclosure details.
- Other `appendix/*.tex` files: one included result or background purpose each.
  They must not redefine a canonical protocol differently.
- `figures/neurips_paper_2026/`: only active displays and their compact source
  data. `_tables/` contains generated table inputs; `_data/`, when present,
  contains frozen row-level evidence needed to reproduce a paper result;
  `manifest.json` maps displays to generators and inputs.
- `figures/neurips_paper_2026/ground_truth_vector_fields/manifest.json`: exact
  ordered inventory and SHA-256 hashes for the composite plus 15 individual
  benchmark PDFs. The top-level manifest points to this nested inventory
  instead of encoding a human-readable filename placeholder.
- `archive/`: dated, non-current provenance, old drafts, and superseded plans.
  Nothing in the archive is an active instruction or status tracker.

The complete code-to-evidence-to-PDF procedure is maintained once in
[`../experiments/neurips_2026/README.md`](../experiments/neurips_2026/README.md).

Exact code/configuration names and cluster paths belong in machine-readable
provenance or reproduction commands, not in the paper-facing explanation to
senior coauthors.

## Result workflow

When a new experiment completes:

1. Report concrete results.
2. Explain them in the experiment context.
3. State the warranted interpretation and its limits.
4. State the implications for the paper.
5. Suggest the next step.
6. Update the main draft or the single relevant appendix if the result changes
   a claim, protocol, display, narrative, or priority. Otherwise create one
   dated archive note.

Do not recreate `EXPERIMENTS.md`, `PAPER_TRACK_STATUS.md`,
`PAPER_EXPERIMENT_EVIDENCE_MAP.md`, or any parallel status log.

## Maintenance rules

- Keep active authored text files outside `archive/` at or below 500 lines.
- Keep each active appendix fragment responsible for one purpose.
- Preserve compact source data, sample counts, seeds, splits, hyperparameters,
  aggregation rules, and artifact provenance needed to support paper claims.
- Do not treat benchmark basin labels or known basin counts as training-time
  inputs; they are allowed for evaluation only.
- Keep only the current paper PDF beside the source. LaTeX intermediates and old
  PDFs are generated or archived material.
- Before removing an artifact, confirm that it is neither included by active TeX
  nor the sole surviving source for a reported value.
- Run `uv run skae-paper check` before a paper handoff, then build the TeX source;
  run all Python commands inside a compute allocation.
