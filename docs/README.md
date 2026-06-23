# Documentation Map

This directory is organized around the active NeurIPS draft:

- `neurips_sparse_koopman_multibasin.tex`: primary paper source of truth for claims, narrative, experiment ordering, and paper-facing interpretation.
- `neurips_sparse_koopman_multibasin.bib`: bibliography database for the draft.
- `appendix/*.tex`: single-purpose appendix fragments included by the draft. Keep each fragment scoped to one appendix topic.
- `figures/neurips_paper_2026/`: active paper figures, tables, and generated display artifacts referenced by the draft.
- `archive/`: historical notes, old plans, handoff packets, old drafts, literature notes, and build artifacts.

Rules:

- Keep active authored text files outside `archive/` at or below 500 lines.
- Do not create parallel status trackers for the same responsibility. If a claim, result, or priority is paper-facing, put it in the draft or the relevant appendix fragment.
- If material is useful but not part of the current paper source, move it under `archive/` with a dated folder or short README.
- Build products and historical PDFs should not live beside the active draft unless they are the current paper PDF.
