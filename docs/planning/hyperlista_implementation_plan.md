# HyperLISTA Implementation Plan (Superseded)

This planning document has been superseded by the unified LISTA refactor.

Current direction:
- Use a single `LISTAKM` model class.
- Select encoder behavior through `cfg.MODEL.ENCODER.ENCODER_TYPE` (`"lista"` or `"hyperlista"`).
- Keep strict model-factory behavior for unsupported model names.
- Keep training/evaluation tooling aligned with unified encoder dispatch.

Active references:
- `docs/planning/lista-refactoring.md`
- `docs/planning/lista_generic_sparse_parity_depth_plan.md`
- `docs/planning/lista_depth_first_forecasting_plan.md`
