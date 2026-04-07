# `generic_sparse + block_diagonal K` Fairness Plan

Date: March 11, 2026

## Objective

Add a matched `generic_sparse + block_diagonal K` control to every paper-facing experiment that currently uses `lista_blockdiag`.

This is required before freezing any paper claim about block-diagonal gains.

## Why this is required

- Current `lista_blockdiag` comparisons change both the encoder family and the Koopman structure relative to `generic_sparse`.
- In the current code, `GenericKM` still instantiates a dense `K`, so there is no MLP baseline with the same block-diagonal dynamics constraint.
- Existing `lista_blockdiag` results are therefore valid as end-to-end model comparisons, but they do not isolate whether the gain comes from the LISTA encoder, the block-diagonal Koopman structure, or their interaction.

## Fairness rule

For every paper-facing `lista_blockdiag*` root, add a mirrored `generic_sparse_blockdiag*` root that matches all non-encoder settings:

- same systems
- same seeds
- same `dt`
- same `num_steps`
- same optimizer knobs
- same loss coefficients
- same `k_structure=block_diagonal`
- same `k_block_size`
- same eval profile
- same environment-specific overrides

Only the encoder family should differ.

Naming rule:
- replace the `lista_blockdiag` prefix with `generic_sparse_blockdiag`

Examples:
- `lista_blockdiag` -> `generic_sparse_blockdiag`
- `lista_blockdiag_ns200k_denseopt_sc6em3` -> `generic_sparse_blockdiag_ns200k_denseopt_sc6em3`
- `lista_blockdiag_targeted` -> `generic_sparse_blockdiag_targeted`

## Required code changes

### 1. Add structured-`K` support to `GenericKM`

Files:
- [skae/model.py](/home/mila/l/lia/skae/skae/model.py)
- [tests/test_model.py](/home/mila/l/lia/skae/tests/test_model.py)

Work:
- Teach `GenericKM` to honor `cfg.MODEL.K_STRUCTURE in {dense, diagonal, block_diagonal}`.
- Reuse the same `K_BLOCK_SIZE` semantics already used by `LISTAKM`.
- Implement `kmatrix()` and `step_latent()` for diagonal and block-diagonal `K`.
- Keep MLP encoder/decoder behavior unchanged.
- Keep latent normalization behavior unchanged.
- Do not add block-usage losses to `GenericKM` in the first pass; the fairness requirement is structural `K`, not auxiliary block losses.

Acceptance:
- unit tests cover dense, diagonal, and block-diagonal `GenericKM`
- `GenericKM.step_latent()` matches `GenericKM.kmatrix()` semantics
- existing dense `generic_sparse` behavior remains unchanged

### 2. Add a canonical paper-benchmark variant

Files:
- [skae/benchmarks/paper_benchmark_manifest.py](/home/mila/l/lia/skae/skae/benchmarks/paper_benchmark_manifest.py)
- [tests/test_paper_benchmark_manifest.py](/home/mila/l/lia/skae/tests/test_paper_benchmark_manifest.py)

Work:
- Add `generic_sparse_blockdiag` to the manifest.
- Use `config_name="generic_sparse"`.
- Match the current canonical `lista_blockdiag` non-encoder recipe: coefficients, `k_structure`, and `k_block_size`.

Acceptance:
- manifest exposes the new variant
- task builders can request it by name

### 3. Update task builders that currently emit `lista_blockdiag`

Files:
- [tools/build_paper_benchmark_tasks.py](/home/mila/l/lia/skae/tools/build_paper_benchmark_tasks.py)
- [tools/build_paper_followup_recipe_tasks.py](/home/mila/l/lia/skae/tools/build_paper_followup_recipe_tasks.py)
- [tools/build_kuramoto_dimension_sweep_tasks.py](/home/mila/l/lia/skae/tools/build_kuramoto_dimension_sweep_tasks.py)
- [tools/build_hopfield_basin_sweep_tasks.py](/home/mila/l/lia/skae/tools/build_hopfield_basin_sweep_tasks.py)
- [tools/build_kuramoto_mode_support_audit_tasks.py](/home/mila/l/lia/skae/tools/build_kuramoto_mode_support_audit_tasks.py)

Tests:
- [tests/test_paper_benchmark_tasks.py](/home/mila/l/lia/skae/tests/test_paper_benchmark_tasks.py)
- [tests/test_kuramoto_dimension_sweep_tasks.py](/home/mila/l/lia/skae/tests/test_kuramoto_dimension_sweep_tasks.py)
- [tests/test_kuramoto_mode_support_audit_tasks.py](/home/mila/l/lia/skae/tests/test_kuramoto_mode_support_audit_tasks.py)

Work:
- Add mirrored `generic_sparse_blockdiag*` variants anywhere `lista_blockdiag*` is currently enumerated.
- For follow-up recipes, use `base_variant=generic_sparse_blockdiag` so recipe labels can mirror the existing blockdiag LISTA roots exactly.
- For support-alignment and audit builders, add the new root label and family mapping.

Acceptance:
- local task tables contain both `lista_blockdiag*` and `generic_sparse_blockdiag*`
- root labels are stable and deterministic
- task counts update as expected

### 4. Update queue scripts and root-spec generation

Files:
- [scripts/queue_paper_benchmark_chain.sh](/home/mila/l/lia/skae/scripts/queue_paper_benchmark_chain.sh)
- [scripts/queue_paper_followup_recipes.sh](/home/mila/l/lia/skae/scripts/queue_paper_followup_recipes.sh)
- [scripts/queue_kuramoto_dt00625_200k_compare.sh](/home/mila/l/lia/skae/scripts/queue_kuramoto_dt00625_200k_compare.sh)
- [scripts/queue_intrinsic_hd_dt_rescue.sh](/home/mila/l/lia/skae/scripts/queue_intrinsic_hd_dt_rescue.sh)
- [scripts/sweep_intrinsic_hd_dt_rescue.sh](/home/mila/l/lia/skae/scripts/sweep_intrinsic_hd_dt_rescue.sh)
- [scripts/queue_kuramoto_dimension_sweep.sh](/home/mila/l/lia/skae/scripts/queue_kuramoto_dimension_sweep.sh)
- [scripts/queue_hopfield_basin_sweep.sh](/home/mila/l/lia/skae/scripts/queue_hopfield_basin_sweep.sh)
- [scripts/queue_competitive_lv_retrain.sh](/home/mila/l/lia/skae/scripts/queue_competitive_lv_retrain.sh)
- [scripts/queue_kuramoto_mode_support_audit.sh](/home/mila/l/lia/skae/scripts/queue_kuramoto_mode_support_audit.sh)

Work:
- Add the new mirrored MLP-blockdiag variants to default variant lists and recipe CSVs.
- Ensure root-spec files include the new roots.
- Ensure compare jobs generate the right pairwise tables.

Required pairwise comparisons for any affected paper-facing experiment:
- `generic_sparse_blockdiag*` vs `generic_sparse*`
- `lista_blockdiag*` vs `generic_sparse_blockdiag*`
- keep `lista_blockdiag*` vs `generic_sparse*` only as the end-to-end comparison

### 5. Update collectors and summary scripts that hardcode family lists

Likely files:
- [tools/summarize_kuramoto_dimension_sweep.py](/home/mila/l/lia/skae/tools/summarize_kuramoto_dimension_sweep.py)
- [tools/evaluate_paper_benchmark_support_alignment.py](/home/mila/l/lia/skae/tools/evaluate_paper_benchmark_support_alignment.py)
- any compare or collection helper that assumes only `{generic_sparse, lista_dense, lista_blockdiag}`

Work:
- add the new blockdiag MLP root where tables or comparisons would otherwise omit it
- keep old tables readable by separating the dense MLP anchor, the blockdiag MLP control, and the blockdiag LISTA candidate

## Paper-facing experiments that must gain the new control

### Priority 1: current headline block-diagonal claims

- focused Kuramoto `dt=0.00625`, `200k`
- Kuramoto `N=32` confirmation
- Kuramoto dimension sweep
- Kuramoto robustness (uniform spread)
- intrinsic-HD `dt` rescue pilot
- Hopfield `dt=0.00625`, `200k`

### Priority 2: benchmark-wide fairness for any remaining blockdiag narrative

- canonical `v4` benchmark
- `200k` follow-up blockdiag recipe reruns
- `competitive_lv` multi-basin retrain groups that currently include `lista_blockdiag`

### Priority 3: downstream evaluation-only consumers after checkpoints exist

- Kuramoto mode-support audit
- label-free clustering on blockdiag roots
- `competitive_lv` support-alignment reruns
- Hopfield basin-count sweep

## Recommended execution order

1. Implement structured `K` in `GenericKM` and land tests.
2. Add `generic_sparse_blockdiag` to the paper benchmark manifest.
3. Update task builders and queue scripts.
4. Run local QA on builders and focused tests.
5. Re-run the highest-value headline studies first:
   - focused Kuramoto `dt=0.00625`, `200k`
   - Kuramoto `N=32`
   - Kuramoto dimension sweep
   - Hopfield `dt=0.00625`, `200k`
6. Only after that, decide whether the canonical `v4` and `200k` blockdiag benchmark-wide reruns are still needed for the final paper framing.
7. Refresh evaluation-only analyses on the new checkpoints.

## Local QA before queueing

Run at minimum:

```bash
uv run pytest tests/test_model.py tests/test_paper_benchmark_manifest.py tests/test_paper_benchmark_tasks.py tests/test_kuramoto_dimension_sweep_tasks.py tests/test_kuramoto_mode_support_audit_tasks.py -q
```

Also smoke-check builder outputs:

```bash
uv run python tools/build_paper_benchmark_tasks.py --phase full --output_tsv /tmp/paper.tsv --model_variants_csv generic_sparse,generic_sparse_blockdiag,lista_blockdiag
uv run python tools/build_kuramoto_dimension_sweep_tasks.py --output_tsv /tmp/kuramoto_dim.tsv --model_variants_csv generic_sparse,generic_sparse_blockdiag,lista_blockdiag
uv run python tools/build_hopfield_basin_sweep_tasks.py --output_tsv /tmp/hopfield.tsv --model_variants_csv generic_sparse,generic_sparse_blockdiag_targeted,lista_blockdiag_targeted
```

Verify:
- task counts are correct
- root labels follow the naming rule
- `k_structure=block_diagonal` appears on the new MLP control rows
- no LISTA-only flags are required on the new MLP control rows

## Writing rule until this is done

Do not write or imply:
- “block-diagonal `K` causes the Kuramoto gain”
- “structured Koopman dynamics outperform the MLP baseline on Kuramoto”

Allowed wording until completion:
- `lista_blockdiag` is currently the strongest tested end-to-end block-diagonal variant on the targeted Kuramoto setting
- the current evidence is comparative, not a clean structure-isolation ablation
