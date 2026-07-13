# Repository audit for SKAE benchmark extension

Date: 2026-06-24 00:03 EDT.

## Revision and working tree

- Commit inspected: `7e93a7239991a17eb24d5968cc09f012a88a4435`.
- The working tree was already dirty before benchmark changes. Pre-existing modified paths included `skae/model.py`, `skae/config.py`, `skae/evaluation.py`, `tools/train.py`, many paper figures/tables, scripts, tests, `pyproject.toml`, and `uv.lock`. I did not revert or normalize those changes.

## Model implementation found

The repository model factory is `skae.model.make_model(cfg, observation_size)`. It maps:

- `GenericKM` and `SparseKM` to the same `GenericKM` class.
- `LISTAKM` to LISTA/HyperLISTA sparse-code Koopman models.
- `StructuredLISTAKM` to a structured latent extension.

For `GenericKM`, the mathematical form is:

\[
z_t = E_\theta(x_t), \qquad
\hat{x}_t = D_\phi(z_t), \qquad
z_{t+1}=z_t K,
\]

where `E_theta` and `D_phi` are MLPs configured by `MODEL.ENCODER` and `MODEL.DECODER`. `K` is learned as a dense matrix by default, or as diagonal/block-diagonal when `MODEL.K_STRUCTURE` requests it. The discrete rollout applies repeated right multiplication by `K`.

For `LISTAKM`, the decoder is a learned dictionary and the encoder is LISTA-style iterative soft-thresholding or HyperLISTA. The transition remains a learned linear latent map, with optional diagonal/block/soft-block/structured variants. Homogeneous coordinates can append a constant coordinate internally.

## Architecture details

- Encoder/decoder: MLP for the main repository SKAE (`MLPCoder`); LISTA/HyperLISTA sparse encoders are also implemented. A separate benchmark module has convolutional Koopman models for spatialized reaction-diffusion fields.
- Latent evolution: autonomous discrete-time linear latent update `z_{t+1}=z_t K` for the main SKAE. Controlled dynamics are not part of `GenericKM`; separate benchmark utilities implement additive controlled world models.
- Sparsity mechanism in `GenericKM`: `MODEL.SPARSITY_COEFF` penalizes the L1 norm of latent activations, not the L1 norm of `K`. `SparseKM` is therefore a configuration of `GenericKM`, not a distinct class.
- Sparsity mechanism in LISTA: elementwise soft-thresholding, optional group shrinkage/top-k groups, and optional structured latent penalties.
- Current loss terms in the standard trainer: latent alignment, reconstruction, prediction, latent activation sparsity, plus optional homogeneous, block, structured, and decoder-coherence terms depending on model type. Observation losses are dimension-normalized by `MODEL.OBS_LOSS_DIM_NORMALIZATION`.
- Multi-step training: supported through `TRAIN.SEQUENCE_LENGTH`; `tools/train.py` encodes every state in a sampled sequence, rolls out from the initial latent, and compares all rollout horizons.
- Convolutional encoders: not supported by the main `make_model` path; available in `skae/benchmarks/spatialized_conv_koopman.py` for the spatialized benchmark code.
- Checkpoints and metrics: `tools/train.py` writes `config.json`, `checkpoint.pt` for best validation final error, optional `last.pt`, `final_metrics.json`, `metrics_summary.json`, and optional history/evaluation artifacts under a timestamped run directory.

## Pre-change execution checks

All Python execution below was run through `salloc` and `srun` on `cn-f001.server.mila.quebec`.

Full test suite command:

```bash
salloc --mem=8G -c 4 --partition=long --time=00:45:00 \
  srun --cpu-bind=none bash -lc 'hostname; date; uv run pytest -q'
```

Result: 371 passed, 8 failed in 144.78 s. Failures were pre-existing in the inspected tree:

- `tests/test_claude_catalog_packet_tasks.py::test_claude_catalog_packet_custom_subset_uses_requested_systems_and_models`
- `tests/test_claude_catalog_packet_tasks.py::test_claude_catalog_packet_manifest_payload_tracks_selected_metadata`
- `tests/test_config.py::test_unknown_train_key_raises`
- `tests/test_model.py::TestUnifiedLossInterface::test_horizon_scaling_is_inverse_h`
- `tests/test_paper_benchmark_manifest.py::test_paper_benchmark_manifest_shape`
- `tests/test_transition_rich_basin_partition_manifest.py::test_transition_rich_basin_partition_manifest_shape`
- `tests/test_transition_rich_basin_partition_manifest.py::test_transition_rich_basin_partition_manifest_jsonable`
- `tests/test_transition_rich_basin_partition_tasks.py::test_transition_rich_basin_partition_default_matrix`

Existing smoke example:

```bash
salloc --mem=8G -c 4 --partition=long --time=00:15:00 \
  srun --cpu-bind=none bash -lc \
  'hostname; uv run python tools/train.py --config generic_sparse --env duffing \
   --sequence_length 1 --num_steps 3 --batch_size 8 --eval_every 2 \
   --eval_num_steps 2 --device cpu --skip_eval --skip_basin_eval \
   --log_dir runs/audit_smoke_generic_sparse_duffing'
```

Result: completed and wrote `runs/audit_smoke_generic_sparse_duffing/20260624-000305/checkpoint.pt`. A previous attempt failed before training because `tools/train.py` does not expose `--data_size`.

## Benchmark adaptation implications

The requested benchmark focuses on sparse Koopman operators. The repository's generic sparse preset instead applies latent-activation L1. The benchmark harness therefore reports the repository behavior explicitly and includes benchmark-local operator-density diagnostics for `K`. Any operator-L1 run is labeled as a benchmark loss adaptation, not as an unchanged repository training objective.
