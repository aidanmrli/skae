# Changelog

## Unreleased

- Optional label-free support losses, added after the paper: `SupportLossConfig` and `WindowSimilarity`
  ([`skae/losses.py`](skae/losses.py)), the `support` and `pair_weights` arguments of `training_loss` and
  `return_activity` in the encoders. They are off by default, so the paper's models train exactly as before.
- New options of `fit`: `support`, `koopman_lr` (a separate learning rate for `K`) and
  `lr_schedule="cosine_tail"`.

## 1.0.0

Initial public release, accompanying the NeurIPS 2026 paper
[*Sparse Koopman Autoencoders Identify Local Dynamical Regimes in Multibasin Systems*](https://arxiv.org/abs/2608.29057).
