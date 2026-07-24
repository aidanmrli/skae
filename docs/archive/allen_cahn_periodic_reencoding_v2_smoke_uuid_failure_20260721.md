# Allen–Cahn periodic-reencoding v2 smoke UUID failure — 2026-07-21

## Concrete result

Outcome-free GPU smoke job `10168433` completed the exact-shape synthetic
forecast workload at 98% GPU utilization, then failed before its smoke audit
could issue a passing receipt. The real CUDA UUID probe correctly observed the
PyTorch runtime type `_CUuuid`, but v2 incorrectly required `str(_CUuuid)` to
start with `GPU-`. On this runtime, that string is the bare canonical 36-character
UUID; `nvidia-smi` reports the same identity with the `GPU-` prefix.

No trained checkpoint, prospective PDE trajectory, validation choice, sealed
scientific metric, or v1 scientific payload was opened. No v2 scientific or
summary job was submitted. The v2 card SHA-256 is
`e5af3746ae9a537f4c7860221228c9f39fc92acd7ce6442b85cd48d1f35cab4f`;
its source-manifest SHA-256 is
`0fa8b1994c1634164224441c3baf19f48314fb9db9ae89472c2802d491d82265`.

## Context and interpretation

This is an operational provenance failure, not a scientific result. It neither
supports nor weakens periodic reencoding because the outcome quarantine held
and the scientific computation was never launched. The high smoke utilization
does confirm that the unchanged exact-shape workload remains appropriate for
auditing the requested A100L utilization contract.

## Project implication

V2 cannot authorize a scientific launch because it has no passing, hash-bound
smoke receipt. Its files and failed smoke root remain historical provenance and
must not be reused as successful evidence.

## Next step

Use a new execution-only packet with unique roots. Derive the PyTorch UUID from
the exact 16 values in `_CUuuid.bytes`, validate the bare string representation,
canonicalize it with `uuid.UUID`, and require equality with an independent,
single-device `nvidia-smi` query. Run that helper in both the outcome-free probe
and the post-compute metric-free runtime-lineage writer before permitting the
unchanged scientific protocol.
