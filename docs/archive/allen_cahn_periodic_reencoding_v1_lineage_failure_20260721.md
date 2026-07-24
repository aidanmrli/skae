# Allen–Cahn periodic-reencoding v1 lineage failure (2026-07-21)

## Concrete result

Prospective GPU job `10167687` completed validation selection and all sealed
H200/H400 forecast computation, wrote the sealed selection and scientific
payloads, and then failed before outcome authorization. The failure occurred
while serializing metric-free runtime lineage: PyTorch returned the CUDA UUID
as an internal `_CUuuid` object, which strict JSON rejected.

No validation choice, forecast metric, scientific payload, or result summary
was opened. Dependent summary job `10167688` was cancelled by `afterok`.

Frozen and sealed provenance:

- prediction card: `97716bb6dc2c0e6a4f8389d362c06ad67045ac2a85574c794f1966a388ce9e17`
- source manifest: `a4729c04d1981c031d1531304668dc01b4165c1d5b1610d4d780d9730d123c6f`
- outcome-free smoke receipt: `cc366eeb760144c96fb9a1b55eef274c3ced03dc732584032722ed39f83f9a08`
- sealed scientific payload: `c0626c435bfcf8388585ed42e541c3748d9e88f0975ec6b033c17771a58c3c12`
- sealed selection decision: `2e2e0a79751af55c636e1792715636c8f7175e349085bbaff97bcacbc329e4b0`
- validation field manifest: `0c26866228a6b99665e73da364d16a9572d19cd7f4ecb56bf4d051f492e22a3c`
- test field manifest: `364251ff1bf3fb4d6464a17a78a5c697a7c2ef99297f41584c43750fb3b85c87`
- raw GPU telemetry: `5d69bca15d06c305ba17ff692dae63028f079800dfe7441c9969d0b1518bd226`
- evaluation-end marker: `e0ff9fc3fe76f8380f893f924173bd5337b65b1c45343c37bbd40a93eef5f1a8`
- compute-end marker: `24318bab313c60d50e46ddaeb9388ebcf6ff24d1263c17296983bda6112885f7`

The two evaluation-end markers contain the same epoch, and the sealed payload
hash in the marker matches the payload. The original source manifest still
verifies. The output root is
`/network/scratch/l/lia/skae/allen_cahn_periodic_reencoding_confirmation_20260721_v1`.

## Context

The job ran for 1:18:44 on one A100L. NVIDIA accounting reports 99% process
GPU utilization, 26,768 MiB maximum device memory, and 4,713,156 ms of process
time. Outcome-blind recomputation from the surviving telemetry gives 99.96%
and 99.95% mean retained utilization in the selection and evaluation windows,
respectively, with no retained sample at or below 80%.

Missing artifacts are `runtime_lineage.json`, `markers/job_end.json`, the copied
telemetry audit, and the outcome guard. The exact PyTorch allocator peak was
process-local and is no longer recoverable with its frozen semantics.

## Interpretation

This is a post-compute operational-provenance failure, not evidence of a
nonfinite forecast, low GPU utilization, or a scientific comparison. The v1
payload cannot pass the original card because that card required an
uninterrupted GPU-job → runtime-lineage → telemetry-guard → dependent-summary
chain. Reconstructing the chain later would be an undeclared protocol repair.

## Project implication

V1 remains unopened and is classified as invalid historical provenance. It
must not contribute a forecast value, cadence identity, figure, table, or
directional claim.

## Next step

Before any outcome access, run a clean v2 packet with every scientific choice
unchanged. Preserve all v1 files byte-for-byte. V2 may change only the
post-compute UUID serialization path, card/protocol identifiers, unique output
roots, provenance tests, and launch bindings. It requires a new source freeze,
outcome-free exact-shape smoke, scientific job, telemetry guard, and dependent
summary.
