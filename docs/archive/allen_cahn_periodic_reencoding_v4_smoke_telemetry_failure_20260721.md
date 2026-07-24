# Allen--Cahn periodic reencoding V4 smoke telemetry failure (2026-07-21)

## Concrete result

Outcome-free GPU smoke job `10169084` completed the exact two-model,
twenty-call H400 synthetic workload on one A100 80 GB GPU. Its all-sample
telemetry was strong: mean retained utilization was `99.57959183673469%`,
P10 was `100%`, peak memory fraction was `0.31209716796875`, and all 245
retained samples were nonzero. The smoke nevertheless failed because one
gap between consecutive one-second `nvidia-smi` records was
`2.9700000286102295` seconds, exceeding the preregistered two-second maximum.
Every other smoke telemetry check passed.

No passing smoke receipt was issued, so no V4 scientific job or dependent
summary was submitted. No trained checkpoint, physical prospective dataset,
forecast outcome, or scientific payload was accessed.

The frozen V4 card and source-manifest SHA-256 digests were respectively
`9d2374f986164941771af076576046358bce6bdbf19501a10a079d84081bc6f7`
and `015a305244ba0fc037b5af8d09fc3e15b633991f6f628d3ee7d27ac6a28dd593`.
The raw telemetry, telemetry audit, smoke runtime, and UUID-probe digests were
respectively
`f64a9a351c3ffa069fe4c12bd60c7a57010dcdec4788189455f676c0f1b8f2be`,
`d77ddec5c036f4267fb2a2b07a4a95611eb76e2c3b996ab6741b715062fa6e3b`,
`a0fa7d231df217f91aa8ff64a78c20b221d2ccf8668bb507c0210883ea73731c`,
and `ec33cb5b54e15527b06f31b8501f08768b0142db662f3aaa980638b14845fc4f`.

## Experimental context

The sole failed condition measured recorder punctuality, not GPU workload
utilization, memory safety, numerical finiteness, CUDA identity, or model
behavior. The raw stream contains a single jump from
`2026/07/21 18:38:16.501` to `2026/07/21 18:38:19.471`; utilization was 100%
on both adjacent records.

## Interpretation

V4 provides no scientific forecasting evidence. Its outcome-free smoke does
show that the exact workload saturates the allocated GPU while remaining well
inside the memory bound. Treating a transient monitoring-process delay as a
scientific failure would conflate recorder continuity with compute validity.

## Project implications

V5 changes only operational monitoring and unique execution provenance. It
uses natural 60-second samples, keeps every in-window sample without filtering,
boundary deletion, or padding, requires at least three samples and at least 90%
mean utilization in each disjoint GPU-compute window, and retains strict GPU
identity and memory checks. P10, cadence, maximum gap, and marker-edge distance
remain recorded descriptively but do not gate the result. Mila post-job NVIDIA
accounting may be inspected independently as descriptive corroboration.

## Next step

Freeze V5 under new roots, run its unchanged exact-shape outcome-free smoke,
and submit the unchanged scientific protocol only if the V5 smoke issues its
hash-bound receipt and passes the duplicate-safe guard.
