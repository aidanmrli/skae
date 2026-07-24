# Allen–Cahn early-fate probe V1: invalid target tie

Date: 2026-07-21 (America/Toronto)

V1 is permanently archived as `invalid_target_tie`. The outcome-blind card required zero exact top-two terminal pixel-occupancy ties. The authenticated CPU reducer stopped at that gate before any probe analysis because at least one such tie existed. No V1 result may be used as evidence for early support readout, representation quality, or forecasting mechanism.

## Frozen roots and jobs

- Prediction card SHA-256: `17987b43e40acbbc0c59d7a9d12ae2f1efa1343ff8904bbd9f58037a9be43bb2`
- Task manifest SHA-256: `10d47b500ebd79be4ab33983fa8a51e51c75aa7209a233deced925596b94837c`
- Source manifest SHA-256: `28dd4626283e2629f391d9618e48b672ff266ecfd82925f68680712a37e2c496`
- Synthetic A100 profile job `10164842`: completed, exit `0:0`, elapsed `00:01:25`.
- Field-only A100 extraction job `10164843`: completed, exit `0:0`, elapsed `00:00:43`.
- CPU validity/reduction job `10164844`: failed closed, exit `1:0`, elapsed `00:00:20`.
- Compute-node tests allocation `10164838`: 7/7 tests passed.

## Artifact provenance

- Profile decision SHA-256: `428c298742735b6594aca11f310bf01e8ad18e2347f687e025a9739899858bd3`
- Sealed field-only features SHA-256: `57b7b578597d7ad9d43151f3a92f31a50977de29c2e10f782467df279e1d21ea`
- Authenticated telemetry receipt SHA-256: `78be122734b4e3fe58958cbd78be15ca45b64dc97e2a762819394b7fb45c43be`
- Reducer stderr SHA-256: `ab642c8c85c3982c739a57df0976e66e0413038e3152410e38f171692f4d8dbd`
- Reducer stdout was empty (SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).
- Scratch root: `/network/scratch/l/lia/skae/allen_cahn_early_fate_probe_20260721_v1`.

GPU utilization was not hidden: the selected synthetic batch had 99.0% mean all-sample utilization; the scientific encoder scope had 98.8% mean all-sample utilization over 15 samples with no idle sample. The extraction allocation-wide mean was 41.49% over 36.02 seconds because checkpoint/data preload and 1.0 GB CPU serialization were outside the GPU kernel.

## Exact outcome-access attestation

The reducer derived labels at `experiments/neurips_2026/allen_cahn_early_fate_probe/summarize.py:104-107`, transient class counts at lines 112-116, and terminal diagnostics at lines 118-126 solely to evaluate validity. No label, class-count value, or diagnostic value was printed, persisted, or inspected. The reducer raised `RuntimeError("Exact terminal top-count tie gate failed")` at line 128.

No probe fit, prediction, accuracy or other score, permutation, contrast, decision packet, rows table, provenance packet, or evidence manifest was computed or created. The `summary/` directory does not exist. The only opened semantic fact is the binary validity fact that at least one exact terminal top-two occupancy tie exists; its count, trajectory identity, dataset, class, and margin were not inspected.

## Consequence

The benchmark generator itself defines a modal label with lowest-index `argmax` tie-breaking, but V1 predeclared that such an arbitrary tie was unacceptable for a fate claim. The gate is not relaxed post-outcome. Any successor must be transparently labeled, use entirely new dataset seeds frozen before generation, preserve the V1 scientific analysis, and stop permanently if its independently reviewed validity rule fails.
