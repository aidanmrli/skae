# Allen--Cahn early-fate probe V2 telemetry failure (2026-07-21)

## Concrete result

The independently red-team-approved V2 chain used prediction-card SHA-256
`db6f57e45568284e8903520df0dc2838ac2410d8adf8839d21ce366dc446a20c`,
source-manifest SHA-256
`dfe1c66166488befe119757b0de9e3bcf7e4be564fb81dc74038aa1ceef9618d`,
and task-manifest SHA-256
`30c508c2201d8e37ad1d8d3409d3317685bb4db686c20eb74773dda772379a89`.
GPU job 10165245 completed successfully. Dependent CPU telemetry job 10165246
failed with exit code 1 and the sole error `Telemetry authentication must be
CPU-only.` Dependent label-aware summary job 10165247 was consequently
canceled. No telemetry receipt or summary directory was created.

The GPU stage created three field-only datasets and one frozen feature payload.
The dataset SHA-256 values, in frozen seed order, are
`48f6cc8890093286729d8e1b9d8d947c6dbdef2f9e2d7d70376a4d96fc4aa3a1`,
`6838fb14251af8194ec3dd8f6b4860b14b4e9b7fe698ca29f9a3a08da99d65e7`,
and `fd478c7c79b33a4427e30887abeb24eb540edbfc53f860eccd4006e37ffdcbc3`.
The dataset manifest SHA-256 is
`dab14d685b25472414f0d4cc423211ea8cb876cf14069ccdcd662f5e05ae1b22`;
the feature-payload SHA-256 is
`941ed0e53d344f604955eb6b7aba8a0cacba0486813630836feff54b50caae25`;
the marker SHA-256 is
`2e08caa15df200e663ffb96244dcbf1aa67e689b9d6401f85d251724b9aaf7c9`;
and the unverified raw-telemetry SHA-256 is
`2b9a240a8bac5dcf42adc776b91fd4d58d0d7c61ceb46a9a2fe982564c5597d3`.
Markers and the dataset manifest attest zero semantic outcomes accessed or
derived. The telemetry-job stderr SHA-256 is
`b16164bb6e63305bdaf3f104f2f5921be250a3a9cb786135368e5e2e171380d6`.

## Context

V2 was the one permitted successor to V1's exact-terminal-tie target failure.
It restored exact equality to the authenticated V1 Allen--Cahn generator,
including beta 8.0, used entirely new frozen RNG streams, and predeclared exact
top-two terminal-occupancy tie exclusion. The GPU firewall allowed only fields,
split indices, neutral metadata, and representation encoding. Ties, eligibility,
labels, probes, scores, and contrasts were reserved for the receipt-authenticated
CPU summary stage.

The wrapper rejected the CPU telemetry task before its Python authenticator ran
because `CUDA_VISIBLE_DEVICES` was nonempty and not `NoDevFiles`, despite the
SLURM task requesting no GPU. The exact value was not logged. The likely cause
is inherited environment state through the launcher's `--export=ALL`, but this
is an inference rather than an authenticated observation.

## Interpretation

This is an execution/authentication failure, not a null or positive scientific
result. No target was constructed and no model was fit or scored. Therefore V2
provides no evidence about whether sparse x0 support predicts decisive T=20
modal-well occupancy or outperforms dense, raw-state, physics-summary, or
initial-modal controls.

As a non-scientific utilization diagnostic only, the unverified trace shows
that the synchronized 15.26-second encoder scope had
16 one-second samples, 14 active samples, 99.0% active-sample utilization,
86.625% all-sample utilization, and two zero samples. Those values would clear
the frozen encoder thresholds, but they are descriptive because the
authentication job failed. Across the complete 74-sample monitored GPU runtime,
mean utilization was only 21.203%, active-sample mean was 78.45%, and 72.973% of
samples were zero. Generation alone averaged 6.5% across 26 samples. Thus the
encoder kernel was highly utilized while CPU initial-condition work, loading,
and serialization made allocation-level utilization low; only the encoder
scope was a predeclared validity gate.

## Project implications

The V2 early-fate probe cannot support the rebuttal or paper. Its frozen failure
policy treats any telemetry or authentication failure as terminal and forbids a
V3 endpoint, exclusion, seed, or gate revision. The field and latent artifacts
must remain outcome-blind provenance rather than be reduced post hoc.

## Next step

Do not retry V2, alter its roots, derive its terminal labels, or launch a
successor. Base the rebuttal on independently authenticated Allen--Cahn
forecasting, dense-baseline, support-closure, and local-law evidence, with their
existing claim boundaries. Any future early-outcome study would need to be a
separately motivated experiment outside this frozen V1/V2 sequence and should
not be represented as rescuing it.
