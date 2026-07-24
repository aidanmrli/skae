# Global-K residual forecast V1 smoke-gate failure (2026-07-21)

## Concrete result

The first prospective support-routed residual forecast chain used prediction-card
SHA-256
`d60d833d84961da0c5931e6ee6cf3dbf763c12ccf3607764c5700f7cba2808dc`,
task-manifest SHA-256
`86a3dce2ce8fd6ca569aebcccb6812ac6c3ee206ec21ba8e2ccf2642305fb024`,
and source-manifest SHA-256
`03e19d4efa6e98cc1b429403223de6159a8a787841e3218bed690af27cc750b8`.
Queue, data-preparation, and outcome-blind smoke jobs 10165645, 10165655, and
10165656 completed. The dependent smoke-gate job 10165657 failed; science job
10165658 and its dependent telemetry and summary jobs 10165659--10165660 were
canceled. No scientific task ran and no forecast outcome was inspected.

The smoke shard, telemetry trace, and synchronized compute-window record have
SHA-256 values
`b8688a9a83527b8fe981757a121a06cc166df4c80f8b87aef6511a3fd51574d6`,
`6cf85b75c8945abfc9a432d09015376be12f3404bb4b160c5c82ab63b231e4bb`,
and
`c88f7ef042ab5fdb0d29478893f4985bab37285bb3cf175faf42583effdf25c8`.
The smoke completed all 41 frozen methods with finite, schema-valid metadata;
it persisted no MSE, curve, basin label, routing-alignment value, null score,
comparison, or decision.

The synchronized forecast interval lasted 4.619 seconds and retained five
one-second utilization samples: 33%, 98%, 96%, 99%, and 85%. Their mean was
82.2%, p10 was 53.8%, and the peak memory observation was 4,433 MiB. The whole
15-sample allocation trace averaged 27.93% utilization. These values failed the
predeclared minimum duration, sample-count, mean, p10, rolling-window, and
allocation-wide utilization checks. Independently, telemetry assessment raised
an exception while serializing a NumPy Boolean; an undefined rolling statistic
would also have violated strict JSON had execution reached it.

## Context

The experiment asks whether label-free sparse support families make one
unchanged learned global Koopman matrix useful inside the autonomous
per-step-reencoded residual map

`x_next = x + D(((E(x) P_f) K) P_f) - D(E(x) P_f)`.

The V1 smoke used 8,192 newly generated trajectories solely to validate the
method roster, finiteness, routing/null completion, and GPU observability before
ten model-seed science tasks. Its firewall deliberately prevented scientific
forecast outcomes or evaluation labels from being written.

## Interpretation

V1 is operationally invalid, not a positive or negative scientific result. The
firewall and dependency chain behaved correctly: two independent predeclared
failures stopped the experiment before science. The trace also showed that the
real forecast kernels can saturate the A100 once active, but the frozen workload
was too small to sustain utilization or provide a valid observation window.

## Project implications

Nothing from V1 supports forecasting, coordinate specificity, multiple local
laws, invariant subspaces, or sparse superiority. Its scratch root
`/network/scratch/l/lia/skae/global_k_residual_forecast_v1_20260721` must remain
unchanged as failure provenance. A repair may address only the operational
causes while preserving every scientific seed, checkpoint, map, comparator,
horizon, metric, inferential unit, gate, and decision rule.

## Next step

Use a fresh V2 root and independently freeze a narrow serialization repair:
cast telemetry checks to native Booleans and serialize an undefined rolling
statistic as JSON `null` while retaining strict nonfinite rejection. Increase
the prospective smoke and evaluation corpora to 98,304 distinct trajectories
each so that useful forecast evaluation, rather than dummy computation, spans a
robust telemetry window. Re-run independent byte-level review before launch;
do not reuse any V1 trajectory or outcome artifact.
