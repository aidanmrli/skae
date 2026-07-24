# Allen–Cahn LISTA batch-16 utilization diagnostic — 2026-07-22

## Concrete result

Smoke job `10173588` was cancelled incomplete after 23:31. Shared frozen-`S` pretraining completed, and Q0/Q2/Q3 branches reached roughly 200--250 of 500 forecast updates. Their step-0 losses were identical. The final telemetry trace had 139 post-header samples: 71.64% all-sample mean, 89.71% active-sample mean, 100% peak, and 20,245 MiB maximum allocation. The isolated three-branch phase averaged only 41.58% over all samples and 74.55% over active samples.

## Experimental context

Shared pretraining used batch 32 and achieved 97.57% mean utilization. Three concurrent forecast branches used batch 16. The paired production design would have run only Q2 and Q3 at batch 16, making the high-utilization pretraining fraction negligible over 3,500 forecast updates.

## Interpretation

The partial run showed finite early optimization but did not complete its stability, learned-`S`, forecast-checkpoint, or external-validation gates. It therefore selects no refinement depth and is not scientific evidence. The branch phase was too underfilled to justify production.

## Project implications

The dependent batch-16 production and selector jobs never ran. The replacement keeps the scientific protocol and utilization thresholds fixed while increasing forecast batch size to 48. The observed memory supports a conservative three-branch estimate of about 60.7 GiB on an 80-GiB A100, leaving headroom; production uses only two branches.

## Next step

Require the fresh batch-48 smoke to pass all semantic, stability, forecast-improvement, provenance, memory, and utilization gates before launching the two paired seeds.
