# Allen–Cahn LISTA refinement smoke operational failure — 2026-07-22

## Concrete result

SLURM job 10173307 was cancelled after 22:12 before any branch completed pretraining. Logged progress was Q0 step 1200, Q2 step 1200, and Q3 step 700 of 2000. It produced no forecast metrics, learned-(S) update evidence, checkpoint comparison, or stability decision. Across 21 post-initial one-minute samples, GPU utilization was 17.7% mean, 26.5% conditional-active mean, and 42% peak, with approximately 8.6 GiB resident memory.

## Experimental context

The smoke redundantly ran the same frozen-(S) pretraining in three concurrent Q0/Q2/Q3 processes at batch size 8. Because (S=0) and is frozen, refinement count cannot affect this pretraining stage; the repeated work weakened pairing and underused the A100L.

## Interpretation

This is an invalid/incomplete operational result, not evidence against two or three learned refinements. A simple checkpoint copy is unsafe because a step-(-1) payload can rerun pretraining on resume and can retain Q0 model metadata if no forecast checkpoint improves.

## Project implications

The replacement protocol performs one shared frozen-(S) pretrain per seed, imports its exact model, optimizer, and minibatch RNG state through explicit warm-start semantics, and rewrites every branch checkpoint with the requested depth. It records source/model/RNG/first-minibatch hashes and requires nonzero (S) gradients and updates for positive depths.

## Next steps

Run a versioned Q0/Q2/Q3 smoke with batch 32 for shared pretraining, batch 16 for concurrent forecast branches, 10-second GPU telemetry, and fail-closed stability, identity, gradient, checkpoint-depth, and utilization gates. Only then launch paired seeds 64 and 65 for Q2 versus Q3 and choose depth by direct H200 cumulative field MSE.
