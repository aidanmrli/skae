# GPU-utilization optimization provenance (2026-08-21)

This is a dated archive note, not an active paper claim or experiment-status
tracker. It records the utilization work merged through PR #14 on `main` at
`b5f5e79`. The original working tree contained unrelated uncommitted paper
and experiment changes; that tree was preserved and was not used as the base
for this note.

## Concrete results

The work was split into small, reviewable pull requests:

- [PR #10](https://github.com/aidanmrli/skae/pull/10) added the utilization
  measurement harness and explicit provenance/eligibility receipts. The
  CPU/requeue validation was job `10438891` (one restart, 12/12 checks). The
  first RTX8000 baseline was job `10438950`: 19.603 steps/s and 2.0% GPU
  utilization over 13 samples. SM occupancy was unavailable because the
  counter query returned `ERR_NVGPUCTRPERM`; the receipt was therefore marked
  ambiguous/non-production.
- [PR #13](https://github.com/aidanmrli/skae/pull/13) bounded measurement
  windows so that profiling does not silently grow with a long run. The
  default window is 256 steps, with an explicit 256--8192 override; warmup,
  profile, and one-second sampling settings remain part of the receipt. Its
  4096-step baseline job `10439075` completed 4096 steps in 212.336 s
  (19.290 steps/s), with 2.0% GPU utilization over 211 samples and a
  260.476 s allocation wall time. A mismatched-window run was rejected before
  training, preserving its artifacts.
- [PR #11](https://github.com/aidanmrli/skae/pull/11) added complete,
  recoverable checkpoints: model, optimizer, progress, RNG states, data
  order, logging/best-state metadata, source/configuration identity, and
  storage identity. Writes are atomic, recovery selects the newest valid
  generation across scratch and permanent storage, retention is bounded, and
  preemption/time-limit signals can requeue the same job. The final real
  signal/requeue validation (job `10439340`) interrupted the task, requeued
  it, resumed at step 318, and completed the 1000-step test with matching
  scratch/permanent checkpoint hashes.
- [PR #12](https://github.com/aidanmrli/skae/pull/12) cached the exact
  deterministic gated-transfer sequence once and kept the contiguous data and
  geometry tensors on the device. This removes repeated CPU generation and
  host-to-device synchronization from the training hot path while preserving
  the FP32 sequence, loss, gradients, parameters, optimizer state, and
  serialized resume behavior covered by the PR tests.
- [PR #14](https://github.com/aidanmrli/skae/pull/14) scheduled metric/diagnostic
  work outside the hot path. It retains the diagnostics at their declared
  cadence and preserves the training objective; the tests cover the metric
  contracts and the homogeneous/zero-penalty cases.

The paired RTX8000-class comparison for PR #12 used main job `10439404` and
candidate job `10439405`:

| quantity | main | exact-cache candidate | change |
| --- | ---: | ---: | ---: |
| throughput | relative baseline | 7.326x baseline | 7.326x |
| GPU utilization | 2.0588% | 16.1429% | +14.0841 percentage points |
| measured training time | relative baseline | 13.65% of baseline | -86.35% |
| allocation wall time | relative baseline | 40% of baseline | -60% |

The PR #12 correctness and requeue evidence was recorded in jobs
`10439391`, `10439392`, and `10439422`; these covered the CPU/GPU equivalence
checks and the serialized interruption/resume path for the cache-enabled
runner.

The exact-GPU-UUID comparison for PR #14 used main job `10440121` and
candidate job `10440153`:

| quantity | main | scheduled-diagnostic candidate | change |
| --- | ---: | ---: | ---: |
| throughput | 283.385 steps/s | 312.026 steps/s | +10.10% |
| GPU utilization | 31.379% | 33.0% | +1.621 percentage points |
| measured training time | relative baseline | 90.82% of baseline | -9.18% |
| unprofiled phase time | relative baseline | 92.93% of baseline | -7.07% |
| total diagnostic allocation wall time | relative baseline | 104.53% of baseline | +4.53% |

The PR #14 correctness and requeue evidence was recorded in jobs
`10440116`, `10440117`, and `10440124`.

## Experiment context

The comparisons used the same declared workload and the same GPU class, and
the PR #14 pair used the same GPU UUID. The measurement harness separates the
warmup/profile window from setup and diagnostic overhead, reports both
measured and allocation wall time, and records configuration, source, device,
and continuation identity. This distinction matters because a change that
improves the measured training loop can still add fixed setup or diagnostic
work to the total allocation.

The PR #12 change targets the dominant CPU/data-transfer stall observed in the
baseline. PR #14 targets repeated diagnostics that were not needed on every
optimization step. Checkpointing is an efficiency safeguard rather than a
per-step speed optimization: a clean resume avoids discarding already-paid
GPU hours after a time limit or preemption.

## Interpretation and limits

GPU utilization is the fraction of time during which at least one kernel was
running. It is not SM occupancy. SM occupancy remained unavailable for these
runs because NCU's counter access returned `ERR_NVGPUCTRPERM`; no GPU-utilization
number above should be read as an SMO result. Consequently, these runs do not
establish a production SMO gate or an absolute RGU result.

For a paired run on the same GPU model, GPU count, and allocation protocol,
the per-hour RGU rate is constant up to the cluster's GPU coefficient. Thus
the ratio of elapsed allocation wall times is also the ratio of allocated
RGU-hours, even though the absolute RTX8000 coefficient was not available in
these receipts. The PR #12 allocation-wall reduction supports a relative
RGU-hour reduction under that assumption; its larger measured-window
reduction describes the training loop only. The PR #14 pair supports a
relative measured-window reduction, but its +4.53% total allocation-wall
change means that it does not support an end-to-end RGU-hour reduction.

The improvements are therefore evidence of less idle time and lower relative
allocation for the paired workloads, not evidence that the GPU reached high
SM occupancy. Results are sensitive to node state, sampling windows, and
fixed harness overhead; the receipts should be retained with any future
comparison.

## Project implications

The cache/device path is the largest demonstrated improvement: it moved the
measured GPU-utilization observation from about 2% to about 16% and reduced
paired allocation wall time by 60%. Scheduled diagnostics provide a smaller
additional gain in the measured loop when the baseline is already better
utilized, while making clear that diagnostic overhead must be included in any
end-to-end resource claim. Complete checkpoint state and requeue handling
protect these gains from being lost on interruption.

These provenance results are not incorporated into the active NeurIPS paper
claims or displays. They are retained here so that future code changes and
coauthor discussions can distinguish training-loop efficiency, allocation
efficiency, GPU utilization, SM occupancy, and RGU accounting.

## Next steps

1. Repeat the paired benchmark on a cluster configuration that grants the
   required NCU performance-counter permissions, and record SM occupancy with
   the same bounded-window protocol.
2. Obtain or document the authoritative GPU-to-RGU coefficient before making
   an absolute RGU-hour claim; until then report only paired relative timing.
3. Keep the cache and diagnostic changes behind their existing correctness,
   continuation-identity, and measurement receipts when testing additional
   workload sizes or GPU models.
