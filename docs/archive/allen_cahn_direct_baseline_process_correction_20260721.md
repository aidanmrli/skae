# Allen--Cahn matched-direct process correction (2026-07-21)

During outcome-blind validation of the matched nonlinear direct-forecast
baseline, two early commands used `salloc ... bash` without an explicit
`srun`. The shell therefore remained on the login host rather than entering
the allocated compute node. One resource query and a partial unit-test attempt
(allocation 10164839) are discarded in full. They produced no model
checkpoint, selector score, development evaluation, new-initial-condition
evaluation, or accepted scientific artifact.

The workflow was corrected to require `salloc ... srun ...` for every Python
or test command. The first corrected test allocation, 10165008, ran on a
compute node and exposed a test-only dynamic-import problem; that test was
repaired before the final outcome-blind freeze. The final frozen packet was
then validated in allocation 10165172 on `cn-f004` using an explicit `srun`:
all 10 focused tests passed.

The frozen dependency launcher also reads
`source_locked_command_graph.full_launch_authorized` from the task lock and
refuses to submit any scientific job while it is false. Only a non-scientific
GPU smoke is authorized at this stage.

The first authenticated A100L smoke, job 10165178, was finite and did not run
checkpoint selection or evaluation. It used 17,475 MiB peak GPU memory but
failed the frozen utilization gate (core mean 49.9375%, p10 6.0%). The raw
telemetry SHA-256 is `4845e863...e1265`; the failed audit SHA-256 is
`ac9f9de0...7340`. This failure is retained rather than filtered.

An outcome-blind execution repair batched finiteness reductions and tested
fixed-shape CUDA-graph replay. Allocation 10165214 correctly failed because
capture alone had not applied the second update. After adding exactly one
initial replay, allocation 10165223 passed parameter equivalence. Expanded
allocation 10165233 passed eager-versus-graph loss, parameter, and every Adam
state-tensor comparison through three updates. No selector or evaluation
outcome was opened during these repairs.

The authenticated 20-update graph smoke, job 10165256, was also finite and
did not run selection or evaluation. It reduced replay time to 0.891673 s and
then sustained 16 consecutive 98--100% utilization samples, but the unchanged
all-loop audit correctly failed because seven eager/capture samples dominated
the short 23-sample core (mean 75.913%, p10 1%). Peak memory was 18,511 MiB.
The raw telemetry SHA-256 is `6dd4488e...b250`; the audit SHA-256 is
`448ceef2...1e68`.

One final outcome-blind amendment fixes the smoke at 80 genuine updates while
retaining every startup, eager, capture, replay, and tail sample and keeping
the 85% mean and 80% p10 gates unchanged. No further smoke-length or
utilization amendment is allowed: failure makes scientific launch a NO-GO.

The terminal authenticated smoke, job 10165282 on `cn-d003`, completed all 80
updates without an OOM or non-finite value. CUDA-graph replay averaged
0.892644 s over 78 updates; PyTorch peak allocated and reserved memory were
16,631,233,536 and 17,874,026,496 bytes, respectively. The marker-bounded
optimizer loop had 73 retained samples, including five zero-utilization
samples, with mean utilization 88.863% but p10 utilization 56%. It therefore
passed the frozen 85% mean gate and failed the frozen 80% p10 gate. The job
exited 2 as designed. Raw telemetry SHA-256 is
`34815a1a8b41827817e3b85289b20b54456961c8fff63ecaf29d021c4d8ef1be`;
the failed telemetry-audit SHA-256 is
`de81cd05a4a0ade67853d1dda0807f91f5be45e4942c2f5cece127abbdf92598`;
the authenticated run-manifest and training-summary SHA-256 values are
`1f6b7483feda0c674f2cf365286bb61afd6edff8de5bace633d5524b92fd6891`
and `918dca8969d19207f59de23c033ea005c5efb8ee5e0123840546de8811f217d7`.

Under the predeclared terminal rule, the matched nonlinear direct baseline is
a scientific-launch **NO-GO**. No checkpoint selection, development
evaluation, new-initial-condition evaluation, or comparator outcome was run,
and this abandoned branch cannot support a forecasting claim.
