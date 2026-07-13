# SLURM Startup Diagnostic Report

Date: May 21, 2026
Working directory: `/home/mila/l/lia/skae`
Login host: `login-3.server.mila.quebec`

## Summary

The May 21 retry did not reproduce the earlier immediate SLURM startup
failure (`RaisedSignal:53` / exit `0:53`). Diagnostic jobs completed, compute
nodes wrote to both home and scratch, the baseline parent launcher completed,
all four child launchers completed, and two actual training arrays have reached
the Python trainer.

The only remaining non-start condition in this snapshot is scheduler priority:
the oracle-basin array is still pending with `Reason=Priority`, `ExitCode=0:0`,
and no assigned node, so it has no experiment stdout/stderr yet.

The login session still reports no Kerberos credential cache:

```text
klist: Credentials cache keyring 'persistent:1500001740:1500001740' not found
```

That said, the current compute-node write probes succeeded, so the previous
scratch/home write failure is not present in this run.

## Diagnostic Jobs

Shared-log GPU diagnostic:

```text
JobID: 9612278
State: COMPLETED
ExitCode: 0:0
Partition: main
Node: cn-a007
Elapsed: 00:00:20
Stdout: /network/scratch/l/lia/skae/diag-shared-9612278.out
Stderr: /network/scratch/l/lia/skae/diag-shared-9612278.err
```

The stdout begins with:

```text
START_SHARED_LOG
cn-a007.server.mila.quebec
Thu May 21 11:34:39 EDT 2026
```

The stderr file is empty.

Node-local-log and filesystem-write diagnostic:

```text
JobID: 9612279
State: COMPLETED
ExitCode: 0:0
Partition: main
Node: cn-a004
Elapsed: 00:00:21
```

The job successfully wrote:

```text
/home/mila/l/lia/skae/diagnostics/slurm_startup_20260521/home-write-9612279.txt
/network/scratch/l/lia/skae/scratch-write-9612279.txt
```

## Baseline Launcher Retry

Parent launcher:

```text
JobID: 9612280
State: COMPLETED
ExitCode: 0:0
Partition: main-cpu
Node: cn-m001
Stdout: /network/scratch/l/lia/skae/queue-cstab-route-baselines-9612280.out
Stderr: /network/scratch/l/lia/skae/queue-cstab-route-baselines-9612280.err
```

Parent stdout:

```text
Submitting support_family as staged_cstab_baseline_support_family_lista_full_20260519
Submitting oracle_basin as staged_cstab_baseline_oracle_basin_lista_full_20260519
Submitting latent_kmeans as staged_cstab_baseline_latent_kmeans_lista_full_20260519
Submitting random_matched as staged_cstab_baseline_random_matched_lista_full_20260519
Queued matched C_stab route baselines.
Submission record: results/staged_cstab_route_baselines_20260519_admin_retry/automation/route_baseline_queue_submissions.tsv
```

Parent stderr is empty.

Child launchers:

```text
support_family  queue job 9612281  COMPLETED  ExitCode=0:0  array=9612285
oracle_basin    queue job 9612282  COMPLETED  ExitCode=0:0  array=9612297
latent_kmeans   queue job 9612283  COMPLETED  ExitCode=0:0  array=9612292
random_matched  queue job 9612284  COMPLETED  ExitCode=0:0  array=9612289
```

All child launcher stderr files are empty. Each child launcher generated 225
tasks.

## Actual Training Array Snapshot

Snapshot time: Thu May 21 11:42:58 EDT 2026

```text
support_family array 9612285:
  0-15 completed with ExitCode=0:0 after skip/resume checks
  16-31 running on long partition
  32-224 pending due JobArrayTaskLimit

random_matched array 9612289:
  0,2-8 completed with ExitCode=0:0 after skip/resume checks
  1 running on long partition
  9-224 pending with Reason=Priority

latent_kmeans array 9612292:
  0 later started cleanly on cn-l082; 1-224 pending with Reason=Priority

oracle_basin array 9612297:
  0-224 pending with Reason=Priority, ExitCode=0:0, StartTime=Unknown
```

The running support-family task `9612285_16` has reached the trainer and is
printing stage-1 losses, e.g.

```text
Stage 1 step 7400/100000 loss=0.0842512 pred=0.0101174
Stage 1 step 7500/100000 loss=0.0827754 pred=0.00974953
Stage 1 step 7600/100000 loss=0.0831449 pred=0.0104519
```

The running random-matched task `9612289_1` has reached the trainer and is
also printing stage-1 losses. Its stderr contains only the CUDA module load
line:

```text
[=== Module cudatoolkit/12.6.0 loaded ===]
```

At Thu May 21 11:51:45 EDT 2026, latent-kmeans task `9612292_0` had started:

```text
State: RUNNING
ExitCode: 0:0
Reason: None
Node: cn-l082
Stdout: /network/scratch/l/lia/skae/staged-fabs-local-k-9612292_0.out
Stderr: /network/scratch/l/lia/skae/staged-fabs-local-k-9612292_0.err
```

Its stdout shows successful startup through imports, model construction, and
stage-1 training:

```text
Host: cn-l082.server.mila.quebec
Start Time: Thu May 21 11:50:21 EDT 2026
Routing object: latent_kmeans
Model loaded.
All core imports loaded.
Stage 1 step 0/100000 loss=3.22801 pred=2.16789
Stage 1 step 2200/100000 loss=0.216105 pred=0.0291897
```

Its stderr also contains only the CUDA module load line.

No tracebacks, OOM strings, CUDA errors, or Python exceptions were found in the
checked support/random/latent stdout/stderr logs.

Latest scheduler snapshot checked at Thu May 21 11:53:01 EDT 2026:

```text
support_family 9612285: tasks 16-31 running; remaining tasks pending by JobArrayTaskLimit
random_matched 9612289: tasks 1 and 16-29 running; remaining tasks pending by JobArrayTaskLimit
latent_kmeans 9612292: task 0 running; remaining tasks pending by Priority
oracle_basin 9612297: all tasks pending by Priority
```

`scontrol show job 9612297` reports:

```text
JobState=PENDING Reason=Priority ExitCode=0:0
StartTime=Unknown
Partition=long
ReqTRES=cpu=4,mem=24G,node=1,billing=1,gres/gpu=1
StdOut=/network/scratch/l/lia/skae/staged-fabs-local-k-9612297_4294967294.out
StdErr=/network/scratch/l/lia/skae/staged-fabs-local-k-9612297_4294967294.err
```
