# Paper Experiment Evidence Map

Date: June 23, 2026

This is the paper-facing map for organizing the NeurIPS experiments section. It
compresses the live experiment record around the evidence chain in the draft:
multibasin Koopman learning motivates sparse supports, sparse supports produce
inspectable support objects, and merged support families can identify basin
structure without basin supervision. Labels and known basin counts remain
evaluation-only signals on benchmark systems; they are not part of
training-time method design or deployment.

Current manuscript organization: the main text now focuses on basin-support
identifiability, active-coordinate functional diagnostics, periodic support
refresh, forecasting performance, and a compact \(F_{\rm abs}\)-routed
learned-intercept local affine result on controlled multibasin systems. The
more elaborate \(C_{\rm stab}\) route has been moved to appendix diagnostic
context because matched route controls show that it is not necessary for the
main local-prediction gain.

May 14 related-literature positioning note: the Discussion now adds a bounded
connection to two sparsity literatures outside Koopman learning. LLM sparse
autoencoder work supports the idea that sparse dictionaries can expose
inspectable active features from dense activations, while continual-learning
sparsity work supports the broader idea that masks, sparse subnetworks, or
non-overlapping activations can reduce interference. This should remain framing,
not evidence: the paper's empirical claim is still basin-support alignment and
forecasting in sparse Koopman autoencoders, not LLM interpretability or
continual-learning performance.

May 26 matched route-baseline completion note: the reviewer-facing route
controls for the \(C_{\rm stab}\)-routed local affine result are now complete
and paper-facing. After fixing checkpoint resume (`last.pt` or
`checkpoint.pt`) and manually resubmitting aggregate jobs whose original
`afterok` dependencies had been canceled, support-family, random-matched,
latent-kmeans, and oracle-basin routing each produced `225/225` training rows
and `225/225` wide-periodic rows. All clean aggregate jobs exited `0:0`. The
controls match the retained \(C_{\rm stab}\) run on task roster, training
budget, stage split, fit data size, learned-intercept local maps,
checkpoint-selection rule, and wide periodic grid. The result is
baseline-sensitive: versus the matched global-\(K\) LISTA anchor, H1000 row
wins / median staged-global ratios are \(C_{\rm stab}\) `188/225` / `0.399`,
support-family `189/225` / `0.328`, random-matched `185/225` / `0.451`,
latent-kmeans `196/225` / `0.341`, and oracle-basin `195/225` / `0.366`.
Direct H1000 paired comparisons to \(C_{\rm stab}\) give support-family
`129/225` better rows and median baseline/\(C_{\rm stab}\) ratio `0.899`,
latent-kmeans `132/225` and `0.903`, random `95/225` and `1.058`, and oracle
`112/225` and `1.000`. Evidence-map implication: the main text should report
the simpler \(F_{\rm abs}\)-routed local \(K_c,d_c\) result, because it is
directly tied to the primary support-family object and is at least as strong
as \(C_{\rm stab}\) for forecasting. \(C_{\rm stab}\) should remain an
appendix support-flow diagnostic, not the headline route.

May 25 dense continuous tail-fate local-\(K_c\) control readout: no existing
staged dense-fate local-map run was found. The completed dense control only
clustered continuous latent tail summaries post hoc for basin-fate
identification, and the active `latent_kmeans` route baseline clusters LISTA
latent states rather than dense zero-sparsity MLP fate routes. The staged
trainer now supports `latent_tail_fate`: it clusters route-fit dense latent
trajectory tail summaries (tail mean/std/final features, silhouette-selected
\(k\le 12\)), assigns training transitions by trajectory fate, and routes
rollout states by nearest latent route center so future tail labels are not
used at deployment. A compute-node compile check completed as `9615576` with
exit `0:0`. The five-seed control completed under
[results/staged_dense_tail_fate_local_k_mlp_zero_seed5_20260521](/home/mila/l/lia/skae/results/staged_dense_tail_fate_local_k_mlp_zero_seed5_20260521):
launcher `9615608`, training array `9615609`, collect `9615610`, compare
`9615611`, and wide-periodic re-evaluation `9615612` all completed with exit
`0:0`, yielding `75/75` staged/global wide-periodic rows. Dense-tail-fate
local maps strongly improve the dense global-\(K\) anchor (`73/75`,
`71/75`, and `71/75` row wins at `H100/H500/H1000`; `70/75` all-horizon
wins; median `H1000` candidate/global ratio `0.0418`; geometric mean
staged/global ratios `0.149/0.109/0.108`). However,
against sparse \(C_{\rm stab}\)-routed LISTA on the same `75` system/seed
pairs, dense-fate does not match prediction performance: dense row wins are
`0/75` at `H100`, `2/75` at `H500`, `2/75` at `H1000`, and `0/75` on all
horizons. Median absolute best-periodic MSEs are dense-fate
`0.0522/0.0922/0.0962` versus sparse \(C_{\rm stab}\)
`0.000446/0.000297/0.000297` at `H100/H500/H1000`; geometric mean
dense/sparse ratios are `55.0/94.0/104.3`. The conceptual contrast should be
written explicitly in the paper: the dense control clusters trajectory-tail
features and routes a rollout state by nearest dense latent center for those
fate labels, whereas \(C_{\rm stab}\) routes from the current sparse support
mask through a learned support-transition graph and recurrent support-flow
component. This result supports keeping \(C_{\rm stab}\)-routed local maps as
prediction-relevant controlled-suite evidence rather than reducing the claim
to interpretability only. The interpretability claim is still central, but it
should be phrased as ``fate is represented as an inspectable support-flow
quotient of active coordinates,'' not as ``only sparse models can encode
fate.'' Data efficiency remains unclaimed because no sample-size sweep was run.

June 9 ManiSkill activation audit and corrected `5k` dense-tanh versus
sparse-KAE tuning note: the
previously reported dense ManiSkill periodic rows used the original ReLU MLP
blocks, not tanh. The controlled ManiSkill trainer now exposes an explicit
activation option, and dense no-sparsity baselines now default to tanh and
reject non-tanh baseline launches. This makes `dense_tanh_sp0` a true
dense-tanh baseline. The corrected packed-GPU `5k` comparison and follow-up
optimizer-fairness control, evaluated from `last.pt` with best periodic decoded-prediction
re-encoding over `{1,2,5,10,20,50,100}`, finds sparse KAE rows beating dense
tanh on seeds `0`--`2`. Standard dense tanh H10--H50 mean state MSE is
`0.002079`; applying the LISTA-favored optimizer setting
`lr=5e-4,weight_decay=0` improves dense tanh to `0.002023`. With the same
optimizer, sparse-MLP ReLU is strongest: `sparsity=0.003` reaches `0.001837`
and `sparsity=0.01` reaches `0.001860`. Best LISTA in the fairness rerun is
the standard optimizer row at `0.001881`; LISTA with `lr=5e-4,weight_decay=0`
is weaker on H10--H50 (`0.001968`) but stronger at H125 than standard LISTA.
Evidence-map implication: this keeps ManiSkill alive as a promising
benchmark-development candidate for sparse KAEs, but the fair
optimizer-control winner is sparse-MLP ReLU rather than LISTA. It should not
enter the main evidence order until checkpoint selection, periodic-cadence
selection, semantic outcome/contact labels, and support diagnostics are fixed
in advance.

June 9 ManiSkill two-track benchmark note: the robotics extension should now
be written as two separate paper-adjacent streams. The perturbation-balanced
`PegInsertionSide-v1` stream is the only ManiSkill stream aimed at
outcome/contact support-regime discovery, and it is currently blocked on a
semantic label audit because the stored `success/jam/miss/drop/partial`
classes are target perturbation labels rather than validated physical
outcomes. The ManiSkill-10 default-demo stream is a forecasting
generalization benchmark for Koopman representation quality under periodic
decoded-prediction re-encoding; it should not be used as support-alignment
evidence except for exploratory diagnostics. The current roster is
`PickCube-v1`, `PushCube-v1`, `PullCube-v1`, `PokeCube-v1`, `StackCube-v1`,
`RollBall-v1`, `PullCubeTool-v1`, `PushT-v1`, `PegInsertionSide-v1`, and
`PlugCharger-v1`, with `LiftPegUpright-v1` as the first replacement/smoke
task. Protocol artifacts are
[MANISKILL_INSERTION_BENCHMARK.md](/home/mila/l/lia/skae/docs/MANISKILL_INSERTION_BENCHMARK.md),
[MANISKILL10_DEFAULT_TASK_FORECASTING.md](/home/mila/l/lia/skae/docs/MANISKILL10_DEFAULT_TASK_FORECASTING.md),
and
[maniskill10_default_tasks.tsv](/home/mila/l/lia/skae/experiments/maniskill10_default_tasks.tsv).

June 10 ManiSkill execution note: the first CPU-only gates are complete. The
perturbation label audit (`9801691`) confirms the current `e20` packet has
balanced target labels but not validated five-way physical outcomes: available
signals reduce to `15` final successes and `85` untyped non-successes, and no
named contact/depth/distance/grasp/rim feature groups are present. This keeps
perturbation-balanced insertion out of the paper evidence order until a
semantic relabeler or regenerated packet exists. The ManiSkill-10 default-task
data path is viable: smoke `9801692`, full prep `9801694`, and rescue
`9801740` produced compact datasets for all ten selected official tasks
without GPU use. The only current coverage caveat is `RollBall-v1`, whose
downloaded packet reaches `68` transitions rather than the manifest H80
endpoint. Initial packed GPU smoke `9801761` failed immediately on a missing
`jq` dependency and was canceled after negligible accelerator use. Corrected
packed GPU smoke `9801812` completed `8/8` rows for `PickCube-v1` and
`PlugCharger-v1`; sparse rows are mixed on `PickCube-v1` but on
`PlugCharger-v1` beat dense tanh at short horizons and remain competitive at
longer horizons. GPU telemetry reached `100%` utilization, but the combined
active fraction was only `57.7%` because periodic evaluation/support
bookkeeping held the GPU during low-utilization phases. The launcher now
defaults to checkpoint-only GPU training, and
[run_maniskill10_eval_cpu_array.sh](/home/mila/l/lia/skae/scripts/run_maniskill10_eval_cpu_array.sh)
provides the dependent CPU-side evaluation/support pass for scale-up.

May 14 normalized-decoder and sparsity-target queue note: the next table
replacement candidate is `normdec_rollout`, which normalizes linear decoder
atoms for every KAE row, including Dense MLP, while preserving the current
rollout-latent sparsity target. The matched ablation is `normdec_encoded`,
which applies the L1 sparsity term to encoded latents for all sparse KAEs,
including LISTA-based models, because the paper's support objects are measured
on encoded states. Focused test job `9553712` gates controlled launchers
`9553720/9553721` and Dysts launchers `9553722/9553723`. Group sparsity for
block-diagonal rows is deferred: it is a different structural prior and should
be evaluated as a BD-specific ablation rather than mixed into the fair table
replacement.

May 15 normalized-decoder queue readout: `9553712` passed and the Dysts
`normdec_rollout`/`normdec_encoded` chains completed, but this is not a table
replacement yet. The retained-`10` Dysts `normdec_rollout` Sparse MLP-BD row is
`3.11e-4/0.0900/0.433/0.679` at `H100/H2000/H4000/H5000`, and the
`normdec_encoded` row is `2.66e-4/0.0878/0.472/0.704`. The rollout-target row
is best among normalized-decoder rows at `H2000/H4000`, but several other rows
worsen and the encoded-target ablation is mixed rather than clearly better.
The controlled multibasin launchers `9553720/9553721` failed before training
arrays because the backfill table only held seed-`10`--`14` rows for several
roots. The launcher is patched to merge seed-`0`--`9` sources with the
seed-`10`--`14` backfill; keep the current Table 1 and Table 2 sources active
until the controlled normalized-decoder replacement question is resolved
cleanly.

May 16 controlled normalized-decoder seed-3 readout note: the patched
controlled launcher produced a partial first read rather than a table
replacement. The first attempt (`9561145/9561146`) failed before task-table
generation because of a launcher-side multi-source path-list bug. The
corrected launchers `9561573` and `9561574` completed, emitting rollout
training array `9561575` and encoded training array `9561603`. Rollout
completed `226/270` tasks and timed out `44`; encoded completed `255/270` and
timed out `15`; downstream forecasting/support diagnostics collected for the
finished runs. Rollout-target sparsity favors plain Sparse MLP at longer
controlled horizons, while encoded-target sparsity gives a more balanced
controlled read and makes Sparse MLP-BD best at `H100`. Keep the current table
sources active until the timeout gaps are repaired or the full replacement
budget is rerun.

May 12 prospective benchmark-extension note: the manuscript now includes a
Discussion bridge and appendix protocol for two future benchmarks:
spatialized multibasin reaction-diffusion fields and controlled ManiSkill
insertion. These are positioned as next benchmark additions, not as completed
evidence. The PDE benchmark is the reliable high-dimensional scaling test; the
ManiSkill benchmark is the application-style contact/outcome regime test. Both
preserve the central protocol: no basin, outcome, contact, or regime labels
during training; labels are evaluation-only. Keep this appendix separate from
the current main evidence order until results exist.

May 12--14 benchmark-extension smoke/control note: separate one-seed-first scaffolds now
exist for both additions. Spatialized reaction-diffusion LISTA smoke job
`9530539` completed and wrote finite metrics, but the first result is not
paper evidence because final-basin consistency is `0.0`, late-training
sparsity collapsed, and final fields are mixed under the final-majority
diagnostic. ManiSkill prepare job
`9530626` completed after switching to raw downloaded `env_states`, producing
a `1000`-episode compact `PegInsertionSide-v1` state/action dataset; controlled
LISTA train/eval job `9530627` completed with finite rollout metrics but one
collapsed support family, so outcome/contact NMI and ARI are `0.0`. Treat
these as implementation validation until a non-degenerate metric file exists.
Follow-up one-seed tuning fixed the support collapses but is still not paper
evidence. The PDE setting `LISTA_ALPHA=0.001`, `SPARSITY_COEFF=0` gives
`H=12` basin consistency `0.917` and compressed support-family NMI `0.709`
with Jaccard `0.7`; the dense same-seed control forecasts better (`H=12`
field MSE `0.885` versus LISTA `1.032`) but has no compact support signal at
the matched threshold and needs `224` validation representatives to reach NMI
`0.616`. The
ManiSkill setting `LISTA_ALPHA=0.2`, `SPARSITY_WEIGHT=0.03` gives H100 state
MSE `0.248` and outcome NMI `0.347` in an overfragmented read; a compressed
`131`-family read has outcome NMI `0.303`. The dense controlled KAE has H100
state MSE `3.644` and no matched-threshold support signal; its best outcome-NMI
sensitivity reaches `0.312` only with `753` families. Keep both in the
benchmark-extension scratch lane: PDE needs a convolutional model-family
comparison, and ManiSkill needs perturbation-balanced state-only rollouts.

May 21 spatialized-PDE convolutional implementation and support-tuning note: the PDE benchmark now
has the missing convolutional model-family path (`conv_lista`, `conv_dense`,
and `conv_sparse_mlp`), label-settling metadata, same-horizon basin-map
metrics, Fourier-band MSE, support-size summaries, and task-table SLURM
launchers. Direct smoke job `9615803` completed with exit `0:0`; task-table
launcher `9615822` submitted array `9615825`, and all `3/3` child tasks
completed with exit `0:0`. A first controlled conv pilot then completed:
launcher `9615832` submitted array `9615833`, and all `6/6` child tasks exited
`0:0` for `cal_square_4` and `transition_routes_4`, seed `0`, and the three
conv variants. Both datasets are numerically clean (`[40, 13, 16, 16, 2]`,
invalid values `0`, clipped values `0`). Best validation MSEs are
`0.8990/0.8997/0.8991` on `cal_square_4` and `1.6977/1.7039/1.7039` on
`transition_routes_4` for `conv_lista`/`conv_dense`/`conv_sparse_mlp`; at
`H=4`, `conv_lista` has the lowest field MSE on both systems (`0.8751`,
`1.6752`). This upgrades the benchmark infrastructure and confirms finite
conv training/evaluation. The follow-up tuning pass diagnosed the collapse as
dense-support collapse and fixed it with `z_dim=128`, `hidden_channels=32`,
`num_blocks=2`, `lista_num_loops=3`, `lista_alpha=0.03`,
`sparsity_weight=0.05`, `support_threshold=0.3`, and `family_jaccard=0.8`.
Matched two-system, two-seed controls completed cleanly: mean
`H(B|F_tuned)` / `H(F_tuned|B)` / family count is `0.103/0.569/5.50` for
LISTA, `0.087/0.509/5.25` for sparse-MLP, and `0.173/0.612/5.25` for dense.
Secondary NMI/purity/H4-MSE is `0.751/0.938/0.970`,
`0.776/0.938/0.977`, and `0.708/0.875/0.962`. This moves the PDE benchmark
from collapsed-support infrastructure to a viable tuned-support benchmark
protocol, but not into the main evidence order: sparse-MLP is slightly lower on
`H(B|F_tuned)` and dense is non-degenerate, so any paper use should be framed
as benchmark/protocol evidence unless the scaled run reveals a clearer
sparse-LISTA advantage.

May 25 spatialized-PDE support-threshold correction: the lower-threshold rerun
confirms that the earlier `family_jaccard=0.8` setting was a non-collapse
diagnostic, not the right basin-scale family resolution. On the selected
LISTA checkpoints, `threshold=0.2`, `Jaccard=0.4` gives exactly `4.00` mean
families with `H(B|F)=0.217`, `H(F|B)=0.363`, and support size `58.8/128`;
the old `threshold=0.3`, `Jaccard=0.8` read overfragments to `5.50` families.
This is calibration history only: the screen used `grid16` states
(`d_x=512`) with `d_z=128`, violating the required overcomplete Koopman-lift
rule `d_z >= 4*d_x`. Future spatialized PDE runs must use `d_z>=2048` for
grid `16` and `d_z>=8192` for grid `32`.
A higher-sparsity LISTA screen makes `threshold=0.1` usable
(`alpha=0.2`, `sparsity=0.1`, `Jaccard=0.4`: `3.75` families,
`H(B|F)=0.227`, `H(F|B)=0.303`, support size `47.2`) and gives an exact
low-threshold read at `threshold=0.05`, `Jaccard=0.4` with
`alpha=0.2`, `sparsity=0.2` (`4.00` families, `H(B|F)=0.222`,
`H(F|B)=0.363`, support size `53.1`). `threshold=0.01` remains too dense or
unstable for compact families. Matched controls completed and sparse-MLP can
match the tuned read, so this does not move PDE into the main evidence order.
Use it as a benchmark-calibration note and require matched-control reporting
for any future PDE claim.

May 25 spatialized-PDE overcomplete rerun: the first valid overcomplete
interactive screen now satisfies the Koopman-lift rule with `grid16`,
`d_x=512`, and `d_z=2048`. The run covers `cal_square_4` and
`transition_routes_4`, seed `0`, with matched dense controls. At the fixed
PDE diagnostic rule `support_threshold=0.2`, `Jaccard=0.2`, and
`majority_fraction>=0.7`, `conv_lista` with `alpha=0.2`,
`sparsity_weight=0.05` gives dominant-slice `H(B|F),H(F|B)=0.000,0.231` on
`cal_square_4` and `0.000,0.277` on `transition_routes_4`, compared with
dense `0.231,0.549` and `0.382,0.277`. Sparse supports are far smaller than
dense (`78.4` and `106.9` active coordinates out of `2048`, versus `681.6`
and `610.7`). Forecasting is not uniformly better but is paper-plausible:
the sparse row wins H12 on `cal_square_4` and H4/H8/H12 on
`transition_routes_4`. Display implication: this justified the first
overcomplete expansion, but it is no longer sufficient to support a main-text
benchmark claim because the follow-up matched dense/sparse-MLP run is mixed.

May 25 spatialized-PDE expansion queue: the next-run gate is complete and
summarized. Parent jobs `9647692`, `9647693`, and `9647695` launched training
arrays `9647700`, `9647701`, and `9647702` for selected LISTA, dense, and
sparse-MLP. The expansion covers `cal_high_cross_3`, `var_l_shape_5`, and
`cal_pentagon_5` with seeds `0`--`4`, `grid16`, `d_x=512`, `d_z=2048`, and
`20,000` steps under `long`, `gpu:1`, `16G`, `4` CPUs, and `02:50:00`.
Dependent support-sweep queue parent `9647703` submitted support array
`9648659`, and all training/support tasks completed with exit `0:0`.
The aggregate weakens the main-text case: dense has the best short/mid-horizon
field forecasting (`H4/H8/H12` mean `0.4278/0.4374/0.5339`) versus LISTA
`0.4842/0.4817/0.5364` and sparse-MLP `0.4777/0.4798/0.5476` under the
time-averaged rollout MSE. Endpoint-only H12 `final_field_mse` is better for
LISTA (`0.7641`) than dense (`0.8746`) or sparse-MLP (`0.8121`), but this
still cannot support a long-horizon claim because the generated fields have
only `16` forecastable observation intervals. At the fixed
support diagnostic (`threshold=0.2`, `J=0.2`, `majority_fraction>=0.7`), LISTA
has much smaller supports (`19.3` active coordinates versus dense `78.5`) and
slightly lower `H(F|B)` (`0.623` versus `0.671`), but worse `H(B|F)` (`0.074`
versus dense `0.063`); sparse-MLP is best on `H(B|F)=0.034`. Display plan:
do not draft this as a main dominance result. Treat PDE as an appendix stress
test or hold it pending targeted sparse tuning that restores forecasting while
preserving compact supports.

May 26 spatialized-PDE long-horizon pilot: queued to replace the short-horizon
PDE read with valid H256/H512 forecast targets. Parent `9656822` completed and
submitted training array `9656825` plus dependent support parent `9656826`.
The task table stores `trajectory_length=512` and `label_extra_observations=512`
so H512 has ground truth, while `train_observation_limit=128` keeps training
windows inside the requested 128-observation training interval. The pilot
covers `cal_square_4` and `transition_routes_4`, seeds `0,1`, and
`conv_lista`/`conv_dense`/`conv_sparse_mlp` at `sequence_length=8`, `d_z=2048`,
and `eval_horizons=1,4,8,16,32,64,128,256,512`. The training/evaluation array
completed all `12/12` tasks, but the original `conv_dense` rows are superseded
because the PDE convolutional dense path used GELU hidden blocks instead of
the tanh dense-control convention. The code is patched, old support parent
`9656826` was canceled, and corrected dense-only parent `9659093`,
training/evaluation array `9659165`, and support-sweep array `9659514`
completed with exit `0:0`. Corrected tanh dense is the strongest forecaster
through H256, with H8/H16/H32/H64/H128/H256 field MSE
`0.242/0.302/0.917/1.441/1.819/2.126`, versus LISTA
`0.695/0.773/1.208/1.623/2.015/2.750` and sparse-MLP
`0.668/0.778/1.260/1.693/2.010/2.942` on finite rows; H512 MSE is non-finite
for every variant. Fixed support alignment favors sparse rows: corrected dense
deep-slice `H(B|F)=0.678`, support size `151.0`; LISTA `0.111`, support size
`7.6`; sparse-MLP `0.043`, support size `11.7`. Display plan: do not draft
this as a positive sparse forecasting result. Use it only as a high-dimensional
forecast/support tradeoff diagnostic unless targeted sparse tuning recovers
dense-matched forecasting. Follow-up status: the capacity audit found matched
width, depth, `d_z`, decoder, dense `K`, and training/data protocol; the
remaining confound is tanh dense blocks versus GELU sparse blocks. The
tanh-matched sparse/LISTA tuning pilot is complete: queue parent `9661532`,
rescue array `9662516`, replacement support parent `9662517`, and support
array `9662576` all exited `0:0`, producing `48/48` checkpoints, evaluations,
and support JSONs. Corrected tanh dense remains best through H32; tanh sparse
rows are only modestly better at H64/H128 and unstable by H256/H512. Support
alignment improves strongly: at `threshold=0.3`, `Jaccard=0.4`, LISTA
`alpha=0.05`, `sparsity=0` gives `H(B|F)=0.283`, `H(F|B)=0.241`, `4.25`
families, support size `17.5`, versus corrected dense `0.658/0.200`, `2.75`
families, support size `57.7`. Display plan: use this as a tradeoff
diagnostic, not a main positive result, and run a paper-consistent ReLU
`generic_sparse`-style convolutional sparse path before deciding whether the
PDE benchmark strengthens the paper.

June 2 spatialized-PDE controlled expansion note: the ReLU sparse path has now
been run at scale across the five high-dimensional systems and seeds `0`--`4`
under `spatial_rd_controlled_expansion_20260602`. Final counts are `25/25`
datasets, `125/125` checkpoints, `125/125` evaluations, and `125/125` support
sweeps after CPU rescue jobs. Forecasting does not support a main sparse win:
finite H1--H128 mean field MSE is tanh dense `0.593`, ReLU dense `0.737`,
ReLU LISTA `sp=0.01` `0.762`, ReLU LISTA `sp=0.05` `0.765`, and ReLU
sparse-MLP `0.775`; finite H1--H256 means are `0.725/0.866/0.868/0.884/0.908`
in the same order, and H512 remains mostly null/non-finite. The positive read
is support compactness/alignment: at `tau=0.05,J=0.4`, sparse rows reach deep
`H(B|F)≈0.049--0.053` with about `4.2--4.4` formed families and support sizes
`47--138`, while ReLU dense has `H(B|F)=0.119`, support size `522`, and tanh
dense has `H(B|F)=0.412`, support size `1048`. Display plan: keep this out of
the main evidence order; use it, if needed, as an appendix high-dimensional
stress test showing that sparsity makes basin identity readable as compact
support families even when dense tanh remains the better forecaster.

May 22 spatialized-PDE support-object correction: the selected conv PDE
checkpoints were re-evaluated with the manuscript's actual support objects,
not the tuned diagnostic threshold. With fixed \(S_{\rm abs}=\{|z_i|>10^{-3}\}\)
and \(F_{\rm abs}\) Jaccard `0.5`, \(F_{\rm abs}\) collapses to one family for
all `12` selected rows. On `deep_final_test_states`, mean `H(B|F_abs)=1.057`
and purity `0.469` for LISTA, sparse-MLP, and dense; on `all_test_states`,
mean `H(B|F_abs)=1.131` and purity `0.438`. Exact \(S_{\rm abs}\) has
`H(B|S_abs)=0.000` for LISTA and sparse-MLP, but with dense fragmented
supports (`~118/128` active coordinates, `95/97` exact supports on all states
for LISTA/sparse-MLP). This means the prior clean PDE support read is a
scale-tuned diagnostic, not evidence that paper-defined \(F_{\rm abs}\)
discovers compact basin families. Keep PDE out of the main evidence order
unless the next pass justifies a scale-normalized PDE support rule or shows a
clean \(C_{\rm stab}\) support-flow object.

May 19 ManiSkill perturbation-balanced assessment note: the first
perturbation-balanced state-only packet now runs end to end, but it is not
paper evidence. Generation job `9598665` produced `100` simulator rollouts
from `20` source episodes with target labels `success`, `jam`, `miss`,
`drop`, and `partial`, split by source episode into `70/15/15` train/val/test
rollouts. One-seed model jobs completed for LISTA, dense controlled KAE, and
sparse-MLP controlled KAE. Held-out state MSE at `H10/H25/H50/H100` is LISTA
`0.00369/0.00700/0.0496/15.91`, dense
`0.00384/0.00379/0.0231/7.83`, and sparse-MLP
`0.00352/0.00437/0.0651/143.43`. LISTA gives stronger target-label support
alignment than dense at the matched threshold (`22` families, outcome NMI
`0.616`; dense has one family and NMI `0`), but dense forecasts better at long
horizons and the current labels are perturbation targets rather than validated
physical outcomes. Do not add this benchmark to the main evidence order unless
the next pass validates actual post hoc outcome/contact labels and LISTA is
competitive with dense forecasting on held-out source episodes.

May 12 HyperLISTA d_z=256 diagnostic note: local interactive paired runs found
HyperLISTA can repeatedly beat matched d_z=256 LISTA on autonomous/no-reencode
stability, with positives on `cal_octagon_8`, `cal_square_4`,
`cal_hexagon_6`, `cal_pentagon_5`, `cal_high_cross_3`, and `var_diamond_4`.
The best `var_diamond_4` HyperLISTA variant used `c_theta=0.2` and reduced
no-reencode MSE from LISTA `18.75/1.731e15/2.098e33` to `1.64/4.31/4.37` at
`H100/H500/H1000`. `snic_multi` is positive only at later autonomous horizons
(`0.770/0.813` versus LISTA `4.48/364` at `H500/H1000`) and loses H100
(`0.374` versus LISTA `0.139`). `transition_routes_4` is likewise a
short-horizon counterexample under the fair sequence-8 comparison (`H100`
no-reencode `~2.10` for HyperLISTA versus LISTA `0.96`), and the sequence-32
adaptation is not a HyperLISTA win after matching LISTA's sequence length and
non-encoder knobs (`0.149/6.53e3/4.95e9` LISTA versus
`0.565/3.76e4/5.98e10` HyperLISTA at `H100/H500/H1000`). Standard LISTA still
wins every-step/best-reset on every checked system. This does not change the
main evidence order; treat HyperLISTA as an autonomous-stability diagnostic
unless a later run also improves reset-corrected encoder quality. The unfair
sequence-32 all-system queue was canceled. The corrected fair sequence-8
launcher `9529830` submitted array `9529842` and collect job `9529843`, but
those jobs were also canceled while the interactive search continued; no broad
HyperLISTA queue is active. A follow-up removed the duplicate base-loss
`1/horizon` scaling and tested larger HyperLISTA \(K\) learning rates. The
cleanup is conceptually correct but did not improve the fair results by itself:
the `transition_routes_4` `k_matrix_lr=1e-5` run reproduced the earlier
`2.027/4.811/5.204` no-reencode profile, and `snic_multi` became unstable at
`k_matrix_lr=5e-5` (`0.397/33.3/3.21e4`).

May 12 decoder-normalization fairness note: LISTA-style decoders normalize
dictionary atoms, while the MLP sparse Koopman autoencoder decoder has been an
unconstrained linear map. This creates a possible scale confound: decoder atom
norms can absorb scale that would otherwise appear in sparse coefficients and
their L1 penalty/support thresholds. `GenericKM` now has an opt-in normalized
linear decoder path, and launcher `9530394` queued a one-seed retained-10 Dysts
`dt x30` sensitivity pilot for Sparse MLP and Sparse MLP-BD with
`--normalize_decoder_atoms true`. Cache array `9530395`, GPU training array
`9530396`, long-horizon queue `9530397`, eval cache `9530746`, validation
`9530747`, eval array `9530748`, and collector `9530749` all completed with
exit `0:0`. The diagnostic is mixed: normalized Sparse MLP improves over the
matched seed-`0` old Sparse MLP row, but normalized Sparse MLP-BD is worse than
the matched seed-`0` repaired Sparse MLP-BD row, and neither pilot row beats the
displayed multi-seed Dysts table rows. Keep this outside the main evidence
order; it is a fairness sensitivity result, not a replacement table row. The
May 14 `normdec_rollout` queue supersedes the one-seed pilot as the only
candidate for replacing paper table rows.

May 12 strict staged local-\(K_c\) control note: fixed-center launcher
`9523663` generated `225` retained-Table-1 tasks and submitted array
`9523664`, but that predecessor was canceled after a strongly negative partial
read. The first source-target affine replacement
(`9527160 -> 9527170/9527171/9527172`) was canceled before completion to avoid
spending the full 15-seed budget before the pilot read. Reduced launcher
`9527220` submitted array `9527234`, collect `9527235`, and compare `9527236`
for a source-target affine same-budget LISTA experiment on the retained
multibasin roster with seeds `0,1`. The first `100000` of `200000` steps train
encoder, decoder, and global \(K\); then the autoencoder/global \(K\) are
frozen, label-free \(F_{\rm abs}\) families are formed with `absolute:0.001`
and Jaccard `0.4`, and all local \(K_c\) matrices train for the remaining
`100000` steps using \(z_{t+1}=d_c+(z_t-c_c)K_c\),
\(d_c=c_cK_{\rm global}\). The interactive target metric has been corrected
to the user's intended best-periodic comparison: compare staged local maps
against the best matched global-\(K\) periodic reencoding baseline over
periods `1,2,5,10,20,25,50,100`. Under that stricter read,
`claude:transition_routes_4` seed `0` is a clean multi-family positive
(`3` support families; staged `0.00510/0.0181/0.0901` versus global
`0.00728/0.0451/0.339` at `H100/H500/H1000`). Seed `1` is mixed under a
single-checkpoint requirement. The full same-budget two-seed read has now
landed: staged array `9527234`, collect `9527235`, compare `9527236`, and
wide-periodic re-evaluation job `9531292` all completed with exit `0:0`. After
re-evaluating all `30` staged/global checkpoint pairs over the same wide period
grid with batch size `100`, the result is heterogeneous rather than a
significant forecasting win: staged wins `19/30` rows at `H100`, `15/30` at
`H500`, `15/30` at `H1000`, and `15/30` on all horizons. Clean two-seed
all-horizon positives are `claude:cal_asymmetric_3`,
`claude:cal_pentagon_5`, `claude:transition_routes_4`, and
`claude:var_l_shape_5`; clean two-seed all-horizon negatives are
`claude:arrested_spiral`, `claude:cal_hexagon_6`,
`claude:cal_octagon_8`, and `gated_local_linear`. Keep this outside the main
evidence order as a heterogeneous mechanism/control result unless a tuned
recipe later produces a significant win. The aggregate artifact is
[wide_periodic_reeval_aggregate.md](/home/mila/l/lia/skae/results/staged_fabs_local_affine_k_lista_table1_seed2_20260512/wide_periodic_reeval_full/wide_periodic_reeval_aggregate.md).
A follow-up artifact audit found `144` trained local maps total, `4` to `6`
per run, from support families fitted on `4096` generated training
trajectories and `36864` latent states per run. The clean negatives are not
simply under-fragmented; they usually have a much stronger best-periodic global
baseline. The diagnostic artifact is
[local_k_diagnostics_summary.md](/home/mila/l/lia/skae/results/staged_fabs_local_affine_k_lista_table1_seed2_20260512/local_k_diagnostics/local_k_diagnostics_summary.md).
A learned-intercept contrast pilot completed but is not yet paper evidence:
`9553697 -> 9553698 -> 9553699` tested a trainable affine target center
\(d_c\) on two clean-positive and two clean-negative systems before any broader
scale-up. All jobs exited `0:0`, but the completed re-evaluation is
H100/periodic-1 only rather than the intended H100/H500/H1000 wide-periodic
grid. The restricted read wins `5/8` paired rows versus matched same-budget
global-\(K\) LISTA, with strong `transition_routes_4` wins but two
`cal_pentagon_5` losses. Treat this as tuning evidence; a corrected wide-grid
re-evaluation is needed before any scale-up decision.

The \(C_{\rm stab}\)-routed learned-intercept local-linearization probe has now
completed both the four-system/two-seed contrast and the full retained
multibasin scale-up. The contrast
`9562611 -> 9562654 -> 9562821 -> 9562824` replaces the instantaneous
\(F_{\rm abs}\) route with \(C_{\rm stab}\) stable support components and keeps
the learned affine intercept local-map parameterization. On the corrected
wide-periodic `H100/H500/H1000` grid, it wins `7/8` paired rows at every
horizon, with row-geometric staged/global ratios `0.271/0.431/0.453`.
The full retained multibasin Table 1 roster with all `15` seeds completed as
`9565762 -> 9565763 -> 9565764 -> 9565765`; all jobs exited `0:0`, and the
matched wide-periodic table has `225/225` `status=ok` staged/global rows.
Against the strong matched global-\(K\) LISTA baseline, staged local maps win
`189/225`, `188/225`, and `188/225` paired rows at `H100/H500/H1000`,
respectively, and `176/225` paired rows on all three horizons. Row-geometric
staged/global ratios are `0.269/0.202/0.182`; by system-level seed-geometric
ratio, staged wins `14/15` systems at each horizon and on all horizons, with
`duffing_triple_well` the only exception. This is statistically strong
controlled-multibasin evidence, but it should be positioned separately from
external robustness. Compact aggregate:
[wide_periodic_reeval_aggregate.md](/home/mila/l/lia/skae/results/staged_cstab_learned_intercept_k_lista_full_20260516/wide_periodic_reeval/wide_periodic_reeval_aggregate.md).
The retained `10`-system Dysts `dt x30` extension has
landed a partial negative read: training completed all `150` staged rows, but
the Dysts evaluation array left `44` staged rows missing after failed/OOM
packs. On the `106` complete staged/global pairs, staged wins `0/106` at
`H100/H500/H1000`, `15/106` at `H5000`, and `0/106` on all horizons. This is
not paper-positive Dysts evidence for the local \(K_c,d_c\) recipe.

May 18 manuscript integration note: the controlled-multibasin result is now
drafted as a self-contained main-paper section labeled
`sec:cstab_local_linearizations`. The section presents the local-linearization
motivation, the actual \(C_{\rm stab}\) support-flow routing object used in
the staged run, the exact absolute-mask and base support-family construction,
the definitions of \(u_t\), \(u\to v\), and route assignment thresholds, a new
routing diagram, the half-global/half-local learned-intercept affine training
recipe, and compact result table
[table_cstab_local_k_forecasting.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table_cstab_local_k_forecasting.tex).
Keep this display positioned as in-domain controlled-multibasin evidence; the
Dysts negative stress test remains a limitation rather than a table-positive
extension.

May 26 matched route-baseline note: the local-linearization section now has
matched route controls for the retained \(C_{\rm stab}\)-routed
learned-intercept recipe. The completed controls compare against instantaneous
\(F_{\rm abs}\) support-family routing, oracle basin routing, latent k-means
with matched route count, and random matched routes, while keeping the same
retained `225` task roster, `200000` total steps, `100000/100000` staged split,
learned-intercept local maps, stable-fit data size, checkpoint-selection rule,
and final wide-periodic grid. The repaired checkpoint-resume arrays completed
for oracle-basin `9645554`, support-family `9645583`, random-matched
`9645587`, and latent-kmeans `9645669`, and all resubmitted collect/compare/wide
jobs exited `0:0`. Against the global-\(K\) LISTA anchor, H1000 row wins and
median staged/global ratios are support-family `189/225` / `0.328`,
random-matched `185/225` / `0.451`, latent-kmeans `196/225` / `0.341`, and
oracle-basin `195/225` / `0.366`, compared with \(C_{\rm stab}\)'s `188/225` /
`0.399`. Direct H1000 paired comparisons to \(C_{\rm stab}\) show support-family
and latent-kmeans slightly ahead by median ratio, random-matched behind, and
oracle-basin tied. Use this as a local-map robustness/control result, not as
evidence that \(C_{\rm stab}\) is the uniquely best forecasting route.

May 14 stable support component diagnostic note: the new
\(C_{\rm stab}\) object merges base support-family labels by empirical
support-flow fate rather than by instantaneous Jaccard overlap alone. It is
still label-free at construction time and uses basin labels only for benchmark
audit. The four-system/two-seed p256 LISTA hard-init contrast is positive for
the support-object question but not for the local-map question: on the
per-basin deep slice, \(C_{\rm stab}\) gives coverage `1.0`,
\(H(B\mid C)=0\), \(H(C\mid B)=0\), and basin-matched object counts on all
`8/8` contrast pairs, while current \(F_{\rm abs}\) does so on only `4/8`.
The all-state read improves mean \(H(B\mid\cdot)\) from `0.420` to `0.221`
and mean \(H(\cdot\mid B)\) from `0.486` to `0.384` at mean coverage `0.959`.
The one-step local affine map probe remains negative: \(C_{\rm stab}\) is
never better than global on the deep slice and improves over current
\(F_{\rm abs}\) on only `3/8` pairs. Keep \(C_{\rm stab}\) outside the main
evidence order until validation-gated local-map or forecasting follow-up shows
that the improved support object also improves dynamics. A May 15 matched
encoder-family comparison gives the same caution for architecture claims:
\(C_{\rm stab}\) cleanly separates sparse-support encoders from dense
zero-sparsity MLP, but it does not distinguish LISTA from sparse MLP on the
four-system/two-seed contrast. Both LISTA roots and sparse MLP reach
deep-slice \(H(B\mid C)=H(C\mid B)=0\), NMI `1.0`, and `8/8` count matches,
while dense MLP collapses to one component with \(H(B\mid C)=1.685\), NMI
`0`, and `0/8` count matches. Do not use \(C_{\rm stab}\) yet as a
LISTA-over-sparse-MLP paper claim. The retained-15 two-seed follow-up is
stronger but not perfect. Sparse-root
array `9562395` completed `150/150` shards with `0` failures: perfect
deep-slice \(C_{\rm stab}\) matches are LISTA dense p256 `26/30`, LISTA
soft-block p256 `25/30`, LISTA-BD `26/30`, sparse MLP `27/30`, and repaired
sparse MLP-BD `24/30`. The dominant failures are overfragmentation on
`duffing_triple_well` and `gated_transfer_linear`. Dense latent-fate array
`9562516` completed `30/30` shards with `0` failures and changes the
interpretation: dense zero-sparsity MLP still collapses as a support object,
but continuous latent-tail clustering can recover strong basin information
(unsupervised deep \(H(B\mid{\rm fate})=0\), \(H({\rm fate}\mid B)=0.182\),
NMI `0.927`, `24/30` count matches). Therefore \(C_{\rm stab}\) should be
framed as evidence for an inspectable sparse support-flow quotient, not as
evidence that dense latents cannot encode basin fate. Do not promote
\(C_{\rm stab}\) to the sole main support object until the overfragmentation
and sensitivity checks are resolved.

May 6/12 paper-reorganization note: Table 1 is now a forecasting-only table
over all displayed horizons for the 15 controlled multibasin systems and the
10 Dysts \(dt{\times}30\) systems, including standalone state-space/local-linear
controls. Basin-support diagnostics are now a separate main-text support table,
and periodic \(F_{\rm top8}\) support-family refresh remains an appendix table.
The appendix file `docs/appendix/support_conditioned_predictors.tex` owns the
moved \(F_{\rm top8}\)-local predictor material.

May 7 supplementary-materials note: each item in the "Appendix or supplement
displays" todo list now has a separate LaTeX fragment under
`docs/supplementary_materials/`, plus an input manifest at
`docs/supplementary_materials/supplementary_materials.tex`. The fragments are
supplement-only: they preserve definitions, inventories, robustness
comparisons, controls, falsification diagnostics, and provenance without
changing the main-text evidence order.

May 7 Dysts robust-aggregation appendix note: the Dysts appendix now includes
an IQM-over-IQM sensitivity table generated with the repaired retained-`10`
Dysts `dt x30` Sparse MLP-BD row. The statistic is seed IQM within each system
followed by IQM across retained systems. Repaired Sparse MLP-BD is the best
row at every displayed horizon in this robust view; keep this as appendix
evidence for aggregation sensitivity rather than replacing the main
arithmetic-mean Dysts estimand. SLURM job `9491319` regenerated the matching
curve PDFs/PNGs from the repaired summary.

May 7 ground-truth multibasin visualization note: the multibasin benchmark
inventory appendix now includes the generated ground-truth vector-field
visualizations for all retained `15` systems. The appendix displays one
overview composite plus detailed three-panel figure groups with per-system
colorbars. Treat these as benchmark-geometry documentation only: the learner
still observes stored states, not vector fields, attractor centers, or basin
labels.

May 6 Sparse MLP-BD validity note: the old Sparse MLP-BD rows were not valid
block-diagonal MLP controls. The task specs requested
`k_structure=block_diagonal`, but `GenericKM` ignored `MODEL.K_STRUCTURE` and
always used a dense `kmat`. The code is now patched; the Dysts Sparse MLP-BD
row and the controlled Sparse MLP-BD row have both been rerun from the repaired
code path and now support the block-diagonal MLP transition claims shown in the
rebuilt table sources.

May 7/12 repaired Sparse MLP-BD result note: the retained-10 Dysts `dt x30`
repaired packet completed all `150/150` system-seed rows under
[results/dysts_dt30_sparse_mlp_bd_repaired_20260506](/home/mila/l/lia/skae/results/dysts_dt30_sparse_mlp_bd_repaired_20260506).
The main Dysts Sparse MLP-BD cells are now `3.96e-4`, `0.113`, and `0.436` at
`H100/H2000/H4000`; `H2000` and `H4000` carry exact system-sign/Holm `\ast`
superscripts, and `H4000` is now the best displayed Dysts cell. Controlled
multibasin forecasting plus per-basin deep support diagnostics also completed
under
[results/transition_rich_sparse_mlp_bd_repaired_table1_20260506](/home/mila/l/lia/skae/results/transition_rich_sparse_mlp_bd_repaired_table1_20260506)
with `225/225` forecasting rows, `6075` support rows, and `0` support failures.
Rescue array `9526863` and collection/support chain
`9526864 -> 9526865 -> 9526866` all completed with exit `0:0`. Table rebuild
job `9529647` regenerated the compact Table 1 and support-diagnostic sources.
The displayed controlled Sparse MLP-BD cells are `0.0473`, `0.136`, and
`0.159` at `H100/H500/H1000`, with support cells `0.358` for `H(B|F_abs)` and
`3.0` for `|F_abs|`.

May 6 support-refresh replacement completion note: the old refresh subtable
was tied to support-gated continuation and is superseded for the main text.
The replacement controlled-transfer support-switch audit completed under
[results/controlled_support_refresh_table1_seed15_20260506](/home/mila/l/lia/skae/results/controlled_support_refresh_table1_seed15_20260506)
for periods `1/10` and
[results/controlled_support_refresh_table1_seed15_periods5_20_20260506](/home/mila/l/lia/skae/results/controlled_support_refresh_table1_seed15_periods5_20_20260506)
for periods `5/20`. Both packets cover the six Table 1 controlled model roots,
retained 15 systems, seeds `0`--`14`, and `topk:8`, and both merged with
`1,345/1,345` specs, `285,804` rows, and `0` failures. The paper-facing table
object is \(F_{\rm top8}\): sparse-family after-reencoding target rates are
`0.994`--`1.000` over periods `1/5/10/20`, with no-transfer false target-family
rates only `0.009`--`0.019`. Table 2b should show each period as stale
pre-refresh family \(\to\) refreshed family. Exact top-`8` supports remain
useful as a fragmentation diagnostic, but not as the main refresh table object.

May 6 protocol clarification: the controlled multibasin models are trained on
broad boundary-emphasized rollout windows from the generator, not on the deep
slice. Table 1 now places the H100/H1000 forecasting columns first; those
columns use all held-out rollouts, not only deep states. The support-diagnostic
columns now use the evaluation-only per-basin deep-state slice, defined by the
top quartile of basin-depth margin within each benchmark basin. The
manuscript now states the rationale explicitly: broad training matches the
label-free deployment problem and preserves boundary/transient coverage, while
the per-basin deep slice isolates the cleaner basin-stratified question for static
support--basin alignment.

May 6 \(F_{\rm abs}\) versus exact \(S_{\rm abs}\) display note: the current
per-basin deep-slice rows now have two companion figures,
[fig_fabs_vs_sabs_basin_identification.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_fabs_vs_sabs_basin_identification.pdf)
and
[fig_fabs_vs_sabs_utility_tradeoff.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_fabs_vs_sabs_utility_tradeoff.pdf).
They support the calibrated claim that exact \(S_{\rm abs}\) masks are often
basin-pure but overfragment each basin, while \(F_{\rm abs}\) families are the
more useful basin-scale identification object. Use this as a table/appendix
companion if coauthors ask why the main count diagnostic is \(|F_{\rm abs}|\)
rather than \(|S_{\rm abs}|\).

May 6 initial-coordinate intervention note: a single-system p256 dense LISTA
seed-`0` ablation completed as replacement GPU SLURM job `9487431` under
[results/support_coordinate_interventions_20260506/gated_local_linear_lista_seed0](/home/mila/l/lia/skae/results/support_coordinate_interventions_20260506/gated_local_linear_lista_seed0).
The first CPU submission `9487332` and earlier GPU submission `9487372` were
canceled while pending; the `main` job requested one GPU, `1` CPU, `8G`, and
`03:00:00`, and completed in `27` seconds.
It directly perturbs the initial latent support on `gated_local_linear`: top
active coordinates of \(S_{\rm abs}=10^{-3}\) are cumulatively dropped from
`1` to `10`, and `20` random-support shuffles move the active coefficient
values to inactive latent indices. A follow-up coordinate-dropping-only pass
with `100` initial states writes
[results/support_coordinate_interventions_20260506/gated_local_linear_lista_seed0_n100](/home/mila/l/lia/skae/results/support_coordinate_interventions_20260506/gated_local_linear_lista_seed0_n100).
At horizon `21`, mean accumulated MSE is `0.0158` for the standard rollout and
`0.508/1.37/2.08/3.26/8.51` after dropping top `1/2/3/5/10` active
coordinates; the figure shows `95%` bootstrap confidence bands over initial
states. A random-support-only rerun with `100` initial states and `20`
shuffles per state writes
[results/support_coordinate_interventions_20260506/gated_local_linear_lista_seed0_n100_random](/home/mila/l/lia/skae/results/support_coordinate_interventions_20260506/gated_local_linear_lista_seed0_n100_random).
Across those `2000` point-shuffle outcomes, horizon-`21` accumulated MSE has
mean `1.69e3`, median `874`, and IQR `376`--`2.06e3`. The compact H21 table
now includes mean `±` standard deviation, median with interquartile range, and
paired sign-test \(p\)-values versus the standard rollout; all intervention
rows have \(p<10^{-16}\). The paper-style display
should use the linear-axis
absolute-MSE figures
[fig_support_coordinate_dropping_accumulated_mse.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_support_coordinate_dropping_accumulated_mse.pdf)
and
[fig_support_coordinate_random_shuffle_accumulated_mse.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_support_coordinate_random_shuffle_accumulated_mse.pdf),
not the earlier ratio plots. The output also contains horizon-MSE companions
for horizons `{1,3,5,7,9,11,13,15,17,19,21}` over `15` per-basin-deep stable
starts. The main text now includes the two accumulated-MSE panels and the
random-support trajectory panel as a representative mechanistic figure, with
the cross-system wrong-support-freeze table remaining the broader functional-use
evidence. This should be treated as strong companion evidence for the existing
wrong-support-freeze functional-use claim.

May 6 support-family value schematic:
[fig_support_family_value_diagram.svg](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_support_family_value_diagram.svg)
is a non-statistical diagram for coauthors asking what \(F_{\rm abs}\) is
actually good for. It uses the current alluvial example's concrete counts
(`393` exact masks to `3` support families on `4,656` deep states) and frames
the value as compression from brittle exact-mask IDs into a stable learned key
that can be named, inspected as a prototype barcode, and reused for routing or
caching local statistics.

May 6 spatial alluvial prototype note: the current alluvial is now accompanied
by three layout prototypes in
[spatial_alluvial_prototypes](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/spatial_alluvial_prototypes)
plus a data-driven renderer at
[tools/make_spatial_alluvial_prototypes.py](/home/mila/l/lia/skae/tools/make_spatial_alluvial_prototypes.py).
The prototypes test whether the main alluvial should show state-space source
points directly: (A) a spatial/alluvial/codebook triptych, (B) an alluvial with
spatial source thumbnails replacing the left basin blocks, and (C) spatial
support-family callouts to prototype barcodes. Treat these as display options
from existing assets, not new evidence.

May 6 reviewer-scope add-on note: three requested checks are partially landed.
The oracle/local-K control already exists in compact Table 2 partition-control
form, and a plain p256 LISTA autonomous-rollout add-on is still running under
[results/oracle_vs_learned_local_koopman_20260506](/home/mila/l/lia/skae/results/oracle_vs_learned_local_koopman_20260506).
The explicit regime-discovery baseline packet is complete under
[results/regime_discovery_local_koopman_20260506](/home/mila/l/lia/skae/results/regime_discovery_local_koopman_20260506),
covering k-means, diagonal GMM, and spectral clustering on raw state, dense
latent, sparse latent values, and binary supports. It is cautionary:
oracle-basin local-K is strong (`latent/global=0.2174`) and basin-count
clustering is also strong (`~0.30` best rows), while learned support-family
local-K is near global (`0.9827`). The supplemental out-of-generator packet has
completed training/collect, interpretability, and one-step regime discovery
under
[results/out_of_generator_multistable_p256_lista_20260506](/home/mila/l/lia/skae/results/out_of_generator_multistable_p256_lista_20260506)
for a gene toggle, thermal reactor, modified FitzHugh-Nagumo, and buckled
beam. It supports stable forecasting and deep-slice support purity, but its
one-step local-K routing result is negative. The autonomous OOD
oracle/local-K merge has also landed (`120` rows, `0` failures) and is
negative: family-local and oracle-basin rollouts are worse than global at H100
and unstable at longer horizons. Treat these as appendix/falsification audits,
not as main-text generality evidence for local-K routing.

May 12 baseline-coverage expansion note: additional reviewer-facing control
implementations have landed, and the standalone forecasting controls are now
tabled for both manuscript forecasting rosters. Corrected standalone tools add
DMD, polynomial EDMD,
RBF-dictionary EDMD, k-means hard local-linear dynamics, GMM hard local-linear
dynamics, and GMM soft-gated local-linear dynamics. Validation `9526657` and
smoke `9526658` passed. Launcher `9526668` submitted standalone array
`9526673` with seeds `0,1,2`; all `54/54` tasks completed with exit `0:0`
and no `status=error` rows under
[paper_baseline_suite_20260512_corrected](/home/mila/l/lia/skae/results/paper_baseline_suite_20260512_corrected).
That packet used a nine-system low-dimensional smoke/provisional roster, not
the retained `15` multibasin systems and `H100/H500/H1000` horizons in
[neurips_sparse_koopman_multibasin.tex](/home/mila/l/lia/skae/docs/neurips_sparse_koopman_multibasin.tex).
Do not use its aggregates as manuscript evidence; retained-15 standalone
launcher `9527268` completed after updating launcher defaults to the paper
roster and submitted baseline array `9527269` (`90` tasks, `0-89%32`), which
completed `90/90` tasks with exit `0:0`, `810` `status=ok` rows, and no
nonempty stderr logs. On the retained-15 paper roster, raw cumulative MSE
aggregates at `H100/H500/H1000` are:
RBF-dictionary EDMD `0.660/0.808/0.751`, polynomial EDMD
`0.864/0.968/0.887`, DMD `2.58/2.14/1.70`, k-means local-linear
`1.40/3.04/137.7`, GMM-hard local-linear `1.65/3.57/3.93`, and GMM-soft
local-linear `1.66/3.71/4.07`. RBF-dictionary EDMD wins the aggregate at all
three horizons and wins `11/15`, `11/15`, and `12/15` systems respectively.
Compared with the existing Table 1 KAE rows, it beats the dense-latent MLP
baseline (`0.830/2.68/2.93`) but not any sparse KAE row; the best sparse rows
are `0.0387/0.0940/0.107`. The Dysts `dt x30` extension completed under
[paper_baseline_dysts_dt30_20260512](/home/mila/l/lia/skae/results/paper_baseline_dysts_dt30_20260512):
array `9530093` completed `59/60` tasks, repair task `9530233_48` completed
the remaining Sakarya mixture task after bounded-state filtering, and the final
output has `540` `status=ok` rows with `0` `status=error` rows. On Dysts,
RBF-dictionary EDMD is the best standalone row (`0.890/2.84/2.91` at
`H100/H2000/H4000`) but is worse than Dense MLP (`0.00110/0.224/0.754`) and all
sparse KAE rows; local-linear mixture rows are unstable at long horizons. The
main manuscript now includes these controls directly in the forecasting-only
Table 1, while support diagnostics are displayed in a separate main-text table.
Launcher `9523987` also produced the merged p256 checkpoint-backed
regime/oracle local-K packet under
[paper_regime_oracle_baselines_20260512](/home/mila/l/lia/skae/results/paper_regime_oracle_baselines_20260512)
with `51/51` completed runs, `1377` rows, and `0` run-level failures. Treat
the `43` failed row-local `gmm_diag` assignments as a route-family caveat.
Off-target smoke aggregates: standalone H50 raw cumulative MSE ranks fixed-count
k-means hard first (`0.227`), RBF-dictionary EDMD second (`0.304`), polynomial
EDMD third (`0.347`), then GMM-soft (`0.438`), GMM-hard (`0.441`), and DMD
(`0.617`). In checkpoint-backed local-K, oracle basin labels give the
upper-bound diagnostic (`0.211` latent/global), sparse-latent k-means with the
evaluation-only basin count is the best explicit clustering route (`0.282`),
and learned support families are the only label-free support-family-count
route below global (`0.954`).
Keep checkpoint-backed local-K controls with the appendix
baseline/falsification material first. The standalone forecasting controls are
now suitable for the compact main-text table because they directly answer the
classical Koopman/local-linear baseline question; they do not change the
primary sparse KAE comparison because they are worse than sparse rows on both
rosters.

June 23 label-free local Koopman/EDMD baseline note: the standalone baseline
table now includes the fair multi-model/switching-style control requested for
multi-regime systems. The new implementation fits local polynomial EDMD and
local RBF-EDMD operators behind k-means routes, selects `k` from `1,2,4,8,16`
by validation rollout error on held-out training trajectories, and never uses
basin labels or known basin counts for training or selection. CPU-only arrays
`9914514` and `9914513` completed all retained-multibasin and Dysts tasks.
The updated standalone table is
[table_standalone_state_space_baselines.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table_standalone_state_space_baselines.tex).
Local polynomial EDMD is the new strongest stable standalone row:
`0.150/0.252/0.275` on retained multibasin H100/H500/H1000 and
`5.0e-4/2.17/2.97` on Dysts H100/H2000/H4000. This is strong enough to report
as a credible baseline, but it still does not beat the sparse KAE rows on
retained-multibasin forecasting and it has no support-alignment readout.
Relative to the current paper table, retained-multibasin local polynomial
EDMD is `3.7x/1.8x/1.7x` worse than LISTA and `3.9x/2.7x/2.6x` worse than the
best sparse KAE row at H100/H500/H1000, while beating Dense MLP by
`5.5x/10.6x/10.7x`. On Dysts, it is a serious H100 control but loses by
`22x/6.8x` to the best sparse KAE row at H2000/H4000 and is worse than Dense
MLP at those long horizons.
Local RBF-EDMD has ordinary medians but enormous arithmetic means from rollout
blow-up outliers, so it should be shown as unstable rather than omitted.
Evidence-map implication: cite the label-free local EDMD protocol in the
baseline/falsification discussion, and keep the main evidence order focused on
support alignment, functional support interventions, refresh, and sparse-KAE
forecasting.

May 6 support-family local-K follow-up: the first p256 support-family
hyperparameter iteration is active under
[results/regime_support_family_hparam_p256_20260506_iter1](/home/mila/l/lia/skae/results/regime_support_family_hparam_p256_20260506_iter1).
It sweeps support definitions and Jaccard thresholds with explicit clustering
disabled, so each worker reports only global \(K\), learned support-family
local \(K\), and oracle-basin local \(K\). The purpose is to check whether the
near-global learned-family result (`0.9827`) is a poor route-construction
setting rather than an intrinsic limitation. Keep the result appendix/control
facing unless it materially closes the oracle gap. A partial `09:16 EDT`
diagnostic points to high-J absolute-threshold supports as a promising region,
but this is incomplete and should not be drafted as a result. Conditional
refinement launcher `9480079` is held behind the first summary to test that
region over min-transition and ridge settings.

April 29 live recomputation note: the expanded table-refresh artifacts have
landed and the manuscript tables have been recomputed. The fixed-`17`
five-root forecasting source has `1261` rows; Table 1 interpretability has
`34,047` rows and `0` failures; Table 2 self-routed forecasting has `9,796`
rows and `0` failures; the support-refresh packet has `189,708` rows and `0`
failures; the Dysts packet collected `894/900` requested rows, with the six
missing rows confined to LISTA-BD. The controlled multibasin tables strengthen
the paper. This note is superseded for the current Dysts display by the fully
collected `dt x30` packet and the retained-`10` aggregation that excludes the
six-dimensional `dysts:LorenzCoupled` and Chua-family duplicate
`dysts:MultiChua`: Dense MLP is worst at every horizon, all primary sparse
rows beat Dense in aggregate at every displayed horizon. With the repaired
Sparse MLP-BD row, the compact displayed aggregate-best rows are Sparse MLP at
`H100`, LISTA at `H2000`, and repaired Sparse MLP-BD at `H4000`.
Treat Dysts as an external forecasting/discretization-sensitivity stress test,
not as support-basin evidence. In the manuscript, LISTA-SB is framed as a
soft-block ablation for the controlled support/routing mechanism and as an
appendix-only Dysts diagnostic.

April 30 verification note: the active Table 1 aggregator and the Tables 2-4
within-system paired-test script were rerun on compute allocation `9411806` and
reproduced the manuscript-facing counts. The draft tables are current with the
expanded seed artifacts.

April 30 Table 2 two-horizon/matched-dimension update: at the user's request,
Table 2 now reports both `H100` and `H1000` routed/global ratios side by side,
and the LISTA-SB row has been replaced by the matched `d_z=256` self-routing
artifact. The paired-test JSON and routing summary CSV were regenerated on
compute allocation `9420481` from the expanded five-model seed-`0`--`14`
self-routed artifact plus the matched `d_z=256` LISTA-SB packet. Displayed
point estimates in that intermediate pass used the old trimmed across-system
aggregate, before the May 3 mean-over-systems correction.
The `[K/17]` counts use within-system Wilcoxon/Holm tests over all seed slots
with censored handling for invalid H-step routed/global comparisons. The
matched row weakens exact-support routing but preserves strong family-local
routing, especially at `H1000`.

May 3 Table 2 statistics/narrative update: the main routing table now
foregrounds `F_top8` family-local routing and moves exact/gated `S_top8`
route forms to appendix diagnostics. The displayed ratio is
`10^{mean_s D_s}`, where `D_s` is the within-system IQM of finite seed-level
`log10(routed/global)` ratios and the outer mean is taken across retained
systems. The companion system-wins column uses the censored per-system
comparison, counting finite routed/global-invalid cases as routed wins. Main
significance stars come from a one-sided exact sign-flip test across
system-level finite log effects, Holm-corrected over the displayed
model--horizon cells. This replaces the previous within-system `[K/17]` count
as the main Table 2 inference.

May 3 layout update: the compact routing table and compact support-refresh
table now appear side by side in the manuscript as subtables 2a and 2b. The
Dysts actual-MSE display is therefore the current manuscript Table 3, although
historical notes may still call that Dysts artifact Table 4.

May 4 manuscript prose/display update: the visible paper todos have been
cleared in `docs/neurips_sparse_koopman_multibasin.tex`. The manuscript now
includes the method overview diagram, a compressed experiments protocol, a
completed related-work section and conclusion, and an appendix
ratio-to-Dense Dysts table that supports the average-case interpretation of
the main actual-MSE Dysts display. This is a prose/display alignment update,
not a new experiment result.

May 5 Methods visual update: use
[fig_methods_support_family_pipeline.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_methods_support_family_pipeline.pdf)
as the main Methods schematic. It replaces the earlier sectorized overview with
a cleaner two-panel white-background diagram: a label-free support route from
\(x_t\) through sparse codes and support-family keys to evaluation/routing,
and a zoomed greedy Jaccard support-family construction panel without basin
labels or basin counts. This is a schematic display artifact, not a new
experiment result.

May 6 training-dynamics appendix note: use
[appfig_training_dynamics_gated_local_seed0.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/appfig_training_dynamics_gated_local_seed0.pdf)
and
[appfig_training_dynamics_dysts_dt30_seed0.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/appfig_training_dynamics_dysts_dt30_seed0.pdf)
only as appendix diagnostics of optimization and checkpoint selection. The
controlled panel plots `gated_local_linear`, seed `0`, for the six controlled
rows. The Dysts panel aggregates seed-`0` traces over the retained `10` Dysts
`dt x30` systems by the median over finite per-system metric values at each
step. Both show validation final error, training objective, \(\rho(K)\), and
the logged sparsity-ratio diagnostic. The controlled validation-selected
checkpoint markers mostly occur late for sparse rows, and \(\rho(K)\) stays
close to the unit circle; the Dysts validation proxy is noisier because
chaotic rollout errors can become very large early in training. Neither figure
should be presented as aggregate benchmark evidence or as a claim that
closest-to-one spectral radius is universally optimal.

May 5 p256 visual-draft update: the Support Barcode Map and Alluvial
Basin-to-Support-to-Family drafts have been regenerated from p256
LISTA-family checkpoints only. Use
[fig_support_barcode_map_p256.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_support_barcode_map_p256.pdf)
as the barcode candidate: p256 dense LISTA on `gated_local_linear`, with the
displayed basin-family prototypes overlaid on one shared `0..255` barcode.
Use
[fig_basin_support_family_alluvial_p256_gated_deep_unbold_headers.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_basin_support_family_alluvial_p256_gated_deep_unbold_headers.pdf)
as the current alluvial candidate: p256 dense LISTA on `gated_local_linear`,
seed `0`, global deep-state slice, `S_abs=10^{-3}`, family Jaccard threshold
`0.5`, `3` represented evaluation basins, `3` support families, `393` exact
supports, and family-dominant basin agreement `1.0`. It keeps the original
alluvial geometry and changes only the column headers to normal weight. The
compact polished and all-state `J=0.32` p256 versions remain
sensitivity/alternate artifacts; earlier p64 drafts are superseded and should
not be used as paper-facing figures.

May 6 spatial alluvial follow-up: if the alluvial remains in the main text,
consider replacing the pure-block layout with the source-column prototype or a
triptych generated by
[tools/make_spatial_alluvial_prototypes.py](/home/mila/l/lia/skae/tools/make_spatial_alluvial_prototypes.py).
The unresolved display question is whether spatial context belongs inside the
alluvial panel itself or as an adjacent companion panel.

May 5 Jaccard-threshold sensitivity update: the sweep under
[results/support_family_jaccard_threshold_sweep_20260505/full_retained15_deep_subsetfit](/home/mila/l/lia/skae/results/support_family_jaccard_threshold_sweep_20260505/full_retained15_deep_subsetfit)
confirms that the family Jaccard threshold sets the resolution of the
support-family object. Low thresholds merge distinct basins into too few
families; high thresholds can make \(H(B\mid F)\) vanish by splitting each
basin into many families. Keep the main paper at the fixed `J=0.5`
convention, and use the sweep only as appendix/sensitivity evidence. Do not
make a main narrative claim from high-J zero entropy unless the corresponding
family count is shown.

Whole-slice sensitivity note: at the fixed `J=0.5` threshold, the closest
all-state `F_abs` family-count match is Sparse MLP-BD (`4.217` families versus
`4.20` basins), followed by Sparse MLP (`4.244`). This should be interpreted
as count compression only, because the same rows have higher whole-slice
\(H(B\mid F_{\rm abs})\) than the LISTA-family rows. The main paper should
continue to use the deep-state slice as the cleaner basin-interior alignment
test.

May 5 routed-forecasting-MSE completion note: the experiment plan is
[ROUTED_FORECASTING_MSE_PLAN.md](/home/mila/l/lia/skae/docs/ROUTED_FORECASTING_MSE_PLAN.md).
Two smoke jobs completed with `0` failures (`9466086` controlled multibasin,
`9466087` Dysts), and the full add-on completed raw-row collection:
`9466089`--`9466223` plus merge `9466224` wrote `675` retained-`15`
controlled rows, and `9466225`--`9466314` plus merge `9466315` wrote `450`
retained-`10` Dysts rows. Both merges have `0` failures and complete
model-system-seed coverage. Aggregation job `9467050` then compared routed
`F_top8` family-local centered rollouts with the current same-model
best-periodic Table 1/Table 3 rows using seed IQM within each system followed
by arithmetic means over systems. The result is negative: routed forecasting
wins `0/15` controlled systems and `0/10` Dysts systems for every LISTA-family
model at every evaluated horizon. Do not alter the main evidence order or add
these rows to Table 1/Table 3 as superior forecasting evidence; at most, keep
the result as a falsification/mechanism check showing that useful support
families for local routing do not automatically yield stable autonomous
forecasting rollouts.

May 5 periodic routed-forecasting rerun note: the autonomous routed result did
not settle the user's intended route-refresh hypothesis, so the evaluator was
extended with periodic decode/re-encode via `--reencode_periods`. The rerun is
now complete for periods `5,10,20,30`: compile job `9467340`, smoke jobs
`9467341`/`9467342`, controlled shards `9467343`--`9467477` plus merge
`9467478`, Dysts shards `9467479`--`9467568` plus merge `9467569`, and
aggregation job `9467598` completed with `0` recorded failures. The merged raw
outputs contain `2700` controlled rows and `1800` Dysts rows. Periodic refresh
reduces many autonomous blow-ups and improves route coverage, but it does not
make post-hoc \(F_{\rm top8}\)-family local maps a superior forecasting method:
every aggregate routed/best-periodic ratio remains above `1`, with only
isolated system wins in individual cells. Keep this as a
falsification/mechanism check, not as a main Table 1/Table 3 forecasting row.

May 6 stage-2 fixed-setting expansion result: the `F_top8`, `J=0.40`,
period-`5`, `reroute_each_step` support-family-local \(K_c\) expansion has
now landed enough evidence to decide paper positioning. The 10-seed 50k
controlled multibasin run completed with `0` failures and high route coverage
(`0.9982`), but it is mixed against the same-root best-periodic sparse-LISTA
rollout: routed/best-periodic ratios are
`1.70/0.91/0.84` at `H100/H500/H1000`, with only `4/15`, `5/15`, and `5/15`
system wins. Against the current hard-init Dense MLP Table 1 comparator, the
same controlled row is positive: stage-2/Dense ratios are
`0.0866/0.0513/0.0490`, `14/15` systems have lower per-system seed-IQM MSE at
each horizon, and system-level one-sided Wilcoxon p-values Holm-correct to
`9.16e-4` across the three horizons. The matching 10-seed 50k Dysts run is strongly negative:
`0/10` system wins at every horizon and routed/best-periodic ratios already
`6.9e3` at `H100`. The 100k controlled continuation completed but is worse
than 50k, while Dysts 100k has `97/100` successful rows, three local-map
class-count failures, and unstable available metrics. Table 1 now includes
this as `LISTA + local \(K_c\)`, labeled as a trained local-map forecasting
variant with support diagnostics inherited from the frozen LISTA encoder. It
should be interpreted as Dense-competitive/significant, not as a best-periodic
sparse-LISTA replacement or a Dysts-robust forecasting method. A no-refresh
controlled rollout over the same 50k period-`5` trained \(K_c\) checkpoints
completed with `0` failures and is strongly negative: routed/best-periodic
ratios are `5.26e8/3.84e33/4.80e34`, system wins are `0/15` at all three
horizons, route coverage drops to `0.2161`, and aggregate routed/no-reencoding
global-\(K\) ratios are `2.12e5/1.23e9/2.31`. This means the Table 1
local-\(K_c\) row must be described as a periodically refreshed local-map
rollout.

May 6 Dysts stage-2 re-encode-period sweep queued: the period-`5` Dysts
support-family-local result above is being stress-tested over periods
`{1,2,5,10,20}` by reusing the completed period-`5` rows and queuing the
missing periods `{1,2,10,20}`. Seed-half merges are `9478074` and `9478276`;
combined analysis job `9478278` will write
`results/routed_stage2_local_maps_20260506/combined_best_lista_dysts_j040_50k_period_sweep_labelnone_seed0_9`.
The controlled multibasin result is fairly competitive at `H500/H1000`, but
it is already significantly better than the current hard-init Dense MLP Table
1 comparator. The display plan should therefore allow a controlled-table
stage-2 row if needed, while keeping Dysts robustness pending and avoiding an
external-robustness claim unless the period sweep overturns the Dysts failure.

May 6 calibrated-global stage-2 ablation result: the matched
`stage2_map_mode=global_dense_calibrated` 50k raw-row batches completed for
controlled and Dysts with `0` failures, but this is not a positive calibration
control. Controlled simple means are `H100/H500/H1000 = 16.4/6.86e31/0.125`,
and Dysts simple means are catastrophic from `H100` onward. Combined
aggregation jobs exposed script/schema issues, so raw rows and seed-half
summaries are the current source. The paper-facing interpretation is that
second-stage rollout calibration alone is not a reliable forecasting fix.

Current manuscript layout: remove the alluvial candidate from the main-text
support-alignment display and wrap a cropped seed-level
\(H(B\mid F_{\rm abs})\) panel on the right side of the body text. Omit the
\(|F_{\rm abs}|\) and wrong-support ratio strips from the main-text visual
display; the count and functional wrong-support evidence remain in Table 1 and
appendix per-system tables. Keep the active-index codebook as a separate
full-width figure.

May 3 retained-benchmark aggregation update: at the user's direction,
`multiwell_strong_transition` and `claude_checkerboard_potential` are now
excluded from the controlled multibasin benchmark, leaving `15` retained
systems. Main point estimates use seed-IQM within each retained system
followed by an arithmetic mean across retained systems. Tables 1--3 have been
regenerated under this estimator: every sparse-latent row beats Dense MLP on
retained-system mean raw MSE at `H100`, `H500`, and `H1000`; Table 2 routing
denominators are `15` retained systems; and the support-refresh display has
`12` eligible transfer systems. The Dysts `dt x30` table has also been
regenerated after excluding `dysts:LorenzCoupled`, the only six-dimensional
member of the original Dysts shortlist, and `dysts:MultiChua`, to keep a
nonredundant `10`-system three-dimensional shortlist. On the retained systems,
all primary sparse rows beat Dense MLP in aggregate at every displayed
horizon. Under the exact system-sign/Holm main-table rule, the current compact
Dysts display uses `H100/H2000/H4000`; repaired Sparse MLP-BD clears `H2000`
and `H4000`. The displayed aggregate-best rows are Sparse MLP at `H100`, LISTA
at `H2000`, and repaired Sparse MLP-BD at `H4000`.

April 30 matched Table 1/support-refresh completion note: the matched
`d_z=256` Table 1 interpretability diagnostics and support-refresh rows completed with
`0` failures. Interpretability jobs `9412879 -> 9412880` wrote `6,885` rows;
support-refresh jobs `9412881`--`9412883 -> 9412884` wrote `191,400` rows.
Tables 1--3 were recomputed on allocation `9420481` and the manuscript/PDF now
use the matched LISTA-SB row where available. Dysts matching is deferred to the
later full-horizon run.

May 1 dense-K LISTA add-on completion note: the plain dense-K LISTA row is now
complete and folded into Tables 1--3 under
[results/transition_rich_lista_dense_p256_hardinit_table123_20260430](/home/mila/l/lia/skae/results/transition_rich_lista_dense_p256_hardinit_table123_20260430).
The new `lista_dense_signsplit_p256_hardinit_basin_partition` task table has
`255` rows (`17` systems x `15` seeds) and matches the current LISTA Table
1--3 training settings: hard-init sampling, `d_z=256`, `sequence_length=8`,
`200000` optimization steps, sparsity coefficient `0.003`, and a dense Koopman
matrix with no soft-block or block-diagonal K regularizer. Training array
`9423749` completed `255/255` tasks, and collector/evaluation jobs
`9423750`--`9423760` completed with exit `0:0`. The artifacts contain `255`
forecasting rows, `6,885` interpretability rows, `1,980` self-routed rows,
and `191,400` support-refresh rows with `0` failures. The table/figure
builders were rerun on compute allocation `9431949`, and the manuscript
fragments now show the plain LISTA row.

May 1 support-refresh redesign completion note: the support-refresh display has
been rebuilt with period groups rather than a `Fallback` column. Period means the
number of Koopman rollout steps between decode/re-encode support refreshes
after controlled target-basin entry. The missing MLP-control refresh rows
completed under
[results/periodic_support_refresh_mlp_controls_seed0to14_20260430](/home/mila/l/lia/skae/results/periodic_support_refresh_mlp_controls_seed0to14_20260430):
Sparse MLP-BD shards `9423980`--`9423982`, Sparse MLP shards
`9423983`--`9423985`, Dense MLP shards `9423986`--`9423988`, merge `9423991`.
All shards completed with exit `0:0` and shard-level `failure_count=0`.
The first SLURM merge `9423991` ran out of memory and dependent table job
`9423996` was cancelled; clean merge job `9432117` was rerun with `64G`,
completed with exit `0:0`, and wrote `572,654` data rows with `0` failures.
The current support-refresh fragment displays LISTA, LISTA-SB, LISTA-BD, Sparse
MLP-BD, Sparse MLP, and Dense MLP.

May 2 compact control-table rebuild note: the Table 2 partition-control
summary and compact TeX fragment have been rebuilt with the matched
`d_z=256` LISTA-SB row, and the paper-facing compact display now reports
`H100` only. The combined input is
[results/transition_rich_table2_controls_p256_compact_20260502/self_routed_controls/self_routed_forecasting_rows.csv](/home/mila/l/lia/skae/results/transition_rich_table2_controls_p256_compact_20260502/self_routed_controls/self_routed_forecasting_rows.csv);
it keeps the non-LISTA-SB rows from the original control packet and replaces
the p64 LISTA-SB rows with the completed p256 follow-up
(`9429041`--`9429043 -> 9429044`). Final rebuild allocation `9437124`
rewrote
[table2_partition_controls_h100_censored_seed15_summary.csv](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table2_partition_controls_h100_censored_seed15_summary.csv)
and
[table2_partition_controls_compact.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table2_partition_controls_compact.tex).
Use the compact table as matched-dimension appendix/audit evidence. Within
each model, the selectors are ordered Oracle basin labels, Support family,
Latent clusters, then Random matched.

April 30 statistical-protocol documentation note: the manuscript appendix and
[EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md) now give the
repeatable testing procedure. For each table cell, construct paired deltas
within each system, run one-sided paired Wilcoxon tests inside systems,
Holm-correct the per-system p-values across the eligible systems for that
cell, and report `[K/N]` as the count of Holm-cleared systems. The point
estimate and the significance count are deliberately separate: IQM summarizes
finite observed values, while `[K/N]` summarizes within-system reproducibility.
For Table 2 only, seed slots `0`--`14` are retained with censored deltas for
invalid H-step routed/global comparisons; this is an evaluation convention for
evaluable versus unevaluable forecasts, not a causal diagnosis of learning
failure.

May 1 higher-dimensional hard-system correction note: the focused sparse-KAE
forecasting redo for fixed 8-basin CLV, Hopfield `N=16/P=16`, and Kuramoto
identical `N=16` is complete and corrected. The purpose was to revisit the old
hard-system forecasting evidence under the shifted sparse Koopman-autoencoder
narrative with a clean tanh Dense MLP baseline instead of the old ReLU
zero-sparsity control. Training array `9412218`, collector `9412219`, and
comparisons `9412220`--`9412222` completed with exit `0:0` under
[results/hard_system_sparse_kae_redo_p1024_seq8_100k_halflr_sc6em3_tanh_dense_20260429](/home/mila/l/lia/skae/results/hard_system_sparse_kae_redo_p1024_seq8_100k_halflr_sc6em3_tanh_dense_20260429).
The initial collector output under-counted Hopfield because it selected latest
runs per `n_16/seed` wrapper; the fixed collector and allocation `9432839`
refresh now yield the expected `270` rows. It uses `d_z=1024`,
`sequence_length=8`, `100k` steps, half learning rates, `15` seeds, and six
recipes: Dense MLP, Sparse MLP, Sparse MLP-BD, LISTA, LISTA-BD, and LISTA-SB.
The result is negative for the sparse/LISTA recipes: tanh Dense MLP wins all
CLV/Hopfield/Kuramoto system-median comparisons at `H100`, `H500`, and
`H1000`.

Use this file as the first stop for drafting. Use
[EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md) for detailed
experiment logs, [PAPER_TRACK_STATUS.md](/home/mila/l/lia/skae/docs/PAPER_TRACK_STATUS.md)
for high-level claim status and wrap-up priorities, and
[SUPPORT_OBJECT_GLOSSARY.md](/home/mila/l/lia/skae/docs/SUPPORT_OBJECT_GLOSSARY.md)
for exact support-object definitions.

## Recommended figures and tables

The experiments section should use figures and tables in the same order as the
reader's current chain: multibasin problem, forecasting competence, supports as
basin indicators, functional support diagnostics, and periodic support refresh.
The main text should avoid
queue chronology and instead use each display item to close one link in that
chain.

### Main-text display order

1. **Figure 1: Support maps agree with basin structure.**
   Purpose: show that the model produces discrete support objects that align
   spatially with benchmark basins when labels are used only for evaluation.
   The current main-text Figure 1 is the generated composite
   [fig_benchmark_support_dysts_composite.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_benchmark_support_dysts_composite.pdf).
   Top row: `gated_local_linear`, `claude:transition_routes_4`,
   `claude:cal_square_4`, and `claude:cal_high_cross_3` with true
   vector-field streamlines, evaluation basin maps, and LISTA `topk:8`
   support families mapped post hoc to their dominant evaluation basin. The
   polished display uses the concise row label
   `Support-basin alignment`, removes in-panel agreement annotations, and
   keeps the support/basin match-mismatch cue outside the panels with a
   slightly larger legend font. Bottom row:
   Chua, Dadras, Shimizu-Morioka, and Lu-Chen-Cheng H5000 Dysts `dt x30`
   phase portraits from the best seed-`0` primary model in the existing
   all-model manifest, with a slight Dadras zoom. Caption rule: explicitly
   state that basin labels are evaluation overlays, not training inputs, and
   that Dysts is only a forecasting benchmark.
   Candidate-screen note:
   [fig_benchmark_support_grid_agreement_candidates.json](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_benchmark_support_grid_agreement_candidates.json)
   scores the same visual grid-agreement metric for fixed-`17` systems with
   explicit basin maps or catalog attractor-center conventions.
   `claude:cal_high_cross_3` (`0.902`) is now included as the fourth panel
   because it was the strongest unused screen candidate.
   Companion active-index display:
   [support_family_index_codebook_claude_cal_asymmetric_3.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/support_family_codebooks_retained15/support_family_index_codebook_claude_cal_asymmetric_3.pdf)
   shows the actual latent coordinate indices activated by `d_z=256` LISTA-BD
   `topk:8` support-family prototypes on the three-basin Asymmetric wells
   system. Use it when the reader needs to see that the support-family labels
   correspond to concrete coordinate sets, not just colors on a basin map. Row
   labels and tick opacity encode within-basin family coverage; thin tick
   position on the shared `0..255` axis remains the binary active-coordinate
   index. Do not reintroduce the removed printed active-index list column.
   Screening batch:
   [support_family_codebooks_retained15](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/support_family_codebooks_retained15)
   contains one LISTA/LISTA-BD active-index codebook per retained controlled
   multibasin system plus a contact sheet and manifest. Use it for appendix
   selection or for swapping the main companion subfigure; note that the
   Asymmetric wells triplet has been overwritten by the one-panel LISTA-BD
   main-text figure. Do not include all `15` in the main text unless the
   narrative explicitly needs benchmark-wide visual coverage.
   Exploratory \(F_{\rm abs}\) screen:
   [support_family_codebooks_retained15_fabs](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/support_family_codebooks_retained15_fabs)
   repeats the retained-`15` LISTA/LISTA-BD codebooks with support rule
   `absolute:0.001`. Use it only as a diagnostic view of absolute-threshold
   support structure unless the manuscript needs to foreground \(F_{\rm abs}\)
   rather than the cleaner fixed-size \(F_{\rm top8}\) routing object.
   Deep-slice \(F_{\rm abs}\) screen:
   [support_family_codebooks_retained15_fabs_deep](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/support_family_codebooks_retained15_fabs_deep)
   repeats the retained-`15` LISTA/LISTA-BD codebooks on generated observation
   states restricted to the global top-quartile basin-margin deep slice
   (`128 x 128`, eval seed `42`). This is the visual diagnostic closest to the
   main \(F_{\rm abs}\) Table 1 slice, with the caveat that the global criterion
   can leave some systems with only one represented deep basin.
   Additional p256 visual candidates:
   [fig_support_barcode_map_p256.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_support_barcode_map_p256.pdf)
   gives a compact state-space support-family map plus a shared latent
   coordinate barcode for p256 LISTA on `gated_local_linear`, and
   [fig_basin_support_family_alluvial_p256_deep.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_basin_support_family_alluvial_p256_deep.pdf)
   gives a clean deep-slice basin-to-exact-support-to-family flow for p256
   LISTA on `claude:cal_hexagon_6` seed `6`, with exact supports shown as
   unlabeled middle bars and a `6`-basin-to-`6`-family match. The polished
   Image-1-style deep-slice alluvial is
   [fig_basin_support_family_alluvial_p256_gated_deep_unbold_headers.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_basin_support_family_alluvial_p256_gated_deep_unbold_headers.pdf):
   p256 LISTA on `gated_local_linear`, seed `0`, `393` exact supports, and a
   `3`-basin-to-`3`-family match. The all-state sensitivity alluvial is
   [fig_basin_support_family_alluvial_p256_gated_all_j032.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_basin_support_family_alluvial_p256_gated_all_j032.pdf):
   p256 LISTA on `gated_local_linear`, seed `7`, `S_abs=10^{-3}`, family
   Jaccard threshold `0.32`, `5635` exact supports, and a
   `3`-basin-to-`3`-family match.
   Treat these as alternatives or appendix companions to the current composite
   and active-index codebook, not as p64 carryovers.

2. **Table 1: Forecasting on both benchmark families.**
   Purpose: quantify the predictive baseline for the paper before the
   identifiability diagnostics. The current Table 1 source is
   [table1_forecasting_multibasin_dysts.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table1_forecasting_multibasin_dysts.tex).
   It combines all displayed multibasin horizons (`H100/H500/H1000`) and three
   representative Dysts `dt x30` horizons (`H100/H2000/H4000`) in one table. Keep
   forecasting separate from support diagnostics so reviewers see that support
   identifiability is not being inferred from forecast MSE alone. Multibasin
   forecasting suppresses the all-`15/15` count and uses `\ast` superscripts;
   Dysts superscripts use exact system-sign tests with Holm correction across
   the displayed non-Dense Dysts cells, so a star means the row improves on
   Dense MLP for all retained `10/10` Dysts systems after correction. Signed-rank
   p-values are audit sidecars, not the main-table annotation. The current
   source remains active because the May 14 `normdec_rollout` replacement
   landed only on Dysts and the controlled launchers failed before training;
   do not mix partial normalized-decoder rows into this table.
   The standalone baseline companion table
   [table_standalone_state_space_baselines.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table_standalone_state_space_baselines.tex)
   now includes label-free local polynomial EDMD and local RBF-EDMD. Use it to
   answer the reviewer-facing multi-model/local-Koopman baseline question
   without changing the main support-diagnostic table.

3. **Table 2: Support diagnostics and periodic support refresh.**
   Purpose: make the main interpretability evidence a compact two-subtable
   display. Subtable 2a is the per-basin deep-slice support diagnostic:
   \(H(B\mid F_{\rm abs})\), \(\overline{|F_{\rm abs}|}\), and wrong-support
   ratios at `h=1` and `h=20`. Subtable 2b is the controlled-transfer
   \(F_{\rm top8}\) support-family refresh result. The current support-diagnostics
   source is
   [table2_support_diagnostics_per_basin_deep.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table2_support_diagnostics_per_basin_deep.tex),
   and the new refresh source is
   [table2_support_refresh_controlled_transfer_family.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table2_support_refresh_controlled_transfer_family.tex).
   The support-family alignment and wrong-support columns use the evaluation-only
   per-basin deep-state slice and show single `\ast` significance superscripts
   instead of `[K/N]` counts. The May 14 `normdec_encoded` queue remains the
   conceptual sparsity-target ablation because these diagnostics are computed
   from encoded latents, not rollout latents, but it still needs controlled
   support diagnostics before changing the training objective. The refresh
   subtable should report stale
   pre-refresh \(\to\) refreshed \(F_{\rm top8}\) target-family rates over the
   `1/5/10/20` cadence grid, plus the no-transfer false-target control.
   Forecasting columns do not belong in this table.
   If the table needs a non-statistical companion, use
   [fig_support_family_value_diagram.svg](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_support_family_value_diagram.svg)
   to show the practical role of \(F_{\rm abs}\): many exact masks become one
   stable support object, and that object can be inspected and reused as a
   route/cache key.

4. **Appendix-only support-routed local predictors.**
   Purpose: preserve the local-support-predictor evidence without making it a
   main-text claim. The moved material now lives in
   [support_conditioned_predictors.tex](/home/mila/l/lia/skae/docs/appendix/support_conditioned_predictors.tex).
   It can be used as a diagnostic that support families can key post-hoc local
   maps in some settings, but the main paper should not cite it or rely on it
   for the NeurIPS narrative.

5. **Figure 2: Forecasting horizon trends.**
   Purpose: visually support Table 1. Use the retained multibasin horizon curve
   and the Dysts `dt x30` horizon curve side by side, with the caption
   explaining seed-bootstrap uncertainty and the log-MSE scale.

### Appendix or supplement displays

- **Appendix Table A1: Support object definitions and sensitivity.**
  Include `absolute:0.001`, `relative:0.1`, top-`8`, exact support, support
  family, and dominant group. This prevents the main text from collapsing
  thresholded identifiability, local-law diagnostics, and deployment routing
  into one overloaded "support" claim.
- **Appendix Table A2: Fixed-`17` benchmark system inventory.**
  List system name, basin count used for evaluation, basin-label source,
  whether it is clean away from basin boundaries, transition-rich, boundary-stress, or
  mechanism/falsification oriented, and which main figure/table uses it.
- **Appendix Table A3: Global-deep comparison.**
  Since the manuscript Table 1 now uses the per-basin deep-state slice for the
  support diagnostics, keep the older global-deep Table 1 support columns as an
  appendix comparison if space permits. The current-roster per-basin rerun
  completed as jobs `9477837`--`9477845` with `0` failures and wrote
  `interpretability_per_basin_deep_current_table1_pass0/` outputs in the three
  current Table 1 source packets. The per-basin replacement Table 1 gives
  \(H(B\mid F_{\rm abs})=0.130/0.156/0.153\) for LISTA/LISTA-BD/LISTA-SB,
  wrong-support ratios `2.15e4/2.20e4/2.53e4` at `h=1`, and
  `93.6/84.0/129` at `h=20`; Dense MLP remains poor with
  \(H(B\mid F_{\rm abs})=1.28\) and one support family.
- **Appendix Table A4: Matched hard-init controls.**
  Summarize the near-boundary-sampling sparse MLP, block-diagonal sparse MLP,
  and no-shrink dense MLP controls. This is the guardrail that prevents a
  LISTA-only claim.
- **Appendix Table A5: Centered local-law, zero-intercept, true-geometry, and
  random-partition diagnostics.**
  Use this to show the calibrated mechanism story: centered local laws are
  useful, zero-intercept local laws fail, true-geometry recovery is mixed, and
  support families are not merely random partitions.
- **Appendix Figure A1: Operator geometry diagnostic.**
  Only include in the main text if the paper claims geometric recovery. With
  the current mixed result, this is better as a falsification/limitation figure
  comparing learned local operators, true Jacobians/eigendirections, basin
  baselines, and random partitions.
- **Appendix Table A6: Fair `200k`, hard-system, and historical-provenance
  benchmark summary.**
  Keep the repaired `200k` cross-system benchmark, Kuramoto/Hopfield
  hard-system results, dense recipe-selection provenance, and `50k` audit in
  one compact appendix area rather than spreading them through the main
  experiments narrative. Add the corrected `d_z=1024` tanh-dense-baseline
  higher-dimensional redo here as a dense-baseline hard-system stress test
  rather than as a main evidence pillar.
- **Appendix Dysts IQM-over-IQM sensitivity.**
  Use
  [table_dysts_dt30_iqm_over_iqm.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table_dysts_dt30_iqm_over_iqm.tex)
  and
  [fig_dysts_dt30_iqm_over_iqm_horizon.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_dysts_dt30_iqm_over_iqm_horizon.pdf)
  when coauthors ask how the Dysts model ordering changes under a robust
  cross-system statistic. The refreshed table shows repaired Sparse MLP-BD
  best at every horizon after seed IQM within systems and IQM across retained
  systems. Keep it
  appendix-only because the main Dysts table intentionally reports the
  arithmetic mean across systems as the average-case MSE estimand.
- **Appendix planned benchmark extensions.**
  Use
  [benchmark_extensions.tex](/home/mila/l/lia/skae/docs/appendix/benchmark_extensions.tex)
  as the protocol source for the two proposed additions: spatialized
  multibasin reaction-diffusion fields and controlled ManiSkill insertion.
  This appendix is a planning object and should not be cited as evidence until
  the corresponding results land. The companion progress docs now record the
  first smoke, tuning, dense-control, and activation-corrected jobs. The PDE
  pilot should not be promoted from the flattened run because dense forecasts
  better, even though LISTA has cleaner compact support alignment. The
  ManiSkill pilot is the stronger next candidate. Array `9725895` completed
  the matched `50k`-step CPU-only retry over dense, dense+L1 sparse-MLP, and
  LISTA settings for seeds `0`--`2`, but those dense rows used ReLU MLP blocks,
  not tanh. The retry fixes the old undertraining failure but remains a
  ReLU-control forecast/support tradeoff rather than a clean dense-tanh sparse
  forecasting result. Focused long-horizon eval `9729181` supports focusing on
  LISTA `alpha=0.03,sparsity=0.01` through H175 relative to the old dense-ReLU
  control, but H300--H500 require a newly generated longer rollout packet. The
  activation-corrected `5k` pilot is the first true dense-tanh comparison:
  sparse KAE rows beat `dense_tanh_sp0` over seeds `0`--`2` under `last.pt`
  and best periodic cadences `{1,2,5,10,20,50,100}`. The direct comparison
  first had sparse-MLP sparsity `0.01` best, and the June 9 optimizer-fairness
  run confirms that applying the LISTA-favored optimizer to the controls keeps
  sparse-MLP ReLU on top (`sparsity=0.003` H10--H50 `0.001837`, versus tuned
  dense `0.002023` and best LISTA `0.001881`). This is promising sparse-KAE
  forecasting evidence, but it still needs a fixed checkpoint rule, validated
  perturbation-balanced state-only outcome/contact labels, and support
  diagnostics before any paper-facing result. The current display candidate
  from the old `50k` packet is the fine H10--H125
  forecasting graph at horizons
  `10,20,30,40,50,60,70,80,90,100,110,120,125`,
  [fig_maniskill_h125_forecasting_fine.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_maniskill_h125_forecasting_fine.pdf),
  with CSV/manifest provenance under
  [results/maniskill_h125_forecast_fine_20260603](/home/mila/l/lia/skae/results/maniskill_h125_forecast_fine_20260603).
  This graph should be described as autonomous latent rollout, not periodic
  re-encoding: the model encodes the initial state once, rolls forward in
  latent space under the action sequence, and decodes predictions. No
  periodic re-encoding periods were used for the ManiSkill display.
  A matched best-periodic display is now also generated:
  [fig_maniskill_h125_forecasting_best_periodic.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_maniskill_h125_forecasting_best_periodic.pdf),
  with CSV/manifest provenance under
  [results/maniskill_h125_forecast_best_periodic_20260603](/home/mila/l/lia/skae/results/maniskill_h125_forecast_best_periodic_20260603).
  The periodic cadences are `{10,20,50,100}` and the best cadence is selected
  per model/seed/horizon. This is deployment-style re-encoding from decoded
  predictions rather than future-state leakage. It removes the autonomous
  H100--H125 blow-up almost entirely, but it is a reset-stabilized forecasting
  protocol; use it as an auxiliary stability diagnostic unless the paper
  explicitly wants a periodic-refresh result. The best H100/H125 mean state
  MSEs are sparse-MLP `0.00186/0.00205`, dense-ReLU `0.00188/0.00211`,
  LISTA `alpha=0.03,sparsity=0.003` `0.00207/0.00231`, and LISTA
  `alpha=0.03,sparsity=0.01` `0.00207/0.00237`.
  The H220 extension is also complete:
  [fig_maniskill_h220_forecasting_best_periodic.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_maniskill_h220_forecasting_best_periodic.pdf),
  with CSV/manifest provenance under
  [results/maniskill_h220_forecast_best_periodic_20260603](/home/mila/l/lia/skae/results/maniskill_h220_forecast_best_periodic_20260603).
  For current reporting, H175 is the sensible maximum horizon because H125,
  H150, and H175 all keep `15` eligible test episodes per seed, while H200 and
  H220 drop to `9` and `5`. Best-periodic H175 mean state MSEs are dense-ReLU
  `0.00301`, sparse-MLP `0.00304`, LISTA `alpha=0.03,sparsity=0.003`
  `0.00399`, and LISTA `alpha=0.03,sparsity=0.01` `0.00394`.

## 1. Do sparse supports align with basins when labels are used only for evaluation?

Paper role:
- This is the first main result. It tests whether the learned support object is
  an interpretable indicator of basin membership after fully label-free
  training.
- It is deliberately a static test. It asks whether the support says where the
  state is, not whether the support already proves which latent coordinates or
  local operator should advance the state.

Current result:
- On the retained `15`-system multibasin benchmark, support families on states
  far from basin boundaries have low conditional basin entropy for every
  sparse-latent row: `H(B|F_abs)=0.0391` for LISTA, `0.0456` for LISTA-BD, and
  `0.0442` for LISTA-SB, versus `0.765` for the dense no-shrink MLP baseline.
- The same retained table shows that wrong-support interventions are strongly
  functional for sparse models: wrong-support/base ratios at `h=1` are
  `2.10e4`--`7.08e4` for sparse rows versus `20.0` for Dense MLP, and the
  effect persists at `h=20`.
- On the same retained-`15` table, every sparse-latent row has lower
  arithmetic-mean raw MSE than Dense MLP at `H100`, `H500`, and `H1000`.
  The displayed `H100/H1000` values are `0.0407/0.166` for LISTA,
  `0.0411/0.135` for LISTA-BD, `0.0387/0.130` for LISTA-SB, and `0.830/2.93`
  for Dense MLP. The main retained-benchmark horizon display uses
  fixed-system, log-relative seed-bootstrap `95%` bands, matching the Dysts
  horizon display.
- The causal claim should be "induced sparse latent structure supports
  support objects that agree with basin labels," not "LISTA alone is
  responsible." The
  implementation makes this a latent-magnitude penalty claim: `GenericKM` forms
  the sparsity term as the mean latent `L1` norm and adds
  `SPARSITY_COEFF * sparsity_loss` to the objective
  ([model.py](/home/mila/l/lia/skae/skae/model.py:1173),
  [model.py](/home/mila/l/lia/skae/skae/model.py:1278)). The fixed-`17`
  manifest makes the sparse-versus-no-sparsity comparison explicit: sparse MLP
  controls use `generic_sparse` with `sparsity_coeff=0.003`, while no-shrink
  controls use `generic_no_shrink` with `sparsity_coeff=0.0`
  ([transition_rich_basin_partition_manifest.py](/home/mila/l/lia/skae/skae/benchmarks/transition_rich_basin_partition_manifest.py:1177),
  [transition_rich_basin_partition_manifest.py](/home/mila/l/lia/skae/skae/benchmarks/transition_rich_basin_partition_manifest.py:1251)).
- Sparse MLP controls are themselves competitive: under the standard-sampling
  control row in the locked final packet, the sparse MLP already has supports
  that do not mix basin labels on the states-far-from-boundaries slice and
  competitive forecasting (`H(B|S_abs)=0.0000`,
  `H(S_abs|B)=0.2449`, `U_exact=0.9772`, `H(B|F_abs)=0.0000`, and
  `H100/H500/H1000 = 0.0297 / 0.0614 / 0.0608`)
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:12)).
  The dense hard-init sparse-latent Koopman finalist gives the cleanest exact
  support-compression result (`H(S_abs|B)=0.0543`, `U_exact=0.9923`), but that row
  is mixed-regime because the promoted LISTA roots use hard-init oversampling
  while the MLP rows in the locked packet use standard sampling
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:6),
  [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:14)).
- The matched near-boundary-sampling control packet is an important causal
  guardrail. Under the same hard-init sampling regime, sparse MLP controls are
  almost tied on the selected deep slice (`H(B|S_abs)=0.0000`,
  `H(S_abs|B)=0.2068`, `U_exact ~= 0.98`) and forecast strongly
  (`H100/H500/H1000 = 0.0082 / 0.0260 / 0.0273` for plain sparse MLP and
  `0.0082 / 0.0252 / 0.0264` for block-diagonal sparse MLP), while the
  no-shrink dense MLP control is much worse functionally
  (`0.5704 / 2.6733 / 3.8044`)
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1/transition_rich_final_comparison.md:10)).

Interpretation:
- Away from basin boundaries, sparse supports often identify basin structure.
  The direct evidence is that the locked final comparison is explicitly on the
  `absolute:0.001` / states-far-from-boundaries support slice and all sparse
  rows in that table have `H(B|S_abs)=0.0000`, meaning that an absolute-threshold support object does
  not mix basin labels on that slice
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:3),
  [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:12)).
  The same rows have low fragmentation or high dominant-support mass
  (`H(S_abs|B)=0.0543`, `U_exact=0.9923` for the dense sparse-latent finalist;
  `H(S_abs|B)=0.2449`, `U_exact=0.9772` for the standard sparse MLP control), and
  support families saturate (`H(B|F_abs)=0.0000`). The matched hard-init sparse MLP
  controls replicate the support-label agreement read under the oversampled regime
  (`H(B|S_abs)=0.0000`, `H(S_abs|B)=0.2068`, `U_exact ~= 0.98`, `H(B|F_abs)=0.0000`)
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1/transition_rich_final_comparison.md:3),
  [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1/transition_rich_final_comparison.md:10)).
- Exact support identity is not a global invariant. Near separatrices and weak
  coordinates, support families or fixed-size top-`k` supports are more robust
  than a single absolute threshold.
- This agreement result is necessary but insufficient. It says that supports
  carry information about basin membership, but it does not say that the active
  support selects useful columns of the learned transition or a useful centered
  local law. That is the role of the non-oracle routing experiments.
- The no-shrink control is a useful negative/guardrail mainly on functional
  robustness, not on every entropy metric: it can still show thresholded
  supports that do not mix basin labels on the selected states-far-from-boundaries
  slice, but it is much worse in
  hard-init forecasting and freeze robustness. Therefore the interpretation
  should emphasize induced sparsity plus functional stability, not only one
  entropy column.

Project implication:
- The paper should write this as support agreement with basin labels, not
  basin-block alignment and not oracle clustering. Basin labels are only used
  to score support purity, fragmentation, and dominant-support mass.
- Standard-sampling and near-boundary-sampling rows should stay separate in
  tables. The hard-init controls support induced sparsity more than a
  LISTA-specific architecture claim.
- Table 1 should therefore have one row family for sparse-latent Koopman, one
  for sparse MLP Koopman, and one for no-shrink / no-sparsity MLP controls, with
  standard-sampling and hard-init rows visibly separated. The main prose should
  say that latent sparsity can produce support objects that agree with basin
  labels, while the strongest exact-support compression belongs to the dense
  sparse-latent finalist and the strongest architecture-isolating hard-init
  evidence belongs to the broader induced-sparsity claim.

Next drafting step:
- Build the main interpretability figure: true basin labels, learned
  support/support-family labels, support entropy, and a support-versus-basin
  confusion matrix on representative systems.

Primary artifacts:
- [FIXED17_LISTA_RESULTS_INDEX.md](/home/mila/l/lia/skae/docs/FIXED17_LISTA_RESULTS_INDEX.md)
- [transition_rich_basin_support_metric_definitions.md](/home/mila/l/lia/skae/docs/transition_rich_basin_support_metric_definitions.md)
- [results/transition_rich_hardinit_mlp_controls_seed10_20260416](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416)
- [EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md), fixed-`17`
  support-label agreement entries

## 2. Do those same supports select useful local predictors without oracle basin labels?

Paper role:
- This is the functional test of the support-as-dynamical-selector claim. The first
  experiment asks whether learned supports align with basin labels when labels
  are used only for evaluation. This second experiment asks whether the same
  model-produced support objects do useful dynamical work: can a support
  inferred from the current latent state choose a better local predictor than
  applying one learned global latent transition everywhere?
- In the manuscript logic up to the experiments section, this closes the
  second arrow in the chain
  `sparse latent code -> support agrees with basin labels -> support-routed local
  prediction`. Without this result, support alignment could be only a static
  clustering observation. With it, the supports become evidence for a
  label-free routing variable that selects local Koopman-style prediction.
- This is the counterfactual the reader needs in mind: a support can label a
  basin well and still fail dynamically if the coordinates that mark basin
  membership are not the coordinates whose transition columns carry the local
  dynamics.

What is being tested:
- Freeze the trained model and infer a support object from the latent state;
  do not use true basin labels, known basin counts, or trajectory-to-basin
  assignments for routing.
- Use the inferred support in two distinct prediction diagnostics:
  support-gated prediction, where the same learned global transition is
  applied after masking the centered latent input by the current support, and
  centered local-law prediction, where the current support or support family
  selects a post-hoc centered local operator fitted after model training.
- Evaluate on disjoint forecast trajectories and compare routed prediction
  against the model's learned global transition. Ratios below `1` and high win
  rates mean the support selected a useful local predictor.
- Foreground fixed-size top-`8` support families because every state first
  gets a threshold-free route key and nearby exact keys are then merged without
  basin labels into a more robust deployment-facing selector.

Terminology clarification:
- "Routing" should not be written as if the trained model contains separate
  learned Koopman matrices per support. The base trained Koopman autoencoder
  has one learned global transition.
- Support-gated prediction is support-conditioned masking of the global
  transition, not switching among independently trained local Koopman matrices.
  Algebraically, the support selects which centered latent coordinates drive
  the transition, so it is closest to testing whether the support selects useful
  columns or subspaces of the learned global transition.
- Centered local-law prediction is a literal local-law routing diagnostic. It
  freezes the trained encoder and decoder, fits centered local slopes after
  training, and then uses the current model-produced support or family as the
  key that selects which local slope to apply.
- These local laws are not used during model training. They are post-hoc
  inference/evaluation objects. In the full non-oracle rollout packet, however,
  they are actually used to produce forecasts in the centered support-local and
  centered family-local modes; they are not merely scored passively.

Centered local-law mechanism packet:
- This is a one-step latent-dynamics diagnostic, not an autonomous forecasting
  study. For each trained checkpoint, generate benchmark trajectories, encode
  each state, and form held-out one-step latent pairs `(z_t, z_{t+1})`.
- Label each transition by the current basin, exact support, or support family.
  Basin labels are included as an evaluation/control partition only, not as a
  training or deployment router.
- Split transitions into fit/test subsets. For each partition class `c` with
  enough transitions, fit a ridge-regularized centered local law
  `z_{t+1} ~= center_c + A_c (z_t - center_c)`.
- Compare held-out one-step latent MSE against the learned global transition, one
  global centered latent slope, count-matched random partitions, and sometimes
  latent-k-means controls. Also evaluate support-gated versions of the learned
  transition using support prototypes or block-union masks.
- This packet tests mechanism: whether support/family partitions behave like
  local regime labels for one-step latent dynamics, and whether masking the
  learned global transition by support exposes useful local subspaces. It does
  not test full autonomous forecasting.

Full non-oracle rollout packet:
- This is the deployment-facing forecasting diagnostic. For each trained
  checkpoint, use one generated trajectory set to fit support/family centers
  and post-hoc local operators, and a disjoint trajectory set for held-out
  autonomous forecasts.
- Start each forecast from only the initial observation: encode `x_0` to
  `z_0`, then roll forward to horizons `H100`, `H500`, and `H1000`.
- In the global-transition baseline, forecast with the base trained transition.
- In support-gated prediction, infer the support from the current predicted
  latent state, then apply the learned global transition to the centered masked
  latent state. If the required support center is unavailable, fall back to the
  global transition.
- In centered local-law prediction, infer the exact support from the current
  predicted latent state, select the post-hoc local slope for that support, and
  apply it around the support center. In the family-level version, first map
  the exact support to a support family and then select the family-level local
  law. Missing routes fall back to the global transition.
- The route is recomputed from the model's own predicted latent state during
  rollout, not from the true future state and not from an oracle basin label.
  This is the packet that tests whether the supports can actually be used for
  inference-time forecasting.

Why this matters:
- This packet is now appendix-only mechanism context. It tests whether learned
  supports can key post-hoc local prediction behavior, but the main paper no
  longer relies on support-conditioned rollouts.
- The main-text functional-use evidence is the wrong-support intervention plus
  the controlled-transfer support-refresh audit. Local-law routing remains a
  useful stress test for whether support objects can be reused beyond
  identification.

Current result:
- The non-oracle self-routed forecasting packet is complete with `510/510`
  runs, `24,600` rows, and `0` failures, but later routed-forecasting sweeps
  show that post-hoc local maps are not robust enough to carry the main
  forecasting claim.
- The old periodic support-gated refresh rows are superseded for the main text
  by the completed controlled-transfer global-\(K\) support-refresh audit.
  The new main read is \(F_{\rm top8}\) support-family switching: sparse-family
  after-reencoding target rates are `0.994`--`1.000` across periods
  `1/5/10/20`, while no-transfer false target-family rates remain
  `0.009`--`0.019`. Display the table as stale pre-refresh family \(\to\)
  refreshed family so the support switch is visible.

Interpretation:
- Support families are the stable paper-facing object. Exact top-`8` supports
  are too fragmented for the refresh table, even though they remain useful
  diagnostics.
- The periodic-refresh claim should be about representation switching under
  controlled basin transfer, not about support-conditioned forecasting or
  local-\(K_c\) rollouts.

Project implication:
- Keep local support-gated and local-law predictors in
  `docs/appendix/support_conditioned_predictors.tex`. Do not use them as a
  main-text claim.
- Main Table 2b should use the new controlled-transfer \(F_{\rm top8}\)
  support-family refresh subtable and explicitly state that only the learned
  global \(K\) is used between refresh events.

Next drafting step:
- Keep the appendix local-predictor section concise, and keep the main text
  focused on basin-support identifiability, active-coordinate interventions,
  global-\(K\) support refresh, and forecasting.
- Landed router audit: the matching \(F_{\rm abs}\) stage-2 replay used
  `support_definition=absolute:0.001`, `J=0.40`, period `5`,
  `reroute_each_step`, `50000` steps, one GPU per worker on `long`, and
  `DEVICE=cuda`, matching the existing \(F_{\rm top8}\) fixed-setting protocol
  except for the support selector. The result is failed/negative:
  controlled multibasin is not aggregate-usable because `149/150` workers
  OOMed, and the partial Dysts aggregate (`96/100` workers) has coverage
  `0.2345`, fallback `0.7655`, `0/10` wins versus best-periodic at every
  horizon, and routed/best-periodic ratios `6.39e3` at `H100`, `5.37e27` at
  `H500`, `5.22e25` at `H1000`, and `6.74e4` at `H5000`. Use this as a
  negative \(F_{\rm abs}\) stage-2 router audit, not as a new main-table
  forecasting row.
- Reviewer add-ons for this section: the plain p256 autonomous
  oracle-vs-learned local-\(K\) comparison is still incomplete because shard
  `9478599` is running and merge `9478603` is held. The one-step
  regime-discovery packet has landed: `9478633`--`9478647 -> 9478648` wrote
  `6,885` rows with `0` failures. It shows that standard unsupervised
  partitions over dense/sparse latent values and binary supports become strong
  when given the evaluation-only basin count, while learned support-family
  local-\(K\) is near global on this one-step metric. This should be framed as
  a caution/appendix audit, not as main positive routing evidence.

Primary artifacts:
- [results/transition_rich_self_routed_forecasting_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420)
- [results/periodic_support_refresh_fixed17_seed0_20260425](/home/mila/l/lia/skae/results/periodic_support_refresh_fixed17_seed0_20260425)
- [controlled_transfer_switching_experiment_20260423.md](/home/mila/l/lia/skae/docs/planning/controlled_transfer_switching_experiment_20260423.md)

## 3. Do sparse-latent Koopman models remain competitive for long-horizon forecasting?

Paper role:
- This is the external forecasting stress test. It shows the sparse-latent
  models are not only interpretable on the controlled multibasin benchmark.
- A supplemental out-of-generator multistability packet is now partly landed as
  a separate generality check rather than as part of the fixed retained
  controlled benchmark. It covers `claude:toggle_switch_3gene`,
  `claude:bistable_reactor`, `claude:fitzhugh_nagumo_3eq`, and
  `claude:buckled_beam` with the current p256 LISTA recipe. Training/collect,
  interpretability, and one-step regime discovery landed cleanly; autonomous
  oracle/local-K is still running. Use the landed part to support stable
  forecasting and deep-slice support purity, not local-K routing generality.

Current result:
- *2026-04-29 refresh:* the `seq_len=10`, up-to-`15`-seed, `12`-system
  `H{5K..60K}` Dysts rerun has been collected under
  [results/dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428/collect](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428/collect).
  The collector has `894/900` requested rows. The six missing or invalid rows
  are all LISTA-BD (`dysts:DequanLi` seed `3` and `dysts:LorenzCoupled` seeds
  `1,2,4,8,9`), and within-system paired tests use common completed seeds.
- Dense-transition LISTA had the best old trimmed aggregate at every reported
  horizon: `0.198`, `1.02`, `2.88`, `3.81`, `4.30`, `4.65`, and `4.93` for
  `H5000` through `H60000`. However, the within-system Holm-corrected counts
  are weak (`2/12`, `2/12`, `0/12`, `0/12`, `2/12`, `0/12`, `0/12`), so this
  should be described as aggregate competitiveness rather than broad
  systemwise dominance.
- LISTA-BD is not a positive Dysts result in this rerun. Its prior trimmed aggregate is
  dominated by catastrophic finite errors (`7.9e29` at every displayed
  horizon) and it clears `0/12` systems against Dense MLP. Sparse MLP and
  Sparse MLP-BD stay near the Dense MLP baseline and clear at most `1/12`
  systems at the longest horizons.
- *2026-05-03 aggregation-corrected `dt x30` sensitivity:* the full `12`-system coarser-step packet
  completed under
  [results/dysts_dt30_basinblock_p256_seq10_100k_20260430/long_horizon_eval/collect](/home/mila/l/lia/skae/results/dysts_dt30_basinblock_p256_seq10_100k_20260430/long_horizon_eval/collect)
  with `1080/1080` rows, `0` pending/invalid rows, and median full-horizon
  finite coverage of `1` for every root at every reported horizon. The
  paper-facing rebuild now excludes `dysts:LorenzCoupled`, the only
  six-dimensional member of the original shortlist, and `dysts:MultiChua`, the
  Chua-family duplicate. It reports arithmetic means across per-system
  seed-IQMs on the retained `10` three-dimensional systems. After the repaired
  Sparse MLP-BD refresh, the displayed Sparse MLP-BD values are
  `3.96e-4/0.113/0.436` at `H100/H2000/H4000` and `0.658` at `H5000`. The
  displayed aggregate-best rows are Sparse MLP at `H100`, LISTA at `H2000`,
  and repaired Sparse MLP-BD at `H4000`.
- The within-system Wilcoxon/Holm read against Dense MLP is positive but not
  universal and remains a diagnostic rather than the main inference. On the
  retained `10` systems, LISTA improves `10/10` systems at every horizon,
  LISTA-BD improves `9/10`--`10/10`, Sparse MLP improves `9/10`--`10/10`,
  and Sparse MLP-BD improves `9/10`--`10/10`. The aggregate Wilcoxon/Holm test
  over retained systems clears LISTA at every horizon, LISTA-BD except `H500`,
  and both sparse MLP rows through `H4000`; LISTA-SB is displayed in the
  manuscript-linked trend figure but remains a diagnostic Dysts row because it
  does not survive the all-comparison correction.
  The paper-facing raw-MSE trend uses log scale because the displayed IQM MSEs
  span about `4.8e3x`; linear-scale and side-by-side scale-check plots are
  retained as diagnostics. Trend bands are fixed-system seed-bootstrap `95%`
  intervals after system-wise log-relative normalization and are anchored to
  the displayed arithmetic mean, so they quantify typical finite-seed
  uncertainty within an average system rather than variation across the
  retained `10` systems or raw-scale sensitivity to one high-MSE resample. The
  `_raw_seed_ci` plots retain the original raw-MSE fixed-system interval,
  `_system_ci` plots resample fixed per-system seed-IQMs across systems, while
  `_log_seed_bootstrap` plots resample seeds after a `log10` MSE transform. The table and
  plots are under
  [docs/figures/neurips_paper_2026](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026),
  and phase portraits for all `12` originally collected systems at `H1000`--`H5000` are under
  [docs/figures/dysts_dt30_phase_portraits_seed0_h1000_h5000_all_models_20260501](/home/mila/l/lia/skae/docs/figures/dysts_dt30_phase_portraits_seed0_h1000_h5000_all_models_20260501).
- *2026-05-07 IQM-over-IQM appendix sensitivity:* the appendix now includes a
  robust cross-system Dysts display generated from the same retained-`10`
  `dt x30` summary with the repaired Sparse MLP-BD row. It uses seed IQM
  within each system followed by IQM across retained systems. The resulting
  Sparse MLP-BD values are `1.39e-4`, `0.00209`, `0.00676`, `0.0142`,
  `0.0268`, `0.0912`, `0.238`, and `0.425` from `H100` through `H5000`,
  making Sparse MLP-BD best at every displayed horizon. Use this to answer the
  robust-aggregation question; do not silently swap it into the main
  average-case table.
- A separate LISTA soft-block `d_z=256` Dysts sensitivity row has completed
  under
  [results/dysts_long_horizon_eval_dysts_seq10_lista_softblock_p256_sc6em3_seeds0to14_20260428](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_dysts_seq10_lista_softblock_p256_sc6em3_seeds0to14_20260428).
  It covers the same `12` Dysts systems and seeds `0`--`14` with `180/180`
  completed evaluations, but the available collector is the older
  `H5000`--`H30000` horizon grid. Treat it as appendix sensitivity, not as a
  row in the main `H5000`--`H60000` Dysts table.
- The older seven-root `H<=30000` comparison is now superseded for the
  manuscript table. Keep it only as provenance for older visual packets or
  exploratory discussion.
- The older fair `200k` cross-system benchmark remains useful supporting
  context at `H100-H3000`. It is a three-way read rather than a simple
  sparse-versus-dense result: sparse MLP is best at `H100/H1000`
  (`2.947e-4 / 0.0240`), promoted dense sparse-latent Koopman is best at
  `H500` and `H1500-H3000` (`0.0047 / 0.0449 / 0.0627 / 0.0880 / 0.1039`),
  and zero-sparsity improves late-horizon coverage on more systems than it
  wins by median.

Interpretation:
- Sparse-latent Koopman models remain competitive in aggregate on Dysts, and
  under the `dt x30` sensitivity they usually improve over the Dense MLP
  baseline. The per-system Holm counts are moderate rather than universal, so
  this remains forecasting support rather than a broad dominance claim.
- Stronger transition structure is fragile in the older tiny-step packet but
  not in the `dt x30` packet. The correct paper phrasing is sensitivity to the
  discrete-time benchmark design, not categorical block-diagonal failure.
- The robust retained-system Dysts statistic now favors repaired Sparse MLP-BD
  across the whole `H100`--`H5000` range, which is useful appendix evidence
  that the block-diagonal MLP control is competitive when heavy-tailed system
  effects are trimmed.
- The `dt x30` sensitivity changes that diagnosis from "hard block-diagonal
  LISTA is categorically weak on Dysts" to "the old tiny-step/very-long
  composition benchmark exposes instability." Because the sensitivity changes
  the timestep and horizon index, it should be labeled as a
  benchmark-discretization diagnostic rather than pooled into the old
  `H5000`--`H60000` table.
- This benchmark is supporting evidence, not a support-label agreement test,
  because Dysts is not the controlled labeled multibasin benchmark.

Project implication:
- Put Dysts after the multibasin alignment and routing results. It should
  answer "do we remain competitive?" rather than replace the central
  interpretability story.
- Use the fair `200k` benchmark as supporting context if the paper needs a
  shorter-horizon all-system aggregate table, but do not let it reorder the main
  experiments section.

Next drafting step:
- Keep the Dysts table concise and explicitly secondary. If a Dysts visual is
  included, use the `dt x30` log-scale mean-MSE trend as the main view and keep
  the ratio-to-Dense, linear-scale, or IQM-over-IQM plots as supporting
  diagnostics rather than replacing the statistical table.

Primary artifacts:
- [results/dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428/collect](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428/collect)
- [results/dysts_seq10_lista_softblock_p256_sc6em3_seeds0to14_20260428](/home/mila/l/lia/skae/results/dysts_seq10_lista_softblock_p256_sc6em3_seeds0to14_20260428)
- [results/dysts_long_horizon_eval_20260414](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414)
- [results/dysts_long_horizon_eval_mlp_blockdiag_20260415](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_mlp_blockdiag_20260415)
- [docs/figures/dysts_phase_portraits/dysts_h30000_best_root_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h30000_best_root_phase_portraits_manifest.json)
- [results/paper_zero_sparse_benchmark_200k_20260321](/home/mila/l/lia/skae/results/paper_zero_sparse_benchmark_200k_20260321)

## 4. Other useful things

Keep these as supporting, appendix, or falsification evidence unless the paper
needs a specific response to a reviewer-style objection.

- Centered local-law mechanism: support-conditioned centered local laws beat
  the learned global transition on states far from basin boundaries, with win rates
  `93.1%` for block-diagonal sparse-latent Koopman, `98.6%` for dense
  sparse-latent Koopman, and `100%` for the dense non-sparse MLP control on
  the deepest slice. This supports the local-dynamics framing but is not
  LISTA-specific.
- True local geometry: the second-audited true-Jacobian/eigendirection packet
  is mixed. LISTA support families often beat random partitions near
  attractors, but the dense non-sparse MLP can have lower absolute projected
  Jacobian error because its latent representation is closer to the identity
  map. Use this as a
  falsification diagnostic, not a headline claim.
- Controlled transfer: dense sparse-latent exact top-`8` supports switch well
  after a deliberate state-space bridge, while family-level switching is strong
  for multiple roots. This supports support switching but should be written
  separately from a deployment-ready intervention/control claim.
- Negative zero-intercept operator selection: the earlier affine-only local-law
  result failed even with oracle basin fits. Keep it because it justifies the
  centered local-law diagnostic and prevents overclaiming exact operator recovery.
- Phase portraits and visual packets: useful for senior-coauthor handoff and
  appendix figures, but not substitutes for the three main evidence questions.
- Hard-system forecasting: the Kuramoto/Hopfield and step-size-rescue packets
  are decision-grade limitation/support evidence. They show that step size is
  a real bottleneck, that block-diagonal structure gives targeted Kuramoto
  advantages at moderate dimension, and that Hopfield remains mostly
  MLP-better. The corrected `d_z=1024` tanh Dense MLP redo closes the loop in
  the same direction: Dense MLP wins all CLV/Hopfield/Kuramoto system-median
  comparisons at `H100`, `H500`, and `H1000`. This belongs in limitations or
  appendix unless the main paper needs a hard-system stress-test paragraph.
- Label-free clustering and broad support-audit diagnostics: keep these as
  appendix/provenance for why literal binary support uniqueness is not enough.
  They explain known negatives such as Kuramoto, mixed continuous-separation
  cases such as Hopfield, and the shift toward support families, top-`k`
  supports, and support-label agreement on the fixed-`17` systems.
- Historical benchmark provenance: dense-LISTA recipe selection, the symmetric
  `50k` audit, invalidated historical block-`K` controls, and March queue
  repair notes should stay as provenance. The main text should cite the
  repaired `200k` and fixed-`17` packets instead.
- Queue history, failed/superseded packets, tuning provenance, and selector
  sensitivity should be linked from the detailed log or archive rather than
  repeated in the main experiment narrative.

Primary artifacts:
- [results/transition_rich_centered_chart_mechanism_20260420](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420)
- [results/true_jacobian_geometry_fixed17_seed0_20260424_reaudit](/home/mila/l/lia/skae/results/true_jacobian_geometry_fixed17_seed0_20260424_reaudit)
- [results/controlled_transfer_switching_fixed17_seed0_20260424_reaudit](/home/mila/l/lia/skae/results/controlled_transfer_switching_fixed17_seed0_20260424_reaudit)
- [results/kuramoto_dimension_sweep_dt00625_200k_20260309](/home/mila/l/lia/skae/results/kuramoto_dimension_sweep_dt00625_200k_20260309)
- [results/hard_system_sparse_kae_redo_p1024_seq8_100k_halflr_sc6em3_tanh_dense_20260429](/home/mila/l/lia/skae/results/hard_system_sparse_kae_redo_p1024_seq8_100k_halflr_sc6em3_tanh_dense_20260429)
- [PAPER_TRACK_STATUS.md](/home/mila/l/lia/skae/docs/PAPER_TRACK_STATUS.md), broad support-audit and label-free clustering entries
- [EXPERIMENTS_ARCHIVE.md](/home/mila/l/lia/skae/docs/EXPERIMENTS_ARCHIVE.md)
