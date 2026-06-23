# Paper Track Status

Date: June 23, 2026
Evidence organization last refreshed: `2026-05-07`
Paper-critical live queue status last refreshed: `2026-06-23`

## Paper-Track Summary

Problem being solved:
- The paper now needs an evidence-first experiments narrative matching the
  draft setup: multibasin Koopman learning, sparse supports as inspectable
  support objects, and support families as label-free basin-identification
  variables.
- The main manuscript now foregrounds the simpler \(F_{\rm abs}\)-routed
  learned-intercept local affine result as the controlled-multibasin local
  linearization section. The more elaborate \(C_{\rm stab}\) support-flow
  construction has been moved to appendix diagnostic context because matched
  route controls show that it is not needed for the main forecasting gain.

Current solution:
- Use [PAPER_EXPERIMENT_EVIDENCE_MAP.md](/home/mila/l/lia/skae/docs/PAPER_EXPERIMENT_EVIDENCE_MAP.md)
  as the drafting order and display plan for the experiments section.
- June 23 label-free local Koopman/EDMD baseline completion: the fair
  multi-model/switching-style standalone baseline is now implemented, tested,
  run, and tabled. It fits k-means-routed local EDMD operators without basin
  labels or basin counts, selects the route count from `1,2,4,8,16` on held-out
  training trajectories, and refits on the full train split. CPU-only arrays
  `9914514` and `9914513` completed all `45/45` retained-multibasin and
  `30/30` Dysts tasks. Local polynomial EDMD is the strong stable row:
  `0.150/0.252/0.275` on retained multibasin H100/H500/H1000 and
  `5.0e-4/2.17/2.97` on Dysts H100/H2000/H4000. Local RBF-EDMD remains in the
  table but should be interpreted as unstable because a few rollouts dominate
  the arithmetic means. Relative to the current paper forecasting rows, local
  polynomial EDMD is a credible baseline but not a replacement for the
  LISTA/sparse KAE rows: on retained multibasin it is
  `3.7x/1.8x/1.7x` worse than LISTA and `3.9x/2.7x/2.6x` worse than the best
  sparse KAE row at H100/H500/H1000, although it beats Dense MLP by
  `5.5x/10.6x/10.7x`. On Dysts it is only short-horizon competitive; it is
  `22x/6.8x` worse than the best sparse KAE row at H2000/H4000 and worse than
  Dense MLP at those long horizons. The updated standalone artifacts are
  [table_standalone_state_space_baselines.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table_standalone_state_space_baselines.tex)
  and the paired summary CSVs under
  [docs/figures/neurips_paper_2026/_tables](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables).
- May 26 local-linearization manuscript pass: the main paper now has a compact
  \(F_{\rm abs}\)-routed local affine subsection,
  `sec:fabs_local_linearizations`, in
  [neurips_sparse_koopman_multibasin.tex](/home/mila/l/lia/skae/docs/neurips_sparse_koopman_multibasin.tex).
  It reports support-family local \(K_c,d_c\) maps as the primary local
  linearization result because they are simpler, directly tied to the paper's
  main support-family object, and empirically at least as strong as
  \(C_{\rm stab}\). The table
  [table_fabs_local_k_forecasting.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table_fabs_local_k_forecasting.tex)
  reports paired wins, mean best-periodic MSEs, median ratios, and geometric
  ratios versus matched global-\(K\) LISTA. The \(C_{\rm stab}\) route,
  diagram, and retained result now live in
  [support_stable_local_linearizations.tex](/home/mila/l/lia/skae/docs/appendix/support_stable_local_linearizations.tex)
  as an inspectable support-flow diagnostic.
- May 26 matched sparse route-baseline completion: the paper-critical matched
  route controls for the retained \(C_{\rm stab}\) local-map result are now
  complete. The final resume arrays reached `225/225` rows for oracle-basin
  `9645554`, support-family `9645583`, random-matched `9645587`, and
  latent-kmeans `9645669`; clean manual collect/compare/wide aggregate jobs
  all exited `0:0`. These controls match the retained \(C_{\rm stab}\) run on
  task roster, `200000` total steps, `100000/100000` stage split, stable-fit
  data size, learned-intercept local maps, checkpoint-selection rule, and wide
  periodic period grid. Versus matched global-\(K\) LISTA, H1000 row wins and
  median staged/global ratios are: \(C_{\rm stab}\) `188/225`, `0.399`;
  support-family `189/225`, `0.328`; random-matched `185/225`, `0.451`;
  latent-kmeans `196/225`, `0.341`; oracle-basin `195/225`, `0.366`.
  Direct H1000 comparison to \(C_{\rm stab}\) shows support-family better on
  `129/225` rows with median baseline/\(C_{\rm stab}\) ratio `0.899`,
  latent-kmeans better on `132/225` with ratio `0.903`, random worse
  (`95/225`, ratio `1.058`), and oracle essentially tied (`112/225`, ratio
  `1.000`). This resolved the positioning decision: the main text now reports
  support-family local maps as the primary result, while \(C_{\rm stab}\) is
  retained only as appendix evidence for inspectable support-flow semantics.
- May 25 dense continuous tail-fate local-\(K_c\) control readout: a prior
  staged dense-fate local-map run was not found. The existing dense control was
  only post-hoc continuous latent-fate clustering, while the active
  `latent_kmeans` route baseline clusters LISTA latent states rather than the
  dense zero-sparsity MLP fate object. The trainer now has a
  `latent_tail_fate` route that clusters dense latent trajectory tail
  summaries using tail mean/std/final features with silhouette-selected
  \(k\le 12\), assigns route-fit transitions by trajectory fate, and routes
  deployment rollouts by nearest latent route center. A compute-node compile
  check completed as `9615576` with exit `0:0`. The five-seed control completed
  under
  [results/staged_dense_tail_fate_local_k_mlp_zero_seed5_20260521](/home/mila/l/lia/skae/results/staged_dense_tail_fate_local_k_mlp_zero_seed5_20260521):
  launcher `9615608`, training array `9615609`, collect `9615610`, compare
  `9615611`, and wide-periodic re-evaluation `9615612` all completed with
  exit `0:0`, yielding `75/75` staged/global wide-periodic rows (`15` systems
  x seeds `0`--`4`). Dense-tail-fate staged local maps beat their dense
  global-\(K\) anchor on `73/75`, `71/75`, and `71/75` rows at
  `H100/H500/H1000`, `70/75` rows on all three horizons, and `15/15` systems
  at `H1000`; the median `H1000` candidate/global ratio is `0.0418`. Direct
  paired comparison against sparse \(C_{\rm stab}\)-routed LISTA on the same
  `75` system/seed pairs is strongly negative for dense-fate: dense row wins
  are `0/75` at `H100`, `2/75` at `H500`, `2/75` at `H1000`, and `0/75` on
  all horizons. Median absolute best-periodic MSEs are dense-fate
  `0.0522/0.0922/0.0962` versus sparse \(C_{\rm stab}\)
  `0.000446/0.000297/0.000297` at `H100/H500/H1000`, with geometric mean
  dense/sparse ratios `55.0/94.0/104.3`. Mechanistically, the dense control
  clusters continuous latent trajectory-tail summaries and routes a rollout
  state by nearest center of the current dense latents associated with those
  fate clusters. \(C_{\rm stab}\) instead routes from the current sparse
  support: current mask \(\rightarrow\) base support family \(\rightarrow\)
  recurrent support-flow component \(\rightarrow\) local affine map. This
  control therefore does not force an interpretability-only repositioning:
  sparse \(C_{\rm stab}\)-routed local maps remain prediction-superior to the
  dense-fate route on the retained controlled suite, while also exposing an
  inspectable support-transition graph. Keep data-efficiency language
  conservative because no sample-size sweep was run, and keep the section
  baseline-sensitive until the unfinished sparse route baselines complete.
- June 9 ManiSkill activation audit and corrected `5k` dense-tanh versus
  sparse-KAE pilot: the
  previously reported dense ManiSkill periodic rows used the original ReLU MLP
  blocks, not tanh. The controlled trainer now has an explicit activation
  option, and dense no-sparsity baselines now default to tanh and reject
  non-tanh baseline launches. Thus `dense_tanh_sp0` is a true dense-tanh
  baseline. The corrected packed-GPU `5k` comparison and follow-up optimizer
  fairness control,
  evaluated with `last.pt` and best periodic decoded-prediction re-encoding
  over `{1,2,5,10,20,50,100}`, finds sparse KAE rows beating true dense tanh
  on seeds `0`--`2`. Standard dense tanh H10--H50 mean state MSE is
  `0.002079`; applying the LISTA-favored optimizer setting
  `lr=5e-4,weight_decay=0` improves dense tanh to `0.002023`. Under the same
  optimizer, sparse-MLP ReLU is strongest: `sparsity=0.003` reaches `0.001837`
  and `sparsity=0.01` reaches `0.001860`. Best LISTA in the fairness rerun is
  the standard optimizer row at `0.001881`; LISTA with
  `lr=5e-4,weight_decay=0` is weaker on H10--H50 (`0.001968`) but stronger at
  H125 than standard LISTA. Treat this as a promising benchmark-development
  pilot for sparse KAEs, but not as a LISTA-specific forecasting win: the fair
  optimizer-control winner is sparse-MLP ReLU. ManiSkill still needs validated
  outcome/contact labels and support diagnostics before paper-facing claims.
- June 9 ManiSkill workstream split: the robotics evidence is now separated
  into two non-interchangeable benchmarks. The perturbation-balanced insertion
  stream tests whether sparse support families align with contact/outcome
  regimes in `PegInsertionSide-v1`; it remains blocked on semantic relabeling
  because the current five classes are target perturbation labels, not
  validated physical outcomes. The new ManiSkill-10 default-task stream is a
  forecasting-generalization benchmark over official demo tasks with periodic
  decoded-prediction re-encoding; it tests representation quality, not
  support-regime discovery. Protocol sources are
  [MANISKILL_INSERTION_BENCHMARK.md](/home/mila/l/lia/skae/docs/MANISKILL_INSERTION_BENCHMARK.md),
  [MANISKILL10_DEFAULT_TASK_FORECASTING.md](/home/mila/l/lia/skae/docs/MANISKILL10_DEFAULT_TASK_FORECASTING.md),
  and
  [maniskill10_default_tasks.tsv](/home/mila/l/lia/skae/experiments/maniskill10_default_tasks.tsv).
- June 10 ManiSkill CPU/GPU gates: perturbation label audit job `9801691`
  completed and confirms that the current `e20` packet should remain
  pipeline/debugging evidence only. It exposes balanced target labels but only
  binary final-success semantics (`15` success, `85` non-success), with no
  named contact, insertion-depth, peg-hole-distance, grasp/drop, or
  rim/alignment feature groups for five-way physical relabeling. ManiSkill-10
  default-task data preparation completed for all ten selected tasks after
  smoke `9801692`, full array `9801694`, and trajectory-discovery rescue
  `9801740`; no GPUs were used for data prep. Initial GPU smoke `9801761`
  failed immediately because `jq` was unavailable on the GPU node and the
  pending task was canceled, limiting wasted accelerator time to about one
  second. Corrected packed GPU smoke `9801812` completed `8/8` expected rows
  for `PickCube-v1` and `PlugCharger-v1` with four concurrent workers per GPU.
  It provides a viable end-to-end default-task forecasting smoke: sparse rows
  are mixed on `PickCube-v1` but on harder `PlugCharger-v1` beat dense tanh at
  H10/H20 and are competitive through H150/H200. GPU telemetry reached `100%`
  utilization, but combined active fraction was only `57.7%` because the
  longer task held the GPU during mostly low-utilization periodic
  evaluation/support bookkeeping. The launcher now defaults to checkpoint-only
  GPU training, and
  [run_maniskill10_eval_cpu_array.sh](/home/mila/l/lia/skae/scripts/run_maniskill10_eval_cpu_array.sh)
  provides the dependent CPU evaluation/support pass for scale-up.
- May 14 sparsity-adjacent literature pass: the Discussion in
  [neurips_sparse_koopman_multibasin.tex](/home/mila/l/lia/skae/docs/neurips_sparse_koopman_multibasin.tex)
  now briefly positions sparse Koopman supports relative to LLM sparse
  autoencoders/dictionary-learning interpretability and sparsity-based
  continual-learning interference mitigation. This is framing only, not a new
  paper claim: SKAEs are neither post-hoc probes of a frozen language model nor
  sequential task learners with task identities. The defensible connection is
  that sparse support identities can separate otherwise interfering
  computations; in this paper those computations are local Koopman-compatible
  dynamics and the support is evaluated as a regime variable.
- May 14 normalized-decoder fairness replacement queued: the paper-facing
  table replacement candidate now normalizes linear decoder atoms for every
  KAE row, including Dense MLP, while preserving the current rollout-latent
  sparsity target (`normdec_rollout`). A matched conceptual ablation
  (`normdec_encoded`) applies the L1 sparsity term to encoded latents for all
  sparse KAEs, including LISTA-based rows, because the support diagnostics are
  computed from encoded states. Focused test job `9553712` gates controlled
  launchers `9553720/9553721` and Dysts launchers `9553722/9553723`. Group
  sparsity for block-diagonal rows is deferred because it changes the
  objective and block prior rather than only correcting decoder fairness.
- May 15 normalized-decoder queue readout: test job `9553712` passed and the
  Dysts `normdec_rollout`/`normdec_encoded` chains completed, but this packet
  should not replace paper tables. On retained-`10` Dysts, `normdec_rollout`
  gives Sparse MLP-BD `3.11e-4/0.0900/0.433/0.679` at
  `H100/H2000/H4000/H5000`, while `normdec_encoded` gives
  `2.66e-4/0.0878/0.472/0.704`. The rollout-target row is best among
  normalized-decoder rows at `H2000/H4000`, but the normalized-decoder packet
  worsens several other rows and encoded-latent sparsity is mixed. The
  controlled multibasin launchers `9553720/9553721` failed before submitting
  arrays because one source table only had seed-`10`--`14` rows for several
  roots; the launcher has been patched to merge the seed-`0`--`9` source
  tables with the backfill before requeueing.
- May 16 controlled normalized-decoder seed-3 readout: the patched controlled
  launcher now has a collected initial read, but not a table-ready one.
  Corrected launchers `9561573` (`normdec_rollout_seed3`) and `9561574`
  (`normdec_encoded_seed3`) completed. Rollout training array `9561575`
  completed `226/270` tasks and timed out `44`; encoded training array
  `9561603` completed `255/270` and timed out `15`. Forecasting and
  per-basin-deep support diagnostics collected for the completed runs, and no
  related SLURM jobs remain active. Rollout-target sparsity still favors
  plain Sparse MLP at longer controlled horizons, while encoded-target
  sparsity gives a more balanced controlled read and makes Sparse MLP-BD best
  at `H100`; coverage gaps and the mixed Dysts result mean the current paper
  tables should not be replaced from this packet.
- May 7 supplementary-materials split: the appendix/supplement todo items A1--A6
  from the evidence map now each have their own LaTeX fragment under
  [docs/supplementary_materials](/home/mila/l/lia/skae/docs/supplementary_materials),
  with an input manifest in
  [supplementary_materials.tex](/home/mila/l/lia/skae/docs/supplementary_materials/supplementary_materials.tex).
  These files preserve support-object definitions, the fixed-system inventory,
  the old global-deep comparison, matched hard-init controls, local-law
  diagnostics, and historical hard-system provenance as supplement-only
  material; they do not change the main evidence order.
- May 7 Dysts robust aggregation appendix: the Dysts appendix now has a
  separate IQM-over-IQM table generated with the repaired retained-`10` Dysts
  `dt x30` Sparse MLP-BD row. The statistic is seed IQM within system followed
  by IQM across retained systems; it makes repaired Sparse MLP-BD the best row
  at every displayed horizon from `H100` through `H5000`. This is an appendix
  sensitivity view, not a replacement for the main arithmetic-mean Dysts
  table. SLURM job `9491319` regenerated the matching curve PDFs/PNGs from the
  repaired summary.
- May 7 ground-truth multibasin visualization appendix: the multibasin
  benchmark inventory appendix now includes the generated vector-field
  composite plus detailed three-panel figure groups for all retained `15`
  controlled systems. These displays document benchmark geometry only; the
  learner still receives stored states rather than vector fields, attractor
  centers, basin labels, or basin counts.
- May 12 prospective benchmark-extension draft: the manuscript now has a
  Discussion bridge plus
  [appendix/benchmark_extensions.tex](/home/mila/l/lia/skae/docs/appendix/benchmark_extensions.tex)
  describing two future benchmark additions. The reliable addition is a
  spatialized multibasin reaction-diffusion field benchmark that lifts the
  existing procedural vector fields to high-dimensional two-channel grids
  while keeping attractor metadata evaluation-only. The application-style
  addition is controlled ManiSkill insertion, where support families would be
  evaluated against hidden contact/outcome labels. This is not current
  evidence and should not be phrased as a completed result.
- May 12--14 benchmark-extension smoke scaffolds and same-seed controls: separate progress documents now
  track the one-seed-first execution path for the spatialized PDE and
  ManiSkill insertion benchmarks. PDE job `9530539` completed and produced
  finite metrics, but this is only pipeline validation: field MSE is finite at
  `H=1/4/8/12`, while final-basin consistency is `0.0`, the sparse latent
  collapsed late in training, and final fields are mixed under the
  final-majority diagnostic.
  ManiSkill prepare job `9530626` completed after switching the default path
  to raw downloaded `env_states`, yielding a `1000`-episode compact
  `PegInsertionSide-v1` state/action dataset; controlled LISTA train/eval job
  `9530627` completed with finite rollout metrics but collapsed all states to
  one support family. One-seed tuning then produced non-degenerate support
  reads. For PDE, `LISTA_ALPHA=0.001`, `SPARSITY_COEFF=0` reached `H=12`
  final-basin consistency `0.917` and compressed support-family NMI `0.709`
  with `7` validation representatives at Jaccard `0.7`; the same-seed dense
  control forecasts better at `H=12` (`0.885` versus LISTA `1.032`) but has no
  compact support alignment at the matched threshold and only reaches NMI
  `0.616` by overfragmenting into `224` validation representatives. For ManiSkill,
  `LISTA_ALPHA=0.2`, `SPARSITY_WEIGHT=0.03` reached H100 state MSE `0.248`
  and outcome NMI `0.347` in an overfragmented read; a compressed `36`-family
  partial read gives outcome NMI `0.288`, and the completed compact sweep gives
  `131` families with outcome NMI `0.303` at support threshold `0.2`, Jaccard
  `0.4`. The dense controlled KAE has much worse H100 state MSE (`3.644`) and
  no support signal at the matched threshold; its best outcome-NMI sensitivity
  reaches `0.312` only with `753` families. These are still one-seed tuning
  results, not paper evidence.
- May 21 spatialized-PDE convolutional implementation and support-tuning pass:
  the high-dimensional
  benchmark now has a convolutional Koopman model family (`conv_lista`,
  `conv_dense`, `conv_sparse_mlp`), a label-settling dataset generator, expanded
  basin-map/Fourier/support diagnostics, and task-table SLURM launchers. Direct
  smoke job `9615803` completed with exit `0:0`, and task-table launcher
  `9615822` submitted array `9615825`, where all `3/3` child tasks completed
  with exit `0:0`. A first controlled conv pilot also completed: launcher
  `9615832` submitted array `9615833`, and all `6/6` child tasks completed
  with exit `0:0` for `cal_square_4` and `transition_routes_4`, seed `0`, and
  the three conv variants. Both generated datasets are numerically clean
  (`[40, 13, 16, 16, 2]`, invalid values `0`, clipped values `0`). Best
  validation MSEs are `0.8990/0.8997/0.8991` on `cal_square_4` and
  `1.6977/1.7039/1.7039` on `transition_routes_4` for
  `conv_lista`/`conv_dense`/`conv_sparse_mlp`; at `H=4`, `conv_lista` has the
  lowest field MSE on both systems (`0.8751`, `1.6752`). This validates
  benchmark mechanics and finite conv training/evaluation. A follow-up tuning
  loop then fixed the dense-support collapse. The selected setting is
  `z_dim=128`, `hidden_channels=32`, `num_blocks=2`, `lista_num_loops=3`,
  `lista_alpha=0.03`, `sparsity_weight=0.05`, evaluated with
  `support_threshold=0.3` and `family_jaccard=0.8`. Matched controls over
  `cal_square_4` and `transition_routes_4`, seeds `0,1`, completed cleanly:
  mean `H(B|F_tuned)` / `H(F_tuned|B)` / family count is
  `0.103/0.569/5.50` for LISTA, `0.087/0.509/5.25` for sparse-MLP, and
  `0.173/0.612/5.25` for dense. Secondary NMI/purity/H4-MSE is
  `0.751/0.938/0.970`, `0.776/0.938/0.977`, and `0.708/0.875/0.962`.
  This fixes the support-read blocker under a tuned diagnostic object, but it
  is still benchmark-development evidence rather than a main paper claim
  because sparse-MLP is slightly lower on `H(B|F_tuned)` and dense is
  non-degenerate.
- May 25 spatialized-PDE support-threshold correction: the user's concern that
  `family_jaccard=0.8` was too high is confirmed. Re-scoring the selected
  checkpoints shows the old LISTA diagnostic overfragmented to `5.50` mean
  families with `H(B|F)=0.103`, `H(F|B)=0.569`; `threshold=0.2`,
  `Jaccard=0.4` gives exactly `4.00` mean families with `H(B|F)=0.217`,
  `H(F|B)=0.363`, support size `58.8/128`. This read is now explicitly
  demoted to calibration history because the screen used `grid16` states
  (`d_x=512`) with `d_z=128`, violating the required Koopman lifting rule
  `d_z >= 4*d_x`. Future spatialized PDE runs must use at least `d_z=2048`
  for grid `16` and `d_z=8192` for grid `32`. A higher-sparsity LISTA screen
  makes lower thresholds usable but does not rescue `threshold=0.01`:
  `alpha=0.2`, `sparsity=0.1`, `threshold=0.1`, `Jaccard=0.4` gives `3.75`
  families, `H(B|F)=0.227`, `H(F|B)=0.303`, support size `47.2`, and
  `alpha=0.2`, `sparsity=0.2`, `threshold=0.05`, `Jaccard=0.4` gives
  `4.00` families, `H(B|F)=0.222`, `H(F|B)=0.363`, support size `53.1`.
  Matched controls are not favorable to a LISTA-specific claim at this small
  scale: sparse-MLP with `alpha=0.2`, `sparsity=0.1`, `threshold=0.03`,
  `Jaccard=0.6` also reaches `4.00` families with `H(B|F)=0.217`,
  `H(F|B)=0.406`. Paper implication: keep spatialized PDE as benchmark
  development unless the next pass justifies a scale-calibrated support rule
  or a \(C_{\rm stab}\)-style support-flow object and preserves matched
  controls.
- May 25 spatialized-PDE overcomplete rerun: the undercomplete-latent issue is
  now fixed in code and in the first interactive results. The current valid
  screen uses `grid16`, `d_x=512`, and `d_z=2048` (`4x`) on `cal_square_4`
  and `transition_routes_4`, seed `0`, with matched dense controls. At the
  fixed diagnostic setting `support_threshold=0.2`, `Jaccard=0.2`, evaluated
  on basin-dominant fields with `majority_fraction>=0.7`, the best current
  row is `conv_lista`, `alpha=0.2`, `sparsity_weight=0.05`. It gives
  `H(B|F),H(F|B)=0.000,0.231` on `cal_square_4` and `0.000,0.277` on
  `transition_routes_4`; dense gives `0.231,0.549` and `0.382,0.277`.
  LISTA support sizes are also much smaller (`78.4` and `106.9` active
  coordinates out of `2048`) than dense (`681.6` and `610.7`). Forecasting is
  competitive enough to scale: LISTA wins H12 on `cal_square_4` and wins
  H4/H8/H12 on `transition_routes_4`, though it is mildly worse than dense at
  H4/H8 on `cal_square_4`. Paper implication: this was the first promising
  high-dimensional overcomplete PDE result and justified the follow-up
  expansion. Keep it out of the main evidence order because the subsequent
  matched sparse-MLP/dense expansion is mixed.
- May 25 spatialized-PDE expansion queue: the first overcomplete statistical
  expansion is complete and summarized. Queue parents `9647692`, `9647693`,
  and `9647695` submitted arrays `9647700`, `9647701`, and `9647702` for
  selected LISTA, dense, and sparse-MLP respectively. Scope was the three
  additional systems `cal_high_cross_3`, `var_l_shape_5`, and
  `cal_pentagon_5`, seeds `0`--`4`, `grid16`, `d_x=512`, `d_z=2048`, and
  `20,000` steps. Runner limits were `long`, `gpu:1`, `16G`, `4` CPUs,
  `02:50:00`, with checkpoint/resume enabled every `1000` steps. All training
  arrays and the dependent support array `9648659` completed with exit `0:0`.
  Result: dense is still the best short/mid-horizon forecaster on the added
  systems (`H4/H8/H12` mean `0.4278/0.4374/0.5339`) versus LISTA
  `0.4842/0.4817/0.5364` and sparse-MLP `0.4777/0.4798/0.5476` using
  time-averaged rollout MSE. Endpoint-only H12 `final_field_mse` is more
  favorable to LISTA (`0.7641` versus dense `0.8746` and sparse-MLP `0.8121`),
  but the packet is not a long-horizon test because `trajectory_length=16`
  leaves only H16 as the largest ground-truth forecast horizon. At the fixed
  support diagnostic (`threshold=0.2`, `J=0.2`, `majority_fraction>=0.7`),
  LISTA gives compact supports (`19.3` active coordinates versus dense
  `78.5`) and modestly lower fragmentation (`H(F|B)=0.623` versus `0.671`),
  but not better purity entropy (`H(B|F)=0.074` versus dense `0.063`);
  sparse-MLP is best on `H(B|F)=0.034`. Paper implication: the PDE expansion
  is a useful high-dimensional stress test, not a landed main-text dominance
  claim. Use it, if at all, to show forecast/support tradeoffs until targeted
  sparse tuning recovers forecasting quality.
- May 26 spatialized-PDE long-horizon pilot: parent `9656822` completed and
  submitted training array `9656825` plus dependent support parent `9656826`.
  This queue fixes the prior horizon limitation without silently clipping
  H256/H512: stored trajectories have `trajectory_length=512` and
  `label_extra_observations=512`, while the trainer samples only from the first
  `128` observation intervals with `sequence_length=8`. Scope is
  `cal_square_4` and `transition_routes_4`, seeds `0,1`, and
  `conv_lista`/`conv_dense`/`conv_sparse_mlp`, all at `grid16`, `d_x=512`,
  `d_z=2048`, and `eval_horizons=1,4,8,16,32,64,128,256,512`. The
  training/evaluation array completed all `12/12` tasks with exit `0:0`, but
  its original `conv_dense` rows are superseded: the PDE convolutional dense
  path used GELU hidden blocks instead of the tanh dense-control convention
  used by `generic_no_shrink`. The code is now patched so `conv_dense` uses
  tanh, old support parent `9656826` was canceled, and corrected dense-only
  parent `9659093`, training/evaluation array `9659165`, and support-sweep
  array `9659514` all completed with exit `0:0`. Corrected tanh dense is the
  strongest forecaster through H256, with H8/H16/H32/H64/H128/H256 field MSE
  `0.242/0.302/0.917/1.441/1.819/2.126`, versus LISTA
  `0.695/0.773/1.208/1.623/2.015/2.750` and sparse-MLP
  `0.668/0.778/1.260/1.693/2.010/2.942` on finite rows. H512 MSE is
  non-finite for every variant. Fixed support alignment goes the other way:
  corrected dense has poor dominant-slice support alignment (`H(B|F)=0.678`,
  support size `151.0`), while sparse rows remain compact and basin-aligned
  (`H(B|F)=0.111`, support size `7.6` for LISTA; `H(B|F)=0.043`, support size
  `11.7` for sparse-MLP). Paper implication: this is a forecast/support
  tradeoff diagnostic, not a sparse forecasting win to expand yet. A follow-up
  capacity audit found no width/depth/latent/K advantage for dense; the
  remaining activation confound is that corrected dense used tanh conv blocks
  while sparse rows used GELU. The tanh-matched sparse/LISTA tuning pilot is
  now complete: `9661532`, rescue array `9662516`, replacement support parent
  `9662517`, and support array `9662576` all exited `0:0`, producing `48/48`
  checkpoints, evaluations, and support JSONs. Corrected tanh dense remains
  best through H32 (`0.289/0.260/0.242/0.302/0.917` at H1/H4/H8/H16/H32)
  versus the best tanh sparse rows (`0.307/0.276/0.259/0.412/0.973`), while
  tanh sparse/LISTA is only modestly better at H64/H128 (`1.409/1.761` versus
  dense `1.441/1.819`) and remains unstable by H256/H512. Support alignment is
  much better than dense: at `threshold=0.3`, `Jaccard=0.4`, LISTA
  `alpha=0.05`, `sparsity=0` gives `H(B|F)=0.283`, `H(F|B)=0.241`, `4.25`
  families, and support size `17.5`, versus corrected dense
  `0.658/0.200`, `2.75` families, support size `57.7`. Paper implication:
  keep this as a forecast/support tradeoff diagnostic and add a
  paper-consistent ReLU `generic_sparse`-style convolutional sparse path rather
  than treating GELU as the sparse high-dimensional default.
- June 2 spatialized-PDE controlled expansion: the paper-consistent ReLU
  sparse path is complete across all five high-dimensional systems and seeds
  `0`--`4` under `spatial_rd_controlled_expansion_20260602`. Final artifacts
  have `25/25` datasets, `125/125` checkpoints, `125/125` evaluations, and
  `125/125` support sweeps after CPU rescues. The run uses `grid16`,
  `d_x=512`, `d_z=2048`, `trajectory_length=512`,
  `label_extra_observations=512`, and train windows of length `8`. Forecasting
  is not a sparse win: finite H1--H128 mean field MSE is tanh dense `0.593`,
  ReLU dense `0.737`, ReLU LISTA `sp=0.01` `0.762`, ReLU LISTA `sp=0.05`
  `0.765`, and ReLU sparse-MLP `0.775`; finite H1--H256 means are
  `0.725/0.866/0.868/0.884/0.908` in the same order, with H512 mostly
  null/non-finite. Support alignment is the positive result: at the calibrated
  PDE support read `tau=0.05,J=0.4`, sparse rows form about `4.2--4.4`
  deep-test families on systems with `3--5` basins and reach deep
  `H(B|F)≈0.049--0.053` with support sizes `47--138`, while ReLU dense has
  `H(B|F)=0.119` and support size `522`, and tanh dense has `H(B|F)=0.412`
  and support size `1048`. GPU utilization was high posthoc (`96.6%` mean over
  `16` packed jobs), but `PACK_SIZE=8` caused several 32-GiB V100 evaluation
  OOMs; future packed launches should separate training from evaluation or use
  a lower pack size on 32-GiB GPUs. Paper implication: keep the spatialized PDE
  result as an appendix high-dimensional support-alignment/compactness stress
  test, not as a main sparse forecasting claim.
- May 22 spatialized-PDE paper-object re-evaluation: the selected PDE conv
  checkpoints were re-scored with the manuscript support objects rather than
  the tuned diagnostic threshold. The run covered `12` matched-control
  checkpoints (`cal_square_4` and `transition_routes_4`, seeds `0,1`,
  variants `conv_lista`/`conv_sparse_mlp`/`conv_dense`) and wrote
  [paper_support_rows.csv](/home/mila/l/lia/skae/results/spatial_rd_paper_support_eval_20260522/paper_support_rows.csv)
  plus
  [paper_support_summary.json](/home/mila/l/lia/skae/results/spatial_rd_paper_support_eval_20260522/paper_support_summary.json).
  Concrete result: with fixed \(S_{\rm abs}=\{|z_i|>10^{-3}\}\) and fixed
  \(F_{\rm abs}\) Jaccard `0.5`, \(F_{\rm abs}\) collapses to one family for
  all selected PDE rows. On `deep_final_test_states`, mean `H(B|F_abs)=1.057`
  and purity `0.469` for LISTA, sparse-MLP, and dense; on `all_test_states`,
  mean `H(B|F_abs)=1.131` and purity `0.438`. Exact \(S_{\rm abs}\) has
  `H(B|S_abs)=0.000` for LISTA and sparse-MLP, but the supports are not
  compact: mean support size is about `118/128`, exact support count is
  `95/97` on all states and `7/7` on deep-final states. Dense is almost fully
  active (`~127.6/128`) and has worse exact-support basin entropy.
  Interpretation: the previous clean-support result was a tuned scale
  diagnostic, not a paper-object result. Do not use the current PDE packet as
  evidence that \(F_{\rm abs}\) discovers compact basin families; keep it as
  benchmark construction evidence unless a justified PDE-scale support rule or
  \(C_{\rm stab}\) support-flow evaluation changes the readout.
- May 19 ManiSkill perturbation-balanced one-seed assessment: the simulator
  generation path now runs on a GPU node and produced a grouped
  `20`-source-episode packet with five target perturbation classes
  (`success`, `jam`, `miss`, `drop`, `partial`). One-seed LISTA, dense
  controlled KAE, and sparse-MLP controlled KAE runs completed on the same
  held-out split. Held-out state MSE at `H10/H25/H50/H100` is LISTA
  `0.00369/0.00700/0.0496/15.91`, dense
  `0.00384/0.00379/0.0231/7.83`, and sparse-MLP
  `0.00352/0.00437/0.0651/143.43`. LISTA has nontrivial target-label support
  alignment (`22` families, outcome NMI `0.616`) while dense has one matched
  support family, but dense is the better long-horizon forecaster and the
  labels are still perturbation targets rather than validated physical
  outcomes. Keep ManiSkill as benchmark-development evidence only; do not
  promote it to a main paper result until the labeler is semantic and LISTA is
  competitive with dense forecasting.
- May 12 strict staged \(F_{\rm abs}\) local-\(K_c\) LISTA control queued and
  revised: fixed-center launcher `9523663` generated `225` retained-Table-1
  tasks and submitted GPU array `9523664`, but that run was canceled after a
  strongly negative partial read. The first source-target affine replacement
  (`9527160 -> 9527170/9527171/9527172`) was then canceled before completion
  to avoid spending the full 15-seed budget before the pilot read. The active
  reduced launcher `9527220` completed and submitted array `9527234`, collect
  `9527235`, and compare `9527236`. It keeps the same 200k-step dense LISTA
  Table 1 recipe on the retained `15` multibasin systems with seeds `0,1`,
  split as `100000` joint encoder/decoder/global-\(K\) steps plus `100000`
  frozen-autoencoder local-\(K_c\) steps, but uses
  \(z_{t+1}=d_c+(z_t-c_c)K_c\) with \(d_c=c_cK_{\rm global}\), so the local
  stage initializes exactly at the frozen global-\(K\) map. Families remain
  label-free (`absolute:0.001`, `J=0.4`) and the collect job compares directly
  against the existing same-budget global-\(K\) LISTA row. The interactive
  target metric has been corrected to best periodic reencoding over a wider
  period grid (`1,2,5,10,20,25,50,100`). Under that stricter baseline,
  `claude:transition_routes_4` seed `0` is a clean multi-family positive:
  staged source-target affine maps fit `3` support families and beat matched
  global-\(K\) LISTA at `H100/H500/H1000`
  (`0.00510/0.0181/0.0901` versus `0.00728/0.0451/0.339`). Seed `1` is
  promising but not a single-checkpoint sweep: a `4000`-step staged run fits
  `2` families and can beat the global baseline per horizon using best/last
  staged checkpoints, but no single checkpoint has beaten all three horizons.
  The same-budget `200000`-step two-seed pilot now has its full stricter read:
  staged array `9527234`, collect `9527235`, compare `9527236`, and
  wide-periodic re-evaluation job `9531292` all completed with exit `0:0`.
  Re-evaluating all `30` staged/global pairs over periods
  `1,2,5,10,20,25,50,100` gives a heterogeneous, non-significant forecasting
  result: staged wins `19/30` rows at `H100`, `15/30` at `H500`, `15/30` at
  `H1000`, and `15/30` on all horizons. Clean two-seed all-horizon positives
  are `claude:cal_asymmetric_3`, `claude:cal_pentagon_5`,
  `claude:transition_routes_4`, and `claude:var_l_shape_5`; clean two-seed
  all-horizon negatives are `claude:arrested_spiral`,
  `claude:cal_hexagon_6`, `claude:cal_octagon_8`, and `gated_local_linear`.
  The aggregate artifact is
  [wide_periodic_reeval_aggregate.md](/home/mila/l/lia/skae/results/staged_fabs_local_affine_k_lista_table1_seed2_20260512/wide_periodic_reeval_full/wide_periodic_reeval_aggregate.md).
  A follow-up artifact audit found `144` trained local maps total, `4` to `6`
  per run, from support families fitted on `4096` generated training
  trajectories and `36864` latent states per run. Clean negatives are not
  simply under-fragmented; their global periodic baseline is usually already
  near the error floor. The diagnostic artifact is
  [local_k_diagnostics_summary.md](/home/mila/l/lia/skae/results/staged_fabs_local_affine_k_lista_table1_seed2_20260512/local_k_diagnostics/local_k_diagnostics_summary.md).
  The next tuning probe completed but is not yet decisive evidence:
  learned-intercept parameterization compile gate `9553697`, contrast array
  `9553698`, and dependent re-evaluation `9553699` all exited `0:0`. The
  completed re-evaluation is H100/periodic-1 only, not the intended
  H100/H500/H1000 wide-periodic grid. It gives `5/8` paired wins versus
  matched same-budget global-\(K\) LISTA, with strong `transition_routes_4`
  wins but two `cal_pentagon_5` losses, so it is not a scale-up result. A
  follow-up \(C_{\rm stab}\)-routed learned-intercept contrast completed:
  compile gate `9562611`, launcher `9562654`, array `9562821`, collector
  `9562822`, compare job `9562823`, and wide-periodic re-evaluation job
  `9562824` all exited `0:0`. On the corrected `H100/H500/H1000` wide-periodic
  grid, it wins `7/8` paired seed rows at every horizon, with geometric mean
  staged/global ratios `0.271/0.431/0.453`. The system-level read is weaker:
  `3/4` systems win by seed-geometric ratio, while `cal_hexagon_6` loses
  because seed `0` has a very strong global-\(K\) periodic baseline. This is
  promising tuning evidence but not yet statistically decisive by itself. The
  full retained multibasin 15-seed run then completed as
  `9565762 -> 9565763 -> 9565764 -> 9565765`: training finished `225/225`
  rows, collection/comparison/wide-periodic re-evaluation all exited `0:0`,
  and the matched staged/global wide-periodic table has `225/225` `status=ok`
  rows. Staged local maps win `189/225`, `188/225`, and `188/225` paired rows
  at `H100/H500/H1000`, respectively, and `176/225` paired rows on all three
  horizons; geometric mean staged/global ratios are `0.269/0.202/0.182`.
  By system-level seed-geometric ratio, staged wins `14/15` systems at each
  horizon and on all horizons, with `duffing_triple_well` the only exception.
  Compact aggregate:
  [wide_periodic_reeval_aggregate.md](/home/mila/l/lia/skae/results/staged_cstab_learned_intercept_k_lista_full_20260516/wide_periodic_reeval/wide_periodic_reeval_aggregate.md).
  The same \(C_{\rm stab}\)-routed learned-intercept recipe has produced a
  negative retained-`10` Dysts `dt x30` read: training completed `150/150`, but
  Dysts evaluation collected only `106/150` staged rows due failed/OOM eval
  packs; on those complete staged/global pairs, staged wins `0/106` at
  `H100/H500/H1000` and `15/106` at `H5000`, with all-horizon wins `0/106`.
- May 14 stable support component post-hoc diagnostic: the new protocol in
  [SUPPORT_FAMILY_LOCAL_KC_TRAINING_PROTOCOL.md](/home/mila/l/lia/skae/docs/SUPPORT_FAMILY_LOCAL_KC_TRAINING_PROTOCOL.md)
  defines a label-free stable support component \(C_{\rm stab}\) by support
  transition fate rather than instantaneous Jaccard overlap alone. The first
  evaluator,
  [evaluate_stable_support_components.py](/home/mila/l/lia/skae/tools/evaluate_stable_support_components.py),
  compiled under SLURM job `9554707` and smoke-tested on existing p256 LISTA
  hard-init checkpoints under job `9554713`; the four-system/two-seed contrast
  completed under job `9554718` with `48` rows and `0` failures. On the
  per-basin deep slice, \(C_{\rm stab}\) gives coverage `1.0`,
  \(H(B\mid C)=0\), \(H(C\mid B)=0\), and basin-matched object counts on all
  `8/8` contrast pairs. Current \(F_{\rm abs}\) does so on `4/8`, because it
  merges `cal_hexagon_6` and `cal_octagon_8` basins. On all states,
  \(C_{\rm stab}\) improves mean \(H(B\mid\cdot)\) from `0.420` to `0.221`
  and mean \(H(\cdot\mid B)\) from `0.486` to `0.384` at mean coverage
  `0.959`. The local one-step affine diagnostic is not positive:
  \(C_{\rm stab}\) is never better than global on the deep slice (`0/8` pairs)
  and improves over current \(F_{\rm abs}\) on only `3/8` pairs. Treat this as
  strong basin-support-object evidence and a recipe for future training, not as
  paper-facing local-\(K_c\) evidence. A matched May 15 encoder comparison
  completed four post-hoc roots with `0` failures: LISTA dense soft-block p256
  `9559715`, LISTA blockdiag `9559716`, sparse MLP `9559718`, and dense
  zero-sparsity MLP `9559717`. \(C_{\rm stab}\) separates sparse encoders from
  dense MLP, but not LISTA from sparse MLP on this contrast: both LISTA roots
  and sparse MLP reach deep-slice \(H(B\mid C)=H(C\mid B)=0\), NMI `1.0`,
  and `8/8` count matches, while dense MLP collapses to one component with
  \(H(B\mid C)=1.685\), NMI `0`, and `0/8` count matches.
- May 15 retained-15 \(C_{\rm stab}\) and dense latent-fate control:
  sparse-root array `9562395` completed `150/150` shards with `0` failures,
  and dense continuous latent-fate array `9562516` completed `30/30` shards
  with `0` failures. On the retained `15` systems and seeds `0,1`,
  \(C_{\rm stab}\) is strong but not perfect on per-basin deep states:
  perfect matches are LISTA dense p256 `26/30`, LISTA soft-block p256 `25/30`,
  LISTA-BD `26/30`, sparse MLP `27/30`, and repaired sparse MLP-BD `24/30`.
  Common misses are overfragmentation on `duffing_triple_well` and
  `gated_transfer_linear`. The dense zero-sparsity MLP still has collapsed
  support under \(C_{\rm stab}\), but a separate continuous latent-tail fate
  control is strong: unsupervised KMeans gives deep \(H(B\mid{\rm fate})=0\),
  \(H({\rm fate}\mid B)=0.182\), NMI `0.927`, and `24/30` count matches;
  oracle-\(k\) KMeans gives deep \(H(B\mid{\rm fate})=0.173\),
  \(H({\rm fate}\mid B)=0.006\), NMI `0.871`, and `16/30` count matches. This
  supports a narrower paper claim: sparse encoders expose basin fate as an
  inspectable support-flow quotient, while dense MLP can still contain basin
  fate in continuous amplitudes that require external clustering.
- May 12 additional baseline coverage completion: corrected standalone
  DMD/polynomial EDMD/RBF-dictionary EDMD and fixed-count mixture/local-linear
  baselines completed an off-target nine-system smoke/provisional packet under
  [paper_baseline_suite_20260512_corrected](/home/mila/l/lia/skae/results/paper_baseline_suite_20260512_corrected)
  after validation `9526657` and smoke `9526658`; launcher `9526668` submitted
  array `9526673`, and all `54/54` tasks completed with exit `0:0` and no
  `status=error` rows. These H50 aggregates are implementation validation, not
  manuscript baseline evidence for the retained `15`-system benchmark in
  [neurips_sparse_koopman_multibasin.tex](/home/mila/l/lia/skae/docs/neurips_sparse_koopman_multibasin.tex).
  The standalone baseline tools and launcher defaults now target the retained
  `15` multibasin systems at `H100/H500/H1000` for seeds `0,1,2`; retained-15
  launcher job `9527268` completed and submitted baseline array `9527269`
  (`90` tasks, `0-89%32`), which completed `90/90` tasks with exit `0:0`,
  `810` `status=ok` rows, and no nonempty stderr logs. RBF-dictionary EDMD is
  the strongest standalone retained-15 row at `H100/H500/H1000`
  (`0.660/0.808/0.751` raw cumulative MSE), ahead of polynomial EDMD
  (`0.864/0.968/0.887`) and DMD (`2.58/2.14/1.70`). It beats the dense-latent
  MLP KAE baseline (`0.830/2.68/2.93`) but not any sparse KAE row; the best
  sparse rows remain `0.0387/0.0940/0.107`. The Dysts `dt x30` standalone
  extension is also complete: array `9530093`
  completed `59/60` tasks, repair task `9530233_48` completed the remaining
  Sakarya mixture task, and the final output has `540` `status=ok` rows and `0`
  `status=error` rows. RBF-dictionary EDMD is the best Dysts standalone row
  (`0.890/2.84/2.91` at `H100/H2000/H4000`) but is worse than Dense MLP
  (`0.00110/0.224/0.754`) and all sparse KAE rows. The manuscript now includes
  these controls directly in the forecasting-only Table 1, while support
  diagnostics are displayed in a separate main-text support table.
  June 23 extension: the same standalone table now includes label-free local
  polynomial EDMD and local RBF-EDMD. These route by k-means on states and
  select `k` by validation rollouts rather than benchmark basin counts. Local
  polynomial EDMD is substantially stronger than the earlier standalone
  controls but still does not beat the sparse KAE rows on retained
  multibasin; quantitatively it is `3.7x/1.8x/1.7x` worse than LISTA and
  `3.9x/2.7x/2.6x` worse than the best sparse KAE row at H100/H500/H1000.
  On Dysts, it is strong at H100 but `22x/6.8x` worse than the best sparse KAE
  row at H2000/H4000. Local RBF-EDMD is retained as an unstable variant with
  visible blow-up outliers.
  The p256 checkpoint-backed
  clustering/oracle local-K packet under
  [paper_regime_oracle_baselines_20260512](/home/mila/l/lia/skae/results/paper_regime_oracle_baselines_20260512)
  is merged (`51/51` runs, `1377` rows, `0` run-level failures), but should be
  filtered to the retained paper roster before paper-facing aggregation. Treat
  `gmm_diag` route rows cautiously because `43` row-local clustering
  assignments are marked failed.
- May 12 HyperLISTA method-side note: d_z=256 interactive paired runs found a
  replicated HyperLISTA substitution positive for autonomous/no-reencode
  stability, but not for reset-corrected forecasting. HyperLISTA improves
  no-reencode rollouts on `cal_octagon_8`, `cal_square_4`, `cal_hexagon_6`,
  `cal_pentagon_5`, `cal_high_cross_3`, and `var_diamond_4`; on
  `snic_multi` and `transition_routes_4`, the fair sequence-8 comparison loses
  `H100` no-reencode but wins later autonomous horizons because LISTA
  diverges. A sequence-32 HyperLISTA attempt was explicitly checked against a
  matched sequence-32 LISTA baseline and is not a win
  (`0.149/6.53e3/4.95e9` LISTA versus `0.565/3.76e4/5.98e10` HyperLISTA at
  `H100/H500/H1000`). LISTA still wins every-step/best-reset on all checked
  systems, so this remains a method diagnostic rather than a paper-facing
  near-universal encoder claim. The first all-system queue with the unfair
  sequence-32 arm was canceled; corrected fair sequence-8 launcher `9529830`
  submitted training array `9529842` plus collect job `9529843`, but those jobs
  were also canceled while the interactive fair-comparison search continued.
  No broad HyperLISTA queue is active. A follow-up removed the duplicate
  base-loss `1/horizon` scaling and tested higher HyperLISTA \(K\) learning
  rates; this did not fix the reset/predictive deficit. The transition
  `k_matrix_lr=1e-5` probe was effectively unchanged, while `snic_multi`
  with `k_matrix_lr=5e-5` lost late autonomous stability.
- May 12 decoder-structure fairness diagnostic completed: `GenericKM` now exposes
  opt-in normalized linear decoder atoms, matching the dictionary-normalization
  constraint used by LISTA-style decoders. The motivation is to test whether
  unconstrained MLP decoder atom norms were a confound in the Sparse MLP and
  Sparse MLP-BD rows by allowing decoder scale to trade off against L1
  coefficient scale. Focused SLURM test job `9530385` passed. Launcher
  `9530394` queued a one-seed retained-10 Dysts `dt x30` pilot for Sparse MLP
  and Sparse MLP-BD with `--normalize_decoder_atoms true`; cache array
  `9530395`, GPU training array `9530396`, long-horizon queue `9530397`, eval
  cache `9530746`, validation `9530747`, eval array `9530748`, and collector
  `9530749` all completed with exit `0:0`. The seed-`0` arithmetic-mean
  best-periodic values at `H100/H2000/H4000/H5000` are `5.20e-4/0.182/0.619/0.962`
  for Sparse MLP norm-dec and `5.71e-4/0.125/0.624/0.952` for Sparse MLP-BD
  norm-dec. Normalization improves plain Sparse MLP versus its matched seed-`0`
  old row but worsens repaired Sparse MLP-BD versus its matched seed-`0`
  repaired row, so this diagnostic alone does not change Table 1. The later
  May 14 full-row `normdec_rollout` queue is the only candidate for replacing
  the current table rows.
- May 6/12 paper-reorganization pass: the main manuscript has been rewritten
  around basin identifiability. Table 1 is now a forecasting-only table covering
  all displayed horizons for the 15 controlled multibasin systems and the 10
  Dysts \(dt{\times}30\) systems, including standalone state-space/local-linear
  controls. Basin-support diagnostics are now shown in their own main-text
  table, and the support-refresh appendix table remains the periodic
  \(F_{\rm top8}\) support-family refresh display. The
  \(F_{\rm top8}\)-local predictor/routing table and method are moved to a new
  appendix file, `docs/appendix/support_conditioned_predictors.tex`, and are
  not referenced from the main text.
- May 6 Sparse MLP-BD blocker: the old `GenericKM` implementation ignored
  `MODEL.K_STRUCTURE`, so `generic_sparse --k_structure block_diagonal` trained
  a dense transition. `skae/model.py` is now patched to create diagonal and
  block-diagonal MLP Koopman parameters. The retained-10 Dysts Sparse MLP-BD
  row and the controlled multibasin Sparse MLP-BD row have now both been rerun
  from the repaired code path.
- May 7/12 repaired Sparse MLP-BD result: the retained-10 Dysts `dt x30`
  packet completed all `150/150` system-seed rows under
  [results/dysts_dt30_sparse_mlp_bd_repaired_20260506](/home/mila/l/lia/skae/results/dysts_dt30_sparse_mlp_bd_repaired_20260506).
  The main Dysts Sparse MLP-BD cells are now `3.96e-4`, `0.113`, and `0.436`
  at `H100/H2000/H4000`; `H2000` and `H4000` carry exact system-sign/Holm
  `\ast` superscripts, and `H4000` is the best displayed Dysts cell.
  Controlled multibasin forecasting plus per-basin deep support diagnostics
  also completed under
  [results/transition_rich_sparse_mlp_bd_repaired_table1_20260506](/home/mila/l/lia/skae/results/transition_rich_sparse_mlp_bd_repaired_table1_20260506)
  with `225/225` forecasting rows, `6075` support rows, and `0` support
  failures. Rescue array `9526863` and collection/support chain
  `9526864 -> 9526865 -> 9526866` all completed with exit `0:0`. Table rebuild
  job `9529647` regenerated the compact Table 1 and support-diagnostic table
  sources. The displayed controlled Sparse MLP-BD cells are `0.0473`, `0.136`,
  and `0.159` at `H100/H500/H1000`, with support cells `0.358` for
  `H(B|F_abs)` and `3.0` for `|F_abs|`.
- May 6 support-refresh replacement completed: because support-conditioned
  rollouts are now appendix-only, the periodic refresh mechanism test has been
  replaced by a controlled-transfer support-switch audit that uses only the
  learned global \(K\). The seed-matched packet under
  [results/controlled_support_refresh_table1_seed15_20260506](/home/mila/l/lia/skae/results/controlled_support_refresh_table1_seed15_20260506)
  covers periods `1/10`, and the matched additive packet under
  [results/controlled_support_refresh_table1_seed15_periods5_20_20260506](/home/mila/l/lia/skae/results/controlled_support_refresh_table1_seed15_periods5_20_20260506)
  covers periods `5/20`. Both completed with `1,345/1,345` specs,
  `285,804` merged rows, and `0` failures. The paper-facing result is the
  \(F_{\rm top8}\) support-family read: sparse-family after-reencoding target
  rates are `0.994`--`1.000` across periods `1/5/10/20`, while no-transfer
  false target-family rates remain `0.009`--`0.019`. Table 2b now shows each
  period as stale pre-refresh family \(\to\) refreshed family. Exact top-`8`
  supports are overfragmented and should stay diagnostic.
- May 6 protocol-clarity pass: the manuscript now distinguishes the broad
  boundary-emphasized multibasin training distribution from the evaluation-only
  per-basin deep-state slice used for Table 1 support diagnostics. Table 1 now
  places the H100/H1000 forecasting columns before the support diagnostics and
  clarifies that those forecasting columns are computed on all held-out
  rollouts, not only deep states. The main experimental procedure now also
  states the rationale for the train/evaluate split: broad training is the
  deployment-facing learning problem, while the per-basin deep slice is a clean
  basin-stratified diagnostic for static support--basin alignment.
- May 6 per-basin deep-slice Table 1 replacement: the current-roster rerun
  completed cleanly. Shards `9477837`--`9477840`, `9477842`, and `9477844` and
  merges `9477841`, `9477843`, and `9477845` all completed with exit `0:0`;
  merged outputs contain `27,351`, `6,885`, and `6,885` data rows with `0`
  failures. Compute job `9479694` rebuilt the manuscript Table 1 source using
  all-held-out forecasting columns plus per-basin-deep support diagnostics.
  The key support values are \(H(B\mid F_{\rm abs})=0.130/0.156/0.153\) for
  LISTA/LISTA-BD/LISTA-SB versus Dense MLP `1.28`, with LISTA-family counts
  `4.0/3.8/3.8` versus Dense `1.0`.
- May 6 \(F_{\rm abs}\) versus exact \(S_{\rm abs}\) display: SLURM job
  `9482111` generated
  [fig_fabs_vs_sabs_basin_identification.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_fabs_vs_sabs_basin_identification.pdf)
  and
  [fig_fabs_vs_sabs_utility_tradeoff.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_fabs_vs_sabs_utility_tradeoff.pdf)
  from the same current-roster per-basin deep-slice rows. The result supports
  the current paper positioning: exact \(S_{\rm abs}\) masks are basin-pure on
  the clean slice (`H(B|S_abs)=0` for sparse rows) but overfragment basins
  (`H(S_abs|B)` about `0.89`--`1.12`, mean counts about `459`--`659`), whereas
  \(F_{\rm abs}\) gives a compact basin-scale support object with low
  uncertainty (`H(B|F_abs)` about `0.13`--`0.26`) and counts near the retained
  benchmark basin scale (`3.3`--`4.0` vs `4.20`).
- May 6 initial-latent coordinate intervention ablation completed: a focused
  single-checkpoint support-necessity test ran as replacement GPU SLURM job
  `9487431` under
  [results/support_coordinate_interventions_20260506/gated_local_linear_lista_seed0](/home/mila/l/lia/skae/results/support_coordinate_interventions_20260506/gated_local_linear_lista_seed0).
  The first CPU submission `9487332` and earlier GPU submission `9487372` were
  canceled while pending; the `main` job requested one GPU, `1` CPU, `8G`,
  and `03:00:00`, and completed in `27` seconds.
  It uses p256 dense LISTA, `gated_local_linear`, seed `0`, `15`
  per-basin-deep true-basin-stable starts, absolute support `S_abs=10^{-3}`,
  top-`1` through top-`10` coordinate dropping, and `20` random-support
  shuffles. A follow-up coordinate-dropping-only pass with `100` initial states
  writes
  [results/support_coordinate_interventions_20260506/gated_local_linear_lista_seed0_n100](/home/mila/l/lia/skae/results/support_coordinate_interventions_20260506/gated_local_linear_lista_seed0_n100).
  At horizon `21`, mean accumulated MSE is `0.0158` for the standard rollout
  and `0.508/1.37/2.08/3.26/8.51` after dropping top `1/2/3/5/10` active
  coordinates; the figure shows `95%` bootstrap confidence bands over initial
  states. A random-support-only rerun with `100` initial states and `20`
  shuffles per state is complete under
  [results/support_coordinate_interventions_20260506/gated_local_linear_lista_seed0_n100_random](/home/mila/l/lia/skae/results/support_coordinate_interventions_20260506/gated_local_linear_lista_seed0_n100_random).
  Across its `2000` point-shuffle outcomes, horizon-`21` accumulated MSE has
  mean `1.69e3`, median `874`, and IQR `376`--`2.06e3`. The compact H21
  table now includes mean `±` standard deviation, median with interquartile
  range, and paired sign-test \(p\)-values versus the standard rollout; all
  displayed interventions have \(p<10^{-16}\). Paper-style
  linear-axis absolute-MSE figures are now
  available as
  [fig_support_coordinate_dropping_accumulated_mse.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_support_coordinate_dropping_accumulated_mse.pdf)
  and
  [fig_support_coordinate_random_shuffle_accumulated_mse.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_support_coordinate_random_shuffle_accumulated_mse.pdf).
  Clipped-axis trajectory companions are available as
  [fig_support_coordinate_trajectories_drop_top10.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_support_coordinate_trajectories_drop_top10.pdf)
  and
  [fig_support_coordinate_trajectories_random_support_19.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_support_coordinate_trajectories_random_support_19.pdf);
  they set the plot window from the true/standard rollouts so bad red
  intervention paths can leave the chart.
  The main paper now includes the two accumulated-MSE panels plus the
  random-support trajectory panel in the functional-use results text, with
  prose connecting the intervention protocol back to the support-identity
  claim. Treat it as a representative mechanistic display supporting the
  cross-system wrong-support table; broaden across more systems only if
  coauthors want it elevated beyond representative evidence.
- May 6 support-family value schematic:
  [fig_support_family_value_diagram.svg](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_support_family_value_diagram.svg)
  has been added as a coauthor-facing explanation of what support families are
  good for. It reuses the current alluvial example's concrete counts
  (`393` exact \(S_{\rm abs}\) masks to `3` \(F_{\rm abs}\) families on
  `4,656` deep states) and frames the value as stable basin-scale identity,
  an inspectable active-coordinate codebook, and a route/cache key.
- May 6 spatial alluvial prototype pass: three layout prototypes now live under
  [spatial_alluvial_prototypes](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/spatial_alluvial_prototypes),
  and the data-driven renderer is
  [tools/make_spatial_alluvial_prototypes.py](/home/mila/l/lia/skae/tools/make_spatial_alluvial_prototypes.py).
  These address the missing spatial context in the current alluvial by testing
  a triptych, a spatial-source-column alluvial, and a spatial callout layout.
  They are display prototypes from existing assets, not new evidence.
- May 6 \(F_{\rm abs}\) router-comparison result: the matching stage-2
  support-family-local replay under
  [results/routed_stage2_local_maps_20260506](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506)
  used `support_definition=absolute:0.001`, `J=0.40`, period `5`,
  `reroute_each_step`, `50000` steps, one GPU per worker, `DEVICE=cuda`, and
  worker partition `long`. It is failed/negative under the current recipe.
  Controlled multibasin is not aggregate-usable because `149/150` workers OOMed
  and only `claude:snic_multi` seed `6` completed. Dysts produced a partial
  `96/100` aggregate with route coverage `0.2345`, fallback `0.7655`, `0/10`
  wins versus best-periodic at every horizon, and routed/best-periodic ratios
  `6.39e3` at `H100`, `5.37e27` at `H500`, `5.22e25` at `H1000`, and `6.74e4`
  at `H5000`.
- May 5 Methods visual update:
  [fig_methods_support_family_pipeline.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_methods_support_family_pipeline.pdf)
  is now wired into `docs/neurips_sparse_koopman_multibasin.tex` as the
  non-floating Methods schematic. It replaces the older sectorized overview
  with a two-panel route-and-construction diagram: the top panel shows
  label-free support extraction, evaluation-only \(F_{\rm abs}\), and
  \(F_{\rm top8}\) routing; the lower panel shows greedy Jaccard
  support-family construction without basin labels or basin counts. This is a
  display artifact, not a new experiment result.
- May 6 appendix training-dynamics update:
  [appfig_training_dynamics_gated_local_seed0.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/appfig_training_dynamics_gated_local_seed0.pdf)
  and
  [appfig_training_dynamics_dysts_dt30_seed0.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/appfig_training_dynamics_dysts_dt30_seed0.pdf)
  are now wired through
  [appendix/training_dynamics.tex](/home/mila/l/lia/skae/docs/appendix/training_dynamics.tex).
  The first is a representative `gated_local_linear`, seed-`0` diagnostic for
  the six controlled model rows. The second aggregates seed-`0` traces across
  the retained `10` Dysts `dt x30` systems by the median over finite
  per-system metric values at each step. Together they plot validation final
  error, training objective, \(\rho(K)\), and the logged sparsity-ratio
  diagnostic, supporting the appendix discussion of optimization/checkpoint
  progression and the qualitative spectral-radius observation. They are not
  new aggregate benchmark rankings and do not change the main evidence order.
- May 4 manuscript-editing pass: visible `\todo{}` notes in
  `docs/neurips_sparse_koopman_multibasin.tex` have been resolved. The pass
  tightened the introduction/control framing, added the method overview
  diagram, compressed the methods/experimental procedure, replaced related
  work and conclusion placeholders with prose, and added an appendix
  Dysts ratio-to-Dense table as the scale-normalized companion to the main
  Dysts actual-MSE table. No new experiment results were introduced.
- Keep the main paper evidence in this order:
  1. forecasting performance on the controlled multibasin and Dysts benchmarks;
  2. support-family agreement with basin labels, with labels used only for
     evaluation;
  3. wrong-support interventions showing functional dependence on active
     coordinates;
  4. periodic support refresh after controlled basin transfer.
- Paper-critical router comparison status: the \(F_{\rm abs}\) replay is
  failed/negative against the existing \(F_{\rm top8}\) fixed-setting results.
  It supports keeping `topk:8` as the practical stage-2 routing object, but the
  paper should still phrase this carefully: the Dysts comparison is negative
  for \(F_{\rm abs}\), while the controlled \(F_{\rm abs}\) replay is an OOM
  operational failure rather than a completed controlled head-to-head.
- May 6 reviewer-scope add-ons partially landed: the requested oracle/local-K,
  explicit regime-discovery, and out-of-generator multistability checks are now
  tracked and partly complete. Existing compact partition-control artifacts already include
  learned-support, latent-k-means, and oracle-basin local-\(K\) controls for
  the Table 2 compact packet, and a plain p256 LISTA add-on is still running under
  [results/oracle_vs_learned_local_koopman_20260506](/home/mila/l/lia/skae/results/oracle_vs_learned_local_koopman_20260506)
  because shard `9478599` remains running and merge `9478603` is held. The new
  explicit regime-discovery evaluator is complete under
  [results/regime_discovery_local_koopman_20260506](/home/mila/l/lia/skae/results/regime_discovery_local_koopman_20260506)
  with `6,885` rows and `0` failures. Its result is cautionary: oracle basin
  local-K is strong (`latent/global=0.2174`), evaluation-only basin-count
  clustering is also strong (`~0.30` best rows), but learned support-family
  local-K is near global (`0.9827`). The supplemental out-of-generator packet
  has completed training/collect, interpretability, and one-step regime
  discovery under
  [results/out_of_generator_multistable_p256_lista_20260506](/home/mila/l/lia/skae/results/out_of_generator_multistable_p256_lista_20260506)
  for a gene toggle, thermal reactor, modified FitzHugh-Nagumo, and buckled
  beam. Forecasting and deep-slice support purity are positive, but one-step
  local-K routing is negative; learned support-family, oracle-basin, and
  clustering partitions are all worse than global \(K\). The OOD autonomous
  oracle/local-K job `9478621 -> 9478622` has now completed with `120` rows
  and `0` failures; it is also negative for local-K routing, with
  family-local and oracle-basin rollouts worse than global at H100 and
  unstable at longer horizons. This supports appendix/scope framing rather
  than a main-text generality claim.
- May 6 support-family local-\(K\) follow-up queued: because the p256
  regime-discovery row leaves a large gap between learned support-family
  (`0.9827`) and oracle basin (`0.2174`) local-\(K\), the active follow-up is
  to tune the label-free support-family construction before drawing a final
  conclusion. The first iteration is under
  [results/regime_support_family_hparam_p256_20260506_iter1](/home/mila/l/lia/skae/results/regime_support_family_hparam_p256_20260506_iter1):
  worker jobs `9479775`--`9479851` sweep support definitions
  `topk:4/6/8/12/16`, absolute and relative thresholds, and Jaccard thresholds
  `0.20`--`0.80`, with summary job `9479852`. This is label-free tuning of
  the support-family route; basin labels remain evaluation-only. A partial
  `09:16 EDT` diagnostic is promising but not final: `absolute:0.001`, `J=0.8`
  is much closer to oracle on the incomplete row set. Conditional refinement
  launcher `9480079` is held behind `9479852` to sweep high-J
  absolute/relative support settings over min transitions and ridge.
- Build the main displays in the current order: benchmark/support visual,
  forecasting-only Table 1 with standalone baselines, the separate support
  diagnostics table, basin-support alignment figure, and forecasting horizon
  trends. Treat the support-routed local predictor and support-refresh tables as
  appendix-only diagnostic material.
- Figure 1 has been replaced by the paper-facing composite
  [fig_benchmark_support_dysts_composite.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_benchmark_support_dysts_composite.pdf):
  top row shows controlled multibasin vector fields, evaluation basin maps,
  and LISTA `topk:8` support-family overlays; bottom row shows H5000 Dysts
  `dt x30` phase portraits from the best seed-`0` primary model. The current
  polished version is a 2-by-4 display: multibasin panels are
  `gated_local_linear`, `claude:transition_routes_4`, `claude:cal_square_4`,
  and `claude:cal_high_cross_3`; Dysts panels are Chua, Dadras,
  Shimizu-Morioka, and Lu-Chen-Cheng. It removes in-panel agreement
  annotations, uses the concise row label `Support-basin alignment`, adds an
  external support/basin match cue with a slightly larger legend font, and
  slightly zooms Dadras. The added `claude:cal_high_cross_3` panel is the
  highest visual grid-agreement candidate from the screen (`0.902`).
- May 5 visual companion update:
  [support_family_index_codebook_claude_cal_asymmetric_3.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/support_family_codebooks_retained15/support_family_index_codebook_claude_cal_asymmetric_3.pdf)
  is now the manuscript support-codebook figure. It shows the actual active
  latent coordinate indices for `d_z=256` LISTA-BD `topk:8` support-family
  prototypes on the three-basin Asymmetric wells system. The one-panel layout
  was chosen after inspecting retained-system candidates; the old right-side
  active-index list column was removed so thin tick position on the shared
  `0..255` axis carries the coordinate-index information directly. Row labels
  and tick opacity encode within-basin family coverage. The accompanying JSON
  records all displayed and non-displayed family prototype index sets. This is
  a display artifact from existing checkpoints, not a new experiment result.
- A retained-`15` screening batch of the same active-index codebook has been
  rendered under
  [support_family_codebooks_retained15](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/support_family_codebooks_retained15),
  with one LISTA/LISTA-BD figure per retained controlled multibasin system,
  a manifest, and a contact sheet. Treat this as an appendix/replacement
  selection pool; the Asymmetric wells triplet has been overwritten by the
  one-panel LISTA-BD main-text figure.
- An exploratory \(F_{\rm abs}\) retained-`15` screening batch is also available
  under
  [support_family_codebooks_retained15_fabs](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/support_family_codebooks_retained15_fabs).
  It uses the absolute support rule `absolute:0.001` for both LISTA and
  LISTA-BD. The panels are much denser than the \(F_{\rm top8}\) codebooks and
  should be treated as visual diagnostics unless the paper narrative shifts
  toward absolute-threshold support structure.
- A global-deep-slice \(F_{\rm abs}\) version is available under
  [support_family_codebooks_retained15_fabs_deep](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/support_family_codebooks_retained15_fabs_deep).
  It uses `128 x 128` generated observation states, eval seed `42`,
  `absolute:0.001`, and the global top-quartile basin-margin deep slice. This
  is the closer visual analogue of the Table 1 \(F_{\rm abs}\) slice, but
  several systems represent only one or two basins under the global deep
  criterion.
- May 5 p256 visual-draft update: the paper-facing Support Barcode Map and
  Alluvial Basin-to-Support-to-Family candidates have been regenerated from
  p256 LISTA-family checkpoints, superseding the earlier p64 drafts. The
  barcode draft is
  [fig_support_barcode_map_p256.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_support_barcode_map_p256.pdf),
  using p256 dense LISTA on `gated_local_linear`; it overlays the displayed
  basin-family prototypes on one shared `0..255` latent-coordinate barcode.
  The alluvial draft is
  [fig_basin_support_family_alluvial_p256_gated_deep_unbold_headers.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_basin_support_family_alluvial_p256_gated_deep_unbold_headers.pdf),
  using p256 dense LISTA on `gated_local_linear`, seed `0`, the global
  deep-state slice, `S_abs=10^{-3}`, and family Jaccard threshold `0.5`; it
  shows `3` represented evaluation basins, `3` support families, `393` exact
  supports, and family-dominant basin agreement `1.0`. It keeps the original
  alluvial geometry and changes the three column headers to normal weight. The
  compact polished and all-state `J=0.32` versions remain sensitivity/alternate
  artifacts, but this unbold-header deep-slice version is the current
  paper-facing alluvial candidate.
- May 6 spatial alluvial alternatives: use
  [prototype_a_triptych_composite.svg](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/spatial_alluvial_prototypes/prototype_a_triptych_composite.svg),
  [prototype_b_spatial_source_column.svg](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/spatial_alluvial_prototypes/prototype_b_spatial_source_column.svg),
  and
  [prototype_c_family_callouts.svg](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/spatial_alluvial_prototypes/prototype_c_family_callouts.svg)
  to decide how much spatial context should be folded into the alluvial. The
  implementation target is a matched data-driven PDF/PNG render from the same
  selected state-space grid points once a compute slot is available.
- May 5 Jaccard-threshold sensitivity update: the threshold sweep under
  [full_retained15_deep_subsetfit](/home/mila/l/lia/skae/results/support_family_jaccard_threshold_sweep_20260505/full_retained15_deep_subsetfit)
  confirms that the support-family Jaccard threshold is a major
  interpretability-resolution parameter. Low thresholds merge basins; high
  thresholds can drive `H(B|F)` to zero by overfragmenting the representation.
  At the fixed paper convention `J=0.5`, `F_abs` gives basin-scale sparse-row
  family counts (`3.17/3.03/3.05` for LISTA/LISTA-BD/LISTA-SB) with low
  `H(B|F_abs)` (`0.0385/0.0380/0.0396`), while Dense MLP remains one family
  with `H(B|F_abs)=0.7648`. For `F_top8`, very high thresholds can make Dense
  MLP appear pure as well (`H(B|F_top8)=0` at `J>=0.8`) by splitting it into
  `13.8` families. The paper should therefore keep `J=0.5` fixed and always
  read entropy together with family count; do not tune Jaccard threshold for
  best entropy.
- May 5 routed-forecasting-MSE add-on: the plan is
  [ROUTED_FORECASTING_MSE_PLAN.md](/home/mila/l/lia/skae/docs/ROUTED_FORECASTING_MSE_PLAN.md).
  The raw evaluation completed after two passing smoke jobs, and aggregation
  job `9467050` completed the seed-IQM-over-seeds and mean-over-systems
  comparison against the current Table 1/Table 3 best-periodic rows. The
  merged outputs have complete seed coverage (`675` controlled rows, `450`
  Dysts rows, `0` failures). The result is negative for paper-table
  forecasting: routed `F_top8` support-family centered local rollouts win
  `0/15` controlled systems and `0/10` Dysts systems against the same-model
  best-periodic baseline for every LISTA-family model at every evaluated
  horizon. Do not add these routed rows to Table 1 or Table 3 as superior
  forecasting evidence.
- May 5 periodic routed-forecasting rerun: the stronger route-refresh test
  requested by the user is complete for periods `5,10,20,30`. Compile job
  `9467340`, smoke jobs `9467341`/`9467342`, controlled shards
  `9467343`--`9467477` with merge `9467478`, Dysts shards
  `9467479`--`9467568` with merge `9467569`, and aggregation job `9467598`
  completed with `0` recorded failures. The merged raw outputs contain `2700`
  controlled rows and `1800` Dysts rows. Periodic decode/re-encode improves
  stability relative to the fully autonomous routed rollout, but every
  aggregate routed/best-periodic ratio remains above `1`; only isolated cells
  have any system wins. This keeps post-hoc \(F_{\rm top8}\)-routed local maps
  as mechanism/falsification evidence rather than a replacement for the
  best-periodic forecasting rows.
- May 6 stage-2 fixed-setting expansion results: the `F_top8`, `J=0.40`,
  period-`5`, `reroute_each_step` support-family-local \(K_c\) expansion has
  now completed the 10-seed `50000`-step controlled and Dysts passes. The
  controlled result is mixed against the same-root best-periodic sparse-LISTA
  rollout: `0` failures, route coverage `0.9982`, and routed/best-periodic
  ratios `1.70/0.91/0.84` at `H100/H500/H1000`, with system wins `4/15`,
  `5/15`, and `5/15`. Against the current hard-init Dense MLP Table 1
  comparator, however, it is paper-positive: on the same `15` systems and
  common seed slots `0`--`9`, stage-2/Dense ratios are
  `0.0866/0.0513/0.0490`, `14/15` systems have lower per-system seed-IQM MSE,
  and system-level one-sided Wilcoxon p-values Holm-correct to `9.16e-4`
  across the three displayed horizons. The Dysts
  50k result is strongly negative: `0` failures but `0/10` system wins and
  routed/best-periodic ratios already `6.9e3` at `H100`. The 100k controlled
  continuation completed but is worse than 50k, and the Dysts 100k continuation
  has `97/100` successful rows with three local-map class-count failures and
  unstable available metrics. The controlled row is now included in Table 1 as
  `LISTA + local \(K_c\)`, with forecasting values only and support diagnostics
  marked as inherited from the frozen LISTA encoder/support representation; do
  not present it as beating the best periodic sparse-LISTA rollout or as
  externally robust until Dysts improves. A follow-up no-refresh controlled
  rollout over the same 50k period-`5` trained \(K_c\) checkpoints completed
  with `0` failures and is strongly negative: routed/best-periodic ratios are
  `5.26e8/3.84e33/4.80e34`, system wins are `0/15` at all three horizons,
  route coverage falls to `0.2161`, and aggregate routed/no-reencoding-global
  ratios are `2.12e5/1.23e9/2.31`. Therefore Table 1's local-\(K_c\) row
  should be understood as a periodically refreshed local-map rollout, not as a
  stable autonomous local-map forecaster.
- May 6 Dysts stage-2 period-sensitivity queue: because the Dysts 50k
  support-family-local result above used only re-encode period `5`, the
  missing periods `{1,2,10,20}` have been queued and will be combined with the
  completed period-`5` outputs to evaluate `{1,2,5,10,20}`. Seed batches
  `0`--`4` and `5`--`9` use merge jobs `9478074` and `9478276`; combined
  analysis job `9478278` will write
  [combined_best_lista_dysts_j040_50k_period_sweep_labelnone_seed0_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/combined_best_lista_dysts_j040_50k_period_sweep_labelnone_seed0_9).
  The controlled multibasin result is competitive enough for an appendix
  ablation discussion at `H500/H1000`, but it should not enter the main table
  unless the Dysts robustness check changes the external-stress-test readout.
- May 6 calibrated-global stage-2 ablation results: the matched
  `stage2_map_mode=global_dense_calibrated` 50k raw-row batches completed for
  controlled and Dysts with `0` failures and perfect route coverage, but they
  are unstable rather than a positive calibration control. Controlled simple
  means are `H100/H500/H1000 = 16.4/6.86e31/0.125`; Dysts simple means are
  catastrophic from `H100` onward. The combined readout jobs hit aggregation
  script issues, so raw rows and seed-half summaries are the current source.
- Whole-slice count alignment is not the same as the main deep-slice
  alignment result. At fixed `J=0.5`, the closest all-state `F_abs` count match
  is Sparse MLP-BD (`4.217` families versus `4.20` basins), followed by Sparse
  MLP (`4.244`), but their whole-slice `H(B|F_abs)` values are higher
  (`0.3373/0.3127`) than the LISTA-family rows. LISTA-family models create
  more all-state families because boundary/transition states fragment the
  support object, while the deep-state slice remains the cleaner
  basin-interior alignment test.
- Current manuscript display layout: the alluvial candidate has been removed
  from the main-text support-alignment display. A cropped seed-distribution
  panel for \(H(B\mid F_{\rm abs})\) only is wrapped on the right side of the
  body text. This display uses the current per-basin deep-slice Table 1 source
  rows. The \(|F_{\rm abs}|\) and wrong-support ratio strip panels have been
  removed from the main-text display, and the Asymmetric wells active-index
  codebook remains a separate full-width figure.
- Keep active supports, not pre-specified latent blocks, as the primary
  interpretability object. Do not reframe the paper around basin-block
  alignment or around training-time access to basin counts.
- State explicitly that support-label agreement is necessary but insufficient:
  a support may be an excellent basin label while the predictive dynamics are
  carried by different latent coordinates, continuous coefficient values, or
  cross-coordinate couplings in the learned Koopman transition.
- The main manuscript displays now use tests and summaries matched to each
  claim. The Dysts table reports actual MSEs and marks cells passing an exact
  system-sign test with Holm correction over per-system seed-IQM log-MSE
  differences; the older `[K/N]` within-system seed-paired reproducibility
  counts remain diagnostic artifacts rather than the main display. The current
  Table 2 combines basin-support diagnostics and controlled-transfer periodic
  support refresh. The support-diagnostics subtable suppresses `[K/N]` counts
  in favor of a single `*` superscript for significance, while the refreshed
  Table 2b reports stale pre-refresh \(\to\) refreshed \(F_{\rm top8}\)
  target-family rates over periods `1/5/10/20`, plus the no-transfer
  false-target control. Family-local routing and exact/gated `S_top8` routes remain
  appendix diagnostics. The
  completed jobs `9388212-9388218` are
  per-basin deep-slice interpretability outputs for Table 1 appendix
  robustness; they are not Table 2-4 seed packets.
- May 3 retained-benchmark aggregation update: at the user's direction,
  `multiwell_strong_transition` and `claude_checkerboard_potential` are now
  excluded from the controlled multibasin benchmark, leaving `15` retained
  systems. Main point estimates use seed-IQM within each retained system
  followed by an arithmetic mean across retained systems. Tables 1--3 have
  been regenerated under this estimator: every sparse-latent row beats Dense
  MLP on retained-system mean raw MSE at `H100`, `H500`, and `H1000`; Table 2
  routing denominators are `15` retained systems; and the support-refresh
  display has `12` eligible transfer systems. The Dysts `dt x30`
  aggregation has also been regenerated after excluding `dysts:LorenzCoupled`,
  the only six-dimensional member of the original Dysts shortlist, and
  `dysts:MultiChua`, to keep a nonredundant `10`-system three-dimensional
  shortlist. On the retained systems, all primary sparse rows beat Dense MLP
  in aggregate at every displayed horizon. Under the exact system-sign/Holm
  main-table rule, the current compact Dysts display uses
  `H100/H2000/H4000`; repaired Sparse MLP-BD clears `H2000` and `H4000`. The
  displayed aggregate-best rows are Sparse MLP at `H100`, LISTA at `H2000`,
  and repaired Sparse MLP-BD at `H4000`.
- The April 29 table-refresh pass has now landed for the manuscript tables.
  The fixed-`17` forecasting source has `1261` rows across `17` systems and
  `5` roots; LISTA-BD, Sparse MLP, and Dense MLP have full `15/15` coverage on
  every system, while LISTA-SB and Sparse MLP-BD each miss `7` system-seed
  cells. Table 1 interpretability has `34,047` rows and `0` failures; Table 2
  self-routed forecasting has `9,796` rows and `0` failures; the
  support-refresh packet has `189,708` rows and `0` failures; the Dysts packet
  collected
  `894/900` requested long-horizon rows, with the six missing rows confined to
  the LISTA-BD recipe.
- April 30 verification reran the active table aggregators on compute
  allocation `9411806`. The recomputed Table 1 artifacts and Tables 2-4
  paired-test JSON reproduce the manuscript-facing counts, so the remaining
  work is PDF/prose polish rather than additional paper-critical aggregation.
- Later on April 30, Tables 1--3 were revised to use the matched `d_z=256`
  LISTA-SB artifacts where available. The table builders and paired-test JSON
  were regenerated on compute allocation `9420481`. This was superseded on
  May 3 for Table 2: the main display now keeps only `F_top8` family-local
  routing at `H100` and `H1000`, reports system-level finite log-ratio effects,
  and uses exact sign-flip/Holm stars rather than `[K/15]` counts as the main
  inference.
- On May 2, the main Table 1 count diagnostic was changed from `|S_abs|` to
  arithmetic-mean `|F_abs|` because the family count is the paper-facing
  basin-scale compression diagnostic; `|S_abs|` remains an appendix
  active-coordinate-size diagnostic. The current retained-benchmark mean family
  counts are `3.2` for LISTA, `3.0` for LISTA-BD, `3.1` for LISTA-SB, `2.6`
  for both sparse MLP rows, and `1.0` for the dense no-shrink baseline. The
  manuscript now treats the represented basin count as the reference for
  interpreting this non-directional family-count column: retained-benchmark
  basin-count mean/median is `4.20/4`, and the global deep slice represents
  `3.00/3` basins on average/median.
  The current main support-diagnostic display keeps the count diagnostic in the
  table rather than the visual: it shows only the per-seed
  \(H(B\mid F_{\rm abs})\) strip wrapped beside the body text.
- The same-protocol Table 2 partition-control inference pass completed under
  [results/transition_rich_table2_controls_20260430/self_routed_controls](/home/mila/l/lia/skae/results/transition_rich_table2_controls_20260430/self_routed_controls).
  Jobs `9425249`--`9425263 -> 9425264` completed with exit `0:0`; the merged
  artifact has `12,245` rows and `0` failures. A matched LISTA-SB p256
  follow-up also completed under
  [results/transition_rich_table2_controls_lista_sb_p256_20260501/self_routed_controls](/home/mila/l/lia/skae/results/transition_rich_table2_controls_lista_sb_p256_20260501/self_routed_controls)
  as shards `9429041`--`9429043`, with merge `9429044`, yielding `2,475`
  rows and `0` failures. The robust summary and compact TeX fragment were
  rebuilt on `2026-05-02` from a combined input that replaces the original p64
  LISTA-SB row with the matched p256 row, and the paper-facing appendix
  display now reports `H100` only:
  [docs/figures/neurips_paper_2026/_tables](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables).
  The controls should remain appendix/audit evidence for now: support-family
  local predictors help, but Oracle basin labels, latent clusters, and
  count-matched random partitions are diagnostic comparators rather than
  deployment-time selectors. The compact control table now uses the matched
  `d_z=256` LISTA-SB row.
- On May 3, Table 2 was refocused on the paper's support-family claim. The
  main table now reports only `F_top8` family-local routing and gives
  system-level exact sign-flip/Holm stars rather than within-system
  Wilcoxon/Holm `[K/17]` counts. A later May 3 display pass replaced the
  missingness-oriented finite-system column with a system-wins column from the
  censored comparison. Regeneration on compute allocations `9449287` and
  `9449702`
  rewrote
  [table2_self_routing_h100_h1000.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table2_self_routing_h100_h1000.tex)
  and the paired-test provenance JSON/CSV. This resolves the previous
  mismatch between large cross-system family-local gains and conservative
  per-system seed-count brackets.
- The May 3 layout pass then placed the compact routing table and the compact
  support-refresh table side by side as subtables 2a and 2b in the manuscript.
  The Dysts actual-MSE table consequently becomes the current manuscript
  Table 3; historical experiment notes may still refer to the Dysts display as
  Table 4.
- A plain dense-K LISTA add-on is complete for Tables 1--3 under
  [results/transition_rich_lista_dense_p256_hardinit_table123_20260430](/home/mila/l/lia/skae/results/transition_rich_lista_dense_p256_hardinit_table123_20260430).
  The new `lista_dense_signsplit_p256_hardinit_basin_partition` row matches
  the current Table 1--3 LISTA protocol (`d_z=256`, `sequence_length=8`,
  `200000` steps, sparsity coefficient `0.003`, hard-init sampling) but removes
  K-structure regularization (`k_structure=dense`, `soft_block=0`). Training
  array `9423749` completed `255/255` tasks, and collector/evaluation jobs
  `9423750`--`9423760` completed with exit `0:0`. The regenerated manuscript
  artifacts now include the plain LISTA row in Tables 1--3 and in the
  fixed-`17` figures.
- The support-refresh fragment has been rebuilt around period-grouped support refresh rather
  than the old `Fallback` column. The period is the number of rollout steps
  between decode/re-encode support refreshes. The missing MLP-control refresh
  rows completed under
  [results/periodic_support_refresh_mlp_controls_seed0to14_20260430](/home/mila/l/lia/skae/results/periodic_support_refresh_mlp_controls_seed0to14_20260430)
  as shards `9423980`--`9423988` with exit `0:0` and shard-level
  `failure_count=0`. The first SLURM merge `9423991` ran out of memory and
  dependent table job `9423996` was cancelled; clean merge job `9432117` was
  rerun with `64G`, completed with exit `0:0`, and wrote `572,654` data rows
  with `0` failures. The current manuscript subtable includes LISTA, LISTA-SB,
  LISTA-BD, Sparse MLP-BD, Sparse MLP, and Dense MLP. The support-refresh
  prose now
  frames periodic re-encoding as a representation-agnostic stale-route repair
  on the controlled transfer benchmark: refreshing the state-dependent support
  or route helps after the trajectory moves to a new region, independent of
  which displayed Koopman representation produced that local description.
- The statistical testing procedure is now documented as a repeatable
  protocol in the manuscript appendix and in
  [EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md). The fixed rule
  for within-system \(K/N\) reproducibility counts is to build paired deltas within each
  system, run one-sided paired Wilcoxon tests inside systems, Holm-correct
  across eligible systems within each table cell, and report \(K/N\) as the
  number of Holm-cleared systems. Table 1 suppresses the all-`15/15`
  multibasin forecasting counts and uses `\ast` superscripts, while
  support/refresh diagnostics keep bracketed counts. The main Dysts columns
  are reduced to `H100/H2000/H4000`; the full Dysts horizon trend remains in
  the figure. Table 2 main inference is now system-level:
  finite routed/global log-ratios are summarized within each system and tested
  across systems with an exact sign-flip/Holm procedure. The older censored
  seed-slot deltas remain provenance/appendix diagnostics for route-form
  ablations.
- A targeted Dysts diagnosis now explains the LISTA-BD negative result more
  sharply. Seed-`0` `H30000/H40000/H50000/H60000` phase portraits comparing
  LISTA-BD with Dense MLP show LISTA-BD non-finite rollouts on `11/12` systems,
  while Dense MLP remains finite. Transition spectra point to unstable hard
  block-diagonal LISTA transitions (`rho(K)` roughly `1.06`--`1.16` in the old
  checkpoints), and the evaluator's finite-prefix `nanmean` can hide
  non-finite suffixes. A `dysts:Chua` prototype with `dt x30`, horizons `/30`,
  and `50000` optimization steps is fully finite through reduced `H2000`
  with selected-checkpoint MSE `0.0005300`, so coarser sampling is promising
  but still a benchmark-design hypothesis.
- The Dysts `dt x30` follow-up is now collected. The long-horizon evaluator
  and collector report full-horizon finite coverage, finite-step coverage,
  valid-MSE fractions, and finite-prefix lengths beside the historical
  finite-prefix MSE. The full `12`-system packet under
  [results/dysts_dt30_basinblock_p256_seq10_100k_20260430/long_horizon_eval/collect](/home/mila/l/lia/skae/results/dysts_dt30_basinblock_p256_seq10_100k_20260430/long_horizon_eval/collect)
  has `1080/1080` rows, `0` pending/invalid rows, and median full-horizon
  finite coverage of `1` for every root at every reported horizon. The
  paper-facing aggregation now excludes `dysts:LorenzCoupled`, the only
  six-dimensional member of the original shortlist, and `dysts:MultiChua`, to
  avoid Chua-family redundancy. It uses seed-IQM within systems followed by
  arithmetic means across the retained `10` three-dimensional systems:
  [table4_dysts_dt30_iqm.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table4_dysts_dt30_iqm.tex)
  reports means of per-system seed-IQM MSEs and marks actual-MSE
  cells with `*` when the exact system-sign test against Dense MLP passes
  after Holm correction across all model-horizon comparisons. Under the
  retained `10`-system denominator this means the displayed star marks
  improvement on all `10/10` systems. In the current compact
  `H100/H2000/H4000` Dysts display, repaired Sparse MLP-BD is starred at
  `H2000` and `H4000`, with `9/10` system wins and no star at `H100`. The
  manuscript-linked Dysts trend figure has now been regenerated with LISTA-SB
  included at the historical `no_lista_sb` filename; the table and trend
  display it as a diagnostic row, where it does not survive the
  all-comparison correction. The supporting
  ratio/log-ratio diagnostics remain in
  [dysts_dt30_aggregate_tests_vs_dense.csv](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/dysts_dt30_aggregate_tests_vs_dense.csv)
  for appendix/audit use rather than as a main-text table.
  Dense MLP is worst at every displayed horizon. With the repaired Sparse
  MLP-BD row, the compact displayed aggregate-best rows are Sparse MLP at
  `H100`, LISTA at `H2000`, and repaired Sparse MLP-BD at `H4000`. Ratio
  summaries include hierarchical system+seed bootstrap
  intervals and system-level/log-ratio SDs for uncertainty auditing, but raw
  MSE SDs should not be foregrounded because the Dysts errors are heavy-tailed
  and span orders of magnitude. The paper-facing raw-MSE trend
  uses log scale because the displayed MSEs span about `4.8e3x`; linear and
  side-by-side scale-check versions remain as diagnostics. The trend bands now
  show fixed-system seed-bootstrap `95%` intervals after system-wise
  log-relative normalization, anchored to the displayed arithmetic mean, so
  they summarize seed uncertainty within an average system rather than
  cross-system `25`--`75` percentile heterogeneity or raw-scale sensitivity to
  one high-MSE resample. For uncertainty audit, `_raw_seed_ci` keeps the
  original raw-MSE fixed-system interval, `_system_ci` plots resample fixed
  per-system seed-IQMs across systems, and `_log_seed_bootstrap` plots
  resample seeds after a `log10` MSE transform.
  New plots
  `fig_dysts_dt30_forecasting_performance_no_lista_sb.pdf`,
  `fig_dysts_dt30_forecasting_performance.pdf`,
  `fig_dysts_dt30_forecasting_performance_linear.pdf`,
  `fig_dysts_dt30_forecasting_performance_scale_check.pdf`,
  `fig_dysts_dt30_iqm_horizon.pdf`, `fig_dysts_dt30_ratio_to_dense.pdf`, and
  `fig_dysts_dt30_winner_counts.pdf` are in
  [docs/figures/neurips_paper_2026](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026),
  and seed-`0` phase portraits for all `12` originally collected systems are in
  [docs/figures/dysts_dt30_phase_portraits_seed0_h1000_h5000_all_models_20260501](/home/mila/l/lia/skae/docs/figures/dysts_dt30_phase_portraits_seed0_h1000_h5000_all_models_20260501).
  Treat this as a benchmark-discretization sensitivity result, not as a silent
  replacement for the old `H<=60000` Dysts stress-test table.
- A clean Dysts LISTA-SB diagnostic smoke is complete. The
  `dt x30` builder now makes LISTA-SB match LISTA-BD's encoder
  (`1` LISTA loop, ReLU final operator) and changes only the `K` structure:
  hard block-diagonal for LISTA-BD versus dense `K` with a soft off-block
  penalty for LISTA-SB. Launcher `9430988` completed the small `dysts:Chua`,
  seed-`0`, `50000`-step smoke; the collected rows are `2/2` complete with
  full-horizon finite coverage through `H2000`. LISTA-SB/LISTA-BD
  best-periodic MSE ratios are `0.859/0.938/1.191/1.436/1.507` at
  `H100/H500/H1000/H1500/H2000`: cleaned LISTA-SB is slightly better at the
  two shortest horizons and worse from `H1000` onward on this one Chua seed.
  Do not change Table 4 claims from this one-system smoke alone.
- A one-seed all-`12`-system clean LISTA-SB run is now complete under the
  paper-facing Dysts protocol. Launcher `9432830`, cache array `9432834`,
  training array `9432835`, evaluation queue launcher `9432836`, long-horizon
  eval jobs `9433586`--`9433588`, and collector `9433589` completed with exit
  `0:0`, producing `12/12` complete rows under
  [results/dysts_dt30_clean_lista_sb_seed0_12sys_100k_20260501](/home/mila/l/lia/skae/results/dysts_dt30_clean_lista_sb_seed0_12sys_100k_20260501),
  with horizons `H100/H500/H1000/H1500/H2000/H3000/H4000/H5000`.
  The generated task table verifies the fix across all rows: `bad_rows=0`,
  `lista_num_loops=1`, `lista_final_op=relu`, dense `K`, soft-block enabled,
  and `soft_block_num_blocks` equal to the documented basin/scroll/lobe count.
  The row is finite across the full horizon grid. Median best-periodic MSEs
  are
  `0.000299/0.005165/0.024618/0.057127/0.087170/0.233537/0.513065/0.943133`;
  seed-`0` prior trimmed aggregates are
  `0.000657/0.013247/0.046570/0.132622/0.274048/0.605152/0.991347/1.37441`.
  These aggregates are worse than the old seed-`0` LISTA-SB row at every horizon,
  though the clean recipe improves the median system and wins `7/12` systems
  at the longest horizons. The superseded `50000`-step/`H<=2000` all-system
  chain from launcher `9432796` was canceled before training started.
- The focused higher-dimensional hard-system redo is collected and corrected.
  Launcher `9412191`, training array `9412218`, collector `9412219`, and
  comparisons `9412220`--`9412222` all completed with exit `0:0`. The first
  collector artifact under-counted Hopfield because it selected latest runs
  per `n_16/seed` wrapper; after fixing the collector and refreshing on
  allocation `9432839`, the run has the expected `270` rows. Dense MLP beats
  every sparse/LISTA candidate on fixed 8-basin CLV, Hopfield `N=16/P=16`,
  and Kuramoto identical `N=16` at `H100`, `H500`, and `H1000` system
  medians.

Outstanding problem:
- The normalized-decoder replacement and encoded-sparsity ablation are
  partially collected but not paper-ready. The Dysts side completed with small
  retained-seed coverage gaps and mixed forecasting effects; the patched
  controlled seed-`0,1,2` side now has forecasting/support diagnostics, but
  the 3-hour GPU cap left partial coverage (`226/270` rollout tasks and
  `255/270` encoded tasks completed). If a clean `normdec_rollout` packet
  later improves or preserves forecasting/support diagnostics, it is the fair
  table-replacement candidate because every KAE row uses the same normalized
  linear decoder convention. If `normdec_encoded` improves support diagnostics
  without damaging forecasting, the paper objective should be updated
  coherently for all sparse KAEs rather than only for MLP rows. Group sparsity
  remains a separate BD-specific ablation, not a replacement criterion for the
  current tables.
- The new spatialized-PDE and ManiSkill insertion benchmarks are not landed
  paper evidence. The PDE convolutional path is implemented and the first
  full controlled expansion is execution-complete, but the aggregate is mixed:
  tanh dense is the strongest finite H1--H128/H256 forecaster, while ReLU
  sparse rows provide compact, basin-aligned support families under the
  calibrated PDE-scale support rule. The next PDE step is paper positioning,
  not broad scale-up under a stronger claim: use this as an appendix
  high-dimensional support-alignment stress test unless a later method recovers
  dense-tanh forecasting while retaining compact sparse supports. Robotics
  remains a separate candidate. Array `9725895` completed a matched `50k`-step
  CPU-only ManiSkill retry over dense, dense+L1 sparse-MLP, and LISTA settings
  for seeds `0`--`2`, but those dense rows used ReLU MLP blocks and therefore
  do not answer the dense-tanh-control question. Longer training substantially
  improves all rows and mild sparse settings slightly improve mean H100 over
  dense-ReLU, but dense-ReLU remains best at shorter horizons and the cleanest
  support-alignment rows forecast worse. Focused long-horizon eval `9729181`
  shows target LISTA `alpha=0.03,sparsity=0.01` remains better than dense-ReLU
  through H175 on the stored packet, but all focused rows are unstable by
  H200/H220 and H300--H500 require newly generated longer rollouts. The
  activation-corrected `5k` pilot now gives a stronger sparse-KAE forecasting
  signal:
  with true `dense_tanh_sp0`, `last.pt`, and best periodic decoded-prediction
  re-encoding over `{1,2,5,10,20,50,100}`, sparse KAE rows beat dense tanh over
  seeds `0`--`2`. The June 9 optimizer-fairness run shows that giving
  `lr=5e-4,weight_decay=0` to dense and sparse-MLP controls makes sparse-MLP
  ReLU the best small-pilot row (`sparsity=0.003` H10--H50 `0.001837`, versus
  tuned dense `0.002023` and best LISTA `0.001881`). This should be read as a
  promising fixed-protocol sparse-KAE candidate, not yet as landed paper
  evidence, because checkpoint selection, semantic label/support evaluation,
  and scale-up need to be fixed before paper-facing claims. The current
  display candidate from the
  old `50k` packet is
  the finer H10--H125 forecasting graph at horizons
  `10,20,30,40,50,60,70,80,90,100,110,120,125`,
  [fig_maniskill_h125_forecasting_fine.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_maniskill_h125_forecasting_fine.pdf),
  with CSV/manifest provenance under
  [results/maniskill_h125_forecast_fine_20260603](/home/mila/l/lia/skae/results/maniskill_h125_forecast_fine_20260603).
  This is an autonomous latent-rollout evaluation: the model encodes the
  initial state once, rolls latent dynamics through the action sequence, and
  decodes the rollout. It does not use periodic re-encoding; the re-encoding
  period is none/infinite.
  A matched best-periodic re-encoding follow-up is now complete with periods
  `{10,20,50,100}` and display
  [fig_maniskill_h125_forecasting_best_periodic.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_maniskill_h125_forecasting_best_periodic.pdf),
  with CSV/manifest provenance under
  [results/maniskill_h125_forecast_best_periodic_20260603](/home/mila/l/lia/skae/results/maniskill_h125_forecast_best_periodic_20260603).
  This protocol refreshes from decoded predictions rather than ground-truth
  future states and selects the best periodic cadence per model/seed/horizon.
  It makes H100/H125 errors nearly flat (`~0.0019`--`0.0024`) and removes the
  autonomous blow-up through H125, but it should be positioned as
  reset-stabilized forecasting, not strict autonomous Koopman rollout.
  Sparse-MLP `sparsity=0.003` is marginally best at H100/H125
  (`0.00186/0.00205`), dense-ReLU is close (`0.00188/0.00211`), and LISTA is
  not the best-periodic forecasting winner within that ReLU-control packet.
  Winning cadences are mostly
  `periodic_10` and `periodic_20`; `periodic_100` is never selected.
  A H220 extension is now complete under the same period grid. The stored
  packet has full held-out coverage through H175 (`15` episodes per seed),
  then drops to `9` episodes at H200 and `5` at H220. Use H175 as the longest
  clean current-packet horizon and treat H200/H220 as reduced-coverage stress
  diagnostics. Best-periodic H175 mean state MSEs are dense-ReLU `0.00301`,
  sparse-MLP `0.00304`, LISTA `alpha=0.03,sparsity=0.003` `0.00399`, and
  LISTA `alpha=0.03,sparsity=0.01` `0.00394`. The H220 display is
  [fig_maniskill_h220_forecasting_best_periodic.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_maniskill_h220_forecasting_best_periodic.pdf),
  with CSV/manifest provenance under
  [results/maniskill_h220_forecast_best_periodic_20260603](/home/mila/l/lia/skae/results/maniskill_h220_forecast_best_periodic_20260603).
  ManiSkill still needs validated perturbation-balanced outcome/contact labels
  before any paper-facing claim.
- The controlled repaired Sparse MLP-BD compute/table blocker is closed. Dysts
  and controlled rows now use repaired block-diagonal `GenericKM` artifacts in
  the rebuilt paper table sources. Remaining work for this row is narrative
  calibration, not another rerun.
- The reviewer-requested additions are partially landed. The completed
  one-step regime-discovery read already narrows the regime-variable claim:
  basin-count clustering is a strong evaluation-only baseline, while learned
  support-family local-\(K\) is near global on the p256 one-step latent metric.
  The OOD packet supports forecasting and deep-slice support purity but is
  negative for local-K routing. Do not cite the autonomous long-horizon
  oracle/local-K add-ons until p256 merge `9478603` and OOD merge `9478622`
  land. A full temporal switching LDS remains unimplemented; the completed
  diagonal-GMM routed local-\(K\) comparison is the current mixture-of-Koopman-
  style baseline.
- The standalone state-space/local-linear baseline suite is now interpreted and
  tabled for both paper forecasting rosters, now including label-free local
  polynomial EDMD and local RBF-EDMD. On controlled multibasin, local
  polynomial EDMD is the best standalone baseline but remains behind the best
  sparse KAE rows by `3.9x/2.7x/2.6x` at H100/H500/H1000 and behind LISTA by
  `3.7x/1.8x/1.7x`. On Dysts `dt x30`, it is excellent at H100 but does not
  produce a long-horizon sparse-KAE replacement, losing by `22x/6.8x` at
  H2000/H4000. Local RBF-EDMD and older
  local-linear mixtures can diverge at long horizons. The remaining baseline
  work is not display selection for these standalone controls; it is filtering
  the checkpoint-backed oracle/clustering local-K packet to the retained paper
  roster and deciding whether to repair, exclude, or explicitly caveat the
  `43` row-local failed `gmm_diag` clustering assignments before using that
  route family.
- The stage-2 trained-local-map expansion is no longer a queue blocker, but it
  needs careful positioning. The controlled 50k result is significantly better
  than the hard-init Dense MLP Table 1 comparator, but remains mixed against
  same-root best-periodic; Dysts 50k is strongly negative, controlled 100k is
  worse than 50k, and Dysts 100k has three shard failures plus unstable
  available rows. The Dysts re-encode-period sweep is now queued as the focused
  robustness check. Unless periods `{1,2,5,10,20}` materially change the Dysts
  conclusion, include stage-2 only as a controlled multibasin/local-map row or
  appendix ablation, not as an external-robust forecasting claim; do not spend
  more queue time on budget-only continuations.
- The strict staged source-target affine \(F_{\rm abs}\) local-\(K_c\)
  2-seed pilot is complete and heterogeneous, not a significant forecasting
  win over same-budget global-\(K\) LISTA with best periodic reencoding. The
  clean positives are useful mechanism/control cases, but row-level wins are
  tied at the longer horizons (`15/30` at `H500` and `H1000`), so do not scale
  this recipe to a broad seed sweep without further tuning. The artifact audit
  points to baseline-headroom and local-map damage, not too few support
  families, as the immediate tuning target. A learned-intercept contrast pilot
  (`9553697 -> 9553698 -> 9553699`) completed, but only the restricted
  H100/periodic-1 re-evaluation landed. It wins `5/8` paired rows and confirms
  the local maps are active, but the mixed system pattern and missing
  long-horizon wide-grid read mean this remains tuning evidence rather than a
  forecasting result to scale. The \(C_{\rm stab}\)-routed local prediction
  probe (`9562611 -> 9562654 -> 9562821 -> 9562824`) completed and wins
  `7/8` paired rows at `H100/H500/H1000`, but only `3/4` systems by
  seed-geometric ratio. The full retained multibasin Table 1 roster with seeds
  `0`--`14` has now resolved the scale-up question:
  `9565762 -> 9565763 -> 9565764 -> 9565765` all completed with exit `0:0`,
  producing `225/225` matched wide-periodic staged/global rows. Staged local
  maps win `189/225`, `188/225`, and `188/225` paired rows at
  `H100/H500/H1000`, `176/225` paired rows on all three horizons, and `14/15`
  systems by seed-geometric ratio at each horizon. This is a statistically
  strong controlled-multibasin win over same-budget global-\(K\) LISTA, with
  `duffing_triple_well` the lone system-level exception. Compact aggregate:
  [wide_periodic_reeval_aggregate.md](/home/mila/l/lia/skae/results/staged_cstab_learned_intercept_k_lista_full_20260516/wide_periodic_reeval/wide_periodic_reeval_aggregate.md).
  The retained
  `10`-system Dysts `dt x30` extension is not positive: `9565795` finished
  training, `9573792` collected the long-horizon read, and the completed
  staged/global pairs are orders of magnitude worse for staged local maps.
  Treat Dysts as a negative external stress test for this exact local
  \(K_c,d_c\) recipe.
- The combined-stage2 aggregation utility needs cleanup before using 100k or
  calibrated-global combined readouts in any appendix table.
- The training-dynamics appendix displays are execution-complete and are not a
  blocker. Their current role is optimization/checkpoint diagnostics: one
  representative controlled trace and one retained-`10` Dysts median companion.
  Expand them to a multi-seed or per-system audit only if coauthors ask for
  stronger appendix coverage of training progression.
- The Dysts IQM-over-IQM appendix table/curve are execution-complete and did
  not require new training or evaluation jobs. Keep them as a robust
  aggregation sensitivity result unless coauthors ask to change the main Dysts
  estimand.
- The routed-forecasting-MSE add-on is execution-complete and negative for the
  main forecasting tables. The controlled multibasin shards
  `9466089`--`9466223` merged in `9466224` with `675` rows and `0` failures;
  the Dysts shards `9466225`--`9466314` merged in `9466315` with `450` rows
  and `0` failures; aggregation job `9467050` compared seed-IQM system means
  against the current same-model best-periodic Table 1/Table 3 rows. The
  routed rows win `0/15` controlled systems and `0/10` Dysts systems at every
  evaluated horizon, so they should not be promoted as superior forecasting
  evidence. The periodic decode/re-encode follow-up also completed with
  `2700` controlled rows, `1800` Dysts rows, and `0` recorded failures; it
  improves stability relative to autonomous routing but leaves every aggregate
  routed/best-periodic ratio above `1`. The remaining paper issue is how
  briefly to mention this as a falsification/mechanism check, if at all.
- The remaining paper work is now presentation and claim calibration. Tables
  1--3 have been recomputed on the retained `15` controlled systems after
  excluding `multiwell_strong_transition` and `claude_checkerboard_potential`,
  with matched `d_z=256` LISTA-SB artifacts where available. The current Dysts
  `dt x30` table should be
  framed as forecasting competitiveness and benchmark-discretization
  sensitivity: after excluding `dysts:LorenzCoupled` and `dysts:MultiChua`,
  Dense MLP is worst at every displayed horizon, every primary sparse row is
  better than Dense in aggregate at every displayed horizon, and the compact
  displayed aggregate-best rows are Sparse MLP at `H100`, LISTA at `H2000`,
  and repaired Sparse MLP-BD at `H4000`. The main mechanistic claims should
  still rest on the controlled
  multibasin evidence: basin-support alignment, non-oracle support-routed
  prediction, and support refresh under transfer.
- The Dysts appendix now adds the robust IQM-over-IQM read for the same
  retained systems. In that sensitivity, repaired Sparse MLP-BD is best at
  every displayed horizon; cite it only as an aggregation-sensitivity appendix
  result because the main table's arithmetic-mean estimand still captures
  average-case MSE.
- The Dysts follow-up no longer needs queue monitoring; it has been folded into
  the manuscript as the current Dysts display with an explicit `dt x30`
  sensitivity label. Full-horizon coverage reporting shows the setup is fully
  collected and finite across all `12` originally queued Dysts systems with
  matched Dense MLP, Sparse MLP, and LISTA-family rows at `100000` training
  steps. The paper-facing display now retains `10` three-dimensional systems
  and excludes `LorenzCoupled` for state-dimension comparability plus
  `MultiChua` for Chua-family redundancy. The result
  supports the rescale diagnosis, but it changes the discrete-time benchmark
  and reduces `H60000` from `60000` learned-map compositions to `2000`; keep
  the old tiny-step packet as a separate stress-test diagnostic.
- Separately, the LISTA-SB recipe comparison on Dysts is under diagnostic
  review. The existing Dysts LISTA-SB row came from the older two-loop,
  sign-split soft-block setup; the completed Chua smoke tests the cleaner
  hard-vs-soft `K` comparison on one system. The smoke is finite and mixed,
  with cleaned LISTA-SB better than LISTA-BD at `H100/H500` but worse at
  `H1000/H1500/H2000`. The completed seed-`0` all-system clean LISTA-SB run
  with the correct `100000`-step, `H100`--`H5000` Dysts protocol fixes the
  setup bug but does not improve the paper-facing IQM summary. This is not a
  paper blocker and does not justify changing the Dysts display.
- The Table 2 testing/display procedure is now fixed for the manuscript.
  Subtable 2a reports `F_top8` family-local routed/global ratios at `H100`
  and `H1000` plus system-wins counts; exact gated/support-local `S_top8`
  routes are appendix diagnostics.
  The main significance stars use the benchmark system as the independent
  unit: seed-level finite log-ratios are summarized within system and tested
  across systems with an exact one-sided sign-flip test, Holm-corrected across
  the displayed model--horizon cells. LISTA, LISTA-SB, and LISTA-BD clear this
  corrected system-level test at both horizons; Sparse MLP-BD clears at `H100`
  only; Sparse MLP and Dense MLP do not clear. Subtable 2b uses transfer pairs
  within system and clears
  `12/12` systems for every displayed row (LISTA, LISTA-SB, LISTA-BD, Sparse
  MLP-BD, Sparse MLP, and Dense MLP) at both periods. The Dysts table
  now uses the `10` retained per-system seed-IQM log-MSE differences for
  system-level model-vs-Dense inference, shown as `*`/`**` on the actual-MSE
  cells. The older `[K/N]` seed-paired within-system Wilcoxon/Holm counts should stay in
  diagnostics rather than the main table. It should be presented as a mixed
  external forecasting stress test rather than as an interpretability claim.
- The Table 2 censoring rule should be described as a provenance/appendix
  diagnostic, not as the main significance test. A finite routed H-step
  forecast with an invalid same-model global H-step forecast is counted as a
  censored routed win in the diagnostic count because the benchmark ranks an
  evaluable forecast above an unevaluable forecast for the same system, seed,
  subset, and horizon. The main table uses mutually finite routed/global pairs
  and reports the number of contributing systems.
- The completed Table 2 partition-control audit supports the intended
  evidence chain for support routing: Oracle basin labels ask whether useful
  local predictors exist when the benchmark partition is supplied, support
  families ask whether the learned sparse representation supplies a label-free
  substitute, and random/cluster partitions test the generic-subdivision
  alternative. The compact control table has been rebuilt with the matched
  `d_z=256` LISTA-SB follow-up (`9429041`--`9429043 -> 9429044`) and now
  reports `H100` only; keep it in the appendix as audit evidence rather than
  as a main routing-table replacement.
- The old per-basin deep-slice interpretability rerun (`9388212`--`9388218`) is
  superseded by the current-roster rerun (`9477837`--`9477845`). Table 1 now
  deliberately uses the per-basin deep support-diagnostic estimand. Forecasting
  columns remain all-held-out-rollout MSE, so the table should be described as
  all-held-out forecasting plus basin-stratified mechanistic support diagnostics.
- Matched-dimension LISTA-SB `d_z=256` replacement is landed for Tables 1--3.
  The fixed-`17` packet produced `255` forecasting rows, `6,885`
  interpretability rows, and `1,980` self-routed rows with `0` failures. The
  matched support-refresh packet produced `191,400` rows with `0` failures.
  Tables 1--3 were recomputed on allocation `9420481` and the PDF was rebuilt.
  The Dysts add-on produced `180/180` rows but only through `H30000`, so Table
  4 replacement is deferred to a later full-horizon matched run.
- The plain dense-K LISTA add-on has finished and is now a main controlled
  benchmark row in the regenerated Table 1--3 artifacts. The fixed-`17`
  horizon curves use all currently recorded fixed-`17` horizons
  (`H100/H500/H1000`) with the same fixed-system, log-relative seed-bootstrap
  `95%` bands as the Dysts horizon trend; no current Table 1--3 collected fixed-`17` row
  exposes an `H3000` horizon.
- Table 3 should be interpreted as a broad periodic-re-encoding effect rather
  than a LISTA-only support mechanism. All displayed LISTA, sparse MLP, and
  Dense MLP rows benefit from refreshing their state-dependent local
  description after transfer; representation-specific Koopman-structure claims
  remain grounded in the alignment, ablation, and routing tests.
- The paper-facing prose should always explain why the next
  experiment follows: label agreement asks whether the support says where the
  state is; routing asks whether the support helps predict where it goes next.
- The higher-dimensional hard-system redo is no longer a queue blocker. Use
  it as a negative stress-test result: once the collector bug was fixed, tanh
  Dense MLP was best on CLV, Hopfield, and Kuramoto at all evaluated horizons.

## Goal

Fixed-`17` LISTA root/result lookup:
- Use [FIXED17_LISTA_RESULTS_INDEX.md](/home/mila/l/lia/skae/docs/FIXED17_LISTA_RESULTS_INDEX.md) as the canonical quick-reference page for the fixed-`17` LISTA roots, their packets, and the headline results that currently matter for the paper write-up.
- Use [SUPPORT_OBJECT_GLOSSARY.md](/home/mila/l/lia/skae/docs/SUPPORT_OBJECT_GLOSSARY.md) for the paper-facing definitions of `absolute:0.001`, `relative:0.1`, `topk:8`, exact support, support family, and dominant group.

The paper target is now explicit:

- make the lead live branch a fixed-`17` basin-separation comparison:
  do the models learn reusable sparse supports for distinct basins on the
  selected systems, and is some induced sparsity essential for good
  finite-dimensional Koopman representations when multiple basins or fixed
  points coexist? LISTA is one structured way to induce that sparsity, not
  the claim by itself.
- keep the fair `200k` forecasting packet, the hard-system packet, and the
  existing mechanism packet as **decision-grade supporting evidence** rather
  than the lead live branch
- execute that branch on the 17 systems that we can analyze mechanistically in the paper through the plan in [docs/planning/transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md)
- use [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md) as the current ground-truth planning note for ablation design choices when iterating on items `3` and `4` of that transition-rich plan, and replace planning assumptions with experiment-backed conclusions once those axes are run systematically
- distinguish **endpoint basin** from **finite-horizon transition**; the new branch is intentionally transition-rich even though endpoint basins remain well defined
- write **tests before any system-specific code**, then calibrate toy systems before queueing model sweeps
- keep training-time method design label-free: do not assume known basin counts or basin labels outside benchmark diagnostics
- prioritize support agreement with basin labels, support-view clustering,
  recurring-support reuse, and local predictive structure over MSE-only
  reporting
- measure long-horizon forecasting at `H100`, `H500`, and `H1000` as a
  downstream functional test of the support-label hypothesis: if sparsity
  helps because the Koopman state retains basin identity and that identity is
  dynamically useful, forecasting should be strongest away from separatrices
  and weakest near separatrices where the relevant local law is ambiguous
- use oracle basin-depth / separatrix-proximity slices only for benchmark
  evaluation; do not turn those signals into training-time assumptions
- treat local-law interpretability as basis-aware and symmetry-aware: do not require different basin eigenvalues as the success criterion; instead compare raw and similarity-aligned operator/Jacobian families together with support-family uniqueness up to alignment
- treat the local-versus-global evidence as currently strongest against the
  trained model's learned global transition, not against every possible global
  centered refit. The centered local-law packet supports support-conditioned
  laws on covered states, especially away from basin boundaries, but the
  separately refit global-centered comparison remains weaker and should be
  written cautiously.
- use the two reviewer-response branches as falsification diagnostics:
  [true_jacobian_geometry_experiment_20260423.md](/home/mila/l/lia/skae/docs/planning/true_jacobian_geometry_experiment_20260423.md)
  tests true local-geometry agreement, and
  [controlled_transfer_switching_experiment_20260423.md](/home/mila/l/lia/skae/docs/planning/controlled_transfer_switching_experiment_20260423.md)
  tests support switching under deliberate basin transfer. A tabula-rasa audit
  found evaluator mistakes in the first seed-`0` fixed-`17` outputs, and those
  first outputs are superseded. Corrected reruns are now complete. The
  true-geometry result is a secondary, mixed falsification diagnostic:
  LISTA support families often beat random count-matched partitions near
  attractors, but the zero-sparse MLP often has lower absolute
  projected-Jacobian error because its latent representation is closer to the
  identity map. The
  controlled-transfer result is sharper: dense LISTA exact `topk:8` supports
  switch well after a deliberate state-space bridge, while zero-sparse MLP
  exact supports remain weak; support-family switching is strong for all roots
  and therefore is not sufficient for a LISTA-specific sparsity claim.
- keep the matched standard MLP encoder control as the main causal comparator
  on this branch; use the clean tanh / no-shrink MLP when isolating the
  induced-sparsity effect directly
- for paper-facing comparisons that claim an architecture effect, keep the
  training sampling regime matched across compared roots; treat hard-init
  oversampling as a separate factor and report standard versus hard-init as
  separate table entries rather than folding them into one causal read
- do not treat local-switch localization or sparse-only mechanism claims as the branch acceptance criterion; those are supporting context, while the live branch question is basin separation on the fixed `17` systems
- treat **`200k` as the only main-text training budget** for the frozen benchmark and hard-system supporting packets and this should only be done once we have compelling and significant results at a 20k budget
- use **`20k` as the working training budget** for forward interpretability and metric-diagnosis runs on the fixed `17`-system shortlist while the recipe is still moving, and reserve `200k` on that branch for the final locked confirmatory rerun plus the now-completed narrow default-sampling forecast-floor check on the best `v6` / `v7` roots
- use **`1` seed as the default working budget** for forward interpretability diagnostics on the fixed `17`-system shortlist while the branch is still choosing methods and metrics, and expand to `10` seeds only after a seed-`0` result looks strong enough to justify a paper-facing robustness check
- use the **default sampling regime** for forward LISTA comparator sweeps on
  the fixed `17` systems so the comparison against the MLP controls does not
  inherit a sampling confound; keep hard-init or other altered sampling
  schemes as separate ablation rows once the base recipe is fixed
- treat the dense LISTA Stage 1-4 chain as **appendix-only comparator-selection provenance**
- the MLP `+ block_diagonal K` fairness controls were rerun on March 17/19; full audit: [docs/mlp_block_k_audit_20260317.md](/home/mila/l/lia/skae/docs/mlp_block_k_audit_20260317.md)

Active execution note:
- Completed higher-dimensional hard-system forecasting redo: queue launcher
  `9412191`, training array `9412218`, collector `9412219`, and comparison
  jobs `9412220`--`9412222` completed with exit `0:0` under
  [results/hard_system_sparse_kae_redo_p1024_seq8_100k_halflr_sc6em3_tanh_dense_20260429](/home/mila/l/lia/skae/results/hard_system_sparse_kae_redo_p1024_seq8_100k_halflr_sc6em3_tanh_dense_20260429).
  The corrected collector artifact has `270` rows after fixing the `n_16`
  wrapper grouping bug. The answer is negative for the sparse/LISTA recipes:
  tanh Dense MLP is best on CLV, Hopfield, and Kuramoto at `H100`, `H500`,
  and `H1000` system medians.
- The forecasting packet is now decision-grade, and the raw-source seed-statistics companion report is in [docs/PAPER_SEED_STATISTICS_20260331.md](/home/mila/l/lia/skae/docs/PAPER_SEED_STATISTICS_20260331.md). It verifies raw-vs-collector agreement and records the remaining raw finite-value coverage gaps explicitly.
- The true-Jacobian/eigendirection and controlled-transfer support-switching
  branches are now scaffolded, revised, smoke-tested, audited, corrected, and
  rerun for fixed-`17` seed-`0`. Implemented files:
  [tools/evaluate_transition_rich_true_jacobian_geometry.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_true_jacobian_geometry.py),
  [tools/evaluate_transition_rich_controlled_transfer_switching.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_controlled_transfer_switching.py),
  [run_transition_rich_true_jacobian_geometry.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_true_jacobian_geometry.sh),
  and
  [run_transition_rich_controlled_transfer_switching.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_controlled_transfer_switching.sh).
  Shell syntax checks pass for the wrappers, and compute-node py-compile
  validation passed after the audit fixes. Corrected smoke jobs `9347587` and
  `9347588` completed with `COMPLETED 0:0`. Corrected true-geometry job
  `9347593` completed in `17m38s` with `49/49` runs, `62,460` rows,
  `30,014` ok rows, and `0` failures under
  [results/true_jacobian_geometry_fixed17_seed0_20260423_corrected](/home/mila/l/lia/skae/results/true_jacobian_geometry_fixed17_seed0_20260423_corrected).
  Corrected controlled-transfer coverage jobs `9347590-9347592` completed
  with `1,776` total rows, `1,632` ok rows, `144` skipped rows, and `0`
  failures under
  [results/controlled_transfer_switching_fixed17_seed0_20260423_corrected](/home/mila/l/lia/skae/results/controlled_transfer_switching_fixed17_seed0_20260423_corrected).
  Interpretation is now claim-calibration rather than execution coverage:
  true geometry is not a headline result, while dense LISTA exact `topk:8`
  controlled-transfer switching is a useful support-switching diagnostic.
  April 25 verification: the second-audited `20260424_reaudit` jobs
  `9347926-9347929` all completed with exit `0:0`, no matching jobs are
  currently queued, and the controlled-transfer packet remains organized as
  three completed root shards rather than a merged top-level summary file.
  Claim-framing note: the MLP roots should be treated as optional specificity
  controls for these two branches, not as required comparators. The geometry
  question can be written as a LISTA-only support/family versus random and
  attractor/basin-baseline diagnostic. The controlled-transfer packet shows
  encoder support switching along a measured state-space basin transfer; a
  stronger periodic-reencoding claim should be phrased as a rollout mechanism
  or backed by an explicit no-reencoding versus reencoding ablation.
- That explicit ablation is now queued. New files:
  [evaluate_transition_rich_periodic_support_refresh.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_periodic_support_refresh.py),
  [run_transition_rich_periodic_support_refresh.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_periodic_support_refresh.sh),
  [queue_transition_rich_periodic_support_refresh.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_periodic_support_refresh.sh),
  and
  [merge_transition_rich_periodic_support_refresh_shards.sh](/home/mila/l/lia/skae/scripts/merge_transition_rich_periodic_support_refresh_shards.sh).
  Compute-node smoke job `9361455` completed with `32` ok rows and `0`
  failures under
  [results/periodic_support_refresh_smoke_20260425_cal_square](/home/mila/l/lia/skae/results/periodic_support_refresh_smoke_20260425_cal_square).
  Full fixed-`17` seed-`0` LISTA-only shards `9361464` and `9361465` completed
  cleanly under
  [results/periodic_support_refresh_fixed17_seed0_20260425](/home/mila/l/lia/skae/results/periodic_support_refresh_fixed17_seed0_20260425).
  Dense LISTA completed `16/16` specs with `34,440` rows (`34,176` ok,
  `264` skipped, `0` failures), and blockdiag LISTA completed `17/17` specs
  with `38,280` rows (`38,016` ok, `264` skipped, `0` failures). Merge job
  `9361470` is still pending on scheduler priority, but the per-root summaries
  already provide the science read. The decisive positive result is dense
  LISTA exact `topk:8` after the trajectory is clearly in the target basin:
  refreshed-support routing reaches route-target fraction `0.8552/0.8886`,
  fallback `0.1392/0.1058`, and
  refreshed-versus-previous-support MSE ratio `0.0093/0.0131`. Dense LISTA
  `topk:8` family and blockdiag `topk:8` family also support the mechanism,
  but blockdiag exact supports do not. Therefore the stronger mechanism claim
  should be written for dense LISTA exact `topk:8` and for support families,
  not as a universal statement about every LISTA support definition.
- The supporting Dysts long-horizon visual packet for senior-coauthor handoff
  is now complete at
  [H5000](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h5000_lista_phase_portraits_manifest.json),
  [H20000](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h20000_lista_phase_portraits_manifest.json),
  the older LISTA-only shared-batch
  [H30000](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h30000_lista_phase_portraits_manifest.json),
  and the benchmark-aligned all-roots
  [H30000 best-root packet](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h30000_best_root_phase_portraits_manifest.json).
  The new all-roots `H30000` selector uses the completed seven-root collector
  rows and picks the lowest `H30000` best-periodic MSE per system across all
  checked-in roots, seeds, and periodic cadences; it selects block-diagonal
  LISTA on `14/15` systems (`sc=6e-3` on `10`, `sc=3e-3` on `4`) and dense
  LISTA only on `dysts:QiChen`. Use that as the preferred `H30000` visual
  appendix. The older dense-heavy `H30000` shared-batch LISTA packet remains
  useful only as a selector-sensitivity artifact; it does not replace the
  root-level aggregate result that still favors block-diagonal LISTA
  (`sc=6e-3`) at `H30000`.
- The seed-`10` Dysts long-horizon benchmark reevaluation packet under
  [results/dysts_long_horizon_eval_20260414](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414)
  is now complete. The refreshed collector summary
  [summary.md](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/collect/summary.md)
  reports `750/750` complete tasks and `0/750` pending or invalid tasks.
- The cache-length infrastructure bug is fixed and no longer the blocker.
  Cache prebuild `9273655`, validation `9273656`, collector `9273658`, and
  replacement validation probe `9273675` all completed cleanly, so the only
  remaining issue at that point was rescue coverage on the failed `9273657`
  array tasks.
- The concrete rescue blocker is now identified and fixed in
  [skae/config.py](/home/mila/l/lia/skae/skae/config.py): older checkpoints
  serialize legacy environment fields such as `ENV.COMPETITIVE_LV.SYSTEM_SEED`,
  and the prior `Config.from_dict()` path rejected those unknown keys before
  evaluation began. A previously failing zero-sparse MLP reevaluation
  (`dysts:Chua`, seed `0`) now completes cleanly after that compatibility
  patch.
- Rescue pass `1` is complete. `9278881` and recollect `9278882` both
  finished cleanly, and the queue metadata remains in
  [rescue_pass1_queue_record.json](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/queue/rescue_pass1_queue_record.json).
- The Dysts long-horizon benchmark is now complete and decision-grade as
  supporting evidence: dense LISTA is the best aggregate benchmark root at
  `H5000/H10000` (`0.1285/0.9778`), while block-diagonal LISTA `sc=6e-3` is
  best at `H20000/H30000` (`1.9150/2.2720`). Both MLP controls are now fully
  measured too: sparse MLP `0.1953/1.2373/3.2524/3.6981` and zero-sparse MLP
  `0.2474/1.4564/3.2354/3.7893`.
- The paper Dysts launcher now supports the missing block-diagonal MLP family
  directly. `generic_sparse_blockdiag` is now a paper benchmark variant in
  [skae/benchmarks/paper_benchmark_manifest.py](/home/mila/l/lia/skae/skae/benchmarks/paper_benchmark_manifest.py),
  and
  [scripts/queue_paper_followup_recipes.sh](/home/mila/l/lia/skae/scripts/queue_paper_followup_recipes.sh)
  no longer fails if the historical selected-`dt` table is absent; it falls
  back to benchmark-default `dt` values and keeps the comparison anchors
  stable for partial reruns.
- The Dysts block-diagonal MLP extension is now complete at the paper
  collector horizons under
  [results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415).
  Replacement wrapper `9282328`, collector `9282331`, and compare jobs
  `9282332-9282334` all finished, and replacement training array `9282330`
  ended with `299/300` successful tasks. The lone historical miss was
  `generic_sparse_blockdiag_ns200k_sc3em3` on `dysts:Dadras`, seed `0`
  (`9282330_150`), which failed with a CUDA uncorrectable ECC error on a
  Quadro RTX 8000 rather than a model-side crash; retry `9286093_150` later
  cleared that gap and enabled the full long-horizon packet below. On the Dysts-only
  `H100/H500/H1000` packet, system-median best-periodic MSE is
  `9.227e-05 / 0.001412 / 0.004684` for `sc=3e-3` and
  `7.454e-05 / 0.001399 / 0.004821` for `sc=6e-3`; among the two new roots,
  `sc=6e-3` is slightly better at `H100/H500` while `sc=3e-3` is slightly
  better at `H1000`.
- The matching long-horizon custom-root Dysts packet is now also complete under
  [results/dysts_long_horizon_eval_mlp_blockdiag_20260415](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_mlp_blockdiag_20260415).
  Retry `9286093_150` cleared the earlier hardware-transient `dysts:Dadras`
  gap, launcher `9286094` completed, and the chained jobs `9289755-9289758`
  finished cleanly. The collector
  [summary.md](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_mlp_blockdiag_20260415/collect/summary.md)
  reports `300/300` complete tasks and `0` pending tasks at
  `H5000/H10000/H20000/H30000`.
- The seven-root Dysts architecture audit is now complete. Aggregate median
  best-periodic MSE remains dense LISTA `0.1285/0.9778` at
  `H5000/H10000` and blockdiag LISTA `sc=6e-3` `1.9150/2.2720` at
  `H20000/H30000`. The new blockdiag-MLP roots land at
  `0.1501/1.1401/3.0536/3.5891` (`sc=3e-3`) and
  `0.1945/1.2761/2.9519/3.4785` (`sc=6e-3`), so they sharpen the fairness read
  but do not overturn the long-horizon headline. The strongest blockdiag-MLP
  root is the runner-up aggregate model at `H5000/H10000`, while neither
  blockdiag-MLP root wins any systems at `H20000/H30000`.
- The fixed-`17` matched hard-init MLP control follow-up is no longer just
  queued. Under
  [results/transition_rich_hardinit_mlp_controls_seed10_20260416](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416),
  launcher `9285895`, initial array `9285897`, pass-`0` collect / resolve /
  advance `9285898 -> 9285899 -> 9285900`, and rescue pass `1`
  `9291399 -> 9291400 -> 9291401 -> 9291402` have all finished. Pass-`1`
  [dt_resolution/pass1/dt_resolution.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/dt_resolution/pass1/dt_resolution.md)
  shows all `51/51` arm-system pairs as `accepted_default`, so the rescue /
  `dt` blocker is closed. The finalized forecasting summary under
  [collect_pass1/forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/collect_pass1/forecasting_summary.md)
  reports system-median best-periodic
  `H100/H500/H1000 = 0.0082 / 0.0260 / 0.0273` for the sparse hard-init MLP,
  `0.0094 / 0.0359 / 0.0383` for the structured blockdiag hard-init MLP, and
  `0.5704 / 2.6733 / 3.8044` for the tanh / no-shrink hard-init control. That
  strengthens the induced-sparsity story while weakening any architecture-only
  hard-init forecasting claim.
- The replacement hard-init interpretability chain is now complete after a
  small merge-reader patch that normalizes empty CSV cells. Shards
  `9304602-9304604` had already finished cleanly; the patched merge reran as
  `9304747` and the dependent summary as `9304748`, producing
  [interpretability_final_pass1](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/interpretability_final_pass1)
  with `13,554` rows and `0` failures plus the finalized matched-sampling
  comparison in
  [final_comparison_pass1](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1).
  On the paper slice (`absolute:0.001` / `deep`), the two sparse hard-init MLP
  controls are almost tied: blockdiag sparse MLP gives
  `0.0082 / 0.0252 / 0.0264` at `H100/H500/H1000`, plain sparse MLP gives
  `0.0082 / 0.0260 / 0.0273`, both have `H(B|S_abs)=0.0000`,
  `H(S_abs|B)=0.2068`, `U_exact ~= 0.98`, and `H(B|F_abs)=0.0000`, while the
  tanh / no-shrink control remains far worse. That closes the matched
  hard-init fairness table and says induced sparsity matters more than the
  specific sparse-encoder architecture in this oversampled setting.
- The new operator-selection mechanism package is also complete under
  [results/transition_rich_operator_selection_hardinit_matched_20260418](/home/mila/l/lia/skae/results/transition_rich_operator_selection_hardinit_matched_20260418).
  Smoke `9304650`, shards `9304655-9304659`, and merge `9304660` all finished
  cleanly, and the merged packet writes `56,538` rows with `0` failures. This
  is the first direct held-out `A_global` versus `A_basin` versus
  `A_support/family/group` study with count-matched random controls,
  latent-kmeans controls, and masked-`K` projections on the matched hard-init
  root family. Its headline result is negative for the strong paper claim:
  even oracle basin-conditioned fits do not beat one global latent law on the
  fixed `17` systems, and the best support-family fits beat random controls
  but still remain worse than global. The execution blocker is therefore
  closed; the remaining blocker is claim positioning.
- The reframed centered-chart mechanism packet is now complete under
  [results/transition_rich_centered_chart_mechanism_20260420](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420).
  Shards `9310546-9310548` and merge `9310549` all finished cleanly, and the
  merged packet writes `74,369` rows with `0` failures. This reruns the
  mechanism question with centered local charts, depth strata, and the actual
  dense `tanh` / no-shrink MLP control. On `relative:0.1` exact support,
  `persistent_current`, and deep `q4` states, centered support-conditioned
  local slopes beat the learned global `K` on `93.1%` of evaluated blockdiag
  LISTA rows (`130`), `98.6%` of dense LISTA rows (`141`), and `100%` of the
  dense no-sparsity MLP rows (`140`). Deep support-gated `K` is also strongly
  positive, especially for blockdiag LISTA, where q4 input-gated/global-`K`
  and block-submatrix/global-`K` both win on `100%` of evaluated rows.
- The updated paper-safe reading is therefore more specific. The April 18
  zero-intercept operator-selection failure was mainly a charting/comparator
  problem, not evidence that local laws were absent. Once the read is done in
  centered local charts, support-, family-, and basin-conditioned local laws
  appear across much of the fixed `17`, including most of the proxy-labeled
  benchmark. But the dense no-sparsity MLP also shows the same deep centered
  local-law effect, so the paper still cannot claim that LISTA-style induced
  sparsity uniquely creates those local laws. The defensible claim is now:
  induced sparsity improves basin-support identifiability, stability, and
  forecasting; centered local-law evidence is real but not LISTA-specific; and
  the cleanest direct support-gated `K` interpretation is the blockdiag LISTA
  case. Boundary-adjacent `q1` states remain the main negative slice.
- The non-oracle self-routed forecasting packet is now complete rather than
  merely queued. It is implemented in
  [tools/evaluate_transition_rich_self_routed_forecasting.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_self_routed_forecasting.py)
  with shard / merge launchers in
  [scripts/run_transition_rich_self_routed_forecasting.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_self_routed_forecasting.sh),
  [scripts/merge_transition_rich_self_routed_forecasting_shards.sh](/home/mila/l/lia/skae/scripts/merge_transition_rich_self_routed_forecasting_shards.sh),
  and
  [scripts/queue_transition_rich_self_routed_forecasting_shards.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_self_routed_forecasting_shards.sh).
  Smoke validation is complete under
  [results/transition_rich_self_routed_forecasting_smoke_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_smoke_20260420)
  with `270` rows and `0` failures, and merge-path validation is complete
  under
  [results/transition_rich_self_routed_forecasting_merge_smoke_20260420/merged](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_merge_smoke_20260420/merged).
  This is the direct deployment-facing test of whether the model's own support
  or support family can route forecasting better than one global `K` without
  oracle basin labels. The full fixed-`17` packet is complete under
  [results/transition_rich_self_routed_forecasting_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420),
  with the original root-only queue `9314170-9314173`, the first coarse
  seed-split `long-cpu` queue `9314196-9314214`, the temporary `main-cpu`
  workaround `9314400-9314406`, and the `12h` long-queue retry
  `9314431-9314437` all canceled after scheduler inspection. The final
  successful queue was the one-seed `long-cpu` submission:
  `9314443-9314472` at `03:00:00`, plus dependency-held merge `9314473`.
  `sacct` now shows every shard and the merge as `COMPLETED 0:0`; shard
  elapsed times ranged from `00:39:51` to `02:08:48`, and the merge finished
  in `00:00:20`. The merged packet writes `510/510` completed runs,
  `24,600` rows, and `0` failures. At the time, the paper-facing Table 2
  display used `H100/global` IQM ratios plus paired Wilcoxon/Holm counts
  because that was where the packet gave the strongest confirmatory
  non-oracle routing evidence. That display is now superseded by the April 30
  `H1000` Table 2 update. The
  evaluator was also patched for resumable intra-shard reruns with atomic per-spec
  flushing; that applies to future reruns rather than retroactively to the
  completed queue. The resume path is compute-validated too: validation job
  `9315112` completed in `16s` and confirmed that rerunning the completed
  one-spec smoke shard skips work immediately with `1/1` completed.
- The fixed-`17` LISTA phase-portrait handoff packet is now complete at
  [H1000/H3000/H5000](/home/mila/l/lia/skae/docs/figures/fixed17_lista_phase_portraits_20260414/fixed17_h1000_h3000_h5000_lista_phase_portraits_manifest.json).
  It writes one figure per system per horizon (`51` total) plus per-system
  selection metadata, selecting each system's run by the lowest saved
  `H1000` best-periodic mean across collected transition-rich LISTA rows and
  then reusing that run's saved `H1000` best-periodic mode for all three
  horizons. The packet spans `8` distinct LISTA roots, so it should be read
  as supporting presentation material for coauthors and appendix use rather
  than as new causal shortlist evidence.
- The one-seed `17 x 2` LISTA basin-partition sweep at default `dt` was
  launched under
  [transition_rich_basin_partition_20260407](/network/scratch/l/lia/skae/transition_rich_basin_partition_20260407),
  with the `dt`-rescue continuation already dependency-chained behind it. That
  April 7 default-source queue used the older `200k` training budget and
  should now be read as a legacy long-budget LISTA root rather than the
  forward default. The submitted rescue task tables in that same chain were
  rewritten to `20k`.
- Forward interpretability queues should now use `20000` training steps and `1` seed while
  we are still diagnosing model behavior and choosing the right metrics; only
  the final locked paper rerun on this branch should return to `200000` and `10` seeds,
  except for the now-completed narrow default-sampling `v6` / `v7`
  forecast-floor follow-up, which did not reopen the shortlist.
- An April 13 audit of the last 7 days of paper-critical SLURM work found no
  finished packets missing from this file. The required cleanup is status
  hygiene: the clean tanh / no-shrink control and the `v7`-screened method
  axes are complete and should no longer be described as pending.
- The seed-`0` hard-init follow-up is now complete under
  [transition_rich_basin_partition_hardinit_seed0_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409),
  with chain `9209614 -> 9209615 -> 9209616` completed cleanly. Forecasting
  summary:
  [forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409/collect/forecasting_summary.md).
  Interpretability summary:
  [interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409/reduce/interpretability_summary.md).
- The hard-init read is useful but not uniform. The block-diagonal hard-init
  variant is the clearer interpretability positive: at `absolute:0.001` on
  deep-basin states it improves `H(S_abs|B)` (`1.4297 -> 1.3493`), `U_exact`
  (`0.7181 -> 0.7447`), `H(F_abs|B)` (`0.1129 -> 0.1018`), own-basin projection
  ratio (`25.5197 -> 7.7018`), and wrong-support ratio (`0.7599 -> 0.3034`)
  with nearly neutral forecasting. The dense `p64` hard-init variant is more
  mixed on raw support compression, but it improves forecasting strongly
  (`H1000` system-median best `0.1358 -> 0.0794`) and also reduces the
  damage from deep-basin canonical-support interventions
  (`own/base 9.9799 -> 3.0431`, `freeze/base 0.8926 -> 0.6768`).
- The first post-hard-init cross-root paper-facing reduction is now already
  launched under
  [transition_rich_post_hardinit_crossroot_eval_20260409](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409).
  Its phase-`1` dependency chain was `9210427` (`collect_tr_crossroot`,
  `afterok:9209616`) -> `9210429` (`tr_interp_crossroot`,
  `afterok:9210427`). Both jobs completed. Its selected comparison bundle is fixed in
  [selected_roots.txt](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409/root_specs/selected_roots.txt).
  That queued bundle is the minimal paper-facing comparison set: strongest
  current `v5` forecast-retaining and exact-support roots, strongest current
  `v6` identifiability follow-ups, matched `v1` sparse MLP control, and the
  hard-init packet's base / variant pairs.
- That first submitted cross-root phase-`1` reduction was not usable.
  `9210429` wrote `0` rows and `17` failures under
  [failures.json](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409/interpretability/failures.json).
  The log shows `ROOT_LABELS_CSV` arrived as only `v5_blockdiag_signsplit`,
  and every attempted row then failed checkpoint load because the saved
  block-diagonal LISTA checkpoints still expose `encoder.We.*` keys while the
  current loader expects the newer `precode_module` / `dict_param` layout.
  That run should be treated only as a failed first attempt, not as evidence.
- That clean rerun has now completed under
  [interpretability_rerun_compat_20260409](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409/interpretability_rerun_compat_20260409)
  as job `9211252`, with `4131` interpretability rows and `0` failures. The
  rerun validates the same shortlist choice already suggested by the hard-init
  packet: the forecast-retaining interpretability finalist is the
  block-diagonal hard-init root, the stronger forecasting / intervention
  tradeoff is the dense `p64` hard-init root, and the matched sparse MLP
  control does not beat either finalist on the main branch objective.
- The study-plan-aligned state-level reducer is now smoke-validated on the
  historical native trio under
  [transition_rich_interpretability_smoke_20260409/native_seed0](/home/mila/l/lia/skae/results/transition_rich_interpretability_smoke_20260409/native_seed0).
  Its current paper consequence is sharper than the earlier support-group
  reducer: deep-basin `H(B|S_abs)` is already approximately zero for both dense
  and block-diagonal LISTA on that subset, but `H(S_abs|B)` remains large unless
  supports are forced into aggressive top-`k` masks. The active positive is
  basin purity, not exact-support uniqueness.
- The next study-plan metric tier is now also smoke-validated on that same
  native trio under
  [transition_rich_interpretability_smoke_20260409/native_seed0_v2_family_projection](/home/mila/l/lia/skae/results/transition_rich_interpretability_smoke_20260409/native_seed0_v2_family_projection).
  It adds greedy support-family clustering plus deep-basin canonical-support
  projection diagnostics. Its current paper consequence is that the branch now
  has a plausible family-level positive even before the new ablations land:
  family entropy within basin is already much lower than exact-support entropy
  on the native trio, but the exact canonical-support intervention still hurts
  one-step prediction even when the wrong-basin intervention is catastrophic.
  That means the defensible near-term story is basin-selective support
  families or dominant groups, not yet one exact canonical support per basin.
- All submitted fixed-`17` interpretability packets `v1-v6` are now complete
  through arrays, collect, resolve, and interpretability reduction.
- The completed packet ladder now covers the full planned shortlist sweep on
  this branch: `v1` initial dense / blockdiag / MLP control, `v2`
  HyperLISTA and `2 x basin-count` blocks, `v3` reset-policy and dynamics-
  aware reencoding, `v4` structured and soft-block penalties, `v5` sign-split
  plus latent-size / block-count sweeps, and `v6` restrained pre-code plus
  decoder coherence.
- `v5` is the sign-split shortlist tier in concrete terms:
  hard block-diagonal and dense soft-block LISTA families with sign-split
  codes, `2` versus `4` LISTA refinement loops, doubled block-count variants,
  and latent-size sweeps including `p=64` and `p=128`.
- `v6` is the identifiability follow-up to `v5`:
  the same sign-split shortlist families with either a restrained linear
  pre-code, a decoder-coherence penalty, or both. Decoder coherence here
  means penalizing off-diagonal similarity among normalized decoder atoms so
  the decoder dictionary has fewer redundant atoms and fewer interchangeable
  exact supports.
- The best current shortlist exact-support result is the `v5` root
  `lista_dense_softblock_signsplit_p64_basin_partition`. On deep-basin states
  at `absolute:0.001`, it reaches `mean H(S_abs|B)=0.7719`,
  `mean U_exact=0.8064`, and `mean H(F_abs|B)=0.0521` while still keeping
  `16/17` systems under the `H1000` good-forecast gate. This is the first
  branch result that compresses exact supports within basin strongly enough to
  matter without collapsing forecasting outright. It is also worth considering `v6` as well.
- The best forecast-retention result across the new shortlist tiers remains
  the `v5` root `lista_blockdiag_signsplit_basin_partition`
  (`H1000 system-median = 0.0119`, `17/17` good systems). The strongest `v6`
  forecast-preserving root is
  `lista_dense_softblock_signsplit_coherence_basin_partition`
  (`H1000 system-median = 0.0585`, `17/17` good systems).
- `v6` resolves the next mechanism tradeoff but does not displace `v5`.
  Coherence-only roots help forecasting; linear-encoder roots help exact-
  support compression; their combination still leaves a large forecasting cost
  and does not beat the `v5 p64` root on the combined frontier. That being said, we should consider `v6` as well.
- The next design-note LISTA tier is now implemented locally as well:
  standard LISTA supports adaptive residual/prior-gap thresholds, groupwise
  base thresholds over inferred latent groups, and explicit free-MLP,
  dictionary-tied, or hybrid tied-plus-residual pre-codes. The fixed-`17`
  transition-rich manifest now has runnable shortlist variants for those
  axes. The block-diagonal adaptive/groupwise-threshold arm plus the dense
  dictionary-tied and hybrid pre-code arms have now all been screened in the
  completed `v7` packet, so they no longer sit in the unrun backlog; they are
  now negative / mixed shortlist provenance.
- The next still-open LISTA design-note axis is now implemented locally too:
  standard LISTA supports fixed-beta momentum refinement, and the fixed-`17`
  transition-rich manifest now includes runnable sign-split momentum variants
  on both the forecast-retaining block-diagonal root and the dense soft-block
  `p64` root. Those runs were also screened in `v7` and did not beat the
  locked finalists, so momentum is no longer a tooling blocker or an unrun
  shortlist gap.
- The stronger soft-block sweep that was still open in the design note is now
  exposed locally too: the fixed-`17` transition-rich manifest now includes
  sign-split dense soft-block `p64` shortlist variants at `5e-4` and `1e-3`
  in addition to the earlier `1e-4` weight. Those runs have now been
  screened in the completed `v7` packet, so the open question is now
  evidence rather than configuration coverage.
- A narrow hard-init follow-up is now complete under
  [transition_rich_basin_partition_hardinit_seed0_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409).
  Its chain `9209614` (training array) -> `9209615` (`collect_tr_bp`) ->
  `9209616` (`tr_interp_reduce`) finished cleanly. This packet was the first
  forward execution of the near-separatrix hard-initialization axis on the
  fixed shortlist. It kept the working `20k` budget, used seed `0` only, and
  compared four roots: the current `v5` block-diagonal and dense soft-block
  `p64` anchors against their corresponding hard-init variants.
- The matched sparse MLP control and the clean tanh / no-shrink `200k`,
  `10`-seed control are now both in hand, so the main missing paper-side step
  is no longer training-side queue completion. It is the state-conditioned
  evaluation read: reduce the locked finalists and controls on the study-plan
  metrics by basin depth / separatrix proximity and decide whether any method
  tie-break remains worth budget.
- One narrow default-sampling LISTA-only refinement around the `v5 p64`
  recipe has now been run. The highest-value remaining method-side axes from
  the design notes included stronger soft-block penalties, momentum LISTA,
  adaptive or blockwise thresholds, dictionary-tied or hybrid pre-codes,
  and group-aware sparse-group shrinkage / top-`k` group-first support
  selection. Those variants are now implemented and at least smoke-screened on
  the fixed `17` systems, so they no longer belong to the “unrun” list.
- That narrow follow-up is now complete under
  [transition_rich_basin_partition_20260410_seed0_smoke_v7](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260410_seed0_smoke_v7)
  as wrapper `9226563`. It keeps the working `20k`, seed-`0` budget and
  queues only unrun shortlist variants around the current winners:
  block-diagonal adaptive/groupwise thresholds, block-diagonal sparse-group
  shrinkage, block-diagonal sign-split momentum, dense soft-block
  dictionary-tied pre-code, dense soft-block hybrid pre-code, denser
  soft-block penalties at `5e-4` and `1e-3`, and dense soft-block sign-split
  momentum. The chain `9226564_[0-135] -> 9226565 -> 9226566 -> 9226567` is
  now complete with `0` reducer failures. Its task table leaves
  `hard_init_oversample` unset, so this packet stays on the default sampling
  regime and remains directly comparable to the MLP controls. The best new
  forecasting root is `lista_blockdiag_sparsegroup_basin_partition`
  (`H1000 system-median best = 0.0846`), while the strongest new deep-basin
  support-compression read is
  `lista_dense_softblock_signsplit_p64_softblock5em4_basin_partition`
  (`H(S_abs|B)=0.6795`, `U_exact=0.8453`, `H(F_abs|B)=0.0634` at
  `absolute:0.001` / `deep`), but that root is missing `1/17` systems after a
  fast training failure. `lista_blockdiag_adaptive_groupwise_threshold` is not
  a serious contender because it is numerically unstable. No new `v7` root
  cleanly dominates both forecasting and support alignment, so treat this wave
  as shortlist provenance rather than as a promotion over the locked
  finalists. The completed hard-init packet remains the separate
  sampling-axis ablation and the main promoted comparator row.
- One narrow default-sampling `200k`, `10`-seed forecasting-only follow-up is
  now complete under
  [transition_rich_basin_partition_v6_v7_200k_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_basin_partition_v6_v7_200k_seed10_20260410).
  It promotes `lista_dense_softblock_signsplit_coherence_basin_partition` and
  `lista_blockdiag_sparsegroup_basin_partition`, the best forecasting roots
  from `v6` and `v7`, to test whether longer training lowers their
  default-sampling forecast floor enough to reopen the shortlist. Wrapper
  `9228393` wrote the `340`-task default table, but its attempted model-wise
  dt-rescue continuation was canceled because that protocol would allow
  different models on the same system to run at different `dt` values and
  would therefore confound the forecasting comparison. The actual fair queue,
  `9228394_[0-339] -> 9228395`, completed on April 11, 2026 and wrote
  `329/340` forecasting rows across both roots and all `17` systems. The
  better root, `lista_dense_softblock_signsplit_coherence_basin_partition`,
  reaches system-median best-periodic `H100/H500/H1000 = 0.0416 / 0.0761 / 0.0796`;
  `lista_blockdiag_sparsegroup_basin_partition` lands at
  `0.0437 / 0.1142 / 0.1193`. This is not a full branch reopen; it is a
  completed long-budget forecasting floor check at matched system-level `dt`,
  and it is negative for promotion. The coherence root remains slightly worse
  than the promoted dense hard-init finalist at all three horizons
  (`0.0196 / 0.0733 / 0.0775`) and worse than the matched sparse MLP control
  at `H500/H1000` (`0.0614 / 0.0608`). Because this packet is forecasting-
  only, it adds no new basin-support reduction. Return the remaining branch
  budget to state-conditioned evaluation rather than more training.
- The main remaining evaluation-side gaps from
  [docs/planning/interpretability_study_plan.md](/home/mila/l/lia/skae/docs/planning/interpretability_study_plan.md)
  have narrowed after the corrected reviewer-response reruns. Controlled-
  transfer switch-timing diagnostics now give a useful dense LISTA exact
  `topk:8` support-switching read, while true Jacobian/eigendirection
  diagnostics are mixed and should remain secondary. Remaining paper-side work
  is mainly merge/artifact finalization for the completed
  periodic-support-refresh/no-refresh ablation, final state-conditioned
  long-horizon forecasting figures at
  `H100/H500/H1000` by basin depth / separatrix proximity, seed/threshold
  robustness only if the transfer branch becomes main-text evidence, and
  basis-aware or similarity-aligned operator-family wording if we go beyond
  the current support-routing claim.
- The reducer-side tooling for those evaluation items now exists locally:
  canonical wrong-support rollout metrics, first-switch timing summaries,
  sampled effective-Jacobian family summaries, and optional visual artifacts
  for support families are implemented in the state-level interpretability
  reducer. The remaining gap is no longer missing code; it is running those
  diagnostics on the completed shortlist roots and deciding how they change
  the final claim. One caveat after the April 9 basis-aware update: the live
  reducer already supports the **raw** operator-family and Jacobian-family
  summaries, but it does **not** yet implement the new similarity-aligned
  operator distances, eigendirection comparisons, or invariant-subspace-angle
  metrics. Those alignment-aware diagnostics remain a tooling gap.
- The overnight dependency chain behind that root is now effectively complete
  through pass `6`, and its scientific consequence is stronger default-`dt`
  evidence rather than smaller-`dt` rescue evidence:
  [collect_pass0](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/collect_pass0/forecasting_summary.md)
  already reports `17/17` good systems at `H1000` for both dense and
  block-diagonal LISTA on the fixed shortlist, and the rescue-only `20k`
  reruns keep all `16/16` rerun systems below the same gate by system median
  for both roots.
- The same overnight chain also exposed a queueing / collection bug rather
  than a scientific need for more halving: the collector preserved `env_dt`
  only for `multiwell_strong_transition`, so the resolver kept re-emitting
  default-`dt` task tables for `gated_*` and `claude:*` arms. That collector
  gap is now fixed locally in
  [collect_forecasting_roots.py](/home/mila/l/lia/skae/tools/collect_forecasting_roots.py).
  Do not interpret the completed overnight waves as evidence for a
  smaller-`dt` effect.
- Natural live-sweep `evaluation_best/.../rollout_artifacts.pt` outputs have
  not appeared yet; the current native-trio read comes from manual compute-node
  reevaluation of saved checkpoints under
  [manual_eval](/network/scratch/l/lia/skae/transition_rich_basin_partition_20260407/manual_eval).
- The first manual native-system read from that live sweep already says the
  native trio is not bottlenecked by default `dt` under the current user-facing
  rescue rule, and it already gives the branch's key positive signal:
  support groups remain basin-pure in every inspected dense / block-diagonal
  LISTA arm.
- The current lead blocker is therefore no longer queue completion,
  step-size rescue, or missing implementation support. The locked `200k`,
  `10`-seed LISTA-vs-control packet is now fully reduced, but it is still a
  mixed-regime comparison because the promoted LISTA roots use hard-init
  oversampling while the completed MLP controls in that packet use the
  standard training sampling regime. The matched-sampling hard-init control
  packet under
  [results/transition_rich_hardinit_mlp_controls_seed10_20260416](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416)
  is now forecasting-finalized and says the hard-init forecasting gain is not
  LISTA-exclusive: the sparse hard-init MLP control reaches
  `H100/H500/H1000 = 0.0082 / 0.0260 / 0.0273`, the structured blockdiag
  hard-init MLP reaches `0.0094 / 0.0359 / 0.0383`, and both beat the locked
  hard-init LISTA forecasting winner at `H1000` (`0.0516`), while the clean
  zero-sparse hard-init control is much worse. The replacement sharded
  interpretability rerun is also complete, so the remaining blocker is no
  longer missing reduction output. The remaining claim-calibration work is to
  keep the sampling-regime split explicit, decide how prominently to keep the
  block-diagonal hard-init forecast-retaining companion result, and decide
  whether the final wording should stay at exact-support reuse or soften to
  family / dominant-group or symmetry-aware alignment in light of the now-
  completed centered-chart mechanism packet and the now-completed non-oracle
  self-routed / state-conditioned
  `H100/H500/H1000` forecasting read.
- Immediate wrap-up priority should therefore be:
  first update the coauthor-facing docs so the locked hard-init packet is read
  as mixed-regime evidence rather than architecture-only evidence; second
  finish or freeze the remaining matched hard-init MLP control coverage and
  then prepare the paper-facing figures / tables with an explicit
  sampling-regime split; third run only the extra diagnostics still needed for
  wording, chiefly the basis-aware alignment readouts and the final visual
  summaries built from the completed self-routed long-horizon forecast read.
- The concrete post-hard-init evaluation bundle is now complete in phase `1`
  form under
  [transition_rich_post_hardinit_crossroot_eval_20260409](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409).
  It compares `v5` forecast-retaining and exact-support leaders, `v6`
  coherence and restrained-linear follow-ups, the matched non-LISTA control,
  and the hard-init packet's near-separatrix oversampling variants under one
  reduction protocol.
- Rationale for that exact bundle:
  it is the smallest set that can answer the paper-facing causal question
  without spending more training budget first. That bundle has now done its
  job: it fixed the seed-`0` shortlist before the locked multi-seed rerun, and
  the locked packet then answered the causal comparison at the final budget.
- The branch now has the fully reduced final confirmatory rerun in hand. The
  seed-`10` packet is
  [transition_rich_basin_partition_final_seed10_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409):
  default array `9211290_[0-679]`, pass-`0` collect / resolve
  `9211291 -> 9211292`, rescue pass-`1` array `9214917_*`, pass-`1`
  collect / resolve `9214918 -> 9214919`, final reducer `9218036`, and final
  paired summary `9218037`.
  It runs the promoted LISTA roots
  `lista_blockdiag_signsplit_hardinit_basin_partition` and
  `lista_dense_softblock_signsplit_p64_hardinit_basin_partition` against both
  `mlp_sparse_basin_partition_control` and the newly exposed
  `mlp_zero_sparse_basin_partition_control`, which in that locked packet
  should now be read as a zero-`L1` ReLU ablation rather than the clean tanh /
  no-shrink baseline, at the locked `200k`, `10`-seed budget on the fixed
  `17` systems.
- Because those promoted LISTA roots are hard-init oversampled while the two
  MLP controls are standard-sampling, this packet should be read as mixed
  architecture-plus-sampling evidence. It is useful for ranking and for
  motivating a matched-regime follow-up, but it is not the final
  architecture-isolating comparison.
- The early failures in that array are operational rather than scientific:
  tasks `238`, `255`, `267`, `279`, `295`, and `303` all failed on `cn-a009`
  with `torch.AcceleratorError: CUDA error: uncorrectable ECC error
  encountered`.
  Rescue pass `1` resolved that hardware-only cluster; no further rescue rows
  were requested afterward.
- The finalized forecasting summary under
  [collect_pass1/forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/collect_pass1/forecasting_summary.md)
  reports system-median best-periodic `H1000` values of `0.0516` for the
  block-diagonal hard-init LISTA finalist, `0.0775` for the dense soft-block
  `p=64` hard-init LISTA finalist, `0.0608` for the matched sparse MLP
  control, and `0.0909` for the locked packet's zero-`L1` ReLU MLP
  control, with all four roots at `17/17` good systems.
- That zero-`L1` ReLU arm should not be treated as disposable. It is not the
  clean tanh / no-shrink anti-sparsity control, but it is still
  scientifically meaningful because it removes the explicit `L1` penalty while
  retaining ReLU-induced architectural shrinkage. In other words, it is a
  ReLU-only sparsity ablation that can still speak to basin-support
  identification and forecasting quality.
- That locked forecasting summary is useful but not sufficient for the
  coauthor-facing mechanism writeup on this branch. The next paper-facing read
  should expand it to `H100`, `H500`, and `H1000`, then stratify those
  horizons by basin depth / separatrix proximity. The explicit prediction is
  that sparsity-driven gains should be largest deep in a basin and smallest
  near a separatrix if the latent is really preserving basin identity.
- The detailed fixed-`17` per-system forecasting table for the four locked
  roots now lives in
  [docs/EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md).
  By lowest per-system `H1000`, the block-diagonal hard-init LISTA root is
  best on `7/17` systems, the dense `p64` hard-init LISTA root is best on
  `5/17`, the sparse MLP control is best on `5/17`, and the zero-`L1`
  ReLU MLP control is best on `0/17`.
- A new working-budget zero-sparsity no-shrink control screen is now also
  complete under
  [transition_rich_zero_sparse_control_noshrink_20k_seed3_20260410](/home/mila/l/lia/skae/results/transition_rich_zero_sparse_control_noshrink_20k_seed3_20260410)
  as `9223056_[0-50] -> 9223057 -> 9223058`. It keeps all `17/17` systems
  inside the default-`dt` `H1000 < 50` gate, but its system-median
  best-periodic forecasting is weak (`H100/H500/H1000 = 1.8317 / 3.5797 /
  4.1857`), so it strengthens the sparse-vs-zero-sparse framing without
  changing the branch ranking.
- The older locked-budget zero-sparsity control expansion under
  [transition_rich_zero_sparse_control_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_zero_sparse_control_seed10_20260410)
  as array `9221521_*` is misconfigured relative to the requested clean
  tanh / no-shrink control because its task table maps
  `mlp_zero_sparse_basin_partition_control` to `generic_sparse` rather than
  `generic_no_shrink`. But it should still remain in the paper-side evidence
  stack as a ReLU-only ablation, not be thrown away outright. Operationally it
  now runs through collect / resolve pass `4`, still accepts default `dt` on
  all `17/17` systems, and remains at system-median best-periodic
  `H100/H500/H1000 = 0.5764 / 2.0556 / 2.6532` with no requested smaller
  `dt`.
- The corrected locked-budget tanh / no-shrink control is now complete under
  [transition_rich_zero_sparse_tanh_control_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_zero_sparse_tanh_control_seed10_20260410)
  as wrapper `9224111`, using a fresh results tag so the task table is rebuilt
  from the corrected manifest mapping. That launcher has now completed and the
  fresh task table is verified to use `config_name=generic_no_shrink`. The
  default pass now completes as `9224263_* -> 9224264 -> 9224265`; despite one
  fast array failure, the collector writes `169` rows, all `17/17` systems
  accept default `dt`, and the packet reports system-median best-periodic
  `H100/H500/H1000 = 0.5763 / 1.7924 / 2.4279`. This is the clean no-shrink
  control. It is slightly stronger than the ReLU-only zero-`L1` ablation but
  remains much weaker than the locked sparse LISTA roots, which strengthens
  the broader induced-sparsity claim.
- The final paired state-level comparison under
  [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md)
  is now the branch's main mixed-regime read. On the selected `absolute:0.001`
  / `deep` slice, the dense `p64` hard-init LISTA root beats the
  standard-sampling sparse MLP control in `H(S_abs|B)` (`0.2449 -> 0.0543`), `U_exact`
  (`0.9772 -> 0.9923`), and `freeze/base@20` (`0.3923 -> 0.1691`), with
  paired wins on `15/17`, `14/17`, and `16/17` systems while remaining
  forecast-competitive. The block-diagonal hard-init root instead carries the
  best finalized locked-packet forecasting value (`H1000 = 0.0516`) plus
  better freeze robustness than both standard-sampling MLP controls, but it
  loses exact-support compression to the matched sparse MLP control and
  therefore should be written as the forecast-retaining companion rather than
  the lead basin-support win.
- The remaining paper-critical evidence gap is therefore still partly the
  LISTA-versus-MLP comparison: the matched-sampling hard-init forecasting read
  is now complete and already says sparse hard-init MLPs can inherit the
  hard-init forecasting gain, while the structured blockdiag hard-init MLP is
  also competitive. The stronger induced-sparsity claim is no longer blocked
  on training-side queue completion: the clean tanh / no-shrink `200k`,
  `10`-seed control is in hand, and the hard-init control packet now has full
  pass-`1` forecasting coverage plus all `51/51` arm-system pairs accepted at
  the default `dt`. The remaining evaluation-side gap is now the missing
  matched-hard-init state-level interpretability reduction, plus the planned
  `H100/H500/H1000` depth-versus-separatrix read needed to keep the fairness
  language precise.
- On that same selected slice `H(B|F_abs)` is `0.0000` for all four roots, so the
  locked-packet discrimination is no longer family entropy. It is exact-
  support fragmentation, intervention stability, persistence, and forecasting.
- Mila rejected the fully pre-expanded rescue chain for that packet under
  `AssocMaxSubmitJobLimit`, so the live confirmatory execution now uses an
  incremental queueing pattern instead of a fully chained one: run the default
  array first, inspect resolve output, and only then submit the specific
  rescue pass that is actually needed. The branch now has a dedicated one-pass
  launcher for that path at
  [queue_transition_rich_basin_partition_rescue_pass.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_basin_partition_rescue_pass.sh).
- The final paired paper readout is also prepared locally at
  [summarize_transition_rich_final_comparison.py](/home/mila/l/lia/skae/tools/summarize_transition_rich_final_comparison.py),
  and the branch has now used it to emit one combined LISTA-vs-control readout
  under
  [final_comparison_pass1](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1)
  instead of another manual aggregation pass.
- The same packet now also has an auto-advance watcher at
  [advance_transition_rich_basin_partition_packet.sh](/home/mila/l/lia/skae/scripts/advance_transition_rich_basin_partition_packet.sh):
  job `9211747` completed after `9211292` and emitted the pass-`1` rescue
  array `9214917_*`. That same path then queued the final interpretability
  reduction and paper-facing comparison automatically and finalized at pass
  `1`; the recorded status is in
  [advance_pass1.json](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/automation/advance_pass1.json).
- Run that bundle in two phases:
  phase `1` is already queued with the existing reducer on raw support-family /
  projection / operator-family / Jacobian-family metrics; phase `2` remains a
  tooling task and should rerun the same bundle only after the reducer gains
  the April 9 basis-aware alignment readouts (similarity-aligned operator
  distances, eigendirection similarity, and symmetry-aware support alignment).
- Step-size rescue should therefore no longer be written as an open blocker on
  the one-seed LISTA shortlist. The current queue-era evidence already says the
  fixed `17`-system LISTA packet is operationally fine at default `dt`; what
  remains missing is the matched standard-MLP contrast and any follow-up
  ablation needed to explain why high-purity recurring supports still fail the
  stronger local-linearity test.
- Operational queue caveat:
  `v4` still carries a launcher-level failure record because `9202903` hit the
  submit cap, but all emitted default / rescue / reducer jobs completed
  successfully. There is no live backlog now, so any next packet can be
  submitted deliberately rather than under queue-pressure triage.
- Do not write this branch as if it needs to prove the stronger
  chart-switch-localization claim or a sparse-only mechanism claim. Those
  older questions are supporting context; the paper decision for this branch is
  whether LISTA gives cleaner basin-separated support structure than the
  standard MLP control on the fixed shortlist.
- The current ablation-design source of truth for the next interpretability
  loop is
  [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md).
  It should guide matrix-structure, reset, loss, and diagnostic sweeps for
  plan items `3` and `4` until those axes are executed and written back into
  the live docs.
- The forward interpretability scope is now frozen to `17` systems and no
  others:
  - native transition-rich trio:
    `multiwell_strong_transition`, `gated_local_linear`,
    `gated_transfer_linear`
  - Claude-catalog subset:
    `arrested_spiral`, `cal_asymmetric_3`, `cal_high_cross_3`,
    `cal_hexagon_6`, `cal_octagon_8`, `cal_pentagon_5`, `cal_square_4`,
    `checkerboard_potential`, `duffing_triple_well`, `snic_multi`,
    `transition_routes_4`, `var_depth_gradient_4`, `var_diamond_4`,
    `var_l_shape_5`
- The conceptual inventory in
  [docs/planning/transition_rich_system_inventory_20260406.md](/home/mila/l/lia/skae/docs/planning/transition_rich_system_inventory_20260406.md),
  the elite sketches in
  [docs/planning/transition_rich_elite_system_sketches_20260406.md](/home/mila/l/lia/skae/docs/planning/transition_rich_elite_system_sketches_20260406.md),
  and the companion figures under
  [docs/figures/transition_rich_inventory_20260406](/home/mila/l/lia/skae/docs/figures/transition_rich_inventory_20260406)
  now serve as historical design-space provenance only. Do not use their
  broader `16`-system / `8`-system shortlists as the live experiment scope.
- The worktree now also has an artifact-backed audit of the already
  implemented Claude catalog in
  [docs/planning/claude_catalog_audit_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_audit_20260407.md)
  with a companion figure under
  [docs/figures/claude_catalog_audit_20260407](/home/mila/l/lia/skae/docs/figures/claude_catalog_audit_20260407):
  - `112` systems are registered in `skae/claude_catalog`
  - the combined grounded screen now covers `83`
  - `29` implemented systems remain unscreened
  - `12` screened systems are now accepted under the fast-screen gates, with an
    `8`-system strict-crossing core
  - `var_depth_gradient_4` is now part of that strict core and `cal_hexagon_6`
    is now part of the relaxed-accept subset after targeted retune refreshes,
    while `cal_octagon_8` remains a screened high-basin near miss
  - the companion packet now includes a combined audit atlas plus separate
    strict-crossing and accepted-pass portrait galleries
  - the implemented Claude catalog should therefore be treated as a grounded
    small benchmark pool plus retune frontier rather than as an already
    validated large benchmark packet
  - use that audit to describe what is grounded and what remains frontier
  - do not treat the older `6`-system packet in
    [docs/planning/claude_catalog_handoff_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_handoff_20260407.md)
    as the active recommendation; forward Claude-catalog experiments are now
    restricted to the fixed `14`-system subset above
  - the same supersession is now also written in senior-coauthor-facing
    language in
    [docs/planning/claude_catalog_senior_review_packet_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_senior_review_packet_20260407.md),
    so the protocol and active scope are readable without relying on internal
    code names
- The first frozen `2D` pair remains useful, but it is now understood as the
  secondary half of a three-system suite:
  - `gated_local_linear` is now the cleanest oracle-validated chart-switching
    toy
  - `multiwell_strong_transition` is the weaker shared-corridor toy
  - `gated_transfer_linear` is the explicit-transfer stress test where periodic
    decode/re-encode may still help at chart changes, but the oracle state-space
    gap is only modest
- The first implementation tasks are complete locally:
  - write tests
  - add the minimal interface scaffolding
  - calibrate deterministic toy systems
  - freeze two native-plot secondary `2D` candidates
  - implement and calibrate the explicit-transfer toy
- The paper-facing 100-trajectory figure now exists at [transition_rich_100_trajectories.svg](/home/mila/l/lia/skae/docs/figures/transition_rich_20260331/transition_rich_100_trajectories.svg).
- The transfer-system figure packet now also exists under
  [docs/figures/chart_switching_transfer_20260331](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331).
- Short training-entry smokes also pass under [runs/transition_rich_smoke](/home/mila/l/lia/skae/runs/transition_rich_smoke), so the new environments are now verified through the actual training CLI as well as through direct calibration.
- The standardized checkpoint evaluation path now also contains a deterministic flow-branching diagnostic for the frozen `2D` systems, with normalized tolerance sweeps, ground-truth-null thresholds, region-wise breakdowns, and companion `2D` visual artifacts.
- A new oracle state-space benchmark-validity read is now complete under
  [results/transition_rich_oracle_chart_switch_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_chart_switch_20260401).
  Its key consequence is a stricter benchmark ranking:
  - `gated_local_linear` is the clean main-text mechanistic positive
  - `gated_transfer_linear` should be written as the harder explicit-transfer
    stress test, not as the sole chart-switching flagship
  - `multiwell_strong_transition` remains a secondary toy
- A second oracle benchmark-validity read is now also complete under
  [results/transition_rich_oracle_refresh_cadence_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_refresh_cadence_20260401).
  It sharpens the same role split:
  - `gated_local_linear` is the clean periodic-refresh positive
  - `multiwell_strong_transition` is weaker and needs faster refresh
  - `gated_transfer_linear` is strongly cadence-sensitive and should be read as
    a hard transfer stress test whose relevant comparison is stale local charts
- The full seed-robust screening matrix on the calibrated three-system suite is
  now complete:
  - Stage 1 array `9135303` finished with `26/27` completed cells; the single
    failed task was `lista_dense_promoted_stage4 x multiwell_strong_transition x seed_2`
  - collect `9135304` completed `0:0`
  - comparisons `9135305-9135307` completed `0:0`
  - post hoc chart-change attribution `9135358` completed `0:0`
  - post hoc support local-linearity `9135411` completed `0:0`
- The collected read fixes the current paper-facing role split more sharply than
  the oracle-only stage:
  - `gated_local_linear` is the clean learned-model mechanistic positive
  - `gated_transfer_linear` is also a clean forecasting positive, but the
    stronger localization claim on true chart-change windows is negative
  - `multiwell_strong_transition` is not a clean forecasting positive at the
    paper `200k` budget
- The support local-linearity pass is encouraging supporting evidence for the
  branch, but it should be read as prior mechanism context rather than as the
  final fixed-`17` LISTA-versus-MLP reduction.
- The remaining clearly undercovered older mechanism artifacts are corrected competitive-LV support alignment under [results/zero_sparse_mechanisms_20260321/competitive_lv_representation_followup/support_alignment](/home/mila/l/lia/skae/results/zero_sparse_mechanisms_20260321/competitive_lv_representation_followup/support_alignment) with only seeds `0,1,2`, and the direct Kuramoto mode-support audit with only `5` seeds per root and sampling strategy; neither is the lead paper blocker anymore.
- Do not queue another broad benchmark or hard-system rerun by default. The
  immediate work is to turn the collected transition-rich read into a clean
  paper claim and plan any next runs only on the fixed `17`-system shortlist
  above. Do not reopen broader conceptual-inventory or full Claude-catalog
  selection for this branch; the only remaining selection question is the run
  order within the fixed shortlist and whether the single missing dense-LISTA
  `multiwell` seed is worth finishing.


## Consolidated Paper-Facing Families

For the NeurIPS draft's experiments section, prefer the evidence order in
[PAPER_EXPERIMENT_EVIDENCE_MAP.md](/home/mila/l/lia/skae/docs/PAPER_EXPERIMENT_EVIDENCE_MAP.md):
basin-support alignment, support-routed local prediction, long-horizon
forecasting competitiveness, then supporting/falsification diagnostics. The
family map below remains useful for provenance and artifact ownership, but it
should not dictate the paper's result order.

For drafting, compress the completed evidence into the family map below rather than citing each queue-era branch as its own experiment, and make the new lead branch explicit.

| Family | Merge these experiments | Main paper question | Writing rule |
|---|---|---|---|
| **Transition-rich basin partitioning** | tests-first toy-system calibration; fixed `17`-system shortlist; dense / block-diagonal LISTA live sweep; matched standard-MLP control; transition diagnostics; deterministic flow-consistency / flow-branching read; recurring-support local-linearity reuse metrics | Do LISTA encoders define reusable basin-aligned sparse supports on the fixed `17` systems, and is that basin separation stronger than for a matched standard MLP encoder? | This is the lead live family. Write it around basin separation first: count the systems where LISTA support views are basin-pure and reusable, use `gated_local_linear` as the clean positive anchor, treat `gated_transfer_linear` and the Claude subset as stress tests, and report the standard MLP control as the main contrast. |
| **Cross-system forecasting** | fair `200k` benchmark checkpoint family; matched zero-sparsity MLP benchmark extension; full-benchmark block-diagonal dense-opt transfer | What is the benchmark headline versus the MLP anchor once the dense comparator is fixed, and does explicit sparsity help beyond the same MLP with `lambda_sparse=0`? | Cite the fair `200k` benchmark as the supporting quantitative anchor. Do not let it crowd out the new transition-rich branch. |
| **Hard-system forecasting** | `dt`-rescue audit; focused smaller-`dt` Kuramoto/Hopfield follow-ups; long-horizon reevaluation of those same checkpoints; Kuramoto robustness/dimension sweeps; matched hard-system parity; matched block-diagonal fairness controls; higher-basin Hopfield / CLV robustness probes; matched zero-sparsity MLP hard-system extension; corrected `d_z=1024` tanh-dense-baseline redo from `9412218`--`9412222` | Where do step size, structure, and the sparsity penalty help, and where do LISTA families still fail? | Write this as one connected limitation/support family, not as a separate live execution branch. Treat the higher-dimensional redo as negative dense-baseline stress-test evidence, not a positive mechanism result. |
| **Appendix-only provenance** | dense LISTA recipe-selection/tuning sweeps; matched `50k` `v4` four-model audit | What tuning/provenance material justifies the fixed comparator choices and historical benchmark context? | Keep appendix-only. Do not present hyperparameter selection as a main result family. |

## Current Best Evidence

Paper-priority note:
- The experiments section should now be drafted from
  [PAPER_EXPERIMENT_EVIDENCE_MAP.md](/home/mila/l/lia/skae/docs/PAPER_EXPERIMENT_EVIDENCE_MAP.md)
  rather than from queue chronology: basin-support alignment first,
  support-routed prediction second, Dysts long-horizon forecasting third, and
  supporting/falsification diagnostics last.
- The live open branch is now the deterministic transition-rich basin-partitioning plan in [docs/planning/transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md).
- The evidence below remains the fixed supporting packet while that branch is
  being interpreted and written into the paper story.
- Main-text benchmark claims should be drawn from the `200k` results below.
- The older `50k` `v4` matrix is appendix-only historical context unless a reviewer specifically asks for the matched four-model snapshot.
- For the dense MLP-versus-LISTA benchmark, “fair” means matched systems, pass-2 `dt`, `200k` budget, `target_size=256`, `sequence_length=8`, and matched `[64,64]` front-end hidden widths/depths. It is not a strict equal-parameter comparison, because LISTA also keeps its learned recurrent encoder matrix on top of that shared front end.
- A matched zero-sparsity MLP extension of this same benchmark family is now collected under [results/paper_zero_sparse_benchmark_200k_20260321](/home/mila/l/lia/skae/results/paper_zero_sparse_benchmark_200k_20260321); it uses the same `200k` recipe as `generic_sparse_ns200k_best`, except `lambda_sparse = 0.0`, and the current table-facing forecasting packet now uses exact 10-seed coverage for every displayed benchmark and hard-system row. Its current read is mixed: zero-sparsity broadens good-system coverage on some late-horizon slices, but it is no longer the benchmark-median winner at `H1000`.

### Lead live branch: deterministic transition-rich basin partitioning

- The locked execution plan is [docs/planning/transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md).
- The current design-source companion for the unrun ablation axes is
  [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md).
  Treat it as planning ground truth for the next loop over items `3` and `4`,
  not as already-validated evidence.
- Training-budget policy for this live branch:
  use `20000` steps for forward interpretability diagnosis while the metrics
  and recipe are still being set; reserve `200000` only for the final locked
  paper-facing rerun.
- The branch should be written around basin separation on the fixed `17`
  systems, not around best-periodic forecasting.
- It should also not be written as a chart-switch-localization or sparse-only
  mechanism branch. Those are supporting side questions, not the branch
  acceptance criterion.
- The already-running April 7 queue is LISTA-only evidence. The matched
  standard MLP encoder control is still required before the intended fixed-`17`
  branch claim is decision-grade.
- The overnight continuation strengthens the same point: default `dt` is
  already adequate on the one-seed LISTA shortlist, and the repeated overnight
  `20k` reruns should be read as additional default-`dt` robustness checks
  rather than as smaller-`dt` rescue evidence.
- A paper-facing local reduction of the finished LISTA shortlist now exists
  under
  [basin_support_metrics_20260408_v3](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/basin_support_metrics_20260408_v3).
  The canonical prose definition note for these metrics is
  [docs/transition_rich_basin_support_metric_definitions.md](/home/mila/l/lia/skae/docs/transition_rich_basin_support_metric_definitions.md).
  Its branch-level read is clear:
  - mean support-group purity is `0.9883` for block-diagonal LISTA and
    `0.9961` for dense LISTA
  - mean retained-trajectory coverage is `0.8729` and `0.8787`, with
    `15/17` systems above the `0.60` coverage gate for both roots
  - local `H=20` NRMSE beats the matched global fit on `0/17` systems for
    both roots
  - local `H=20` NRMSE beats the shuffled baseline on `0/17`
    block-diagonal systems and `1/17` dense systems, and that lone dense win
    is `claude:duffing_triple_well` at only `0.1172` coverage
  - the current LISTA packet therefore supports a basin-pure recurring-support
    claim much more strongly than a local-linearity mechanism claim
- The first live LISTA basin-partition sweep on the fixed `17`-system shortlist
  was launched at each system's default `dt`, with a dependency-chained
  `dt`-halving rescue continuation if any arm fails `H1000 best-periodic < 50`.
  That already-launched April 7 sweep predates the new `20k` diagnostic-budget
  policy and should not be treated as the forward default.
- The older manual native-trio audit remains useful supporting detail, but it
  should now be subordinated to the full fixed-`17` reduction above:
  - all six dense/block-diagonal native arms pass the default-`dt` rescue gate
  - support groups remain basin-pure in every inspected native LISTA arm
  - `gated_local_linear` and `gated_transfer_linear` stay high-coverage purity
    positives
  - `multiwell_strong_transition` remains the weakest native case because
    coverage is still below the `0.60` gate
- The required test suite now exists and passes, along with the supporting calibration module [skae/transition_calibration.py](/home/mila/l/lia/skae/skae/transition_calibration.py) and the reproducible calibration entry point [tools/calibrate_transition_system.py](/home/mila/l/lia/skae/tools/calibrate_transition_system.py).
- Two native-plot `2D` deterministic candidates are now frozen from ground-truth calibration on the fixed `17x17` grid:
  - `multiwell_strong_transition`
    - `5` endpoint basins
    - per-endpoint crossing fractions `0.500 / 0.500 / 0.500 / 0.500 / 0.585`
    - overall crossing `0.512`
    - label stability `1.000`
  - `gated_local_linear`
    - `3` endpoint basins
    - per-endpoint crossing fractions `0.670 / 0.577 / 0.539`
    - overall crossing `0.599`
    - label stability `1.000`
- The explicit-transfer toy is now implemented and calibrated:
  - `gated_transfer_linear`
    - `3` endpoint basins
    - source-neighborhood transfer fractions `0.333 / 0.444 / 0.333`
    - overall source-neighborhood transfer `0.370`
    - core-retention fractions `1.000 / 1.000 / 1.000`
    - overall core retention `1.000`
    - label stability `1.000`
- The paper-usable mechanics figures for that transfer toy now exist:
  - [gated_transfer_linear_region_map.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_region_map.svg)
  - [gated_transfer_linear_chart_trajectories.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_chart_trajectories.svg)
  - [gated_transfer_linear_endpoint_trajectories.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_endpoint_trajectories.svg)
  - [gated_transfer_linear_transfer_summary.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_transfer_summary.svg)
- The same calibration gives one current caveat:
  - `multiwell_gradient` under the new corridor-aware labeling is not yet a clean low-transition control (`overall crossing = 0.401`, center-basin crossing `0.242`), so it should remain a secondary reference until we decide whether to retune it.
- The branch now also has a paper-usable mechanics figure:
  - [transition_rich_100_trajectories.svg](/home/mila/l/lia/skae/docs/figures/transition_rich_20260331/transition_rich_100_trajectories.svg)
- Updated interpretation of those two frozen systems:
  - they are good secondary toys for transition handling and local chart
    mechanics
- A new oracle chart-switch benchmark-validity read is now complete under
  [results/transition_rich_oracle_chart_switch_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_chart_switch_20260401):
  - `gated_local_linear`
    - oracle-vs-global gain at `H20` on all starts: `0.129974`
    - oracle-vs-fixed-chart gain at `H20` on all starts: `0.0219174`
  - `gated_transfer_linear`
    - oracle-vs-global gain at `H20` on all starts: `0.0206368`
    - oracle-vs-fixed-chart gain at `H20` on all starts: `0.00237432`
    - oracle-vs-global gain on transfer starts: `-0.0557357`
    - oracle-vs-fixed-chart gain on transfer starts: `0.00993001`
  - `multiwell_strong_transition`
    - oracle-vs-global gain at `H20` on all starts: `0.0144757`
    - oracle-vs-fixed-chart gain at `H20` on all starts: `0.00539429`
- Revised benchmark role split:
  - `gated_local_linear` is the cleanest chart-switching positive
  - `gated_transfer_linear` is the explicit-transfer stress test
  - `multiwell_strong_transition` is the weaker shared-corridor toy
- A second oracle benchmark-validity read is now complete under
  [results/transition_rich_oracle_refresh_cadence_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_refresh_cadence_20260401):
  - `gated_local_linear`
    - median dwell: `171`
    - recovered oracle fraction at `H20`: `0.988651` at `c=2`, `0.90577` at
      `c=5`, `0.665843` at `c=10`
    - largest cadence preserving at least `90%` of the oracle-vs-fixed gain on
      all starts: `c=5`
  - `multiwell_strong_transition`
    - median dwell: `92`
    - recovered oracle fraction at `H20`: `0.968223` at `c=2`, `0.858159` at
      `c=5`, `0.639087` at `c=10`
    - largest cadence preserving at least `90%` of the oracle-vs-fixed gain on
      all starts: `c=2`
  - `gated_transfer_linear`
    - median dwell: `33`
    - recovered oracle fraction at `H20`: `0.88425` at `c=2`, `0.503731` at
      `c=5`, `0.0577701` at `c=10`
    - largest cadence preserving at least `90%` of the oracle-vs-fixed gain on
      all starts: only `c=1`
    - on transfer starts, oracle refresh still beats stale local charts but not
      the global fit
- The standardized evaluation stack now has one additional forecast-side diagnostic for these systems:
  - sweep a normalized same-state tolerance on pooled rollout states
  - set the next-step divergence threshold from the ground-truth simulator so the true system is exactly zero under the chosen sweep
  - report close-pair counts, flow-branching rate, next-step / `K`-step divergence severity, and basin-core vs transition-region breakdowns
- First model pass is also fixed:
  - `generic_sparse`
  - matched zero-sparsity MLP
  - promoted dense LISTA
  - `lista_blockdiag` only on the strongest positive case if the screening read is genuinely informative
- The Stage 1 execution wrapper and branch-specific summary collector now exist:
  - [tools/build_transition_rich_screening_tasks.py](/home/mila/l/lia/skae/tools/build_transition_rich_screening_tasks.py)
  - [tools/summarize_transition_rich_screening.py](/home/mila/l/lia/skae/tools/summarize_transition_rich_screening.py)
  - [scripts/collect_transition_rich_screening.sh](/home/mila/l/lia/skae/scripts/collect_transition_rich_screening.sh)
  - [scripts/queue_transition_rich_screening_stage1_20260401.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_screening_stage1_20260401.sh)
- The post hoc chart-change attribution pass now also exists as a separate,
  scientifically cleaner stage:
  - [tools/evaluate_transition_rich_chart_change_attribution.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_chart_change_attribution.py)
  - [tools/collect_transition_rich_chart_change_attribution.py](/home/mila/l/lia/skae/tools/collect_transition_rich_chart_change_attribution.py)
  - [scripts/collect_transition_rich_chart_change_attribution.sh](/home/mila/l/lia/skae/scripts/collect_transition_rich_chart_change_attribution.sh)
  - [scripts/queue_transition_rich_chart_change_attribution_20260401.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_chart_change_attribution_20260401.sh)
- The post hoc support-local-linearity pass now also exists as a separate,
  scientifically cleaner stage:
  - [tools/evaluate_transition_rich_support_local_linearity.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_support_local_linearity.py)
  - [tools/collect_transition_rich_support_local_linearity.py](/home/mila/l/lia/skae/tools/collect_transition_rich_support_local_linearity.py)
  - [scripts/collect_transition_rich_support_local_linearity.sh](/home/mila/l/lia/skae/scripts/collect_transition_rich_support_local_linearity.sh)
  - [scripts/queue_transition_rich_support_local_linearity_20260401.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_support_local_linearity_20260401.sh)
- Undertrained smoke validation exists only as a tooling check, not as a paper
  result, under
  [results/transition_rich_smoke_chart_change_attr_20260401](/home/mila/l/lia/skae/results/transition_rich_smoke_chart_change_attr_20260401):
  it shows a small positive chart-change localization delta but no meaningful
  transfer-trajectory gain, which is exactly the expected “pipeline works,
  science not claimed yet” outcome for a `5`-step smoke root.
- Undertrained support-local-linearity smoke validation also exists only as a
  tooling check, not as a paper result, under
  [results/transition_rich_smoke_support_local_linearity_20260401](/home/mila/l/lia/skae/results/transition_rich_smoke_support_local_linearity_20260401):
  on the collected `generic_sparse` transfer smoke, retained support groups
  cover `0.858` of all trajectories, `0.865` of chart-switch trajectories, and
  `0.700` of transfer trajectories, with `H20` local/global/shuffled latent
  NRMSE `0.0701 / 0.1309 / 0.2018` and weighted endpoint / source-endpoint-pair
  purity `1.000 / 0.891`; that is the right nondegenerate tooling read, but it
  is still not a paper claim because the smoke checkpoints trained for only `5`
  steps.
- The collected Stage 1 screen is now the real model-side read:
  - `26/27` runs completed under
    [results/transition_rich_screening_stage1_20260401](/home/mila/l/lia/skae/results/transition_rich_screening_stage1_20260401)
  - `gated_local_linear` is a clean positive for all three roots, with
    `H1000` best-periodic medians `0.0890151`, `0.000710212`, `0.000783794`
    against no-reencode medians `27651.5`, `2.86748`, `206.989`
  - `gated_transfer_linear` is also a clean positive for all three roots, with
    `H1000` best-periodic medians `1.7954`, `1.80077`, `1.09863` against
    no-reencode medians `6.54401e+20`, `2.45499e+34`, `1.25639e+12`
  - `multiwell_strong_transition` is not a clean forecasting positive at the
    paper budget; `H100` and `H500` best-periodic medians remain unstable and
    enormous across roots
- The collected chart-change attribution pass is now also the real model-side
  causal read on the explicit-transfer toy:
  - summary:
    [results/transition_rich_chart_change_attribution_20260401/summary.md](/home/mila/l/lia/skae/results/transition_rich_chart_change_attribution_20260401/summary.md)
  - all three localization deltas are negative for all three roots, so the
    stronger claim that learned periodic gains localize at true chart-change
    windows is not currently supported
- The collected support local-linearity pass is also now the real
  partition-reuse read:
  - summary:
    [results/transition_rich_support_local_linearity_20260401/summary.md](/home/mila/l/lia/skae/results/transition_rich_support_local_linearity_20260401/summary.md)
  - `gated_local_linear` is the strongest reusable-partitions positive,
    `gated_transfer_linear` is intermediate, and `multiwell_strong_transition`
    is weakest on retained-coverage
  - the matched zero-sparsity MLP is also strong, so this is not a sparse-only
    mechanism win
- Interpretation:
  - the branch now has a valid three-system deterministic suite before any
    model training
  - the suite now has two independent ground-truth benchmark-validity reads:
    one for per-step chart switching and one for periodic refresh cadence
  - the branch now also has the first collected model-side diagnostic read, and
    it supports a narrower claim than the most ambitious version of the story:
    strong periodic-refresh positives on `gated_local_linear` and
    `gated_transfer_linear`, but not a clean `multiwell_strong_transition`
    positive and not a chart-localization win on the transfer toy
- The supporting benchmark, hard-system, and mechanism evidence below remains the context for interpreting this new branch.

### 1. Supporting benchmark packet: one fair `200k` checkpoint family with `H100-H3000` coverage

- The table-facing benchmark family should now be read from [results/paper_zero_sparse_benchmark_200k_20260321/collect/forecasting_rows.csv](/home/mila/l/lia/skae/results/paper_zero_sparse_benchmark_200k_20260321/collect/forecasting_rows.csv), [results/paper_zero_sparse_benchmark_200k_20260321/collect/paper_benchmark_summary.json](/home/mila/l/lia/skae/results/paper_zero_sparse_benchmark_200k_20260321/collect/paper_benchmark_summary.json), and [results/paper_zero_sparse_benchmark_200k_20260321/fixed_cadence_periodic_100/fixed_cadence_summary.json](/home/mila/l/lia/skae/results/paper_zero_sparse_benchmark_200k_20260321/fixed_cadence_periodic_100/fixed_cadence_summary.json), not from the older mixed-coverage benchmark summaries.
- The verified best-periodic cross-system medians are:
  - sparse MLP: `2.947e-4 / 0.0051 / 0.0240 / 0.0795 / 0.1123 / 0.1591 / 0.2201`
  - zero-sparsity MLP: `3.021e-4 / 0.0063 / 0.0353 / 0.1067 / 0.1351 / 0.1738 / 0.1951`
  - promoted dense LISTA: `3.440e-4 / 0.0047 / 0.0250 / 0.0449 / 0.0627 / 0.0880 / 0.1039`
  - horizons are `H100/H500/H1000/H1500/H2000/H2500/H3000`
- The verified good-system counts are:
  - sparse MLP: `27/25/25/26/25/25/25`
  - zero-sparsity MLP: `27/26/26/27/27/27/27`
  - promoted dense LISTA: `28/25/26/25/25/25/25`
- Interpretation:
  - sparse MLP is still the clean anchor at `H100` and `H1000`
  - promoted dense LISTA is the strongest cross-system LISTA result and is median-best at `H500` and `H1500-H3000`
  - zero-sparsity does not produce the benchmark-median `H1000` win implied by the earlier mixed-coverage snapshot, but it does improve late-horizon coverage on more systems than it wins by median
  - the paper therefore needs a genuine three-way benchmark read rather than a sparse-versus-dense summary plus a side-note control
- Under one global deployment-like cadence `periodic_100`, the verified late-horizon medians are:
  - sparse MLP: `0.1070 / 0.2140 / 0.3690 / 0.6148`
  - zero-sparsity MLP: `0.1729 / 0.4542 / 0.7041 / 0.9349`
  - promoted dense LISTA: `0.0744 / 0.2688 / 0.4437 / 0.6508`
  - horizons are `H1500/H2000/H2500/H3000`
- Fixed-cadence interpretation:
  - explicit sparsity still helps materially relative to the matched zero-sparsity MLP under one deployment-like cadence
  - promoted dense LISTA remains the late-horizon fixed-cadence median winner in the benchmark packet, so the deployment-style story is again three-way rather than a sparse-only positive
- The dense-optimizer block-diagonal transfer still does not rescue block-diagonal LISTA globally; keep block-diagonal claims restricted to targeted hard-system evidence and explicit fairness-control tables.

### 2. Appendix-only dense LISTA comparator-selection provenance

This is supporting provenance for the dense comparator used in Section 1. It should be mentioned only briefly in the main text and detailed in the appendix if needed.

- The dense-LISTA easy-system parity Stage 1 is complete under [dense-LISTA easy-system Stage 1 summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/collect/paper_benchmark_summary.md):
  - same dense LISTA architecture and benchmark-selected `dt` on all `8` target systems
  - Stage 1 changed only `num_steps`, `lr`, and `k_matrix_lr`
- Best Stage-1 recipes against the fixed `generic_sparse` anchor:
  - `lista_dense_ns100k_lr5em5_klr5em6_wd1em4` wins `6/8` target systems with median dense/generic ratio `0.8699`
  - `lista_dense_ns200k_lr5em5_klr5em6_wd1em4` wins `5/8` target systems with median ratio `0.7888`
- Best per-system dense recipe still loses on:
  - `competitive_lv` (`1.764x` vs `generic_sparse`)
  - `duffing` (`1.041x` vs `generic_sparse`)
- Positive details:
  - all `9` Stage-1 dense-LISTA recipes keep `8/8` target systems under the good-forecast band
  - no Stage-1 dense recipe is catastrophic on the target set
- Primary audit files:
  - [Stage-1 paper summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/collect/paper_benchmark_summary.md)
  - [Stage-1 forecasting summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/collect/forecasting_summary.md)
  - [best win-count comparison](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/compare/lista_dense_ns100k_lr5em5_klr5em6_wd1em4_vs_generic_sparse/forecasting_comparison.md)
  - [best median-ratio comparison](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/compare/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_vs_generic_sparse/forecasting_comparison.md)
- Interpretation:
  - The dense-LISTA gap on the easier accepted-default systems is now clearly partly optimization-limited, not purely architectural.
  - This is strong support for a fairness-preserving dense-LISTA recovery story.
  - It is still not enough to claim dense LISTA is better than `generic_sparse` on most systems overall, because the result is limited to the targeted `8`-system subset and the holdouts remain real.
- The coefficient-only Stage 2 holdout sweep is now complete under [Stage-2 forecasting summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage2_20260308/collect/forecasting_summary.md):
  - `duffing` is flipped only by the specialized `100k, sc=0.012` recipe (`0.0182` vs `0.0309`, `0.590x`)
  - `competitive_lv` is not flipped by any coefficient-only recipe
  - the best global-compromise holdout recipe is `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3`
- The exact `8`-system validation Stage 3 is now complete under [Stage-3 paper summary](/home/mila/l/lia/skae/results/dense_lista_recipe_validation_stage3_20260309/collect/paper_benchmark_summary.md):
  - `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3` wins `6/8` shared systems vs `generic_sparse`
  - shared-system median `H1000` best-periodic ratio is `0.6928`
  - `8/8` systems stay under the good-forecast band with `0` catastrophic systems
  - all seeds are good on all `8` systems
  - the cheaper `100k, sc=0.003` recipe reaches `3/8` wins
  - the Duffing-fixing `100k, sc=0.012` recipe falls to `2/8` wins overall
- Interpretation:
  - the dense-LISTA fairness question is now resolved: promote `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3` as the single fair dense-LISTA recipe
  - the paper story is no longer “LISTA almost catches up if tuned enough”; it is “a fixed dense LISTA architecture recovers most easy-system near-misses with one fair external recipe, but still leaves a persistent `competitive_lv`-style holdout and does not overturn the global `generic_sparse` ranking”
- The promoted dense-LISTA full `29`-system rerun is now complete under [Stage-4 paper summary](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/collect/paper_benchmark_summary.md):
  - same dense LISTA architecture, promoted Stage-3 recipe, and benchmark-selected pass-2 `dt` table
  - compared against the fixed `generic_sparse` `v4` anchor, dense LISTA wins `21/29` shared systems with median dense/generic ratio `0.6455`
  - cross-system median `H1000` best-periodic improves from `0.0328` to `0.0232`
  - good-system count improves from `25/29` to `26/29`
  - there are `0` systems where the promoted dense recipe fails the good-forecast band while `generic_sparse` passes
  - the remaining dense failures are still concentrated on the hard systems, especially `kuramoto` (`48.50`), `hopfield` (`1.578e+06`), and `multiwell_strong_transition_hd` (`4.533e+04`)
- Primary audit files:
  - [Stage-4 paper summary](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/collect/paper_benchmark_summary.md)
  - [Stage-4 forecasting summary](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/collect/forecasting_summary.md)
  - [Stage-4 dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/compare/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3_vs_generic_sparse/forecasting_comparison.md)
- Interpretation:
  - the dense-LISTA parity story is now stronger than the Stage-3 subset result alone: one fixed dense recipe beats the fixed `generic_sparse` anchor on most benchmark systems overall
  - for paper writing, the fair `200k` comparison in Section 1 is the dense-vs-MLP headline; the older Stage-4-vs-`v4` comparison is supporting evidence for recipe promotion, not the main-text benchmark claim
  - when writing that headline, say explicitly that the dense-vs-MLP comparison is width/depth-aligned but not parameter-matched
  - the remaining dense-LISTA paper risk is no longer “can it catch up on the easier systems?”; it is whether the paper cleanly separates the cross-system parity win from the unresolved hard-system failures

### 3. `dt` rescue and hard-system follow-up are now decision-grade: step size is a real bottleneck, Kuramoto has a targeted `200k` rescue, and Hopfield remains MLP-better

- A matched zero-sparsity MLP extension of every paper-facing hard-system setting cited in this section is now collected under [results/zero_sparse_hard_systems_20260321](/home/mila/l/lia/skae/results/zero_sparse_hard_systems_20260321). It covers Kuramoto (`N=16` identical, `N=16` uniform-spread, and the dimension sweep), Hopfield (`N=16` and quarter-step `N=64`), corrected 4-basin competitive LV, and fixed 8-basin competitive LV at both cited step sizes. The recovery chain cleared on March 22 when requeued task `9007966_4` completed and refreshed collectors/comparisons `9016645-9016660` all finished `0:0`.

- The focused higher-dimensional redo is collected and corrected under
  [results/hard_system_sparse_kae_redo_p1024_seq8_100k_halflr_sc6em3_tanh_dense_20260429](/home/mila/l/lia/skae/results/hard_system_sparse_kae_redo_p1024_seq8_100k_halflr_sc6em3_tanh_dense_20260429)
  and replaces the old ReLU zero-sparsity control with a tanh dense MLP
  baseline. The chain `9412218 -> 9412219 -> 9412220/9412221/9412222`
  completed with exit `0:0`; the collector was then fixed and refreshed on
  allocation `9432839` after the first artifact selected latest runs per
  `n_16/seed` wrapper and dropped most Hopfield rows. The corrected artifact
  has `270` rows for Dense MLP, Sparse MLP, Sparse MLP-BD, LISTA, LISTA-BD,
  and LISTA-SB on CLV, Hopfield, and Kuramoto. Dense MLP wins every
  candidate-vs-Dense system-median comparison at `H100`, `H500`, and `H1000`.
  At `H1000`, CLV/Hopfield/Kuramoto medians are Dense MLP
  `0.2999/2.8034/7.7872`, LISTA-SB `0.3210/8.3277/15.5167`, Sparse MLP
  `0.4383/18.8994/56.4700`, Sparse MLP-BD `0.4336/18.8998/56.4689`, LISTA
  `0.8004/1.704e+06/3569.6968`, and LISTA-BD
  `3.0786/20.8668/1326.8682`.

- The repaired `dt` resolution completed through pass `2`:
  - `15/29` systems accept default `dt`
  - `4/29` systems accept after at least one halving
  - `10/29` systems remain `integration_hard`
- The most important remaining `integration_hard` systems are:
  - `kuramoto` (`selected dt = 0.0125`)
  - `hopfield` (`selected dt = 0.0125`)
  - `lotka_volterra`
  - `multiwell_strong_transition`
  - `multiwell_gradient_hd`
  - `multiwell_rotational_hd`
  - `multiwell_strong_transition_hd`
  - `dysts:DequanLi`
  - `dysts:WangSun`
  - `dysts:LorenzCoupled`
- Current high-dimensional bottlenecks at the selected smaller `dt` are still bad:
  - `kuramoto`, `generic_sparse`, system-median `H1000` best-periodic: `65.7014`
  - `hopfield`, `generic_sparse`, system-median `H1000` best-periodic: `199.4978`
  - `kuramoto`, `lista_blockdiag`, system-median `H1000` best-periodic: `14.2618`
  - `hopfield`, dense LISTA, system-median `H1000` best-periodic: `7.241e+09`
- Primary audit file:
  - [v4 pass-2 `dt` resolution summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/dt_resolution/pass2/dt_resolution.md)
- Interpretation:
  - the benchmark is no longer blocked by queue completion
  - step size is a real scientific bottleneck, not just a scheduling artifact
  - the open paper problem shifted from "finish the rerun" to "what to do with systems that stay hard even after the allowed `dt` rescue"

- The repaired focused intrinsic-HD rerun is complete under [intrinsic-HD `dt` rescue rerun summary](/home/mila/l/lia/skae/results/intrinsic_hd_dt_rescue_20260308_rerun1/forecasting_summary.md):
  - all `48` rows are collected
  - official selection is still based on `evaluation_results_best.json`
- Best current intrinsic-HD arms at `H1000` best-periodic:
  - `kuramoto`:
    - `lista_blockdiag`, `dt=0.0125`, `sp=0.0005`: `14.36`
    - matched `generic_sparse`, `dt=0.0125`, `sp=0.0005`: `25.93`
  - `hopfield`:
    - `generic_sparse`, `dt=0.0125`, `sp=0.0005`: `71.02`
    - best `lista_blockdiag`, `dt=0.0125`, `sp=0.0010`: `80.54`
- Smaller `dt=0.0125` beats `dt=0.025` for both systems in both model families, so the step-size hypothesis is now directly supported by a full focused rerun.
- A diagnostic recollection from `evaluation_results_last.json` shows checkpoint-selection mismatch on Kuramoto:
  - `lista_blockdiag`, `dt=0.0125`, system-median `H1000` improves from `23.40` to `14.64` across the focused pilot grid
  - on the winning `lista_blockdiag`, `dt=0.0125`, `sp=0.0005` arm, the last-checkpoint median is `13.91`
  - this is a diagnostic, not yet the official paper metric, but it shows late training still matters on Kuramoto
- Completed focused Kuramoto `dt=0.00625`, `200k` follow-up under the official short-horizon compare [Kuramoto comparison](/home/mila/l/lia/skae/results/kuramoto_dt00625_200k_compare_20260308/compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md) plus the later long-horizon collector [Kuramoto forecasting summary](/home/mila/l/lia/skae/results/kuramoto_dt00625_200k_compare_20260308/collect/forecasting_summary.md):
  - `generic_sparse`: seed-median `H1000` best-periodic `27.02`
  - dense LISTA: seed-median `H1000` best-periodic `13.84`
  - `lista_blockdiag`: seed-median `H1000` best-periodic `6.98`
  - all five `lista_blockdiag` seeds are good and tightly clustered in `6.89-7.13`
  - the underlying focused summary is again not `H1000`-only: at `H100`, `generic_sparse=0.0343`, `lista_blockdiag=0.1536`, `lista_dense=0.2194`; at `H500`, `generic_sparse=1.4972`, `lista_blockdiag=2.6834`, `lista_dense=4.1222`; the `lista_blockdiag` win appears only at `H1000`
  - the same run family now also has explicit `H1500/H2000/H2500/H3000` medians from the collector: `generic_sparse = 547.37 / 1.208e+04 / 3.370e+05 / 9.207e+06`, `lista_blockdiag = 10.93 / 14.52 / 17.94 / 21.58`, dense LISTA `= 54.85 / 205.26 / 541.19 / 1519.09`
  - **Fixed-cadence ablation (completed):** `periodic_100` exactly reproduces the official `best_periodic` `H1000` ranking for all three roots (`6.98`, `13.84`, `27.02`), so the long-horizon Kuramoto block-diagonal win is already a fixed-cadence result, not a `best_periodic` oracle artifact. At `H500`, `generic_sparse` still has lower error than `lista_blockdiag` under the fixed cadence.
  - **Checkpoint-selection ablation (completed):** switching from `evaluation_results_best.json` to `evaluation_results_last.json` on the current `dt=0.00625`, `200k` comparison does not change the model ranking or good-band membership (`lista_blockdiag`: `6.98 -> 7.00`, `lista_dense`: `13.84 -> 17.63`, `generic_sparse`: `27.02 -> 29.43`). The older `dt=0.0125` pilot mismatch (`23.40 -> 14.64`) is real but tied to the earlier, superseded setting. Keep `evaluation_results_best.json` as the official paper rule.
- Completed Hopfield `dt=0.00625`, `200k` follow-up under [Hopfield follow-up summary](/home/mila/l/lia/skae/results/hopfield_dt00625_200k_compare_20260309/forecasting_summary.md):
  - `generic_sparse`: seed-median `H1000` best-periodic `3.36`
  - `lista_blockdiag`: seed-median `H1000` best-periodic `8.82`
  - the same file shows the MLP lead at every collected horizon: `H100` `0.0500` vs `0.1075`, `H500` `0.8835` vs `3.5303`, `H1000` `3.3642` vs `8.8212`
  - the long-horizon continuation stays negative as well: `generic_sparse` is `6.61 / 9.17 / 10.96 / 12.23` at `H1500/H2000/H2500/H3000`, versus `12.23 / 12.83 / 13.23 / 13.58` for `lista_blockdiag`
  - both are inside the good-forecast band on the system median, but every-step errors are still enormous for both
- The completed `H3000` reevaluation sharpens the hard-system long-horizon read across the whole Kuramoto/Hopfield family:
  - Kuramoto `N=16`, `dt=0.00625`, `200k`: `lista_blockdiag` remains much better than `generic_sparse` across `H1500/H2000/H2500/H3000` (`10.93 / 14.52 / 17.94 / 21.58` vs `547.37 / 1.208e+04 / 3.370e+05 / 9.207e+06`), but it is out of band beyond `H1000`
  - Kuramoto `N=32`, `dt=0.00625`, `200k`: `lista_blockdiag` stays better across `H1500/H2000/H2500/H3000` (`10.89 / 16.14 / 21.69 / 27.95` vs `13.60 / 24.27 / 42.19 / 75.25` for `generic_sparse`), but it is also out of band at those later horizons
  - Kuramoto uniform-spread `N=16`: the repaired `generic_sparse_blockdiag` mirror is now slightly better than `lista_blockdiag` at every reported horizon (`8.13 / 24.76 / 91.23 / 399.74 / 1724.45` versus `9.53 / 28.41 / 117.46 / 523.72 / 2129.37`), but both repaired structured roots are still catastrophic by `H3000`
  - Hopfield `N=16`, `dt=0.00625`, `200k`: `generic_sparse` stays better at `H1500/H2000/H2500/H3000` (`6.61 / 9.17 / 10.96 / 12.23`) than `lista_blockdiag` (`12.23 / 12.83 / 13.23 / 13.58`); only `generic_sparse` stays in-band through `H2000`
  - higher-basin Hopfield `N=64`, `dt=0.0015625`, `200k`: quarter-`dt` improves errors but remains strongly negative for LISTA, with `generic_sparse = 309.92 / 520.71 / 711.04 / 873.50` at `H1500/H2000/H2500/H3000` versus dense `591.50 / 842.59 / 1046.18 / 1206.93` and targeted blockdiag `735.26 / 1035.51 / 1274.93 / 1461.51`
- Completed Kuramoto `N=32`, `dt=0.00625`, `200k`, `3`-seed confirmation under the short-horizon compare [Kuramoto `N=32` summary](/home/mila/l/lia/skae/results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md) and long-horizon collector [Kuramoto `N=32` forecasting summary](/home/mila/l/lia/skae/results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/collect/forecasting_summary.md):
  - `generic_sparse`: seed-median `H1000` best-periodic `6.65` (all seeds good)
  - `lista_blockdiag`: seed-median `H1000` best-periodic `6.00` (all seeds good, std `0.33`)
- Completed Kuramoto dimension sweep under [Kuramoto dimension summary](/home/mila/l/lia/skae/results/kuramoto_dimension_sweep_dt00625_200k_20260309/collect/kuramoto_dimension_summary.md):
  - dimensions: `N={8,16,24,32,64}`
  - models: `generic_sparse`, promoted dense LISTA, `lista_blockdiag`
  - fixed setting: `dt=0.00625`, `200k`, `5` seeds
  - `H1000` seed-median best-periodic by dimension:
    - `generic_sparse`: `813.57`, `30.18`, `6.71`, `6.68`, `208.93`
    - promoted dense LISTA: `495.07`, `13.44`, `14.99`, `92.28`, `208.71`
    - `lista_blockdiag`: `8.11`, `7.07`, `6.57`, `5.92`, `23.27`
  - seed robustness:
    - `lista_blockdiag` is all-seeds-good at `N=16/24/32`
    - `lista_blockdiag` is median-good but not fully robust at `N=8` (`4/5` good seeds, worst seed `10.89`)
    - `lista_blockdiag` falls out of band at `N=64` (`2/5` good seeds, worst seed `209.20`)
- Completed repaired block-diagonal MLP mirrors for the missing Kuramoto families under the retry1 roots:
  - uniform-spread `N=16`: `generic_sparse_blockdiag` reaches `H1000 = 8.13` with all `5/5` seeds good, beating both `generic_sparse` (`44.46`) and `lista_blockdiag` (`9.53`)
  - dimension sweep `H1000`: `generic_sparse_blockdiag = 10.61 / 6.51 / 5.79 / 5.16 / 208.54` at `N=8/16/24/32/64`
  - on that sweep, the repaired MLP block-`K` mirror beats `generic_sparse` at every `N`, beats `lista_blockdiag` at `N=16/24/32`, but not at `N=8` or `N=64`
  - the suspicious `N=64` row is now audited and decision-grade: it is a real seed-fragility limit (`2/5` good seeds, `3/5` collapsed seeds), not a leftover March 17 invalid-control artifact
- Interpretation:
  - smaller `dt` is the dominant hard-system lever in the current regime
  - `lista_blockdiag` is the strongest hard-system LISTA result on Kuramoto and the only model that cleanly wins the `N=16` three-way comparison there at `H1000`; the shorter-horizon audit shows this is a specifically long-horizon advantage rather than across-horizon dominance
  - the completed repaired fairness mirrors narrow the causal read: block structure alone helps on the targeted `N=16` family, on uniform-spread `N=16`, and across the sweep at `N=16/24/32`, but it does not rescue `N=8` and the `N=64` failure is real but seed-sensitive rather than a provenance glitch
  - promoted dense LISTA does not transfer as a robust Kuramoto solution under this sweep
  - Hopfield is no longer a catastrophic boundary case in the targeted `N=16`, `dt=0.00625`, `200k` setting, but it is still not a structured-LISTA success story because `generic_sparse` remains better through `H2000` and both models are out of band by `H2500-H3000`
  - the remaining scientific limitation is autonomous rollout stability, not whether periodic reencoding can rescue the hard systems at all
  - the remaining Kuramoto paper question is no longer missing fairness controls; it is how to present a targeted moderate-dimension success with an explicit `N=8` and `N=64` limit
- Completed Kuramoto robustness evaluation (uniform frequency spread, `N=16`, `dt=0.00625`, `200k`, `5` seeds) under the short-horizon compare [Kuramoto robustness comparison](/home/mila/l/lia/skae/results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/compare/lista_blockdiag_uniform_spread_vs_generic_sparse_uniform_spread/forecasting_comparison.md) and long-horizon collector [Kuramoto robustness forecasting summary](/home/mila/l/lia/skae/results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/collect/forecasting_summary.md):
  - `lista_blockdiag`: seed-median `H1000` best-periodic `9.53` (4/5 seeds good, std `0.64`)
  - `generic_sparse`: seed-median `H1000` best-periodic `44.46` (0/5 seeds good)
  - `4.7x` improvement; the Kuramoto block-diagonal positive is not a single-regime artifact
  - every-step errors remain catastrophic for both models under heterogeneity; periodic re-encoding is essential

### 4. Appendix-only `50k` `v4` audit: useful for matched four-model symmetry, not for main-text claims

- Completed `v4` full-matrix result (`29` systems, `4` baselines, `3` seeds) under the repaired `dt`-rescue chain:
  - `generic_sparse` is best by cross-system median `H1000` best-periodic (`0.0328`)
  - dense LISTA is second (`0.0388`)
  - block-diagonal LISTA is third (`0.1508`)
  - diagonal-K LISTA is worst (`1.2110`)
- `H1000` good-system counts (`best-periodic < 10`):
  - `generic_sparse`: `25/29`
  - dense LISTA: `24/29`
  - block-diagonal LISTA: `24/29`
  - diagonal-K LISTA: `24/29`
- Shared-system comparison against `generic_sparse`:
  - dense LISTA wins `15/29`
  - block-diagonal LISTA wins `3/29`
  - diagonal-K LISTA wins `3/29`
- Interpretation note:
  - `lista_diagonal` is now retired from active paper scope. Keep the completed diagonal numbers only as historical context; do not allocate new experiment budget to diagonal reruns.
- Primary audit files:
  - [v4 paper summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/paper_benchmark_summary.md)
  - [v4 final forecasting summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/forecasting_summary.md)
  - [dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_dense_vs_generic_sparse/forecasting_comparison.md)
  - [block-diagonal vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md)
- Interpretation:
  - `v4` is still a useful matched four-model snapshot and the cleanest source for the `dt`-resolution audit.
  - It should be treated as appendix-only historical context once the `200k` reruns materially improve the same story.
  - Do not use `v4` as the source of main-text rankings or headline model comparisons when a `200k` counterpart exists.

## Recent Queue Activity

As of `2026-05-26`, the matched route-baseline packet for the retained
\(C_{\rm stab}\)-routed learned-intercept local-\(K_c\) controlled result is
complete and paper-facing. The original May 19 failure chain is retained only
as diagnostic history: those arrays timed out or failed before aggregate
collection, and later 3-hour checkpoint-resume attempts exposed that rows with
`checkpoint.pt` but no `last.pt` were not being resumed. The trainer now
resumes from either checkpoint name, the route queue wrappers pass
`SAVE_LAST_CHECKPOINT`, and the final patched arrays completed for oracle-basin
`9645554`, support-family `9645583`, random-matched `9645587`, and
latent-kmeans `9645669`, with `225/225` completed tasks each. The manually
resubmitted collect/compare/wide jobs all exited `0:0`, including support
recollection after fixing the collector to choose the latest evaluable run
rather than an empty resume-start directory. Final H1000 row wins and median
staged/global ratios versus the global-\(K\) LISTA anchor are:
\(C_{\rm stab}\) `188/225` / `0.399`, support-family `189/225` / `0.328`,
random-matched `185/225` / `0.451`, latent-kmeans `196/225` / `0.341`, and
oracle-basin `195/225` / `0.366`. The conclusion is that local affine maps are
robustly useful, but \(C_{\rm stab}\) is not uniquely best as a forecasting
router.

The March 9-20 paper-strengthening program is fully closed and now belongs to
the archived planning set. As of `2026-05-16`, the first normalized-decoder
controlled requeue has produced a partial seed-3 initial read: the initial
`9561145/9561146` launcher attempt failed before emitting arrays, and the
corrected launchers `9561573` and `9561574` completed. Rollout training array
`9561575` completed `226/270` tasks and timed out `44`; encoded training
array `9561603` completed `255/270` and timed out `15`; downstream
forecasting and support collectors completed for the finished runs. No
related SLURM jobs remain active. The previous normalized-decoder table
replacement queue is otherwise closed: test job `9553712` passed, Dysts
launchers `9553722/9553723` completed and were summarized under
[normdec_dysts_summary_20260515](/home/mila/l/lia/skae/results/normdec_dysts_summary_20260515),
and controlled launchers `9553720/9553721` failed before array submission
because of the source-table seed-coverage issue now patched in the launcher.
As of `2026-05-12`, the other live paper-side queues added
here are: the strict staged source-target affine \(F_{\rm abs}\) local-\(K_c\)
LISTA Table 1 pilot, where reduced launcher `9527220` submitted array
`9527234`, collect `9527235`, and compare `9527236` under
[results/staged_fabs_local_affine_k_lista_table1_seed2_20260512](/home/mila/l/lia/skae/results/staged_fabs_local_affine_k_lista_table1_seed2_20260512)
after canceling the fixed-center predecessor `9523664` and the full 15-seed
source-target affine predecessor `9527170`, and dependent wide-periodic
re-evaluation job `9531292` completed the final mixed decision read under
[wide_periodic_reeval_full](/home/mila/l/lia/skae/results/staged_fabs_local_affine_k_lista_table1_seed2_20260512/wide_periodic_reeval_full);
matched interactive periodic-grid
probes under
[runs/staged_local_affine_periodic_target_20260512](/home/mila/l/lia/skae/runs/staged_local_affine_periodic_target_20260512)
show a clean `claude:transition_routes_4` seed-`0` best-periodic win with `3`
support families and a mixed seed-`1` read;
and the additional baseline tooling/smoke results, where validation `9526657`
and smoke `9526658` passed, corrected standalone DMD/EDMD/mixture launcher
`9526668` submitted and completed an off-target nine-system array `9526673` under
[results/paper_baseline_suite_20260512_corrected](/home/mila/l/lia/skae/results/paper_baseline_suite_20260512_corrected),
which is implementation validation rather than retained-15 manuscript evidence.
Retained-15 standalone launcher `9527268` completed and submitted baseline
array `9527269` (`90` tasks, `0-89%32`), and the array completed all `90/90`
tasks with exit `0:0`. RBF-dictionary EDMD is the retained-15 standalone
winner at all three paper horizons (`0.660/0.808/0.751` raw cumulative MSE).
It beats Dense MLP, but not the sparse KAE rows in the manuscript table.
Dysts standalone launcher `9530090` submitted array `9530093`; after repairing
the single failed Sakarya mixture task as `9530233_48`, the Dysts packet has
`540` ok rows and no error rows. Summary job `9530325` wrote the baseline
summary CSVs, and those rows are now folded into the manuscript forecasting
table.
Checkpoint-backed
clustering/oracle local-K launcher `9523987` produced the merged packet under
[results/paper_regime_oracle_baselines_20260512](/home/mila/l/lia/skae/results/paper_regime_oracle_baselines_20260512).
As of `2026-06-23`, the label-free local EDMD extension is also closed:
launcher jobs `9914507` and `9914508` submitted CPU-only arrays `9914514` and
`9914513`, which completed all `45/45` retained-multibasin and `30/30` Dysts
tasks with exit `0:0`. Allocation `9914609` rebuilt
[table_standalone_state_space_baselines.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table_standalone_state_space_baselines.tex)
and the summary metadata now records the appended local EDMD result roots.
Earlier live queues are documented in the summary above.

### Just Closed

- **Kuramoto uniform-spread dense-LISTA completion run:** the March 20 appendix-table audit found that the original March 9 uniform-spread robustness manifest never launched dense LISTA, so the missing row was genuine rather than a collector omission. Completion chain `8989946 -> 8989947 -> 8989948/8989949/8989950` finished cleanly under [results/kuramoto_uniform_spread_dense_20260320](/home/mila/l/lia/skae/results/kuramoto_uniform_spread_dense_20260320). Training tasks `8989946_[0-4]` ended between `2026-03-20 02:58 EDT` and `2026-03-20 03:31 EDT`; collector `8989947` finished at `03:32 EDT`, and compare jobs `8989948/8989949/8989950` finished at `03:33 EDT`. The completed row is `0.2081 / 4.1470 / 16.55 / 46.38 / 132.35 / 415.24 / 1530.83` at `H100/H500/H1000/H1500/H2000/H2500/H3000`.
- **Kuramoto uniform-spread valid MLP `+ block-K` rerun:** retry chain `8988801 -> 8988802 -> 8988803/8988804` closed successfully under [results/kuramoto_uniform_spread_gs_blockdiag_retry1_20260319](/home/mila/l/lia/skae/results/kuramoto_uniform_spread_gs_blockdiag_retry1_20260319). Training tasks ended between `2026-03-19 22:33 EDT` and `2026-03-19 23:47 EDT`; the collector finished at `23:48 EDT`, and both compare jobs finished at `23:49 EDT`.
- **Kuramoto dimension-sweep valid MLP `+ block-K` rerun:** retry chain `8988805 -> 8988806 -> 8988807/8988808` also closed successfully under [results/kuramoto_dimension_sweep_gs_blockdiag_retry1_20260319](/home/mila/l/lia/skae/results/kuramoto_dimension_sweep_gs_blockdiag_retry1_20260319). Training tasks ended between `2026-03-19 22:06 EDT` and `2026-03-19 23:54 EDT`; the collector finished at `23:54 EDT`, and both compare jobs finished at `23:55 EDT`.
- **Queue hygiene note:** the first submission attempt (`8988754`, `8988758`) was invalidated by external startup termination after accidental duplicate-submission cleanup. Use only the retry1 roots above for paper-facing interpretation.

### Last 48 hours (completed state)

- **Paper-facing horizon reevaluation:** closed successfully after one tooling repair. Original kickoff array `8929393` failed `300/300` tasks because `run_manifest.tsv` used CRLF line endings. The repaired chain `8933469 -> 8933470 -> 8935507 -> 8936342 -> 8937045 -> 8937956 -> 8938574` completed all `1528` reevaluations and refreshed the benchmark / compare artifacts through `H3000`.
- **Hard-system parity sweep:** closed successfully after a small anchor repair. Stage-2 confirm array `8931671` ended with two failed anchor rows (`87`, `137`); the repair / recollect / resummary chain `8933794 -> 8933795 -> 8933796` completed and produced the final negative-for-LISTA parity summary now used in this document.
- **Corrected `competitive_lv` representation follow-up:** closed after one repair. Wrapper `8933876` completed, support-alignment job `8939086` completed, the first label-free clustering array `8939087` failed `39/39` because checkpoint paths in `lfc_task_specs.tsv` retained a trailing carriage return, and March 13 repair chain `8945129 -> 8945133 -> 8945135` then completed cleanly. The final scientific read is still negative on the clean `4`-basin paper roots.
- **Hopfield quarter-`dt` higher-basin rerun:** fully complete. Base sweep `8926091` and matched fairness control `8926089` both finished; the smaller `dt` improves errors materially but does not move any root into the good band or change the ordering (`generic_sparse` remains best).
- **Other completed paper-side jobs:** the historical matched block-diagonal fairness-control jobs (`8922796`, `8922810`, `8922811`) all finished at the queue level, but their MLP `+ block-K` outputs were later invalidated by the March 17 audit; the CLV high-basin scalability / smaller-`dt` sweeps (`8923108`, `8926090`) and the clean 4-basin CLV retrain / recovery / extension chain (`8922033`, `8922374`, `8922879`) are complete and already reflected in the current paper claims above.
- **Accounting note:** the only other failed job in the last-48-hour accounting window was one unrelated `interactive` shell session (`8936456`); it does not affect paper state.

### Recently completed

- **Paper-facing `H3000` horizon reevaluation:** complete. The refreshed fair benchmark is now split by horizon: `generic_sparse_ns200k_best` is best by cross-system median at `H100/H500/H1000` (`3.146e-4`, `0.0050`, `0.0233`), while promoted dense Stage 4 becomes median-best at `H2000/H2500/H3000` (`0.0627`, `0.0910`, `0.0940`) but falls to `24/29` good systems at `H3000` versus `26/29` for the MLP anchor.
- **Long-horizon fixed-cadence benchmark rescore:** complete offline from the refreshed fair `200k` benchmark JSONs; no queue submission was needed. Under one global `periodic_100`, dense remains breadth-competitive at `H1500-H3000` but no longer has a clean late-horizon median edge: the MLP is better by cross-system median at `H2000/H2500`, and at `H3000` the MLP also keeps better coverage (`20/29` vs `17/29`).
- **Hard-system parity sweep:** complete. `generic_sparse` is best on all `14` confirmed settings; dense LISTA records `0` wins / `10` losses / `4` worse-seed-robustness rows, and `lista_blockdiag` records `0` wins / `8` losses / `6` worse-seed-robustness rows.
- **Corrected `competitive_lv` representation follow-up:** complete. The repaired March 13 clustering rerun wrote the missing `label_free_clustering_v2/` outputs, but the corrected CLV result remains negative for the intended support-based narrative: no clean `4`-basin paper root clears the support-view gate, all keep negative cosine separation, and support-view clustering collapses to `5/2/1` discovered groups across seeds.
- **Competitive LV high-basin smaller-`dt` halving rerun:** complete on array `8926090` (`96/96`). Halving to `dt=0.0025` removes the remaining fixed-system `200k` `lista_blockdiag` seed failures on the higher-basin CLV probes, but `generic_sparse` remains best and `generic_sparse_blockdiag` stays neutral.
- **Kuramoto unique mode-support audit (completed March 10):** All `30/30` array tasks and collector finished under `results/kuramoto_mode_support_audit_20260310/`. The strong negative claim is confirmed: Kuramoto winding-number basins do not have meaningful basin-specific support patterns. Mode-support uniqueness is trivially degenerate — every trajectory has its own unique support (`traj_unique=1.0`), mode supports are singletons, basin consistency is negligible (`0.0625` balanced / `0.309` random), and Hamming geometry is flat (`ratio≈1.0`). This holds identically across all 3 model families (`generic_sparse`, `lista_dense`, `lista_blockdiag`), all 5 seeds, both sampling protocols (`random`, `balanced`), all support modes (`mean`, `majority`, `modal`), and all threshold values (`1e-4` to `1e-1`). This closes the gap left by label-free clustering v2 and directly confirms both claims: basins are not recoverable from latent features, and basins do not have literal reusable mode supports.
- **Broad support-alignment audit on labelable `v4` systems:** complete under [support-alignment summary](/home/mila/l/lia/skae/results/paper_benchmark_support_alignment_20260311_v4_labelable/summary.md). Across `11` valid labelable systems (`132` checkpoints), binary `mode_uniqueness_rate` saturated at `1.0` on all `44/44` system-root medians, while cosine separation still cleanly split the systems: multiwell positive, Duffing negative, Kuramoto negative, and Hopfield mixed.
- **Fair `200k` follow-up benchmark:** complete. `generic_sparse_ns200k_best` is the best full-benchmark root by cross-system median `H1000` best-periodic (`0.0233`), while promoted dense Stage 4 still wins `18/29` shared systems and keeps `26/29` good systems at that horizon.
- **Dense LISTA promoted Stage 4 rerun:** complete. One fixed fair dense recipe wins `21/29` systems against the fixed `generic_sparse` anchor and improves the dense median to `0.0232`.
- **Kuramoto dimension sweep:** complete. `lista_blockdiag` is robust through `N=32`, not fully robust at `N=8`, and no longer in-band at `N=64`.
- **Focused Kuramoto and Hopfield smaller-`dt` follow-ups:** complete. `lista_blockdiag` wins the Kuramoto `dt=0.00625`, `200k` comparison at `H1000`, but the refreshed `H1500-H3000` read reduces that win to a relative-only advantage; Hopfield remains `generic_sparse`-best through `H2000` and both models fail by `H2500-H3000`.
- **Label-free clustering v2:** complete on array `8919951` with collector `8919952`. Multiwell systems are strongly positive, Duffing is weakly positive, and Kuramoto is negative.

### 6. Broad support-alignment audit: binary mode uniqueness saturates, cosine separation carries the real signal

Together with the corrected `competitive_lv` representation rerun and the March 13 local-linearity study, this section belongs to the basin-support/mechanism family rather than a standalone side analysis.

- A matched zero-sparsity MLP extension of this mechanism family is now complete under [results/zero_sparse_mechanisms_20260321](/home/mila/l/lia/skae/results/zero_sparse_mechanisms_20260321). The original wrapper `9007983` was canceled during March 22 recovery, but the refreshed wrapper `9016661` finished the same support-alignment, label-free clustering, Kuramoto support-audit, and recurring-support local-linearity analyses on the no-sparsity MLP checkpoints. The completed Kuramoto support-audit rerun shows that removing the sparsity penalty does not recover reusable mode supports.

- **Result (COMPLETE, March 11, local audit):** Evaluated the canonical `v4` checkpoints on all currently valid labelable benchmark systems under [support-alignment summary](/home/mila/l/lia/skae/results/paper_benchmark_support_alignment_20260311_v4_labelable/summary.md).
  - scope: `11` systems (`duffing`, `8` `multiwell*` variants, `kuramoto`, `hopfield`) x `4` roots x `3` seeds = `132` checkpoints
  - excluded: `competitive_lv`, because the canonical `v4` checkpoints used the invalidated 1-basin configuration
  - settings: `100` trajectories, length `500`, `5000`-step basin rollout, `support_threshold=1e-3`, `support_mode=mean`
- **Concrete result:**
  - binary mode-support uniqueness saturates completely: **all `44/44` system-root medians have `mode_uniqueness_rate=1.0`**
  - support reuse is weak almost everywhere: **`40/44` system-root medians have `mean_basin_consistency < 0.2`**
  - trajectory-level supports are often unique: **`24/44` system-root medians have `trajectory_unique_support_rate = 1.0`**
  - all multiwell system-root medians are positive by cosine separation (`0.250` to `0.706`)
  - Duffing is negative across all roots (`-0.129` to `-0.084`) despite perfect mode uniqueness
  - Kuramoto is negative across all roots (`-0.307` to `-0.264`) despite perfect mode uniqueness; the random `100`-trajectory audit again produces singleton `q=±2` basins, so the apparent `mean_basin_consistency≈0.424` is inflated and not evidence of reusable basin supports
  - Hopfield is mixed: cosine separation is positive across all roots (`0.459` to `0.607`), but `mean_basin_consistency` is only `0.043` for every root and `trajectory_unique_support_rate=1.0` throughout
- **Interpretation:**
  - the literal binary question "does each basin have a unique mode support?" is too weak as a broad benchmark diagnostic, because it returns a perfect score even on known negatives like Duffing and Kuramoto
  - cosine separation reproduces the known qualitative split and should remain the primary support-alignment metric
  - Hopfield currently shows continuous basin separation without reusable sparse support signatures: basin centroids separate, but trajectories do not reuse a stable support within each basin
- **Paper implication:** do not make a benchmark-wide paper claim based on `mode_uniqueness_rate`. Keep the support story tied to multiwell cosine/clustering positives, scope Duffing and Kuramoto as negatives, and treat Hopfield as a mixed continuous-separation-only case.

### 7. Label-free basin recovery: v2 validates label-free clustering on potential-well systems

This is the label-free clustering subfamily inside the same basin-support/mechanism family.

- **v1 result (methodology limitation, March 10):** The initial label-free clustering evaluation used trajectory-mean cosine k-means on 128 trajectories in 256 dimensions. Results:
  - Duffing (2 basins): ARI=`0.134` — all three models produce **identical** scores, confirming the feature extraction protocol (not the encoder) is the bottleneck
  - Kuramoto (5 basins): ARI≈`0` for all models
  - Competitive LV: only 1 basin observed (trivial, now fixed — see competitive_lv retrain below)
- **Root cause:** v1 protocol destroyed per-timestep support signal via trajectory averaging, suffered from concentration of measure (no PCA), and tested only the cosine feature view. The identical Duffing scores across all encoder families confirmed this was a feature-extraction issue.
- **v2 result (COMPLETE, March 10, array `8919951`, collector `8919952`):** Revised evaluation with 6 feature views, PCA to 20d, 256 trajectories, 10 systems.
  - **Multiwell systems (8 variants, 5 basins each): strong positive.**
    - `multiwell_gradient/gradient_hd`: mean ARI `0.976/0.991`, near-perfect recovery (max `1.000`)
    - `multiwell_rotational/rotational_hd`: mean ARI `0.963/0.971`
    - `multiwell_energy/energy_hd`: mean ARI `0.794/0.916`
    - `multiwell_strong_transition/strong_transition_hd`: mean ARI `0.931/0.918`
    - `generic_sparse` tends to have highest ARI; LISTA families close behind
  - **Duffing (2 basins): weak positive.** Mean ARI `0.19–0.24` across all views. Root cause: within-basin support consistency is only ~10% (basin 0: 12.8%, basin 1: 7.5%), so ~90% of trajectories activate a different support than their basin's mode. The encoder learned basin-discriminative continuous representations but not basin-aligned sparse supports on this system.
  - **Kuramoto (5 basins): negative.** Mean ARI ~`0` across all views. Supports are genuinely non-separable: within-basin vs between-basin Hamming distance ratio is 1.004 (flat). Winding-number basin distribution is highly imbalanced (q=0: 59%, q=±2: <1%). **Bug fix:** the v2 linear accuracy (~0.92–0.99 on support views) was a measurement artifact — singleton basins caused a fallback to train accuracy with no CV; corrected 3-class CV gives `0.427` (below majority baseline). Fix applied in `evaluate_label_free_clustering_v2.py`: singleton classes are now dropped before CV.
  - **Direct uniqueness update:** this v2 negative is strong evidence against basin recoverability, and the completed Kuramoto unique mode-support audit now directly confirms that Kuramoto also lacks meaningful literal basin-specific mode supports. Uniqueness is trivially degenerate (every trajectory has its own singleton support), so the Kuramoto negative is established on both the clustering and literal-support-uniqueness fronts.
  - **Feature view comparison:** `last_step_cosine` is strongest on multiwell; discrete support views (`majority_support`, `modal_support`) are competitive but do not clearly outperform PCA'd cosine features; `traj_mean_cosine` (v1 baseline) is comparable after PCA, suggesting v1 failure was primarily concentration-of-measure rather than averaging.
- **Interpretation:**
  - The multiwell positives (8/8 systems, ARI 0.71–1.00) upgrade the basin-support claim from "per-timestep uniqueness" to **"label-free basin recovery is possible"** without training-time basin labels. This is a key paper claim.
  - The kuramoto negative is genuine — supports carry zero basin-discriminative signal (flat Hamming geometry, every trajectory unique, ~93 active dims in all basins). This limits the label-free claim to potential-well systems.
  - The duffing weak result demonstrates that per-timestep support uniqueness (2/2) does not guarantee trajectory-level basin-support alignment when within-basin consistency is low (~10%). This is an honest limitation worth reporting.
- **Competitive LV multi-basin retrain and representation family (forecasting complete March 11; representation family complete March 13):** The previous `competitive_lv` benchmark was trivial (1 observed basin at `INTERACTION_SCALE=0.35`). The config is now `0.70`, producing 4 major basins. All 28/28 training tasks completed (arrays `8922033` + recovery `8922374`). Forecasting collection and comparison are done. All models are inside the good-forecast band; the paper-facing `200k` result is `generic_sparse=0.1254`, with `lista_blockdiag` and `lista_dense` close behind. The older `50k` comparison is only an auxiliary sanity check. Competitive LV is not a problematic forecasting holdout for any architecture. The related representation family is now also complete: support alignment on the corrected checkpoints remains weak/trivial, the repaired March 13 label-free clustering v2 rerun finished cleanly, and the final clustering read stays negative for stable support-defined basin recovery on the clean `4`-basin paper roots. Do NOT re-use any old 1-basin `competitive_lv` evaluation results.

## Decision Rules

- If default `dt` is poor and the benchmark rescue chain requests halving, prefer **smaller `dt`** before broader model changes or `10x` longer training.
- **The `200k` results are now the primary paper evidence.** The `v4` `50k` matrix is retained only as appendix / historical context for the matched four-model audit. All headline paper claims, cross-system rankings, and model comparisons should be drawn from the `200k` runs in `results/paper_followup_recipes_200k_20260309`.
- If a `200k` rerun materially improves over a `50k` result, do not foreground the `50k` result in the paper body.
- Use `generic_sparse_ns200k_best` as the primary paper anchor (not the `50k` `generic_sparse`).
- Use the promoted dense Stage-4 root (`lista_dense_promoted_stage4`) as the primary dense LISTA comparator.
- Use dense LISTA as the cross-system LISTA reference, but keep `lista_blockdiag` as the only LISTA-family candidate for intrinsic-HD follow-up unless new evidence clearly overturns that ranking.
- Treat the completed dense-LISTA easy-system Stage-1 sweep as evidence that external optimization alone can recover most easy dense-LISTA near-misses without changing architecture or `dt`.
- Treat the completed dense-LISTA Stage-2 / Stage-4 chain as the parity decision point:
  - stop coefficient-only holdout tuning
  - promote `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3` as the single fair dense recipe
  - use the completed Stage-4 rerun as the current dense parity evidence (`21/29` wins, `26/29` good systems, `0` dense-fails-anchor-passes systems)
  - when discussing the fair `200k` comparison, make the split explicit: dense wins more systems and keeps more systems good, while `generic_sparse_ns200k_best` has the best overall median
  - `v4` remains an appendix-only symmetric four-model audit; the `200k` follow-up is the primary paper-facing comparison
- Treat the full-benchmark block-diagonal dense-opt transfer as a negative result for global parity:
  - do not promote `lista_blockdiag_ns200k_denseopt_sc3em3` or `lista_blockdiag_ns200k_denseopt_sc6em3` as paper baselines
  - mention them only as targeted positives on `multiwell_strong_transition` / `multiwell_strong_transition_hd` or as evidence that the dense optimizer does not transfer cleanly to block-diagonal LISTA
- Treat historical block-diagonal claims through the repaired March 17/19 MLP controls only; the old mislabeled controls remain invalid provenance and should not be reused.
- Use the repaired intrinsic-HD rerun as the current decision-grade targeted evidence.
- Keep `evaluation_results_best.json` as the official checkpoint-selection rule for now, but treat `evaluation_results_last.json` as an important diagnostic on Kuramoto when discussing model-selection limits.
- Use the completed `dt=0.00625`, `200k` follow-ups and Kuramoto dimension sweep as the hard-system evidence:
  - on Kuramoto, emphasize that `lista_blockdiag` is robustly in-band at `N=16/24/32`, is not fully robust at `N=8`, and fails by `N=64`
  - on Hopfield, emphasize that smaller `dt` rescues periodic-reencoding forecasts for both models, but `generic_sparse` remains better
  - do not claim autonomous stability on the hard systems; every-step rollout errors remain the main limitation
- Treat the active Hopfield basin-count sweep as mechanism mapping only:
  - use it to test whether higher basin count changes the architecture ordering on Hopfield
  - do not use it to overwrite the canonical Hopfield paper claim unless the modified environment is explicitly framed as a new benchmark setting
- Use the completed Kuramoto dimension sweep to frame the hard-system claim:
  - claim a smaller-`dt` Kuramoto rescue for `lista_blockdiag` through `N=32`, not through `N=64`
  - make the `N=64` failure (`23.27`, `2/5` good seeds) and the non-robust `N=8` result (`8.11`, `4/5` good seeds) explicit
  - do not present promoted dense LISTA as a positive Kuramoto transfer result

## Highest-Value Audit Files

- [Current experiment log](/home/mila/l/lia/skae/docs/EXPERIMENTS.md)
- [Broad labelable-system support-alignment summary](/home/mila/l/lia/skae/results/paper_benchmark_support_alignment_20260311_v4_labelable/summary.md)
- [follow-up `200k` paper summary](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/paper_benchmark_summary.md)
- [follow-up `200k` forecasting summary](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/forecasting_summary.md)
- [fair `200k` `generic_sparse` vs canonical `generic_sparse`](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/compare/vs_canonical_generic_sparse/generic_sparse_ns200k_best_vs_generic_sparse/forecasting_comparison.md)
- [promoted dense Stage 4 vs fair `200k` `generic_sparse`](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/compare/vs_generic_sparse_ns200k_best/lista_dense_promoted_stage4_vs_generic_sparse_ns200k_best/forecasting_comparison.md)
- [Stage-4 dense rerun summary](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/collect/paper_benchmark_summary.md)
- [Stage-4 dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/compare/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3_vs_generic_sparse/forecasting_comparison.md)
- [Kuramoto dimension summary](/home/mila/l/lia/skae/results/kuramoto_dimension_sweep_dt00625_200k_20260309/collect/kuramoto_dimension_summary.md)
- [Stage-1 dense-LISTA easy-system summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/collect/paper_benchmark_summary.md)
- [Kuramoto recovery summary](/home/mila/l/lia/skae/results/kuramoto_recovery_seq8_20260305/forecasting_summary.md)
- [intrinsic-HD `dt` rescue rerun summary](/home/mila/l/lia/skae/results/intrinsic_hd_dt_rescue_20260308_rerun1/forecasting_summary.md)
- Appendix-only `50k` references:
  - [v4 paper summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/paper_benchmark_summary.md)
  - [v4 final forecasting summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/forecasting_summary.md)
  - [v4 pass-2 `dt` resolution summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/dt_resolution/pass2/dt_resolution.md)
  - [dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_dense_vs_generic_sparse/forecasting_comparison.md)
  - [block-diagonal vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md)
