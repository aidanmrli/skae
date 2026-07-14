# Control world-model state suite, 2026-06-23

## Concrete results

The CPU state-observation control world-model suite completed all 240 runs:
four DeepMind Control Suite tasks (`cartpole_swingup`, `finger_spin`,
`cheetah_run`, `walker_walk`), five model variants, four data fractions
(`0.1`, `0.25`, `0.5`, `1.0`), and three seeds. The summary artifacts are:

- `results/control_world_model_state_suite_20260623/summary.csv`
- `results/control_world_model_state_suite_20260623/summary.json`
- `/network/scratch/l/lia/skae/control_world_model_state_suite_20260623/runs`

At the full-data setting, mean test open-loop MSE at horizon 50 was lowest for
the residual MLP latent dynamics baseline (`0.297`), followed by dense
bilinear Koopman (`0.460`), dense additive Koopman (`0.616`), sparse bilinear
Koopman (`0.693`), and sparse additive Koopman (`1.062`). Reward prediction at
full data followed the same broad ordering: MLP (`1.036`), dense bilinear
(`1.433`), dense additive (`1.973`), sparse additive (`3.359`), and sparse
bilinear (`4.598`). No variant produced unstable rollouts under the suite's
norm/finite-value diagnostic. Additive Koopman variants were fastest for
rollout and random-shooting latency; at full data, sparse additive reached
about `17.3k` rollouts/s and `16.2 ms` random-shooting latency, while bilinear
and MLP transitions were slower.

## Experiment context

This was a first offline, random-policy, state-observation experiment to test
the new action-conditioned world-model scaffold. The models used compact
datasets containing observations, actions, rewards, continuations, valid masks,
and train/validation/test splits. Labels and task metadata were not used during
training. The implemented variants were sparse/dense additive Koopman,
sparse/dense bilinear Koopman, and a residual MLP latent transition baseline.
The suite did not include TD-MPC/TD-MPC2 integration, online data collection,
policy learning, pixel observations, or stochastic latent states.

## Interpretation

The first-pass sparse Koopman world-model variants do not yet show a prediction
or reward-modeling advantage over dense Koopman or MLP transitions on these
random-policy state datasets. The clearest positive signal is computational:
additive Koopman transitions are substantially faster for latent imagination
and simple random-shooting evaluation. The dense bilinear model is the strongest
Koopman predictor in this initial sweep, suggesting that state-dependent action
effects matter more than the current sparse matrix penalty in these tasks.

## Project implications

These results should not be promoted as a paper claim that sparse Koopman
world models outperform modern model-based RL baselines. They are useful
provenance for the code path and for narrowing the next experiment: the current
implementation can train and evaluate action-conditioned latent dynamics
reproducibly, but sparsity needs a better controlled comparison before it can
support a NeurIPS-facing world-model claim.

## Next steps

The next controlled run should compare against a TD-MPC-style transition on the
same replay data, tune sparse bilinear regularization separately for \(K_0\) and
\(K_j\), and evaluate on behavior-policy or expert replay rather than only
uniform random actions. If prediction becomes competitive, add return or MPC
control evaluation, then move to pixel/history encoders and stochastic
ensembles.
