# Reproduction-Grade Main Results Packet for Senior Review

Date: March 14, 2026

This note is the paper-facing handoff for senior coauthors. It is written so that a colleague who does not have this repository can still understand what was run, why it was run, how fairness was enforced, and how to reproduce the reported numbers.

The completed evidence should be written in the paper as **three main scientific families plus one appendix-only provenance bucket**:

| Family | What belongs here | Paper-writing rule |
|---|---|---|
| **Cross-system forecasting** | The full 29-system benchmark at the final 200k training budget, using the fixed sparse MLP anchor and the fixed dense LISTA comparator | This is the benchmark headline. Treat the dense recipe-selection chain as appendix-only provenance, and write the benchmark as a breadth-versus-typical-system comparison: the sparse MLP is the safer anchor through `H1000`, dense LISTA wins more systems, and dense LISTA becomes median-best only at the longest horizons. |
| **Hard-system forecasting** | Smaller-step-size rescue studies, long-horizon reevaluations of those same checkpoints, Kuramoto robustness and dimension checks, and matched parity/fairness controls | Write this as one family about when smaller step size and latent structure help, and when they still do not. |
| **Basin-support and mechanism** | Basin-support alignment, label-free clustering, the direct Kuramoto support audit, the corrected competitive Lotka-Volterra representation follow-up, and recurring-support local-linearity | Write this as one family about when latent supports do and do not align with basins. |
| **Appendix-only provenance** | Dense LISTA hyperparameter selection and the old 50k historical benchmark audit | Mention briefly if needed, but do not present it as a main scientific result family. |

## 1. Global reproduction protocol

### 1.1 Software and deterministic conventions

- Use Python 3.10+ with the locked environment from `uv.lock`.
- The recorded run environment used:
  - PyTorch `2.9.1`
  - Dysts `0.96`
  - scikit-learn `1.7.2`
- Built-in systems and Dysts systems are both integrated with fourth-order Runge-Kutta.
- Do not assume one universal seed set across the paper. The exact seed coverage is part of each family definition below, and some benchmark families intentionally use mixed seed counts across systems.
- Report medians across seeds. Do not report best-seed results.
- Select the final checkpoint by the lowest **final** validation error on a fixed 200-step validation rollout from 16 held-out initial states.
- Reproduction should preserve the exact random-number conventions:
  - a training run with seed `s` uses `s` as the master model seed
  - the fixed validation corpus is generated with seed `s + 999999`
  - forecasting evaluation initial conditions are generated with seed `s + 12345`
  - support-alignment and label-free clustering evaluations use evaluation seed `42` unless stated otherwise
- For the built-in systems, evaluation initial conditions are generated directly from the system reset distribution.
- For Dysts systems, evaluation samples from the held-out cached test trajectories described below rather than from fresh online resets.

### 1.2 Human-readable model families used throughout this note

| Name used in this note | Encoder and decoder | Latent linear dynamics | Main benchmark recipe |
|---|---|---|---|
| **Sparse MLP anchor** | Encoder: 2 hidden layers of width 64, bias terms enabled, ReLU at the encoder output. Decoder: one learned linear map from latent space to state space, with no hidden layer. | Dense latent transition matrix. | 200k steps, latent dimension 256, main learning rate `1e-4`, latent-transition learning rate `1e-5`, weight decay `1e-4`, alignment weight `1.0`, reconstruction weight `0.03`, prediction weight `1.0`, sparsity weight `0.0025`. |
| **Dense LISTA comparator** | Same 64-64 front end as the sparse MLP, followed by one LISTA refinement step after the initial shrinkage map. The LISTA hyperparameter is `alpha = 0.15`; the effective shrinkage threshold is `alpha / L`, where `L` is computed once at initialization from the decoder dictionary. Final encoder output uses ReLU after the last shrinkage. Decoder: normalized linear dictionary decoder. | Dense latent transition matrix. | 200k steps, latent dimension 256, main learning rate `5e-5`, latent-transition learning rate `5e-6`, weight decay `1e-4`, alignment weight `1.0`, reconstruction weight `0.03`, prediction weight `1.0`, sparsity weight `0.003`. |
| **Block-diagonal LISTA** | Same encoder and decoder as dense LISTA. | Block-diagonal latent transition matrix with block size 16. | Same as dense LISTA except the sparsity weight is either `0.003` or `0.006`, depending on the specific comparison. |
| **Block-diagonal MLP fairness control** | Same encoder and decoder as the sparse MLP anchor. | Block-diagonal latent transition matrix with block size 16. | Used only in fairness controls to isolate whether gains come from encoder structure or from imposing block structure on the latent transition. |

Important fairness note:

- The sparse MLP and dense LISTA headline comparison is **budget-matched** and **front-end-matched**.
- It is **not** exact parameter-count matching, because the LISTA encoder contains an additional learned recurrent encoder matrix on top of the same 64-64 front end.
- In the paper, describe this comparison as budget-matched and architecture-aligned, not as strictly parameter-matched.
- The decoder is linear in every paper-facing comparison in this note.

### 1.2A Exact encoder and decoder maps

The sparse MLP anchor uses:

- `h1 = ReLU(W1 x + b1)`
- `h2 = ReLU(W2 h1 + b2)`
- `z = ReLU(W3 h2 + b3)`
- `x_hat = W_dec z`

The dense LISTA and block-diagonal LISTA families use the same front end to produce a dense pre-activation `c = f(x)`, then apply:

- initial shrinkage: `u^(0) = shrink(c, alpha / L)`
- one recurrent refinement step: `z = ReLU(shrink(u^(0) S + c, alpha / L))`

where:

- `S` is a learned latent interaction matrix inside the encoder
- `alpha = 0.15` in the promoted dense recipe
- `L` is computed once at initialization as `1.05` times a 10-step power-iteration estimate of the spectral norm of `W_d^T W_d`, where `W_d` is the initial decoder dictionary

The LISTA-family decoder is a learned linear dictionary:

- learn a latent-to-state dictionary matrix `D`
- renormalize each latent atom of `D` to unit norm at decode time
- decode with `x_hat = z D_normalized`

### 1.3 Shared training procedure

Every paper-facing 200k training run follows the same outer protocol unless the targeted family section below says otherwise:

1. Draw a mini-batch of `256` trajectories.
2. For each trajectory, use a window of `9` consecutive states, which gives `8` one-step transitions for training.
3. Encode all states in the window once.
4. Roll out the latent linear dynamics for `8` steps from the first encoded state.
5. Decode the latent rollout back to state space.
6. Optimize the weighted sum of:
   - latent alignment loss
   - reconstruction loss
   - multi-step prediction loss
   - L1 sparsity penalty on the predicted latent states
7. Use AdamW with:
   - one learning rate for all non-latent-transition parameters
   - a smaller learning rate for the latent linear transition
   - zero weight decay on the latent linear transition
   - standard weight decay on the other parameters

To reproduce the exact training data stream rather than only the high-level recipe:

- use `8` persistent mini-batch random-number streams
- for a run with training seed `s`, initialize those `8` streams with seeds:
  - `s + 0`
  - `s + 256`
  - `s + 512`
  - `s + 768`
  - `s + 1024`
  - `s + 1280`
  - `s + 1536`
  - `s + 1792`
- at optimization step `t`, use stream number `t mod 8`
- keep each stream persistent across the full run rather than reinitializing it every time it is reused

The intended effect of this design is that:

- each run is exactly reproducible from the training seed
- the training batches are deterministic
- the same training seed reproduces the same model, checkpoint choice, and evaluation set

Shared paper-facing settings:

- Training steps: `200000`
- Mini-batch size: `256`
- Latent dimension: `256`
- Training window: `8` transitions per example
- Validation rollout horizon for checkpoint selection: `200`
- Validation initial states: `16`
- Validation checkpoint-selection cadence: every `500` training steps, plus the final training step

Exact objective used for paper-facing benchmark families:

Let:

- `H = 8` be the number of one-step transitions in the training window
- `d_x` be the observation dimension of the current system
- `z_true[t]` be the encoder output on the true state at step `t`
- `z_pred[t]` be the latent rollout after applying the learned linear latent map `t` times from the encoded initial state
- `x_true[t]` be the true state at step `t`
- `x_pred[t]` be the decoded latent rollout at step `t`
- `x_recon[t]` be the decoder output applied to `z_true[t]`

Then the per-batch losses are:

- alignment: `(1 / H) * mean_{b,t} ||z_pred[b,t] - z_true[b,t]||_2`
- reconstruction: `(1 / (H * sqrt(d_x))) * mean_{b,t} ||x_recon[b,t] - x_true[b,t]||_2`
- prediction: `(1 / (H * sqrt(d_x))) * mean_{b,t} ||x_pred[b,t] - x_true[b,t]||_2`
- sparsity: `(1 / H) * mean_{b,t} ||z_pred[b,t]||_1`

The total loss is:

- `alignment_weight * alignment`
- `+ reconstruction_weight * reconstruction`
- `+ prediction_weight * prediction`
- `+ sparsity_weight * sparsity`

Important exactness notes:

- the reconstruction term is applied to decoded **true** future latents, not to the initial state
- the sparsity penalty is applied to the predicted latent rollout, not to a separately encoded latent sequence
- built-in systems generate training windows on the fly from fresh resets
- Dysts systems draw training windows from the fixed trajectory cache in Section 1.4

Exact validation checkpoint rule:

- at each checkpoint-selection event, evaluate a 200-step rollout with **every-step re-encoding**
- use the fixed 16-state validation set generated with seed `s + 999999`
- compute the Euclidean prediction error at each step, average over the 16 validation trajectories, and select the checkpoint with the smallest error at the final validation step
- do not use best-periodic cadence, no-re-encoding rollouts, or horizon-averaged MSE for checkpoint selection

System-generation details that matter for exact reproduction:

- Hopfield:
  - the stored patterns are random `+1/-1` patterns
  - unless a system seed is explicitly fixed in a targeted study, the stored patterns change with the training seed
  - the stored-pattern generator uses seed `s + 29`, where `s` is the training seed
- Competitive Lotka-Volterra:
  - unless a fixed interaction seed is explicitly stated, the interaction matrix changes with the training seed
  - in the benchmark setting, the off-diagonal interaction coefficients are sampled uniformly on `[0, 0.70]`, symmetrized, and the diagonal is then set to `1.0`
  - with no fixed system seed, the growth-rate generator uses seed `s + 44` and the interaction-matrix generator uses seed `s + 45`
- Kuramoto benchmark:
  - the benchmark uses identical natural frequencies, so the system does not vary across seeds
  - when a robustness study uses random frequencies instead of the deterministic uniform-spread setting, the frequency generator uses seed `s + 13`
- Multiwell systems:
  - the five canonical attractor centers listed below are fixed, not resampled
  - if random centers are ever used outside the paper benchmark, they are generated from seed `s`

### 1.4 Dysts-specific training and evaluation rules

The Dysts systems were not treated like the built-in systems. To reproduce the Dysts part of the benchmark, use all of the following rules:

- Use the named systems from Dysts `0.96` with their default parameterization and default initial condition.
- Standardize each state coordinate using the mean and standard deviation stored in Dysts metadata.
- Sample the reset state as:
  - default initial condition
  - plus Gaussian noise with standard deviation `0.2 x` the coordinate-wise Dysts standard deviation
- Use a deterministic trajectory cache for training, validation, and test:
  - `200` trajectories per split
  - `30000` steps per cached trajectory
  - `2000` warmup steps before the cached segment starts
  - cache build seed `0` for the training split
  - cache build seed `1` for the validation split
  - cache build seed `2` for the test split
- Evaluate Dysts checkpoints on the held-out test cache rather than generating a fresh on-the-fly rollout set.
- The Dysts cache is independent of the model seed. This is important: training seed changes model initialization and batch selection inside the cache, but it does not rebuild the train/validation/test cache contents.

### 1.5 Shared forecasting evaluation protocol

All forecasting tables in this packet use the same evaluation definition.

For each checkpoint and each system:

- Evaluate on `100` initial conditions.
- For a run with training seed `s`, generate those initial conditions with evaluation seed `s + 12345`.
- Roll out three classes of protocol:
  - no re-encoding after the initial state
  - re-encode at every step
  - periodic re-encoding
- Test periodic re-encoding cadences:
  - built-in systems: every `10`, `25`, `50`, and `100` steps
  - Dysts systems: every `1`, `5`, `10`, `20`, `40`, `60`, `80`, `100`, `200`, `300`, `400`, `500`, and `1000` steps

Definition of the horizon metric at horizon `H`:

1. Compute the squared Euclidean prediction error at each forecast step `1, ..., H`.
2. Average those squared errors over the first `H` steps for each initial condition.
3. Average over initial conditions.

When this note says **best periodic re-encoding**, it means:

- for a given horizon,
- among the tested periodic cadences above,
- choose the one with the lowest mean horizon error.

This is an evaluation-time oracle over a fixed, predeclared set of re-encoding cadences. It is not a learned or tuned per-system policy.

Aggregation rule for every paper-facing forecasting table:

- first compute the system-level median across the available seeds for that model and system
- then compute the cross-system median across the 29 system-level medians
- if a system has only `3` collected seeds in the final benchmark family, use those `3`; do not impute missing seeds or replace them with another run

Thresholds used throughout the benchmark comparisons:

- **Good system:** horizon-1000 system-median best periodic error `< 10`
- **Catastrophic system:** horizon-1000 system-median best periodic error `>= 1000`

### 1.6 Step-size table fixed before the final benchmark

All final benchmark and follow-up experiments reuse the pass-2 step-size table below. The step size was chosen system-by-system from the default step size, the half step size, or the quarter step size before the final 200k benchmark comparisons were run.

## 2. Benchmark system catalog

### 2.1 Low-dimensional built-in systems

| System | State dimension | Exact definition used in the paper | Reset distribution | Selected dt |
|---|---:|---|---|---:|
| Duffing | 2 | `x' = v`, `v' = x - x^3` | `x` uniform on `[-1.5, 1.5]`, `v` uniform on `[-1, 1]` | `0.01` |
| Lotka-Volterra | 2 | Predator-prey system with `alpha = beta = gamma = delta = 0.2`: prey' = `0.2*prey - 0.2*prey*predator`, predator' = `0.2*prey*predator - 0.2*predator` | Both populations uniform on `[0.02, 3.0]` | `0.0025` |
| Blended linear system | 2 | `x' = sum_{k=1}^3 w_k(x) A_k (x - c_k)`, with `w_k(x) = exp(-||x-c_k||^2 / (2*sigma^2)) / sum_j exp(-||x-c_j||^2 / (2*sigma^2))`, `sigma = 1.5`, centers `c_1=(2,0)`, `c_2=(-2,2)`, `c_3=(-2,-2)`, and matrices `A_1=[[-0.5,-3.0],[3.0,-0.5]]`, `A_2=[[-0.2,0],[0,-5.0]]`, `A_3=[[-5.0,0],[0,-2.0]]`. | Uniform on `[-4, 4]^2` | `0.05` |

### 2.2 Multiwell family

All multiwell systems use the same five canonical attractor centers:

- `(-1, -1)`
- `(1, -1)`
- `(-1, 1)`
- `(1, 1)`
- `(0, 0)`

Shared multiwell constants:

- Gaussian-well width `sigma = 0.7`
- center scale `2.0`
- minimum center separation `0.6`
- reset range `[-2.5, 2.5]^d`
- exact scalar field in the 2D core:
  - `V(x) = -sigma^2 * sum_j exp(-||x - c_j||^2 / sigma^2)`
  - `grad V(x) = 2 * sum_j (x - c_j) * exp(-||x - c_j||^2 / sigma^2)`
- `R90(a, b) = (-b, a)`
- the 8-dimensional variants use an **embedded** construction:
  - the first two coordinates follow the 2D multiwell dynamics
  - the remaining six coordinates decay linearly toward zero

| System | State dimension | Exact velocity field | Selected dt |
|---|---:|---|---:|
| Multiwell gradient | 2 | `x' = -grad V(x)` | `0.02` |
| Multiwell rotational | 2 | `x' = -grad V(x) + 0.6 * R90(grad V(x))` | `0.02` |
| Multiwell energy | 2 | `x' = -grad V(x) + 1.2 * sin(||x||) * x` | `0.02` |
| Multiwell strong transition | 2 | `x' = -grad V(x) + 0.8 * R90(grad V(x)) + 1.0 * cos(2 ||x||) * x` | `0.005` |
| Multiwell gradient, embedded high-dimensional | 8 | First 2 coordinates use `x' = -grad V(x)`; coordinates `3-8` use `x_i' = -x_i` | `0.005` |
| Multiwell rotational, embedded high-dimensional | 8 | First 2 coordinates use `x' = -grad V(x) + 0.6 * R90(grad V(x))`; coordinates `3-8` use `x_i' = -x_i` | `0.005` |
| Multiwell energy, embedded high-dimensional | 8 | First 2 coordinates use `x' = -grad V(x) + 1.2 * sin(||x||) * x`; coordinates `3-8` use `x_i' = -x_i` | `0.02` |
| Multiwell strong transition, embedded high-dimensional | 8 | First 2 coordinates use `x' = -grad V(x) + 0.8 * R90(grad V(x)) + 1.0 * cos(2 ||x||) * x`; coordinates `3-8` use `x_i' = -x_i` | `0.005` |

### 2.3 High-dimensional built-in systems with explicit basin labels

| System | State dimension | Exact definition used in the paper | Reset distribution | Basin label used only for evaluation | Selected dt |
|---|---:|---|---|---|---:|
| Kuramoto | 16 in the benchmark | Ring-coupled phase oscillators with periodic boundary conditions: `theta_i' = omega_i + (4.0 / N) * (sin(theta_{i+1} - theta_i) + sin(theta_{i-1} - theta_i))`. In the benchmark, `omega_i = 0` for all `i`. | Each phase uniform on `[-pi, pi]` | Integer winding number around the ring | `0.0125` |
| Hopfield | 16 in the benchmark | Continuous Hopfield dynamics `x' = (-x + W tanh(x) + b) / 1.0`, `16` neurons, `4` stored random patterns in the benchmark, `beta = 1.0`, `tau = 1.0`, zero bias. The stored patterns are drawn as independent random `+1/-1` vectors and the recurrent weight matrix is `(P^T P) / m` with the diagonal reset to zero. | Gaussian with standard deviation `1.0` in each coordinate | Index of stored pattern with maximal sign overlap | `0.0125` |
| Competitive Lotka-Volterra | 10 in the benchmark | `N`-species competitive system `x_i' = x_i (r_i - sum_j A_ij x_j)` with `r_i = 1`, `A_ii = 1`, symmetric `A_ij` generated by sampling off-diagonal entries uniformly on `[0, 0.70]` and then symmetrizing, positivity clipping after each RK4 step, and survival threshold `1e-3`. | Each coordinate uniform on `[0.05, 1.5]` | Survivor-support bitmask | `0.01` |

Exact competitive Lotka-Volterra equation used in the repaired controls:

- `x_i' = x_i (r_i - sum_j A_ij x_j)` for `i = 1, ..., N`
- `r_i = 1` for all species in the repaired paper-facing settings
- `A_ii = 1`; for symmetric settings, the off-diagonal entries are sampled on `[0, s]` and then symmetrized
- after each RK4 step, coordinates are clipped below `0`
- the evaluation-time basin label is the survivor-support bitmask `sum_i 1{x_i > 1e-3} 2^i`

### 2.4 Dysts systems

For all Dysts systems below:

- use Dysts `0.96`
- use the Dysts default parameterization of the named system
- use the Dysts default initial condition plus the Gaussian perturbation rule in Section 1.4
- standardize coordinates using Dysts metadata

| Dysts system | State dimension | Native Dysts dt | Selected dt used in the benchmark |
|---|---:|---:|---:|
| Dadras | 3 | `0.0006578296382730287` | `0.0006578296382730287` |
| Duffing | 3 | `0.0014901683613936711` | `0.0003725420903484178` |
| QiChen | 3 | `7.837106184364728e-05` | `7.837106184364728e-05` |
| Sakarya | 3 | `0.0009970461743625946` | `0.0009970461743625946` |
| SprottTorus | 3 | `0.0027218536422742544` | `0.0027218536422742544` |
| Chua | 3 | `0.0002847474579095888` | `0.0002847474579095888` |
| MultiChua | 3 | `0.00045179529921654196` | `0.00022589764960827098` |
| DequanLi | 3 | `1.6763993998996084e-05` | `4.190998499749021e-06` |
| LuChenCheng | 3 | `0.00018469678279714685` | `0.00018469678279714685` |
| SanUmSrisuchinwong | 3 | `0.0014933288881479711` | `0.0014933288881479711` |
| WangSun | 3 | `0.005392498749791912` | `0.001348124687447978` |
| ShimizuMorioka | 3 | `0.002408001333556058` | `0.0006020003333890145` |
| LorenzCoupled | 6 | `0.0003241940323387382` | `8.104850808468456e-05` |
| RikitakeDynamo | 3 | `0.0011796966161026034` | `0.00029492415402565086` |
| Hadley | 3 | `0.00029086847807974437` | `0.00029086847807974437` |

## 3. Cross-system forecasting family

### 3.1 Aim

Test whether dense LISTA is globally competitive against a fair sparse MLP anchor once the following are matched:

- training budget
- latent dimension
- front-end width and depth
- per-system step size
- seed set
- forecasting evaluation protocol

### 3.2 What is being changed, and what is being held fixed

Intervention:

- Replace the sparse MLP encoder with a one-iteration dense LISTA encoder while keeping the decoder, latent dimension, task suite, step-size table, and forecasting metric fixed.

Secondary intervention:

- Replace the dense latent linear transition with a block-diagonal transition to test whether the transition structure alone yields a broad benchmark gain.

Held fixed across the main sparse-MLP versus dense-LISTA benchmark:

- same 29 systems
- same selected step size for each system
- same `200000` training steps
- same latent dimension `256`
- same 64-64 front end before the final encoder map
- same evaluation horizons
- matched seed coverage for the sparse MLP anchor and dense LISTA comparator

Not held fixed:

- exact parameter count

### 3.3 Exact experimental setup

- Systems: all 29 systems in Section 2
- Training budget: `200000` steps per run
- Sparse MLP anchor and dense LISTA comparator use matched seed coverage:
  - seeds `0-4` on `21` systems
  - seeds `0-2` on `8` systems: `competitive_lv`, `kuramoto`, `hopfield`, `multiwell_gradient_hd`, `multiwell_rotational_hd`, `multiwell_energy_hd`, `multiwell_strong_transition_hd`, and `dysts:LorenzCoupled`
- Block-diagonal LISTA transfer checks use seeds `0, 1, 2` on all 29 systems
- Forecast horizons in the original benchmark summary: `100`, `500`, `1000`
- Long-horizon reevaluation: the **same trained checkpoints** were reevaluated at `1500`, `2000`, `2500`, and `3000`
- Sparse MLP recipe: Section 1.2
- Dense LISTA recipe: Section 1.2
- Block-diagonal LISTA transfer checks:
  - same dense LISTA optimizer as the promoted dense comparator
  - block size `16`
  - sparsity weight `0.003` or `0.006`
- Checkpoint selection for the headline table:
  - use the best validation checkpoint selected during training
  - do not switch to the final checkpoint after seeing the test results
- Long-horizon extension:
  - reuse the exact same selected checkpoints from the `H100-H1000` benchmark
  - do not retrain for the `H1500-H3000` table

### 3.4 Why this design isolates the intended claim

- The dense LISTA comparator is fixed **before** the final 29-system fair benchmark. This prevents per-system retuning on the benchmark itself.
- The sparse MLP anchor and dense LISTA comparator see the same systems, same step sizes, same budget, same seeds, and same forecasting metric.
- The block-diagonal transfer checks test whether the dense optimizer itself rescues the block-diagonal model family benchmark-wide.
- The fixed-cadence re-encoding ablation checks whether the dense win-count advantage depends on an oracle choice of re-encoding cadence.

### 3.5 How to read the benchmark tables

- The benchmark needs to answer two different questions:
  - which model is better on a **typical** system after taking the system-level median across seeds
  - on how many of the `29` matched systems one model has the lower system-level seed median
- Dense LISTA and the sparse MLP trade off these two notions of strength, so the main table should report both rather than forcing the reader to infer one from the other.

### 3.6 Main result

| Horizon | Sparse MLP anchor median | Dense LISTA comparator median | Dense LISTA wins | Good systems, Sparse MLP | Good systems, Dense LISTA |
|---|---:|---:|---:|---:|---:|
| `H100` | **`3.146e-4`** | `4.337e-4` | `11/29` | `27/29` | `28/29` |
| `H500` | **`0.0050`** | `0.0052` | `17/29` | `25/29` | `25/29` |
| `H1000` | **`0.0233`** | `0.0257` | `18/29` | `25/29` | `26/29` |
| `H1500` | **`0.0390`** | `0.0430` | `19/29` | `26/29` | `26/29` |
| `H2000` | `0.1015` | **`0.0627`** | `18/29` | `26/29` | `26/29` |
| `H2500` | `0.1593` | **`0.0910`** | `17/29` | `26/29` | `24/29` |
| `H3000` | `0.2087` | **`0.0940`** | `18/29` | `26/29` | `24/29` |

Key benchmark read:

- Through `H1000`, the sparse MLP keeps the better cross-system median even after dense LISTA starts winning more systems.
- Dense LISTA becomes median-best only at `H2000-H3000`.
- That late-horizon gain comes with worse coverage than the sparse MLP at `H2500-H3000`.
- Packet-table alignment note: the built-in appendix table should use fixed `8`-basin `competitive_lv`, `dt = 0.005` as the canonical CLV row, and the matching canonical Kuramoto row is now being rebuilt under [results/kuramoto_dt0p01_200k_canonical_20260323](/home/mila/l/lia/skae/results/kuramoto_dt0p01_200k_canonical_20260323) for `N = 16`, identical frequencies, `dt = 0.01`, `200k`, with sparse MLP, zero-sparsity MLP, block-diagonal MLP, dense LISTA, and block-diagonal LISTA all included.

### 3.7 Benchmark-wide block-structure read

- The same `29`-system family can isolate two causal changes cleanly:
  - replace the sparse-MLP encoder with dense LISTA while keeping a dense latent transition
  - impose a block-diagonal latent transition within the LISTA family while keeping the LISTA encoder family and dense-optimizer recipe fixed
- It cannot isolate the pure effect of a block-diagonal transition within the MLP family on the full `29`-system benchmark, because no benchmark-wide block-diagonal MLP rerun was collected.
- That narrower control is available only in the targeted hard-system fairness studies below.

| Contrast on the same `29`-system benchmark | `H1000` seed-median evidence | Systems won | Good systems | Causal read |
|---|---|---:|---|---|
| Dense LISTA versus sparse MLP anchor | `0.0257` versus `0.0233` | `18/29` | `26/29` versus `25/29` | Changing the encoder family improves breadth at `H1000`, but not the typical-system median. |
| Block-diagonal LISTA, sparsity `0.003`, versus sparse MLP anchor | `0.0524` versus `0.0233` | `7/29` | `26/29` versus `25/29` | Dense-optimizer transfer does not rescue block-diagonal LISTA benchmark-wide. |
| Block-diagonal LISTA, sparsity `0.006`, versus sparse MLP anchor | `0.0766` versus `0.0233` | `5/29` | `27/29` versus `25/29` | The stronger sparsity setting slightly improves coverage, but it is even worse on the typical system and on win count. |
| Block-diagonal LISTA, sparsity `0.003`, versus dense LISTA | `0.0524` versus `0.0257` | `4/29` | `26/29` versus `26/29` | Within LISTA, imposing block structure hurts the benchmark median without improving breadth. |
| Block-diagonal LISTA, sparsity `0.006`, versus dense LISTA | `0.0766` versus `0.0257` | `4/29` | `27/29` versus `26/29` | Within LISTA, block structure buys at most a one-system coverage gain while sharply worsening typical performance. |

### 3.8 Oracle-dependence ablation

- Under one fixed global `periodic_100` cadence, dense LISTA still wins `17/29` systems against the sparse MLP, so the win-count edge is not an oracle-only artifact.
- But the good-system counts drop to a `22/29` tie and dense LISTA newly fails the blended linear system while the sparse MLP passes it.
- The fixed-cadence check therefore weakens any claim that dense LISTA is the safer default.

### 3.9 Interpretation for the paper

- The fair benchmark headline is **not** "LISTA wins the benchmark."
- The fair benchmark headline is:
  - sparse MLP is the safer benchmark anchor through `H1000`
  - dense LISTA is globally competitive
  - dense LISTA becomes median-best only at longer horizons `H2000-H3000`
  - that longer-horizon gain comes with worse catastrophic-outlier coverage

## 4. Hard-system forecasting family

### 4.1 Family aim

Test whether the poor benchmark behavior on Kuramoto, Hopfield, and the repaired competitive Lotka-Volterra controls is mainly a step-size problem, a model-family problem, or both.

### 4.2 Family-wide causal design

Across all hard-system studies below, the intended intervention is always one of:

- reduce the integration step size
- add block structure to the latent transition
- replace the sparse MLP encoder with LISTA

The family is designed to isolate those effects as follows:

- within each targeted setting, all compared models use the **same** system, **same** step size, **same** training budget, **same** latent dimension, and **same** evaluation protocol
- the block-diagonal MLP control is used wherever needed to separate encoder effects from latent-transition structure
- the parity sweep matches tuning budget across families:
  - Stage 1: `6` coarse configurations per family per setting, `100000` steps, seeds `0, 1`
  - Stage 2: `1` selected configuration per family per setting, `200000` steps, seeds `0, 1, 2`
- parity-stage selection is pre-registered:
  1. lower median `H1000` best periodic error
  2. more good seeds
  3. fewer catastrophic seeds
  4. lower median `H1000` every-step error
  5. simpler setting if still tied: larger step size, lower sparsity, or inherited recipe over specialized recipe

### 4.3 Smaller-step-size Kuramoto and Hopfield rescue studies

#### Aim

Test whether step-size reduction alone substantially rescues hard-system forecasting, and whether block-diagonal LISTA benefits more from the smaller step size than the sparse MLP.

#### Exact settings

| Setting | Exact system specification | Models compared | Seeds |
|---|---|---|---:|
| Kuramoto targeted rescue | `16` oscillators, ring coupling, identical natural frequencies, coupling constant `4.0`, step size `0.00625` | Sparse MLP, dense LISTA, block-diagonal LISTA | `0-6` |
| Hopfield targeted rescue | `16` neurons, `16` stored patterns, random stored patterns, `beta = 1.0`, `tau = 1.0`, step size `0.00625` | Sparse MLP, block-diagonal LISTA | `0-2` |

Recipes used:

- Sparse MLP: same as the main benchmark anchor
- Dense LISTA and block-diagonal LISTA in Kuramoto:
  - same one-iteration LISTA encoder
  - LISTA `alpha = 0.15`, so the effective shrinkage threshold is `alpha / L`
  - block size `16` when block-diagonal
  - sparsity weight `0.006`
- Hopfield targeted block-diagonal LISTA:
  - same one-iteration LISTA encoder
  - LISTA `alpha = 0.15`, so the effective shrinkage threshold is `alpha / L`
  - block size `16`

#### Results

| Setting | Quantitative result | Interpretation |
|---|---|---|
| Kuramoto, `H100/H500/H1000` | Sparse MLP / block-diagonal MLP / block-diagonal LISTA / dense LISTA = `0.0343 / 0.1278 / 0.1536 / 0.2194`, `1.4972 / 2.4366 / 2.6834 / 4.1222`, `27.02 / 6.39 / 6.98 / 13.84` | The repaired block-diagonal MLP control still loses early, but it becomes the best model by `H1000`. |
| Kuramoto, `H1500/H2000/H2500/H3000` | Sparse MLP / block-diagonal MLP / block-diagonal LISTA / dense LISTA = `547.37 / 9.87 / 10.93 / 54.85`, `1.208e+04 / 12.74 / 14.52 / 205.26`, `3.370e+05 / 15.13 / 17.94 / 541.19`, `9.207e+06 / 17.59 / 21.58 / 1519.09` | The long-horizon Kuramoto positive is real for the repaired block-diagonal MLP control as well, although all models are outside the good band beyond `H1000`. |
| Hopfield, `H100/H500/H1000/H1500/H2000/H2500/H3000` | Sparse MLP = `0.0500 / 0.8835 / 3.3642 / 6.6116 / 9.1682 / 10.9565 / 12.2317`; block-diagonal LISTA = `0.1075 / 3.5303 / 8.8212 / 12.2297 / 12.8285 / 13.2329 / 13.5834` | Smaller step size helps both models, but Hopfield remains sparse-MLP-better at every horizon. |

### 4.4 Kuramoto robustness and dimension checks

#### Aim

Test whether the Kuramoto rescue is a narrow one-setting artifact or whether it survives moderate changes in dimension and frequency heterogeneity.

#### Exact settings

| Study | Exact intervention | Models compared | Seeds |
|---|---|---|---:|
| Moderate-dimension confirmation | Same as the targeted rescue, but `32` oscillators instead of `16`, step size `0.00625` | Sparse MLP and block-diagonal LISTA in the dedicated confirmation; matched block-diagonal MLP and dense LISTA also exist from the four-model dimension sweep at the same setting | `0-4` |
| Frequency-spread robustness | Same as the targeted rescue, but natural frequencies fixed to a linear spacing from `-0.5` to `0.5`, still `16` oscillators and step size `0.00625` | Sparse MLP, block-diagonal MLP, dense LISTA, and block-diagonal LISTA; dense LISTA was backfilled on March 20 after being omitted from the original March 9 manifest | `0-4` |
| Dimension sweep | `N in {8, 16, 24, 32, 64}`, ring coupling, step size `0.00625`, same 200k budget | Sparse MLP, block-diagonal MLP, dense LISTA, and block-diagonal LISTA | `5` seeds per setting |

#### Results

| Setting | Quantitative result | Interpretation |
|---|---|---|
| Kuramoto `N = 32`, `H1000/H1500/H2000/H2500/H3000` | Sparse MLP / block-diagonal LISTA = `6.65 / 6.00`, `13.60 / 10.89`, `24.27 / 16.14`, `42.19 / 21.69`, `75.25 / 27.95` | The moderate-dimension rescue survives to `H3000`, but only as a relative advantage, not as an in-band result. |
| Kuramoto with uniform frequency spread, `H1000/H1500/H2000/H2500/H3000` | Sparse MLP / block-diagonal MLP / dense LISTA / block-diagonal LISTA = `44.46 / 8.13 / 16.55 / 9.53`, `913.45 / 24.76 / 46.38 / 28.41`, `3.069e+04 / 91.23 / 132.35 / 117.46`, `8.926e+05 / 399.74 / 415.24 / 523.72`, `2.729e+07 / 1724.45 / 1530.83 / 2129.37` | The March 20 dense-LISTA backfill does not change the main read: the repaired block-diagonal MLP remains best through `H2500`, and dense LISTA becomes slightly lower only at `H3000` after all structured roots are already out of band. |

Packet-table audit note:

- The appendix table now backfills Kuramoto `N = 32` with the recovered block-diagonal MLP row from the March 19 dimension-sweep retry root: `0.1136 / 1.6177 / 5.16 / 9.28 / 13.32 / 17.04 / 20.42` at `H100/H500/H1000/H1500/H2000/H2500/H3000`.
- The same appendix row now also includes the dense-LISTA dimension-sweep values: `52.6191 / 77.2023 / 92.28 / 102.11 / 109.53 / 114.68 / 119.39`.
- Those two recovered rows are same-setting results, but they do not come from the dedicated March 9 `N = 32` confirmation collector; the sparse-MLP and block-diagonal-LISTA entries in that packet row remain the dedicated confirmation values.
- The March 20 dense-LISTA backfill for uniform-spread Kuramoto `N = 16` completed cleanly under [results/kuramoto_uniform_spread_dense_20260320](/home/mila/l/lia/skae/results/kuramoto_uniform_spread_dense_20260320): `0.2081 / 4.1470 / 16.55 / 46.38 / 132.35 / 415.24 / 1530.83` at `H100/H500/H1000/H1500/H2000/H2500/H3000`.

Kuramoto dimension sweep, seed-median best-periodic errors:

| N | Horizon | Sparse MLP | Block-diagonal MLP | Dense LISTA | Block-diagonal LISTA |
|---:|---|---:|---:|---:|---:|
| 8 | `H1000` | 813.57 | 10.61 | 495.07 | **8.11** |
| 8 | `H3000` | `3.145e11` | **26.48** | `4.128e11` | 51.67 |
| 16 | `H1000` | 30.18 | **6.51** | 13.44 | 7.07 |
| 16 | `H3000` | `1.056e7` | **17.22** | 433.29 | 21.99 |
| 24 | `H1000` | 6.71 | **5.79** | 14.99 | 6.57 |
| 24 | `H3000` | 5021.08 | **19.38** | 694.54 | 26.48 |
| 32 | `H1000` | 6.68 | **5.16** | 92.28 | 5.92 |
| 32 | `H3000` | 73.75 | **20.42** | 119.39 | 25.01 |
| 64 | `H1000` | 208.93 | 208.54 | 208.71 | **23.27** |
| 64 | `H3000` | 224.07 | 223.71 | 224.11 | **87.64** |

Interpretation:

- The repaired block-diagonal MLP is best by median at `N = 16/24/32` at both `H1000` and `H3000`.
- At `N = 8`, it is slightly worse than block-diagonal LISTA at `H1000`, but lower by `H3000`; both structured models are still far outside the paper’s good band by that horizon.
- At `N = 64`, the repaired block-diagonal MLP remains catastrophic, and block-diagonal LISTA is clearly better.
- Recheck note: the `N = 64` repaired block-diagonal MLP row was verified directly from the raw per-seed `evaluation_results_best.json` files. It is a real bimodal seed split, not a collector typo: seeds `0` and `4` are good (`5.97/24.70` and `5.56/23.92` at `H1000/H3000`), while seeds `1-3` fail near `209/224`.

### 4.5 Higher-basin Hopfield and repaired competitive Lotka-Volterra fairness checks

#### Aim

Test whether the negative hard-system read changes once the repaired block-diagonal MLP controls are substituted for the historical invalid rows.

#### Exact settings

Reported settings:

- Higher-basin Hopfield fairness control:
  - `N = 64`, quarter-step `dt = 0.0015625`, `200k`
- Competitive Lotka-Volterra fairness / follow-up settings:
  - both repaired CLV families use the same `N`-species ODE `x_i' = x_i (r_i - sum_j A_ij x_j)` with uniform growth `r_i = 1`, `A_ii = 1`, symmetric off-diagonal interactions, RK4 integration, positivity clipping after each step, reset `x_i(0) ~ Uniform[0.05, 1.5]`, and basin labels defined by the survivor-support bitmask at survival threshold `1e-3`
  - corrected `4`-basin rebuild: `N = 10`, interaction scale `s = 0.70`, symmetric `A_ij ~ Uniform[0, 0.70]` then symmetrized, `dt = 0.01`, `SYSTEM_SEED = -1` so the interaction matrix varies with the training seed, full `15`-seed `200k` rebuild
  - fixed `8`-basin probe: `N = 15`, interaction scale `s = 0.83`, symmetric `A_ij ~ Uniform[0, 0.83]` then symmetrized, fixed `SYSTEM_SEED = 0`, `200k`
  - reported step sizes for the fixed `8`-basin probe: `dt = 0.005` and `dt = 0.0025`
  - the labels `4`-basin and `8`-basin are empirical shorthand for the observed major survivor-support basins of those settings; they are not quantities assumed known at training time

Model families reported for these controls:

- sparse MLP
- block-diagonal MLP fairness control
- dense LISTA
- block-diagonal LISTA

Recipe note:

- The block-diagonal MLP control mirrors the non-encoder recipe of the paired block-diagonal LISTA setting exactly.
- For the corrected `4`-basin rebuild, the paper-facing source is the full `15`-seed collector, not the earlier mixed-coverage intermediate collector.

#### Results

Higher-basin Hopfield fairness control, with the best value at each horizon in bold:

| Model | H100 | H500 | H1000 | H1500 | H2000 | H2500 | H3000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sparse MLP | **0.1920** | **8.8258** | **111.75** | **309.92** | **520.71** | **711.04** | **873.50** |
| Block-diagonal MLP | 0.5782 | 45.3432 | 322.81 | 653.30 | 997.21 | 1284.10 | 1515.62 |
| Dense LISTA | 56.5293 | 94.1468 | 312.42 | 591.50 | 842.59 | 1046.18 | 1206.93 |
| Block-diagonal LISTA | 40.3230 | 99.0692 | 378.82 | 735.26 | 1035.51 | 1274.93 | 1461.51 |

Interpretation:

- The repaired block-diagonal MLP control is also strongly negative; Hopfield remains sparse-MLP-best at every horizon.

Competitive Lotka-Volterra fairness / follow-up results, consolidated into one table with one row block per setting and the best value in each setting-horizon column in bold:

| Setting | Model | H100 | H500 | H1000 | H1500 | H2000 | H2500 | H3000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| corrected `4`-basin, full `15`-seed rebuild | Sparse MLP | **0.0108** | **0.1077** | 0.1530 | 0.1864 | 0.1992 | 0.2026 | 0.2218 |
|  | Block-diagonal MLP | 0.0126 | 0.1615 | 0.2016 | 0.2729 | 0.3384 | 0.3962 | 0.4457 |
|  | Dense LISTA | 0.0119 | 0.1108 | **0.1489** | **0.1618** | **0.1874** | **0.1938** | **0.2101** |
|  | Block-diagonal LISTA (`sc6em3`) | 0.0570 | 0.1594 | 0.1660 | 0.1916 | 0.2031 | 0.2308 | 0.2577 |
|  | Block-diagonal LISTA (`sc3em3`) | 0.0492 | 0.9919 | 1.702e+12 | 1.702e+12 | 1.702e+12 | 1.702e+12 | 1.702e+12 |
| fixed `8`-basin, `dt = 0.005` | Sparse MLP | 0.0732 | 0.3554 | 0.4630 | 0.5081 | 0.5351 | 0.5548 | 0.5707 |
|  | Block-diagonal MLP | 0.2477 | 0.5037 | **0.4253** | **0.4020** | **0.3961** | **0.3976** | **0.4024** |
|  | Dense LISTA | **0.0427** | 0.3687 | 0.4856 | 0.5298 | 0.5573 | 0.5762 | 0.5909 |
|  | Block-diagonal LISTA | 0.0708 | **0.3403** | 0.4376 | 0.4770 | 0.4999 | 0.5163 | 0.5296 |
| fixed `8`-basin, `dt = 0.0025` | Sparse MLP | **0.0025** | **0.1222** | **0.2576** | **0.3266** | **0.3637** | **0.3879** | **0.4055** |
|  | Block-diagonal MLP | 0.0105 | 0.2517 | 0.4537 | 0.5525 | 0.6096 | 0.6467 | 0.6733 |
|  | Dense LISTA | 0.0096 | 0.2624 | 0.4731 | 0.5742 | 0.6302 | 0.5104 | 0.5398 |
|  | Block-diagonal LISTA | 0.0198 | 0.3424 | 0.5702 | 0.6758 | 0.7342 | 0.7708 | 0.7969 |

Interpretation:

- Corrected `4`-basin CLV stays negative for the block-diagonal MLP control. The system remains easy overall, and dense LISTA is best from `H1000` through `H3000`.
- Fixed `8`-basin CLV at `dt = 0.005` is the only CLV setting where the repaired block-diagonal MLP becomes best, and it does so only from `H1000` onward.
- Halving the fixed `8`-basin step size to `dt = 0.0025` removes that positive; the sparse MLP is again best at every horizon.

| Setting | Quantitative result | Interpretation |
|---|---|---|
| Matched hard-system parity sweep | On `14` confirmed settings, the sparse MLP is best on all `14`. Dense LISTA has `0` wins, `10` outright losses, and `4` rows with worse seed robustness. Block-diagonal LISTA has `0` wins, `8` outright losses, and `6` rows with worse seed robustness. | Matched tuning budget does not rescue LISTA on Hopfield or corrected higher-basin competitive Lotka-Volterra. |

### 4.6 Interpretation for the paper

- Smaller step size is a real causal lever on the hard systems.
- The strongest positive hard-system result is still the smaller-`dt` Kuramoto family, but the repaired MLP controls broaden the causal read beyond the original targeted `N=16` identical-frequency table.
- The repaired block-diagonal MLP mirrors are clean positives on identical-frequency Kuramoto at `N=16/24/32` and on the uniform-spread `N=16` follow-up. In the dimension sweep they are not best at `N=8` at `H1000`, though they do beat block-diagonal LISTA there by `H3000`; `N=64` remains a real catastrophic failure with a verified bimodal seed split rather than a table typo.
- That Kuramoto result must be written carefully: it is a targeted structure-helping-positive on a narrow oscillatory family, not a general hard-system or benchmark-wide LISTA win.
- Hopfield remains the clearest negative result for LISTA under matched tuning.
- Corrected `4`-basin competitive Lotka-Volterra is not a block-diagonal MLP success story; dense LISTA becomes best from `H1000` onward, and the repaired block-diagonal MLP control is worse than both dense roots at every long horizon.
- Fixed `8`-basin competitive Lotka-Volterra gives one narrow block-diagonal MLP positive at `dt = 0.005`, but that gain disappears at `dt = 0.0025`, so the CLV structure effect is step-size-specific rather than robust.

## 5. Basin-support and mechanism family

### 5.1 Family aim

Test whether good forecasting is accompanied by **basin-support alignment**:

- trajectories from the same basin should reuse the same sparse support
- different basins should separate in latent support space
- recurring supports should support locally linear latent dynamics

This family uses basin labels **only for evaluation**, never for training.

### 5.2 Family-wide causal design

This family is designed to separate representation effects from training-budget effects:

- it reuses already-trained forecasting checkpoints instead of launching special representation-only training runs
- for a given system, all models are evaluated on the same trajectory corpus
- basin labels are assigned from long rollouts of the same corpus, so the label source is shared across models
- the direct Kuramoto audit and the corrected competitive Lotka-Volterra rerun act as negative controls
- the recurring-support local-linearity study compares each support-conditioned fit against:
  - a single global linear map
  - a shuffled-group baseline with the same group sizes

### 5.3 Labeled support-alignment audit

#### Aim

Test whether a basin corresponds to a reusable sparse support pattern, rather than only to a continuous latent centroid.

#### Exact protocol

- Systems evaluated:
  - Duffing
  - all 8 multiwell variants
  - Kuramoto
  - Hopfield
  - corrected 4-basin competitive Lotka-Volterra
- Models:
  - sparse MLP
  - dense LISTA
  - block-diagonal LISTA
  - historical diagonal LISTA where available for diagnostics only
- Seeds:
  - `0, 1, 2`
- Trajectories per system-model-seed:
  - `100`
- Trajectory length before basin assignment:
  - `500`
- Extra rollout for basin assignment:
  - `5000` steps
- Support threshold:
  - `1e-3`
- Support representation used in the main table:
  - average the latent trajectory over time
  - mark a latent coordinate as active if the absolute value of that time average exceeds `1e-3`
- Cosine-separation representation:
  - average the latent trajectory over time
  - compute mean within-basin cosine similarity
  - compute mean between-basin cosine similarity
- report cosine separation as within minus between

Exact metric definitions for the support-alignment audit:

- for each basin, define the **basin mode support** as the most common binary support among that basin's trajectories
- **mean basin consistency** = average, over basins, of the fraction of trajectories in that basin that realize the basin mode support
- **mode uniqueness rate** = fraction of basin pairs whose basin mode supports are different
- **trajectory-unique support rate** = number of distinct trajectory supports divided by number of evaluated trajectories
- **mean within-basin Hamming distance** and **mean between-basin Hamming distance** are computed over all trajectory-support pairs; the reported ratio is between divided by within
- **cosine separation score** = mean within-basin cosine similarity of the aggregated continuous latent vectors minus mean cosine similarity between basin centroids

Exact trajectory-generation rule:

- use evaluation seed `42`
- for trajectory index `i`, reset the environment with seed `42 + i`
- roll out the trajectory for the stated trajectory length
- append the initial state, so each stored trajectory contains the initial state plus all rollout states
- assign the basin label by taking the final state of that stored trajectory and rolling it forward for `5000` additional steps
- use the basin reached after that additional rollout as the evaluation label

Sampling rule:

- use random sampling, not balanced-per-basin sampling, unless a study explicitly says otherwise
- the main results in this packet use random sampling throughout

#### Why this tests the intended claim

- If a model has true basin-support alignment, trajectories from the same basin should repeatedly activate the same sparse support.
- The long rollout used for basin assignment avoids using a transient label from the initial condition alone.
- Because all models on a system share the same evaluation corpus, differences are attributable to the learned representation, not to different sampled trajectories.

### 5.4 Label-free clustering audit

#### Aim

Test whether basin structure can be recovered from latent supports **without** using basin labels as clustering inputs.

#### Exact protocol

- Systems:
  - the multiwell family
  - Duffing
  - Kuramoto
  - Hopfield
  - corrected competitive Lotka-Volterra
- Trajectories per checkpoint:
  - `256`
- Trajectory length:
  - `256`
- Extra rollout for basin labeling used only for scoring:
  - `5000`
- Support threshold:
  - `1e-3`
- Feature views:
  - modal support over time
  - majority support over time
  - support at the final time step
  - final latent vector after cosine normalization
  - thresholded support of the trajectory-mean latent vector
  - cosine-normalized trajectory-mean latent vector
- Clustering:
  - K-means
  - number of clusters fixed to the true number of basins for that evaluation system
- random state `42`
- `10` restarts
- Dimensionality reduction:
  - for continuous feature views only, if the feature dimension is larger than 20, project to 20 principal components before clustering
- Metrics:
  - adjusted Rand index
  - normalized mutual information
  - silhouette score
  - K-means purity
  - linear classifier accuracy from cross-validated logistic regression
- Cross-validated logistic-regression accuracy uses up to `5` folds, but never more folds than the smallest class count in the evaluated basin partition.
- If a class has fewer than `2` examples, that class is dropped for the logistic-regression score only; clustering metrics are still computed on the full label set.

#### Why this tests the intended claim

- A model can separate basins in a continuous latent space without learning reusable sparse supports.
- By evaluating both binary support views and continuous latent views, this audit distinguishes:
  - support-based recovery
  - continuous-separation-only recovery

### 5.5 Direct Kuramoto support audit

#### Aim

Test whether the good relative forecasting result on rescued Kuramoto comes from stable basin-reused supports.

#### Exact question asked

On the rescued Kuramoto setting, do multiple trajectories from the same winding-number basin actually reuse the same support?

#### Main finding

- No. The trajectory-unique support rate is `1.0`.
- The Hamming geometry is almost flat, with between/within ratio only about `1.004-1.012`.
- Label-free clustering adjusted Rand index is approximately zero.

Interpretation:

- Good relative forecasting on rescued Kuramoto does **not** imply reusable basin-aligned supports.

### 5.6 Corrected competitive Lotka-Volterra representation follow-up

#### Aim

Re-run the representation analysis on the corrected multi-basin competitive Lotka-Volterra setting after invalidating the old one-basin setup.

#### Exact clean paper setting

- `10` species
- symmetric interactions
- interaction scale `0.70`
- uniform growth
- positivity clipping enabled
- survival threshold `1e-3`
- step size `0.01`

The follow-up also included selected higher-basin parity checkpoints, but the clean 4-basin paper-facing setting above is the main reference point.

#### Why this matters

- The old one-basin results cannot be used in the paper.
- This rerun asks whether the corrected multi-basin version strengthens the intended support-alignment narrative.

#### Main finding

- It does not.
- Best clean support-view median among the paper-facing roots:
  - adjusted Rand index `0.1203`
  - normalized mutual information `0.1802`
- All clean paper-facing roots keep negative cosine separation:
  - from `-0.0681` to `-0.0556`
- Support-view clustering collapses to discovered cluster counts `5`, `2`, and `1` across seeds instead of recovering a stable 4-basin structure.

### 5.7 Recurring-support local-linearity study

#### Aim

Test whether recurring supports correspond to locally linear latent dynamics that generalize better than:

- one global linear map
- a shuffled partition with the same group sizes

#### Exact protocol

Positive system:

- Multiwell strong-transition system
- 2-dimensional version
- selected step size `0.005`

Negative control system:

- rescued Kuramoto
- `16` oscillators
- ring coupling
- identical natural frequencies
- step size `0.00625`

Models:

- sparse MLP
- dense LISTA
- block-diagonal LISTA

Shared evaluation corpus:

- `256` trajectories
- trajectory length `256`
- evaluation seed `42`
- trajectory `i` is generated from reset seed `42 + i`

Grouping rule:

- compute the majority support for each trajectory using threshold `1e-3`
- keep only support groups with at least `5` trajectories in total
- after a global trajectory split:
  - at least `3` training trajectories per retained group
  - at least `1` test trajectory per retained group

Train/test split:

- split by trajectory, not by time step
- `80/20`
- split seed `42`
- shuffled-group baseline seed `42`

Local linear fit:

- center the training latent states
- project onto the top `32` principal directions of the centered training latent states
- fit a ridge-regression linear map with regularization `1e-4`
- evaluate:
  - one-step normalized root mean squared error
  - 20-step normalized root mean squared error

Baselines:

- a single global linear map fit on all retained training trajectories
- a shuffled-group baseline built from a random reassignment of the retained groups while preserving the train/test group sizes of every retained group

Exact error definition:

- `NRMSE(y, y_hat) = sqrt(mean((y - y_hat)^2)) / (sqrt(mean(y^2)) + 1e-9)`

Coverage gate used for deciding whether the result is broad enough for a strong mechanism claim:

- retained trajectory coverage `>= 0.60`

#### Why this tests the intended claim

- If recurring supports are mechanistically meaningful, a support-conditioned local map should beat both the global map and the shuffled partition.
- The held-out trajectory split prevents leakage across time steps from the same trajectory.
- The coverage gate prevents overclaiming based on a tiny retained subset.

### 5.8 Main mechanistic results

How to read the numeric summaries:

- `ARI` = adjusted Rand index from label-free K-means basin recovery. `1.0` is perfect; `0` is chance-like.
- `CosSep` = mean within-basin cosine similarity minus mean between-basin cosine similarity on trajectory-mean latent vectors. Positive is better.
- `Cons` = mean basin consistency, the fraction of trajectories in a basin that match that basin's modal support. Higher is better.
- `Uniq` = trajectory-unique support rate, i.e. distinct trajectory supports divided by trajectories. Lower is better; `1.0` means every trajectory gets its own support.

The numeric summary is split so every data cell stays numeric.

Label-free recovery summary:

| Case | ARI_lo | ARI_hi |
|---|---:|---:|
| Multiwell family | `0.794` | `0.991` |
| Duffing | `0.190` | `0.240` |
| Kuramoto | `0.000` | `0.001` |
| Corrected competitive Lotka-Volterra | `0.120` | `0.120` |

Support-reuse summary:

| Case | CosSep_lo | CosSep_hi | Cons_lo | Cons_hi | Uniq_lo | Uniq_hi |
|---|---:|---:|---:|---:|---:|---:|
| Multiwell family | `0.250` | `0.706` | `0.053` | `0.157` | `0.840` | `1.000` |
| Duffing | `-0.129` | `-0.084` | `0.050` | `0.079` | `0.880` | `0.920` |
| Kuramoto | `-0.307` | `-0.264` | `0.424` | `0.424` | `1.000` | `1.000` |
| Hopfield | `0.459` | `0.607` | `0.043` | `0.043` | `1.000` | `1.000` |
| Corrected competitive Lotka-Volterra | `-0.068` | `-0.056` | `0.107` | `0.107` | `1.000` | `1.000` |

Read by case:

- Multiwell is the only family with both high label-free recovery (`ARI = 0.794-0.991`) and positive support separation (`CosSep = 0.250-0.706`).
- Duffing is weak: label-free recovery stays low, cosine separation stays negative, and support reuse remains poor (`Cons = 0.050-0.079`).
- Kuramoto is a hard negative: `ARI` is effectively zero, `CosSep` is negative, and `Uniq = 1.0`. The raw `Cons = 0.424` is inflated by singleton basins in the random `100`-trajectory audit and should not be read as positive evidence.
- Hopfield shows continuous separation without reusable sparse supports: `CosSep > 0`, but `Cons = 0.043` and `Uniq = 1.0`.
- Corrected competitive Lotka-Volterra stays below the support-view gate: best clean support-view `ARI = 0.1203`, `Cons = 0.107`, `Uniq = 1.0`, and discovered support-view cluster counts collapse to `5/2/1` across seeds instead of a stable `4`.

Recurring-support local-linearity uses a different readout:

- `SeedRet` = fraction of seeds with at least one retained support group.
- `Cov` = retained trajectory coverage.
- `L20`, `G20`, `S20` = 20-step NRMSE of the local, global, and shuffled fits; lower is better.
- `W1` = fraction of seeds where the one-step local fit beats the matched global fit.
- A strong mechanism claim requires `Cov >= 0.60`.

| Model | SeedRet | Cov | L20 | G20 | S20 | W1 |
|---|---:|---:|---:|---:|---:|---:|
| Sparse MLP | `1.000` | `0.4141` | `0.0205` | `0.0356` | `0.1117` | `1.000` |
| Block-diagonal LISTA | `1.000` | `0.4336` | `0.0159` | `0.0380` | `0.0895` | `0.333` |
| Dense LISTA | `1.000` | `0.4414` | `0.0246` | `0.0390` | `0.0784` | `0.333` |

Local-linearity read:

- On `multiwell_strong_transition`, all three roots beat their own global and shuffled baselines at `20` steps, but every row stays below the `0.60` coverage gate.
- The best `20`-step fit is block-diagonal LISTA (`L20 = 0.0159`), but only the sparse MLP beats its own one-step global baseline on every seed (`W1 = 1.000`; both LISTA roots are `0.333`).
- Targeted smaller-step-size Kuramoto is a clean negative: `0/15` root-seed evaluations retained any support group, so the local-fit errors are undefined there.

### 5.9 Interpretation for the paper

- Basin-support alignment is **not** a universal property of these sparse Koopman models.
- The strongest positive mechanism story is the multiwell family.
- Duffing is weak and mixed.
- Kuramoto is a clean negative for support reuse.
- Hopfield shows continuous basin separation but not reusable sparse supports.
- Corrected competitive Lotka-Volterra is negative.
- The recurring-support local-linearity study is useful, but only for a bounded claim:
  - partial positive on multiwell
  - negative on Kuramoto
  - not uniquely favorable to LISTA at one step

## 6. What should and should not be claimed in the paper

| Claim | Supported? | Evidence-based wording |
|---|---|---|
| Dense LISTA is globally competitive under fair 200k conditions. | **Yes** | It wins more systems than the sparse MLP by `H1000` and becomes median-best at `H2000-H3000`. |
| Dense LISTA is the safer benchmark model. | **No** | The sparse MLP is still better by cross-system median through `H1000` and keeps the better `H3000` coverage. |
| Block-diagonal LISTA broadly improves on the sparse MLP. | **No** | Its strongest evidence is the targeted smaller-step-size Kuramoto rescue, not the full benchmark or the parity sweep. |
| Basin-support alignment is a general property of the learned latent space. | **No** | It is strong on multiwell, weak on Duffing, negative on Kuramoto, mixed on Hopfield, and negative on corrected competitive Lotka-Volterra. |
| The mechanism follow-up broadened the pro-LISTA story. | **No** | The mechanism follow-up narrowed the claim: partial on multiwell, negative on Kuramoto, negative on corrected competitive Lotka-Volterra. |

## 7. Appendix-only provenance

This section is included only so the handoff is complete. It should not become a main-text experiment section.

### 7.1 Dense LISTA comparator selection

Purpose:

- choose one fixed dense LISTA recipe before the final fair benchmark

Design:

1. Easy-system optimizer sweep
2. Coefficient holdout sweep
3. Validation on an 8-system subset
4. One fixed 29-system rerun with the promoted recipe

How to write it:

- one sentence in the main paper is enough:
  - the dense LISTA comparator was fixed in separate validation sweeps and then held fixed for the final fair benchmark
- if reviewers ask, move the detailed recipe-selection chain to the appendix

### 7.2 Historical 50k benchmark matrix

Purpose:

- historical context and step-size-selection provenance

How to write it:

- appendix only
- do not use it for main benchmark claims when the 200k reruns materially improve the same comparison
