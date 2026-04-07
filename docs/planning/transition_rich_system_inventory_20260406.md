# Transition-Rich System Inventory

Date: April 6, 2026

## Purpose

This file tracks deterministic, autonomous, native-plot `2D` toy systems for
the transition-rich basin-partitioning branch in
[docs/planning/transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md).

This is a design-screening document, not a calibration report. A system is
accepted into this inventory only if it is structurally consistent with the
branch criterion, but every accepted system still needs the usual fixed-grid
calibration pass before it can be treated as a benchmark.

Implementation-oriented elite sketches:

- [transition_rich_elite_system_sketches_20260406.md](/home/mila/l/lia/skae/docs/planning/transition_rich_elite_system_sketches_20260406.md)

Companion audit of the already implemented Claude catalog:

- [claude_catalog_audit_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_audit_20260407.md)

Companion shortlist figure:

- [transition_rich_shortlist_design_map.png](/home/mila/l/lia/skae/docs/figures/transition_rich_inventory_20260406/transition_rich_shortlist_design_map.png)
- [transition_rich_shortlist_design_map.svg](/home/mila/l/lia/skae/docs/figures/transition_rich_inventory_20260406/transition_rich_shortlist_design_map.svg)
- [transition_rich_shortlist_design_map.pdf](/home/mila/l/lia/skae/docs/figures/transition_rich_inventory_20260406/transition_rich_shortlist_design_map.pdf)
- [transition_rich_mechanism_atlas.png](/home/mila/l/lia/skae/docs/figures/transition_rich_inventory_20260406/transition_rich_mechanism_atlas.png)
- [transition_rich_mechanism_atlas.svg](/home/mila/l/lia/skae/docs/figures/transition_rich_inventory_20260406/transition_rich_mechanism_atlas.svg)
- [transition_rich_mechanism_atlas.pdf](/home/mila/l/lia/skae/docs/figures/transition_rich_inventory_20260406/transition_rich_mechanism_atlas.pdf)
- [transition_rich_elite_cards.png](/home/mila/l/lia/skae/docs/figures/transition_rich_inventory_20260406/transition_rich_elite_cards.png)
- [transition_rich_elite_cards.svg](/home/mila/l/lia/skae/docs/figures/transition_rich_inventory_20260406/transition_rich_elite_cards.svg)
- [transition_rich_elite_cards.pdf](/home/mila/l/lia/skae/docs/figures/transition_rich_inventory_20260406/transition_rich_elite_cards.pdf)

The design-map and atlas scores in those figures are qualitative design scores,
not empirical metrics.

## Screening rubric used for this inventory

Hard design gates:

- deterministic and autonomous
- native-plot `2D`
- intended endpoint basin count between `3` and `10`
- explicit procedural transition mechanism rather than plain nearest-well
  collapse
- geometry simple enough to explain in a paper figure
- compatible with the paper's label-light training story

Soft calibration gates to verify later:

- balanced endpoint occupancy
- meaningful but not overwhelming crossing fraction
- stable long-rollout endpoint labels
- no spurious center trap, limit cycle, or boundary chattering

## Inventory Status

- Procedural generation families used:
  smooth multiwell/corridor, gated piecewise-affine transfer, and slow-fast /
  relay hybrids
- Candidate concepts reviewed: `100`
- Accepted into the design inventory: `63`
- Rejected or held out: `37`
- Shortlisted for likely benchmark follow-up: `16`
- Elite shortlist for the first serious implementation pass: `8`

Current outstanding problem:

- We no longer lack candidate systems. The next bottleneck is selecting `10-20`
  systems worth actual implementation and calibration in the current restored
  worktree, and making sure the shortlist spans genuinely different mechanism
  classes rather than many near-duplicates of the same corridor idea.

## Accepted Systems

### A. Smooth Multiwell And Corridor Systems

| ID | System | Basins | Transition device | Why it is interesting | Main calibration risk |
| --- | --- | --- | --- | --- | --- |
| A1 | Triangle swirl with shared center corridor | `3` | Weak central swirl plus radial breathing around three wells | Minimal symmetric positive control for shared-corridor switching | One basin dominates if swirl is too weak |
| A2 | Four-well breathing ring | `4` | Annular breathing term opens an intermediate-radius passage ring | Clean square geometry with a single reusable transition band | Ring opens too much and effectively merges basins |
| A3 | Five-well annular shear | `5` | Thin annular angular shear pushes trajectories past local saddles | Simple ring scaffold with clear nontrivial handoffs | Shear under-tuned and crossing becomes too weak |
| A4 | Six outer wells with weak center lift-hub | `6` | Outward lift near a weak central hub, gradient return outside | Hub-vs-ring routing is easy to see and explain | Center becomes too sticky or too empty |
| A5 | Asymmetric five-well cut-channel oval | `5` | Two low-barrier cuts through an anisotropic ridge | Useful asymmetric benchmark with repeated channel reuse | One or two wells fall below occupancy gate |
| A6 | Seven-well diagonal ridge bridge | `7` | Ridge suppresses local neighbor hops but leaves diagonal bridge routes | Tests whether models track indirect basin changes | Effective basin count may collapse below the nominal count |
| A7 | Eight-well staggered double ring | `8` | Inner/outer ring crossing with staggered barrier gaps | Visually rich but still native-plot and structured | Inner and outer rings become imbalanced |
| A8 | Nine-well grid with diagonal saddle bands | `9` | Saddle bands and central cross favor diagonal movement over direct descent | Natural high-basin structured stress test | Basins may be too similar to tell a clean story |
| A9 | Three-basin cycle lane triangle | `3` | One explicit clockwise lane plus return shells | Makes ordered repeated handoffs explicit | Cycling can overpower stable endpoints |
| A10 | Three-basin hub-only lane | `3` | All cross-basin travel is forced through a central hub corridor | Very clean shared-passage geometry | Hub becomes an unintended extra basin |
| A11 | Four-basin square X-lanes | `4` | Diagonal `X` transfer lanes bypass nearest-neighbor routes | Strong non-nearest transfer benchmark with simple geometry | Center region becomes label-ambiguous |
| A12 | Five-basin asymmetric ring transfer | `5` | Clockwise and counterclockwise lanes have different strength | Good directed-transfer benchmark without leaving autonomous `2D` | Symmetry breaking may create occupancy skew |
| A13 | Six-basin spoke-to-transport ring | `6` | Radial spokes feed a weak transport ring before capture | Tests spoke-vs-ring route choice | Transport ring can become too dominant |
| A14 | Seven-basin selective bridge hub | `7` | Only two sector-dependent bridges connect the ring to the hub | Sharp test of selective shared routes | Bridge use may be too rare on the calibration grid |
| A15 | Eight-basin ring with chord skips | `8` | Short chord corridors skip over immediate neighbors | Useful nonlocal jump benchmark | Chords can partially merge nearby basins |
| A16 | Nine-basin switchback lattice | `9` | Alternating switchback corridors force zig-zag travel | Good high-basin path-geometry stress test | Switchbacks may become too trivial or too busy |
| A17 | Ten-basin two-tier lattice | `10` | Four inner and six outer wells connected by narrow tier bridges | Maximal basin-count candidate that is still structured | The figure may be too busy for the main text |

### B. Gated And Piecewise-Linear Transfer Systems

| ID | System | Basins | Transition device | Why it is interesting | Main calibration risk |
| --- | --- | --- | --- | --- | --- |
| B1 | Triangle central gate and sectors | `3` | Stable local-linear cores plus one shared gate disk and angular exits | Closest clean follow-up to the existing gated-local-linear story | Central gate may be too weak to create enough handoffs |
| B2 | Four-way crossroads | `4` | Orthogonal corridors cross at the origin and reroute states | Very legible bottleneck benchmark | The crossroads can become an attractor instead of a passage |
| B3 | Five-basin directed ring relay | `5` | Each source shell has a strong clockwise lane and weak return lane | Controlled multi-hop relay benchmark | Directed drift may overwhelm occupancy balance |
| B4 | Six-basin star transfer network | `6` | Cores feed a radial star network with destination branches | Strong explicit-chart-transfer system with many routes | Excess symmetry may make basins too interchangeable |
| B5 | Seven-basin center hub with gated exits | `7` | Six outer basins route through a central staging hub with gated exits | Good hub-and-spoke transfer benchmark | Center hub may capture too much mass |
| B6 | Four-basin nested triad with staging center | `4` | Outer basins flow inward before gated release to a destination basin | Separates source, staging, and endpoint roles | Staging center may become terminal |
| B7 | Six-basin bipartite bridge | `6` | Left-half and right-half basin sets only connect through narrow bridges | Strong modular-transfer benchmark | Bridges may be too narrow to yield enough crossings |
| B8 | Four-basin arc bridge | `4` | Curved bridge segments intentionally bend away from shortest paths | Clean test of curved rather than straight transfer lanes | Curvature can introduce numerical stiffness |
| B9 | Three-basin split shell | `3` | Each shell has one return sector and two destination sectors | Compact label-light partition benchmark | Return sectors may dominate and kill transitions |
| B10 | Six-basin conic pair bridge | `6` | Three paired wedges each share a local hub before global switching | Good structured-transfer benchmark with modularity | Pair structure may be too strong and suppress global switching |
| B11 | Five-basin double hub | `5` | Left and right half-planes route through different hubs before release | Useful test of multi-stage chart choice | One hub may dominate and the other may not activate |
| B12 | Nine-basin nearest-neighbor lattice | `9` | Only four-neighbor lanes are allowed; diagonals need two transfers | High-basin percolation benchmark | Too many basins may dilute occupancy |
| B13 | Three-basin sector lift | `3` | Shared outer lift region feeds basin-specific inner gates | Isolates one reusable transition region very clearly | Shared lift region may swallow too much phase space |

### C. Slow-Fast, Relay, And Hybrid Systems

| ID | System | Basins | Transition device | Why it is interesting | Main calibration risk |
| --- | --- | --- | --- | --- | --- |
| C1 | Four-well diagonal corridor cuts | `4` | Smooth diagonal strip cuts lower the barrier through a four-well potential | Smooth counterpart to the explicit lane systems | Corridors may open too much and merge basins |
| C2 | Pinched triangle canard | `3` | Slow-fast folds create delayed jumps between three settling branches | Mechanistically distinctive and easy to explain with nullclines | Narrow parameter window may make it fragile |
| C3 | Sector relay-4 | `4` | Four local-linear sectors plus one transition strip | Very clean piecewise-affine chart-switch benchmark | Over-sharp switching can chatter on boundaries |
| C4 | Sector relay-6 | `6` | Six sectors with two special transfer sectors | Higher-complexity piecewise chart-switch benchmark | Thin sectors may hurt occupancy balance |
| C5 | Slow-fast three-branch oblique manifold | `3` | Three stable slow branches with fold-triggered jumps | Strong test of branch tracking without explicit lanes | One branch may dominate the endpoint mass |
| C6 | Slow-fast four-branch double pinch | `4` | Two pinches create several jump routes before final capture | Good benchmark for multiple slow-fast transfer opportunities | Periodic orbit risk if the folds are mistuned |
| C7 | Heteroclinic lane-3 | `3` | Smoothed heteroclinic lane network with off-lane sinks | Very clear channel-then-capture geometry | Time spent near saddles may be too long |
| C8 | Heteroclinic lane-5 | `5` | Five saddle neighborhoods and soft channels with route choices | Richer graph-based transition benchmark | Endpoint occupancy can skew badly |
| C9 | Rotating barrier-4 | `4` | Barrier orientation changes with radius and rotates the preferred exit direction | Strong moving-separatrix stress test | Quasi-circling instead of clean endpoints |
| C10 | Mirror gate-3 | `3` | Direction-selective slab gates open only in narrow aligned windows | Good directionality benchmark with simple symmetry | Gates may be too narrow for enough crossings |
| C11 | Mirror gate-5 | `5` | Five angular gate windows on a ring of wells | Tests selective exits with more basins but still clean geometry | Privileged gates may cause occupancy imbalance |
| C12 | Deadzone relay-3 | `3` | Central deadzone forces regime changes by quadrant | Strong relay-style control system with explicit boundaries | Chattering near deadzone edges |
| C13 | Deadzone relay-4 | `4` | Two deadzones plus hysteresis strips create entry/exit asymmetry | Good hysteresis benchmark for chart reuse | Sliding and stiffness near switching lines |
| C14 | Spiral funnel-5 | `5` | Weak funnel temporarily captures states before re-releasing them to spiral sinks | Good re-anchoring benchmark with clear geometry | Funnel can turn into the endpoint attractor |
| C15 | Funnel-tunnel-6 | `6` | Single shared tunnel with several exits into final wells | Strong shared-passage stress test | Tunnel can dominate and erase basin separation |

### D. Higher-Novelty Smooth Geometry Systems

These were accepted only if they felt genuinely different from the first-wave
ring/hub/lane templates.

| ID | System | Basins | Transition device | Why it is interesting | Main calibration risk |
| --- | --- | --- | --- | --- | --- |
| D1 | Lens-warp triad | `3` | Smooth coordinate-lens warp bends trajectories through a compressed transition strip | Clean test where dynamical geometry differs from Euclidean geometry | Lens can create a quasi-trapping strip |
| D2 | Curvature-saddle quartet | `4` | Passage opens preferentially at high-curvature ridge segments | More novel than a plain corridor because the transition route depends on barrier shape | Curvature-selected passages may be too sparse |
| D3 | Braided-metric pentad | `5` | Mobility tensor twists with angle and radius, yielding continuous anisotropic route selection | Continuous transport bias without explicit gates or lanes | The anisotropy may be too weak to matter |
| D4 | Oblique-trench hex | `6` | One slanted low-friction trench creates a shared slide region before capture | Gives a visually distinctive transport manifold that is not a hub | The trench may become an unintended attractor corridor |
| D5 | Anisotropic-ridge quartet | `4` | Tangential mobility along ridges encourages sliding before descent | Tests directional resistance instead of only basin depth | Ridge wandering may become too slow |
| D6 | Skew-barrier sextet | `6` | A radius-dependent oblique barrier biases travel direction globally | Nonlocal continuous transport bias instead of local gates | Near-separatrix sticking may dominate |

### E. Higher-Novelty Slow-Fast And Rotating-Separatrix Systems

| ID | System | Basins | Transition device | Why it is interesting | Main calibration risk |
| --- | --- | --- | --- | --- | --- |
| E1 | Folded tri canard | `3` | One shared fold throat creates delayed jumps into three final basins | Strong causal slow-fast story with compact geometry | Parameter window may be narrow |
| E2 | Twin pinch bowtie | `4` | Two sequential pinch regions create a two-stage routing web | Much richer than a single bottleneck without adding many basins | One pinch can dominate and collapse the structure |
| E3 | Rotating separatrix quartet | `4` | Barrier principal axis twists with radius, so the separatrix itself rotates with state | A dynamic boundary is more novel than a fixed corridor | Trapping or trivial collapse if twist is mis-scaled |
| E4 | Twist-gated quad | `4` | An annular twist gate opens only when radius and phase align | Delayed, state-dependent switching rather than static geometry | The gate can be too narrow for enough crossings |
| E5 | Lens canard hexad | `6` | Coupled lens-like slow-fast regions delay then release trajectories through off-axis throats | Combines canard delay with multi-exit selection | The lens region may become terminal |
| E6 | Sheet-to-sink hexad | `6` | A folded quasi-invariant sheet transports states before they drop into six sinks | Strong “folded sheet” picture that is visually distinct from lane systems | The sheet may act like an attractor sheet |

### F. Higher-Novelty Basin-Graph And Routing-Topology Systems

| ID | System | Basins | Transition device | Why it is interesting | Main calibration risk |
| --- | --- | --- | --- | --- | --- |
| F1 | Triad fork graph3 | `3` | One shared saddle region feeds two staged exits before final capture | The object of interest is the routing graph, not the well geometry | One exit may dominate and trivialize the graph |
| F2 | Braided diamond4 | `4` | Two saddles braid their manifolds through the interior before capture | Explicit multi-step route choice without a hub | The braid may become too thin and numerically brittle |
| F3 | Fan saddle6 | `6` | One upstream saddle fans into branches that split again before capture | Good hierarchical partition-reuse benchmark | Outer branches may become underoccupied |
| F4 | Horseshoe leaf6 | `6` | Mirrored horseshoe routing structures release trajectories into leaf sinks | Distinct from hub-and-lane systems while still geometrically legible | Interior strip may become weakly attracting |
| F5 | Petal cycle5 | `5` | A repelling petal-shaped skeleton routes states through a shared interior structure | Controlled heteroclinic-style mechanism without turning the center into the endpoint | Central skeleton may steal mass |
| F6 | Arc DAG4 | `4` | Curved separatrix arcs encode a directed acyclic transition graph | Very clean causal graph story with no return loops | The DAG may collapse into near-direct capture |

## Rejected Or Held-Out Systems

| System idea | Why it was rejected or held out |
| --- | --- |
| Plain five-well gradient descent | Usually collapses directly to the nearest well and is not transition-rich enough |
| Periodically forced Duffing-style switching | Non-autonomous in native `2D`, so it is the wrong object for this branch |
| Two-basin double well | Fails the `3-10` endpoint-basin gate |
| Stochastic basin hopping | Violates determinism and reproducibility |
| Dense `12+` basin mesh | Exceeds the intended basin-count range and is too crowded for clean interpretation |
| Classical double-scroll chaotic flow | Does not give stable endpoint-basin labels in the sense needed here |
| Global monobasin hub system | Trivial endpoint structure even if trajectories look visually busy |
| Over-sharp relay with hard sign switches only | Too prone to chattering and boundary artefacts |
| Center-trap hub variant | The shared hub becomes the endpoint attractor instead of a transition region |
| Undamped spiral transport field | Too likely to create long-lived quasi-cycles instead of clean settling |
| Warped-checkerboard septet | Interesting geometry, but too close to turning the figure into a visual puzzle rather than a clean benchmark |
| Saddle-lens octet | Likely odd/even occupancy imbalance with limited narrative gain |
| Curvilinear-bifurcation quintet | Dynamical adjacency departs too far from visible geometry to stay senior-coauthor-friendly |
| Double-focus septet | Reusable transient structure is appealing, but accidental-attractor risk is too high |
| Curvature-budget decet | Organic high-basin idea, but too busy for the cleanest paper figures |
| Barrier-twist quintet | Useful idea, but too close to the stronger rotating-separatrix quartet |
| Skew-prism pentad | Attractive Morse-Smale geometry, but likely too thin and brittle on a fixed screening grid |
| Nullcline-fan septet | One-fold-to-many-branches story is interesting, but occupancy imbalance risk is too severe |
| Rotating-pinch triad | Compact and elegant, but too close to the rotating-separatrix quartet and folded tri canard |
| Curvature-flip quintet | Rich re-anchoring story, but likely to degrade into sluggish drift instead of clean switching |
| Folded-diamond septet | Good geometry, but too much risk of a central transient trap |
| Zigzag Morse5 | Nice staged graph path idea, but the sequential chain may still behave like ordinary direct descent |
| Cusp ladder7 | Ladder structure is appealing, but fold stability across all rungs looks brittle |
| Twist separatrix8 | Topologically rich, but likely too visually complex for the main story |
| Tree branch9 | Strong hierarchical graph idea, but leaf occupancy balance is likely poor |
| Triple cascade3 | Good timing benchmark, but too close to the cleaner triad-fork graph system |
| Split skein6 | The woven-skein idea is novel, but calibration fragility is likely too high |
| Time-warped multiwell / forced Morse ribbon / periodic canard lattice | Non-autonomous, so out of scope for this branch |
| Noisy geodesic escape / stochastic heteroclinic lace | Stochastic, so out of scope for this branch |
| Twelve-basin hypergrid / dendritic 14-basin lattice / 12plus wedge mesh | Too crowded for the intended `3-10` clean-benchmark window |
| Chaotic saddle endpoint / chaotic flow claim | Does not match the stable endpoint-basin requirement |

## Proposed Benchmark Shortlist

This is a revised `16`-system shortlist after the higher-novelty second pass.
The aim is to keep enough diversity to build a real benchmark family while
avoiding many near-duplicates of the same corridor template.

| Priority | System | Basins | Role in a paper benchmark | Why shortlist it now |
| --- | --- | --- | --- | --- |
| High | Triangle central gate and sectors | `3` | Clean chart-switching positive | Direct follow-up to the current gated-local-linear story with clearer gate geometry |
| High | Four-way crossroads | `4` | Explicit bottleneck transfer benchmark | Very easy to draw, explain, and diagnose |
| High | Sector relay-4 | `4` | Piecewise-affine positive control | The region chart is explicit and mechanistically defensible |
| High | Rotating barrier-4 | `4` | Moving-separatrix stress test | Useful if we want a cleaner localization challenge than the current transfer toy |
| High | Lens-warp triad | `3` | Warped-geometry positive | Clean way to separate visible layout from transport geometry |
| High | Braided diamond4 | `4` | Explicit route-choice topology benchmark | Gives a real interior braid without using a hub |
| High | Twin pinch bowtie | `4` | Two-stage slow-fast routing benchmark | Stronger transition topology than a single bottleneck |
| High | Arc DAG4 | `4` | Directed acyclic transition-graph benchmark | Very clean causal graph story for coauthor-facing figures |
| Medium | Folded tri canard | `3` | Compact slow-fast benchmark | Preserves the qualitative canard mechanism with fewer moving parts |
| Medium | Heteroclinic lane-3 | `3` | Channel-network benchmark | Captures route choice without requiring many basins |
| Medium | Anisotropic-ridge quartet | `4` | Directional-resistance benchmark | Useful because the transition story is “slide along a barrier,” not “go through a gate” |
| Medium | Sheet-to-sink hexad | `6` | Folded-sheet transport benchmark | Distinctive geometry with a clear narrative if it calibrates |
| Medium | Triad fork graph3 | `3` | Minimal basin-graph benchmark | The routing graph is the main object, not the well geometry |
| Medium | Fan saddle6 | `6` | Hierarchical branching benchmark | Good test of multi-stage partition reuse |
| Medium | Six-basin bipartite bridge | `6` | Modular transfer benchmark | Good for partition reuse at the module level rather than only local hops |
| Medium | Oblique-trench hex | `6` | Shared-manifold transport benchmark | Adds a clean trench/sliding-manifold mechanism missing from the current suite |

## Elite Shortlist

This is the stricter `8`-system subset that currently looks most worth actual
implementation if we want a smaller, more coherent benchmark packet instead of
just a larger candidate pool.

| System | Why it made the elite subset |
| --- | --- |
| Triangle central gate and sectors | Best clean follow-up to the current chart-switching positive-control story |
| Four-way crossroads | Cleanest bottleneck-transfer geometry for senior-coauthor-facing figures |
| Sector relay-4 | Most defensible explicit local-chart positive control |
| Lens-warp triad | Most legible warped-geometry system where dynamical and visible geometry differ |
| Braided diamond4 | Strongest explicit route-choice topology system without hub dependence |
| Twin pinch bowtie | Best two-stage slow-fast routing candidate |
| Rotating barrier-4 | Strongest moving-separatrix / dynamic-boundary stress test |
| Arc DAG4 | Cleanest causal transition-graph benchmark in the whole inventory |

Current recommendation:

- If we only implement `4`, take one each from:
  clean control, bottleneck/graph, warped geometry, and moving-boundary or
  slow-fast.
- If we implement `8`, use the elite subset above as the first complete pass.

## Systems Not Shortlisted For First Implementation

These are still worth keeping in reserve, but they do not look like first-wave
implementation priorities:

- the `9-10` basin lattice and tree variants, because they are likely too busy
  for the cleanest first paper figures
- the more heavily directed ring and chord systems, because they are easier to
  mistune into occupancy imbalance without adding much new mechanism value
- the deadzone relay systems, because boundary artefacts are likely to dominate
  the debugging budget
- the more fragile higher-novelty variants, such as woven skeins, large twist
  separatrix systems, and multi-rung cusp ladders, because their main risk is
  calibration brittleness rather than conceptual weakness

## Suggested next implementation order

1. Add one clean positive control:
   triangle central gate and sectors, or sector relay-4.
2. Add one explicit bottleneck / graph benchmark:
   four-way crossroads, braided diamond4, or arc DAG4.
3. Add one smooth-geometry novelty:
   lens-warp triad, anisotropic-ridge quartet, or oblique-trench hex.
4. Add one slow-fast / moving-boundary benchmark:
   folded tri canard, twin pinch bowtie, or rotating barrier-4.
5. Add one harder multi-stage stress test only after the above calibrate:
   sheet-to-sink hexad, fan saddle6, or six-basin bipartite bridge.
