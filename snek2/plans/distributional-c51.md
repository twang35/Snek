# Distributional RL — C51 (categorical DQN)

**Status:** proposed 2026-08-15, **phases 0-2 done and phase 3 armed the same day. Phase 4 is
deliberately not started** — the real batch is to be shaped with the user.

- **Phase 0** picked the support — `v_min = -5`, `v_max = 120`, `num_atoms = 51`, spacing exactly 2.5 —
  while **falsifying the premise the support section rested on**
  ([results](#phase-0-results--the-support-measured-2026-08-15)). The control moved from `b24` to `b25`
  for the head-size reason [below](#the-phase-34-launch-line).
- **Phase 1** shipped as `categorical_agent.py` plus edits to `under_the_hood.py`, `policy_arch.py`,
  `snek2.py`, `eval_agent.py`, `watch.py` and four diagnostics; `training.py` needed **no change**, as
  designed. Suite **24 modules / 643 tests / 0 failed**, and all **18** mutants in the table below fail a
  test. **The code is uncommitted pending review** (CLAUDE.md's code rule).
- **Phase 2** ran end to end: a 3000-step `smoke` arm trains, writes `arch.json` with `algo=c51` and its
  support, resumes cleanly, refuses a resume under a changed `V_MAX` or a changed `ALGO`, is watchable
  through `watch.py`, and is measurable by `eval_checkpoints.py`. Throughput at `fc 200,100,100` is
  **within ~10% of ddqn** (9000 steps: c51 146/151 s, ddqn 165/166 s — c51 *faster*, which is inside the
  noise of episode-length variation with a close-out competing for cores, so read it as "no meaningful
  slowdown" rather than as a speedup).
- **Phase 3** is armed but waiting: `hyperparamTuning/launch_c51_pilot.sh` launches the four pilot arms
  when b30's close-out exits, per this file's own host note.

A throwaway feasibility probe was also run against this repo's own environment and specs, and its
results are in [what the probe established](#what-the-probe-already-established) — they remove most of
the implementation risk and change two design choices from what a naive port would have done.

**One line:** replace the scalar `DdqnAgent` with a categorical (C51) agent that predicts a
*distribution* over the return on a fixed atom grid and trains it by cross-entropy, behind a
`SNEK_ALGO=c51` switch, leaving the environment, the rewards, the collector, the shield and the
epsilon schedule untouched.

Reading order for context: the record config this would be measured against is in
[`../hyperparamTuning/runs.md`](../hyperparamTuning/runs.md#record-status), and the reason a
value-representation change is on the table at all is that the post-peak collapse has no established
mechanism — the obvious candidate was ruled out in
[`findings.md`](../hyperparamTuning/findings.md#-falsified-2026-08-14-there-is-no-plasticity-loss--the-collapsed-networks-fit-a-new-target-better-than-their-own-peak).

## Why this, and what the honest prior is

C51's headline result (Bellemare et al. 2017) is a large median improvement on Atari at 200M frames
with pixel inputs and a conv net. **This project is none of those things** — a 30-value hand-built
observation, an MLP of a few tens of thousands of parameters, and useful learning finished by ~1.5M
steps. Transferring the *size* of that result here is not a reasonable expectation, and the plan
should not be sold on it.

The reason to run it anyway is narrower and fits what this project has actually measured:

**1. The problem here is instability, not ceiling.** Nine network shapes never raised the ceiling
(batch 20), and the selected column has been flat at 93-99% since batch 11. What separates a 94.2%
batch from a 97.6% one is whether a strong region *holds*. The most robust empirical claim in the
distributional literature is not higher final score, it is **reduced value-estimate variance and
gentler degradation** — the failure mode this folder keeps documenting as post-peak collapse. And that
collapse now has *no* mechanism: plasticity loss was ruled out directly (a collapsed net fits a new
target **better** than its own peak did), so the value estimate itself is one of the few candidates
left untried.

**2. The returns here are genuinely bimodal exactly where the game is decided.** From a packed
endgame state the return is approximately "win: the terminal +100 discounted" or "die: a small
negative". A scalar Q must regress the *mean* of those two, which is a value no outcome ever pays.
C51 represents both modes. This is the strongest a-priori argument and it is specific to this reward
structure, not a generic appeal to the paper.

**3. It is cheap, because `tf_agents` ships most of it.** `CategoricalDqnAgent`,
`CategoricalQNetwork`, `CategoricalQPolicy` and the distribution projection are all present in the
pinned 0.18.0. The work is a ~90-line subclass, a network builder, four `arch.json` fields, and one
agent-construction switch.

**4. Its loss has bounded gradients by construction**, which is the property `GRADIENT_CLIPPING=10`
was adopted to buy and failed to deliver — and the stated motivation for that arm was *exactly* "the
large terminal reward produces occasional huge gradients"
([falsified](../hyperparamTuning/findings.md), 1 of 3 seeds against 3 of 3). A cross-entropy over a
fixed support cannot produce a large gradient from a large target, because the target is a probability
vector. Clipping tried to fix the symptom outside the loss; this fixes it inside.

**5. It is orthogonal to the whole backlog, and the backlog is running dry at the top.** Every open
item is a reward term, an architecture width or a schedule. `CHASE_SAFE_SHAPING` has now returned
**null twice** — b27 (pooled 85.2 vs 87.9) and b30 (best-30 92.9 vs 93.6) — and batch 20 already
established that architecture never raises the ceiling. Nothing else queued changes the *loss*, so a
result here is not confounded with anything behind it, and nothing above it in the backlog is
currently promising.

**6. There is no prior art in this folder at all.** `c51`, `distributional`, `rainbow` and `quantile`
appear nowhere in the docs or code, and the agent class is hardcoded at `snek2.py:320` with no
`SNEK_*` knob — so unlike every batch since b1, this arm cannot be launched by environment variable
and needs the code change first. That is the reason it has never been tried, not a judgement that it
would not work.

**Falsifiable prior, pre-registered:** ~40% that C51 is a null on `best_perfect30` at a matched
horizon; ~35% that it is a modest improvement in *drawdown* (peak-to-later decline) with a null on
peak; ~15% a real improvement in both; ~10% that it is clearly worse, most likely through the
learning-rate/loss-scale interaction in [risks](#risks-in-the-order-they-are-likely-to-bite).

## What actually changes, mathematically

| | now (`DdqnAgent`) | with C51 |
|---|---|---|
| network head | 3 units, one Q per action | 3 × `num_atoms` logits |
| prediction | `Q(s,a)` scalar | `p(s,a)` over a fixed support `z_1..z_N` |
| target | `r + γ·d·max_a' Q'(s',a')` | project `r + γ·d·z` onto the support, weighted by `p'(s',a*)` |
| loss | element-wise Huber on the TD error | cross-entropy against the projected target |
| greedy action | `argmax_a Q(s,a)` | `argmax_a Σ_i z_i·p_i(s,a)` |
| PER priority | `\|td_error\|` | KL(target ‖ prediction) — see [priority signal](#the-priority-signal-kl-not-cross-entropy) |

Everything else is untouched: the observation, all rewards, `discount` coming from the environment's
time step (0.99 mid, **0.0 on `LAST`**, with agent `gamma=1.0`), the forking collector, the
exploration shield, the epsilon schedule, perfect-game counting, checkpointing, charts, and the eval
pipeline's episode logic.

## What the probe already established

Run 2026-08-15 against `SnakeEnvironment` and this repo's own `ShieldedEpsilonGreedyPolicy` and
`TrajectoryPrioritizedReplayBuffer` (throwaway script, scratchpad, nothing committed). All six are
verified facts, not expectations:

| question | answer |
|---|---|
| does `CategoricalQNetwork` build on this env's specs and match `build_q_net`'s initializers? | **yes** — `encoding_network`'s default kernel init is exactly `VarianceScaling(2.0, fan_in, truncated_normal)`, the same as `dense_layer()`. Head is `RandomUniform(±0.03)` with bias `-0.2` vs our `0.0`, which is a constant shift that cancels in the per-action softmax |
| does the exploration shield work over the categorical greedy policy? | **yes** — `agent.policy.info_spec == ()`, which is what `ShieldedEpsilonGreedyPolicy` requires, and it wrapped and ran |
| is the replay-buffer spec unchanged? | **yes** — `collect_data_spec` is identical, so a c51 arm can even warm-start from a ddqn arm's saved buffer |
| **does upstream's `_loss` honour the IS weights we pass?** | **no — it silently drops them.** `weights` is accepted and never used; skewing them 32× moved our loss and left upstream's bit-identical |
| **does upstream give PER a priority signal?** | **no** — it returns `DqnLossInfo(td_loss=(), td_error=())` with a `TODO(b/127318640)` in the source. `training.py`'s `signal.numpy()` would raise on an empty tuple |
| is a from-scratch reimplementation of the loss correct? | **yes, exactly** — our unweighted single-selection total equals upstream's to 6 decimals (4.706757 both) on the same batch |

Two things the probe changed in the design:

- **The loss must be overridden, not merely subclassed for convenience.** Two upstream defects
  (dropped IS weights, no priority signal) sit in the one method, and both are load-bearing here:
  PER is on in every arm, and `SNEK_IS_WEIGHTS` is a live knob. A vendored copy is not needed — a
  ~35-line reimplementation reproduces upstream's number exactly, which is a better test than
  trusting a copy.
- **The double-vs-single mutation is invisible at initialisation.** `agent.initialize()` syncs the
  target net to the online net, so double and single atom selection returned *identical* losses in
  the probe (4.706757 both). A test for double selection that does not first desync the target
  network passes for the wrong reason and its mutant survives.

Also measured: 200 `agent.train` steps at batch 128 in **1.33 s (150 train-steps/s)** with
`num_atoms=111` on `fc (50,100,50)`, so the loss step is not near the bottleneck (the desktop's arms
run the *whole* loop at ~92 steps/s). Parameter count on that shape: **28,683 vs 11,853** for ddqn.

## Phase 0 results — the support, measured (2026-08-15)

C51 needs a fixed `[v_min, v_max]` grid and `project_distribution` clamps anything outside it, so the
grid is a modelling choice made before training. Measured with
[`perDiagnostics/return_distribution.py`](../hyperparamTuning/perDiagnostics/return_distribution.py)
over **three checkpoints, 60 greedy episodes, 116,454 states**: the record `b24d` @1342k, and `b30e`
(the control's `200,100,100` trunk) at its peak @681k and mid-climb @200k.

| γ=0.9975 | min | p5 | p50 | p95 | p99.9 | max |
|---|---|---|---|---|---|---|
| `b24d` @1342k (98%/500) | -0.50 | 4.56 | 27.99 | 93.09 | 103.37 | **104.37** |
| `b30e` @681k (peak) | -0.50 | 8.52 | 27.12 | 90.44 | 103.08 | **104.38** |
| `b30e` @200k (mid-climb) | -0.50 | -0.34 | 11.47 | 40.77 | 100.01 | 102.23 |

**‡ This falsifies the premise the section it replaces was built on.** That version said "almost every
other state's return sits in roughly [-5, 25]", so a grid covering the win would waste most of its
atoms, and it offered a nonlinear support as the principled fix. Both are wrong, and the error was
reasoning at **γ=0.99 when the config runs γ=0.9975**. An effective horizon of 400 steps rather than
100 means the terminal +100 is visible from most of a won game, so the returns are spread broadly
across the whole range instead of piling up near zero: **60% of a champion's states are above 25, 24%
above 50, 12% above 75.** There is no crowding problem, the nonlinear support is closed unbuilt, and
the atom count is a free choice rather than a constraint.

**Two numbers the analytic reasoning got wrong, both in the direction that would have clipped:**

- **The max is 104.38, not ~101.** A state a few steps from the win still collects food on the way, so
  the return is `100·γ^k` *plus* those meals. The guard originally written as
  `v_max ≥ PERFECT_GAME_REWARD + FOOD_REWARD` (101) would have permitted a grid that clips the top of
  the distribution — the exact failure it was added to prevent.
- **The observed min is -0.50, not -5.** `STARVE_REWARD`, not `DEATH_REWARD`: there were **zero
  collisions in all 60 greedy episodes** across all three checkpoints, which independently reproduces
  the folder's finding that the modal failure is failing to reach reachable food. `-5` still has to be
  covered, because exploration collides during training even though greedy play does not, and the
  analytic minimum is exactly `DEATH_REWARD` (a one-step death; any earlier death is discounted toward
  zero).

### The support, settled after two revisions (2026-08-15)

**Chosen: `v_min = -5`, `v_max = 120`, `num_atoms = 51` — spacing exactly 2.5.** Settled 2026-08-15
after two revisions: the first draft used the *measured* range (`-6`, `110`, 51 atoms), the second the
γ-free theoretical bound (`-5`, `195`, 81 atoms), and this is a deliberate step back from the second.
It keeps everything that mattered — the same 2.5 spacing, the same 35 atoms across the bulk, the
correct `v_min` — and trades clip-proofness above 120 for 9,090 fewer parameters. See
[why 120 and not 194](#why-120-rather-than-the-theoretical-194).

**The theoretical maximum is exactly 194.** The bound comes
from the reward function: a perfect game pays `FOOD_REWARD` for 94 foods and then
`PERFECT_GAME_REWARD` for the 95th — **not both**, because `Snake.py:517` *overwrites*
`reward = FOOD_REWARD` with the win — so the undiscounted maximum return is `94·1 + 100 = 194`, and it
is attained from the opening state in the limiting case where every food spawns one step from the head.

| max return from `s0` | m=1 | m=2 | m=3 | m=4 | m=10 | m=19 |
|---|---|---|---|---|---|---|
| γ=1 (any discount, upper bound) | **194.00** | 194.00 | 194.00 | 194.00 | 194.00 | 194.00 |
| γ=0.9975 (this config) | 162.49 | 136.94 | 116.17 | 99.23 | 44.97 | 21.39 |
| γ=0.99 | 99.00 | 56.62 | 36.45 | 26.02 | 9.46 | 4.75 |

`m` is steps per meal; the record averages ~19, which is why its *measured* max of 104.4 comes from
near-win states rather than from `s0`. **The reason to prefer a derived bound over the measured one is
that it is γ-independent**: `SNEK_DISCOUNT` is a live knob — 0.99, 0.995 and 0.9975 have all shipped —
and a grid sized to 0.9975's observed distribution silently becomes wrong for an arm that changes it.

#### Why 120 rather than the theoretical 194

At fixed spacing 2.5 on the `b25` trunk, `v_max` buys clip-proofness and costs only head parameters —
resolution is identical in every row, because spacing is what is being held fixed:

| grid | N | total params | vs ddqn control | atoms across the bulk | clipping needs |
|---|---|---|---|---|---|
| `[-5, 110]` (measured max + 5%) | 47 | 50,641 | 1.38× | 35 | 2.98 steps/meal |
| **`[-5, 120]`** (chosen) | **51** | **51,853** | **1.41×** | **35** | **2.80 steps/meal** |
| `[-5, 162.5]` (γ=0.9975 bound) | 68 | 57,004 | 1.55× | 35 | never at γ≤0.9975 |
| `[-5, 195]` (γ-free bound) | 81 | 60,943 | 1.66× | 35 | never |

**120 is not clip-proof, and the honest statement of the risk is that last column.** A return above 120
requires a policy averaging under **2.80 steps per meal** — about **7× faster than the record's ~19** —
against a measured maximum of 104.38 across three checkpoints including the 98%/500 record, so the
headroom is 15%. The value ceiling is also self-limiting rather than distorting: a bootstrap target is
`r + γ·d·z_i`, so with the support capped at 120 the network cannot push its own estimates past it,
and everything below is untouched.

What 120 keeps, against the wider grids: the same 2.5 spacing, the same ~35 atoms across the p5-p95
bulk, and `-5`/`0`/`100`/`120` all landing on atoms. What it gives up: the 120-194 band, which no
measured policy comes within 15 units of. It also happens to halve the initial-optimism confound
below — the grid midpoint is 57.5 rather than 95.0.

**`v_min` is `-5` exactly, and the code proves nothing lower is reachable.** `Snake.step` *assigns*
each outcome reward rather than accumulating (`Snake.py:513-559`), so death, starve, food and the win
are mutually exclusive — a death pays exactly `DEATH_REWARD`, never `DEATH_REWARD + STARVE_REWARD`, so
the original guard's `-5.5` bound described a reward the game cannot pay. Two *additive* terms follow:
`-FOOD_DISTANCE_REWARD`, which is skipped on any step that ends the episode, and the chase-safe
shaping, which does fire on a terminal step as `-c·Φ(s)`. With both at 0 — which is this arm's config —
**the minimum possible return is exactly -5.0**, and a value sitting exactly on `v_min` is not clipped,
since `project_distribution` clamps to the closed interval. So the margin `-6` bought nothing and cost
a little: at `-5` the death outcome lands on **one** atom instead of being split across two.

An arm that turns shaping back on needs `v_min ≤ DEATH_REWARD − c`, which is the startup guard's job
rather than a permanent margin.

#### Atoms landing on `0` and `100` is nearly cosmetic — do not count it as a reason

Spacing 2.5 puts `-5` on atom 0, `0` on atom 2, `100` on atom 42 and `120` on atom 50. That reads like a
design virtue and mostly is not, because **every predicted value is fractional** and alignment has
nothing to do with predictions — only with where *target* mass is placed. Three cases:

- **Non-terminal steps (~99.9% of transitions): alignment does nothing.** The target support is
  `r + 0.9975·z_i`, and the γ contraction takes every atom off the grid regardless — `z = 100` maps to
  99.75, which splits 0.10/0.90 across atoms 97.5 and 100. Fractional, exactly as expected.
- **Terminal steps: this is the only place exact values arise at all.** At `d = 0` the target support
  collapses to `r` for every atom, so the target is a point mass at the literal reward — and the rewards
  *are* round. A death gives a **one-hot** target at `-5`; a win a one-hot target at `100`. Off-grid, the
  same target would split across two atoms with an irreducible cross-entropy equal to that split's
  entropy. And the alignment is partial anyway: `STARVE_REWARD = -0.5` splits 0.20/0.80 for a floor of
  **0.5004 nats**, and `FOOD_REWARD = 1.0` is not on an atom either.
- **The one thing that floor would have mattered for is already fixed.** With cross-entropy as the PER
  priority, a permanently split target carries a permanent priority floor and gets over-sampled forever.
  [Using the KL](#the-priority-signal-kl-not-cross-entropy) subtracts `H(target)`, so the signal reaches
  0 either way — which absorbs most of the benefit alignment would have had.

The residual benefit is **interpretability**: with `100` on an atom, "probability this state wins" reads
straight off atom 42 and death mass off atom 0, with no binning error, which matters only because this
folder does write diagnostics that inspect network internals. Keep the alignment because it is free;
never choose a grid for it.

Note throughout that `project_distribution` *interpolates* between neighbouring atoms, so the
represented **mean is exact** at any spacing; what atoms buy is distributional detail.

**The bimodality argument is now measured, and it is stronger than claimed.** Within a *single* length
band the return is either near zero or near 100, which is precisely what a scalar Q cannot represent:

| checkpoint, band | p50 | p95 | p99.9 | max |
|---|---|---|---|---|
| `b30e` @200k, length 95-97 | **-0.21** | 2.15 | **101.74** | 102.23 |
| `b30e` @200k, length 90-94 | 3.35 | 85.26 | 100.26 | 100.77 |
| `b24d` @1342k, length 90-94 | 64.55 | 102.93 | 103.94 | 104.15 |

A mean-only estimate at length 95-97 for the mid-climb policy regresses toward a value **no outcome
ever pays**, and the spread it is hiding is the whole width of the grid.

**Reward rescaling stays rejected**, and the measurement does not reopen it: `runs.md`'s "explicitly
not planned" keeps `FOOD_REWARD`/`DEATH_REWARD`/`STARVE_REWARD`/`PERFECT_GAME_REWARD` fixed *because*
they rescale `avg_reward`, which is what `BOOTSTRAP_REWARD_THRESHOLDS` fires on — a rescale would
silently move the epsilon schedule and confound the arm.

#### The startup guards, in three levels

Choosing `v_max = 120` deliberately below the derived 194 means a single hard "must cover the
theoretical max" guard would refuse to start the arm this plan is for. So the guard separates a bound
that is never worth violating from a trade-off that is, and **the escape hatch must not silently
disable both**:

| condition | level | why |
|---|---|---|
| `v_min > DEATH_REWARD − CHASE_SAFE_SHAPING` | **hard fail** | costs nothing to satisfy and there is no reason to violate it; a clipped death value is a wrong terminal target |
| `v_max` < the **measured** max, 105 | **hard fail** | clipping returns that real policies demonstrably reach is a mistake, not a trade |
| `v_max` < the **derived** bound, 194 | **warn once, and record it** | a judgement, which is what `120` is. The line prints the derived bound, the measured max, the headroom, and the steps-per-meal a policy would need to exceed the grid, and `run_config` carries it into `runs/<policy>.md` |

`SNEK_C51_ALLOW_CLIPPING=1` overrides only the two hard fails. Both derived numbers are **computed from
the reward constants** — `(MAX_POSSIBLE_SCORE − 1)·FOOD_REWARD + PERFECT_GAME_REWARD` for the bound — so
they move when a reward does instead of going stale, which is the failure the first draft's hand-written
`101` would have had.

### The one new confound the grid creates: initial optimism

**A categorical head's initial expected Q is the grid's midpoint**, because the atom logits start
near-uniform. That is **57.5** for `[-5, 120]`, against a ddqn head that starts at Q ≈ 0 (its final
layer is `RandomUniform(±0.03)`). So a c51 arm begins believing every state is worth about what a
guaranteed perfect game is worth, and **the size of that optimism scales with `v_max`** — i.e. with the
choice just made above.

It is not obviously harmful (optimism in the face of uncertainty is a respectable exploration prior, and
the first thousands of steps run at epsilon 0.4 anyway), but it is a **second difference between the
arm and its control**, which is exactly what this folder's protocol exists to prevent. Two options:

- **Default: standard uniform init**, as in the paper, and measure in phase 2 how many steps the
  optimism takes to wash out. Simple, matches the literature, keeps the confound visible in the write-up.
- **`SNEK_C51_ZERO_INIT=1`: a downward bias ramp on the atom logits** so the initial expected Q is 0,
  matching ddqn. Computed: `bias_i = -λ·(z_i − v_min)` with **λ = 0.1622** puts E[Q] at 0.0000, with 70%
  of the initial mass on the bottom 3 atoms. Three lines in the network builder, and note λ barely
  depends on `v_max` — the mass concentrates in the bottom ~9 atoms either way — so the constant
  survives a grid change, which is also why it must be **computed at build time rather than pasted in**.

Recommendation: **ship the knob, default it off, and decide from phase 2** — if the optimism visibly
delays the first learning, the ramp becomes the default for phase 4 and the write-up says so.

## The priority signal: KL, not cross-entropy

Upstream returns nothing, so this is a free choice. **Use `KL(target ‖ prediction) = CE − H(target)`,
which is what the Rainbow paper uses**, not the raw cross-entropy.

The reason is measurable in the probe: at initialisation the per-example CE ran **4.59-4.71**, a
0.12-wide band around `ln(111) = 4.7095`. Most of that magnitude is the *irreducible* entropy of the
projected target — the projection spreads mass over two atoms, so `H(target) > 0` always — and an
additive near-constant compresses the relative spread that PER's `alpha` exponent acts on. That is
the same defect this repo already documented for `td_loss` as a priority (Huber's log-log slope
against `|δ|` measures 1.92-1.99, so `td_loss` at `alpha=0.6` is an effective exponent near 1.2), and
sharpness is [a variance dial, not a quality dial](../hyperparamTuning/findings.md) — a sharper
signal raised both the ceiling and the death rate.

**One thing is worse here than in the Huber case, which is why this is not a cosmetic choice.**
`td_loss` and `td_error` rank the *same* transitions — Huber is monotone in `|δ|`, top-1000 Jaccard
**1.0000** on 8 of 8 arms, so the signal only ever changed how much mass the top got. CE and KL differ
by `H(target)`, which is **not** monotone in KL, so the two genuinely *reorder* transitions. This
choice can change which experience gets replayed, not just how hard.

`SNEK_PRIORITY_SIGNAL` keeps its two existing values and both map to the KL for a c51 arm, so
**`training.py` needs no change at all** — the subclass returns the per-example KL in *both*
`td_loss` and `td_error`. `SNEK_PRIORITY_SIGNAL=ce` is added as the ablation.

## Design

### Knobs (all default to today's behaviour)

| knob | default | meaning |
|---|---|---|
| `SNEK_ALGO` | `ddqn` | `ddqn` or `c51`. Anything else exits at startup |
| `SNEK_NUM_ATOMS` | 51 | size of the support grid, [chosen from the measurement](#phase-0-results--the-support-measured-2026-08-15). Ignored unless `ALGO=c51` |
| `SNEK_V_MIN` / `SNEK_V_MAX` | -5.0 / 120.0 | support bounds, guarded as above |
| `SNEK_C51_ZERO_INIT` | 0 | 1 ramps the atom-logit biases so the initial expected Q is 0 rather than the grid midpoint — see [initial optimism](#the-one-new-confound-the-grid-creates-initial-optimism) |
| `SNEK_C51_DOUBLE` | 1 | 1 = action for the target atoms chosen by the **online** net (Rainbow-style double, matching today's `DdqnAgent`); 0 = upstream's target-net selection |
| `SNEK_C51_ALLOW_CLIPPING` | 0 | escape hatch for the two support guards |
| `SNEK_PRIORITY_SIGNAL` | `td_error` | gains `ce` as a third value, meaningful only for c51 |

`SNEK_C51_DOUBLE=1` is the default deliberately: the control is `DdqnAgent`, so upstream's
single-selection C51 would change *two* things at once. The knob keeps the confound measurable
instead of baked in.

### `arch.json` — four new fields, and the trap they close

A c51 checkpoint restored into a ddqn network is the exact failure `policy_arch.py` was written for:
the head is `3·N` wide instead of 3, and `expect_partial()` says nothing. Worse, `v_min`/`v_max` are
**part of the policy** — the greedy action is `argmax Σ z_i p_i`, so restoring correct weights against
the wrong support silently produces a different policy with no shape mismatch anywhere.

So `build_arch` gains `algo`, `num_atoms`, `v_min`, `v_max`; `assert_restorable` compares `algo`
(reading a missing field as `'ddqn'`, so every existing directory and every `hallOfFame/` entry keeps
loading untouched); `assert_config_matches` adds all four to the resume check alongside
`fc_layer_params`; and the support comes **from the file** at eval and watch time, never from the
environment — the same rule that removed the reliance on `SNEK_FC_LAYERS`.

### Edits, file by file

| file | change |
|---|---|
| **`categorical_agent.py`** *(new, ~90 lines)* | `SnekCategoricalDqnAgent(CategoricalDqnAgent)`. Overrides `_loss` (per-example KL in `extra`, `sample_weight=weights` into `aggregate_losses`) and `_next_q_distribution` (online-net argmax when `double`). `n_step_update > 1` raises at construction — the n-step target-support branch is ~10 more lines and no recent arm uses n>1 (n=3 measured null in batch 15, n=5 closed), so it is left out rather than shipped untested |
| **`under_the_hood.py`** | `build_categorical_q_net(num_actions, fc_layer_params, num_atoms)` beside `build_q_net`, wrapping `CategoricalQNetwork` so the two share one construction site and one set of initializers |
| **`policy_arch.py`** | the four fields, the `'ddqn'` default, the two extra assertions |
| **`snek2.py`** | the knobs, the two support guards, an `if algo == 'c51'` branch choosing net + agent, and `algo`/`num_atoms`/support in `run_config` so `runs/<policy>.md` records them |
| **`eval_agent.py`** | `build_eval_agent` dispatches on `arch['algo']`, building the categorical net and agent with the **recorded** support. Covers `eval_checkpoints.py` and `eval_workers.py`, which both call it |
| **`watch.py`** | currently builds its own `DdqnAgent` — a fourth construction site the module docstring in `eval_agent.py` claims does not exist. Route it through `build_eval_agent` rather than adding a second c51 branch |
| **`hyperparamTuning/perDiagnostics/*.py`** | `input_sensitivity_over_time.py`, `per_priorities.py`, `plasticity.py`, `plasticity_probe.py` call `agent._q_network` directly and would get `[B, 3, N]` logits. Add a shared `expected_q(net, obs, arch)` helper and use it, so a c51 arm is measurable rather than a shape error |
| **`training.py`** | **none** — that is the point of returning the signal in both `extra` fields |

### Tests — `tests/test_categorical_agent.py` plus additions to `test_policy_arch.py`

Each row names the mutation it is there to kill, per the "mutate the implementation and confirm a
test fails" rule.

| test | mutant it kills |
|---|---|
| unweighted single-selection loss equals upstream's on a fixed batch | any error in the reimplemented projection/CE — the check the probe already passes |
| skewed IS weights change our total; upstream's is unchanged | dropping `sample_weight=weights` (and it pins *why* the override exists) |
| `extra.td_error` is `[batch]`, finite, ≥ 0, and equals `extra.td_loss` | returning `()` again, or only filling one field, which would break `training.py` |
| KL is ~0 when prediction == target while CE is `H > 0` | reverting the priority to raw CE |
| **with a desynced target net**, the selected atoms are the online argmax's row | switching to target-net selection. **Must desync first** — see the probe finding, the mutant survives otherwise |
| loss at init ≈ `ln(num_atoms)` | a support/atom-count mismatch between net and agent |
| a `discount == 0` transition puts target mass at the reward, not spread over the support | anything that breaks the terminal-bootstrap contract `snake_environment.to_tensor_time_step` is built on |
| `v_max` below the **measured** 105, or `v_min` above `DEATH_REWARD - CHASE_SAFE_SHAPING`, raises unless the escape hatch is set — while `v_max = 120` starts and *warns* | collapsing the three guard levels into one, which would refuse to start the arm this plan is for, or into none |
| the derived bound equals 194 for the shipped rewards, and *moves* when `PERFECT_GAME_REWARD` or `MAX_POSSIBLE_SCORE` is changed in the fixture | hard-coding 194 or 105, which would go stale silently |
| `SNEK_C51_ZERO_INIT=1` puts the initial expected Q within a tolerance of 0, and the default leaves it at the midpoint | the ramp silently doing nothing, which would look identical in every metric |
| the shield masks fatal exploration moves over a *categorical* greedy policy | a regression in `ShieldedEpsilonGreedyPolicy`'s `info_spec` contract |
| arch: missing `algo` reads as `ddqn`; c51-vs-ddqn mismatch raises; atoms/support mismatch fails a resume; round trip keeps all four fields | every silent-restore path this repo has already been bitten by twice |

Then the full suite: **24 modules** (up from 23), ~610 tests, 0 failed.

## Phases

| phase | work | gate to the next |
|---|---|---|
| **0** | **done 2026-08-15** — [`return_distribution.py`](../hyperparamTuning/perDiagnostics/return_distribution.py), 3 checkpoints × 60 episodes × 116,454 states. Support chosen, the section's own premise falsified, and the bounds then re-derived from the reward function rather than the sample: [results](#phase-0-results--the-support-measured-2026-08-15) | `v_min = -5`, `v_max = 120`, `num_atoms = 51` |
| **1** | the edits and tests above, plus mutation checks | full suite green; every mutant fails |
| **2** | `snek2.py smoke` with `SNEK_ALGO=c51 SNEK_MIN_CHECKPOINT_SCORE=0 SNEK_MAX_STEPS=3000`: trains, writes `arch.json` with `algo=c51`, checkpoints, `watch.py smoke` loads it, and a one-checkpoint eval measures it. Record steps/s against a matched ddqn smoke | it runs end to end on both hosts' code path. **Note the eval displaces every `evals/` PNG** (CLAUDE.md) — check what is at the top level first and restore afterwards |
| **3** | pilot: one wave of 4 arms, `{lr 1e-5, lr 5e-5} × 2 seeds` at the chosen support, `SNEK_MAX_STEPS=600000`. A **screen, not a result** — does it learn at all, how fast to the first perfect game, is the loss scale sane at 1e-5 | pick one rate. If neither reaches its first perfect game by ~300k while the control did, stop and report rather than sweeping |
| **4** | the batch: 4 seeds of the chosen config, **matched to `b25` (`FC_LAYERS=200,100,100`, IS-off)** so `b25a-d` is the seed-matched control and no control arms need running | the pre-registered criteria below |

### The phase 3/4 launch line

`b25`'s config verbatim, plus the new knobs, one process per seed. The batch prefix is taken at launch
(next free is **`b31`**, but `fc 512` and the four owed `320` seeds are ahead of this in the backlog,
so do not reserve it):

```
cd snek2
SNEK_ALGO=c51 SNEK_NUM_ATOMS=51 SNEK_V_MIN=-5 SNEK_V_MAX=120 \
  SNEK_FC_LAYERS=200,100,100 SNEK_IS_WEIGHTS=0 SNEK_TARGET_UPDATE_PERIOD=1000 \
  SNEK_DISCOUNT=0.9975 SNEK_FOOD_DISTANCE_REWARD=0 SNEK_FORK_BRANCHES=4 \
  SNEK_MAX_STEPS=2000000 SNEK_SEED=<n> \
  PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u snek2.py b31<x>-c51seed<n> \
  > /tmp/b31<x>.log 2>&1 &
```

**The control is `b25`, not the record `b24`, and the reason is the head.** A C51 head is
`last_layer_width × 3·num_atoms`, and `fc 320` is the one shape whose widest layer is also its *last*,
so it pays for the head at 320 wide while `200,100,100` pays at 100. At the chosen 51 atoms that is the
difference between 1.41× and 5.42×:

| fc | ddqn | **c51 N=51 (chosen)** | c51 N=81 | c51 N=111 |
|---|---|---|---|---|
| `50,100,50` | 11,853 | 19,503 (1.65×) | 23,853 (2.01×) | 28,683 (2.42×) |
| **`200,100,100`** | 36,703 | **51,853 (1.41×)** | 60,943 (1.66×) | 70,033 (1.91×) |
| `320` | 10,883 | 59,033 (5.42×) | 88,003 (8.09×) | 116,813 (10.73×) |

`200,100,100` also carries **+10.3 of b24's +12.2** on the IS-off lift, and that lift tracks the widest
layer rather than the parameter count — so this control keeps nearly all of the record config's
strength while cutting the confound from 5.42× to 1.41×. **The cost, stated plainly: `b25` has zero
`≥98%/500` checkpoints** where b24 has two, so the decisive-artifact criterion below becomes one-sided
— a c51 arm producing one is a clear win, producing none says nothing.

`SNEK_FORK_BRANCHES=4` is not optional and nothing in the summary block reports it, so an omission
differs from the control invisibly — the trap b27 nearly fell into. Confirm every arm with
`grep 'hyperparameter override:'` on its log, which should now include `ALGO`, `NUM_ATOMS`, `V_MIN`
and `V_MAX`.

**Host note.** The desktop is full (b28 running, b29 queued) and pushing to `ops` starts real work on
another machine, so it needs its own approval. The laptop has no trainers but is currently running
b30's four close-out evals; the pilot waits for that to finish rather than sharing 14 cores with it.

### Pre-registered criteria for phase 4

**Primary is `best_perfect30` at a matched truncation of 2M**, not `strong_eval_fraction`. That is the
2026-08-14 re-measurement talking: at this level `best_perfect30` has sd **0.67** against `sef`'s
**5.59**, so paired at n=4 it resolves ~**3.6 pp** where `sef` resolves ~21.3 pp, and it still has
3.8 pp of headroom. The `b25` control reads **94.3** on it (`sef` 63.8, pooled 86.0), against b24's
96.2. `sef` is reported alongside for continuity. **Never on `peak_trailing`** — b25 and b24 both sit
on its 95.00 cap; report the count of trailing-95.00 windows instead.

Secondary, and the one the mechanism actually predicts: **drawdown** — peak `pf30` minus the mean over
the 500k that follows it.

Decisive artifact: **the count of full-length ≥98%/500 rows** out of the close-out → HOF-500 chain,
where **the `b25` control is `0 · 0 · 0 · 0`** and b24 is `1 · 1 · 0 · 0`. On the desktop that chain is
automatic; **on the
laptop both passes are by hand**, as 4 parallel `eval_checkpoints.py` processes with `EVAL_WORKERS` ≥ 4.
Note also that a strong arm can finish with an empty HOF job, since that chain gates at 98 — `b25b`'s
plausible 97.2% was abandoned that way — so a gate-97 pass by hand is worth doing if the close-out
looks strong.

| outcome | reading |
|---|---|
| `best_perfect30` up with 4/4 seeds in the same direction | real. Close out at gate 95, then the HOF-500 pass, then promote anything that holds ≥98%/500 |
| `best_perfect30` null but drawdown better in ≥3/4 seeds | the predicted result. Worth 4 more seeds, since drawdown has never been a primary metric and its between-seed variance is unmeasured |
| both null | **C51 is closed for this task** — and given how strong the literature's prior is, that is a `findings.md` entry in its own right |
| clearly worse | check the learning-rate confound *once* (one arm at the pilot's other rate), then close |

Two power facts the reading has to respect. **`n=4` cannot resolve an effect below ~10 pp on `sef`**
(5 pp needs n≈17-37), which is why the criteria are written as "large effect or clear null" rather
than as a comparison of point estimates. And **pair seed by seed**: seed 2 or 4 has been the best arm
in **18 of 18** config waves (paired p=0.00005), so an unpaired mean comparison mostly measures the
seed draw.

## Risks, in the order they are likely to bite

1. **Learning rate.** Cross-entropy has nothing to do with the Huber TD loss's scale, so `1e-5` —
   tuned for the latter, and already flagged in the backlog as "very conservative" — may be simply
   wrong for c51. This is the most likely way a real effect gets read as a null, which is why phase 3
   screens two rates before any seed set is spent.
2. **PER interaction.** Priorities change meaning even with the KL fix, and `alpha=0.6` was chosen
   against `|td_error|`'s spread. Left at 0.6 for the first batch **on purpose** — one change at a
   time — with `alpha` as the named follow-up if the batch is ambiguous. Note the phase-4 control is
   `IS_WEIGHTS=0`, so the IS-weight path is not even exercised there; the fix matters for the default
   config and for not shipping a knob that silently does nothing.
3. **Head-size confound — now 1.41×, down from 10.73×.** C51 multiplies the head by `num_atoms`, and
   choosing the `b25` trunk plus 51 atoms is what shrinks it: 51,853 parameters against the control's
   36,703, where the record's `fc 320` trunk would have cost 5.42× at the same atom count. The licence to treat that as harmless is batch 20's nine-shape sweep ("architecture never
   raises the ceiling") and the b24/b25/b26 finding that the lift tracks the *widest layer*, not the
   parameter count — b25 itself already runs at 3.09× b22's parameters. It still belongs in the
   write-up as a stated confound.
4. **Support mis-specification.** Closed by phase 0 and the two startup guards — and phase 0 found the
   analytic bound would have been too low, so this risk was real.
5. **A reward-derived quantity breaking silently.** This plan touches no reward, but it does touch the
   terminal bootstrap, and the last thing to touch that class cost eight arms 300k+ steps each: the
   perfect-game counter compared a final reward with `PERFECT_GAME_REWARD`, read 0%, and — because
   `training.epsilon_for` uses the trailing perfect rate as its skill signal — **pinned epsilon at its
   ceiling**, so the runs were handicapped as well as mismeasured. The `discount == 0` test above is
   there for that reason, and phase 2's smoke check should confirm `perfect_percent` is non-zero on an
   arm that fills a board.
6. **`tf_agents` 0.18.0 internals.** The subclass depends on `_support`, `_num_atoms`,
   `_next_q_distribution` and `project_distribution`. The equal-to-upstream test is the tripwire: if
   an upgrade changes the projection, that test fails loudly instead of the arm training subtly wrong.

## What this is not

- **Not Rainbow.** No noisy nets, no duelling head, no multi-step. Those are separate arms and
  bundling them would make the result unattributable, which is the mistake batch 10 is still paying for.
- **Not QR-DQN/IQN, and phase 0 removed the reason to want it.** Quantile regression needs no
  `[v_min, v_max]`, which would have been the principled answer if the returns had piled up near zero
  under a grid stretched to 104. They do not — the bulk spans most of the grid — so the fixed support
  costs nothing here, and ~150 lines with no `tf_agents` support buys nothing over C51's ~90.
- **Not a reward change.** The support is chosen to fit the existing rewards, never the reverse.
