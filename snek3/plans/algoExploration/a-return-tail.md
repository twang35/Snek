# Group A: the return tail -- DQN, C51, QR-DQN, IQN, FQF, Munchausen

**Read 2026-09-17, A1 closed and A2's stability batch at 85%** (`docs/runs.md` b35, b36): the paper cell does not reach
competence in 10M moves on either head -- DQN 7-16% perfect and C51 14-25%, both still rising, no stage B -- while the local
cell reaches 90% in 0.6-1.6M counted steps and plateaus at 71-88%. §6's first bullet is the case that applies: the ladder's
rows are read on the local plumbing, with the paper cell kept at A4, unless the user prefers to give the paper cell its full
50M-move budget (~18 h an arm on the desktop). b36's stability criterion never engaged (no arm reached 80%), so A2 carries its
own stability read in its drawdown columns; 51 and 101 atoms were level, and 51 is the setting. Queued the same day as b37 (C51 51 atoms, QR-DQN N 32) and b38 (IQN N = N′ 8, neutral and CVaR-0.25-trained, 2M cap), both
on the local plumbing. **The quantile counts are not the papers'**: measured on this CPU one arm at a time (counted steps/s)
DQN 484, C51 285, QR-DQN N 32 / 64 / 200 at 225 / 74 / 10, IQN N = N′ 64 / 32 / 16 / 8 at 14 / 23 / 52 / 92 -- the quantile
Huber is N × N′ pairs and IQN's embedding is a `Linear` per (sample, τ), so 200 and 64 are days per arm. A5 takes FQF's 32
and A6 inherits its rung's.

**Status: built 2026-09-17** (`algos/dist/`, the linear ε schedule and the Munchausen knobs on `algos/dqn/`, the sidecar's `head`, the `--policy-variant` read; tests `tests/test_dist_*.py`, mutants `tests/mut_dist.json`). **Where each row stands is the `status` column of §3 and the gate table of §4** (added 2026-09-18, after FQF was found not to have run its smoke gate until the day it was queued). Batches queue in the order of §3, A1 first. Group A of [`algorithm-series.md`](algorithm-series.md);
conventions in [`README.md`](README.md). Phase 1 (A1) and phase 2 (A2-A6) of the running order.

The question: does modelling the *distribution* of the return, rather than its mean, help a game whose
champions die of one rare fatal move late in a long game? The ladder is one knob per rung -- the
distribution itself (C51), then dropping the fixed support (QR-DQN), then sampling the fractions
(IQN), then learning them (FQF), and finally an entropy-regularised bootstrap laid over the best
(Munchausen). **A4's risk-sensitive arm is the row this group exists for**: acting on the low quantiles
is the one thing a scalar critic cannot do, and it is the most direct test of the diagnosis in
`docs/findings.md` that the failures are rare fatal moves, not noisy returns.

## 1. What the group shares

All six rows are value-based agents on replay. They reuse `algos/dqn/` for everything that is not the
head or the loss: `replay.py` (prioritised, numpy sum tree), `collect.py` (lanes, n-step, the fork),
`schedules.py` (epsilon and the shield fraction off the eval history), and `agent.py`'s exploration
shield and `build_adam`. **The rungs differ in three places and only three**: the head, the loss, and
how a greedy action is read off the head.

| shared piece | decision |
|---|---|
| package | one package, `algos/dist/`, holding the distributional heads and losses, with one `algo.py` per rung registering `NAME`s `c51`, `qrdqn`, `iqn`, `fqf`. Four thin `algo.py` files over one `heads.py` and one `losses.py`, rather than four packages that each copy the replay wiring. `algos/dqn/algo.py` stays the DQN row's |
| the agent | a `DistAgent` with the same `update`/`target` shape as `DdqnAgent`, parameterised by a head object that answers `q_values(logits) -> (m, actions)` for the greedy read and `loss(online, target, batch)`. Double-DQN action selection is kept on every rung (argmax of the online mean, evaluated on the target), so A2-A5 differ from A1 only in the head |
| the sidecar | `arch.json` gains `head`: `{"type": "c51", "atoms": 51, "v_min": -10, "v_max": 110}`, `{"type": "quantile", "n": 200}`, `{"type": "iqn", "embedding": 64, "n_tau": 64, "n_tau_prime": 64, "k": 32}`, `{"type": "fqf", "n": 32, "embedding": 64}`. The signature includes it. A DQN sidecar has no `head` and the restore path treats absence as the scalar head, so every existing checkpoint still loads |
| restore | `tools/restore.ALGORITHMS` gains the four names; each returns a module whose greedy policy is argmax over the head's **mean**. The risk-sensitive read (A4) is a second policy the sidecar does not select -- it is chosen by the eval, §5 |
| knobs | `SNEK_DIST_*` for what the group owns (`SNEK_DIST_ATOMS`, `SNEK_DIST_V_MIN`, `SNEK_DIST_V_MAX`, `SNEK_DIST_QUANTILES`, `SNEK_DIST_TAU_SAMPLES` / `SNEK_DIST_TAU_PRIME_SAMPLES` / `SNEK_DIST_POLICY_SAMPLES` for IQN's N, N′ and K, `SNEK_DIST_EMBEDDING`, `SNEK_DIST_KAPPA` for the Huber threshold, `SNEK_DIST_FRACTION_LR` and `SNEK_DIST_FRACTION_ENTROPY` for FQF's proposal net). Everything DQN already names (`SNEK_LEARNING_RATE`, `SNEK_BATCH_SIZE`, `SNEK_N_STEP_UPDATE`, `SNEK_TARGET_UPDATE_*`, the fork, the replay, epsilon) keeps its name and default, because it means the same thing. PPO's knobs are refused by name |
| the paper's exploration | every paper in this group anneals ε linearly from 1.0 to a floor over a fixed number of steps and holds it; snek3's `refine_epsilon` reads the eval history instead. The paper cell needs the paper's schedule, so **`algos/dqn/schedules.py` gains `SNEK_EPSILON_SCHEDULE=linear`** with `SNEK_EPSILON_ANNEAL_STEPS` (moves) and the existing `SNEK_INITIAL_EPSILON` / `SNEK_MIN_EPSILON` as its ends; `eval`, the default, is byte-for-byte the schedule A1 ran, and a fixture says so. This is the one change to `algos/dqn/` before A6, made **before A1 runs** so the control has it too. The shield is `SNEK_GUIDED_FRACTION=0` in the paper cell |
| the step | DQN's: `step_granularity` 1, one `collector.step()`, four game moves at the default fork. The x-axis is DQN's, so A2-A6 read directly against A1 |
| the reward scale | the papers clip rewards to [−1, 1] and size every scale against that (`README.md`, "Translating"); here rewards are not clipped, so the support and the Huber threshold are set from the reward configuration the batch runs under. With win 100, food 1, death −5 and γ 0.99 the discounted return lies in roughly [−6, 110]; **C51's `v_min`/`v_max` must bracket that**, and this is the first thing the smoke checks (§4). κ, the Huber threshold, is 1 as in every paper -- on this reward scale that is one meal, which is the paper's unit too |

**Why not fold the heads into `algos/dqn/agent.py`.** The DQN row is the control for everything above
it, and A1 is scheduled to run *before* any head exists. A control whose code changed between its run
and the rows read against it is not a control. `algos/dqn/` does not change in this group.

## 1b. The papers' settings, and how each lands here

Every paper in the group runs the Dopamine-era Atari recipe -- Adam, batch 32, a 1M replay, one update
per 4 agent steps, a target copy every 8,000-10,000 updates, ε 1.0 → 0.01 annealed over the first
250k-1M steps and held, 1-step returns, uniform replay (PER is Rainbow's, not this group's) -- and the
papers differ from each other in the head, the loss and the learning rate. The **paper cell** of every
row runs this recipe on Snake, translated by `README.md`'s rules; the **local cell** runs the same head
on `algos/dqn/`'s tuned defaults (lr 1e-5, batch 128, a 100k replay, a hard target copy every 8 updates,
PER, the eval-driven ε, the shield and the fork). The two cells differ in the plumbing and agree on the
algorithm, which is what makes the gap between them attributable.

| setting | the papers (verified 2026-09-16 against the papers and the Dopamine / authors' configs) | paper cell here | local cell |
|---|---|---|---|
| optimiser, batch | Adam, 32 | Adam, `SNEK_BATCH_SIZE=32` | Adam, 128 |
| learning rate, optimiser | **DQN: RMSProp** 2.5e-4 (decay 0.95, ε 1e-5, centred -- Nature and Dopamine; there is no Adam DQN in the original paper). C51 Adam 2.5e-4; QR-DQN, IQN, FQF, M-DQN, M-IQN Adam **5e-5**; Adam ε 0.01/32 = 3.125e-4 throughout | A1's paper cell uses the **Munchausen paper's DQN-Adam** (Adam 5e-5, ε 3.125e-4), because it is the DQN every other rung's paper is compared against and A6's M-DQN is built on it; the RMSProp original is not run. `SNEK_LEARNING_RATE` 2.5e-4 for A2, 5e-5 for A1 and A3-A6; `SNEK_ADAM_EPSILON` 3.125e-4 | 1e-5, 1e-7 |
| replay | 1M transitions, uniform, prefill 20k steps (Dopamine `min_replay_history`; FQF 50k) | `SNEK_REPLAY_BUFFER_MAX_LENGTH=1000000`, `SNEK_PRIORITY_EXPONENT=0`, `SNEK_INITIAL_COLLECT_STEPS=20000` | 100k, PER 0.6, 2,000 |
| update frequency | one gradient step of batch 32 per 4 agent steps: 8 replayed samples per transition | `SNEK_REPLAY_RATIO=0.25` (the knob is gradient steps per transition, `algos/dqn/algo.py`), batch 32: 8 samples per move | 1.0 at batch 128 |
| target network | hard copy every 8,000 **agent steps** (Dopamine's `target_update_period` counts agent steps; at one update per 4 steps that is 2,000 gradient updates, 32k frames; Nature DQN's "10,000 parameter updates" is the same order) | `SNEK_TARGET_UPDATE_PERIOD=2000` -- **the knob counts gradient updates** (`DdqnAgent.train_step`), so 8,000 here would be 4× the paper's | 8 |
| n-step | 1 in every paper of this group (Dopamine's IQN gin uses 3; the Munchausen paper reverts it to 1 and says so) | `SNEK_N_STEP_UPDATE=1` | 1 |
| exploration | ε linear 1.0 → 0.01 over the first **1M frames = 250k agent steps** (Dopamine `epsilon_decay_period` 250k; Nature DQN's "final exploration frame" 1M); ε 0.001 at eval | `SNEK_EPSILON_SCHEDULE=linear`, `SNEK_INITIAL_EPSILON=1.0`, `SNEK_MIN_EPSILON=0.01`, `SNEK_EPSILON_ANNEAL_STEPS=250000` moves (the earlier draft's 1M was frames read as moves); eval greedy as every snek3 eval is; **shield off, fork off** | the eval-driven schedule, 0.4 → 0.002, shield 0.8, fork 4 |
| discount | 0.99 | 0.99 | 0.99 |
| gradient clipping | none (DQN-family papers); the dueling paper clips the norm at 10 | none | none |
| reward | clipped to [−1, 1] | **not clipped** -- see §1; scales set from the return range instead | same |
| network | Nature CNN → 512 | `SNEK_FC_LAYERS=320`, the reference's trunk; the head is the rung's | same |
| budget | 200M frames = 50M agent steps, 3-5 seeds | 50M moves a cell -- **50M counted steps in the paper cell**, since the fork is off and `collect_envs` is 1, so one counted step is one move; the local cell's four-branch fork makes it 12.5M counted steps for the same moves -- 4 seeds; raised to the reference's 100M if the curve is still rising | as the paper cell |

Per-rung values the papers fix and the plan takes verbatim:

| rung | the paper's values |
|---|---|
| A2 C51 | 51 atoms; support [−10, 10] **on clipped rewards** -- here [−10, 110] at 51 atoms is a 2.4-wide bin, so the stability batch also runs **101 atoms** (the same 2.4 → 1.2 width step the paper's 21 → 51 gave); cross-entropy on the projected target; ε 0.01 |
| A3 QR-DQN | N 200, κ 1 (QR-DQN-1), lr 5e-5 |
| A4 IQN | N = N′ = 64 loss samples (Dopamine's config; the paper says 8 "appears to be sufficient" and does not state the Atari-57 value), K 32 policy samples, cosine embedding 64, κ 1, lr 5e-5. **Risk-sensitive form: the distortion is applied to the acting policy during training and to the policy in the Bellman target, and the score is measured risk-neutrally** -- §4 verbatim (re-read from the PDF 2026-09-16): "This only affects the policy, π_β, used both in Equation 2 and for acting in the environment", and "Figure 3 (left) shows our results in this experiment, with average scores reported under the usual, risk-neutral, evaluation criterion". So the paper cell trains under CVaR and stage B reads the neutral mean; the batch also measures the CVaR read of every checkpoint (§5), which is this plan's addition. CVaR 0.25 and CVaR 0.1 are the paper's two CVaR arms |
| A5 FQF | N 32; quantile net Adam 5e-5; fraction proposal **RMSProp** (centered, momentum 0, ε 1e-5) at 2.5e-9; fraction entropy coefficient 0.001 (the released code's default); target 10k; uniform replay, prefill 50k |
| A6 Munchausen | α 0.9, τ 0.03, l₀ −1; ε-greedy acting (not the softmax); 1-step; M-DQN on Dopamine DQN-Adam at lr 5e-5 and target 8,000, M-IQN on Dopamine IQN with 1-step. **τ is set against a unit reward**; here the food reward is 1, so τ 0.03 means what it meant, and the +100 win is the term it under-weights -- the smoke records the log-policy term's magnitude beside the reward |

## 2. The rows

### A1 -- DQN (double, prioritised, forking; `algos/dqn/`)

Already built. Nothing to implement; the row's work is the batch. This is the first run of `dqn` on the
26-value observation with the current reward preset and the `hist8` history, and the last DQN batches
(`b1`, `b2`, snek3's phase-3 gates) ran under the 30-value observation and the `b2` preset. It also has
to answer whether DQN's step budget is right: `docs/runs.md` b2 crossed 90% at 324k counted steps; the
cap here is chosen from that and F1's answer once F1 has one.

### A2 -- C51 (Bellemare, Dabney & Munos 2017)

A categorical distribution over a fixed support of 51 atoms per action; the Bellman target is projected
back onto the support and the loss is the cross-entropy. What it isolates against A1 is the
distribution itself. snek2 ran C51 (`categorical_agent.py`, b38) and found it unstable enough that a
win-reward shrink was tried and falsified, so **the first C51 batch is a stability batch, not a
comparison**: two doses of `v_max` (110 and 200) at two seeds each, judged on whether the perfect rate
holds once reached. Only a stable pair proceeds to the four-seed comparison against A1.

| module | contents |
|---|---|
| `algos/dist/heads.py::Categorical` | the `(m, actions, atoms)` logits, softmax, the support, `q_values` as the expectation, the projection of a shifted and scaled target distribution onto the support |
| `algos/dist/losses.py::categorical_ce` | cross-entropy against the projected target, per-sample so PER's importance weights apply |
| `algos/dist/algo.py` (`c51`) | `build_config` adds atoms and support; `describe` names them; `fields`/`on_eval` are DQN's |

Tests: the projection on a hand-worked three-atom support (a reward that lands between atoms splits its
mass in the documented proportion; a target beyond `v_max` clips to the top atom); the expectation of a
one-hot distribution is its atom; a batch where every target equals the online distribution has zero
loss. Mutants: the projection's floor/ceil swap, the `(1 − done)` on the target, the support's sign.

### A3 -- QR-DQN (Dabney, Rowland, Bellemare & Munos 2018)

N quantile values per action at fixed fractions τ_i = (2i − 1) / 2N, fitted by the quantile Huber loss;
no support to pick. What it isolates against A2 is dropping the support -- and it is the rung that says
whether A2's instability, if any, was the projection.

| module | contents |
|---|---|
| `heads.py::Quantile` | `(m, actions, n)` values; `q_values` is the mean over quantiles |
| `losses.py::quantile_huber` | the asymmetric Huber over every (online τ_i, target τ_j) pair, threshold κ, per-sample |

Tests: the quantile Huber at κ → 0 reduces to the pinball loss on a two-point example; the loss is
zero when online and target quantiles coincide; the mean of sorted quantiles equals the sample mean.
Mutants: the asymmetry weight `|τ − 1[u < 0]|` dropped, κ's role in the two branches swapped, the mean
over the wrong axis.

### A4 -- IQN (Dabney, Ostrovski, Silver & Munos 2018)

The quantile function itself is the network: τ is sampled each step, embedded with a cosine basis,
multiplied into the trunk, and the same quantile Huber fits it. What it isolates against A3 is
sampling the fractions. **And it adds the risk-sensitive arm**: acting on the mean of quantiles drawn
from τ ∈ [0, 0.25] (a CVaR policy) rather than from [0, 1].

| module | contents |
|---|---|
| `heads.py::Implicit` | the cosine embedding of τ, the Hadamard product with the trunk, `q_values(obs, taus)`; `taus` default to N uniform draws, and a `risk` argument maps them through a distortion (`cvar`: τ ← α·τ) |
| `algo.py` (`iqn`) | `build_config` adds the embedding width, N / N′ / K, and `SNEK_DIST_RISK_ALPHA` (1.0 = neutral) with `SNEK_DIST_RISK_TRAIN` (0/1). **The paper's risk-sensitive agent applies the distortion to the acting policy during training and to the target's policy, and evaluates risk-neutrally** (Dabney et al. 2018 §4); `SNEK_DIST_RISK_TRAIN=1` does exactly that, and is the paper cell. With it at 0 the training policy is neutral and only the greedy `policy_fn` uses α, which is this plan's own addition: a second read of a neutral checkpoint (§5) |

**The trunk is `QNet`'s hidden layers.** `algos/ppo/net.py` reuses `algos/dqn/net.py`'s `QNet` weight for
weight; the implicit head does the same for the hidden stack and replaces only the head, so an IQN arm
can `SNEK_INIT_FROM` a DQN or PPO checkpoint's trunk if that is ever wanted.

Tests: the cosine embedding at τ = 0 is all ones; with `risk_alpha` = 1 the policy equals the neutral
mean; with α → 0 the greedy action follows the lowest quantile on a hand-built two-action example whose
means tie and whose tails differ; with `risk_train` = 1 the target's argmax is taken under the distorted
samples and the target's *values* under undistorted ones (the paper's split). Mutants: the distortion
applied to the target's value samples, `risk_train` applied to the acting policy only, the embedding's
`π` dropped.

**The eval question this row raises** is in §5: a risk-sensitive `policy_fn` is a second greedy policy
over the same checkpoint, and the batch runs both.

### A5 -- FQF (Yang, Zhao, Du, Wei & Liu 2019)

A second network proposes the fractions themselves, trained on the 1-Wasserstein gradient with respect
to the fractions; the quantile network is IQN's. What it isolates against A4 is learning where the
quantiles go, which should matter most when the distribution is bimodal -- a perfect game against a
fatal move is exactly that shape.

| module | contents |
|---|---|
| `heads.py::FractionProposal` | a linear layer on the trunk feature producing N logits, softmax, cumsum to τ_1..τ_{N−1}, midpoints τ̂; its own **RMSProp** optimiser (centered, momentum 0, ε 1e-5, as the authors' code) at `SNEK_DIST_FRACTION_LR` (2.5e-9, the paper's) and an entropy bonus `SNEK_DIST_FRACTION_ENTROPY` (0.001, the released default) |
| `losses.py::fraction_loss` | the closed-form gradient `2 F(τ_i) − F(τ̂_i) − F(τ̂_{i−1})`, detached from the quantile net |

Tests: fractions are monotone in (0, 1) with τ_0 = 0 and τ_N = 1 by construction; the fraction gradient
is zero when the quantile function is linear (equal spacing is optimal); the proposal update leaves the
quantile network's parameters untouched. Mutants: the detach dropped, the cumsum replaced by the raw
softmax, the midpoint index off by one.

### A6 -- Munchausen (Vieillard, Pietquin & Geist 2020)

Not a new head: a log-policy term added to the reward and a soft (log-sum-exp) target, applied to any
of A1-A5. So it is **two knobs on `algos/dqn/` and `algos/dist/`, not a package**: `SNEK_MUNCHAUSEN_ALPHA`
(0 = off, 0.9 in the paper), `SNEK_MUNCHAUSEN_TAU` (the entropy temperature, 0.03), and the log-policy
clip `SNEK_MUNCHAUSEN_L0` (−1). With α = 0 and τ → 0 every algorithm is exactly what it was, and the
test says so byte for byte. The two arms: M-DQN on A1, then M-IQN on whichever of A2-A5 has the best
stage-B density.

**This does change `algos/dqn/agent.py` after the A1 control has run.** The change is behind a knob
whose default reproduces the old arithmetic exactly (a fixture asserts the target tensor is identical
with the knob at its default), so A1's numbers still stand. The `algos/dqn/` freeze in §1 is for the
duration of A2-A5; A6 is the point at which it lifts, and only for this.

Tests: with α = 0 the target equals the double-DQN target to the bit; the log-policy term is clipped
at `l0`; the soft target at τ → 0 recovers the max. Mutants: the clip's sign, the α applied to the
bootstrap instead of the reward, the temperature dropped from the log-softmax.

## 3. The batches, in order

Every row is two cells of four seeds -- one wave -- unless the table says otherwise: the **paper** cell on
§1b's recipe and the **local** cell on `algos/dqn/`'s defaults. Both cells share the reward preset,
`SNEK_OBS_HISTORY=8` and `SNEK_FC_LAYERS=320` of the PPO reference, and the rung's head values from §1b.

| row | status | batch | arms | base | read against | judged on |
|---|---|---|---|---|---|---|
| A1 | **closed** 2026-09-17, read (`docs/runs.md`, `docs/findings.md`) | **b35**, closed 2026-09-17 (`docs/runs.md`): paper 7-16% perfect at 10M moves, no stage B; local 90% by 0.6-1.6M then 71-88% | 4 seeds `dqn` **paper** (§1b: DQN-Adam 5e-5, batch 32, replay ratio 0.25, 1M uniform replay, target 2,000 updates, ε linear 1 → 0.01 over 250k moves, no shield, no fork) + 4 seeds `dqn` **local** (DQN's defaults) | the reference's reward, history and trunk; 50M moves, raised if still rising | PPO's `hist8` table (`docs/runs.md` b27); the two cells against each other | stage-B density, `hof5000`, `hof30k`, drawdown count |
| A2 stability | **closed** 2026-09-17, read | **b36**, closed 2026-09-17: no arm reached 80%, so the criterion never engaged; 51 and 101 atoms level | paper cell, 2 × 2: 51 atoms on [−10, 110] / 101 atoms on [−10, 110], seeds 1-2 each; `v_max` 200 as a third pair only if both clip mass at the top atom in the smoke | A1 paper | -- | does the perfect rate hold after onset; `zero_since` never >200 evals after 80% |
| A2 | **closed** 2026-09-18, read (`docs/results.md`) | **b37a-d**, queued 2026-09-17: `c51local`, 51 atoms on [−10, 110], local plumbing, 3M steps | 4 paper at the stable support + 4 local -- **ran as 4 local** | A1 | A1's two cells | as A1 |
| A3 | **closed** 2026-09-18, read | **b37e-h**, queued 2026-09-17: `qrdqnlocal`, **N 32** (200 runs at 10 steps/s here), κ 1, local plumbing, 3M steps | 4 paper, N 200, κ 1, lr 5e-5 + 4 local -- **ran as 4 local at N 32**, sharing b37's wave with A2 | A1 | A2 | as A1 |
| A4 | **closed** 2026-09-19, read (`docs/results.md`): no checkpoint reached stage B, 55-65% plateau, CVaR cell worse; the CVaR hand pass is moot | **b38**, queued 2026-09-17: `iqnlocal` + `iqncvar25`, **N = N′ 8** (64 runs at 14 steps/s here), K 32, local plumbing, **2M steps** | 4 paper neutral + 4 paper **CVaR 0.25 trained** (`SNEK_DIST_RISK_TRAIN=1`, the paper's risk-sensitive agent); the local cell and a CVaR 0.1 cell run in a second wave only if the first moves. Every neutral checkpoint is also *read* under CVaR 0.25 in stage B (§5) | A1 | A3 | as A1; the trained-CVaR cell against the neutral one, and the neutral-vs-CVaR *read* delta on the same checkpoints |
| A5 | **closed** 2026-09-20, read (`docs/results.md`): climbs where IQN flattened (three seeds cross 90 by 1.65M) but never holds and never reaches 97, stage B empty. Open: FQF at N 16 / 32 in waves of 4 | **b39**: `fqflocal`, **N 8** (32 runs at 16 counted steps/s on the laptop alone, 8 at 68, and the desktop's 8-arm wave runs at a seventh of that), K 32, local plumbing, **2M steps**, one cell of 4 seeds | 4 local at N 8 -- the paper cell dropped by §6's first bullet, a CVaR-trained cell left to A4; every checkpoint read under CVaR 0.25 by hand after | A1 | A4 | as A1 |
| A6 | **live** as **b40**, both waves on the laptop: M-DQN (w2) closed 2026-09-20 -- 100 stage-B rows, best 97.8, drifts late; M-QR-DQN (w3) at 2.4M of 3M, 89-94 after onset with 0-3% of evals below 80, the steadiest value cell yet, stage B owed | **b40**: `mdqnlocal` (M-DQN on b35's local cell) + `mqrdqnlocal` (M-QR-DQN, N 32, on b37's cell -- the densest stage B of A2-A5), α 0.9, τ 0.03, l₀ −1, **3M steps** | 4 paper M-DQN (on A1 paper) + 4 paper M-best (on the best of A2-A5's paper cells) | A1; that rung | A1 paper; that rung | as A1 |

Each row waits for the one above to close. Every arm is a `train` spec on the shared queue with
`SNEK_ALGO` naming the rung; `SNEK_OBS_HISTORY=8` is one depth per wave, as the queue rule requires.
**The paper cell is the row's headline number**; the local cell's job is to say what the fork, the shield,
PER and the fast target copy are worth on this game, and it is dropped from a row once two rows in a
row have shown the same sign.

## 4. Smoke and stability gates before a batch is queued

**Where each rung stands** (2026-09-18; a rung's column is filled in the pass that runs the gate, and an empty
cell means *not run*, not *passed*):

| gate | A1 DQN | A2 C51 | A3 QR-DQN | A4 IQN | A5 FQF | A6 Munchausen |
|---|---|---|---|---|---|---|
| 1 smoke, checkpoint, restore | ran (b35 trained and measured) | ran (b36, b37) | ran (b37) | ran (b38 is training and its stage A measures every checkpoint) | **passed 2026-09-18**: 5,000 steps, `ckpt-5000.pt`, restored through `evaluate.py fqf-smoke one` under the neutral and the `cvar:0.25` read; `watch.py` not run (no window while the laptop is in use) | **passed 2026-09-19**: M-DQN (α 0.9, τ 0.03, l₀ −1 on b35's local cell) and M-QR-DQN (the same on b37's N 32 cell) each ran 5,000 steps and checkpointed, at 458 and 124 counted steps/s alone on the laptop. Not logged: the log-policy term's size beside the reward -- by construction it is α·clip(τ log π, l₀, 0) ∈ [−0.9, 0] a step, up to 90% of a food reward and under 1% of the +100 win |
| 2 C51 support brackets the return, no end-atom mass | -- | ran with b36 (no clipping reported) | -- | -- | -- | -- |
| 3 mutation spec kills every mutant | `mut_seam.json` | **passed 2026-09-18**: `mut_dist.json` 28 / 28 killed, covering every rung's head | same run | same run | same run | same run (the three Munchausen mutants are in the spec) |
| 4 paper cell's ε ramp, shield 0, fork 1 read off the log | ran for b35's paper cell | b36 | -- (no paper cell ran) | -- | -- (no paper cell) | not run |
| 5 a 500k laptop arm reads a non-zero perfect rate | b35's local cell (90% by 0.6-1.6M) | b37 | b37 (onset 1.1-1.8M) | b38's stage A | **not run** (the laptop is in use, 2026-09-18); b39 itself is the gate -- a cell still at zero at 500k is stopped | not run |

The rows A1-A4 were queued before this table existed; their cells record what the batches themselves
showed rather than a separate gate run.


1. `PYTHONPATH=. SNEK_ALGO=<rung> SNEK_MAX_STEPS=5000 ... train.py smoke` runs, checkpoints, and the
   checkpoint restores through `evaluate.py smoke one` and `watch.py`.
2. For C51: `reward config:` is read off the log and the support brackets the discounted return range
   it implies; the smoke asserts no target mass clips to the end atoms on the prefill batch.
3. The mutation spec kills every mutant.
4. The paper cell's ε schedule is read off the log at three points and is the linear ramp asked for;
   the shield line reads 0 and the fork line reads 1 branch.
5. A 500k-step laptop arm reaches a non-zero perfect rate. A rung that cannot is not queued and the
   plan is revisited; the tuning budget for that is one laptop wave.

## 5. The eval decision this group needs: two greedy policies per checkpoint

A4's risk-sensitive read is a second greedy policy over the same weights. The eval protocol measures
"the checkpoint", and every shard loads through `restore.policy_fn_for(arch, net)`. The decision:

- **`arch.json` describes the network, not the acting rule.** The neutral policy is the default read of
  an `iqn`/`fqf` sidecar, so watching, recording and stage A behave as for any checkpoint.
- **A pass names the read.** `tools/closeout` and `evaluate.py` gain `--policy-variant cvar:0.25`,
  threaded to `restore.policy_fn_for(arch, net, variant=...)`; a result file carries `variant` in its
  header and `tools/results.py` names the file `<policy>_<pass>_cvar25.json` so it sits beside the
  neutral one and nothing overwrites it. The HOF row for a risk-sensitive result carries the variant.
- Stage A stays neutral. The training policy is neutral by definition, and stage A is what the epsilon
  schedule reads; changing it would change the training.

This is the same shape as the "no policy" reference form the fixed-path rows already use in the HOF,
and it is what G1's search-versus-network decision (`g-planning.md` §5) reuses.

## 6. What would change the plan

- **The paper cells trail the local cells everywhere.** The codebase's plumbing -- the shield, the fork,
  PER, the fast target -- is what makes value learning work on this game, not the head; the finding is
  written, the local cell becomes the base for A3-A6 and the paper cell is kept at one rung (A4) only.
- **A2 cannot be made stable within its stability batch.** Then A3 runs as the base of the ladder and
  the finding is written; QR-DQN has no support to mis-set and is the usual modern default anyway.
- **A4's CVaR read beats its own neutral read on the 30k top.** That is the group's headline and it
  changes B2 (Beyond the Rainbow acts neutrally; the plan would add the CVaR read to it) and the
  ordering (the CVaR read would be measured on every later value row).
- **The whole ladder is level with A1.** The distribution is not the lever, the tail diagnosis stands
  unexplained by value modelling, and Group D (memory) and Group G (planning) become the candidates.
