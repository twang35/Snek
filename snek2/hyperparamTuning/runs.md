# What is running, and what is next

The live file. Everything here is current state and forward plan — results and
conclusions live elsewhere so this stays short enough to actually keep accurate.

| file | contents |
|---|---|
| `runs.md` (this file) | what is running, what to run next |
| [`completedRuns.md`](completedRuns.md) | every arm that has finished: config, final numbers, verdict |
| [`findings.md`](findings.md) | what is established, what has been falsified |
| [`failureModes.md`](failureModes.md) | the four ways a policy degrades, and how to tell them apart |
| [`hyperparamTuning.md`](hyperparamTuning.md) | the protocol: metrics, how to judge, how to launch |
| [`charts.md`](charts.md) | progress graph per arm |

## The current record: two checkpoints at 96%, both really ~94%

**`b11b-obs30seed2` @855000 and `b14a-disc9975seed1` @3702000 both scored 96/100**, found by batch
11's close-out on 2026-08-04 and batch 14's on 2026-08-06. Both are preserved in
[`../hallOfFame/`](../hallOfFame/README.md) and both load on `master`.

**Neither is really a 96% policy, and they should be read as a tie near 94%.** Each is the maximum
over 176-204 full-length measurements in its own arm, so the headline carries selection luck:

| checkpoint | selected | corrected | how |
|---|---|---|---|
| `b11b` @855k | 96/100 | **~94.0%** | shrunk against a Beta prior on its 104 unselected graph-100% rows |
| `b14a` @3702k | 96/100 | **93.5%** (CI 89.2-96.2) | **re-measured** at 100 fresh episodes: 91/100, pooled 187/200 |

`b14a` is the first champion verified by re-measurement rather than modelling, and it is the better
method — it confirms the copy loads *and* sharpens the estimate. It became the only option because
the 90% abandonment gate destroys the unselected graph-100% rows the shrinkage prior needs; see
[`findings.md`](findings.md#three-measurement-caveats). Prefer a re-run for every future champion.

**The previous record, `b10d-disc995seed4` @1815000 at 95/100, no longer runs.** The two
observations added 2026-08-03 took the vector to 30 values, so it and every other batch-10
checkpoint fail to restore; **`450e66e` is the last commit with the 26-value vector.** That 95% was
measured mid-run over 30 episodes and shrinks to ~87%, and its own arm's close-out found 93%
@1695000 over a full 100 — treat those two as one policy family measured twice rather than a record
and a near-miss. Full batch results:
[`archive/batches1-11.md`](archive/batches1-11.md).

## Reference: the 2026-08-03 changes, and what they froze

Batch 10's own write-up moved to
[`archive/batches1-11.md`](archive/batches1-11.md) once batch 13 made batch 11 the baseline.
The two subsections below stayed here because they are not about batch 10 — they describe the eval
protocol and the observation vector every current arm runs on.

## Evals got ~10x cheaper on 2026-08-03

Two changes, both measured rather than assumed:

| change | effect |
|---|---|
| `EVAL_WORKERS` 2 → 10 | **2.8x** faster per checkpoint (103s → 37s) at *lower* CPU per episode |
| three-stage close-out (default) | **~2.4x** fewer episodes, with every graph-100% checkpoint fully measured |

The close-out protocol is now: every checkpoint whose training graph point was **100%** gets the
full 100 episodes immediately (uncapped — 47/142/7/146 across batch 10's arms), everything else
selected gets 20, and the best `EVAL_CONFIRM_COUNT` **of those screened** get 80 more.

**The 100% tier is coverage, not a champion shortlist.** Graph-100% checkpoints average 73.0%
against 71.2% for graph-90% ones, but the 90% tier holds the higher maximum (93% vs 89%) and
produced *all four* arms' best checkpoint, being ~4x larger. Finding the best checkpoint is the
confirm stage's job.

**`EVAL_CONFIRM_COUNT` is 100, raised from 30 on 2026-08-03.** At 30 the arm's true best non-100%
checkpoint reached the confirm set only **57%** of the time on b10d's 369 candidates — a coin flip
on the headline number. 100 takes that to **97%** and only moves the cost from 2.71x to 2.20x.

The worker-count finding is a correction: this project had concluded eval throughput was
core-bound past ~10 workers and dropped batch 10's close-out to 2 workers to hit a ~50% CPU
target. That made it 2.8x slower *and* cost more CPU per episode, because TensorFlow's thread
pool costs about a core whether its batch has 2 rows or 20. **To be gentler on the machine, run
fewer arms, not fewer workers.**

**Early abandonment was rejected in 2026-08-03 and shipped on 2026-08-05**, and the reversal is
about *which rule*, not new data. The rejected version cut a checkpoint once its running rate
**looked weak** — a predictive rule, which needs a safety margin to avoid discarding something that
would recover, and safe margins saved only 14% against batch 10's 937 measurements.
`EVAL_MIN_ACHIEVABLE` is instead an **arithmetic** rule: stop only when the remaining episodes
cannot carry the checkpoint to the gate even if all of them are perfect. No margin is needed, because
nothing that would reach the bar can be cut.

That the population is a tight blob between 60% and 80% is what *made* it work rather than what
killed it: the gate sits above the whole blob, so nearly all of it is out of contention early.
Measured on batch 13's first 505 full-length rows, full-length work drops to 71% / 52% / **31%** at
gates of 85 / 90 / 95. **The gate is 95**, since that is the bar a checkpoint has to clear to be worth
keeping at all; what it costs is in
[`hyperparamTuning.md`](hyperparamTuning.md#measuring-a-policy-properly-eval_checkpointspy).

A close-out is also resumable now (`EVAL_RESUME=1`), which is what made switching the worker
count mid-run cost only the checkpoint in flight rather than the 333 already measured.

## The environment changed again on 2026-08-03: two new observations

A fourth environment (‡‡‡ in [`completedRuns.md`](completedRuns.md)). The vector went from **26
values to 30**, in two steps.

#### Indices 26-28: following the tail

Three values, one per action, set to **0** when the move lands the head on the cell the tail is
vacating — the tail-chasing move — and 1 otherwise. 1 is good, per the project convention.

One wart of that direction: a **fatal** move also reads 1, since the flag only answers "is this the
tail's cell". Nothing is lost — indices 6-8 mark fatal moves, so the three cases are still
recoverable — but a 1 here on its own does not mean "this move is fine".

**The hypothesis.** Tucking in directly behind its own tail is safe forever and makes no
progress, and it is the same move that closes a pocket down one cell at a time from behind.
Keeping a cell of slack — pushing a bubble of open space along ahead of the tail rather than
closing on it — leaves room to manoeuvre later. Nothing in the vector named that move before:
indices 6-8 call it safe (it is), 9-14 call the tail reachable (it is), and 23-25 fire on it
exactly as they fire on travelling along a wall. Unvalidated — a hypothesis about what the
feature lets a policy express, not a measured effect.

Measured live so it is at least not a dead input, over 24k states of random-but-legal play: the
flag is available on **4.0%** of states, spread across all three actions (380/310/262). The rate
rises with snake length over the range random play reaches (3.9% at length 4-7, 5.7% at 8-11),
which is the direction the hypothesis wants, but random play cannot grow a snake far enough to
say more — that needs a trained policy on this vector, which does not exist yet.

#### Index 29: how cramped the food's space is

One value, not per action — the first observation to describe the food rather than the snake. **0**
when the food is sealed into a single cell, 0.5 when its open region is exactly two cells, 1 for
anything roomier or when there is no food. 1 is safe, per the project convention.

**The hypothesis.** Food sealed into a one- or two-cell pocket cannot be taken by approaching it.
The snake has to wait, or work elsewhere, until its own tail vacates a cell and opens the pocket —
push a bubble of space round to it. Nothing in the vector could express that: `food_observations`
gives direction and distance, which point straight at an unreachable meal, and `safe_to_chase_food`
collapses to 0 without distinguishing "sealed in one cell" from "reachable but the exit is bad".

**Polarity was flipped to the convention deliberately**, after briefly shipping the reverse. The
sign costs nothing in itself — the first layer absorbs it exactly (`w(1-x) + b` = `(-w)x + (b + w)`,
both unconstrained, no weight decay anywhere here) — so the whole vector reading one way is worth
more than the alternative.

The alternative had one advantage, now given up: the common case would sit at **0**. As shipped,
this input reads 1 in **99.95%** of states, which makes it very nearly a constant — it acts much
like a second bias, collecting gradient on almost every sample while carrying information on very
few. That is the shape of the `game_over` trap this project has already been bitten by. The effect
on learning is modest, since a bias absorbs a constant either way, but **do not assume this index's
weights were meaningfully trained** if it is ever repurposed.

**This is a rare-event feature, and how rare depends entirely on length.** Over 48,635 states of
random-but-legal play it left 1 only 14 times — 0.03% — because random play cannot grow a snake past
length ~11. Against a self-avoiding random-walk body as a proxy for a real one:

| snake length | rate below 1 (cramped) |
|---|---|
| 10 | 0.4% |
| 30 | 2.2% |
| 50 | 9.6% |
| 60 | ~26% |

So it is nearly silent early and common in the endgame — which is where an 80%-plus policy spends
its decisive moves, filling an 81-cell board. Treat those figures as a floor: real snakes coil more
tightly than a random walk. Unvalidated either way.

#### Consequences of both

Batch 10's checkpoints no longer load on `master`; `450e66e` is the last commit with a 26-value
vector. The "config vs environment" question `completedRuns.md` raises for batch 10 is now
permanently unanswerable in its original form — nothing isolates whether the 2026-08-02 fixes or
that seed cluster produced the 95%, and the environment has moved on again.

Batch 10's close-outs finished before this mattered, but the hazard is worth keeping in mind: an
eval process works only because it loaded the matching observation code at startup, so `EVAL_RESUME=1`
against a batch-10 arm would now build a 30-value network and fail to restore. **Batch 10's measured
numbers are final and cannot be extended.**

## Closed batches: 11-15 — all null, and what that means for the next one

Nothing is running. **Five consecutive batches on the current 30-value vector have failed to separate
from each other** — 20 arms across four different hypotheses. Per-batch write-ups are in
[`completedRuns.md`](completedRuns.md):

| batch | what it changed | verdict |
|---|---|---|
| [15](#batch-15-is-stopped-at-55-58m-awaiting-evals--n-step-returns-at-n_step_update3) | `N_STEP_UPDATE=3` | **falsified on speed** — 128k slower to pf30 ≥ 40%; level null |
| [14](completedRuns.md#batch-14--disc-09975-at-guided-08-and-the-widest-seed-spread-yet) | `DISCOUNT=0.9975`, `GUIDED_FRACTION=0.8` | null vs 13; `pooled_equal_effort` +0.01 pp |
| [13](completedRuns.md#batch-13--the-epsilon-rewrite-plus-the-exploration-shield-an-exact-null) | eps handover 0.0125 + shield 0.5 | null vs 11 on five metrics |
| [12](completedRuns.md#batch-12--the-deadlock-abandoned-at-1m-of-25m) | eps handover 0.05 | **deadlocked**, abandoned 4/4 |
| [11](archive/batches1-11.md#batch-11--the-same-config-on-the-30-value-vector-no-significant-difference) | the 30-value vector itself | +4 to +5 pp vs batch 10, not significant |

**The 0.995 baseline now has n=8** (batches 11 + 13), and batch 14 adds n=4 at 0.9975 that is indistinguishable from it.

**The binding constraint is seed variance, not ideas.** Batch 14's primary metric spread -16.2 to
+24.8 pp per seed around a +2.05 pp mean; batch 15's spread -9.7 to +22.9 pp. `b15a` is the best arm
on record for `strong_eval_fraction` (39.9%) and `b15c` among the worst (9.4%) — **same config, same
batch, adjacent seeds**. At n=4 nothing below ~10 pp is resolvable, so a fifth null is the expected
outcome of any knob whose true effect is a few points. That is the argument for the shorter-and-wider
batch described under
[what batch 13 leaves for the next batch](#what-batch-13-leaves-for-the-next-batch), not for another
n=4 sweep.

**‡ The ceiling has not moved at all, and that is the most important number here.** Peak trailing
score across the five batches reads 94.92 / 94.80 / 94.90 / 95.00 (b11, b13, b14, b15) — flat inside
0.2 points, even though run length went 3.2-3.6M → 3.4-3.7M → 4.1-4.5M → 5.5-5.8M. What *has* risen
is how long an arm stays good, which is why full-length `strong_eval_fraction` keeps climbing (19.5 →
21.6 → 30.5%) while the equal-effort figures stay flat. **That is a run-length artifact, not
progress**, and it is the honest reading of every "record" in the last three batches.

## Batch 15 is stopped at 5.5-5.8M, awaiting evals — n-step returns at `N_STEP_UPDATE=3`

The first batch that would actually measure n-step returns. Both existing n-step arms are retracted
rather than negative: `b1c-nstep3` and `b2b-nstep2` trained on returns that summed **straight
through episode boundaries**, because the per-step discount is the only truncation in
`r_t + g·d_t·r_{t+1} + g²·d_t·d_{t+1}·r_{t+2} + …` and terminal steps carried 0.9975 until
2026-08-02. Fixed at `snake_environment.py:126` (`discount = 0.0 if step_type == StepType.LAST`),
whose comment names this consequence outright.

### Why n=3 rather than n=2

The standard reason to keep n small is that an uncorrected n-step return is only exact if the
intermediate actions were greedy. **That cost is negligible at this project's epsilon.** Measured
mean epsilon over the back half of each arm: **0.0034** (batch 13), **0.0039** (batch 14).

| epsilon | P(non-greedy)/step | contaminated at n=2 | at n=3 |
|---|---|---|---|
| 0.0125 (refinement ceiling) | 0.83% | 0.83% | 1.66% |
| **~0.004 (what arms actually run at)** | **0.27%** | **0.27%** | **0.53%** |
| 0.002 (floor) | 0.13% | 0.13% | 0.27% |

So n=2 → n=3 moves contamination from ~0.3% to ~0.5% of targets, while the upside scales with n:
propagating the **+100 perfect-game reward** back across a ~1780-step perfect game takes ~890
sequential backups at n=2 against ~593 at n=3. Add that **n=4 arms resolve only a clear win**, so the
variant with the larger expected effect is the one to spend a night on, and that n=3 is Rainbow's
value and therefore the only choice here with outside evidence behind it.

The honest counter, unresolved: priorities come from the n-step TD error, so larger n feeds
larger-magnitude errors into `td_loss` + alpha 0.6, a combination already flagged as effectively
more aggressive than intended. It argues both ways — bigger errors push Huber into its linear
region, which *reduces* the distortion — so it is a thing to watch, not a reason to pick n=2.

### Config: `DISCOUNT=0.995` with `GUIDED_FRACTION=0.8`, control is batch 14

```
SNEK_SEED=1..4  SNEK_DISCOUNT=0.995  SNEK_GUIDED_FRACTION=0.8  SNEK_N_STEP_UPDATE=3
SNEK_PRIORITY_EXPONENT=0.6  SNEK_PRIORITY_SIGNAL=td_loss  SNEK_IS_WEIGHTS=0
SNEK_MAX_STEPS=10000000
```

`GUIDED_FRACTION=0.8` is now the standing value, so batch 15 onward all share it and stay mutually
comparable. That choice decides which control to read this batch against:

| control | differs from batch 15 by | usable? |
|---|---|---|
| **batch 14** | discount (0.9975 → 0.995) **and** n | **yes** — the discount is measured null, so n is the only live variable |
| batch 13 | `GUIDED_FRACTION` (0.5 → 0.8) **and** n | weaker — `GUIDED_FRACTION` has never been isolated at any discount |
| batches 13+14 pooled, n=8 | both of the above | for the primary metric, since 13 and 14 are indistinguishable |

**So judge batch 15 against batch 14, seed-matched**, and use the 13+14 pool for the primary where
the extra arms help. This works only because the discount question came back null: `0.995` and
`0.9975` are measured indistinguishable at n=4 paired, so swapping them is not expected to move
anything, whereas `GUIDED_FRACTION` 0.5 → 0.8 is genuinely unmeasured.

**‡ The residual risk, stated so it is not discovered later:** the discount null was measured *at*
`GUIDED_FRACTION` 0.8 vs 0.5, so it is a null on the pair, not on each knob independently. If the
discount and the shield interact, "0.995 ≡ 0.9975" need not hold at fixed `GUIDED_FRACTION=0.8`.
Nothing suggests they do — at epsilon ~0.003 the shield touches roughly 0.1% of steps, which is too
few events to plausibly interact with a horizon change — but it is an assumption this batch rests on.
If batch 15 comes back with a large effect, **check it against batch 14 before crediting n-step.**

**Do not also change the discount, and leave `GUIDED_FRACTION` at 0.8 from here on.**

**The 10M cap will not stop these arms in reasonable time** — batch 14 took 12.6 h to reach
4.2-4.5M, so 10M is roughly 30 h. Left at the default deliberately: it self-terminates, so an
unattended batch is safe, and `b14c` shows ~1 arm in 4 is still climbing at 4.5M. Stop them by hand
once the curves flatten.

### Stopped at 5.5-5.8M, awaiting evals — the primary says n=3 does **not** accelerate

Ran 2026-08-06 16:26 to 2026-08-07 08:22, **15.9 h**, four arms to 5.79M / 5.75M / 5.46M / 5.81M —
the longest arms on this vector by 1.3M. Stopped by hand well short of the 10M cap. No arm died.

**The pre-registered primary went the wrong way.** n-step's predicted mechanism was faster credit
propagation, read as steps to pf30 ≥ 40%:

| seed | b14 (control) | b15 | delta |
|---|---|---|---|
| 1 | 639k | 620k | -19k |
| 2 | 227k | **524k** | **+297k** |
| 3 | 530k | **707k** | **+177k** |
| 4 | 320k | 378k | +58k |
| **mean** | **429k** | **557k** | **+128k slower**, p=0.250 |

3 of 4 seeds slower. Not significant, but **the direction is against the hypothesis**, and there is
no reading of this batch in which n=3 propagated credit faster.

**Everything else is a fifth null**, equal effort to 4.125M against batch 14:

| metric | b14 | b15 | delta | p |
|---|---|---|---|---|
| `strong_eval_fraction` | 21.53% | 25.59% | +4.05 pp | 0.625 |
| mean perfect | 53.64% | 55.76% | +2.13 pp | 0.625 |
| mean perfect, back half | 62.39% | 63.01% | +0.62 pp | 1.000 |
| `best_perfect30` | 82.50% | 83.17% | +0.67 pp | 0.875 |
| mean trailing | 90.15 | 89.55 | -0.60 | 0.875 |
| peak trailing | 94.74 | 94.64 | -0.10 | 0.875 |
| drawdown | 23.75 pp | 20.42 pp | -3.33 pp | 1.000 |

**‡ The 280k interim read was half wrong, in the same direction as batch 14's.** It reported the
perfect rate down 11.9 pp; at equal effort it finished **up** 2.13 pp. What *did* survive is the early
deficit it was actually measuring — that is the +128k on the primary. So the lesson holds and gets
sharper: **an early snapshot of level reverses, an early snapshot of *timing* does not.** Read
crossings early, read levels late.

**Full-length numbers look much better than the equal-effort ones, and the gap is the point.**
`b15a` set a new peak trailing score of **95.00** (previous best on this vector 94.92, `b11b`) and the
best `strong_eval_fraction` on record at **39.9%**, with `b15b` at 39.0% — so batch 15 has two arms
above the 39.3% that was the previous single best. But `strong_eval_fraction` is a *fraction of an
arm's own evals*, and these arms ran 1.3-1.7M steps longer than batch 14's while still playing at
70-80% perfect. At matched steps the advantage is +4.05 pp at p=0.625. **The honest reading is that
the arms sustained a high level for 5.8M steps, not that n=3 raised the level.**

**‡ 2 of 4 arms were still gaining when stopped**, and this is now the third batch to say the horizon
is longer than assumed. Final 500k band against each arm's best prior band:

| arm | final band (5.5-6.0M) | best prior | |
|---|---|---|---|
| `b15a` | **80.6%** | 80.4% | still gaining |
| `b15d` | **75.8%** | 72.8% | still gaining |
| `b15b` | 62.3% | 78.5% | past peak |
| `b15c` | 53.7% | 59.7% | past peak |

`b15d`'s peak trailing score is at **5799k — its second-to-last eval**. Stopping here truncated two
arms mid-climb. The 10M cap was not the wrong setting; the habit of stopping at a round number is.

### Pre-registered: judge this on *speed*, not ceiling

n-step's predicted effect is faster credit propagation to the same asymptote, so the primary read is
**steps to pf30 ≥ 40%**, with `strong_eval_fraction` secondary. Batch 14 is the control to beat;
batch 13 is shown too because the pooled n=8 is what the primary metric uses:

| seed | **batch 14 (control)** | batch 13 |
|---|---|---|
| 1 | **639k** | 1525k |
| 2 | **227k** | 246k |
| 3 | **530k** | 807k |
| 4 | **320k** | 179k |
| **mean** | **429k** | 689k |

If n=3 helps, that milestone should arrive earlier than batch 14's 429k mean. Note how little room
there is on seeds 2 and 4 — 227k and 320k are close to the floor set by the bootstrap ladder itself,
so a speed gain will show up mostly on the slow seeds, and the mean is the number to read rather than
a count of seeds improved. **If it raises the ceiling without arriving
sooner, that is a surprise and worth writing up as one** — it would mean n-step is doing something
other than accelerating propagation here.

## What five nulls leave for batch 16

### What batch 13 leaves for the next batch

The epsilon axis is closed, so the next batch should spend its four slots on something else. The
binding constraint has not moved: **n=4 resolves ~10-15 pp on `best_perfect30` and ~7 pp on
`strong_eval_fraction`**, and the per-seed spread in this batch was -9.4 to +12.3 pp on an effect
that is genuinely zero. Candidates are in the backlog below, unchanged — `SNEK_LEARNING_RATE=1e-4`
is still the highest-value untested knob.

One thing worth doing first and cheaply: **batch 13's arms are four fresh seeds on the current
config and environment**, so pooled with batch 11 they give **n=8 for the baseline** rather than a
treatment group. That is the widest baseline this project has had, and it is what any future knob
test should be compared against.

**Honest statement of the bet.** The rewrite's original premise — that batches 10-11 running 96.8%
of steps at epsilon exactly 0 was itself a defect — has one falsified prediction against it and no
evidence for it. The part that survives is the *ratchet* being a real defect. So the null
hypothesis here is that b11's near-zero regime is simply correct and every version of this schedule
is a wash at best. 0.0125 is a bet placed to find that out cheaply.

### Why shorter and wider, and what it actually buys

**Three of batch 11's four arms peaked before 1.8M**, then spent 1.5-2.5M further steps getting
worse, so a 10-hour run spent most of its compute past the point of interest. Capping near 2.5M
costs about a third per arm and buys seeds instead — and seed count is the binding constraint.

Be precise about the gain, because an earlier version of this section was not: **n=12 detects ~5 pp
only on a low-variance metric.** On `best_perfect30` it is 8.7 pp, and 5 pp there would need ~37
arms per group, which is not reachable. On `strong_eval_fraction` n=12 gives 5.9 pp and n=8 gives
7.2 pp. See the table in
[`hyperparamTuning.md`](hyperparamTuning.md#the-primary-metric-strong_eval_fraction-the-share-of-an-arms-evals-at-80).

Open question the cap does not answer: whether arms that peak at ~700k would have gone higher with
more steps, or whether the peak is the ceiling. `b11d` peaked at 3468k and was the only arm still
near its peak when stopped, so a 2.5M cap risks truncating that kind of arm — worth letting one or
two run on past the cap once the wave's comparison data is in hand.

### Later candidates

Deferred pending the above. The gate on all four is the same and it is not a code change any more —
it is that **n=4 cannot see an effect this size**, which batch 11 has now demonstrated three
different ways. Running any of these at n=4 would produce another unreadable batch.

| change | why | gate |
|---|---|---|
| second discount value (`0.9975`, or an interior point like `0.996`) | batch 9 left 0.995 vs 0.9975 unsettled; whether that survives the fourth environment is unknown | needs the wider design |
| `SNEK_LEARNING_RATE=1e-4` | the highest-value untested knob | needs the wider design |
| eff exponent ~1.4 (`td_loss` alpha 0.7) at 0.995 | `b4c` and `b7f` tie on ceiling; sharpness may still add on top of the discount | needs the wider design |
| best config + `REPLAY_BUFFER_MAX_LENGTH=500000` | `b4b` beat `b4a` slightly, so diversity may stack | needs the wider design |
| partial IS correction (beta < 1) | full correction cost `b5c` almost everything (2.1%); partial may keep stability without the cost | needs a new knob |
| anything aimed at the post-peak decline | both batch-8 arms peaked at ~2.5-3M and fell away; nothing tried so far addresses *why* | needs a mechanism first |

**Seed count is the binding constraint, not the number of knobs tried.** Six single-seed
conclusions in this document have been overturned or weakened — most recently gradient
clipping, which looked like batch 8's headline twice before failing at 1 of 3. Nothing goes in
[`findings.md`](findings.md) as established without n=3, which is why batch 10 spent all four
slots on one value rather than splitting them.

## Standing backlog

Untested, ordered by expected value. Rationale for the ones that need it follows the
table.

| change | targets | prior |
|---|---|---|
| `LEARNING_RATE=1e-4` | training speed | high, but order it after a stability fix |
| `TARGET_UPDATE_PERIOD=50` / `500` | early learning speed | medium — 2 points to test a hinted trend |
| `TARGET_UPDATE_TAU=0.005`, period 1 | smoothness (soft target updates) | medium |
| `FC_LAYERS=128,128` | capacity | low |
| ~~epsilon ladder *shape*~~ | exploration schedule | **done 2026-08-04** — rewritten, needs measuring |
| `REPLAY_BUFFER_MAX_LENGTH=1000000` | experience diversity | low — the 500k result was ambiguous |

**`LEARNING_RATE=1e-4` — only after a stability fix.** 1e-5 is very conservative and
the in-code comment already suggests 1e-4. With a stable target it may train several
times faster; on its own with `TARGET_UPDATE_PERIOD=8` it would probably make
instability worse. The order matters.

**`TARGET_UPDATE_PERIOD=50` and `=500`.** Batch 1 hinted that longer periods learn
faster early even though they didn't reduce drawdown. Two more points establish
whether that is a trend or noise. Note `b1b-tgt200` was stopped at 104k, well short of
the ~250k horizon, so that hint is weak evidence.

**Epsilon ladder shape — rewritten 2026-08-04, and it is now the highest-value untested
change on this page.** The old ladder ran **96.8% of batches 10-11's training steps at epsilon
exactly 0.0**, bottoming out at median step 15000 while 7 of 8 arms were still at 0% perfect
games, because its rungs were calibrated to `avg_reward` values a non-winning policy clears.
Replaced with two phases — `avg_reward` bootstrap to `INITIAL_EPSILON`/8, then a geometric
descent driven by the trailing-30 perfect rate — neither a ratchet, floor 0.002, and exactly 0
rejected at startup. Design and the measured diagnosis:
[`hyperparamTuning.md`](hyperparamTuning.md#the-epsilon-schedule--rewritten-2026-08-04-and-it-breaks-curve-comparability).

Two consequences for planning. **Learning curves are no longer comparable to batches 1-11** —
checkpoints still load, but every earlier arm trained greedily from step ~15k, so graph shapes
are not like-for-like. And because the refinement phase is a pure function of current skill, a
declining arm now automatically explores more (`b11a` would have gone 0.0020 → 0.0087 across its
42pp drawdown), which makes this the first change in the backlog aimed at the post-peak decline
rather than at the ceiling.

## Explicitly not planned

- **Reward changes** — they would break comparability of `avg_score` with every run
  recorded so far.
- **Reverting to `PyUniformReplayBuffer`** — cpprb is ~2.4x faster with no measured
  learning cost, so cheaper experiments come from keeping it.
- **An LR schedule** — no evidence of optimization instability; degradation is gradual
  in every arm, not spiky.
- **Adding more epsilon knobs** — the schedule was rewritten 2026-08-04 and already exposes
  `INITIAL_EPSILON` and `MIN_EPSILON`; the thresholds, windows and the 80% target are constants
  in `training.py` on purpose. Measure the new schedule before making any of them tunable, or
  the next batch will vary four things at once.
- **Setting `SNEK_MIN_EPSILON=0`** — rejected at startup. See
  [`findings.md`](findings.md#scope-of-that-falsification-added-2026-08-04-it-was-never-about-the-descent-rate)
  for why the batch-3 result does not license it.
- **`N_STEP_UPDATE=5`** — batch 15 measured n=3 and the predicted mechanism is absent: it reached
  pf30 ≥ 40% **128k later** than its control, 3 of 4 seeds slower, with level a null. The whole case
  for a larger n was that the effect scales with n, so its absence at n=3 is the worst possible sign
  for n=5. Contamination was never the obstacle either — 0.53% of targets at n=3, 1.06% at n=5 — so
  there is no cost to blame the null on. **Closed unless the propagation story is revived by
  something other than n.**
- **Resuming any arm from batch 10 or earlier** — every checkpoint on record from before
  batch 11 was trained on an observation vector this project has since changed (20, 23 or 26
  values against the 30 batch 11 trained on), so none of them load; see
  [`../hallOfFame/README.md`](../hallOfFame/README.md#the-entries-below-predate-2026-08-02-and-do-not-run-on-master).
  **Batches 11 onward are all resumable** — 11-15 share the 30-value vector. The arms worth resuming
  are the ones stopped mid-climb: **`b15a` and `b15d`, both still gaining in their final 500k band at
  5.5-6.0M**, then `b14c`, `b11d` and `b13c`. Resuming needs `SNEK_MAX_STEPS` raised above the arm's
  current step or it exits immediately.

### Batch bookkeeping

Each batch keeps its **description** — why it is shaped that way, what each arm isolates,
what outcome would mean what — in this file for as long as any of its arms is running.
When the last arm of a batch stops, move that description and its results to
[`completedRuns.md`](completedRuns.md) and delete it here.

The reason to keep the description live rather than only the status table: the design
rationale is what tells a future session whether a surprising result is informative or
just an arm that was never going to answer anything.

Verify what's actually running with `pgrep -fl "python -u snek2.py"`. Not
`grep "[s]nek2.py"` — git telemetry `curl` processes carry `snek2/snek2.py` in their
payload and inflate the count.
