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

## The current record: three checkpoints read 96-97%, and all three are really ~93-94%

`b15b-nstep3seed2` @3245000 read **97/100**, the highest selected measurement this project has
produced. It does not move the record. **Every champion, corrected, lands in the same place:**

| checkpoint | selected | corrected | how |
|---|---|---|---|
| `b15b` @3245k | **97/100** | **93.0%** (CI 89.5-95.4) | re-measured over 200 fresh episodes: 182/200, pooled 279/300 |
| `b14a` @3702k | 96/100 | 93.5% (CI 89.2-96.2) | re-measured: 91/100, pooled 187/200 |
| `b11b` @855k | 96/100 | ~94.0% | shrunk against a Beta prior on its 104 unselected graph-100% rows |

All three are in [`../hallOfFame/`](../hallOfFame/README.md) and all three load on `master`. Their
intervals overlap almost entirely, so **the honest statement is that five batches have produced the
same ~93-94% policy three times**, not a rising record.

**Re-measure, do not shrink.** Re-measurement confirms the copy loads *and* sharpens the estimate,
and it is the only option left for a gated close-out, since a gate destroys the unselected
graph-100% rows the shrinkage prior needs — see
[`findings.md`](findings.md#three-measurement-caveats).

**‡ A high selected reading is mostly sampling luck, and the numbers say how much.** `b15b`'s 94
full-length rows have **mean 90.7% and median 90.0%**. For a population centred at 90%, a 100-episode
measurement reads ≥95 about 5.7% of the time, so ~5.4 of 94 rows should hit ≥95 by noise alone;
8 did. **`EVAL_MIN_ACHIEVABLE=95` therefore does not find 95% policies — it finds ~90% policies
caught on a good 100 episodes.** Nothing but a second measurement separates the two.

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

**Five consecutive batches on the current 30-value vector have failed to separate from each other** —
20 arms across four different hypotheses, every one of them an optimiser knob. That is the reason
batch 16 goes after the reward function instead. Per-batch write-ups are in
[`completedRuns.md`](completedRuns.md):

| batch | what it changed | verdict |
|---|---|---|
| [15](completedRuns.md#batch-15--n_step_update3-falsified-on-speed-null-on-level-and-a-97100-that-is-really-93) | `N_STEP_UPDATE=3` | **falsified on speed** — 128k slower to pf30 ≥ 40%; evals null, best ckpt +0.05 pp |
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

## Stopped 2026-08-07, close-out pending: batch 16 — the food-distance shaping term, ablated

**Launched 2026-08-07, four arms, `SNEK_FOOD_DISTANCE_REWARD=0`; stopped by hand ~1.25M the same
day.** The first reward change this project has made, and the first batch to test a piece of
hand-designed guidance rather than an optimiser knob. The description stays here until the close-out
runs and its numbers move to [`completedRuns.md`](completedRuns.md).

`FOOD_DISTANCE_REWARD` subtracts 0.001 on every ordinary move that increases the head's Manhattan
distance to the food. It has been on since batch 1 and has never been measured. The case for taking
it away:

- **It rewards the wrong skill.** Approaching food greedily is what a beginner needs and what a 93%
  policy must sometimes refuse — the endgame is about keeping space and following the tail, and a
  detour is often the only safe move. A term that prices every detour at 0.001 is a small permanent
  push against exactly the behaviour the ceiling depends on.
- **It is a bias with no measured benefit.** Its only defence on record is that it was there. Every
  other shaping decision in this project has been measured; this one predates the measurements.
- **The ceiling has not moved in five batches** while every knob tried was in the optimiser. The
  reward function is the part nobody has looked at.

The counter, stated up front: shaping may be load-bearing *early*, where the food signal is sparse
and a random policy needs a gradient to find the first few pieces. If so, the arms will be visibly
slower to leave the bootstrap phase, and that is a real result rather than a failure.

### Config: batch 14's config exactly, minus the shaping

```
SNEK_SEED=1..4  SNEK_DISCOUNT=0.9975  SNEK_GUIDED_FRACTION=0.8  SNEK_FOOD_DISTANCE_REWARD=0
SNEK_PRIORITY_SIGNAL=td_loss  SNEK_IS_WEIGHTS=0
```

**The control is batch 14, and for once it is an exact one** — same four seeds, same discount, same
shield, same priority settings, `N_STEP_UPDATE=1` in both. One variable, and no reliance on a
previous null to make the comparison legitimate. That is what makes this batch worth running at n=4
despite the resolution limit: the difference between two configs is a single reward term.

Batch 15 is *not* the control — its `N_STEP_UPDATE=3` is falsified, so it differs by two things.

### Pre-registered: two reads, and one of them is nearly free

**1. Steps to leave the bootstrap epsilon phase** — the first eval whose epsilon reaches the
refinement ceiling of 0.0125, which happens when trailing `avg_reward` clears the last bootstrap
threshold of 20. This lands in the first ~25k steps, so it is measurable within minutes of launch,
and it is the direct test of "was the shaping scaffolding for early learning".

| batch | handover step per seed (1/2/3/4) | mean |
|---|---|---|
| **14 (control)** | **11k / 10k / 9k / 20k** | **12.5k** |
| 13 | 10k / 10k / 10k / 16k | 11.5k |
| 15 (n=3, not a control) | 20k / 22k / 15k / 19k | 19.0k |

**‡ Part of any speed-up here is arithmetic, not learning.** The bootstrap thresholds are on
`avg_reward`, and removing a penalty raises `avg_reward` for identical play, so a shaping-off arm
clears them slightly sooner by construction. The size of that shift is measurable: with shaping off
`avg_reward - avg_score` is purely the death and starve penalties, so comparing that gap against
batch 14's at matched score isolates it. **Subtract it before crediting the change.**

**2. Impact on training** — `pooled_equal_effort` and `strong_eval_fraction` at equal effort against
batch 14, seed-matched, plus steps to pf30 ≥ 40% as batch 15's primary is still the best available
speed metric. These need the arms to run out, and at n=4 only a ~10 pp swing is resolvable.

What each outcome means:

| result | reading |
|---|---|
| handover unchanged, level unchanged | the shaping is **inert** — 0.001 is too small to matter either way, and it should come out for simplicity |
| handover unchanged, level up | the shaping was a **drag on the ceiling** without being early scaffolding. The most interesting outcome, and the one the hypothesis predicts |
| handover later, level unchanged or up | it was **early scaffolding only** — worth replacing with something that anneals off rather than deleting |
| handover later, level down | it is **load-bearing**. Put it back and note that hand-designed guidance still earns its place here |

**The close-out is the first one at `EVAL_MIN_ACHIEVABLE=95`.** Batch 14 was measured at 90, so
best-checkpoint stays comparable — anything at or above a gate is measured full length under it — but
**the graph-100% tier is not**, and batch 16 will have far fewer full-length rows. Read
`pooled_equal_effort`, which is exact at any gate.

### Answered at 70k steps: the shaping is not early scaffolding

**Read 1 is in, and it is a flat null.** Handover to the refinement phase, seed-matched against
batch 14:

| seed | b14 (shaping on) | b16 (shaping off) | delta |
|---|---|---|---|
| 1 | 11k | 11k | 0 |
| 2 | 10k | 10k | 0 |
| 3 | 9k | 9k | 0 |
| 4 | 20k | **16k** | **-4k** |
| **mean** | **12.5k** | **11.5k** | **-1k** |

Three seeds land in the *same eval*, and the whole difference is seed 4 — the same seed that was the
outlier under shaping. Evals are 1000 steps apart, so "same" means within one eval; the honest
statement is that this measurement cannot see a difference. The sustained form (5 consecutive evals
at the ceiling) is a wash in the other direction: 22k / 14k / 13k / 20k against batch 14's
15k / 14k / 13k / 24k.

**And the arithmetic confound is measured, not just acknowledged — it is worth ~63 steps.** Two
numbers settle it:

| quantity | measured |
|---|---|
| what the shaping cost per episode near the threshold | **0.47 reward** (`avg_reward - avg_score` is -1.168 with shaping, -0.696 without, over evals at score 15-25 with no perfect games) |
| how fast trailing reward rises through the threshold | **7.4 reward per 1000 steps** (mean local slope across all 8 arms) |

0.47 / 7.4 → **≈63 steps of head start, 6% of one eval interval.** So the null is not the artifact
cancelling a real effect; there was almost no artifact to cancel.

**What this rules out:** the shaping was not providing a dense gradient the early policy needed. Arms
find their first food and climb to `avg_reward` 20 at the same rate without it. That leaves the
level question — whether the term was quietly costing the ceiling — which is what the arms are still
running for.

### At 500k: the speed metric is a null too, but the seed spread collapsed

All four arms have crossed pf30 ≥ 40%, so the **primary speed metric is readable already** — and this
is the one kind of early read this project trusts, since a crossing cannot un-happen the way a level
can (see batch 15's interim lesson).

| seed | b14 (shaping on) | b16 (shaping off) | delta |
|---|---|---|---|
| 1 | 639k | 450k | **-189k** |
| 2 | 227k | 400k | +173k |
| 3 | 530k | 379k | **-151k** |
| 4 | 320k | 465k | +145k |
| **mean** | **429k** | **424k** | **-5k**, p=0.875 |

**-5k on a 429k baseline is as flat as this metric gets.** 2 seeds faster, 2 slower.

**‡ The interesting number is the spread, not the mean.** Batch 14's crossings span 227-639k
(sd 189k); batch 16's span 379-465k (sd 41k) — a **4.6x tighter** spread on the metric whose seed
variance has been called this project's binding constraint five batches running. If that survives the
close-out it is worth more than any of the level comparisons, because it would mean the shaping term
was a *variance* source: a small permanent bias whose cost depends on how often a given seed's food
placements punish a detour.

**Treat it as a hypothesis, not a result.** Four samples per group cannot establish a variance
difference — an F-test on sd 189k vs 41k at n=4 is nowhere near significant, and batch 14's spread is
driven mostly by one slow seed. The way to settle it is more seeds at this config, which is exactly
the shorter-and-wider design already argued for below. **Do not put it in
[`findings.md`](findings.md) on this evidence.**

### Stopped at ~1.25M: the ceiling is flat again, the level read waits on the close-out

All four stopped by hand at ~1.25M — much earlier than batch 14/15's 4.2-5.8M, a deliberately shorter
run. Trailing had plateaued in the low-94s and every arm had peaked (816k-1198k), but that is a
shorter horizon than the controls, so the level metrics below are **not comparable to batch 14/15 at
face value** — only the close-out's equal-effort figures against a batch-14 slice at matched steps
will be.

| seed | step | peak trailing | best-30 | `sef` | recent-30 | note |
|---|---|---|---|---|---|---|
| 1 | 1245k | 94.82 @837k | **87.0%** @850k | 20.6% | 77.0% | peaked early, flat since |
| 2 | 1261k | **94.98** @816k | 85.0% @919k | **30.7%** | 79.0% | strongest and flattest |
| 3 | 1257k | 94.36 @**1198k** | 72.7% @1221k | 10.6% | 67.0% | latest peak; recovered from a mid-run dip |
| 4 | 1256k | 94.68 @946k | 73.0% @1032k | 7.2% | **55.7%** | weakest on consistency, mid-pack ceiling |
| **mean** | | **94.71** | **79.4%** | **17.3%** | | |

**‡ The ceiling did not move — a sixth flat batch.** Mean peak trailing 94.71 sits inside 0.3 pp of
the 94.8-95.0 band batches 11-15 hold. Removing the shaping neither raised nor lowered the peak, which
is the level read the training graph *can* give; the eval-level read is what the close-out adds.

**Still to do for this batch:** the close-out at `EVAL_MIN_ACHIEVABLE=95`, seed-matched against a
batch-14 slice truncated to ~1.25M for the equal-effort comparison. `charts.md` entries are done.
Running the close-out will displace every top-level chart in `evals/` — no finished arm's charts are
at risk there right now, but restore from `evals/archive/` afterward if any appear.

## Next up: batch 17 — forked endgame collection (`SNEK_FORK_*`)

**Built 2026-08-07, off by default, not launched.** It is the first change in this document aimed at
the *collect distribution* rather than at the optimiser or the reward, and it was moved ahead of the
target-period batch deliberately — the mechanism is novel here, so it is the more interesting thing
to learn from first.

**The premise it was proposed on is falsified; the one it survives on is narrower.** The idea was that
the endgame is never explored at epsilon ~0.003, so the buffer holds no endgame experience. Measured
across all 20 current-era arms' saved buffers, that is not true:

| arm group | buffer at len ≥ 80 | at len ≥ 90 | collected episodes ending perfect |
|---|---|---|---|
| batches 11, 13, 14, 15, 16 (eps ~0.003) | **20-34%** | 9-21% | **12-81%** |
| batch 12 (eps 0.05, the deadlock) | **0.0%** | 0.0% | **0 of 3142** |

Batch 12 is the calibration: the metric reads exactly zero when the endgame really is missing, so the
20-34% is not a floor artifact. `findings.md`' quote that "the buffer holds no trajectories that eat
the last ~10 food" is a true description of **batch 12**, and false of every arm since.

**What survives.** At an endgame decision point the buffer holds the consequence of the action taken
and **never** that of the alternative — measured mean **2.06 safe actions** per eligible state at
length ≥ 85, so ~1.06 per state are never tried. An arm that dies from state `s` learns that
`Q(s, a_bad)` is low, but nothing raises `Q(s, a_good)` for the action it did not take, so the argmax
has no reason to flip. **The hypothesis to pre-register is counterfactual coverage at endgame decision
points, not endgame volume** — naming it that way is what makes a null closable.

Branch points are also **not rare**, which the original design assumed: ≥ 2 safe actions on **42-45%**
of steps at length 80-84, falling to **9-11%** at 95-100 — about **74-104 eligible states per episode**
at length ≥ 85. "Only a few points" holds above ~95, not from 85.

### Pre-registration

**Launch once batch 16's close-out lands**, because that verdict fixes one setting this batch has to
inherit: whether `SNEK_FOOD_DISTANCE_REWARD` stays at 0 (if the shaping ablation is null or better) or
goes back to 0.001 (if the shaping turns out load-bearing). The control has to match whichever it is.

| item | value |
|---|---|
| config | `SNEK_FORK_BRANCHES=4 SNEK_FORK_PROB=0.5 SNEK_FORK_MIN_LENGTH=85 SNEK_FORK_MAX_STEPS=60`, plus batch 16's settled shaping value |
| control | seed-matched arms at `SNEK_FORK_BRANCHES=1`, same everything else — **batch 16 itself is that control** if the shaping stays off, which saves four slots |
| primary | `strong_eval_fraction` at a common `global_step` horizon, paired, exact permutation over 16 sign flips |
| early read | steps to pf30 ≥ 40% — a crossing, which is the read this project trusts early |
| mechanism metrics | buffer share at len ≥ 80 (baseline **20-34%**), terminal and endgame-death adds per 100k, and the branch share of *sampled* batches |
| abandon at 100k | if the branch share sits outside ~30-60% of collect, or the len ≥ 80 buffer share is not clearly above the control's |

**‡ It is not a one-knob test, and the write-up must say so.** It changes what is collected, how much
per game, and how priority mass is distributed. The three consequences, with sizes, are tabulated in
[`hyperparamTuning.md`](hyperparamTuning.md#forked-endgame-collection--snek_fork_). The one most likely
to produce a *negative* result: branch transitions enter at max priority and are high-TD-error by
construction, and the standing config runs `td_loss` with `IS_WEIGHTS=0`, so nothing corrects the
resulting over-sampling.

**Verified so far, without a batch:** 69 unit tests across `test_game_snapshot.py`,
`test_replay_streams.py` and `test_forking_collector.py`, all mutation-checked; the off path is
byte-identical to pre-change code over 3000 steps on three seeds; and a 10k-step forking smoke run
produced **451 forks at 73% branch share with 0 violations** of the buffer integrity invariant
(length rises by 0 or 1 per frame and the starve budget decrements exactly, within every stored
window), against 0 violations in 400k pairs from three unforked arms.

## After that: batch 18 — `TARGET_UPDATE_PERIOD` 1000

**Not launched.** Designed 2026-08-07, and swapped behind the forking batch on 2026-08-07 so the
branching idea gets measured first. Nothing about the design changes with the reorder: its control is
whatever the batch before it settles, which is now batch 17 rather than batch 16.

`TARGET_UPDATE_PERIOD=8` with `TARGET_UPDATE_TAU=1.0` means the target network is hard-copied from
the online network every 8 gradient steps. **Standard DQN uses hundreds to thousands**, and at 8 the
target is barely a target at all — it tracks the online network closely enough that the bootstrap
target `r + γ·max Q_target(s')` is nearly self-referential, which is the textbook recipe for
oscillation. It has been 8 since batch 1 and has been varied exactly once.

### Why 1000, and why one value rather than two

| option | verdict |
|---|---|
| **1000 at 4 seeds** | **run this** |
| 500 at 4 seeds | the fallback if 1000 overshoots — 500 is 2.5x the only value ever tested, 1000 is 5x |
| 2 seeds at 500 + 2 at 1000 | **do not** — n=2 resolves nothing here, and the result would be two unreadable halves |

The reason to spend the batch on the larger value: **`LEARNING_RATE=1e-5` makes a long period cheap.**
Target staleness is the online network's *drift* over the period, not the period itself, and at
lr 1e-5 with batch 128 the online network moves very slowly — so 1000 steps of drift is modest where
it would be reckless at 1e-4. The knob with the bigger expected effect is the one worth a night, the
same argument that picked n=3 over n=2 in batch 15.

**And it unblocks the highest-value item in the backlog.** `LEARNING_RATE=1e-4` is gated on "after a
stability fix", and a long target period *is* that fix. If 1000 is neutral-or-better, the batch after
it is `TARGET_UPDATE_PERIOD=1000` + `LEARNING_RATE=1e-4`; if 1000 is harmful, that ordering is dead and
the soft-update route (`TARGET_UPDATE_TAU=0.005` at period 1) is next instead.

### The one prior data point, and what it actually says

`b1b-tgt200` (`TARGET_UPDATE_PERIOD=200`, batch 1) predicted smoother curves and smaller drawdowns.
It got the **opposite on drawdown** — 27.4 against the baseline's 19.2 — and what it did do was learn
**much faster early**: score ~55 by 15k steps where `b1a-base` needed ~25k, and an earlier first
perfect game. Full write-up in
[`archive/batches1-11.md`](archive/batches1-11.md#target_update_period200-hypothesis-not-supported-but-interesting).

Two reasons that is a weak hint rather than a finding: it was **stopped at 104k**, well short of the
~250k minimum horizon, and it ran on the 20-value vector under the old epsilon ladder with none of
the current replay or shield settings. Treat "longer learns faster early" as the hypothesis and
"longer worsens drawdown" as the risk to watch, neither as established.

### Config, control and pre-registration

```
SNEK_SEED=1..4  SNEK_TARGET_UPDATE_PERIOD=1000  SNEK_DISCOUNT=0.9975  SNEK_GUIDED_FRACTION=0.8
SNEK_PRIORITY_SIGNAL=td_loss  SNEK_IS_WEIGHTS=0
```

**The control is batch 17**, seed-matched, with forking off and the shaping value batch 16 settles on.
That is the whole reason for the ordering rule: each batch's control is the one before it, so the two
knobs stay separable. **Do not launch this until batch 17 is closed out** — running it against batch
16 instead would leave the target period confounded with forking.

`SNEK_GUIDED_FRACTION=0.8` is shown above for the record; it is the default from 2026-08-07, so
passing it changes nothing.

**Primary metric: steps to pf30 ≥ 40%**, the same speed metric batch 15 pre-registered, because the
live hypothesis is about *early* learning. Secondary: `pooled_equal_effort` and
`strong_eval_fraction` at equal effort. **Also report max drawdown explicitly** — it is the one
number the prior data point predicts will get worse, so it needs to be read whatever the primary
does, rather than looked up afterwards if the batch disappoints.

Risk to check on day one: an arm that learns too slowly never clears
`SNEK_MIN_CHECKPOINT_SCORE=40` and so writes no checkpoint, which makes it unresumable and
unmeasurable. Batch 1's hint points the other way, so this is a glance at the first few evals rather
than a real concern.

## What five nulls leave for the batch after this one

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
| ~~`TARGET_UPDATE_PERIOD`~~ | early learning speed, target stability | **promoted to batch 18** — see [the design](#after-that-batch-18--target_update_period-1000) |
| `TARGET_UPDATE_TAU=0.005`, period 1 | smoothness (soft target updates) | medium |
| `FC_LAYERS=128,128` | capacity | low |
| ~~epsilon ladder *shape*~~ | exploration schedule | **done 2026-08-04** — rewritten, needs measuring |
| `REPLAY_BUFFER_MAX_LENGTH=1000000` | experience diversity | low — the 500k result was ambiguous |

**`LEARNING_RATE=1e-4` — only after a stability fix.** 1e-5 is very conservative and
the in-code comment already suggests 1e-4. With a stable target it may train several
times faster; on its own with `TARGET_UPDATE_PERIOD=8` it would probably make
instability worse. The order matters.

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

- ~~**Reward changes**~~ — **retracted 2026-08-07, and the stated reason was wrong.** The claim was
  that changing a reward breaks comparability of `avg_score`. It does not: `avg_score` is a count of
  food eaten, so it and every eval metric derived from it are on the same scale whatever the rewards
  are. Only `avg_reward` changes scale. `FOOD_DISTANCE_REWARD` is now tunable and batch 16 is testing
  it; `FOOD_REWARD`, `DEATH_REWARD`, `STARVE_REWARD` and `PERFECT_GAME_REWARD` remain fixed, since
  those *do* rescale `avg_reward` enough to move the bootstrap epsilon thresholds a long way. See
  [`hyperparamTuning.md`](hyperparamTuning.md#available-knobs).
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
