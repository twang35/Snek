# Archived runs.md detail

**History only — do not read into context.** Moved out of `runs.md` on 2026-08-08 to keep that file
to current state and forward plan. Contains the record re-measurement derivations, the 2026-08-03
eval/observation reference notes, and the full batch 11-18 write-ups (per-seed numbers also in
[`../completedRuns.md`](../completedRuns.md)). Links from `runs.md` and `completedRuns.md` land here.

## ‡ The record is now ~95%, and the 99/100 that suggested it was mostly selection

**Re-measured 2026-08-08 over 6,600 fresh episodes.** `b17b-forkseed2`'s close-out produced a 99/100
and 25 of 26 full-length rows at ≥95%, which looked like a jump to ~96%. It is not. Every one of the
top rows shrank:

| step | close-out | re-measured, 500 fresh episodes | change |
|---|---|---|---|
| 1190k | 98/100 | **95.0%** (CI 92.7-96.6) | **-3.0** |
| 1205k | **99/100** | **92.4%** (CI 89.7-94.4) | **-6.6** |
| 1231k | 98/100 | 92.6% (CI 90.0-94.6) | -5.4 |
| 1248k | **99/100** | 93.8% (CI 91.3-95.6) | -5.2 |
| **mean** | **98.5%** | **93.45%** | **-5.05** |

**The standing rule was right and I argued against it wrongly** — see
[the reasoning error](#-the-reasoning-error-a-selected-sample-cannot-defend-itself-against-selection)
below, which is the more valuable thing on this page than the number.

**What the record actually is.** `b17b` @1190000, **94.24% over 5,120 fresh episodes** (4825/5120, CI
93.6-94.8). It ended up the most heavily measured checkpoint in the project as a by-product of
validating the independent-worker change, which meant measuring one checkpoint thousands of times.

| | value |
|---|---|
| record checkpoint | `b17b-forkseed2` @1190000 |
| measured | **94.24%**, 4825/5120 fresh episodes, CI **93.6-94.8** |
| previous | `b14a` 93.5% /200, `b15b` 93.0% /300, `b11b` ~94% shrunk |
| honest summary | **~94%, level with the old record once measured properly** |
| steps to get there | **1.19M**, against 3.7M and 3.2M — **the only unambiguous improvement** |

**‡ At 5,120 episodes the level gain mostly disappears.** The earlier 600-episode read of 95.17% sat
above `b14a`'s 93.5%/200 and `b15b`'s 93.0%/300; at ±0.6 pp the honest figure is **94.24%**, which
overlaps both. **So the ceiling still has not moved** — seven batches, and what changed is that this
arm reached the same frontier in a third of the steps. Read the speed, not the level.

**‡ The ordering inside the top rows was pure noise.** The four rows at ≥98% re-measure to **93.45%**
mean, while the six at exactly 97% re-measure to **95.25%** — the group that originally looked *worse*
measured *better*. At n=4 and n=6 that inversion is not significant, and that is the point: the
close-out's ranking among its own best rows carries no information.

**Speed is the durable claim.** ~95% at **1.19M steps** where `b15b` needed 3.2M and `b14a` 3.7M for
~93.5%. Whatever else batch 17 did or did not do, this arm got to the frontier ~3x faster, and that is
measured on step counts rather than on a selected eval.

The checkpoint is in `savedPolicies/b17b-forkseed2/` and is worth promoting to
[`../hallOfFame/`](../hallOfFame/README.md) as a 95.17%/600 champion — a stronger provenance than any
current entry, all of which rest on 100-300 episodes.

## ‡ The reasoning error: a selected sample cannot defend itself against selection

**Worth recording because it was persuasive and wrong.** The argument for trusting the 99/100 was:
`b15b`'s 97 was a lucky draw because its 94 full-length rows averaged 90.7%, whereas `b17b`'s 26
full-length rows averaged **96.2%**, so there was "no ~90% population for the 99 to be drawn from."

**The flaw is that those 26 rows reached full depth *because* they screened well.** The three-stage
protocol promotes checkpoints on a 20-episode screen, so the deep-row population is itself selected on
the very quantity being checked. Its mean is inflated by the same mechanism as the max. Using it to
rule out selection bias is circular.

**A blind grid is the test that is not circular**: 17 checkpoints every 10k across 1110k-1270k, chosen
by *position* alone, 100 fresh episodes each.

| region sample | mean | pooled | CI |
|---|---|---|---|
| `b17b` 1110-1270k, blind grid | **84.06%** | 1429/1700 | 82.2-85.7 |
| the same region's *selected* full-length rows | **96.2%** | — | — |

**84% against 96% is the size of the selection effect**, measured directly. The rule that follows:
**an arm's selected full-length rows describe the checkpoints the screen liked, never the region.** Any
claim about a region needs a position-chosen sample.

**And the region is not flat.** `1140k` reads **12.0%** on the blind grid — a total collapse 30k steps
from a 95% checkpoint, which the close-out never saw because it never screened well enough to be
selected. See
[policy quality changes materially within 1000 training steps](findings.md#policy-quality-changes-materially-within-1000-training-steps).

## The previous record: three checkpoints read 96-97%, and all three are really ~93-94%

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
pool costs about a core whether its batch has 2 rows or 20. **Lowering the worker count to save CPU
does the opposite of what it looks like.**

**Superseded in spirit on 2026-08-08:** the old advice ended "to be gentler on the machine, run fewer
arms, not fewer workers." Being gentle is no longer the goal — see
[evals run hot on purpose](hyperparamTuning.md#-evals-run-hot-on-purpose). The mechanical finding
stands; the priority it served does not.

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

## Closed batches: 11-18 — five nulls, one signal, a null, and a speed result

**Batches 11-15 failed to separate from each other at all** — 20 arms across four hypotheses, every
one of them an optimiser knob. **Batch 16 went after the reward function instead and is the first
non-null**, which is the single most useful thing this table says. Batch 17 went after the *collect
distribution*, the third category, and came back null at the dose it delivered. Per-batch write-ups
are in [`completedRuns.md`](completedRuns.md):

| batch | what it changed | verdict |
|---|---|---|
| **18** | `TARGET_UPDATE_PERIOD` 8 -> **1000** | **the primary metric moved** — 102k faster to pf30 >= 40%, **4/4 seeds** (p=0.125), and drawdown *improved* 20.8 points against a pre-registered risk that it would worsen. Levels +6.7 pp but not separating. Close-out pending |
| [**17**](completedRuns.md#batch-17--forked-endgame-collection-a-null-that-produced-the-project-record) | `SNEK_FORK_BRANCHES=4` — forked endgame collection | **null on the config, record on one arm** — `sef` -1.67 pp (p=0.875) and eq-effort -5.02 pp, both dragged entirely by `b17a`; the other three seeds are **+3.3 to +3.7 pp on eq-effort, 3/3**. And `b17b` produced **99/100 @1205k with a 96.2% region**, the best policy ever measured here |
| [**16**](completedRuns.md#batch-16--the-food-distance-shaping-ablated-the-first-non-null-in-six-batches) | `FOOD_DISTANCE_REWARD=0` | **the first signal** — `sef` +11.35 pp at a matched 1.25M (p=0.250), `best_perfect30` +12.58 pp with 4/4 seeds (p=0.125). Consolidation, not speed or ceiling. Needs replication |
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

**‡ The ceiling still has not moved, and batch 16 does not move it either.** Peak trailing score reads
94.92 / 94.80 / 94.90 / 95.00 / 94.71 (b11, b13, b14, b15, b16) — flat inside 0.3 points across run
lengths from 1.25M to 5.8M. Through batch 15, rising full-length `strong_eval_fraction` (19.5 → 21.6 →
30.5%) was a **run-length artifact** rather than progress, because longer arms accumulate more good
evals. Batch 16 is the first case where that is *not* the explanation: it is the **shortest** batch on
the vector and still beats its control on `sef` by 11 pp at matched steps. So the honest summary of the
six batches is that nothing has raised the ceiling and one thing has raised how much of the time an arm
sits near it.

## Closed: batch 17 — forked endgame collection (`SNEK_FORK_*`)

**Ran 2026-08-07 19:55 to 2026-08-08 00:45, four arms `b17a`-`b17d` at seeds 1-4, stopped by hand at
1.41-1.57M.** Full write-up, per-seed numbers and the mechanism audit:
[`completedRuns.md`](completedRuns.md#batch-17--forked-endgame-collection-a-null-that-produced-the-project-record).
**Closed out 2026-08-08 at `EVAL_MIN_ACHIEVABLE=95`, matching batch 16's protocol exactly** — an
earlier launch at 90 was restarted at 95 after ~4 minutes and no completed rows, because
same-protocol-as-the-control beats better-protocol when the whole point is a paired comparison.

**Verdict: null on the config, and the project record on one arm.** `strong_eval_fraction` **-1.67 pp**
at a matched 1.245M (p=0.875) against its seed-matched batch-16 control, `pooled_equal_effort`
**-5.02 pp** (p=1.000) — and `b17b-forkseed2` holds the
[record at 95.17% over 600 fresh episodes](#-the-record-is-now-95-and-the-99100-that-suggested-it-was-mostly-selection),
reached at 1.19M steps. Both readings are correct and they are about different things. Four things to
carry forward rather than re-derive:

- **One seed carries the result.** Drop `b17a` and `sef` is +4.23 pp. `b17a`'s fork counters are
  normal and `b13a` failed the same way with forking off, so this is most likely seed variance — but
  n=4 cannot prove that, which is the whole finding.
- **The delivered dose was ~60% of design.** Branch share 24-29% against a predicted ~46%, with ~30%
  of eligible fork points skipped because the branch cap was full. **The cap binds, not `FORK_PROB`.**
  So the hypothesis is untested at full strength, not falsified. `SNEK_FORK_BRANCHES=6-8` is the
  re-run if it is ever worth four more slots.
- **The code is sound and stays in.** `forks == retired` on all four arms, `main_steps +
  branch_steps == global_step` exactly, RSS flat, 0 integrity violations over 1.5M steps × 4.
  `FORK_BRANCHES` defaults to 1, so the feature is off unless asked for.
- **‡ The close-out and the graph disagree about the batch, and the close-out is the one with a
  record in it.** On `pooled_equal_effort` the three non-outlier seeds are **+3.34 / +3.66 / +3.56 pp,
  3 of 3 the same direction** — a tighter and more consistent signal than anything in the graph
  metrics — while `b17a` reads **-30.64** and takes the mean to -5.02. **Do not resolve that by
  dropping `b17a`.** Resolve it by noting that n=4 cannot survive one arm this bad, and that the
  question "does forking help" is therefore still open after a batch that spent 20 CPU-hours on it.

**Outstanding, and the highest-value item on this page:** re-measure `b17b` @1205k and @1248k with
`EVAL_MIN_ACHIEVABLE=0` over fresh episodes, then promote to
[`../hallOfFame/`](../hallOfFame/README.md). Until that lands the record is "a ≥95% policy exists",
not "99%".

**Forking stays *on* for batch 18** at the user's direction, which makes batch 18 a clean one-knob
test against batch 17 — see below.

### Pre-registration, as it stood at launch

**`SNEK_FOOD_DISTANCE_REWARD=0` — decided, not conditional on batch 16's close-out.** The distance
shaping stays off from here: the ablation's early read is a measured null (handover 11.5k against the
control's 12.5k, three seeds landing in the same eval, with the arithmetic confound worth ~63 steps),
so nothing suggests it is scaffolding the early game — and it prices every endgame detour at 0.001,
which is the wrong bias for exactly the decisions this batch is trying to teach. A batch about
endgame choice should not be run against a reward that mildly punishes the safe detour.

**And that makes batch 16 the control**, since batch 16 is four seed-matched arms with the same
shaping setting and forking off. **This batch therefore costs four slots rather than eight.**

| item | value |
|---|---|
| config | `SNEK_SEED=1..4 SNEK_FOOD_DISTANCE_REWARD=0 SNEK_FORK_BRANCHES=4 SNEK_FORK_PROB=0.5 SNEK_FORK_MIN_LENGTH=85 SNEK_FORK_MAX_STEPS=60 SNEK_DISCOUNT=0.9975 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0` |
| control | **batch 16**, seed-matched — identical but for `FORK_BRANCHES=1` |
| primary | `strong_eval_fraction` at a common `global_step` horizon, paired, exact permutation over 16 sign flips |
| **horizon** | **~1.25M**, because that is where batch 16's arms stopped. Running longer is fine and probably wise, but the *paired* read is capped at the control's horizon |
| early read | steps to pf30 ≥ 40% — a crossing, which is the read this project trusts early |
| mechanism metrics | buffer share at len ≥ 80 (baseline **20-34%**), terminal and endgame-death adds per 100k, and the branch share of *sampled* batches |
| check at 100k | branch share **above ~5% and rising**. Batch 16's arms first reached length 85 at 8-20k, so forks should be happening by then — a share near 0 means the gate is never being hit and the batch is testing nothing. Stop and lower `FORK_MIN_LENGTH` rather than let it run |
| check at 500k | branch share in **~25-60%** of collect, which is where the cost model puts it once most episodes reach the endgame. Far outside means the arithmetic in this section is wrong and the confound is a different size than advertised |
| abandon | the usual: ≥ 2 of 4 arms not crossing pf30 ≥ 40% by 800k (every batch-16 arm crossed by 465k), or a > 10 pp deficit with a mechanism to explain it |

**‡ The horizon, decided at launch.** `strong_eval_fraction` is a fraction of an arm's *own* evals, so
it is only comparable at a matched step count — and batch 16 stopped at ~1.25M, far short of batches 14
and 15. **The cap is left at the 10M default and the arms run until their curves flatten, with the
paired primary read at a 1.25M truncation.** Running past the control's horizon is free information and
truncating for the paired read costs nothing, whereas stopping at 1.25M and later wanting more would
mean a resume with its buffer and priority discontinuity. What is *not* legitimate is comparing a 4M
forking arm against a 1.25M control and calling the difference forking.

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

## Closed: batch 18 — `TARGET_UPDATE_PERIOD` 1000, forking retained

**Ran 2026-08-08 00:50 to 09:04, four arms `b18a`-`b18d` at seeds 1-4, stopped by hand at
2.40-2.61M.** Charts in
[`charts.md`](charts.md#batch-18--target_update_period1000-forking-retained-stopped-at-240-261m);
close-out evals started 2026-08-08 11:5x. Designed 2026-08-07 and swapped behind the forking batch so
the branching idea got measured first.

**Verdict on the primary metric: it moved, and this is the strongest speed result the project has.**
Paired against batch 17 truncated to a matched 1.406M — one knob, forking on in both:

| metric | b17 (period 8) | b18 (period 1000) | delta | p |
|---|---|---|---|---|
| **steps to pf30 >= 40%** (primary) | 402k | **300k** | **-102.2k** | **0.125** (4/4 seeds) |
| **max drawdown** | 73.97 | **53.20** | **-20.76** | 0.375 |
| `strong_eval_fraction` | 17.56% | 24.29% | +6.73 pp | 0.625 |
| `best_perfect30` | 76.08% | 82.75% | +6.67 pp | 0.625 |
| peak trailing | 94.56 | 94.69 | +0.14 | 0.875 |

0.125 is the floor at n=4, so **every seed got faster** — by 44k to 152k. The levels all point the
same way but none of them separate, which is the expected shape for a speed effect at this sample
size. **And the drawdown risk the batch was pre-registered to watch went the other way**: `b1b-tgt200`
predicted worse drawdown and got it in batch 1; at period 1000 it improved by 20.8 points, 3/4 seeds.

**The ceiling is unmoved for the eighth batch running** — peak trailing 94.94 mean, inside 0.4 of
every batch since 11. Close-out numbers pending; `completedRuns.md` and `findings.md` get written when
the evals finish.

**Forking stays on at batch 17's exact settings**, at the user's direction, and that is the *better*
design rather than a compromise: **batch 17 becomes a clean seed-matched control differing in one knob
only.** The alternative — turning forking off and controlling against batch 16 — would have changed
two things at once. It does mean batch 18 inherits batch 17's ~60%-of-design dose and its
`FORK_BRANCHES` cap pressure, but inherits them *equally on both sides of the comparison*, which is
all a control has to do.

The reading rule that follows: **this batch measures the target period, not forking.** If it wins, the
win is the target period on top of a forking baseline, and whether forking is carrying any of it is
still unmeasured — that question needs the `FORK_BRANCHES=6-8` re-run, not this batch.

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

**As launched**, byte-identical to batch 17 apart from the target period:

```
SNEK_SEED=1..4  SNEK_TARGET_UPDATE_PERIOD=1000
SNEK_FOOD_DISTANCE_REWARD=0  SNEK_DISCOUNT=0.9975  SNEK_GUIDED_FRACTION=0.8
SNEK_FORK_BRANCHES=4  SNEK_FORK_PROB=0.5  SNEK_FORK_MIN_LENGTH=85  SNEK_FORK_MAX_STEPS=60
SNEK_PRIORITY_SIGNAL=td_loss  SNEK_IS_WEIGHTS=0
```

| item | value |
|---|---|
| control | **batch 17**, seed-matched — identical but for `TARGET_UPDATE_PERIOD` 8 → 1000 |
| primary | **steps to pf30 ≥ 40%**, the speed metric batch 15 pre-registered, because the live hypothesis is about *early* learning |
| secondary | `strong_eval_fraction` and `pooled_equal_effort` at equal effort |
| **also report** | **max drawdown, explicitly** — the one number the prior data point predicts gets *worse*, so it gets read whatever the primary does rather than looked up afterwards if the batch disappoints |
| horizon | batch 17's arms reached 1.41-1.57M, so the paired read caps at **~1.41M** |
| check early | a target period 125x longer changes the bootstrap; if an arm has not started scoring by ~50k when every batch-16/17 arm was well up by then, read it as the period being too long rather than a slow seed |
| abandon | the usual: ≥ 2 of 4 arms not crossing pf30 ≥ 40% by 800k, or a > 10 pp deficit with a mechanism |

`SNEK_GUIDED_FRACTION=0.8` is shown for the record; it is the default from 2026-08-07, so passing it
changes nothing. The four `SNEK_FORK_*` values are batch 17's, carried over deliberately so the
comparison is one knob.

Risk to check on day one: an arm that learns too slowly never clears
`SNEK_MIN_CHECKPOINT_SCORE=40` and so writes no checkpoint, which makes it unresumable and
unmeasurable. Batch 1's hint points the other way, so this is a glance at the first few evals rather
than a real concern.

## What six batches leave for the ones after this

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

