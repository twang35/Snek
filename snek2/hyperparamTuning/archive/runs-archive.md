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
[`../hallOfFame/`](../../hallOfFame/README.md) as a 95.17%/600 champion — a stronger provenance than any
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
[policy quality changes materially within 1000 training steps](../findings.md#policy-quality-changes-materially-within-1000-training-steps).

## The previous record: three checkpoints read 96-97%, and all three are really ~93-94%

`b15b-nstep3seed2` @3245000 read **97/100**, the highest selected measurement this project has
produced. It does not move the record. **Every champion, corrected, lands in the same place:**

| checkpoint | selected | corrected | how |
|---|---|---|---|
| `b15b` @3245k | **97/100** | **93.0%** (CI 89.5-95.4) | re-measured over 200 fresh episodes: 182/200, pooled 279/300 |
| `b14a` @3702k | 96/100 | 93.5% (CI 89.2-96.2) | re-measured: 91/100, pooled 187/200 |
| `b11b` @855k | 96/100 | ~94.0% | shrunk against a Beta prior on its 104 unselected graph-100% rows |

All three are in [`../hallOfFame/`](../../hallOfFame/README.md) and all three load on `master`. Their
intervals overlap almost entirely, so **the honest statement is that five batches have produced the
same ~93-94% policy three times**, not a rising record.

**Re-measure, do not shrink.** Re-measurement confirms the copy loads *and* sharpens the estimate,
and it is the only option left for a gated close-out, since a gate destroys the unselected
graph-100% rows the shrinkage prior needs — see
[`findings.md`](../findings.md#three-measurement-caveats).

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
[`archive/batches1-11.md`](batches1-11.md).

## Reference: the 2026-08-03 changes, and what they froze

Batch 10's own write-up moved to
[`archive/batches1-11.md`](batches1-11.md) once batch 13 made batch 11 the baseline.
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
[evals run hot on purpose](../hyperparamTuning.md#-evals-run-hot-on-purpose). The mechanical finding
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
[`hyperparamTuning.md`](../hyperparamTuning.md#measuring-a-policy-properly-eval_checkpointspy).

A close-out is also resumable now (`EVAL_RESUME=1`), which is what made switching the worker
count mid-run cost only the checkpoint in flight rather than the 333 already measured.

## The environment changed again on 2026-08-03: two new observations

A fourth environment (‡‡‡ in [`completedRuns.md`](../completedRuns.md)). The vector went from **26
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
are in [`completedRuns.md`](../completedRuns.md):

| batch | what it changed | verdict |
|---|---|---|
| **18** | `TARGET_UPDATE_PERIOD` 8 -> **1000** | **the primary metric moved** — 102k faster to pf30 >= 40%, **4/4 seeds** (p=0.125), and drawdown *improved* 20.8 points against a pre-registered risk that it would worsen. Levels +6.7 pp but not separating. Close-out pending |
| [**17**](../completedRuns.md#batch-17--forked-endgame-collection-a-null-that-produced-the-project-record) | `SNEK_FORK_BRANCHES=4` — forked endgame collection | **null on the config, record on one arm** — `sef` -1.67 pp (p=0.875) and eq-effort -5.02 pp, both dragged entirely by `b17a`; the other three seeds are **+3.3 to +3.7 pp on eq-effort, 3/3**. And `b17b` produced **99/100 @1205k with a 96.2% region**, the best policy ever measured here |
| [**16**](../completedRuns.md#batch-16--the-food-distance-shaping-ablated-the-first-non-null-in-six-batches) | `FOOD_DISTANCE_REWARD=0` | **the first signal** — `sef` +11.35 pp at a matched 1.25M (p=0.250), `best_perfect30` +12.58 pp with 4/4 seeds (p=0.125). Consolidation, not speed or ceiling. Needs replication |
| [15](batches12-15.md#batch-15--n_step_update3-falsified-on-speed-null-on-level-and-a-97100-that-is-really-93) | `N_STEP_UPDATE=3` | **falsified on speed** — 128k slower to pf30 ≥ 40%; evals null, best ckpt +0.05 pp |
| [14](batches12-15.md#batch-14--disc-09975-at-guided-08-and-the-widest-seed-spread-yet) | `DISCOUNT=0.9975`, `GUIDED_FRACTION=0.8` | null vs 13; `pooled_equal_effort` +0.01 pp |
| [13](batches12-15.md#batch-13--the-epsilon-rewrite-plus-the-exploration-shield-an-exact-null) | eps handover 0.0125 + shield 0.5 | null vs 11 on five metrics |
| [12](batches12-15.md#batch-12--the-deadlock-abandoned-at-1m-of-25m) | eps handover 0.05 | **deadlocked**, abandoned 4/4 |
| [11](batches1-11.md#batch-11--the-same-config-on-the-30-value-vector-no-significant-difference) | the 30-value vector itself | +4 to +5 pp vs batch 10, not significant |

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
[`completedRuns.md`](../completedRuns.md#batch-17--forked-endgame-collection-a-null-that-produced-the-project-record).
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
[`../hallOfFame/`](../../hallOfFame/README.md). Until that lands the record is "a ≥95% policy exists",
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
[`hyperparamTuning.md`](../hyperparamTuning.md#forked-endgame-collection--snek_fork_). The one most likely
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
[`charts-archive.md`](charts-archive.md#batch-18--target_update_period1000-forking-retained-stopped-at-240-261m);
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
[`archive/batches1-11.md`](batches1-11.md#target_update_period200-hypothesis-not-supported-but-interesting).

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
[`hyperparamTuning.md`](../hyperparamTuning.md#the-primary-metric-strong_eval_fraction-the-share-of-an-arms-evals-at-80).

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
[`findings.md`](../findings.md) as established without n=3, which is why batch 10 spent all four
slots on one value rather than splitting them.



## Retired from `runs.md` 2026-08-22 — the batch-27/30 relaunch and the 2026-08-16/18 host snapshots

The perfect-game counter bug is fixed (`b72a5a84`), pinned by `tests/test_perfect_game_counting.py`, and
written up in [`CLAUDE.md`](../../../CLAUDE.md#a-perfect-game-is-identified-by-its-score-never-by-its-reward)
and [`findings.md`](../findings.md#-a-perfect-game-was-identified-by-its-final-reward-and-the-shaping-term-silenced-every-counter).
Every arm this section warns about (`b27a-d`, `b30a-d`) is void and closed, and nothing left in `runs.md`
predates the fix — so the "read this before anything below it" framing had stopped being true.

## ⚠ Batches 27 and 30 were relaunched — the perfect-game counter was reward-based (2026-08-14)

**Read this before anything below it.** Every perfect-game counter identified a win by comparing the
episode's final reward with `PERFECT_GAME_REWARD`, and the chase-safe term shifts that reward by `−c`. So
`perfect_percent` read **0 for every eval of b27a-d and b30a-d** while the arms were filling boards from
step 9k, and because `training.epsilon_for` takes the trailing perfect rate as its skill signal, **epsilon
stayed pinned at 0.0125** — the refinement ceiling — instead of annealing. Full account, numbers and fix:
[`findings.md`](../findings.md#-a-perfect-game-was-identified-by-its-final-reward-and-the-shaping-term-silenced-every-counter).

**The fix shipped as `b72a5a84`** — counting moves to `state_helpers.is_perfect_score(score)`, and
`tests/test_perfect_game_counting.py` pins it (10 tests, both mutants caught, suite 23 modules / 596 tests /
0 failed). Deployed to the desktop the same evening; both boxes now run it.

| batch | void arms | live arms | where |
|---|---|---|---|
| **27** | `b27a-d`, killed at 309-326k | **`b27e-h`** (seeds 1-4), launched 21:31 | desktop, priority 10, close-outs and HOF-500 auto-chained |
| **30** | `b30a-d`, killed at 137-139k | **`b30e-h`** (seeds 1-4), launched 21:40 | laptop, one chart window on `--arms b30` |

**Fresh policies, not resumes.** The void arms' weights and buffers carry 320k steps trained under an
exploration schedule that was never going to descend, which is the one thing a seed-matched comparison
against b24/b25 cannot absorb.

**Batch 30's checkpoints were deleted and the wave relaunched a second time** (21:40), because the first
relaunch's chart window opened with **eight panels**: the viewer's arm registry admitted anything inside a
12 h TTL, and the killed a-d arms were 71 minutes old. `savedPolicies/b30[a-h]` (~330 MB) and the tmp
registry are gone; **`runs/b30a-d*` is kept** — those graphs, reports and eval series are the measured
record of the counter bug, committed at `1cd5a03b`, and `charts/` holds its own copies by design. Say so if
you want them removed as well. The registry rule is now liveness-based
([the mechanism](../../../CLAUDE.md#rendering-is-off-by-default--use-watchpy-to-see-a-game)); the desktop was
checked and needs no change, since its daemon passes explicit PNG paths and reads no registry.

**Progress update, 2026-08-16 — all four chase-safe shaping batches (b27-b30) are done, and the gate is the
lever: gate 85 is null at any dose, gate 75 produced a record region.** All sixteen fixed-counter arms
trained with a real perfect rate and a descending epsilon — the two things b27a-d could not produce in 320k
steps — so the counter fix is confirmed end to end. Where each batch landed (full numbers and graphs in
[`charts.md`](../charts.md); write-ups in [`completedRuns.md`](../completedRuns.md)):

- **b27e-h (desktop): done at 2M, closed out — a null.** Pooled mean **85.2** (eq-effort, gate 95) vs the
  b24 control's ~87.9 (a shade *below*), and **0 of 4** seeds produced a ≥98%/500 checkpoint — best `b27h`
  **97.5%** — against the control's **two** 98.0%/500 records. `c=0.10` on `fc 320` did not reproduce the
  record, let alone beat it.
- **b30e-h (laptop): done at 2M and closed out at 15:05 — also a null, and by the same margin as b27.**
  Training was a dead heat (matched ≤2M best-30 **92.9 vs 93.6**, `sef` **56.9 vs 58.6** against the b25-r2
  control), and the close-out agrees: pooled equal-effort **83.3 mean** (84.3 / 84.3 / 83.8 / 81.0) against
  b25-r2's ~86.1, so **~2.8 behind its seed-matched control** — the same direction and size as b27's 85.2 vs
  87.9 on `fc 320`. **Two architectures, two nulls-or-worse for `c=0.10`.** Ten checkpoints reached ≥98% at
  *100* episodes (6 / 3 / 1 / 0), best 99.0% — but that is not the ≥98%/**500** gate. **The HOF-500 pass has
  since run and is empty**: every one of its ten candidates was abandoned under gate 98 before 500 episodes,
  best partial `b30e` @651k **96.1% at 285 episodes**, then `b30g` 95.3% and `b30f` 90.9%. So the
  shaping×architecture 2×2 is complete and `c=0.10` produced **no record-tier checkpoint on either net** —
  the write-up is in [`completedRuns.md`](../completedRuns.md) and
  [`findings.md`](../findings.md#-chase-safe-reward-shaping-null-at-gate-85-at-any-dose-records-at-gate-75--the-gate-is-the-lever).
- **b28a-d (desktop, `c=0.20`, gate 85): done at 2M, closed out and HOF-500'd — a null, the dose is not the
  issue.** Pooled mean **85.4** (~2.5 under the b24 control's 87.9) and **0 of 4** seeds held ≥98%/500.
  Doubling the dose changed nothing, which — with b27/b30 — rules out both the net and the dose at gate 85.
- **b29a-d (desktop, `c=0.10`, gate 75): done at 2M, closed out and HOF-500'd — the positive result.**
  Pooled **87.8** (a dead heat with b24) but **21 checkpoints held ≥98%/500 across 2 of 4 seeds**, where the
  record-holding control produced only 2 isolated ones. `b29b` @1447k = **99.0%/500 (495/500)**, the head of
  an 18-checkpoint band — **the new project record**, promoted to `hallOfFame/` (see Record status below).
  **The gate, not the dose or the net, is the lever.**
- **Laptop: `b31a-d` was stopped at 538-569k with no close-out**, for the same measurement that started the
  C51 `epsilon` line — the churn is the learning rate, not C51
  ([`findings.md`](../findings.md#-the-c51-arms-chaos-is-the-learning-rate-not-c51--and-the-rate-is-high-because-c51-needs-it)).

**Both hosts as of 2026-08-18 21:24 (fetched). Both are training `b42`/`b43` — the same four
checkpoints, one learning rate each, and both chains are armed. Nothing is owed by hand.**

| host | state | measurement chain |
|---|---|---|
| **laptop** | **idle of trainers.** `b43a-d` are finished on all three instruments. Its last work was the **1000-episode re-measurement of the project's eight best checkpoints** (four `b43`/`b44` continuations, four `hallOfFame/` entries) — 8000 episodes, ~16 min, results in `runs/k1000*_checkpoint_evals_k1000.json`. **All eight fell**, mean −1.35 and −1.45 pp; see [the finding](../findings.md#-the-winners-curse-measured-four-selected-champions-all-fell-and-the-500500-did-not-reproduce-2026-08-20) | **armed.** `scripts/chain_closeout_after_training.sh b43 120` running detached (reparented to pid 1, log `/tmp/b43_chain.log`): polls until the four arms self-terminate at the 3M cap, then close-out at gate 96, then the HOF-500 re-measure on anything ≥98%. **fired on time at 01:07.** The close-out is at 542-709 of 791-1196 checkpoints per arm after 7.3 h — `b43b` is the long pole at ~8 h remaining, so HOF-500 starts this evening. See the close-out cost warning below: this is expected, not stuck |
| **desktop** | **`b45a-d` training**, **85%** of a 5M cap (4.10-4.42M, **+2.75-2.91M past seed**), 4 trainers, 0 evals. `b42`, `b43` and `b44` are all finished on training, close-out and HOF-500; `b44`'s HOF landed 2026-08-20 with all 2235 measurements and **874 rows ≥98%/500**. `b45`'s close-out is queued behind the training | **explicit, not automatic.** A `kill -9` makes the trainings `failed`, and `auto_closeout` fires only on `ok` — so four `<policy>-closeout` specs were queued by hand *before* killing. They use **exactly the id the daemon would synthesize**, which `_scan_pending` keeps in preference to its own projection, so there is no double-run; and `_hof_owed` keys off the *close-out's* success, so HOF-500 still chains by itself |

**The two hosts now run the same chain, which they did not before 2026-08-18.** The desktop daemon has
chained `training → closeout → HOF` since 2026-08-15; the laptop's `chain_closeout_after_training.sh`
stopped after the close-out, so a laptop batch produced *less* than the same batch on the desktop and the
missing half was the one that decides whether a checkpoint is hall-of-fame material. The script now carries
the HOF stage, copied from the daemon's own `HOF_EVAL_ENV`/`HOF_EVAL_ARGS`. **It also pins the close-out
gate at 96** — it used to inherit `eval_checkpoints`' default of 95 while the desktop pinned 96, so the two
hosts were writing close-outs under different gates. That matters because a file's gate lives in its
payload as `min_achievable` and has to be checked before anything is pooled across files, and because the
HOF pass selects `above:98` *from the close-out file* — a gate at or above 98 would abandon the very rows
it needs. Verified against four finished close-outs: the selector reproduces the desktop's own HOF row
counts exactly (b40b 63, b29b 64, b40d 2, b39a 0).

**`b42`'s arms started ~5 minutes before the learning-rate fix was deployed, and it does not matter.** A
running job keeps the code it launched with, so b42 is on the pre-fix `snek2.py` — but its configured rate
*is* the rate its checkpoints carry (1e-5), so `enforce_learning_rate` would compare equal and assign
nothing. The two batches still differ only in the learning rate. Any *future* desktop batch that retunes
the rate on a resume gets the fix, which is live there from `b8d817fd7`.

**Adam's `epsilon` is settled and b32 is closed** — shared-state-set churn **0.119 → 0.088, −26%, 4 of 4
paired, flat to 1M, no dose effect**. The same measurement found that every previously published per-arm
churn figure was inflated ~2×; both are written up under [batch 32](../completedRuns.md#batch-32--adams-epsilon-on-c51-it-works-at-26-churn-and-the-dose-does-not).
**The dose is now closed for good** — b36 vs b38 retried it at 4 seeds a side and pooled 76.77 vs 74.73 at a
matched ≤2M horizon, 3 of 4 favouring `1.5e-4`, p=0.625. `1.5e-4` stays the default on lower seed variance.

**A `failed` training does not publish.** `_publish_results` is gated on the same `ok` flag as
`auto_closeout`, so `b42`'s training artifacts were never going to reach the `results` branch. They were
rsynced across by hand (`rsync -a "the-claw-den:Snek/snek2/runs/b42*" snek2/runs/`) and are what the banded
comparison below is computed from. Re-rsync when the close-out finishes to pick up
`_checkpoint_evals.json`.

**Batch numbering: `b37` is the desktop's b29 replication, so the laptop's dose arm is `b38`.** Worth stating
because both were queued within minutes of each other from different hosts, and `b37` was very nearly used
twice. **`b39` is a C51 zero-init batch (laptop, `launch_b39_zeroinit.sh`); the free-space batch below is
`b40`.**

## Retired from `runs.md` 2026-08-22 — the closed rungs of the b42-b45 ladder

`b42`, `b43` and `b44` are finished on all three instruments and their canonical write-ups are in
[`completedRuns.md`](../completedRuns.md#batch-44--the-same-four-checkpoints-at-lr-1e-7-the-best-rung-of-the-ladder--874-checkpoints-at-98500-and-it-falsified-its-own-pre-registration);
the conclusion is in [`findings.md`](../findings.md#-continuing-a-champion-works-and-lower-is-better--but-the-best-checkpoint-was-the-wrong-one-to-continue-b42b43b44-2026-08-20).
Kept here for the pre-registrations and the two reasoning post-mortems: why the `1e-7` null prediction
failed, and why `peak_trailing` cannot judge a continuation arm.

### ⚠ Read the result against selection bias, not against the numbers in that table

The four starting rates are **the maximum of a noisy statistic over 8 arms and hundreds of checkpoints**, so
they are biased upward — the winner's curse. This file already established the size of the effect:
[a ≥98%/100 checkpoint has roughly a 1-in-60 chance of holding at
500](../findings.md#-corrected-2026-08-18-the-record-region-does-not-replicate--the-98500-count-is-seed-noise-and-pooled-is-the-only-metric-of-this-family-worth-reading),
and `b29b`'s 18-wide band was **a seed, not a config**. So:

- **An arm that continues and later measures ~96-97% has not necessarily decayed.** Regression to the mean
  predicts exactly that, with no contribution from the extra training.
- **The clean comparison is `b42` against `b43`, not either against its own starting rate.** Both start from
  byte-identical checkpoints, so the selection bias is common-mode and cancels.
- **The unbiased per-arm baseline is cheap and has not been run**: re-measure each *starting* checkpoint at
  500 fresh episodes. The close-outs may supply it for free — the seeded dir still contains the start
  checkpoint and it has a graph point, so `top20` can select it — but that is not guaranteed, and it is worth
  one 2000-episode job if the headline reads as a decline.

### What each outcome would mean

| reading | meaning |
|---|---|
| `b42` holds ~99% and extends the band | the record region is reachable by *training longer from inside it*, and every previous arm was stopped early. The most valuable outcome |
| **✅ `b42` decays, `b43` holds — this is what happened, 4 of 4 seeds** | the 1e-5 steps are too large to sit still at a champion — the endgame is a narrow basin and the default rate walks out of it. Makes **low-LR fine-tuning the way to bank a record**, which is a new tool |
| both decay together | the decay is not the step size. Points at the objective or the replay distribution, and says a champion checkpoint is a transient the optimizer does not want to stay in |
| both hold, neither improves | ~99% is the ceiling of this config and the remaining 1% is not a learning problem — consistent with the four null PBRS terms |
| `b43` decays and `b42` does not | would be surprising and is the one reading that suggests a bug; check the reset line is in all four logs first |

### ✅ What happened — `b42` decays, `b43` holds, on 4 of 4 seeds

Banded mean self-eval perfect rate over the **+385k window matched across all eight arms** (the largest window
where every arm has data — `b42` was stopped at +385-421k, `b43` is past +541k, and `sef` is a fraction of each
arm's *own* evals, so bands are the only fair comparison):

| band past seed | `b42` (1e-5) | `b43` (1e-6) | diff | seeds favouring 1e-6 |
|---|---|---|---|---|
| 0-100k | 93.6 | 95.4 | +1.8 | 3 of 4 |
| 100-200k | 91.6 | 96.3 | **+4.7** | **4 of 4** |
| 200-300k | 89.1 | 96.2 | **+7.1** | **4 of 4** |
| 300-385k | 89.5 | 95.6 | **+6.1** | **4 of 4** |

`best_perfect30` and `strong_eval_fraction` agree on all four seeds: **93.3-97.7 / 87.0-97.9** at 1e-5 against
**98.3-100.0 / 97.9-99.8** at 1e-6.

Two refinements on the headline. **The decay is a one-time drop to a lower plateau, not an ongoing collapse** —
1e-5's last two bands are flat at ~89, so the arm re-equilibrates about 5 pp down rather than unravelling.
And **1e-6 is not pure preservation: `b43b` reached a `best_perfect30` of 100.0 at 1667k, 320k steps past its
own seed**, a 30-eval window with no imperfect game, against 97.3 for its byte-identical `b42` twin. So on at
least one seed, continuing a champion at a low rate *improved* it. These are 10-episode self-evals; the
close-outs and HOF-500s decide it.

**⚠ `peak_trailing` cannot judge this family of arms, and it briefly fooled this investigation.** It is
trailing average *score*, which maxes at 95/95 — **all eight `b42`/`b43` arms read exactly 95.0 with the peak
timestamped at their own seed step**, because a policy restored from a 98% checkpoint fills the board on eval
one. That saturation was first read as "no arm ever beat its starting checkpoint", which is false and is
retracted. Use `perfect_percent`, `best_perfect30` or `sef`. The general lesson is the `game_over`/`sef` one
again: **a metric at its ceiling looks like a measurement and carries no information.**

### ‡ b44 at 1e-7 beat 1e-6 on 4 of 4 seeds — the pre-registered null was wrong

Done at the 3M cap. Same four seeded checkpoints, `SNEK_LEARNING_RATE=1e-7` the only change from `b42`.
**The estimate written into the specs before launch put ~65% on a null and ~25% on a win; the 25% branch is
what happened, on every seed.** The readings as pre-registered, with the outcome marked:

| reading | probability | what it would mean |
|---|---|---|
| ~~null against `b43`~~ | ~65% | **did not happen.** The reasoning — that 1e-6 is already doing nothing but failing to damage the policy — was simply wrong: at 1e-7 these arms are still *moving*, and their best-30 peaks arrive 300-1100k steps past seed rather than at it |
| **✅ better than `b43`** | ~25% | **this is what happened, 4 of 4 seeds.** Decay at 1e-6 is still nonzero and the optimum is below it — extend the ladder to **1e-8** |
| ~~worse than `b43`~~ | ~10% | did not happen. The "1e-6 is actively repairing and 1e-7 cannot keep pace" mechanism is not what is going on — **both** rates improve, and the slower one improves further |

**Why the null prediction failed, and it is worth keeping.** The reasoning was that Adam's step is
`lr · m/(√v + ε)`, so lowering the rate scales every step near-uniformly without changing the relative dynamics
— hence 1e-7 should look like 1e-6 at a tenth the movement, and if 1e-6 is already only *preserving*, a tenth of
preserving is still preserving. The scaling argument is fine; **the premise that 1e-6 was only preserving was
not.** It was improving, slowly, and 1e-7 improves *further* because it improves *more slowly and for longer*.
The tell was available before launch and was misread: `b43b`'s 100.0 best-30 window arrived 320k steps past its
seed, which is motion, not stillness. **The general lesson: "this knob has saturated" needs evidence that the
quantity has stopped changing, not that it is changing slowly** — and a peak arriving late in the run is
evidence against saturation.

| band past seed | `b43` 1e-6 | `b44` 1e-7 | diff | seeds `1e-7` ahead |
|---|---|---|---|---|
| 0-250k | 95.9 | 96.3 | +0.4 | 3 of 4 |
| 250-500k | 95.4 | 96.5 | +1.1 | **4 of 4** |
| 500-750k | 92.2 | **96.6** | **+4.4** | **4 of 4** |
| 750-1000k | 93.4 | 97.0 | **+3.6** | **4 of 4** |
| 1000-1250k | 94.8 | 96.6 | +1.8 | 3 of 4 |
| 1250-1487k | 94.5 | 95.2 | +0.7 | 2 of 4 |

Per seed over the whole +1487k common window: **+4.6, +1.0, +1.6, +0.8** — mean **+2.0 pp**, against **+4.9 pp**
for the 1e-5 → 1e-6 step. The gap is widest at 500-1000k because that is where `b43` dips to 92.2 and `b44` does
not: **1e-6 still wanders on a half-million-step timescale, 1e-7 largely does not.** `b44`'s best-30 peaks land
at 1814k / 2190k / 2230k / 2460k, i.e. 300-1100k past seed, against *at the seed step* for `b42`.

### ✅ The 100-episode close-outs agree with the self-evals, and the effect is much larger than the self-evals suggested

All three rungs are closed out (2026-08-19). The comparison below is **the share of a batch's measured
checkpoints that reached ≥98% over 100 episodes** — not the best row, which is a maximum over a noisy statistic
and reads high by construction on whichever arm was measured most.

| starting checkpoint | `b42` 1e-5 | `b43` 1e-6 | `b44` 1e-7 |
|---|---|---|---|
| `b29b` @1447k (the record) | 6.5% (17/261) | 12.8% (166/1297) | **56.3% (853/1516)** |
| `b29a` @1347k | 7.2% (27/373) | 38.7% (607/1568) | **53.7% (867/1616)** |
| `b40b` @1513k | 2.2% (6/279) | 10.0% (133/1325) | **30.1% (415/1377)** |
| `b29c` @1396k | **14.1% (48/340)** | 6.0% (83/1378) | 6.9% (100/1450) |

**Monotone in the rate on 3 of 4 seeds, and the steps are large** — on the record seed the ≥98%/100 share goes
6.5 → 12.8 → **56.3**, i.e. `1e-7` puts more than half of its checkpoints in the record tier where the default
rate puts one in fifteen. The ≥99%/100 shares tell the same story (2.7 → 6.0 → 32.8 on that seed).

**The exception is `b29c`, and it is the weakest of the four starts** (its seed row was 97.1% over 378 episodes,
abandoned by the gate rather than a full-length 500). On that seed the ladder is flat from 1e-6 down and `b42`
is nominally ahead. Consistent with the rest of this file's experience: the rate matters most where there is a
narrow good basin to sit still in, and that seed may not have been in one.

**Two reasons this understates rather than overstates the effect.** The close-out selector measures every
checkpoint whose graph eval cleared the mandatory tier, so the denominators are near-censuses of different
sizes: **65-93% of `b42`'s checkpoints and 84-98% of `b43`/`b44`'s**. `b42`'s pool is therefore the *more*
selected one — its weak checkpoints are missing from the denominator — and it still comes last. And `b44`'s
pool is the least selected of the three (98% of all its checkpoints on two arms), so it carries the most dead
weight and still wins.

**A caveat that applies to the whole table**: these close-outs ran at a gate of 96, so a row below 96% was
abandoned early and is not full length. Every share above counts full-length rows only, which is exactly the
set the gate guarantees, so the comparison is sound — but the *pooled* rate over all rows is not comparable
across arms with different abandonment counts and is deliberately not quoted here.

### ✅ `b43`'s HOF-500 is done, `b44`'s is half done — and its 500/500 did not survive re-measurement

Both re-measures are `above:98` at 500 episodes, flat, and they are the instrument that decides the ladder —
everything above is 100 episodes. `b43`'s finished on the laptop at 09:55 on 2026-08-20 (767 measurements,
12.3 h wall clock, 80% lane utilisation); `b44`'s is at 1176 of 2235 on the desktop, read 10:05.

| arm | ≥98%/500 | ≥99%/500 | best /500 | pass |
|---|---|---|---|---|
| `b43a` (1e-6, from the record) | 16 | 2 | 99.4% @1618000 (497/500) | **done** |
| `b43b` | **170** | **17** | **99.6% @1661000** (498/500), also @1708000 | **done** |
| `b43c` | 1 | 0 | 98.0% @1760000 | **done** |
| `b43d` | 0 | 0 | — (97.7% @1426k over 488, gate-abandoned) | **done** |
| `b44a` (1e-7, from the record) | 105 | — | 99.6% @2207000 | ~53% |
| `b44b` | **155** | — | 100.0% @1886000 (500/500) — **re-measures 98.2%/1000** | ~53% |
| `b44c` | 33 | — | 98.8% @2301000 | ~53% |
| `b44d` | 0 | — | — | ~53% |

**Totals: `b42` 4 rows ≥98%/500 in a completed pass, `b43` 187 in a completed pass, `b44` 293 with roughly half
its pass to go.** The ladder's ordering holds on the deepest instrument available, and the gap is not marginal —
it is 4 → 187 → 293+ from byte-identical starting weights, with learning rate the only difference.

**`b44b` @1886000 scored 500/500, and then failed to replicate.** Re-measured the same day on **1000 fresh
episodes: 982/1000 = 98.2%** (97.2-98.9), a −1.8 pp drop at p=0.0025. The four best checkpoints in the project
were re-measured together and **all four fell**, mean −1.35 pp — so a selected /500 maximum should be treated as
roughly **1.4 pp optimistic**, and the pooled fresh figure for the four is **98.40%**. Nothing is in
`hallOfFame/`, and the re-measurement is why: [full result in `findings.md`](../findings.md#-the-winners-curse-measured-four-selected-champions-all-fell-and-the-500500-did-not-reproduce-2026-08-20).

**Two things this pass corrects in the note it replaces.** First, the candidate record was attributed to the
wrong arm: `b43a` was leading at ~25% of the pass with 99.4%, and `b43b` — a *worse* starting checkpoint, 98.4%
against 99.0% — finished with 99.6% and **170 of `b43`'s 187** ≥98% rows against `b43a`'s 16. **So a checkpoint's
own /500 rate does not predict how well it continues**, and continuing the best checkpoint was not the best move.
Second, the note's own advice — *watch the count, not the maximum* — was right, and it is what makes the ladder
readable: the maxima (99.4 / 99.6 / 100.0) are three noisy draws separated by 2-3 episodes, while the counts are
separated by factors of 40 and up.

**The counts are still one seed's story.** `b43b` alone carries 91% of `b43`'s total and `b44b` 53% of `b44`'s so
far, while the `b29c` seed (`b43d`/`b44d`) produced zero on every rung. The rate multiplies whatever the seed
had; it does not create it. That is consistent with [the retired b29 record
region](../findings.md#-corrected-2026-08-18-the-record-region-does-not-replicate--the-98500-count-is-seed-noise-and-pooled-is-the-only-metric-of-this-family-worth-reading)
and it means **`b45` at `1e-8` should be read on the same four seeds separately**, never pooled into one number.

**`b43` is now closed out of this file.** Its canonical write-up — the per-seed `b42` comparison, pooled
equal-effort figures, the cost breakdown and the two readings that would be wrong — is in
[`completedRuns.md`](../completedRuns.md#batch-43--continuing-the-four-best-checkpoints-at-lr-1e-6-a-record-region-10x-wider-than-anything-before-it-and-the-best-checkpoint-was-the-wrong-one-to-continue), and the selection result it produced is in
[`findings.md`](../findings.md#-continuing-a-champion-works-and-lower-is-better--but-the-best-checkpoint-was-the-wrong-one-to-continue-b42b43b44-2026-08-20).
The ladder's forward plan stays here until `b45` finishes.

## Retired from `runs.md` 2026-08-22 — continuation close-out cost, and the b45 queue block

Both closed. The cost guidance now lives in [`CLAUDE.md`](../../../CLAUDE.md#eval-cost), and the
bookkeeping fix that halved a HOF pass landed 2026-08-20.

### ⚠ A continuation batch's close-out costs 10-20x a normal one's — budget for it

`top20` measured **791, 1196, 803 and 826** checkpoints on `b43`'s four arms, not 20, and `b43`'s close-out has
been running since 01:07. This is documented behaviour rather than a bug: **"N is a target, not a quota"** —
`select_top_checkpoints` measures every checkpoint whose 10-episode graph eval reached **≥90%**, past N. A normal
arm climbs from 0 so few checkpoints qualify; **an arm continued from a 98% checkpoint spends its whole run above
the mandatory threshold, so nearly every checkpoint qualifies.** `b42` is the contrast from the other side — it
decayed, so only 261-373 of its checkpoints qualified and its close-out finished overnight.

**So do not read a continuation close-out still running after 7 hours as hung.** Measured now that all three are
finished: **`b43` 15 h on the laptop, `b44` 11.2-14.9 h on the desktop, `b42` 2.5 h** — the last one because it
decayed, so its pool was a fifth the size.

**The HOF-500 passes are the same shape one instrument deeper**, and they are the larger bill: `above:98` selects
**166 / 607 / 133 / 83** checkpoints on `b43` and **853 / 867 / 415 / 100** on `b44`, each at 500 episodes. `b43`'s
pass prices at ~10 h and `b44`'s at ~18 h; `b42`'s was 0.2-1.0 h per arm, on 4 qualifying rows. **So a rung of
this ladder costs roughly 4 h of training and 25-30 h of measurement**, and the measurement is on the critical
path for the next rung — see the note on `b45` below.

If a future continuation batch needs a cheaper close-out, the lever is the *selector*, not the worker count —
`above:<threshold>` reads a prior close-out's 100-episode numbers instead of the graph, which is what the HOF
pass already uses.

### ✅ Resolved — `b45` was blocked behind `b44`'s HOF-500, and letting it run was the right call

The wave barrier held `b45`'s four trainings while `b44-hof` ran alone on a 4-trainer box for ~18 h. **Nothing
was killed and nothing was requeued**: `b44`'s HOF landed complete on 2026-08-20 with all **2235** measurements
and **874 rows ≥98%/500**, which is the number the whole ladder is scored on, and `b45` started straight after.
Two things this settled for the next time the barrier bites:

- **The 18 h estimate was roughly half bookkeeping, not measurement.** The controller's per-round write was
  O(rows) × O(episodes); `eval_plan.WriteGate` and `RowCache` took a measurement's overhead from 58 s to 1 s
  (fixed 2026-08-20). A comparable pass should now price nearer 9-10 h, so **the trade-off table above is
  priced on the old code** — recheck before using it to justify killing an eval.
- **The stricter-selector options were never needed.** They stay documented above because the reasoning about
  *which* band to cut (`above:99.5`, not `above:99`) is what a future call turns on.

## Retired from `runs.md` 2026-08-22 — the `SNEK_LEARNING_RATE` resume bug

Fixed by `training.enforce_learning_rate`; full account in
[`findings.md`](../findings.md#-snek_learning_rate-was-silently-discarded-by-every-resume--adams-rate-rides-in-the-checkpoint).

### ⚠ This batch only measures anything because of a bug found while setting it up

`SNEK_LEARNING_RATE` was a **no-op on every resume**. Adam's `learning_rate` is a checkpointed `tf.Variable`,
so `initialize_or_restore()` silently restored the saved 1e-5 over the configured 1e-6 — `b43` would have run
four arms identical to `b42` and reported otherwise. Fixed by `training.enforce_learning_rate`; each arm now
prints its reset line at startup, **and that line is the batch's tripwire** — an arm missing it is training at
1e-5. Nothing already measured is invalidated (every prior resume re-used its original rate). Mechanism,
measurement and the two general lessons:
[`findings.md`](../findings.md#-snek_learning_rate-was-silently-discarded-by-every-resume--adams-rate-rides-in-the-checkpoint).

## Retired from `runs.md` 2026-08-22 — closed-batch status sections (b31-b41, the gate ladder, b20-b26)

Every batch below is closed and has a per-batch write-up in [`completedRuns.md`](../completedRuns.md), a
chart section in [`charts.md`](../charts.md) or [`charts-archive.md`](charts-archive.md), and its conclusion
in [`findings.md`](../findings.md). These were the *status* paragraphs `runs.md` carried while they ran,
kept because the summary tables index the family faster than the write-ups do.
**`b41` is the exception: it finished on the desktop and has no write-up anywhere** — it is carried as an
open item in `runs.md`.

## Batch 40 — the free-space term stacked on the record — **closed 2026-08-18: null, and it retires b29's record region**

**Result.** Pooled equal-effort **85.68 / 88.28 / 89.11 / 89.52** (mean **88.15**, a dead heat with b29's 87.83
and b35's 88.20). **Two arms produced a flawless 100.0%/100 checkpoint** (`b40a` @1562k, `b40b` @1424k) and all
four reached the ≥98%/100 tier — 16 / 63 / 9 / 2 = **90 candidates**, close to b29's own 59/64/9/1. **One held
≥98% over 500 episodes**: `b40b` @1513k at **98.2%/500**, third-best /500 on record.

**Verdict: the free-space term is a null**, and read with `b37` (b29's config on fresh seeds, **0 of 4** held) it
also retires the claim that gate 75 produces a *record region* — three batches with indistinguishable /100 tiers
produced 21, 1 and 0 held checkpoints, so that count is seed noise. Full account in
[`findings.md`](../findings.md#-corrected-2026-08-18-the-record-region-does-not-replicate--the-98500-count-is-seed-noise-and-pooled-is-the-only-metric-of-this-family-worth-reading);
rationale, per-arm rows and charts in [`completedRuns.md`](../completedRuns.md) and
[`charts.md`](../charts.md#batch-40--chase-safe-plus-a-global-free-space-term--done-on-the-desktop-null-and-it-makes-b29s-record-region-look-like-seed-luck).

**`b40b` @1513k is a HOF-promotion candidate** — 98.2%/500 sits behind `b29b` (99.0) and `b29a` (98.4) and ahead
of `b24b`/`b24d` (98.0). Promotion is still the manual, verified process; it has not been done.

## Batch 41 — b29 re-run on the **same seeds** — **queued on the desktop** (2026-08-18)

**Sharpens the b37 finding by removing the seed as a variable.** b37 showed b29's /500 "region" does not
replicate on *fresh* seeds; b41 re-runs b29's exact config on the **same** seeds 1-4 to ask whether it
replicates even then. The training is not bit-reproducible — `ParallelPyEnvironment` worker ordering and TF's
threaded FP reductions diverge and compound in the RL loop — so a same-seed re-run is *expected* to diverge,
and the gap `b41x` vs `b29x` measures the **process-noise floor** every seed-matched comparison in this folder
sits on top of, a number nobody has pinned.

| | |
|---|---|
| arms | `b41a-d-b29repro-seed{1..4}`, seeds 1-4, **2M**, priority 30, desktop (same host as b29) |
| config | `b29` verbatim, no free-space term (behaviourally identical under current code, which defaults it off) |
| reads | per-seed vs `b29a-d`: pooled / best-30 / `sef`, the >=98%/500 count, and how far each curve tracks its twin before separating |

**What each outcome means.** Curves that track a long time then separate, pooled within a point or two -> the
floor is small and n=4 seed-matched verdicts here are trustworthy. Wide divergence from step ~0 -> the floor
is large, and *every* n=4 verdict in this folder (the b40 null included) carries that much irreducible noise.
Either way, if `b29b`'s 99.0%/500 does not reappear on its own seed, that is the strongest confirmation yet
that the /500 record was noise rather than a property of the seed **or** the config.

**Caveat:** current desktop code (`>=6bdbe7c3`) is not byte-identical to what b29 first ran — the free-space
addition is off/no-op, but intervening commits make this "current code, same seed", not a binary replay. The
FP/threading floor above dominates any such drift.

## ‡ The desktop marked 14 publishes `done` that never reached the `results` branch (2026-08-18)

**Its DNS resolution for `github.com` is flapping**, and `publish_results` has **no retry**: the artifacts are
written, the ledger records `done`, the push fails once, and nothing tries again. 122 `publish_status` failures
and **14 `publish_results` failures** since 2026-08-17, the last at 07:27 — all four `b40` HOF-500 files plus
`b40b`'s whole close-out were sitting on the box unpublished while `status.json` said `done`.

**Nothing was lost, and the recovery is cheap.** A failed push leaves the commit local, so **the next successful
results push carries the backlog with it** — which is why b35's and three of b37's close-outs eventually appeared
and b40's did not: no results job has completed since 07:27, and the queue is now empty. Retrieved by hand with
`rsync the-claw-den:Snek/snek2/runs/<file> runs/`.

**Two things to internalise.** A `done` in the ledger means "the job finished", **not** "the results are
published" — check `git ls-tree --name-only origin/results:results | grep <policy>` before concluding a HOF pass
found nothing, because *no file* and *an empty file* are the same absence over the git bus, and this batch looked
like four empty HOF passes. And **the fix belongs in the daemon**: either retry `publish_results` on the next
poll, or reconcile at idle by pushing whenever the local `results` worktree is ahead of `origin/results`. That is
a code change and has not been made.

## C51 pilot — closed at 600k, and it handed off to batch `b31` by itself (2026-08-15)

Distributional RL, phase 3 of
[`../plans/distributional-c51.md`](../../plans/distributional-c51.md). The implementation is committed
(`245cf914`).

| | |
|---|---|
| wave A, 15:06 | `c51pilot-lr1e5seed{1,2}`, `c51pilot-lr5e5seed{1,2}` |
| wave B, 16:41 | `c51pilotB-lr1e4seed{1,2}`, `c51pilotB-lr25e4seed{1,2}` |
| config | b25's verbatim (`fc 200,100,100`, `IS_WEIGHTS=0`, `TARGET_UPDATE_PERIOD=1000`, `DISCOUNT=0.9975`, `FORK_BRANCHES=4`, no food-distance shaping) plus `ALGO=c51`, 51 atoms over `[-5, 120]`. Seeds matched across all four rates |
| cap | **600k steps** — a screen, not a result |
| launchers | `launch_c51_pilot.sh` (wave A, which waited out b30's close-out), then the generic `launch_c51_wave.sh` (wave B) |

**What it is asking**, in order: does a categorical agent learn this task at all; how many steps to its
first perfect game against b25's ~9k; is the loss scale sane at 1e-5 (a cross-entropy starts at
`ln 51 ≈ 3.93`, where the Huber TD loss starts near 0, so the same learning rate is not obviously the same
step size). **The gate to phase 4 is one learning rate.**

**Eight trainers on a 14-core laptop is deliberate**, and measured before launching: ~2.3 GB per arm
(0.4 GB parent + 1.9 GB across 11 forked self-eval workers) against 36 GB of RAM, and a swap-in rate of
244 pages per 20 s, so the cost is throughput — roughly half the steps/s per arm — and not paging. It is
also the one place this project's "never more than 4 trainers" rule is knowingly suspended; it was the
user's call, for this screen only.

### The handoff to `b31` ran unattended, and worked

**`launch_c51_batch.sh b31` fired at 20:09**, ~3h20m after it was armed: all eight pilot arms reached the
600k cap, it picked `5e-5`, launched `b31a-d` at 2M, regenerated the tables below and pushed the result as
`c45e8a4f` — with nobody watching any of it. Written this way because cron jobs
in this tool are session-only and fire only while the REPL is idle, so nothing scheduled can be relied on
once the session closes — a detached `nohup` can.

| step | what it does | what it refuses to do |
|---|---|---|
| wait | polls until no `snek2.py c51pilot` trainer remains (one substring covers both waves) | it excludes `chart_viewer` from that `pgrep`, or it would wait forever on the window rather than the arms |
| slots | waits for the laptop's 4-trainer limit to be free | if something else is still training after 6 h it **exits without launching** rather than breaking the limit |
| pick | [`pick_c51_lr.py`](../pick_c51_lr.py)'s pre-registered rule — mean `best_perfect30` at a **common horizon**, then `sef`, then `peak_trailing` | if fewer than two rates have usable data it refuses, and the launcher falls back to `5e-5` so a batch still starts |
| launch | `b31a-d`, 4 seeds, 2M, the chosen rate, otherwise identical to the pilots — so `b25a-d` is the seed-matched control | staggers the four by 5 s rather than leaning on the chart viewer's claim lock with nobody watching |
| docs | regenerates the marked region in this file and `charts.md`, then commits and pushes | a push failure is logged, not retried — the commit is local and recoverable |

**Three guards in the picker exist because a dry run got the answer wrong.** At an early horizon every arm
reads `best_perfect30` 0.0 *and* `sef` 0.0, and a two-level rule then picked whichever rate came out of a
dict first — it chose the slowest rate over the fastest, so `peak_trailing` is now the third key. An arm
that dies early no longer sets everyone's horizon (it would have judged seven healthy arms at 13k of 600k).
And an arm with no eval series is excluded and named rather than counted as a zero, so a failed launch
cannot vote against its own rate.

<!-- C51-PILOT-STATUS:BEGIN -->
*Generated by `pick_c51_lr.py` at 2026-08-15 20:09, when the last pilot arm stopped — the numbers below are read straight off the eval series, and the prose around this block is hand-written.*

**Compared at a common horizon of 600k steps**, the lowest final step any arm reached, because both metrics accumulate over an arm's own evals and a longer arm would otherwise win on horizon alone.

| lr | seeds | mean best-30 | mean `sef` | mean peak trail |
|---|---|---|---|---|
| 5e-05 **← chosen** | 2 | 69.5 | 12.6 | 92.42 |
| 1e-05 | 2 | 56.5 | 3.6 | 89.89 |
| 0.0001 | 2 | 39.0 | 5.3 | 88.19 |
| 0.00025 | 2 | 4.0 | 0.0 | 68.79 |

| arm | lr | seed | step | best-30 | `sef` | peak trail | first perfect |
|---|---|---|---|---|---|---|---|
| `c51pilot-lr1e5seed1` | 1e-05 | 1 | 600k | 85.3 | 7.3 | 93.56 | 141k |
| `c51pilot-lr5e5seed2` | 5e-05 | 2 | 600k | 71.7 | 13.0 | 93.30 | 20k |
| `c51pilot-lr5e5seed1` | 5e-05 | 1 | 600k | 67.3 | 12.1 | 91.54 | 15k |
| `c51pilotB-lr1e4seed2` | 0.0001 | 2 | 600k | 66.3 | 10.6 | 90.80 | 46k |
| `c51pilot-lr1e5seed2` | 1e-05 | 2 | 600k | 27.7 | 0.0 | 86.22 | 92k |
| `c51pilotB-lr1e4seed1` | 0.0001 | 1 | 600k | 11.7 | 0.0 | 85.58 | 8k |
| `c51pilotB-lr25e4seed1` | 0.00025 | 1 | 600k | 5.7 | 0.0 | 70.82 | 49k |
| `c51pilotB-lr25e4seed2` | 0.00025 | 2 | 600k | 2.3 | 0.0 | 66.76 | 59k |

**Chosen: `5e-05`** — best_perfect30 69.5 against 56.5 for the next rate (1e-05).

**Batch `b31` launched at 2026-08-15 20:09** on this rate, 4 seeds, 2M cap, `fc 200,100,100`, otherwise b25's config — so `b25a-d` is the seed-matched control.
<!-- C51-PILOT-STATUS:END -->

**`SNEK_CHART_VIEWER=0` on all four, one window opened by hand.** `chart_viewer.batch_prefix` groups only
`b<n><letters>-` names, so four `c51pilot-*` arms would open four windows; the launcher uses the
`--glob`/`--watch` form an eval wave already uses. The pilot deliberately does **not** claim `b31` — `fc 512`
and the four owed `320` seeds are ahead of C51 in the backlog below.

## Batch 39 — **C51 initialised at expected Q = 0** — **closed 2026-08-18 at the 3M cap: it loses on every metric**

**Result.** Matched at ≤1.87M and seed-paired against `b36`, **−9.4 pp** best-30 and **−7.1 pp** `sef`, 4 of 4
seeds down; pooled **70.18 vs 76.76**, also 4 of 4. **All 650 close-out rows were abandoned under the 95% gate**
— the batch produced no measurable checkpoint at all, where b36 produced 4 and b38 5. Pre-registered **H2
confirmed, H1 falsified**.

**But the predicted mechanism was wrong**, and that is the transferable part: zero-init converged its value
*level* **faster** (half-life 163-202k vs b36a's 304k) from a *larger* initial error, so calibration is not the
channel. The channel is **action separation** — b36a reaches a 12.18 action gap by **8k steps**, b39 sits at
**1.72** and needs ~600k to reach 8.90. **Judge a categorical init by the spread it leaves available, not by how
close its mean is to the truth.** `SNEK_C51_ZERO_INIT` stays off, now on measured grounds. Full account in
[`findings.md`](../findings.md#-zero-init-loses-and-the-channel-is-action-separation-not-calibration--b39-closed-at-3m);
pre-registration, per-arm rows and charts in [`completedRuns.md`](../completedRuns.md) and
[`charts.md`](charts-archive.md#batch-39--c51-initialised-at-expected-q--0-instead-of-the-grid-midpoint--closed-at-the-3m-cap-it-loses-on-every-metric-through-the-heads-capacity-rather-than-its-calibration).

## Batch 38 — b36's config at the **other** Adam epsilon (`3.125e-4`) — **closed 2026-08-17: dead heat, the dose question is settled**

**All four hit the 3M cap and self-terminated**, then closed out at gate 95. At a **matched ≤2M horizon**
b38 pools **74.73 against b36's 76.77** — 3 of 4 seeds favour `1.5e-4`, mean **−2.04 pp**, sign test
**p=0.625**. Best-30 80.0-88.3 vs 84.0-86.7 and best checkpoint 93.4-96.0 vs 91.6-97.0 agree. **No arm
≥98%, so no HOF-500.** So `1.5e-4` stays the default as the lower-variance reference and **the dose is
closed for good** — b32 could not separate the two at n=2, and n=4 says there is nothing to separate.

**Two by-products worth more than the dose answer itself.**

1. **C51 does not benefit from running past ~2M.** Pooling all rows against ≤2M only, **3 of 4 arms got
   worse** past 2M; `b38a` is the exception and holds the batch's best checkpoint at **2355k**. So future
   C51 batches can stop at ~2M — the horizon question b36's launcher raised, answered.
2. **`pooled_equal_effort` is exactly recomputable at any horizon** from each row's stored
   `episode_perfect` flags truncated to the screen depth. Verified by reproducing all 8 published figures
   to the decimal. That removes the horizon caveat that made the first b38-vs-b36 reading unquotable, and
   it is the method to use whenever two arms stopped at different steps.

Per-arm table in
[`archive/charts-archive.md`](charts-archive.md#batch-38--adam-ε-3125e-4-on-b36s-fc-320--closed-the-dose-is-a-dead-heat-at-n4-as-pre-registered)
— batch 38's chart section was retired there on 2026-08-18 to make room for 42/43.

**`launch_b38_eps3125.sh`, identical to b36 with `SNEK_ADAM_EPSILON=3.125e-4` the only change**, seeds 1-4,
3M cap. `b36a-d` is therefore an exact seed-matched control and this is a clean one-variable dose comparison.
**Launched automatically** by `chain_after_evals.sh`, which polls for `eval_checkpoints.py` to drain and then
runs the launcher — log at `/tmp/chain-b38.log`.

**This is the dose question b32 could not answer, retried at 4 seeds a side instead of 2.** b32's shared-set
churn put `1.5e-4` at **0.0865** and `3.125e-4` at **0.0895** — nothing, on n=2, exactly as pre-registered.
b36 + b38 is **the first configuration in this project with 4 seeds per side on one architecture**, so it is
the first that can say anything about the dose at all.

**Read churn first, and only with `--states-from`** — the same reference every C51 reading now uses, so the
numbers stay comparable across batches:

```
PYTHONPATH=. python hyperparamTuning/perDiagnostics/c51_stability.py \
  --policy b36a-c51fc320seed1 --policy b38a-c51fc320eps3125seed1 ... \
  --states 1500 --stride 5000 --points 10 --end 2000000 \
  --states-from hallOfFame/b29b-chase10g75seed2-ckpt1447000
```

| outcome | reading |
|---|---|
| lower churn at `3.125e-4`, best-30 held | the higher dose becomes the C51 default, and the response is still climbing — worth one more rung |
| lower churn, **worse** best-30 | `epsilon` acting as a smaller learning rate in disguise, the known failure mode. This is why best-30 is read *alongside*, not after |
| dead heat at n=4 a side | the dose question closes for good; `1.5e-4` stays the default as the lower-variance reference config |

**b36 stopped at 2M, so match there** rather than at b38's 3M cap.

## Batch 36 (C51 on `fc 320`) and Batch 38 (its Adam-ε dose) — **both closed 2026-08-17**

**Full narratives moved to [`completedRuns.md`](../completedRuns.md#batch-36--c51-on-fc-320-the-better-c51-shape-and-still-far-behind-ddqn), per the runs.md/completedRuns.md split.** The verdicts:

- **`fc 320` is the better C51 shape** — best-30 84.0-86.7 against `b32`'s 77.0/63.0, seed spread 14.0 → 2.7 pp.
- **C51 is still far behind `ddqn` at the identical shape.** `b24a-d` pools 85.97-89.03 with a ≥98% checkpoint
  in every seed; b36 pools 74.77-80.19 and b38 71.79-78.51, **neither batch reaching ≥98% once**.
- **The Adam-ε dose is closed for good** at 4 seeds a side: matched ≤2M, 76.77 vs 74.73, p=0.625.
- **C51 gains nothing past ~2M**, so future C51 batches can stop there.
- **Init optimism is excluded** as the remaining suspect, which is what `b39` now tests directly.

## Batch 32 — Adam's `epsilon` on C51 — **closed 2026-08-16**

**Full narrative moved to [`completedRuns.md`](../completedRuns.md#batch-32--adams-epsilon-on-c51-it-works-at-26-churn-and-the-dose-does-not).** Shared-state-set churn **0.119 → 0.088 (−26%)**, 4 of 4 paired, flat 600k→1M, dose a dead heat — and the dose
stayed a dead heat when b36+b38 retried it at 4 seeds a side. Its chart section is in
[`archive/charts-archive.md`](charts-archive.md); **b32a-d were never closed out**, so they have no rows
in the canonical table.

## Batch 31 — C51 at `lr 5e-5`, 2M — **stopped at 538-569k, no close-out** (2026-08-15)

Launched 20:09 by the pilot's handoff, **killed 23:10 at the user's call** after
[`c51_stability.py`](../perDiagnostics/c51_stability.py) showed the chaos was the learning rate rather than
C51, which made a 2M run at a rate chosen under the old reading not worth the slots. **No close-out was
run** — deliberately, not pending.

Reached 538-569k in 2h44m, all four healthy (no zero stretch), best-30 **21.0 / 53.3 / 66.7 / 71.7** — a
**50.7 pp spread at one config**, which is the n=4 noise problem restated rather than a result. Graphs in
[`charts.md`](../charts.md); the arms are in [`completedRuns.md`](../completedRuns.md) as void.

## Batch 34 — chase-safe `c=0.10`, **gate 70** — *done on the desktop: null*

**Closed 2026-08-16 (training + close-out + HOF-500), and the pre-registered "gate 70 < gate 75" outcome
landed: gate 75 is a narrow sweet spot.** Identical to b29 with `SNEK_CHASE_SAFE_GATE=70` the only change.
Pooled equal-effort **86.4** (~1.5 under the b24 control and just under b29's 87.8) and **0 of 4 seeds held
any ≥98%/500 checkpoint**, against b29's 21 across two seeds — so a single 5-length step off 75 already
collapses the record region. All four healthy throughout (peak 95.00, no zero stretch). This makes the gate a
**band around 75**, not a threshold: 85 null, 75 records, 70 null again. Full numbers and per-arm table in
[`completedRuns.md`](../completedRuns.md#batch-34--chase-safe-c010-gate-70-null--gate-75-is-a-narrow-sweet-spot-not-a-threshold);
finding: [`findings.md`](../findings.md#-chase-safe-reward-shaping-null-at-gate-85-at-any-dose-records-at-gate-75--the-gate-is-the-lever).

## Batch 37 — **b29 replication on fresh seeds 5-8** — **closed 2026-08-18: the /100 band replicates, the /500 record does not**

**Result.** Pooled equal-effort **80.72 / 82.19 / 87.88 / 90.50** (mean **85.32**; `b37b`'s 90.50 is the highest
single arm of the chase-safe family). The ≥98%/100 tier reproduces b29's **2-of-4** shape — 43 and 16 candidates
in two seeds, none in the other two — and **0 of 4 seeds held ≥98% over 500 episodes**, where b29 held 21; the
two best were abandoned under gate 98 at ~360 episodes (97.0%, 96.9%).

**Verdict: the outcome the pre-registration called "b29's region was lucky seeds".** Read with `b40`, three
batches with indistinguishable /100 tiers produced held counts of 21, 1 and 0, so **the ≥98%/500 count is seed
noise** and this family must be judged on pooled. The correction is in
[`findings.md`](../findings.md#-corrected-2026-08-18-the-record-region-does-not-replicate--the-98500-count-is-seed-noise-and-pooled-is-the-only-metric-of-this-family-worth-reading);
rationale, per-arm rows and charts in [`completedRuns.md`](../completedRuns.md) and
[`charts.md`](../charts.md#batch-37--b29-replicated-on-fresh-seeds-5-8--done-on-the-desktop-the-100-band-replicates-the-500-record-does-not).

## Batch 35 — chase-safe `c=0.10`, **gate 40** — *done on the desktop: null*

**Closed 2026-08-17, and the pre-registered "gate 40 < gate 70/75" outcome landed — mid-game shaping is a
null on records.** The ladder's deep rung: `b29`'s config with only `SNEK_CHASE_SAFE_GATE=40` (per-flip dose
held at the 0.10 clamp; total episode dose ~2.5× gate 85). **0 of the 3 measured seeds held any ≥98%/500
checkpoint** (`b35c`'s HOF-500 was still running at check time), best partials abandoned at 96-97%. The twist:
gate 40 posts the **highest pooled equal-effort of any shaped batch (88.2**, above b29's 87.8 and the b24
control's 87.9) — so it grades the mid-game into a healthier *average* board without ever reaching the
record-tier endgame. **Consolidation and the record tier are decoupled**, and across four gates (85, 75, 70, 40)
**only 75 records** — the sweet spot is a narrow, isolated band. Full numbers and per-arm table:
[`completedRuns.md`](../completedRuns.md#batch-35--chase-safe-c010-gate-40-null--the-sweet-spot-at-75-is-isolated-not-a-plateau);
finding: [`findings.md`](../findings.md#-chase-safe-reward-shaping-null-at-gate-85-at-any-dose-records-at-gate-75--the-gate-is-the-lever).

## The chase-safe gate ladder is complete (b27-30, 34, 35, 37, 40) — 75 leads on the /100 tier, and its /500 "region" was seed noise

**Six batches walked the length gate from 85 down to 40, and only 75 records.** All are closed, so per the
bookkeeping rule their descriptions moved to [`completedRuns.md`](../completedRuns.md). The shaping adds
`c·(γΦ(s′) − Φ(s))` with Φ = 1 iff the head and tail share a free region holding the food and the snake is
≥ gate long — potential-based, optimal policy unchanged ([plan](../../plans/chase-safe-reward-shaping.md);
Phase 0 Φ calibration in
[`findings.md`](../findings.md#-measured-the-chase-safe-potential-is-nearly-static-for-a-record-policy-and-busy-for-a-bad-one)).
All arms are **b24's config plus the one knob** (`fc 320` for b27/b28/b29/b34/b35/b37/b40, `fc 200,100,100` for
b30), so `b24a-d`/`b25a-d` are the seed-matched controls at zero extra compute. Cap 2M.

**‡‡‡ Closed 2026-08-18, and the headline is qualified.** `b37` re-ran gate 75 on fresh seeds 5-8 and held
**0 of 4** at ≥98%/500; `b40` added a free-space term and held **1**; b29 held **21** — on
indistinguishable ≥98%/**100** tiers and tied pooled. So **the /500 count is seed noise**, gate 75's lead lives
on the /100 tier and on pooled, and the ladder's nulls at 85/70/40 stand (they are null on *both* tiers). Read the
[correction](../findings.md#-corrected-2026-08-18-the-record-region-does-not-replicate--the-98500-count-is-seed-noise-and-pooled-is-the-only-metric-of-this-family-worth-reading)
before quoting the 21.

| batch | `c` | gate | net | verdict | write-up |
|---|---|---|---|---|---|
| **27** | 0.10 | 85 | `320` | **null** — pooled 85.2 vs b24 87.9, 0 of 4 ≥98%/500 | [Batch 30 2×2](../completedRuns.md#batch-30--chase-safe-shaping-on-fc-200100100-c010-null-and-it-completes-the-shapingarchitecture-22) |
| **30** | 0.10 | 85 | `200,100,100` | **null** — pooled 83.3 vs b25 86.0, 0 of 10 held | [Batch 30](../completedRuns.md#batch-30--chase-safe-shaping-on-fc-200100100-c010-null-and-it-completes-the-shapingarchitecture-22) |
| **28** | **0.20** | 85 | `320` | **null** — pooled 85.4, 0 of 4 held; the dose is not the issue | [Batches 28-29](../completedRuns.md#batches-28-29--chase-safe-dose-and-gate-the-gate-is-the-lever-and-gate-75-produces-a-record-region) |
| **29** | 0.10 | **75** | `320` | **records** — pooled 87.8, **21 held ≥98%/500 in 2 seeds**, best `b29b` @1447k **99.0%/500** | [Batches 28-29](../completedRuns.md#batches-28-29--chase-safe-dose-and-gate-the-gate-is-the-lever-and-gate-75-produces-a-record-region) |
| **34** | 0.10 | **70** | `320` | **null** — pooled 86.4, 0 of 4 held; a 5-length step off 75 loses it | [Batch 34](../completedRuns.md#batch-34--chase-safe-c010-gate-70-null--gate-75-is-a-narrow-sweet-spot-not-a-threshold) |
| **35** | 0.10 | **40** | `320` | **null** — 0 of 3 held (b35c pending), *highest pooled 88.2*; consolidation ≠ records | [Batch 35](../completedRuns.md#batch-35--chase-safe-c010-gate-40-null--the-sweet-spot-at-75-is-isolated-not-a-plateau) |

**The lever is the gate, and 75 is an isolated sweet spot.** Gate 85 is null on `fc 320` (b27), on
`fc 200,100,100` (b30) and at doubled dose (b28); gate 70 (b34) and gate 40 (b35) are null too; only gate 75 (b29) matches the
control's pooled *and* produces a record region the control never did. The Φ calibration is why: the potential carries ~0 at lengths 98-99, so a gate-85
term grades the flat final approach, while gate 75 turns it on ten meals earlier, in the packing decisions
that decide whether the endgame is winnable. Full conclusion:
[`findings.md`](../findings.md#-chase-safe-reward-shaping-null-at-gate-85-at-any-dose-records-at-gate-75--the-gate-is-the-lever).

## Batches 20-26 are closed — where their descriptions went

All seven batches finished and closed out, so per the bookkeeping rule at the end of this file their
descriptions moved to [`completedRuns.md`](../completedRuns.md):

| batch | change | verdict | design + results |
|---|---|---|---|
| **26** | **fc `100,100`** under IS-off | **does not carry the lift** — pooled 79.2, only +3.5 over the control, at **more** parameters than b24's `320`. The arm that showed the lift is width, not size | [write-up](../completedRuns.md#batch-26--fc-100100-under-is-off-the-shallow-shape-does-not-carry-the-lift) |
| **25** | **fc `200,100,100`** under IS-off | **the lift replicates** — pooled 86.0, +10.3, all 4 seeds; peak unmoved. No hall entry: the auto chain's gate-98 abandoned every candidate, `b25b` @911k still 97.2% at 392 episodes | [write-up](../completedRuns.md#batch-25--fc-200100100-under-is-off-the-lift-replicates-at-a-second-shape--but-no-record) |
| **24** | **fc `320`** under IS-off | **first architecture result + a new record** — pooled **87.9**, +12.2 over the control, all 4 seeds (ceiling unmoved). HOF-500: `b24d` @1342k **98.0%/500**, the new record | [write-up](../completedRuns.md#batch-24--fc-width-320-under-is-off-the-first-architecture-result-and-a-new-record) |
| **23** | IS β annealed **0→0.1** | **the best point on the β ladder** — pooled **75.7**, +20.7 over the control, higher on all 4 seeds | [write-up](../completedRuns.md#batch-23--β-annealed-001-the-best-point-on-the-β-ladder-near-the-no-is-extreme) |
| **22** | IS **off** (`SNEK_IS_WEIGHTS=0`) | **dead heat with β→0.1** — pooled 75.7. The consolidation gain saturates by β→0.1 | [write-up](../completedRuns.md#batch-22--is-off-a-dead-heat-with-β01--the-consolidation-gain-saturates) |
| **21** | partial IS (β→**0.5**) | beats the β→1.0 control (pooled 64.3 vs 55.0, 3/4 seeds), well short of no-IS | [write-up](../completedRuns.md#batch-21--partial-is-β05-beats-the-β10-control-still-far-behind-no-is) |
| **20** | `FC_LAYERS`, **nine shapes** | **architecture never raises the ceiling**; capacity binds only below ~0.55× | [the sweep's design](../completedRuns.md#batch-20--the-design-of-the-nine-shape-sweep-complete-2026-08-12) |

**‡ The width result now has three shapes behind it, and "it tracks capacity" is retracted.** Pooled lift
over the b22 control orders by **widest layer**, not by size: `320` **+12.2** at 0.94× the control's
parameters, `200,100,100` **+10.3** at 3.09×, `100,100` **+3.5** at 1.14×. The smallest of the four nets
wins and the second-largest gets almost nothing, so parameter count is not even monotone with the result
([finding](../findings.md#-corrected-2026-08-14-the-is-off-architecture-lift-tracks-the-widest-layer-not-the-parameter-count)).
**The architecture arm this implies is `fc 512` under the b24 config** — a wider first layer, not a bigger
net — and it is now the strongest remaining consolidation direction after the shaping batches.

**The β ladder is the live result to build on.** Gradient concentration ESS/N walks down it — **β→1.0
≈1.0** (batch 20's control, near-uniform) → **β→0.5 ≈0.86** (b21) → **β→0.1** (b23, effective exponent
α·(1−β)=0.54) → **IS off ≈0.38** (b22) → b18's no-IS ≈0.21 — and pooled climbs monotonically with it:
**55.0 → 64.3 → {75.7, 75.7} → ~78.8**, then **flattens at the bottom**. Most of the consolidation is
bought by β→0.1; going all the way to IS-off adds nothing measurable. **The ceiling is unmoved throughout**
(peak 94.4-94.9), so the ladder buys time-near-the-ceiling, not a higher one.

`b23b` holds five full-length checkpoints ≥95/100 around 777k (best 97/100). It was **not** a
hall-of-fame candidate after re-measurement — the selected @777k reads 92.4% over 500 fresh episodes, the
worst of its own cluster. The `b23b` 217-242k collapse was investigated in place and is **not** an escape
from a local minimum; all four seeds make the same level shift
([`findings.md`](../findings.md#-falsified-a-drawdown-is-not-how-a-policy-escapes-a-local-minimum)).

## Retired from `runs.md` 2026-08-22 — max progression across batches, and the batch 11-19 one-liners

Historical. Per-seed numbers are in [`completedRuns.md`](../completedRuns.md).

### Max progression across batches

Best checkpoint per batch, at most three each, newest first. **Two columns, because they say
different things:** `selected` is the close-out's own best row and reads high by construction;
`re-measured` is a later independent sample and is the only column that can be compared across
batches. `*trunc*` means no full-length row survived the gate, so the figure is shorter and noisier.

| batch | change | best selected | re-measured |
|---|---|---|---|
| **24** | **fc `320`** under IS-off | 100% @1633k /100 (`b24a`) · 100% @2126k /100 (`b24c`) · 99% @1031k /100 (`b24b`) | **98.0% /500** ← **new record** (`b24d` @1342k) · 98.0% /500 (`b24b` @2860k) · 97.4% /500 (`b24c` @2982k) |
| **20** | `FC_LAYERS` shapes (**all 9 closed**) | 92% @1470k *trunc* (`b20ah`, 100,200,100) · 91% @1435k *trunc* · 91% @2935k *trunc* (`b20g`) | **none reached full length** — 0 of 36 arms |
| **19** | standard PER + IS | 91% @1536k *trunc* · 77% @1485k *trunc* · 76% @937k *trunc* | **none reached full length** |
| **18** | `TARGET_UPDATE_PERIOD` 1000 | **98% @1588k** · 97% @1601k · 96% @1289k | **97.6% /700** ← **record** · 94.7% /700 · 85.4% /500 |
| 17 | forked endgame collection | 99% @1248k · 99% @1205k · 98% @1231k | **94.24% /5120** · region grid 84% |
| 16 | `FOOD_DISTANCE_REWARD=0` | 93% @913k *trunc* · 92% @1203k · 85% @979k *trunc* | — |
| 15 | `N_STEP_UPDATE=3` | 97% @3245k · 95% @4697k · 91% @3671k | **93.0% /300** |
| 14 | `DISCOUNT=0.9975` | 96% @3702k · 93% @2559k · 90% @2261k | **93.5% /200** |
| 13 | eps handover + shield | 95% @986k · 92% @3367k · 91% @1166k | — |
| 12 | eps handover 0.05 | *deadlocked, not measured* | — |
| 11 | the 30-value vector | 96% @855k · 94% @671k · 88% @3507k | **~94%** (shrunk) |
| 10 ‡‡ | `DISCOUNT=0.995` | 93% @1695k · 90% @1501k · 85% @2344k | 74.9% /66000 pooled |
| ≤9 | earlier environments | `b8f` 92%, `b9d` 70%, `b7f` 51%, `b4c` 50% | not comparable |

**Read the re-measured column and the story is short: 94.2% for a year of batches, then 97.6%.**
Batches 11-19 all train on the same 30-value vector so they are comparable to each other; batch 10
(‡‡) and everything below it are earlier environments where the same checkpoint scores differently,
which is why those rows are not a trend line.

**The selected column is nearly flat from batch 11 on** — 93-99% in every batch that produced a
full-length row — which is exactly why it cannot be used to judge progress. Batch 17's three 98-99%
rows re-measured to 94.2%; batch 18's 98% re-measured to 97.6%. Same selected number, 3.4 pp apart in
reality.


## Closed batches (11-19)

One line each; **batches 20-26 are in the table above**, with write-up links. Full write-ups and
per-seed numbers in [`completedRuns.md`](../completedRuns.md), superseded detail in
[`archive/runs-archive.md`](runs-archive.md).

| batch | change | verdict |
|---|---|---|
| **19** | standard PER (`td_error` + IS on, β→1.0) | **falsified** — worse on all 5 pooled metrics, 4/4 seeds, p=0.125. Drawdown 55.5 → 8.8, but at a lower level |
| 18 | `TARGET_UPDATE_PERIOD` 8 → 1000 | primary moved: 102k faster to pf30≥40%, 4/4 seeds; drawdown improved. **Close-out done** — tightest eq-effort spread of any batch (74.8-81.9), 20 rows ≥95% |
| 17 | `FORK_BRANCHES=4` forked endgame collection | null on config (one seed carried it), **project record on `b17b`**; dose ~60% of design |
| 16 | `FOOD_DISTANCE_REWARD=0` | **first non-null** — `sef` +11.35 pp at 1.25M; consolidation not ceiling. Needs replication |
| 15 | `N_STEP_UPDATE=3` | falsified on speed — 128k slower; evals null |
| 14 | `DISCOUNT=0.9975`, `GUIDED_FRACTION=0.8` | null vs 13 |
| 13 | eps handover 0.0125 + shield 0.5 | null vs 11 on five metrics |
| 12 | eps handover 0.05 | deadlocked, abandoned 4/4 |
| 11 | the 30-value vector itself | +4-5 pp vs 10, not significant |

**The binding constraint is seed variance, not ideas** — n=4 resolves nothing below ~10 pp, and peak
trailing reads 93.8-95.0 flat across batches 11-23. Nothing has raised the ceiling; batches 16 and 21-23
raised how much of the *time* an arm sits near it. **Two things have moved peak trailing downward** —
batch 19's full IS correction (94.16, 4/4 seeds) and batch 20's 0.29× net (93.75, 4/4) — so the invariance
is breakable, just not yet upward. **Fifteen batches of optimiser, PER and architecture knobs have not
raised the ceiling once** — peak trailing reads 94.84-95.00 across `50,100,50`, `100,100`,
`200,100,100` and `320` — which is the argument behind the reward-shaping batches now running (b27-b29).
