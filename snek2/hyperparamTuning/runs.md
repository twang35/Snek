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

## The current record: 96%, and it runs on `master`

`b11b-obs30seed2` @855000 scored **96/100** (CI 90.2-98.4), found by batch 11's close-out on
2026-08-04, and preserved in [`../hallOfFame/`](../hallOfFame/README.md). It is the first record in
a while that loads on `master` as it stands, because batch 11 is the only batch trained on the
current 30-value vector.

**Corrected for the winner's curse it is ~94%.** It is the maximum of 204 full-length measurements
in its own arm. Shrinking it against a Beta prior fitted on `b11b`'s 104 *unselected* graph-100%
rows gives 94.0% — a much smaller haircut than the previous record's, because this measurement is
100 episodes rather than 30. Details and the reason the unselected tier is the only clean prior:
[`completedRuns.md`](completedRuns.md#a-new-best-measured-checkpoint-b11b-855000-96100).

**The previous record, `b10d-disc995seed4` @1815000 at 95/100, no longer runs.** The two
observations added 2026-08-03 took the vector to 30 values, so it and every other batch-10
checkpoint fail to restore; **`450e66e` is the last commit with the 26-value vector.** That 95% was
measured mid-run over 30 episodes and shrinks to ~87%, and its own arm's close-out found 93%
@1695000 over a full 100 — treat those two as one policy family measured twice rather than a record
and a near-miss. Full batch results:
[`completedRuns.md`](completedRuns.md#batch-10--a-fresh-baseline-and-a-new-project-record).

## Batch 10 is closed out — its arms are the control for batch 11

**Batch 10 was stopped 2026-08-03 by request**, all four arms healthy (no `dead_since` on any
of them), to make room for further changes. It was launched as a fresh baseline after seven
observation/reward changes landed 2026-08-02 (fatal-move zeroing, wall/body hugging, normalized
group count, the corrected starve/length split, the terminal-discount fix, safe-to-chase-food,
and the audit that started the day) — nothing had trained on the resulting environment before
it. All four seeds beat every prior post-audit result; full design and results:
[`completedRuns.md`](completedRuns.md#batch-10--a-fresh-baseline-and-a-new-project-record).

**Close-out evals: all four complete.** These four rows are the control batch 11 is measured
against.

| arm | checkpoints | best (100 episodes) | pooled | graph-100% tier |
|---|---|---|---|---|
| `b10d-disc995seed4` | 660 | 93.0% @1695000 | 74.9% | 75.7% (n=146) |
| `b10b-disc995seed2` | 624 | 90.0% @1501000 | 71.8% | 72.9% (n=142) |
| `b10a-disc995seed1` | 272 | 85.0% @2344000 | 67.2% | 68.7% (n=47) |
| `b10c-disc995seed3` | 47 | 79.0% @3965000 | 63.0% | 65.1% (n=7) |

The *pooled* column is over full-length rows only. It is a valid equal-effort arm rate here — these
four predate the screening protocol and were measured flat at 100 episodes each — which is why they
compare directly to each other. It is **not** comparable to a batch-11 pooled figure, where rows
have different depths; the **graph-100% tier** column is the one that crosses the two batches, since
that tier is measured at 100 episodes unscreened in both. See the † note in
[`completedRuns.md`](completedRuns.md).

### Evals got ~10x cheaper on 2026-08-03

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

The early-abandonment idea — cut a checkpoint once its running rate looks weak — was simulated
against all 937 batch-10 measurements and **rejected**: safe thresholds save only 14%, because
the selected population is a tight blob between 60% and 80% rather than a few good runs among
junk. Screening wins by economising on the many mediocre checkpoints instead of the few bad
ones. Full numbers in [`hyperparamTuning.md`](hyperparamTuning.md#screening-eval_screen_episodes-on-by-default).

A close-out is also resumable now (`EVAL_RESUME=1`), which is what made switching the worker
count mid-run cost only the checkpoint in flight rather than the 333 already measured.

### The environment changed again on 2026-08-03: two new observations

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

## Nothing is training, and nothing is evaluating

**Batch 11 stopped 2026-08-04 09:09; its close-outs finished the same day.** Four seeds of batch
10's config on the 30-value vector, and a null result exactly as its own pre-registration predicted.
Full write-up, including two unpredicted differences that are *not* findings:
[`completedRuns.md`](completedRuns.md#batch-11--the-same-config-on-the-30-value-vector-no-significant-difference).

| arm | final step | best-30 perfect | best-30 @3.185M | best ckpt (100 ep) | graph-100% tier | drawdown |
|---|---|---|---|---|---|---|
| `b11b-obs30seed2` | 3.56M | **91.7%** @873k | **91.7%** | **96%** @855k | **81.0%** (n=104) | 18.0 pp |
| `b11a-obs30seed1` | 3.19M | 85.7% @678k | 85.7% | 94% @671k | 79.5% (n=48) | **42.4 pp** |
| `b11d-obs30seed4` | 3.59M | 78.3% @3468k | 76.3% | 88% @3507k | 69.3% (n=40) | 5.6 pp |
| `b11c-obs30seed3` | 3.23M | 73.0% @1718k | 73.0% | 87% @1706k | 69.0% (n=23) | 18.0 pp |

**All three comparisons against batch 10 came out the same way — +4 to +5 pp, none significant:**

| metric | batch 10 | batch 11 | difference | exact p |
|---|---|---|---|---|
| best-30 @3.185M (pre-registered) | 76.2% | 81.7% | +5.4 pp | 0.243 |
| graph-100% tier rate | 70.6% | 74.7% | +4.1 pp | 0.143 |
| best checkpoint | 86.8% | 91.2% | +4.5 pp | 0.157 |

The two new observations are kept under the stated decision rule (keep unless clearly worse). The
agreement across metrics makes a real positive effect more likely than zero, but n=4 cannot separate
+5 pp from +1 pp — which is the whole argument for the next batch being wider.

`b11b` holds both records now: the highest best-30 (91.7%) and the best measured checkpoint (96%).
Both are n=1 of eight arms with heavily overlapping batch means — high-water marks, not findings.

## Proposed next: batch 12 — the epsilon rewrite at n=8, in two waves

| | |
|---|---|
| arms | 8, `SNEK_SEED=1..8`, all identical |
| config | byte-identical to batch 11 plus the new epsilon default |
| cap | 2.5M steps per arm (~7.5 h per wave of 4, from batch 11's ~332k steps/arm/hour) |
| schedule | **two waves of 4** — the 4-trainer cap makes n=8 sequential, one overnight each |
| control | batch 11's four arms, already measured, at a 2.5M horizon |
| pairing | seeds 1-4 pair directly with `b11a`-`b11d` |
| primary metric | `strong_eval_fraction` at 2.5M; `best_perfect30` secondary for continuity |
| decision rule | **keep the schedule unless clearly worse** — revert only on a >10 pp drop |

```
SNEK_SEED=n SNEK_DISCOUNT=0.995 SNEK_PRIORITY_EXPONENT=0.6 \
SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0 \
  /opt/miniconda3/envs/snek/bin/python -u snek2.py b12<x>-eps002seed<n>
```

`eps002` names the floor the way `obs30` named the vector, so it reads as a benchmark label later.
**The control is clean**: the only training-relevant diff since batch 11 launched is the epsilon
schedule and its plumbing, verified by diffing every training file against `83abbd4`.

**This batch cannot prove the epsilon fix helps, and is not meant to.** Even paired, n=4 resolves
~10-15 pp. The fix is justified on mechanism — 96.8% of steps at epsilon exactly 0 is a defect
whatever the effect size — so the pre-registered role of wave 1 is a **regression check** with an
asymmetric rule, and its second purpose is banking the first seeds on the new default. Wave 2
takes n to 8, which is what future knob tests need.

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
- **`N_STEP_UPDATE=5`** — n=2 and n=3 both peak below baseline and then decline, so
  the trend already points the wrong way.
- **Resuming any arm from batch 10 or earlier** — every checkpoint on record from before
  batch 11 was trained on an observation vector this project has since changed (20, 23 or 26
  values against the 30 batch 11 trained on), so none of them load; see
  [`../hallOfFame/README.md`](../hallOfFame/README.md#the-entries-below-predate-2026-08-02-and-do-not-run-on-master).
  **Batch 11 is now the only resumable batch.** Its four arms were stopped healthy at 3.19-3.59M,
  and `b11d` was still near its peak, so resuming *that* arm is the one case worth considering —
  it is also the open question the 2M cap below would truncate.

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
