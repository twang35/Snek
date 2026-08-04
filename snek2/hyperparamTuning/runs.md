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

## The current record: 95%, and it runs on `master` today

`b10d-disc995seed4` @1815000 scored 95/100 (CI 88.8-97.8), measured mid-run, and is preserved in
[`../hallOfFame/`](../hallOfFame/README.md#the-current-record-95-trained-end-to-end-on-todays-environment-2026-08-03).
Unlike every earlier record in this project, **this one needs no checkout** — it trained on the
observation vector that is on `master` right now. Full batch results and the "config vs
environment" open question: [`completedRuns.md`](completedRuns.md#batch-10--a-fresh-baseline-and-a-new-project-record).

## Batch 10 is closed out — its arms are the control for batch 11

**Batch 10 was stopped 2026-08-03 by request**, all four arms healthy (no `dead_since` on any
of them), to make room for further changes. It was launched as a fresh baseline after seven
observation/reward changes landed 2026-08-02 (fatal-move zeroing, wall/body hugging, normalized
group count, the corrected starve/length split, the terminal-discount fix, safe-to-chase-food,
and the audit that started the day) — nothing had trained on the resulting environment before
it. All four seeds beat every prior post-audit result; full design and results:
[`completedRuns.md`](completedRuns.md#batch-10--a-fresh-baseline-and-a-new-project-record).

**Close-out evals: two done, two running.** `b10a` (272 checkpoints) and `b10c` (47) are
complete. `b10b` and `b10d` are still going, resumed 2026-08-03 at `EVAL_WORKERS=10` after the
worker-count measurement below — no training slots are involved either way.

| arm | checkpoints | best (100 episodes) | pooled | state |
|---|---|---|---|---|
| `b10a-disc995seed1` | 272 | 85.0% @2344000 | 67.2% | complete |
| `b10b-disc995seed2` | 624 | 90.0% @1501000 so far | 71.8% | ~58%, ~2h left |
| `b10c-disc995seed3` | 47 | 79.0% @3965000 | 63.0% | complete |
| `b10d-disc995seed4` | 660 | 93.0% @1695000 so far | 74.1% | ~53%, ~3h left |

`b10d`'s 93% @1695000 now beats the 95% @1815000 in the hall of fame on point estimate order
— except it doesn't, quite: the two intervals overlap almost entirely, and the 95% was itself
the max of ~300 noisy measurements. Both are the same policy family measured twice. See the
winner's-curse note in [`hyperparamTuning.md`](hyperparamTuning.md#why-not-abandon-weak-checkpoints-early)
before quoting either as a record.

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
ones. Full numbers in [`hyperparamTuning.md`](hyperparamTuning.md#screening-eval_screen_episodes).

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

**The two close-out evals still running cannot be resumed if they die.** They work only because
their processes loaded the 26-value code at startup; `EVAL_RESUME=1` would now build a 30-value
network and fail to restore. Whatever they have written is what there will be.

## Batch 11 — RUNNING, launched 2026-08-03 ~22:55

**Four seeds of one config on the 30-value environment.** Batch 10's config, unchanged, so the
only difference between the two batches is the two observations added 2026-08-03. **The first
seeded batch in this project's history** — `SNEK_SEED=1..4`, recorded in each `runs/<policy>.md`.

Verify with `pgrep -fl "python -u snek2.py"`. Four watchers are attached via
`watch_when_ready.sh`. `b10d`'s close-out eval was still finishing at launch, by request.

**Early numbers mean nothing yet and are recorded here only so they are not mistaken for a
result later.** At 10k steps batch 11 averages score 41.1 against batch 10's 25.7, and
`b11c` showed 30% perfect where no batch-10 arm had any. That difference is **t=0.90, p≈0.4** —
the within-batch spread is 7.0 to 71.9. This document's own rule is that nothing is judgeable
below ~250k steps, and two runs of one identical config have reached final avg_score 62.5 and
18.0 at 30k. Do not quote this paragraph as evidence of anything.

| arm | policy name | config |
|---|---|---|
| a | `b11a-obs30seed1` | `SNEK_SEED=1` |
| b | `b11b-obs30seed2` | `SNEK_SEED=2` |
| c | `b11c-obs30seed3` | `SNEK_SEED=3` |
| d | `b11d-obs30seed4` | `SNEK_SEED=4` |

Shared base, byte-identical to batch 10:
`SNEK_DISCOUNT=0.995 SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0`

**Names encode the vector length rather than the config**, which departs from the usual convention
of naming what varies (`disc995`, `nstep3`, `unifbuf500k`) — nothing about batch 11's config varies
from batch 10's, so there is nothing to name there. `obs30` identifies the *environment*, and that
is the label worth having: it decides checkpoint compatibility, and it makes this batch reusable as
**the benchmark for every later batch trained on the 30-value vector**. A batch 12 arm at
`LEARNING_RATE=1e-4` compares against `obs30` arms directly; one on a 31-value vector does not, and
its name will say so.

**Batch 10 is a clean control, verified rather than assumed.** The only training-relevant commit
between batch 10's launch and now is `b09c616`, the two observation blocks; the one other commit
touched a chart line width. So batch 11 minus batch 10 isolates exactly those two observations —
the comparison batch 10 could never provide for the 2026-08-02 changes.

**Why four seeds of one config and not a knob test.** The tempting alternative is 2 baseline plus 2
at `LEARNING_RATE=1e-4`, the highest-value untested knob. That is batch 9's documented mistake at
one remove: it buys two n=2 answers instead of one n=4 baseline that every later batch spends.

### What this batch can and cannot show

| metric | batch 10 (n=4) | sd | detectable vs a new n=4 |
|---|---|---|---|
| equal-effort pooled | 69.2% | 5.2 | **9.0 pp** at p<0.05 |
| best-30 perfect (graph) | 80.1% | 5.8 | 10.0 pp |
| best checkpoint | 86.8% | 6.1 | 10.6 pp |
| peak trailing score | 94.47 | 0.50 | 0.9 pp |

**A null result is the most likely outcome, and it will not mean the observations are useless.** If
they help by 3-5pp this design cannot see it — between-seed variance dominates and no amount of
eval precision touches it. Recorded here in advance so a null is not later written up as a finding.

That last row is a trap: peak trailing score sits at 93.8-94.96 against a saturated ceiling, so its
tight sd is compression rather than sensitivity. Useful only as a **regression tripwire**.

### Pre-registered comparison

Fixed before launch, because this document notes that earlier batch rankings compared arms at
horizons where they had not finished improving:

- **Primary:** equal-effort pooled from the close-out, plus best-30 perfect **at a common 4.12M
  horizon** (batch 10's shortest arm) — not at final step, whatever each arm reaches.
- **Secondary:** best checkpoint, with winner's-curse shrinkage applied. Batch 10's headline 93%
  shrinks to 87.2% once it is treated as the max of ~300 noisy measurements.
- **Tripwire:** peak trailing score. A drop below ~93 is a regression signal.
- **Decision rule:** keep the new observations unless clearly worse.

### Horizon: run until they stop improving

No fixed cap, by request. Judged on the trailing-window criteria in
[`hyperparamTuning.md`](hyperparamTuning.md#when-to-keep-a-run-going-and-when-to-stop-it) — last 20
evals vs the previous 20 for score, last 30 vs previous 30 for perfect rate — and **nothing gets
stopped without reporting first**.

Batch 10 managed 4.1-4.65M steps in 15.6 hours with four arms in parallel, so ~281k steps per arm
per hour; expect that to slow as skill rises and eval episodes lengthen. Batch 10 was stopped
*healthy* — `b10b` peaked at 4545k and was stopped at 4652k, `b10c` peaked at 4021k and stopped at
4122k — so its ceiling is unknown, and because the vector changed those arms can never be resumed
to find out. Not repeating that is the reason for an open horizon.

### Infrastructure added for this batch

Both landed before launch and are off by default, so nothing about batch 10 or earlier is affected.

**`SNEK_SEED`** — the first seeding this project has ever had. Nothing seeded anything before, so
`seed1`..`seed4` in batch 10's names were labels and no run was reproducible. The base seed and any
ablation now appear in `runs/<policy>.md`. It buys reduced variance and a roughly repeatable run,
**not** bit-identical replay: exact determinism would also need single-threaded TF and a fixed
arrival order for the parallel workers. It does not help the batch-10 comparison either, since
adding four inputs changes the initialisation RNG stream; it pays off from batch 12 on, where most
tests change only a hyperparameter.

The hazard it had to avoid: food placement uses the *global* `random` module and
`ParallelPyEnvironment` runs one constructor per worker process, so a single shared seed would have
all ten workers deal identical food — turning every 10-episode eval into one episode counted ten
times, with confidence intervals to match, and raising nothing. Each worker gets its own derived
stream, and `tests/test_seed_and_ablation.py` covers it.

**`SNEK_ZERO_OBS`** — zeroes named observation indices *without changing the vector length*, e.g.
`SNEK_ZERO_OBS=26-29`. This is the only way to ever answer "did those two observations help"
cleanly: both groups train on the same 30-value spec, their checkpoints stay mutually loadable, and
the only difference is the information in those indices. Batch 11 can then serve as the treatment
group for a later 4-arm ablated control — a real 4v4 with no environment confound. Deleting a block
instead would change the length and make it a two-environment comparison again, which is the
confound that already cost this project the ability to attribute batch 10's result.

### Later candidates

Deferred pending the above. Still relevant to whatever comes after it:

| change | why | gate |
|---|---|---|
| second discount value (`0.9975`, or an interior point like `0.996`) | batch 9 left 0.995 vs 0.9975 unsettled; whether that survives the third environment is unknown | after the user's pending changes |
| `SNEK_LEARNING_RATE=1e-4` | the highest-value untested knob | after the user's pending changes |
| eff exponent ~1.4 (`td_loss` alpha 0.7) at 0.995 | `b4c` and `b7f` tie on ceiling; sharpness may still add on top of the discount | after the user's pending changes |
| best config + `REPLAY_BUFFER_MAX_LENGTH=500000` | `b4b` beat `b4a` slightly, so diversity may stack | after the user's pending changes |
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
| epsilon ladder *shape* (not floor) | exploration schedule | low |
| `REPLAY_BUFFER_MAX_LENGTH=1000000` | experience diversity | low — the 500k result was ambiguous |

**`LEARNING_RATE=1e-4` — only after a stability fix.** 1e-5 is very conservative and
the in-code comment already suggests 1e-4. With a stable target it may train several
times faster; on its own with `TARGET_UPDATE_PERIOD=8` it would probably make
instability worse. The order matters.

**`TARGET_UPDATE_PERIOD=50` and `=500`.** Batch 1 hinted that longer periods learn
faster early even though they didn't reduce drawdown. Two more points establish
whether that is a trend or noise. Note `b1b-tgt200` was stopped at 104k, well short of
the ~250k horizon, so that hint is weak evidence.

**Epsilon ladder shape.** The floor was tested and the hypothesis falsified. What
remains untested is the *shape*: the ladder is driven by reward thresholds and steps
down once per eval, so it is coupled to `eval_interval` — a latent confound if that
interval is ever changed, and a reason a slower or step-count-based decay is worth
trying.

## Explicitly not planned

- **Reward changes** — they would break comparability of `avg_score` with every run
  recorded so far.
- **Reverting to `PyUniformReplayBuffer`** — cpprb is ~2.4x faster with no measured
  learning cost, so cheaper experiments come from keeping it.
- **An LR schedule** — no evidence of optimization instability; degradation is gradual
  in every arm, not spiky.
- **Making the epsilon last-rung threshold tunable** — the ladder is no longer a
  suspect, see [`findings.md`](findings.md).
- **`N_STEP_UPDATE=5`** — n=2 and n=3 both peak below baseline and then decline, so
  the trend already points the wrong way.
- **Resuming any arm from batch 9 or earlier** — every checkpoint on record from before
  batch 10 was trained on an observation vector this project has since changed (20 or 23
  values against the 26 batch 10 trained on), so none of them load; see
  [`../hallOfFame/README.md`](../hallOfFame/README.md#the-entries-below-predate-2026-08-02-and-do-not-run-on-master).
  Batch 10's own checkpoints *do* still load on `master` as of this close-out — see the
  note above about the user's pending changes for whether that keeps being true.

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
