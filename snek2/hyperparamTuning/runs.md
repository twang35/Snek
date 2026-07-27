# Run status, results, and queue

Companion to [`hyperparamTuning.md`](hyperparamTuning.md) (the protocol) and
[`charts.md`](charts.md) (the graphs). This is the file that changes constantly;
the protocol rarely does.

## What is currently running

Update this section whenever runs start or stop — a future session reads it to
know what is in flight and might have been terminated.

Status as of 21:35. **Batch 3 is running**; batches 1 and 2 are finished.

| policy | config change | steps | best perfect-30 | perfect, latest block | note |
|---|---|---|---|---|---|
| `b3a-epsfloor` | `MIN_EPSILON=0.001` | 313k | **11.0%** | 6.4% | **best batch-3 arm.** Floor active since 267k |
| `b3b-epsfloor2` | `MIN_EPSILON=0.001` | 310k | 8.3% | 3.6% | floor active since 147k — now inside the collapse window |
| `b3c-buf500k` | `REPLAY_BUFFER_MAX_LENGTH=500000` | 293k | 5.7% | 4.5% | **went to epsilon 0.0 at 282k** — accidentally the batch's best test |
| `b2a-base2` | baseline repeat #2 | 762k | 7.0% | 2.9% ↑ | recovering from a long trough; not decaying after all |

All four are **still running** — a progress check never stops an arm (see
`CLAUDE.md`). None has yet earned a verdict.

Logs are in `/Users/tony_wang/.claude/jobs/f3cb1855/tmp/b3{a,b,c}.log`, which is
job-scoped and will not survive; the durable record is `runs/<policy>_evals.json`,
so analyse from there.

Resume any of these by relaunching with the same policy name **and the same
`SNEK_*` overrides** — the overrides are *not* persisted in the checkpoint, so
relaunching without them silently changes the config mid-run and invalidates the
arm. For batch 3 that means `SNEK_MIN_EPSILON=0.001` for `b3a`/`b3b` and
`SNEK_REPLAY_BUFFER_MAX_LENGTH=500000` for `b3c`.

### Matched-step comparison, 250-300k block

Every arm has passed 250k, so these numbers sit on the right side of the judgeability
line. This is the first table in this document that can be taken at face value.

| policy | score mean | score min | perfect mean | best perfect-30 |
|---|---|---|---|---|
| `b3a-epsfloor` | **74.4** | 58.2 | **8.6** | **11.0** |
| `b3b-epsfloor2` | 71.0 | 52.2 | 7.0 | 8.3 |
| `b3c-buf500k` | 70.6 | **60.8** | 4.5 | 5.7 |
| `b2a-base2` | 68.7 | 41.2 | 3.8 | 7.0 |
| `b1a-base` | 45.8 | 10.7 | 8.2 | 16.7 (earlier) |

`b1a-base` was mid-collapse in this block, which is why its numbers are poor here
despite holding the document's best perfect-30 window (16.7% at 200-250k).

**This weakens the "epsilon 0.0 raises the ceiling" reading from the last update.**
That claim rested on every 0.001 arm sitting at 3-7% perfect while `b1a-base` reached
14%. `b3a-epsfloor` has since hit an **11.0% perfect-30 window while floored at
0.001**, which closes most of that gap. Both floored arms are also above every
baseline at matched steps. So the honest current statement is that floored arms are
doing *at least as well* as the baselines and `b1a-base` remains a single high
outlier — not that zero exploration buys a higher ceiling.

That is 4 trainers, i.e. the budget is full — do not launch more until one stops.
Logs are in `/Users/tony_wang/.claude/jobs/f3cb1855/tmp/b3{a,b,c}.log`, which is
job-scoped and will not survive; the durable record is `runs/<policy>_evals.json`,
so analyse from there.

Resume any of these by relaunching with the same policy name **and the same
`SNEK_*` overrides** — the overrides are *not* persisted in the checkpoint, so
relaunching without them silently changes the config mid-run and invalidates the
arm. For batch 3 that means `SNEK_MIN_EPSILON=0.001` for `b3a`/`b3b` and
`SNEK_REPLAY_BUFFER_MAX_LENGTH=500000` for `b3c`.

### Stopped, with verdicts

| policy | config change | stopped at | verdict |
|---|---|---|---|
| `b1a-base` | control: committed defaults | 497k, 19:00 | **The key run of the investigation.** Collapsed at ~265k after peaking at a 14% perfect rate, recovered in score to ~65 but only 2.3% perfect. Question it was kept alive for is answered: recovery is real but score-only |
| `b1c-nstep3` | `N_STEP_UPDATE=3` | 1.12M, 19:00 | **Dead end.** Peaked 76 at 255k, then declined for 850k steps to a flat ~28. Zero perfect games since ~300k |
| `b2b-nstep2` | `N_STEP_UPDATE=2` | 568k, 19:00 | **Dead end**, same shape as n=3: peaked 74.6 at 140k, down to ~35, trailing-30 perfect 0.0%. Two arms ordered by n giving the same result is a trend, not noise |
| `b1b-tgt200` | `TARGET_UPDATE_PERIOD=200` | 104k, 15:10 | Score rising only slowly (+1.8), perfect rate flat (1.0 -> 1.0), hypothesis already answered. Stopped early to free a slot, so it never reached the ~250k horizon where anything is judgeable — **its verdict is weaker than the others here**. Resume with `SNEK_TARGET_UPDATE_PERIOD=200` |
| `train` | human-started, committed defaults | ~15:15 | Stopped by the human, freeing a slot. Do not restart it; it is theirs |

---

## Catastrophic forgetting confirmed — and it is the whole ballgame

**This supersedes the earlier note that no arm showed forgetting.** That note was
written at 83k steps, which turns out to be far too early to see it.

`b1a-base`, the plain committed config, by 50k-step block:

| steps | score mean | score max | score min | perfect mean | perfect max |
|---|---|---|---|---|---|
| 0-50k | 41.8 | 76.1 | 0.0 | 0.4 | 10 |
| 50-100k | 68.9 | 81.6 | 56.6 | 3.0 | 30 |
| 100-150k | 75.0 | **87.5** | 60.6 | 7.8 | 30 |
| 150-200k | 76.2 | 85.0 | 62.0 | 10.2 | 30 |
| 200-250k | 71.9 | 83.1 | 55.9 | **14.0** | **40** |
| 250-300k | 45.8 | 76.4 | 10.7 | 8.2 | 40 |
| 300-350k | 37.1 | 61.6 | 20.4 | 5.0 | 20 |
| 350-400k | 63.6 | 80.6 | 31.5 | 6.2 | 20 |
| 400-450k | 65.1 | 78.7 | 54.8 | **2.9** | 20 |

It held a broad plateau from 30k to ~260k, peaked at **14% mean perfect rate with
40% spikes** in the 200-250k block, and then **collapsed** — score from a 87.5 peak
down to 23 by 343k, a 64-point drawdown, with the perfect rate falling 14 -> 8.2 ->
4.5. See the chart in [`charts.md`](charts.md); the break at ~265k is stark.

Three things follow, and they reshape the investigation:

1. **The collapse happens right after the best perfect rate.** `b1a-base` was on
   track for a genuinely good perfect rate and then threw it away. Preventing the
   collapse is therefore not a side quest — it is plausibly the single
   highest-value intervention available for the stated objective. Everything in
   the queue should be re-prioritized around "does it still hold at 300k".
2. **It is stochastic, not deterministic for the config.** `b2a-base2` is the same
   config and at 275k has *not* collapsed: peak 83.0, latest 74.0, a drop of only
   9.0, with its perfect rate still drifting up. Two runs of one config diverge
   completely after 250k. So the noise this protocol worries about is not just an
   early-phase effect — it extends to *whether a run collapses at all*, which is a
   much bigger deal.
3. **Nothing can be judged below ~250k steps.** Every comparison made so far in
   this document is on the wrong side of that line. Arms need to reach ~300k before
   their verdict means anything, which is expensive and needs planning for.

## When each arm's epsilon treatment actually starts

`MIN_EPSILON` only changes behaviour at the ladder's last rung, which needs
`avg_reward > 100`. Crossing that is uncommon and *late*, so each arm has a
divergence step before which it is indistinguishable from an unfloored run:

| policy | first `avg_reward > 100` | epsilon now | treatment active since |
|---|---|---|---|
| `b1a-base` | 92k (18 evals over) | **0.0** | n/a — the thing being tested against |
| `b3b-epsfloor2` | **147k** | 0.001 (floored) | **147k** |
| `b3a-epsfloor` | **267k** | 0.001 (floored) | **267k** |
| `b3c-buf500k` | **282k** | **0.0** | n/a — floor is 0.0, so it went greedy |
| `b2a-base2` | never (peaked 99.1) | 0.001 | never |

**One crossing is all it takes**, because the ladder is a one-way ratchet: a single
eval over 100 permanently pins epsilon, and score dropping afterwards never raises it
back. So `b1a-base` needed only its 92k crossing to spend the rest of its life fully
greedy.

*Earlier this section said `b3a-epsfloor`'s floor was inert and the arm was an
accidental baseline repeat. That was true at 238k and is no longer: it crossed at
267k, so both floored arms are now genuine treatment arms* — `b3b` with 163k steps of
exposure, `b3a` with 46k.

### `b3c-buf500k` became the sharpest test in the batch, by accident

It crossed at 282k with `MIN_EPSILON` at its 0.0 default, so **it is now running fully
greedy with a 500k buffer** — which is exactly the head-to-head between the two
hypotheses that nothing was designed to test:

- If **epsilon 0.0** is what drives the collapse, `b3c` should break ~150-175k steps
  after 282k, i.e. **around 430-460k**, big buffer or not.
- If **buffer diversity** is what matters, the 5x buffer should protect it and it
  should keep the smooth curve it has had all along.

Either way it is informative, and it costs nothing extra to wait. This is the single
most valuable thing currently running.

### The threshold, not the floor, is the knob worth adding

Every arm's divergence step is set by when it happens to cross `avg_reward > 100`,
which is luck. Making that threshold tunable would let a treatment arm be *forced* to
the last rung early, giving a long exposure window on purpose instead of 46k steps by
accident. That is a better next code change than another `MIN_EPSILON` value.

## The collapse is recoverable in score but not in skill

`b1a-base` was left running specifically to see whether the collapse was permanent.
It isn't — but the recovery is the interesting part, because it is **only half a
recovery**:

| steps | score mean | perfect mean |
|---|---|---|
| 200-250k (pre-collapse peak) | 71.9 | **14.0** |
| 300-350k (trough) | 37.1 | 5.0 |
| 350-400k | 63.6 | 6.2 |
| 400-450k | **65.1** | **2.9** |

Average score climbed back from 37 to ~65 with no intervention, while the
perfect-game rate kept *falling* and is now at a fifth of its pre-collapse level.
So the run relearned how to play competently and did **not** relearn how to finish.

Two things follow, and the second is a change to the protocol:

- **Score and perfect rate decouple after a collapse.** The protocol lists last-5
  score as the workhorse leading indicator, which is right early — but here score
  says "recovered" while the objective says "much worse than before". Late in a
  run, score alone will mislead. Judge post-collapse arms on the perfect rate
  directly.
- **A collapse costs the good result permanently, even though the curve comes
  back.** Riding out a collapse and hoping is not a strategy; the 14% perfect rate
  never returned across 150k further steps. That strengthens the case for
  *preventing* collapse rather than tolerating it.

## `b2a-base2` oscillates on a very long timescale — it does not decay

At 680k this arm looked like it was slowly degrading, and this section said so. **By
763k it had partly recovered, so that reading was premature:**

| steps | score mean | perfect mean |
|---|---|---|
| 150-200k | 69.6 | **5.2** |
| 450-500k | 68.7 | 3.6 |
| 550-600k | 62.8 | **1.0** ← trough |
| 650-700k | 64.8 | 1.4 |
| 750-800k | 66.5 | **2.9** ← recovering |

So the shape is a **slow oscillation with a shallow downward drift**, not a decay and
not a collapse: no break anywhere, a broad trough around 575k, and both metrics
heading back up since. Peak-to-trough on the perfect rate is still large (5.2 → 1.0),
so the swing is real — it just isn't monotonic.

Two things worth keeping from this:

- **The oscillation period is longer than most runs are.** A trough spanning ~100k
  steps means a snapshot at 600k and one at 780k give opposite verdicts on the same
  arm. Trailing-window trends over 20-30 evals are far too short to see this; only the
  50k-block table shows it.
- **This is a third distinct pattern**, alongside `b1a-base`'s sharp collapse and the
  n-step arms' monotonic decline. Max drawdown collapses all three into one number and
  cannot tell them apart.

The lesson is the one this document keeps relearning, now at a fourth timescale: **do
not call a trend from the most recent window.** Two premature calls in a row here came
from doing exactly that.

## Leading hypothesis for the collapse: epsilon reaches exactly 0.0

This is the most actionable thing found so far. The epsilon ladder in
`maybe_update_epsilon()` is a one-way ratchet ending in `epsilon.assign(0.0)` once
`avg_reward > 100`. Tracing when each arm stepped down:

| policy | reached 0.001 | reached **0.0** | collapsed? |
|---|---|---|---|
| `b1a-base` | 37k | **92k** | **yes, at ~265k** |
| `b2a-base2` | 25k | never (still 0.001) | no, through 348k |
| `b1c-nstep3` | 158k | never | declining, no collapse event |
| `b2b-nstep2` | 102k | never | declining, no collapse event |

**The only arm that ever reached epsilon 0.0 is the only arm that collapsed.**

The mechanism is plausible and specific: at epsilon 0.0 the collect policy is fully
greedy, so the replay buffer's contents become entirely determined by the current
policy. With a 100k-transition buffer and ~800-step episodes at high skill, all
exploratory data is flushed out within ~100-200k steps of the switch — and the
collapse landed at 265k, ~173k steps after the switch, which is consistent with
that ordering. From then on it is a closed feedback loop: the policy trains only on
what it already does, so a drift has nothing to correct it. The later
partial recovery also fits — a closed loop can wander back as easily as it wandered
off, which is exactly the oscillating-instability signature rather than a
one-way failure.

Note this was anticipated by the batch-2 queue's "longer epsilon exploration" item
("if forgetting correlates with epsilon getting small, a floor is worth testing")
before there was any evidence for it. There is now.

**Caveats, because this is n=1 and confounded.** Reaching 0.0 requires
`avg_reward > 100`, i.e. only *strong* runs get there — so "reached 0.0" and "was
good enough to collapse from a height" are entangled, and this correlation cannot
separate them. `b2a-base2` is stable partly because it never got good enough to
zero out its exploration; it is stuck at ~4% perfect where `b1a` reached 14%. The
uncomfortable reading is that the ladder's last rung is both what enables the best
performance *and* what destroys it.

The test is cheap and settles it: floor epsilon at 0.001 and never let it hit zero.
That needs a knob (`SNEK_MIN_EPSILON`), which does not exist yet — a small change to
`maybe_update_epsilon()`. This is now the top of the queue.

## n-step returns: closed, negative

Both n-step arms have run long enough to judge, and they agree:

| policy | steps | peak score (at) | latest | trailing-30 perfect | 1st perfect |
|---|---|---|---|---|---|
| `b1c-nstep3` | 858k | 76.0 (255k) | 31.0 | **0.0%** | 206k |
| `b2b-nstep2` | 414k | 74.6 (140k) | 48.9 | **0.0%** | 121k |

Both peak *below* either baseline, both then decline for hundreds of thousands of
steps, and both are at zero perfect games in the trailing window. n=3 has produced
no perfect game at all since 250k and has been flat at ~30 for 200k steps; n=2 is
the same shape one step milder, which is a clean monotonic trend in the wrong
direction rather than noise.

This **overturns the batch-1 read** that n=3 had "the best trajectory of the
batch". That was true through 200k and false afterwards — the momentum that made it
look promising simply ran out at 255k. The lesson from that section ("do not judge
an arm at matched steps alone") survives; the verdict it produced does not. Do not
plan an n=5 arm.

## Prior findings carried in from earlier sessions

These are already established; don't re-litigate them without new evidence.

- **Prioritized replay at alpha=0.6 measured *worse* than uniform** over 3 seeds
  at 30k steps: last-5 avg 46.7 (sd 10.6) vs **60.1 (sd 4.0)** for `alpha=0`.
  Uniform was also far more consistent. At alpha=0.8 with Huber-loss priorities it
  was worse still. Plausible reason: the reward is already dense and shaped
  (`FOOD_DISTANCE_REWARD` on every step), and PER's advantage is largest with
  sparse rewards, so over-sampling high-error transitions mostly adds variance.
  **Unresolved:** that was only 30k steps. PER may pay off closer to 1M. Worth one
  long-horizon retest.
- `alpha=0.6` is nevertheless the committed default, chosen deliberately. Treat it
  as the baseline to beat, not as known-good.
- **Importance-sampling weights must stay mean-normalized.** cpprb normalizes by
  the largest weight in the whole buffer, so raw batch weights average 0.087 at
  beta=0.4 and 0.0027 at beta=1.0 — a silent 11x-370x cut to the learning rate
  that worsens as beta anneals. `normalize_is_weights()` fixes this; don't remove
  it.
- **Priorities come from `|td_error|`, not `td_loss`.** Huber is quadratic below
  |e|=1 so it shrinks small errors, widening its spread; feeding it in gave an
  effective exponent near |e|^1.6 instead of |e|^0.6 and measurably hurt learning.
- **`legacy.Adam` is not faster here** despite TF's M1/M2 warning: measured 0.809
  ms/step vs 0.721 ms for the modern optimizer. Ignore that warning.
- **Throughput is ~230-240 steps/s** for a single run on an idle machine. Expect
  substantially less with 4 runs sharing 14 cores; that affects wall-clock only,
  not learning per step.

---

## Completed runs

### Batch 1 — interim, all still running

Compared at **matched step 83000**, because wall-clock progress is not comparable
between arms (see the eval-cost confound below). Objective metrics first.

| policy | key change | trailing-20 perfect % | 1st perfect | last-5 score | curve mean | max drawdown | peak |
|---|---|---|---|---|---|---|---|
| `b1a-base` | none (control) | **1.5** | 44000 | **66.0** | 52.4 | 19.2 | 78.6 |
| `b1b-tgt200` | `TARGET_UPDATE_PERIOD=200` | 0.5 | **33000** | 59.2 | **53.4** | 27.4 | 76.9 |
| `b1c-nstep3` | `N_STEP_UPDATE=3` | 0.0 | never | 26.3 | 16.5 | 18.6 | 38.1 |

Perfect-game rate is still tiny and very coarse at this horizon — a trailing-20
average of 1.5% is three perfect games across 200 episodes. It is not yet a
reliable way to separate these arms, which is exactly why score is the workhorse
metric this early. Do not pick a winner on perfect % at 83k steps.

Charts for every arm live in [`charts.md`](charts.md), kept separate so this file
stays readable.

**Everything below is n=1 and this domain is very noisy. Treat as hypotheses to
test, not conclusions.** The single most useful next action is repeating the
baseline, because nothing here can be interpreted without knowing its spread.

#### The big surprise: perfect games arrive ~20x earlier than expected

The premise of this investigation was that perfect games need ~1M iterations to
reach ~50%. `b1a-base` produced its **first 10% perfect-game eval at ~44k steps**,
and `b1b-tgt200` at **~33k**. Both plateau in avg_score around 65-75 out of 95 by
40k.

That is a large enough discrepancy to be worth chasing before tuning anything
else, since it changes what "good" means. Candidate explanations, untested:

1. The code changed materially since the 1M-iteration experience was formed — the
   replay buffer is now cpprb prioritized rather than `PyUniformReplayBuffer`, and
   the collect policy runs under `tf.function`. Neither *should* change learning
   per step, but PER changes the sample distribution, so it might.
2. Lucky seed. Entirely plausible at n=1 given the documented 62.5-vs-18.0 spread.
3. The human's long-running `train` policy carries 1.3M steps of history under the
   older code, so its trajectory isn't directly comparable to a fresh run.

Worth asking the human whether ~10% perfect at 44k is genuinely faster than they
have seen, since they have the historical context that these logs don't.

#### `TARGET_UPDATE_PERIOD=200`: hypothesis not supported, but interesting anyway

The prediction was smoother curves and smaller drawdowns. It got the **opposite**
on the drawdown metric (27.4 vs 19.1) and a lower last-5. What it did do is learn
**much faster early** — roughly score 55 by 15k steps where the baseline needed
~25k — and reach its first perfect game sooner.

So the frequent-target-update theory of catastrophic forgetting looks wrong, at
least at this horizon, while "less frequent target updates learn faster early"
looks worth pursuing. Try 50 and 500 to see whether there's a trend or this is
noise.

#### `N_STEP_UPDATE=3`: slower per step, but the best trajectory of the batch

At matched steps it looks like the clear loser, and an early read of this batch
called it exactly that. Judged on trajectory instead it is the most interesting arm
in the batch. Score by 40k-step block:

| steps | mean score | max |
|---|---|---|
| 0-40k | 5.1 | 16.2 |
| 40-80k | 26.9 | 38.1 |
| 80-120k | 26.8 | 35.4 |
| 120-160k | 37.4 | 66.3 |
| 160-200k | 49.7 | 70.1 |
| 200-240k | 57.3 | 67.5 |

Near-monotonic, and at 201k it was still gaining +10.3 per 20 evals while
`b1a-base` had gone flat at ~69. It has produced no perfect game yet, which is the
one real mark against it.

So n-step is not "worse", it is **slower but more consistently improving** — which
is arguably closer to what this investigation wants than a config that sprints to a
plateau. Whether it overtakes the baseline is an open question; that is what the
4-hour cap is for. If it plateaus below ~70 it is merely slow; if it keeps climbing
past that it is the most promising lead so far.

Lesson recorded: **do not judge an arm at matched steps alone.** Matched-step
comparison is right for fairness but blind to momentum, and momentum was the
signal that mattered here.

#### Neither arm showed catastrophic forgetting *by 83k* — since overturned

At 83k both arms oscillated roughly +/-10 around a high plateau rather than
collapsing, and this section originally concluded that forgetting wasn't happening
at all. **That was wrong, and only because the horizon was too short:**
`b1a-base` collapsed hard at ~265k. See the section above.

The lesson worth keeping is about horizon, not about forgetting: a conclusion drawn
at 83k about a phenomenon that appears at 265k is worthless, and there was nothing
in the 83k data to indicate that.

This also exposes a **flaw in the max-drawdown metric**: it cannot tell noisy
oscillation around a good plateau apart from a genuine collapse. `b1b` scores
worse on it purely for oscillating at a high level. A better forgetting metric
would be something like the largest sustained drop — a drawdown that persists over
several consecutive evals rather than one bad eval. Worth implementing before
leaning on drawdown again.

#### Confound found: eval cost scales with policy quality

A better policy eats more food, so its episodes are longer, so its 10-episode eval
takes longer in wall-clock. `b1c-nstep3` reached 161k steps while the other two
were at ~68k in the same elapsed time, purely because it was worse and its evals
were cheap.

Consequences: never compare arms by wall-clock or by "where they got to"; always
compare at matched steps. And a batch of arms will drift apart in step count, so
plan to stop them by step count rather than by time.

---

## Planned queue, and what each is expected to show

Ordered by expected value. Revise freely as results land — this is a plan, not a
commitment.

### Batch 1 — the original plan, for the record

| policy | change | expected | actual |
|---|---|---|---|
| `b1a-base` | none (control) | a *reference*, not a winner | became the key run: collapsed at 265k |
| `b1b-tgt200` | `TARGET_UPDATE_PERIOD=200` | smoother curve, smaller drawdown | faster early, *larger* drawdown |
| `b1c-nstep3` | `N_STEP_UPDATE=3` | earlier learning onset, better late game | slowest to rise, then declined for 850k steps |

The reasoning at the time, kept because two of the three predictions were wrong and
that is worth remembering:

- **`b1a-base`** — a control run under the same machine load as its batch mates,
  since throughput and contention vary between batches and make cross-batch
  comparison weaker than within-batch.
- **`b1b-tgt200`** — the highest-prior change of the batch. 8 gradient steps between
  target-network syncs is extremely frequent where standard DQN uses hundreds to
  thousands, and a target that chases the online network is a classic cause of
  oscillation and forgetting, which was exactly the reported symptom.
- **`b1c-nstep3`** — the food reward is immediate but the perfect-game bonus is
  terminal and extremely sparse, so credit has to crawl back one step per gradient
  update; n-step returns propagate it ~3x faster.

### Batch 3 — reprioritized around the collapse (do these next)

The collapse at ~265k is now the main obstacle to a high perfect rate, so the
queue leads with things that plausibly prevent it. **Every one of these needs to
run past 300k to mean anything**, so expect ~3-4 hours per arm and plan slots
accordingly.

| item | change | status |
|---|---|---|
| A | floor epsilon at 0.001 | **running** as `b3a-epsfloor`, `b3b-epsfloor2` |
| B | `REPLAY_BUFFER_MAX_LENGTH=500000` | **running** as `b3c-buf500k` |
| C | `PRIORITY_EXPONENT=0.0` at long horizon | queued |
| D | `GRADIENT_CLIPPING=10` | queued |
| E | third baseline repeat | queued |
| F | LR schedule | queued, needs a new knob |

#### A. Floor epsilon at 0.001 — never 0.0 ▶ LAUNCHED

Top priority, on the evidence in "Leading hypothesis for the collapse" above: the
one arm that reached epsilon 0.0 is the one arm that collapsed, and at 0.0 the
buffer becomes a closed policy-data feedback loop. The `SNEK_MIN_EPSILON` knob now
exists (defaults to 0.0, so the ladder is unchanged for every other run). Expect:
reaches the same 14%-class perfect rate as `b1a-base` did but holds it past 300k.
Run **twice**, because a single non-collapsing run proves nothing — `b2a-base2`
didn't collapse either.

**Read the arm against `b1a-base`, not against `b2a-base2`.** `b2a-base2` sat at
0.001 for its whole life because it never crossed `avg_reward > 100`, so it is
already a de facto epsilon-floored run — and it is *stable but stuck* at ~4-7%
perfect. That makes it a weak comparator for this hypothesis and raises the
uncomfortable alternative: the last rung may be what enables `b1a`'s 14% *and* what
destroys it, in which case flooring buys stability at the cost of the ceiling and
`b3a`/`b3b` will look like `b2a-base2`. If they plateau at ~5% perfect without
collapsing, that is the answer, and the next move is a *lower* floor (1e-4) rather
than a higher one.

#### B. `REPLAY_BUFFER_MAX_LENGTH=500000` ▶ LAUNCHED

The strongest hypothesis that needs no code change, and it addresses the same
mechanism from the other side. The buffer holds 100k *transitions*, not episodes,
and episode length grows with skill: at score 5 an episode is ~50 steps so the
buffer spans ~2000 episodes, but at score 80 it is ~800+ steps so the buffer spans
only ~125. Experience diversity therefore *shrinks as the policy improves*, which is
exactly the setup for late-stage overfitting and collapse — and it fits the timing,
since the collapse arrives well after the policy gets good. Expect: later or no
collapse.

A and B are complementary, not redundant: **A keeps exploratory data being
*generated*, B keeps it from being *evicted*.**

#### C. `PRIORITY_EXPONENT=0.0` at long horizon

Once the policy is strong, most transitions have small TD error, so prioritization
increasingly samples rare outliers — plausibly destabilizing precisely late in
training. This also finally settles the 30k-step finding that uniform beat PER, at
the horizon that actually matters. Pairs naturally with B.

#### D. `GRADIENT_CLIPPING=10`

Cheap insurance if the collapse is driven by a few exploding updates. Weaker prior
than the above but nearly free.

#### E. A third baseline repeat

Two runs of one config, one collapsing and one not, is a sample size of two on the
most important question in this document. A third would say whether collapse is the
common case or the exception. Lower priority now that A gives the baseline arms a
specific thing to be compared against, but still the cheapest way to firm up the
denominator.

#### F. Lower learning rate late, or an LR schedule

If the collapse is an optimization instability rather than a data-diversity problem,
this addresses it directly. No LR-schedule knob exists yet — would need adding.

### Batch 2 and beyond — the standing backlog

Batch 3 leads because it targets the collapse. Everything still untested lives here,
ordered by expected value. Rationale for the entries that need it is below the table.

| change | targets | prior | status |
|---|---|---|---|
| `DISCOUNT=0.995` / `0.999` | perfect-game reward being reachable at all | **high** | queued |
| `LEARNING_RATE=1e-4` | training speed | high, but ordered after a stability fix | queued |
| `TARGET_UPDATE_PERIOD=50` / `500` | early learning speed | medium — 2 points to test a hinted trend | queued |
| `TARGET_UPDATE_TAU=0.005`, period 1 | smoothness (soft target updates) | medium | queued |
| `FC_LAYERS=128,128` | capacity | low | queued |
| epsilon ladder *shape* (not floor) | exploration schedule | low | partly promoted to 3A |
| baseline repeat 2-3x | knowing the spread at all | — | **done**: `b1a`, `b2a` |
| `N_STEP_UPDATE=2` | credit propagation | — | **closed, negative** |
| `PRIORITY_EXPONENT=0.0` | late-stage sampling noise | — | **promoted to 3C** |
| `GRADIENT_CLIPPING=10` | loss spikes | — | **promoted to 3D** |

#### `DISCOUNT=0.995` or `0.999` — the most under-rated item here

At 0.99 the effective horizon is ~100 steps, but a perfect game is several hundred
steps long, so the terminal bonus is discounted into near-irrelevance. Raising it
should make the perfect-game reward actually reachable by the value function —
plausibly the single most relevant change for the *stated* end goal. It is also a
known source of instability, so it pairs naturally with a stability fix rather than
going first.

#### `LEARNING_RATE=1e-4` — only after a stability fix

1e-5 is very conservative and the in-code comment already suggests 1e-4. With a
stable target it may train several times faster; on its own with
`TARGET_UPDATE_PERIOD=8` it would probably make instability worse. The order
matters.

#### `TARGET_UPDATE_PERIOD=50` and `=500`

Batch 1 hinted that longer periods learn faster early even though they didn't reduce
drawdown. Two more points establish whether that is a trend or noise. Note `b1b-tgt200`
was stopped at 104k, well short of the ~250k horizon, so that hint is weak evidence.

#### Epsilon ladder shape

The floor is now batch 3 item A. What remains untested is the *shape*: the ladder is
driven by reward thresholds and steps down once per eval, so it is coupled to
`eval_interval` — a latent confound if that interval is ever changed, and a reason a
slower or step-count-based decay is worth trying after A lands.

### Explicitly not planned

- Reward changes — they'd break comparability of `avg_score` with every recorded
  run.
- Reverting to `PyUniformReplayBuffer` — cpprb is ~2.4x faster with no measured
  learning cost, so cheaper experiments come from keeping it.
