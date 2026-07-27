# Run status, results, and queue

Companion to [`hyperparamTuning.md`](hyperparamTuning.md) (the protocol) and
[`charts.md`](charts.md) (the graphs). This is the file that changes constantly;
the protocol rarely does.

## What is currently running

Update this section whenever runs start or stop — a future session reads it to
know what is in flight and might have been terminated.

Status as of 09:20. **Nothing is running** — all arms stopped at the human's request
after batch 4 produced a clear winner.

| policy | config change | steps | best 30-eval perfect | best single eval | cumulative | verdict |
|---|---|---|---|---|---|---|
| `b4c-schlongper` | alpha 0.8, `td_loss`, no IS | 1.06M | **34.0%** | **80%** | **11.06%** | **winner — best checkpoint measures 51% over 100 episodes** |
| `b4b-unifbuf500k` | alpha 0 + 500k buffer | 1.23M | 9.3% | 40% | 4.03% | steady but low; no collapse |
| `b4a-uniform` | alpha 0 | 1.25M | 8.7% | 40% | 3.50% | peaked ~575k, drifted down |
| `b3c-buf500k` | 500k buffer, alpha 0.6 | 4.81M | 5.7% | 30% | 0.27% | **died at ~750k**, score 0.0 for 4M steps |

For reference, the best any earlier arm managed was `b1a-base` at a 16.7% window and a
40% single eval.

All four are resumable by relaunching with the same policy name **and the same `SNEK_*`
overrides** — they are not persisted in the checkpoint, so relaunching without them
silently changes the config mid-run:

| policy | overrides needed to resume |
|---|---|
| `b4c-schlongper` | `SNEK_PRIORITY_EXPONENT=0.8 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0` |
| `b4b-unifbuf500k` | `SNEK_PRIORITY_EXPONENT=0.0 SNEK_REPLAY_BUFFER_MAX_LENGTH=500000` |
| `b4a-uniform` | `SNEK_PRIORITY_EXPONENT=0.0` |
| `b3c-buf500k` | `SNEK_REPLAY_BUFFER_MAX_LENGTH=500000` (dead; not worth resuming) |

Resume any of these by relaunching with the same policy name **and the same `SNEK_*`
overrides** — they are not persisted in the checkpoint, so relaunching without them
silently changes the config mid-run and invalidates the arm. For batch 4 that is
`SNEK_PRIORITY_EXPONENT=0.0` (`b4a`), the same plus
`SNEK_REPLAY_BUFFER_MAX_LENGTH=500000` (`b4b`), and
`SNEK_PRIORITY_EXPONENT=0.8 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0` (`b4c`).

## Measured: `b4c-schlongper`'s best checkpoint is a genuine **51%** perfect-game policy

The graph plots one 10-episode eval per point, which moves in 10-point jumps and cannot
distinguish a good policy from a lucky one. So the four best `b4c-schlongper` checkpoints
were reloaded and evaluated over **100 greedy episodes each** with
[`eval_checkpoints.py`](../eval_checkpoints.py). Full results in
`runs/b4c-schlongper_checkpoint_evals.json`.

| checkpoint | perfect % over 100 eps | 95% CI | mean score | median score |
|---|---|---|---|---|
| **869000** | **51.0%** | **41.3-60.6%** | 83.4 | **95.0** |
| 942000 | 29.0% | 21.0-38.5% | 74.3 | 84.0 |
| 775000 | 25.0% | 17.5-34.3% | 75.4 | 76.5 |
| 162000 | 22.0% | 15.0-31.1% | 51.9 | 67.5 |

**Checkpoint 869000 wins more than half the games it plays.** Its median score is 95 of a
possible 95 — over half of its episodes are perfect wins, and the mean of 83.4 is dragged
down by a minority of early deaths (min 13). This is the first hard number in the
investigation that is not a 10-episode estimate.

It also **vindicates the original premise.** The brief said the config should reach ~50%
perfect games around 1M iterations, and `runs.md` previously recorded that the committed
config reached ~1% instead. With `theSchlong`'s PER restored, 51% arrives at 869k steps.
The premise was right; the three PER "corrections" were the regression.

### Selecting checkpoints by peak graph rate overestimates — badly

These four were chosen by highest 5-eval smoothed perfect rate. Comparing what the graph
suggested against what 100 episodes actually measured:

| checkpoint | single eval | smoothed(5) | **true (100 eps)** |
|---|---|---|---|
| 162000 | 30% | 46% | **22.0%** |
| 775000 | 10% | 38% | **25.0%** |
| 869000 | 70% | 44% | **51.0%** |
| 942000 | 10% | 38% | **29.0%** |

The smoothed estimate overshot on **three of four**, by up to 24 points. That is not a bug
in smoothing, it is the **winner's curse**: these checkpoints were picked for having the
highest value of a noisy statistic, so their true rates regress downward. Any "best
checkpoint" chosen off the graph should be expected to measure worse when evaluated
properly.

Two practical consequences:

- **Never quote a graph peak as a policy's perfect rate.** Quote a 100-episode eval. The
  graph is for trajectory; it is not a measurement of a policy.
- **Single evals can also *under*state badly** — 775000 and 942000 both read 10% on the
  graph and measured 25% and 29%. The noise runs both ways, so a low single eval is not
  evidence a checkpoint is bad either.

## `b4c-schlongper`: restoring `theSchlong`'s PER roughly doubles the perfect rate

The three "corrections" made when the buffer was ported to cpprb were, together, the
regression. Reverting all three — alpha 0.8, Huber `td_loss` priorities, no importance
sampling — produces the best arm by a wide margin on every measure:

| measure | `b4c-schlongper` | best of everything else | `theSchlong` |
|---|---|---|---|
| best single eval | **80%** | 40% (`b1a-base`, `b4a`, `b4b`) | 76% (one lucky checkpoint) |
| best 30-eval window | **34.0%** (851-880k) | 16.7% (`b1a-base`) | — |
| cumulative over run | **11.06%** | 5.89% (`b1a-base`) | — |
| peak avg score | **92.0** of 95 | 87.5 (`b1a-base`) | — |
| evals at >=50% perfect | **41** | 0 | — |

**It has beaten the `theSchlong` number it was built to reproduce**, and unlike a lucky
checkpoint it held 50-60% repeatedly across 700k-1000k: 41 separate evals at >=50%,
where no other arm in this investigation has ever produced one.

The block trajectory shows both why it wins and why it is not yet a solved problem:

| steps | score mean | perfect mean |
|---|---|---|
| 100-150k | 74.2 | 12.8 |
| 150-300k | 40.3 → 19.3 | **severe collapse**, score to ~19 |
| 350-400k | 70.4 | 11.0 |
| 700-750k | 75.5 | 13.2 |
| 800-850k | 80.0 | 26.2 |
| 850-900k | **79.0** | **32.2** |
| 950-1000k | 74.2 | 23.2 |
| 1000-1050k | 65.7 | 11.2 |

So it is **higher-performing and higher-variance** — it survived a near-total collapse
around 250k, recovered, and then climbed for 600k steps to a level nothing else has
approached. That matches the human's description of `theSchlong` as "very lucky and not
consistent at all": this config has a much higher ceiling and a much wider spread.

**This needs repeating 2-3x before it is believed.** It is n=1, this domain has produced
62.5-vs-18.0 from one config, and this arm's own history contains a collapse deep enough
to have ended the run if it had been judged at 300k.

### Consistency is now the open problem, not ceiling

The goal is a perfect rate that rises and keeps rising. `b4c` gets much higher but is
not yet consistent: a collapse at 250k, a peak at 875k, and a dip to 0% at ~1046k that
it is currently recovering from (71.4 score / 20% at 1058k). The productive next question
is which of its three changes drives the gain and which drives the variance — they were
reverted together, so that is still unseparated.

### The systemic finding batch 4 is chasing

**Every arm observed so far peaks between 236k and 312k and then degrades.** Five
configs, three epsilon regimes, two buffer sizes, same shape every time:

| policy | peak score (at) | best perfect-30 | where it ended |
|---|---|---|---|
| `b1a-base` | 87.5 (135k) | **16.7%** | collapsed 265k, 2.3% at 503k |
| `b3a-epsfloor` | 83.5 (236k) | 11.0% | 61.4 / 1.3% at 538k |
| `b3b-epsfloor2` | 85.8 (305k) | 8.3% | 52.0 / 3.3% at 541k |
| `b3c-buf500k` | 85.7 (312k) | 5.7% | 66.5 / 2.1% at 514k — **shallowest decline** |
| `b2a-base2` | 83.8 (293k) | 7.0% | 64.3 / 1.1% at 992k |

Degradation is therefore **systemic, not config-specific** — a bigger finding than any
individual arm's result, and the reason batch 4 targets the sampling machinery rather
than another scalar.

Logs are in `/Users/tony_wang/.claude/jobs/f3cb1855/tmp/b3{a,b,c}.log`, which is
job-scoped and will not survive; the durable record is `runs/<policy>_evals.json`,
so analyse from there.

Resume any of these by relaunching with the same policy name **and the same
`SNEK_*` overrides** — the overrides are *not* persisted in the checkpoint, so
relaunching without them silently changes the config mid-run and invalidates the
arm. For batch 3 that means `SNEK_MIN_EPSILON=0.001` for `b3a`/`b3b` and
`SNEK_REPLAY_BUFFER_MAX_LENGTH=500000` for `b3c`.

## Hypothesis A is falsified: epsilon 0.0 is not what causes the collapse

This was the batch's whole point, and the answer is no. Three independent lines of
evidence, all now past the horizon where they mean something:

| arm | epsilon regime | outcome |
|---|---|---|
| `b3b-epsfloor2` | **floored at 0.001** from 147k | peaked 305k, then declined 71 → 52 and 7.0% → 3.3% |
| `b3a-epsfloor` | **floored at 0.001** from 267k | peaked ~300k, then declined 74 → 61 and 8.6% → 1.3% |
| `b3c-buf500k` | **fully greedy at 0.0** from 282k | **did not break**; shallowest decline of any arm, and set its best score at 312k *after* going greedy |

The prediction was explicit and it failed in both directions: flooring epsilon was
supposed to prevent the degradation and did not, while the arm left fully greedy was
supposed to break around 430-460k and instead is the healthiest thing running at 491k.

**What the original correlation actually was.** `b1a-base` reaching epsilon 0.0 and
`b1a-base` collapsing were both consequences of it being the strongest run, not cause
and effect. The confound was noted at the time — "only strong runs reach the last
rung" — and it turned out to be the entire signal. Worth remembering how convincing
the correlation looked: one arm at 0.0, one arm collapsed, same arm, and a clean
mechanism to explain it.

**`MIN_EPSILON` stays in the code** — it defaults to 0.0 and changes nothing unless
set, and knowing epsilon 0.0 is *safe* is a useful result. But the epsilon ladder is
no longer a suspect, and the "make the threshold tunable" follow-up is not worth doing.

## `b3c-buf500k` died completely — which retracts the section below

**Retraction.** The section below called `b3c-buf500k` the most durable arm and read it
as support for hypothesis B. Run further, **it collapsed to score 0.0 at ~750k and never
recovered across the next 4 million steps.** Its cumulative perfect rate is 0.27%, the
worst of any arm that ever learned to play.

| steps | score mean | perfect mean |
|---|---|---|
| 250-300k | 70.7 | 4.2 |
| 650-700k | 58.9 | 0.4 |
| 700-750k | 34.3 | 0.0 |
| 750-800k | **0.6** | 0.0 |
| 800k-4.81M | ~0.0-3.2 | **0.0** |

Score 0 means the snake dies almost immediately, so this is total policy destruction, not
a dip — a **fourth failure mode**, and the most severe: irrecoverable.

Two things to take from it:

- **The "flattest curve in the investigation" praise was premature**, for the same
  reason as every other premature call here: it was a snapshot at 500k of a run that had
  another 4M steps of information in it. The flatness was real and then it died.
- **A dead policy is fast**, which is a trap. Score 0 means episodes end instantly, so
  evals become nearly free and throughput explodes — this arm raced from 523k to 4.81M
  steps in a few hours while the others did ~700k. Rapid step accumulation is a *symptom
  of failure*, the eval-cost confound in reverse. Do not read step count as progress.

Hypothesis B is therefore **not settled**: `b4b-unifbuf500k` (uniform + 500k buffer) is
healthy at 1.23M and steadily rising, while `b3c-buf500k` (PER + 500k buffer) died. The
difference between them is prioritization, not buffer size, which points the same way as
`b4c`.

## ~~Hypothesis B is supported: buffer diversity is the live lever~~ — see retraction above

`b3c-buf500k` is the only arm that has held up, and it is the only arm with a 5x
buffer:

| steps | `b3c-buf500k` (500k buffer) | `b2a-base2` (100k buffer) |
|---|---|---|
| 250-300k | 70.7 | 68.7 |
| 350-400k | 68.1 | 68.7 |
| 450-500k | **66.5** | 68.7 |
| 500-550k | — | 64.9 |

It is not immune — score drifts from 70.7 to 66.5 and its perfect rate is the batch's
lowest — but it degrades far more gently than the floored arms (which lost 13 and 19
points of score over the same span) and it held through the exact window where it was
predicted to break. Its chart is visibly the flattest in the investigation.

The mechanism still fits: the buffer holds *transitions*, and episodes get longer as
the policy improves, so a 100k buffer spans fewer and fewer distinct episodes exactly
when the policy is at its best. A 5x buffer slows that squeeze without eliminating it.

**Next test follows directly**: push further (`REPLAY_BUFFER_MAX_LENGTH=1000000`) and
pair it with `PRIORITY_EXPONENT=0.0`, since prioritized sampling from a large buffer
partly defeats the diversity the larger buffer buys.

## What `theSchlong` did differently — three PER changes, none validated

`theSchlong` (the 2022 code in the repo root, read-only) once reached a **76%
perfect-game rate**. Diffing it against `snek2` is more informative than any
hyperparameter sweep, because it is a working reference rather than a guess.

**What the 76% was**, since this has been got wrong twice: per the human, **a single
checkpoint that got very lucky** — not a mean over a run, and not a sustained level.
An earlier version of this section guessed it was `theSchlong`'s cumulative
`perfect_percentage` metric and compared it against ~6%; that was wrong.

The right comparison is therefore **best single eval**, and on that measure `snek2` has
now surpassed it: `b4c-schlongper` hit **80%** at step 970k, against 76%. The three
numbers worth tracking, for every arm:

| measure | what it answers |
|---|---|
| best single eval | the ceiling — comparable to the 76% |
| best 30-eval window | whether the ceiling is *held*, which is the actual goal |
| cumulative over run | how much of the run was spent playing well |

**Rewards, grid size, network shape and the epsilon ladder are byte-identical**, so
scores compare directly. `Snake.py` and `snake_environment.py` differ only
cosmetically (a policy-name overlay, a moved print, `set_display`/`get_score`
helpers). What actually changed is the replay buffer, and it changed in **three ways
at once**:

| aspect | `theSchlong` (76%) | `snek2` (~6% cumulative) | why it was changed |
|---|---|---|---|
| alpha | **0.8** | 0.6 | "0.6 against raw TD error is the usual choice" |
| priority signal | **`td_loss`** (element-wise Huber) | `abs(td_error)` | Huber shrinks small errors, widening spread → effective exponent ~1.6 |
| IS weights | **none at all** | mean-normalized, beta 0.4→1.0 | prioritizing without IS correction is biased |

Every one of those three is defensible in isolation, and *every one of them was
validated only at 30k steps* — below the ~250k line this document now says nothing can
be judged at. The measurement that justified them (alpha 0.6 beating 0.8-with-Huber)
is therefore worthless for this question.

There is also a mechanical interaction worth noting: **IS weights partially cancel
prioritization.** High-priority samples get downweighted gradients, so
`alpha=0.6 + IS` is a much gentler intervention than `alpha=0.8 + no IS`. The three
changes all push the same direction — less effective prioritization — which is why
they are worth testing together as well as separately.

Two knobs now make the old behaviour reachable: `SNEK_PRIORITY_SIGNAL`
(`td_error`|`td_loss`) and `SNEK_IS_WEIGHTS` (`1`|`0`). Both default to current
behaviour, so nothing else changes.

**Caveat from the human, and it matters:** the 76% was "very lucky and not consistent
at all." So `theSchlong`'s config was never a known-good target — it is evidence about
*variance*, and the prediction that it had a higher ceiling with a wider spread is
exactly what `b4c-schlongper` then showed. The goal remains **consistent learning and a
consistently rising perfect rate**, not chasing a spike.

**Outcome: this diff was the most productive thing in the investigation.** Reverting all
three changes doubled the best sustained perfect rate. See `b4c-schlongper` above.

## The baseline does not reach 50% perfect at 1M steps — it reaches ~1%

`b2a-base2` has now passed **967k steps** with the committed config, which is the
horizon and wall-clock where the premise of this whole investigation says to expect
**~50% perfect games**. Actual:

| steps | score mean | perfect mean |
|---|---|---|
| 150-200k | 69.6 | **5.2** (its best) |
| 500-550k | 64.9 | 3.8 |
| 750-800k | 66.6 | 3.0 |
| 950-1000k | 64.3 | **1.1** |

Its best perfect-game window in the entire run was 7.0%, and it is at 1.1% now. No arm
in this investigation has exceeded `b1a-base`'s 16.7%. That is a **factor of ~7 below
the premise at best, and ~45x below it at 1M steps.**

So one of these is true, and it matters which:

1. **The current code learns materially worse than the code the 50% figure came
   from.** The buffer changed to cpprb prioritized replay and the collect policy now
   runs under `tf.function`. Prioritized replay is the obvious suspect: it was already
   measured *worse* than uniform at 30k steps over 3 seeds (46.7 vs 60.1), and the
   only defence was "it might pay off at 1M". At 967k it has not.
2. The 50% figure came from a run with a much longer history — the human's `train`
   policy carries ~1.3M steps under the older code — and is not comparable to a fresh
   run.

Either way, **`PRIORITY_EXPONENT=0.0` at long horizon is now the highest-value
experiment available**, ahead of everything else in the queue. It is the one change
that both explains the gap and is already known to help at short horizon.

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

| policy | stopped at | verdict |
|---|---|---|
| `b3a-epsfloor` | 538k, 00:02 | Batch 3's best perfect rate (11.0% window) but degraded anyway to 1.3%. Hypothesis A falsified |
| `b3b-epsfloor2` | 541k, 00:02 | Longest floor exposure (from 147k) and the clearest refutation: rounded arc, score 71 → 52 |
| `b2a-base2` | 992k, 00:02 | The 1M-step reference. Answered its question: **2.64% cumulative perfect at ~1M**, best window 7.0% |

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

## ~~Leading hypothesis for the collapse: epsilon reaches exactly 0.0~~ — FALSIFIED

**This hypothesis is dead; see "Hypothesis A is falsified" above.** The section is kept
because the reasoning was sound given what was known, and because it is a clean example
of a correlation that was entirely confound. The evidence and mechanism as they stood:

The epsilon ladder in `maybe_update_epsilon()` is a one-way ratchet ending in
`epsilon.assign(0.0)` once `avg_reward > 100`. Tracing when each arm stepped down:

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

**The caveat recorded at the time turned out to be the whole story.** It read: reaching
0.0 requires `avg_reward > 100`, i.e. only *strong* runs get there, so "reached 0.0"
and "was good enough to collapse from a height" are entangled and the correlation
cannot separate them. That is exactly what happened — `b3c-buf500k` later ran fully
greedy from 282k without breaking, and both floored arms degraded anyway.

Two things worth carrying forward from how this played out:

- **The correlation was as strong as this domain ever produces** — one arm at 0.0, one
  arm collapsed, the same arm, a specific mechanism, and a timing that fit. It was
  still wrong. With n=1 arms and a stated confound, a mechanism that "fits the timing"
  adds no evidence.
- **The test was worth running anyway.** It cost one knob and three arms that were
  going to run regardless, and it converted the leading hypothesis into a closed
  question while incidentally producing the batch's best arm (`b3a`, 11.0%) and the
  natural experiment that settled it (`b3c` going greedy).

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

### Batch 4 — running now, and what follows it

Batch 3 killed the epsilon hypothesis and pointed at sampling and buffer diversity
instead. **Every arm needs to run past ~350k to mean anything** — degradation starts at
236-312k in every arm observed so far — so expect 4+ hours each.

| priority | change | status |
|---|---|---|
| **1** | `PRIORITY_EXPONENT=0.0` | **running** as `b4a-uniform` — modest, 8.7% window |
| **2** | `PRIORITY_EXPONENT=0.0` + 500k buffer | **running** as `b4b-unifbuf500k` — steady, 9.3% window |
| **3** | `theSchlong` PER: alpha 0.8, `td_loss`, no IS | **running** as `b4c-schlongper` — **34.0% window, the clear winner** |

Spanning the whole prioritization axis — none (`b4a`), none-plus-diversity (`b4b`),
maximum (`b4c`) — paid off: the answer was at the end nobody expected. Uniform sampling
was the prior favourite and came in at a third of `b4c`'s rate.

### Batch 5 — the obvious follow-ups

| priority | change | why |
|---|---|---|
| **1** | `b4c` config, **repeated 2-3x** | it is n=1 with a collapse in its own history. Nothing else matters until this holds |
| **2** | alpha 0.8 + `td_loss` + **IS weights ON** | of `b4c`'s three reverted changes, this isolates the IS one — the most likely single cause, since IS weights directly cancel prioritization |
| **3** | alpha 0.8 + `abs(td_error)` + no IS | isolates the priority-signal change |
| 4 | `b4c` config + 500k buffer | `b4b` beat `b4a` slightly, so diversity may stack on top of the winner |
| 5 | `DISCOUNT=0.995` | still untested and still high-prior; the perfect-game bonus is discounted to near-nothing at 0.99 |
| 6 | `GRADIENT_CLIPPING=10` | cheap, independent, and `b4c`'s variance is what needs taming |

Priorities 2 and 3 matter because `b4c` reverted **three** things at once. Knowing which
one carries the gain is worth more than another point on any scalar knob, and the knobs to
separate them already exist.

Dropped: the LR schedule (no evidence of optimization instability — degradation is
gradual in every arm, not spiky) and making the epsilon threshold tunable (the ladder is
no longer a suspect).

### Batch 3 — reprioritized around the collapse (completed)

The collapse at ~265k was thought to be the main obstacle, so this batch led with
things that plausibly prevented it. **Result: A falsified, B supported.**

| item | change | status |
|---|---|---|
| A | floor epsilon at 0.001 | **run and FALSIFIED** as `b3a-epsfloor`, `b3b-epsfloor2` |
| B | `REPLAY_BUFFER_MAX_LENGTH=500000` | **run and SUPPORTED** as `b3c-buf500k` |
| C | `PRIORITY_EXPONENT=0.0` at long horizon | **promoted to batch 4 priority 1** |
| D | `GRADIENT_CLIPPING=10` | still queued |
| E | third baseline repeat | effectively done — `b3a` spent 267k steps as one |
| F | LR schedule | **dropped**, see above |

#### A. Floor epsilon at 0.001 — never 0.0 ▶ RUN, FALSIFIED

Top priority at the time, on the evidence in the epsilon-ladder section: the one arm
that reached epsilon 0.0 was the one arm that collapsed, and at 0.0 the buffer becomes
a closed policy-data feedback loop. Expected: the same 14%-class perfect rate as
`b1a-base` but held past 300k. Run **twice**, because a single non-collapsing run would
prove nothing.

**Outcome: both floored arms degraded anyway, and the arm left fully greedy held up
best.** The prediction below about how to read the arm was also the wrong worry — the
risk was never that flooring would cap the ceiling.

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
