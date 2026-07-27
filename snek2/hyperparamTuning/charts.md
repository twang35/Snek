# Charts

Progress graphs for every tuning arm. Companion to
[`hyperparamTuning.md`](hyperparamTuning.md) (the protocol) and
[`runs.md`](runs.md) (status, results, queue).

In every chart: **blue is average score** (food eaten, out of a possible 95) on the
left axis, **red is perfect-game percentage** on the right. Grey dashed vertical
lines mark points where training was resumed.

## These are snapshots, on purpose

The images here are **copies** taken from `snek2/runs/`, not links to it. The live
graphs in `runs/` are rewritten on every eval and would be lost if that directory
were ever cleaned out, which would silently blank out this file. Copies mean the
documented history survives independently.

The trade-off is that they go stale. Refresh them with:

```
snek2/hyperparamTuning/refresh_charts.sh
```

That re-copies every `runs/*.png` into `charts/` and prints the step each one is
at, so captions can be updated to match.

Snapshot refreshed 21:35, at the steps noted below.

## Every arm at a glance

| policy | change | steps | peak score (at) | best perfect-30 | verdict |
|---|---|---|---|---|---|
| `b1a-base` | none (control) | 503k | **87.5** (135k) | **16.7%** | collapsed at 265k; score recovered, skill did not |
| `b2a-base2` | none (repeat) | 763k | 83.8 (293k) | 7.0% | no collapse; long oscillation, recovering from a trough |
| `b1b-tgt200` | `TARGET_UPDATE_PERIOD=200` | 106k | 76.9 | 1.0% | stopped early, verdict weak |
| `b1c-nstep3` | `N_STEP_UPDATE=3` | 1.14M | 76.0 (255k) | 1.7% | dead end |
| `b2b-nstep2` | `N_STEP_UPDATE=2` | 580k | 74.6 (140k) | 0.7% | dead end |
| `b3a-epsfloor` | `MIN_EPSILON=0.001` | 314k | 83.5 (236k) | **11.0%** | **running**; best batch-3 arm, floored since 267k |
| `b3b-epsfloor2` | `MIN_EPSILON=0.001` | 311k | 85.8 (305k) | 8.3% | **running**; floored since 147k, inside the collapse window |
| `b3c-buf500k` | `REPLAY_BUFFER_MAX_LENGTH=500000` | 294k | 81.3 (144k) | 5.7% | **running**; went to epsilon 0.0 at 282k — key test |

**The comparison to take from this table:** `b1a-base` still holds the best
perfect-game window at 16.7%, but `b3a-epsfloor` has closed most of that gap at 11.0%
and has not collapsed. The n-step arms sit far below everything else at 0.7-1.7%. No
arm has yet been both good *and* durable across a full run, which is what batch 3 is
trying to break.

---

## Batch 1

### b1a-base — control, committed defaults

Step 414k · peak score 87.5 (at 135k) · latest 60.5 · perfect peaked at 14% mean / 40% spikes, now 2.9%

**The most important chart here: catastrophic forgetting, caught in full — and then
a half-recovery.** Broad plateau from 30k to ~260k with the perfect rate (red)
building to 14% mean and 40% spikes, a hard break at ~265k down to 20-40, then an
unaided climb back to ~65 from 350k on.

The thing to look at is the **red trace after 350k**, not the blue one. Blue
recovers; red stays sparse and short. Score came back, the ability to finish games
did not. This is the clearest picture in the investigation of why score alone is
the wrong late-run metric.

This happened *after* its best perfect rate — the run was on track for a good
result and destroyed it. `b2a-base2` below is the identical config and has not
collapsed, so this is stochastic rather than inherent to the config. It is also the
only arm that ever drove epsilon to exactly 0.0 (at 92k), which is the leading
suspect; see `runs.md`.

![b1a-base](charts/b1a-base.png)

### b1b-tgt200 — `TARGET_UPDATE_PERIOD=200`

Step 106k (stopped) · peak score 76.9 · last-5 62.5 · trailing-30 perfect 1.0%

Fastest early riser in the batch, reaching ~55 by 15k where the baseline needed
~25k, and first perfect game at 33k. But it settled into a noisy plateau slightly
below the baseline and its perfect rate never took off. Stopped to free a slot;
resumable with `SNEK_TARGET_UPDATE_PERIOD=200`.

![b1b-tgt200](charts/b1b-tgt200.png)

### b1c-nstep3 — `N_STEP_UPDATE=3`

Step 858k · peak score 76.0 (at 255k) · last-5 29.7 · trailing-30 perfect 0.0%

**A complete arc, and a negative result.** Slowest to rise; at matched steps it
looked like a clear loser; through 200k it was the only arm still gaining while the
others flattened, which made it the most interesting arm in batch 1. Then it peaked
at 76 around 255k and **declined for the next 600k steps**, settling flat at ~30.

Its only perfect games were a handful around 206-300k; nothing since. The long
right-hand tail is the useful part of this chart — it is what "promising trajectory
that simply runs out" looks like, and it is only visible at this horizon.

![b1c-nstep3](charts/b1c-nstep3.png)

---

## Batch 2

### b2a-base2 — baseline repeat

Step 763k · peak score 83.8 (at 293k) · trailing-30 perfect 2.7% · best perfect-30 7.0%

The counterpart to `b1a-base` and the reason repeats matter. Same config, and it ran
**well past the 265k step where its twin collapsed** without ever breaking — no cliff
anywhere on this curve.

The useful feature of this chart is its **very long wavelength**. Score dips from ~70
to a trough around 575k and comes back to ~66 by 760k; the red trace thins to almost
nothing through 550-650k and then returns. At 680k this looked like slow terminal
decay and was written up as such — 80k steps later it is clearly an oscillation
instead. A trough here spans ~100k steps, which is longer than many whole runs, so
snapshots taken 150k steps apart give opposite verdicts.

Also worth noting: it never triggered the last epsilon rung, its `avg_reward` peaking
at 99.1 against the threshold of 100.

![b2a-base2](charts/b2a-base2.png)

### b2b-nstep2 — `N_STEP_UPDATE=2`

Step 414k · peak score 74.6 (at 140k) · last-5 48.3 · trailing-30 perfect 0.0%

Tested whether n-step's steadiness survives at a faster rate than n=3. It does not:
this is the same shape as the n=3 chart, one step milder — peak below either
baseline, then a long slow decline. Trailing-30 perfect rate is **0.0%**, and only
two isolated perfect evals in its whole history.

Two arms, same shape, ordered by n. That is a trend rather than noise, and it closes
the n-step direction.

![b2b-nstep2](charts/b2b-nstep2.png)

---

## Batch 3 — in progress

All three are past 290k and their epsilon treatments have all now engaged, but none
has enough exposure yet for a verdict. Charts are here so the trajectories can be
watched, not judged.

### b3a-epsfloor — `MIN_EPSILON=0.001`, floored since 267k

Step 314k · peak score 83.5 (at 236k) · **best perfect-30 11.0%** · latest block 6.4%

**The strongest batch-3 arm and the best non-`b1a` result in the investigation.** Its
perfect rate climbed 2.6 → 4.4 → 6.2 → **8.6** across the 50k blocks to 300k, and the
250-300k block also has the highest score mean of any arm at any point (74.4) with a
floor of 58.2.

It crossed `avg_reward > 100` at 267k, so the floor has only been engaged for ~46k
steps — everything above was achieved at 0.001 anyway, before the treatment could
matter. That makes this chart mostly evidence about the *baseline* config being better
than `b2a-base2` suggested, not yet evidence for the floor.

![b3a-epsfloor](charts/b3a-epsfloor.png)

### b3b-epsfloor2 — `MIN_EPSILON=0.001`, floored since 147k

Step 311k · peak score **85.8** (at 305k) · best perfect-30 8.3% · latest block 3.6%

The arm with the longest treatment exposure: it crossed `avg_reward > 100` at 147k,
where an unfloored run would have ratcheted permanently to 0.0, so it has been
diverging from the default for 163k steps.

It set its highest score of the run at 305k, but the red trace has thinned over the
last ~30 evals (7.0% in the 250-300k block down to 3.6%) and score has come off that
peak. **This is exactly the window that matters**: `b1a-base` collapsed 173k steps
after its own divergence, which maps to ~320k here. Whether the current dip is that
event starting or ordinary oscillation is the single open question in this batch.

![b3b-epsfloor2](charts/b3b-epsfloor2.png)

### b3c-buf500k — `REPLAY_BUFFER_MAX_LENGTH=500000`

Step 294k · peak score 81.3 (at 144k) · best perfect-30 5.7% · latest block 4.5%

**The smoothest curve in this document, and now the batch's most informative arm.**
The blue trace holds a tight band from 40k on, and its worst eval in the 250-300k block
was 60.8 — the highest floor any arm has managed. A bigger buffer does buy stability,
which is the part of hypothesis B that looks right. Its perfect rate is the batch's
lowest but has been climbing steadily (0.8 → 3.6 → 4.5).

**It crossed `avg_reward > 100` at 282k with `MIN_EPSILON` at its 0.0 default, so it is
now running fully greedy with a 5x buffer** — the head-to-head between the two
hypotheses that nothing was designed to test. If zero exploration is what drives the
collapse, this arm should break around 430-460k regardless of buffer size; if buffer
diversity is what matters, the smooth curve should hold. Watch this one.

![b3c-buf500k](charts/b3c-buf500k.png)
