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

Snapshot refreshed 20:20, at the steps noted below.

## Every arm at a glance

| policy | change | steps | peak score (at) | best perfect-30 | verdict |
|---|---|---|---|---|---|
| `b1a-base` | none (control) | 503k | **87.5** (135k) | **16.7%** | collapsed at 265k; score recovered, skill did not |
| `b2a-base2` | none (repeat) | 681k | 83.8 (293k) | 7.0% | no collapse, but slowly decaying to 1.3% |
| `b1b-tgt200` | `TARGET_UPDATE_PERIOD=200` | 106k | 76.9 | 1.0% | stopped early, verdict weak |
| `b1c-nstep3` | `N_STEP_UPDATE=3` | 1.14M | 76.0 (255k) | 1.7% | dead end |
| `b2b-nstep2` | `N_STEP_UPDATE=2` | 580k | 74.6 (140k) | 0.7% | dead end |
| `b3a-epsfloor` | `MIN_EPSILON=0.001` | 239k | 83.5 (236k) | 6.3% | **running**; floor hasn't fired, so a de facto baseline |
| `b3b-epsfloor2` | `MIN_EPSILON=0.001` | 232k | 84.3 (198k) | 7.7% | **running**; the only real treatment arm |
| `b3c-buf500k` | `REPLAY_BUFFER_MAX_LENGTH=500000` | 215k | 81.3 (144k) | 2.0% | **running**; smoothest curve, lowest ceiling |

**The comparison to take from this table:** `b1a-base` still has more than double any
other arm's perfect rate, and it is the only arm that ran at epsilon 0.0. Everything
else clusters at 0.7-7.7%. Nothing yet has been both good *and* durable — the arms
either peak high and break, or hold steady and never climb.

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

Step 681k · peak score 83.8 (at 293k) · latest 65.7 · trend −2.2 · trailing-30 perfect 1.3%

The counterpart to `b1a-base` and the reason repeats matter. Same config, and it ran
**well past the 265k step where its twin collapsed** without ever breaking — no
cliff anywhere on this curve.

But run out to 681k it is not the stability success it looked like at 350k. The blue
trace sags from ~70 to ~63 after 500k and the red trace thins out to almost nothing:
5.2% perfect in the 150-200k block down to 1.3% now. **This is what slow decay looks
like, as opposed to `b1a-base`'s collapse** — no single event to point at, and max
drawdown barely registers it. Two different ways to lose a good policy.

Also worth noting: it never triggered the last epsilon rung (its `avg_reward` peaked
at 99.1, just under the 100 threshold), and its ceiling was half `b1a-base`'s.

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

All three are still running and none has reached the ~300k horizon where a verdict
means anything. Charts are here so the trajectories can be watched, not judged.

### b3b-epsfloor2 — `MIN_EPSILON=0.001`, the real treatment arm

Step 232k · peak score 84.3 (at 198k) · trend **+3.4** · trailing-30 perfect **6.7%** ↑

The only arm whose epsilon floor has actually fired: it crossed `avg_reward > 100`
once at 147k, where an unfloored run would have ratcheted permanently to 0.0. So this
curve is diverging from the default from 147k onward.

Both traces are still rising at 232k, and the red spikes reach 20-30% and have gotten
denser since 150k. Encouraging, but `b1a-base` looked like this too before it broke —
it collapsed 173k steps after its own divergence, so the equivalent test point here
is ~320k.

![b3b-epsfloor2](charts/b3b-epsfloor2.png)

### b3a-epsfloor — `MIN_EPSILON=0.001`, but inert

Step 239k · peak score 83.5 (at 236k) · trend −2.3 · trailing-30 perfect 5.0%

Same config as `b3b`, but it has **never crossed `avg_reward > 100`**, so its floor
has never engaged and its epsilon sits at 0.001 exactly where an unfloored run's
would. Treat this as an accidental third baseline repeat rather than a treatment arm.

Useful anyway: as a baseline it reached a 6.3% best perfect-30 window against
`b2a-base2`'s 7.0% and `b1a-base`'s 16.7%, which is more evidence that `b1a-base` was
the outlier rather than the norm.

![b3a-epsfloor](charts/b3a-epsfloor.png)

### b3c-buf500k — `REPLAY_BUFFER_MAX_LENGTH=500000`

Step 215k · peak score 81.3 (at 144k) · **drop from peak only 2.3** · trailing-30 perfect 2.0%

**The smoothest curve in this whole document.** The blue trace holds a tight 60-75
band from 40k onward with no excursions, and its worst eval in the 200-250k block was
64.1 — the highest floor any arm has managed. A bigger buffer clearly does buy
stability, which is the part of hypothesis B that looks right.

The red trace is the problem: every spike is a single 10% eval, never the 20-30%
clusters the epsilon arms produce, and the rate is 2.0% against their 5-7%. So this
arm is trading ceiling for consistency — the same trade `b2a-base2` made, arrived at
by a different mechanism.

![b3c-buf500k](charts/b3c-buf500k.png)
