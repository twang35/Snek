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

Snapshot taken at the steps noted below.

---

## Batch 1

### b1a-base — control, committed defaults

Step 164k · peak score 87.5 · last-5 71.1 · **trailing-30 perfect 10.7%**

Steady climb to a ~70 plateau by 40k, then score stops improving while the
**perfect-game rate keeps climbing** — 1.7% → 3.3% → 10.7%. The clearest
demonstration so far that score and perfect rate can move independently, and the
reason both are tracked.

![b1a-base](charts/b1a-base.png)

### b1b-tgt200 — `TARGET_UPDATE_PERIOD=200`

Step 106k (stopped) · peak score 76.9 · last-5 62.5 · trailing-30 perfect 1.0%

Fastest early riser in the batch, reaching ~55 by 15k where the baseline needed
~25k, and first perfect game at 33k. But it settled into a noisy plateau slightly
below the baseline and its perfect rate never took off. Stopped to free a slot;
resumable with `SNEK_TARGET_UPDATE_PERIOD=200`.

![b1b-tgt200](charts/b1b-tgt200.png)

### b1c-nstep3 — `N_STEP_UPDATE=3`

Step 288k · peak score 76.0 · last-5 53.0 · trend **−8.5** · trailing-30 perfect 1.0%

The arm that most changed the reading of this batch. Slowest to rise, and at
matched steps it looked like a clear loser, but through 200k it was the only arm
still gaining while the others flattened. It has since **peaked around 76 and
turned back down**, so the momentum that made it interesting has broken. It did
eventually produce a perfect game.

![b1c-nstep3](charts/b1c-nstep3.png)

---

## Batch 2

### b2a-base2 — baseline repeat

Step 83k · peak score 76.1 · last-5 57.1 · trailing-30 perfect 1.0%

Second sample of the committed defaults, running to establish how much of batch 1
was seed luck. At 83k it is tracking below where `b1a-base` was at the same step
(66.0), which is already useful: it suggests `b1a-base` was on the fortunate side
rather than typical.

![b2a-base2](charts/b2a-base2.png)

### b2b-nstep2 — `N_STEP_UPDATE=2`

Step 101k · peak score 62.1 · last-5 55.2 · trend **+7.8** · trailing-30 perfect 0.0%

Tests whether n-step's steadiness survives at a faster rate than n=3. Still
climbing strongly at 101k. No perfect game yet, same weakness as n=3.

![b2b-nstep2](charts/b2b-nstep2.png)
