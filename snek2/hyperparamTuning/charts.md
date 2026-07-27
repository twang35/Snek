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

Snapshot refreshed 09:00, at the steps noted below.

## Every arm at a glance

| policy | change | steps | peak score (at) | best perfect-30 | best single eval | verdict |
|---|---|---|---|---|---|---|
| `b4c-schlongper` | alpha 0.8, `td_loss`, no IS | 1.06M | **92.0** (869k) | **34.0%** | **80%** | **best arm by 2x**; high ceiling, high variance |
| `b4b-unifbuf500k` | alpha 0 + 500k buffer | 1.23M | 86.6 (743k) | 9.3% | 40% | steady, slowly rising |
| `b4a-uniform` | alpha 0 | 1.25M | 85.9 (550k) | 8.7% | 40% | peaked ~575k, drifting down |
| `b1a-base` | none (control) | 503k | 87.5 (135k) | 16.7% | 40% | collapsed at 265k; score recovered, skill did not |
| `b3a-epsfloor` | `MIN_EPSILON=0.001` | 545k | 83.5 (236k) | 11.0% | 30% | best of batch 3, degraded anyway |
| `b3b-epsfloor2` | `MIN_EPSILON=0.001` | 549k | 85.8 (305k) | 8.3% | 30% | declined despite the floor — falsified hypothesis A |
| `b2a-base2` | none (repeat) | 999k | 83.8 (293k) | 7.0% | 30% | no collapse; long drift down, 1.1% perfect at 1M |
| `b3c-buf500k` | 500k buffer, alpha 0.6 | 4.81M | 85.7 (312k) | 5.7% | 30% | **died at ~750k**, score 0.0 for 4M steps |
| `b1c-nstep3` | `N_STEP_UPDATE=3` | 1.14M | 76.0 (255k) | 1.7% | 10% | dead end |
| `b1b-tgt200` | `TARGET_UPDATE_PERIOD=200` | 106k | 76.9 | 1.0% | 10% | stopped early, verdict weak |
| `b2b-nstep2` | `N_STEP_UPDATE=2` | 580k | 74.6 (140k) | 0.7% | 10% | dead end |

**The comparison to take from this table:** `b4c-schlongper` — which reverts the three
PER changes made when the buffer was ported — beats every other arm by roughly 2x on the
sustained perfect rate and is the only arm that has ever produced an eval above 40%. It
also holds the top score at 92.0 of a possible 95.

Uniform sampling (`b4a`, `b4b`) was the prior favourite and landed at about a third of
`b4c`'s rate, so the axis mattered but the expected direction was wrong. Everything above
`b4a` in this table degraded after peaking somewhere between 236k and 312k; `b4c` peaked
at 875k instead, which is the first real break from that pattern.

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

Step 968k · peak score 83.8 (at 293k) · trailing-30 perfect **0.7%** · best perfect-30 7.0%

The counterpart to `b1a-base` and the reason repeats matter. Same config, and it ran
**well past the 265k step where its twin collapsed** without ever breaking — no cliff
anywhere on this curve.

**This is also the reference run for the premise of the whole investigation, and it
misses badly.** At 967k steps — the horizon where ~50% perfect games was expected —
its 950-1000k block is 64.3 score and **1.1% perfect**. Its best window all run was
7.0%. See "The baseline does not reach 50% perfect at 1M steps" in `runs.md`.

The chart's other useful feature is its **very long wavelength**: score dips to a
trough near 575k, recovers to ~66 by 760k, then drifts down again. At 680k this looked
like terminal decay and was written up as such; 80k steps later it looked like a
recovery; by 967k it is a shallow downward drift with big slow swings. A trough spans
~100k steps, so snapshots 150k apart give opposite verdicts — hence the rule against
calling trends from the most recent window.

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

## Batch 3 — the epsilon hypothesis, settled

All three arms are past 490k, all three epsilon treatments engaged long ago, and the
verdict is in: **the two floored arms degraded and the fully greedy arm did not.**
Compare the three charts below in order — they are the clearest evidence in this
investigation, because the prediction was specific and it failed both ways.

### b3a-epsfloor — `MIN_EPSILON=0.001`, floored since 267k

Step 515k · peak score 83.5 (at 236k) · **best perfect-30 11.0%** · latest block 61.4 / 1.3%

**Batch 3's best perfect rate, and it still degraded.** Its perfect rate climbed
2.6 → 4.4 → 6.2 → **8.6** across the 50k blocks to 300k, with the 250-300k block
holding the highest score mean of any arm at any point (74.4, floor 58.2).

Then it turned: 8.6% → 6.8 → 2.6 → 3.6 → 3.4 → **1.3**, with score sliding 74.4 → 61.4.
The floor was engaged from 267k, i.e. for the entire decline, so it did not prevent
anything.

![b3a-epsfloor](charts/b3a-epsfloor.png)

### b3b-epsfloor2 — `MIN_EPSILON=0.001`, floored since 147k

Step 516k · peak score **85.8** (at 305k) · best perfect-30 8.3% · latest block 52.0 / 3.3%

The arm with the longest treatment exposure — floored from 147k, 369k steps of it — and
**the clearest single refutation of hypothesis A.**

The chart is a rounded arc: up to a peak at ~305k, then a decline that both deepens and
widens, with the blue trace's troughs reaching into the 30s by 500k where earlier lows
were in the 50s. Growing variance alongside a falling mean is the signature to
recognise. Its worst eval in the 500-550k block is 28.2, against 52.2 in the 250-300k
block. Zero exploration was never involved.

![b3b-epsfloor2](charts/b3b-epsfloor2.png)

### b3c-buf500k — `REPLAY_BUFFER_MAX_LENGTH=500000`

Step 492k · peak score **85.7** (at 312k) · best perfect-30 5.7% · latest block 66.5 / 2.1%

**The arm that settled the batch, and the flattest curve in the investigation.** It
crossed `avg_reward > 100` at 282k with `MIN_EPSILON` at its 0.0 default, so it has been
running **fully greedy since 282k** — the condition that was supposed to destroy it.

The prediction was that it would break around 430-460k regardless of buffer size. **It
did not.** It set its best score of the whole run at 312k, *after* going greedy, and at
492k its 450-500k block is 66.5 against 70.7 at 250-300k — a 4-point slide where the
floored arms lost 13 and 19. Its trailing-30 perfect rate is also the only one currently
rising (0.7% → 2.7%).

Compare this chart with `b3b`'s directly: same time span, same peak height, but this one
has no arc and no variance growth. The one config difference that matters is the 5x
buffer.

**Retracted at 4.81M steps: this arm died.** Score fell to 0.0 at ~750k and stayed there
for the next 4 million steps, giving it the worst cumulative perfect rate (0.27%) of any
arm that ever learned to play. The praise above was a snapshot at 500k of a run with 4M
steps of information still in it.

Note the shape of the tail on this chart — it is what total policy destruction looks
like, and it is nothing like the gentle slides elsewhere in this file. Also note that a
dead policy is *fast*: episodes end instantly, so evals become free and this arm raced
to 4.81M steps while its batch mates did ~1.2M. Step count is not progress.

![b3c-buf500k](charts/b3c-buf500k.png)

---

## Batch 4 — the sampling machinery, and the breakthrough

Three arms spanning the prioritization axis: none, none-plus-diversity, and `theSchlong`'s
original maximum. The last one won by a wide margin.

### b4c-schlongper — alpha 0.8, `td_loss` priorities, no IS weights

Step 1.06M · peak score **92.0** of 95 (at 869k) · **best 30-eval perfect 34.0%** · **checkpoint 869k measures 51.0% over 100 episodes**

**The best arm in the investigation, by roughly 2x on every measure that matters.** This
is `theSchlong`'s exact PER configuration — the three changes made during the cpprb port,
all reverted together.

Its checkpoint at 869k was reloaded and evaluated over 100 greedy episodes: **51.0%
perfect (95% CI 41.3-60.6%), median score 95 of 95.** It wins more than half the games it
plays. See "Measured" in [`runs.md`](runs.md) for all four checkpoint measurements.

The red trace is unmistakable against every other chart in this file: sustained 40-60%
spikes from 700k onward, **41 separate evals at >=50%**, and a peak of 80% at 970k. No
other arm has ever produced a single eval above 40%.

It is also the highest-variance arm here, and the chart shows why that matters: a **severe
collapse from 150k to 300k** takes score from 74 down to ~19 before it recovers. Judged at
300k — the horizon this document uses — this arm would have been killed as a failure. It
then climbed for 600k steps to a level nothing else has approached. The dip to 0% at
~1046k is the same phenomenon recurring; it is already recovering (71.4 score / 20% at
1058k).

High ceiling, wide spread — exactly what the human described `theSchlong` as. Consistency
is now the open problem rather than ceiling.

![b4c-schlongper](charts/b4c-schlongper.png)

### b4b-unifbuf500k — alpha 0 (uniform) + 500k buffer

Step 1.23M · peak score 86.6 (at 743k) · best 30-eval perfect 9.3% · cumulative 4.03%

Uniform sampling with a 5x buffer, and the **steadiest arm in the investigation** — the
blue trace holds ~65-69 for a million steps with no collapse and its perfect rate is still
slowly rising (5.7% trailing, up from 4.3%). Contrast with `b3c-buf500k`, which had the
same buffer *with* PER and died: the difference between them is prioritization.

Steady, though, at a low level. This is the "stable but stuck" pattern again, and it is
the trade `b4c` refuses to make.

![b4b-unifbuf500k](charts/b4b-unifbuf500k.png)

### b4a-uniform — alpha 0 (uniform), default buffer

Step 1.25M · peak score 85.9 (at 550k) · best 30-eval perfect 8.7% · cumulative 3.50%

Plain uniform sampling, which was the prior favourite going into this batch and came in at
about a third of `b4c`'s rate. Its perfect rate built steadily to 6.8% around 575k and has
drifted down since, ending near 1.8%.

Useful as the clean control: **removing prioritization entirely is better than the
committed `alpha=0.6 + IS` config but far worse than `theSchlong`'s aggressive
prioritization.** So the relationship is not monotonic in "how much prioritization", which
is why isolating which of `b4c`'s three changes carries the gain is batch 5's priority.

![b4a-uniform](charts/b4a-uniform.png)
