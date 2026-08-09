# Charts

Progress graphs for the **six most recent batches**, newest first. Per-arm numbers live in
[`completedRuns.md`](completedRuns.md); this file is images plus a short reading of each.

**Older sections are retired, not deleted.** Batches 1-11 are in
[`archive/batches1-11.md`](archive/batches1-11.md) and anything retired since is in
[`archive/charts-archive.md`](archive/charts-archive.md). **The PNGs are all still in `charts/`**, so
an archived caption still renders. See
[when an arm is stopped](hyperparamTuning.md#when-you-stop-a-batch-of-arms) for the procedure.

In every chart: **blue is average score** (food eaten, out of 95) on the left axis, **red is
perfect-game percentage** on the right. Grey dashed vertical lines mark resumes; faint red dashed
horizontals mark 20/40/60/80% on the right axis, because the perfect rate is the objective and
was unreadable against left-axis ticks.

**Newest batch first.** Within a batch, best result first.

## These are snapshots, on purpose

The images are **copies** from `snek2/runs/`, not links. The live graphs there are rewritten every
eval and would be lost if that directory were cleaned out, silently blanking this file. Refresh with
`refresh_charts.sh`, which re-copies every `runs/*.png` into `charts/` and prints each one's step.

**The script does not touch this file** — it copies images only, so a new arm gets a PNG and no
entry unless one is written by hand. That drifted once, to 12 undocumented arms across batches 5-7,
because a successful `refresh_charts.sh` looked like the charts were handled. Check this file **and both
archives**, since captions now live in three places — `archive/charts-archive.md` was missing from this
snippet until 2026-08-08, which would have reported every retired arm as undocumented:

```
cd snek2/hyperparamTuning
ls charts/*.png | sed 's|.*/||;s|\.png||' | sort > /tmp/have
grep -ho 'charts/[a-zA-Z0-9-]*\.png' charts.md archive/batches1-11.md archive/charts-archive.md \
  | sed 's|.*charts/||;s|\.png||' | sort -u > /tmp/doc
comm -23 /tmp/have /tmp/doc   # anything listed is an undocumented arm
```

## Batch 20 wave 1 — FC layer capacity (`FC_LAYERS`), the first architecture test, stopped at 1.73-2.78M

Eight arms across two hosts: the control `50,100,50` re-baselined at β=300k (`b20a-d`, laptop) and the
capacity arm `200,100,50`, 2.66× the parameters (`b20e-h`, desktop). This is the first time the network
shape has been varied in the project, aimed at the one quantity nine batches of optimiser knobs never
moved — the ceiling.

**Provisional, and not at a matched horizon.** The laptop control ran to 2.49-2.78M; the desktop
treatment crashed in an OOM cascade at **~1.75M** (checkpoints preserved), so each treatment arm is
~0.8M shorter than its seed-matched control. `strong_eval_fraction` is a fraction of each arm's own
evals, so it is inflated for the longer control and the two columns are **not comparable** until the
control is truncated to ~1.75M or the treatment is rerun to 2.5M — [pending](runs.md).

**The headline survives the caveat: 2.66× capacity did not move the ceiling.** Peak trailing sits at
~94.5 for both shapes — control mean **94.44**, treatment mean **94.50** — squarely inside the 94.7-95.0
band every batch has held since 11. best-30 is a wash too (control 64.0, treatment 65.6), and the
treatment reached its level ~0.8M *sooner*, but from a crashed run that never got the extra steps to
either confirm or give it back. Capacity-up is not, on this evidence, a route to a higher ceiling.

| arm | shape | host | step | peak trail | best-30 | `sef` | recent-30 |
|---|---|---|---|---|---|---|---|
| `b20d` | 50,100,50 | laptop | 2.73M | 94.76 | **80.3%** | 28.4% | 47.7% |
| `b20b` | 50,100,50 | laptop | 2.78M | 94.84 | 78.3% | 16.0% | 63.7% |
| `b20c` | 50,100,50 | laptop | 2.55M | 94.34 | 56.3% | 2.1% | 43.0% |
| `b20a` | 50,100,50 | laptop | 2.49M | 93.80 | 41.3% | 0.2% | 24.0% |
| `b20h` | 200,100,50 | desktop | 1.80M | **94.82** | 76.3% | 10.5% | **68.0%** |
| `b20g` | 200,100,50 | desktop | 1.73M | 94.62 | 72.3% | 11.9% | 59.7% |
| `b20e` | 200,100,50 | desktop | 1.75M | 94.56 | 68.3% | 7.5% | 44.0% |
| `b20f` | 200,100,50 | desktop | 1.77M | 93.98 | 45.3% | 0.6% | 37.0% |

### Control `50,100,50` (laptop), best first

`b20d` — 2.73M, peak 94.76, best-30 **80.3%** @917k, `sef` 28.4%, recent-30 47.7%.
![b20d](charts/b20d-fc50seed4.png)

`b20b` — 2.78M, peak 94.84, best-30 78.3% @1498k, `sef` 16.0%, recent-30 63.7%.
![b20b](charts/b20b-fc50seed2.png)

`b20c` — 2.55M, peak 94.34, best-30 56.3% @1959k, `sef` 2.1%, recent-30 43.0%.
![b20c](charts/b20c-fc50seed3.png)

`b20a` — 2.49M, peak 93.80, best-30 41.3% @1900k, `sef` 0.2%, recent-30 24.0% — the weakest arm of the eight.
![b20a](charts/b20a-fc50seed1.png)

### Treatment `200,100,50` (desktop, 2.66× params), best first

All four crashed together at ~1.75M via the OOM→X-break→fatal-XIO cascade of 2026-08-09, not on their
own trajectory — every one was healthy (`zero_since` null) and still climbing when it died.

`b20h` — 1.80M, peak **94.82**, best-30 76.3% @1755k, `sef` 10.5%, recent-30 **68.0%** — best of the treatment arms.
![b20h](charts/b20h-fc200seed4.png)

`b20g` — 1.73M, peak 94.62, best-30 72.3% @1689k, `sef` 11.9%, recent-30 59.7%.
![b20g](charts/b20g-fc200seed3.png)

`b20e` — 1.75M, peak 94.56, best-30 68.3% @1000k, `sef` 7.5%, recent-30 44.0%.
![b20e](charts/b20e-fc200seed1.png)

`b20f` — 1.77M, peak 93.98, best-30 45.3% @1563k, `sef` 0.6%, recent-30 37.0% — the treatment's laggard, mirroring `b20a` in the control.
![b20f](charts/b20f-fc200seed2.png)

## Batch 19 — standard PER (`td_error` priority + IS on), falsified, stopped at 2.00-2.42M

Four seeds, batch 18's config byte-for-byte with the two PER overrides *dropped* — priority signal
`td_loss` → **`td_error`**, importance sampling **off → on** with β annealing 0.4 → 1.0 over 1M steps.
A clean one-knob-group test with batch 18 as the seed-matched control. **It is the clearest negative
since batch 12's deadlock: every comparable metric moved against it, 4 of 4 seeds, p=0.125** — the
floor at n=4. Paired against batch 18 truncated to a matched **2.004M** (`b19c` is the shortest arm):

| metric | b18 (`td_loss`, no IS) | b19 (standard PER) | delta | p |
|---|---|---|---|---|
| **`strong_eval_fraction`** (primary) | **31.60%** | **13.82%** | **-17.78 pp** | **0.125** (4/4) |
| `best_perfect30` | 85.52% | 63.27% | **-22.25 pp** | **0.125** (4/4) |
| mean perfect, back half | 68.86% | 48.88% | **-19.98 pp** | **0.125** (4/4) |
| peak trailing | 94.85 | 94.16 | **-0.69** | **0.125** (4/4) |
| **max drawdown** | 55.52 | **8.76** | **-46.76** | **0.125** (4/4) |
| steps to pf30 ≥ 40% | 299.5k | 324.7k (3 arms) | slower on 3/3 | — |

**`b19c` never reached pf30 ≥ 40% at all**, so that row has no fourth pair and is left unpooled rather
than filled in — the three seeds that did reach it were all slower (508k vs 460k, 244k vs 180k, 222k vs
216k).

**‡ The ceiling moved for the first time in nine batches, and it moved down.** Peak trailing has read
94.8-95.0 for every batch from 11 through 18 regardless of config; here it is 94.66 / 94.40 / 92.72 /
94.86, mean **94.16**, and lower on 4 of 4 seeds. A −0.69 mean is small in absolute terms, but it is
the first config to shift a quantity that eight consecutive batches could not budge — in the wrong
direction.

**‡ The drawdown result is real, large, and does not rescue the batch.** Max drawdown fell from
55.52 to **8.76**, 4/4 — by far the biggest movement in the table and the strongest anti-forgetting
result the project has. But the arms achieve it by sitting *lower*, not by holding a high level:
recent-30 is 43.3 / 51.3 / 14.3 / 56.0 against the control's much higher figures, and `sef` more than
halved. Flat curves at a worse level is what a full IS correction damping the replay signal looks
like. Since reducing catastrophic forgetting is a *means* in this project and not the goal, a −17.78 pp
primary is not paid for by a smaller drawdown — but the β anneal is now a candidate to pair with any
future change that raises the level.

**Full-length numbers, comparable to batch 18 above and to nothing below it.** These arms ran to
2.0-2.4M, so `sef` is inflated by run length relative to any shorter batch.

| seed | step | peak trailing | best-30 | `sef` | recent-30 | max drawdown |
|---|---|---|---|---|---|---|
| 4 | 2423k | **94.86** | **85.7%** | **40.2%** | **56.0%** | 12.84 |
| 1 | 2192k | 94.66 | 71.0% | 8.0% | 43.3% | **4.94** |
| 2 | 2116k | 94.40 | 66.7% | 4.6% | 51.3% | 10.20 |
| 3 | 2004k | 92.72 | 29.7% | **0.0%** | 14.3% | 7.04 |
| **mean** | | **94.16** | **63.3%** | **13.2%** | **41.2%** | **8.76** |

### b19d-stdperseed4 — standard PER, seed 4

2423k steps, peak trailing **94.86**, best-30 **85.7%** @1944k, `sef` **40.2%**, recent-30 56.0%,
max drawdown 12.84.

![b19d](charts/b19d-stdperseed4.png)

**The seed that escaped, and it is why the batch is a falsification rather than a catastrophe.** It is
level with its batch-18 control on every column (`sef` 40.2 against 41.6, peak 94.86 against 94.92),
so whatever standard PER costs, one seed in four does not pay it. Read against the other three, this
arm is the evidence that the damage is a *distribution* shifting left rather than a mechanism that
cannot work.

### b19a-stdperseed1 — standard PER, seed 1

2192k steps, peak trailing 94.66 @1299k, best-30 71.0% @1301k, `sef` 8.0%, recent-30 43.3%,
max drawdown **4.94** — the smallest in the batch.

![b19a](charts/b19a-stdperseed1.png)

Peaked early at 1.30M and then held a visibly flat, slightly declining band for 900k steps without
ever collapsing. Its control `b18a` reached `sef` 41.2% against this arm's 8.0% while suffering a 56.6
drawdown, so the pair is the whole batch in miniature: far steadier, far worse.

### b19b-stdperseed2 — standard PER, seed 2

2116k steps, peak trailing 94.40 @1889k, best-30 66.7% @1914k, `sef` 4.6%, recent-30 51.3%,
max drawdown 10.20.

![b19b](charts/b19b-stdperseed2.png)

**The slowest consolidator of the four:** it did not reach pf30 ≥ 60% until 1861k where its control
did so at 310k, a 1.55M gap. It was still improving when stopped — peak trailing and best-30 both land
in its final 250k — so this is an arm whose curve had not finished, and the one place the batch's
horizon is a genuine caveat rather than a formality.

### b19c-stdperseed3 — standard PER, seed 3

2004k steps, peak trailing 92.72 @1500k, best-30 **29.7%** @195k, `sef` **0.0%**, recent-30 14.3%,
max drawdown 7.04.

![b19c](charts/b19c-stdperseed3.png)

**Not one eval at ≥80% perfect in 2005 evals**, and its best 30-eval window came at **195k** — the
opening of the run. It never reached pf30 ≥ 40%, and peak trailing 92.72 is more than two points below
anything else in batches 18-19. It was never dead (`zero_since` null, trailing 88.7 at the end) and it
never collapsed — so it is **none of the four failure modes** in
[`failureModes.md`](failureModes.md): it learned to play well, plateaued just short of finishing, and
stayed there. The nearest thing on record is batch 12's deadlock, which was much worse. Seed 3 was
also batch 18's weakest arm, so the seed carries some of this — but `b18c` still managed `sef` 24.3%.

## Batch 18 — `TARGET_UPDATE_PERIOD=1000`, forking retained, stopped at 2.40-2.61M

Four seeds, batch 17's config exactly with the target period taken from 8 to **1000** — a clean
one-knob test, because forking stayed on in both. **The pre-registered primary metric moved, and it is
the strongest speed result this project has:** steps to pf30 >= 40% is **102k earlier, 4 of 4 seeds,
p=0.125** — the floor at n=4. Paired against batch 17 truncated to a matched 1.406M:

| metric | b17 (period 8) | b18 (period 1000) | delta | p |
|---|---|---|---|---|
| **steps to pf30 >= 40%** (primary) | 402k | **300k** | **-102.2k** | **0.125** (4/4) |
| **max drawdown** | 73.97 | **53.20** | **-20.76** | 0.375 |
| `strong_eval_fraction` | 17.56% | 24.29% | +6.73 pp | 0.625 |
| `best_perfect30` | 76.08% | 82.75% | +6.67 pp | 0.625 |
| mean perfect, back half | 57.83% | 64.02% | +6.19 pp | 0.750 |
| peak trailing | 94.56 | 94.69 | +0.14 | 0.875 |

**‡ The one prior data point predicted the opposite on drawdown and was wrong.** `b1b-tgt200` (batch 1,
period 200) got a *worse* drawdown than its baseline, 27.4 against 19.2, and that was recorded as the
risk to watch. At period 1000 the drawdown **improved by 20.8 points**, 3 of 4 seeds. The batch-1 hint
was right about "longer learns faster early" and wrong about the cost.

**Full-length numbers, which are not comparable to the batches below.** These arms ran to 2.4-2.6M
against batch 17's 1.4-1.6M, and `sef` is a fraction of an arm's *own* evals, so the figures below are
inflated by run length relative to any shorter batch. `b18d` at **47.9%** and `b18a` at **41.4%** are
nonetheless the two highest `sef` readings ever recorded here, against `b15a`'s 39.9% at 5.79M.

| seed | step | pf30 >= 40% at | b17 control | peak trailing | best-30 | `sef` (full) | max drawdown |
|---|---|---|---|---|---|---|---|
| 1 | 2612k | 460k | 560k | 94.92 | 88.0% | **41.4%** | 56.58 |
| 2 | 2401k | **180k** | 332k | 94.96 | 86.0% | 25.0% | 85.08 |
| 3 | 2510k | 342k | 386k | 94.94 | 84.3% | 24.3% | 56.14 |
| 4 | 2597k | 216k | 329k | 94.92 | **91.0%** | **47.9%** | **32.62** |
| **mean** | | **300k** | **402k** | **94.94** | **87.3%** | **34.7%** | **57.6** |

**The ceiling still has not moved — an eighth flat result.** Peak trailing reads 94.92 / 94.96 / 94.94 /
94.92, mean **94.94**, inside 0.4 of every batch from 11 on. Whatever the target period does, it is not
raising the maximum an arm reaches.

### b18d-tgt1000seed4 — period 1000, forking on, seed 4

![b18d](charts/b18d-tgt1000seed4.png)

Step 2.60M · peak trailing 94.92 (at 2346k) · **best 30-eval perfect 91.0%** (at 2369k) · `strong_eval_fraction` **47.9%** · recent-30 79.0%

**The steadiest arm on record.** 47.9% of its evals at >=80% perfect is the highest ever, and its max
drawdown of **32.62** is less than half the batch-17 mean — the red trace stays high instead of
collapsing and recovering. Reached pf30 >= 40% at 216k against its control's 329k.

### b18a-tgt1000seed1 — period 1000, forking on, seed 1

![b18a](charts/b18a-tgt1000seed1.png)

Step 2.61M · peak trailing 94.92 (at 1605k) · best 30-eval perfect 88.0% (at **2480k**) · `strong_eval_fraction` 41.4% · recent-30 64.0%

Second-highest `sef` ever, and still finding its best window at 2480k — near the end of the run. Its
seed-1 counterpart in batch 17 was the arm that made that batch a null, so **the pairing that looked
worst going in came out +40 pp on `sef`**. One seed, but worth noting against the temptation to read
seed identity as arm quality.

### b18c-tgt1000seed3 — period 1000, forking on, seed 3

![b18c](charts/b18c-tgt1000seed3.png)

Step 2.51M · peak trailing 94.94 (at 2161k) · best 30-eval perfect 84.3% (at 2168k) · `strong_eval_fraction` 24.3% · recent-30 **79.3%**

**Highest recent-30 of the batch at the moment it was stopped**, with peak and best window both inside
its last 350k — the arm most plausibly cut short.

### b18b-tgt1000seed2 — period 1000, forking on, seed 2

![b18b](charts/b18b-tgt1000seed2.png)

Step 2.40M · peak trailing 94.96 (at 1575k) · best 30-eval perfect 86.0% (at 1600k) · `strong_eval_fraction` 25.0% · recent-30 56.3%

**Fastest start on record: pf30 >= 40% at 180k.** Also the batch's worst drawdown at 85.08, and the
weakest recent-30 — it peaked around 1.6M and gave a lot back. The two facts together are the case for
reading this batch on speed rather than on stability.

## Batch 17 — forked endgame collection (`SNEK_FORK_*`), a null, stopped at 1.41-1.57M

Four seeds, batch 16's config exactly plus forking at length ≥ 85 — one variable, an exact control.
**A null**: `strong_eval_fraction` -1.67 pp at a matched 1.245M (p=0.875). Full write-up in
[`completedRuns.md`](completedRuns.md#batch-17--forked-endgame-collection-a-null-that-produced-the-project-record).

**These four charts are the clearest picture of the seed-variance problem in this document.** Same
config, adjacent seeds, and the batch spans `sef` **1.4% to 36.6%** — a 26x spread that is entirely
between-seed noise, since nothing differs but the seed. `b17b` and `b17a` sitting side by side is why
n=4 cannot resolve an effect below ~10 pp.

| seed | pf30 ≥ 40% at | b16 control | peak trailing | best-30 | `sef` @1.245M |
|---|---|---|---|---|---|
| 1 | 560k | 450k | 93.86 | 54.0% | **1.3%** |
| 2 | 332k | 400k | **95.00** | **92.7%** | **32.0%** |
| 3 | 386k | 379k | 94.78 | 82.0% | 9.9% |
| 4 | 329k | 465k | 94.60 | 75.7% | 18.6% |
| **mean** | **402k** | **424k** | **94.56** | **76.1%** | **15.4%** |

Peak trailing and best-30 above are **full-length**; `sef` is truncated to 1.245M because it is a
fraction of an arm's own evals.

### b17b-forkseed2 — forking on, disc 0.9975 + shield 0.8, seed 2

![b17b](charts/b17b-forkseed2.png)

Step 1.57M · **peak trailing 95.00** (at 1280k) · **best 30-eval perfect 92.7%** (at 1223k) · `strong_eval_fraction` **36.6%** · recent-30 71.0% · **best ckpt 95.17% @1190k over 600 fresh episodes**

**The best arm this project has produced, on the graph and in the close-out, and it is in a null
batch.** 92.7% best-30 is the highest that column has ever read — `b11b` managed 91.7% at 3.56M and
`b15a` 89.7% at 5.79M — and it got there at **1.22M**, a third of the steps. Peak trailing 95.00 ties
`b15a` for the highest on record.

**The close-out read 99/100 @1205k; re-measurement over 6,600 fresh episodes cut that to 92.4%.** The
record still moved — @1190k pools to **95.17% over 600** (CI 93.1-96.6) against the old 93.0-93.5% — but
by ~1.7 pp, not to 96%. A position-chosen grid over 1110-1270k reads **84.06%**, against **96.2%** for
the same region's selected rows, and one point in it (**1140k**) reads **12.0%**: the high-perfect
stretch on the right of this chart is **not** the plateau the close-out implied. `pooled_equal_effort`
**82.42%** is unaffected and is still a record. See
[`completedRuns.md`](completedRuns.md#-the-close-out-b17b-is-the-best-policy-this-project-has-measured-by-a-clear-margin).

### b17d-forkseed4 — forking on, disc 0.9975 + shield 0.8, seed 4

![b17d](charts/b17d-forkseed4.png)

Step 1.51M · peak trailing 94.60 (at 1007k) · best 30-eval perfect 75.7% (at **679k**) · `strong_eval_fraction` 20.7% · recent-30 68.3%

Fastest starter of the batch (pf30 ≥ 40% at 329k) and its best window came earliest of any arm here, at
679k — then 800k steps without beating it. A flat-after-early-peak shape, like `b16a`.

### b17c-forkseed3 — forking on, disc 0.9975 + shield 0.8, seed 3

![b17c](charts/b17c-forkseed3.png)

Step 1.52M · peak trailing 94.78 (at **1386k**) · best 30-eval perfect 82.0% (at **1388k**) · `strong_eval_fraction` 16.4% · recent-30 73.3%

**Still climbing when it was stopped** — both its peak and its best window land in its final 140k
steps, and its `sef` went 6.7% → 16.4% over the last 500k. Also carries the batch's worst drawdown
(85.96). The arm most likely to have been cut short.

### b17a-forkseed1 — forking on, disc 0.9975 + shield 0.8, seed 1

![b17a](charts/b17a-forkseed1.png)

Step 1.41M · peak trailing 93.86 (at 1144k) · best 30-eval perfect 54.0% (at 1166k) · `strong_eval_fraction` **1.4%** · recent-30 26.7%

**The arm that decides the batch, and the failure mode is visible in the chart**: it reaches 95/95
repeatedly and never holds it, so the red perfect-rate trace spikes and collapses rather than settling.
28.2% of its back-half evals fell to ≤10% perfect against 0.2-10.9% across batch 16. It is also the
only arm of the eight that **never reached ε ≤ 0.003** — the schedule is gated on sustained perfect
rate, so the oscillation kept its exploration high, which fed the oscillation. Its fork counters are
normal, and `b13a` failed the same way with forking off.

## Batch 16 — `FOOD_DISTANCE_REWARD=0`, the shaping term ablated, stopped at ~1.25M

Four seeds, batch 14's config exactly minus the food-distance shaping — one variable, an exact
control. **Stopped by hand at ~1.25M**, far short of batch 14/15's 4.2-5.8M, so read these charts for
their *left* halves: the level metrics below (`strong_eval_fraction`, best-30) sit at a much shorter
horizon than the batches beneath and are **not comparable to them at face value** — the comparison that
is valid truncates batch 14 to the same 1.25M, and doing that makes this **the first non-null in six
batches**. Closed out at `EVAL_MIN_ACHIEVABLE=95`; full write-up in
[`completedRuns.md`](completedRuns.md#batch-16--the-food-distance-shaping-ablated-the-first-non-null-in-six-batches).

**The ceiling did not move — a sixth flat result.** Peak trailing across the four arms reads
94.98 / 94.82 / 94.68 / 94.36, mean **94.71**, inside 0.3 pp of the 94.8-95.0 band the previous five
batches sit in. Removing the shaping neither raised nor lowered the peak.

**What did change is the seed spread, and it is the one thing worth watching.** Steps to pf30 ≥ 40%
came out 379-465k (sd 41k) against batch 14's 227-639k (sd 189k) — a **4.6x tighter** spread on the
metric whose seed variance is this project's binding constraint. Flat mean (-5k, p=0.875), collapsed
variance. Four samples cannot establish that; it is a hypothesis for the wider design, not a finding.

| seed | pf30 ≥ 40% at | b14 control | peak trailing | best-30 | `sef` (short horizon) |
|---|---|---|---|---|---|
| 1 | 450k | 639k | 94.82 | 87.0% | 20.6% |
| 2 | 400k | 227k | **94.98** | 85.0% | **30.7%** |
| 3 | 379k | 530k | 94.36 | 72.7% | 10.6% |
| 4 | 465k | 320k | 94.68 | 73.0% | 7.2% |
| **mean** | **424k** | **429k** | **94.71** | **79.4%** | **17.3%** |

### b16b-noshapeseed2 — shaping off, disc 0.9975 + shield 0.8, seed 2

![b16b](charts/b16b-noshapeseed2.png)

Step 1.26M · **peak trailing 94.98** (at 816k) · best 30-eval perfect 85.0% (at 919k) · `strong_eval_fraction` **30.7%** · recent-30 79.0%

**Strongest of the batch and the flattest** — peaked at 816k and held 79% recent perfect at stop. At
this ~1.25M horizon its `sef` matches batch 14's best arms, but the horizons are not comparable; the
close-out is what settles level.

### b16a-noshapeseed1 — shaping off, disc 0.9975 + shield 0.8, seed 1

![b16a](charts/b16a-noshapeseed1.png)

Step 1.25M · peak trailing 94.82 (at 837k) · **best 30-eval perfect 87.0%** (at 850k) · `strong_eval_fraction` 20.6% · recent-30 77.0%

Highest best-30 window of the batch (87.0%), though both peak and best window land early (~840k) and
it has plateaued since. Solid but unremarkable.

### b16c-noshapeseed3 — shaping off, disc 0.9975 + shield 0.8, seed 3

![b16c](charts/b16c-noshapeseed3.png)

Step 1.26M · peak trailing 94.36 (at **1198k**) · best 30-eval perfect 72.7% (at 1221k) · `strong_eval_fraction` 10.6% · recent-30 67.0%

**The latest-peaking arm** — best window at 1221k, at the very end — so unlike its siblings it may not
have plateaued when stopped. Recovered from a mid-run dip to ~86% trailing back to 93.7%, which is
what dragged its `sef` down despite a normal peak.

### b16d-noshapeseed4 — shaping off, disc 0.9975 + shield 0.8, seed 4

![b16d](charts/b16d-noshapeseed4.png)

Step 1.26M · peak trailing 94.68 (at 946k) · best 30-eval perfect 73.0% (at 1032k) · `strong_eval_fraction` **7.2%** · recent-30 **55.7%**

**Weakest of the batch** on `sef` and recent perfect (55.7%), though its peak trailing sits mid-pack —
the gap is consistency, not ceiling. The same seed-4 slot that was the speed outlier in both batch 14
and batch 16.

## Batch 15 — `N_STEP_UPDATE=3`, falsified on speed, and the longest arms on record

Four seeds to 5.5-5.8M in 15.9 h — 1.3M further than any previous batch. n-step's predicted effect
was faster credit propagation, and **the pre-registered metric moved the wrong way**: steps to
pf30 ≥ 40% came out **128k later** than batch 14's control, 3 of 4 seeds slower (p=0.250). The evals
agree it is a null — best checkpoint +0.05 pp (p=1.000), `pooled_equal_effort` +2.24 pp (p=0.625).
Full write-up:
[`completedRuns.md`](completedRuns.md#batch-15--n_step_update3-falsified-on-speed-null-on-level-and-a-97100-that-is-really-93).

**Read these four charts for their right-hand halves, which no earlier batch has.** Two arms were
still gaining in their final 500k band at 5.5-6.0M, and `b15d`'s peak trailing score is at its
second-to-last eval. The long-standing "arms peak by ~3.4M" reading is now falsified three batches
running.

**What has *not* moved is the ceiling.** Peak trailing score across the five batches on this vector
reads 94.92 / 94.80 / 94.90 / 95.00 (b11, b13, b14, b15) — flat inside 0.2 points, with `b15a`'s
95.00 beating `b11b` by 0.08 after 2.2M more steps. Whatever the extra horizon buys, it is not a
higher peak.

| seed | pf30 ≥ 40% at | b14 control | best ckpt | eq-effort | `sef` (full) |
|---|---|---|---|---|---|
| 1 | 620k | 639k | 95% | 77.7% | **39.9%** |
| 2 | **524k** | 227k | **97%** | **79.7%** | 39.0% |
| 3 | **707k** | 530k | 86% *trunc* | 66.4% | 9.4% |
| 4 | 378k | 320k | 91% | 73.5% | 33.8% |
| **mean** | **557k** | **429k** | **92.3%** | **74.3%** | **30.5%** |

### b15a-nstep3seed1 — n=3, disc 0.995 + shield 0.8, seed 1

![b15a](charts/b15a-nstep3seed1.png)

Step 5.79M · **peak trailing 95.00** (at 4716k) · **best 30-eval perfect 89.7%** (at 4705k) · `strong_eval_fraction` **39.9%** · final band 80.6%

**The best arm this project has produced on two measures** — the highest peak trailing score on the
current vector (95.00, past `b11b`'s 94.92) and the highest `strong_eval_fraction` on record (39.9%,
past `b14d`'s 39.3%). It is also **still gaining in its final band**, 80.6% mean perfect over
5.5-6.0M against an 80.4% previous best, so it was stopped mid-climb.

Both records come with the run-length caveat: `strong_eval_fraction` is a share of the arm's own
evals, and this arm spent 1.3M more steps than `b14d` playing at 70-80% perfect. At equal effort the
batch advantage is +4.05 pp at p=0.625. What the chart does show unambiguously is a **shape no
earlier arm has** — a slow, monotone climb that had not turned over by 5.8M.

### b15b-nstep3seed2 — n=3, disc 0.995 + shield 0.8, seed 2

![b15b](charts/b15b-nstep3seed2.png)

Step 5.75M · peak trailing 94.94 (at 4524k) · best 30-eval perfect 89.3% (at 4595k) · `strong_eval_fraction` 39.0% · final band 62.3%

**The best arm this project has measured**, and the arm that cost n=3 the primary. Its
`pooled_equal_effort` is **79.68%**, past `b14d`'s 77.55%, and its best checkpoint read **97/100** —
the highest selected measurement on record. Yet it reached pf30 ≥ 40% at **524k against `b14b`'s
227k**, the +297k that drives the batch mean. The slowest starter in the comparison finished on top,
which is why "steps to a milestone" and "final level" are separate questions.

**The 97/100 is really ~93%.** Re-measured over 200 fresh episodes it read 182/200 = 91.0%, pooling
to **279/300 = 93.0%** (CI 89.5-95.4) — the same haircut `b14a`'s 96/100 took. This arm also holds 8
of the batch's 9 rows at ≥95%, but its 94 full-length rows have **mean 90.7%**, and a population
centred at 90% throws ~5.4 readings of ≥95 per 94 rows by noise alone. The high count is a tail, not
a cluster of near-perfect policies.

Held 78% mean perfect across **four** consecutive bands from 3.0M to 5.0M (78.1 / 78.4 / 78.2 /
78.5), then dropped to 62.3% in its final band. Past peak when stopped.

### b15d-nstep3seed4 — n=3, disc 0.995 + shield 0.8, seed 4

![b15d](charts/b15d-nstep3seed4.png)

Step 5.81M · peak trailing 94.70 (at **5799k**) · best 30-eval perfect 86.3% (at 3687k) · `strong_eval_fraction` 33.8% · final band **75.8%**

**Its peak trailing score is at 5799k — the second-to-last eval it ever ran.** Also still gaining in
its final band (75.8% against a 72.8% previous best). This is the single clearest piece of evidence
in the project that stopping at a round number truncates arms: nothing about this curve suggests it
was finished.

### b15c-nstep3seed3 — n=3, disc 0.995 + shield 0.8, seed 3

![b15c](charts/b15c-nstep3seed3.png)

Step 5.46M · peak trailing 94.38 (at 3808k) · best 30-eval perfect 75.7% (at 2046k) · `strong_eval_fraction` **9.4%** · final band 53.7%

Weakest of the batch by a wide margin and the reason the batch spread is -9.7 to +22.9 pp. Its best
30-eval window is at 2046k, earlier than any sibling, and it never exceeded 60% mean perfect in any
band. **Same config as `b15a`'s 39.9%, adjacent seed, 30 pp apart on the primary metric** — which is
the seed-variance problem that has made five consecutive batches unreadable, stated as compactly as
it can be.

**The first arm in five batches with no full-length eval row at all.** Every 100-episode target was
abandoned by the 90% gate, so its best checkpoint is a truncated 69/80 and
`best_full_length_row`'s half-depth fallback ran in production for the first time. At the 95% gate
this becomes the normal case rather than the exception.

---
