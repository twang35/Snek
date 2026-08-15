# Charts

Progress graphs for the most recent batches — **21 through 26**, a cap of six, newest first. Per-arm
numbers live in
[`completedRuns.md`](completedRuns.md); this file is images plus a short reading of each. A batch appears
here **while it is still running**, with training-only numbers, not just once it has closed.

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

**Three PNGs in `charts/` are not arm charts and will always appear in that list** —
`champion-vs-mediocre`, `drawdown-b23b-vs-b18` and `per-b18-vs-b20-priorities` are diagnostic figures
referenced from [`findings.md`](findings.md) and [`perDiagnostics/`](perDiagnostics/README.md), not
training graphs. Anything *else* the check prints is a real gap.

## Batch 26 — FC `100,100` under IS-off (`SNEK_IS_WEIGHTS=0`), `td_error`, seeds 1-4 — *close-outs done, HOF-500 running (empty)*

The third shape in the width follow-up: a shallow **two-layer `100,100`** net, after b24 (`320`) and b25
(`200,100,100`) both lifted consolidation. It asks whether a shallower shape still gets the gain, or whether
it needs the depth/capacity those two had. Seed-matched control is b22 (`50,100,50`, IS off). Trained on the
desktop.

**All four trained to the 3M cap and closed out (gate 95).** The shallow shape **does not carry the lift**:
close-out pooled mean **79.2** is only **+3.5 over the b22 control's 75.7** — against b24's +12.2 (`320`) and
b25's +10.3 (`200,100,100`). Three seeds learned well (`sef` 44-58); `b26d` is a weak seed (`sef` 13.8,
pooled 69.6) but never died. **No arm produced a ≥98%/100 checkpoint** — the best full-length reads are
`b26b`/`b26c` at 97.0%/100 — so the auto-HOF-500 (gate 98, running now) selects nothing and lands empty; the
record stays b24's. This places the width follow-up cleanly: the consolidation gain needs capacity (`320`
one wide layer, or `200,100,100` deep), and a shallow `100,100` is too small to buy more than a marginal
lift. Sorted by close-out pooled.

| arm | peak trail | best-30 | `sef` (3M) | close-out pooled | best full-length |
|---|---|---|---|---|---|
| `b26b` | 95.00 | **93.7%** @1982k | **58.0%** | **83.8** | 97.0% @1948k |
| `b26c` | 94.96 | 92.0% @2231k | 52.1% | 83.2 | 97.0% @1969k |
| `b26a` | 94.92 | 88.0% @2349k | 44.6% | 80.0 | 95.0% @2904k |
| `b26d` | 94.84 | 79.7% @1073k | 13.8% | 69.6 | none ≥95% (all ab.) |
| **mean — b26 fc100,100 IS-off** | **94.93** | **88.4%** | **42.1%** | **79.2** | 0 of 4 held ≥98%/100 |
| **mean — b22 fc50,100,50 IS-off (control)** | 94.88 | 86.2% | 30.5% | 75.7 | — |

![b26b](charts/b26b-fc100x100noisseed2.png)
**b26b-fc100x100noisseed2**

![b26c](charts/b26c-fc100x100noisseed3.png)
**b26c-fc100x100noisseed3**

![b26a](charts/b26a-fc100x100noisseed1.png)
**b26a-fc100x100noisseed1**

![b26d](charts/b26d-fc100x100noisseed4.png)
**b26d-fc100x100noisseed4**

## Batch 25 — FC `200,100,100` under IS-off (`SNEK_IS_WEIGHTS=0`), `td_error`, seeds 1-4

b22's exact IS-off config with a 3-layer `200,100,100` net (~1.6× the control's params) — the second
shape in the width follow-up after b24's `320` result. It asks whether b24's consolidation lift is width
itself or just more parameters, and whether it survives at a shape other than one wide layer. Seed-matched
control is b22 (`50,100,50`, IS off). Trained on the desktop.

**Fully evaluated — and the first auto-HOF chain ran end to end (training → close-out → HOF-500).** The
close-out (gate 95) pools a mean **86.0** — **+10.3 over the b22 control's 75.7, within 1.9 of b24's
87.9** — so the consolidation lift **replicates at a 3-layer `200,100,100` shape**, which points at
capacity rather than width itself. Peak is unmoved at 95.0. **But the HOF-500 (gate 98) held nothing: every
arm's ≥98%/100 candidates were abandoned, none reaching 98% over 500** — the /100 highs inflated exactly as
b24's did. The strongest was `b25b` @911k, still 97.2% when gate-98 stopped it at 392 episodes; that is a
plausible ~97%/500 holder the folder's gate-97 standard would have run to completion, so it needs a hand
re-measure before any hall claim. **No b25 checkpoint enters the folder on the auto run.** Sorted by
close-out pooled.

| arm | peak trail | best-30 | `sef` | close-out pooled | HOF-500 (gate 98) |
|---|---|---|---|---|---|
| `b25c` | 95.00 | 93.7% | **66.9%** | **87.2** | none ≥98% (best 95.3% @827k, ab.) |
| `b25d` | 95.00 | 94.3% | 62.2% | 85.9 | none ≥98% (best 96.4% @2431k, ab.) |
| `b25a` | 95.00 | 93.7% | 63.2% | 85.6 | none ≥98% (best 92.7% @802k, ab.) |
| `b25b` | 95.00 | **95.3%** | 62.7% | 85.5 | none ≥98% (best **97.2%** @911k, ab.) |
| **mean — b25 fc200,100,100 IS-off** | **95.00** | **94.3%** | **63.8%** | **86.0** | 0 of 4 held ≥98%/500 |
| **mean — b22 fc50,100,50 IS-off (control)** | 94.88 | 86.2% | 30.5% | 75.7 | — |

![b25c](charts/b25c-fc200x100x100noisseed3-r2.png)
**b25c-fc200x100x100noisseed3-r2**

![b25a](charts/b25a-fc200x100x100noisseed1-r2.png)
**b25a-fc200x100x100noisseed1-r2**

![b25b](charts/b25b-fc200x100x100noisseed2-r2.png)
**b25b-fc200x100x100noisseed2-r2**

![b25d](charts/b25d-fc200x100x100noisseed4-r2.png)
**b25d-fc200x100x100noisseed4-r2**

## Batch 24 — FC width `320` under IS-off (`SNEK_IS_WEIGHTS=0`), `td_error`, seeds 1-4

Batch 22's exact IS-off config with the network widened to a single `320` layer (batch 20's `320`
shape) — width is the only change. It asks the question batch 20 could not answer under the β→1.0
control: **does width matter once the prioritisation is fixed at IS-off?** The seed-matched control is
b22 (`50,100,50`, IS off). Trained on the desktop.

**Width raises consolidation under IS-off, and the batch set a new record.** All four peak at **95.00**, so
width does not move the ceiling. But the close-out pools **87.9** (eq-effort, gate 95): **+12.2 over the b22
control's 75.7, and higher on every seed** — above every prior gate-95 arm, the b18b record's 78.5 included.
This is the project's first sign that width and prioritisation interact: width paid nothing under β→1.0
(batch 20), and it pays here under IS-off.

**The HOF-500 re-measured all 199 ≥97%/100 checkpoints; 9 held ≥97%/500 and the batch took the record.** The
new record is **`b24d` @1342k, 98.0%/500** (490/500, CI [96.4,98.9]), edging `b18b` @1588k (97.6%/700) and
tied by `b24b` @2860k (98.0%/500). The /100 rows were badly inflated — `b24a`'s two 100%/100 highs produced
**0 survivors** at 500 episodes (b23b's 97%/100 → 92.4%/500 was the same pattern) — so read the `best HOF-500`
column, not `best /100`.

All at the 3M cap, sorted by close-out pooled. Close-out and HOF-500 ran on the desktop (gate 95 / gate 97,
`EVAL_WORKERS=4`).

| arm | peak trail | best-30 | `sef` | close-out pooled | best /100 | best HOF-500 |
|---|---|---|---|---|---|---|
| `b24a` | 95.00 | 95.3% | 60.5% | **89.03** | **100.0%** @1633k | — (0 of 43 held) |
| `b24b` | 95.00 | **96.7%** | **73.2%** | 88.84 | 99.0% @1031k | **98.0%** @2860k |
| `b24c` | 95.00 | 96.0% | 67.4% | 87.68 | **100.0%** @2126k | 97.4% @2982k |
| `b24d` | 95.00 | 96.7% | 62.9% | 85.97 | 99.0% @1292k | **98.0%** @1342k ← **record** |
| **mean — b24 fc320 IS-off** | **95.00** | **96.2%** | **66.0%** | **87.9** | — | — |
| **mean — b22 fc50,100,50 IS-off** | 94.88 | 86.2% | 30.5% | 75.7% | — | — |

![b24b](charts/b24b-fc320noisseed2.png)
**b24b-fc320noisseed2**

![b24d](charts/b24d-fc320noisseed4.png)
**b24d-fc320noisseed4**

![b24c](charts/b24c-fc320noisseed3.png)
**b24c-fc320noisseed3**

![b24a](charts/b24a-fc320noisseed1.png)
**b24a-fc320noisseed1**

## Batch 23 — β annealed 0→0.1, `td_error` priority, fc 50,100,50

One step further down the IS-β ladder than batch 21: β annealed from **0 to 0.1** over 300k
(`SNEK_IS_BETA=0`, `SNEK_IS_BETA_FINAL=0.1`), so the update keeps **α·(1−β)=0.54** of the priority
signal at the target — between b21's 0.30 (β→0.5) and the full 0.6 with IS off. Otherwise batch 20's
control. It asks whether dialing β toward 0 approaches the no-IS behaviour smoothly.

**β→0.1 lands at the no-IS consolidation level, and the close-out confirms it.** Training graph: mean
best-30 **85.8**, `sef` **33.1** (n=4, `sef` spread 23.5-56.0). The desktop close-out then reads **pooled
75.7** (eq-effort, gate 95) — **+20.7 over the control, +11.4 over b21, and higher on all four seeds than
either** (sign-test p=0.0625 each, the n=4 floor) — closing most of the gap to b18's no-IS ~78.8 (ESS/N
0.21, a different base). Both metrics climb monotonically down the β ladder: pooled control 55.0 → b21
64.3 → **b23 75.7** → b18 ~78.8; `sef` 11.2 → 14.3 → **33.1** → 34.6. Peak trailing **94.90** is flat with
every batch since 11 — ceiling unmoved, only consolidation differs. `b23b` holds a dense strong region on
the graph — **five full-length checkpoints ≥95/100 around 777k, best 97/100** — and looked like a
hall-of-fame candidate, but the re-measurement protocol falsified it: at **500 fresh episodes the
close-out-selected @777k reads 92.4%**, the *worst* of its own cluster (textbook selection bias) and well
below the 97.6% record. No b23 checkpoint enters the hall of fame.

All at the 3M cap, sorted by best-30. Close-out under gate 95, `EVAL_WORKERS=4`.

| arm | peak trail | best-30 | `sef` | close-out pooled | best row |
|---|---|---|---|---|---|
| `b23b` | **95.00** | **91.0%** | **56.0%** | **82.1%** | **97.0% @777k (n=100)** |
| `b23a` | **95.00** | 87.0% | 23.5% | 77.2% | 80.0% @1039k (n=20) |
| `b23c` | 94.78 | 83.0% | 24.0% | 71.5% | 75.0% @1393k (n=20) |
| `b23d` | 94.82 | 82.3% | 28.7% | 72.1% | 75.0% @603k (n=20) |
| **mean — b23 β→0.1** | 94.90 | 85.8% | 33.1% | **75.7%** | — |
| **mean — b21 β→0.5** | 94.69 | 74.1% | 14.3% | 64.3% | — |
| **mean — control β→1.0** | 94.44 | 64.0% | 11.2% | 55.0% | — |
| **mean — b18 no IS** | — | 87.3% | 34.6% | ~78.8% | — |

**Only `b23b` cleared gate 95 at full length** (12 rows, top-3 96.3%); the other three have no full-length
row, so their `best row` is a 20-episode screen (a bound), while `pooled` is exact.

![b23b](charts/b23b-beta01seed2.png)
**b23b-beta01seed2**

![b23a](charts/b23a-beta01seed1.png)
**b23a-beta01seed1**

![b23c](charts/b23c-beta01seed3.png)
**b23c-beta01seed3**

![b23d](charts/b23d-beta01seed4.png)
**b23d-beta01seed4**

## Batch 22 — IS off (`SNEK_IS_WEIGHTS=0`), `td_error` priority, fc 50,100,50

The bottom of the IS-β ladder short of the no-IS extreme: importance sampling **off**, so the gradient
carries the full `|δ|^0.6` prioritisation (**ESS/N ≈0.38**). Same config as b21/b23 — IS is the only
knob. Trained on the desktop, closed out under gate 95.

**Done at 3M — and IS-off is a dead heat with β→0.1.** Close-out **pooled 75.7** (eq-effort, gate 95),
identical to b23's 75.7, with best-30 86.2 vs 85.8 and `sef` 30.5 vs 33.1 — indistinguishable. So the
consolidation gain **saturates by β→0.1**: taking the last of the IS correction off buys nothing more.
The ladder is control 55.0 → b21 64.3 → **{b23 β→0.1, b22 IS-off} 75.7** → b18 ~78.8 (a different base).
Peak trailing 94.88, ceiling unmoved. `b22a` reached a 96/100 full-length checkpoint @1075k and `b22d` a
95/100 @2275k — below the 97.6% record and, by the shrink pattern, not record candidates.

All at the 3M cap, sorted by best-30. Close-out under gate 95, `EVAL_WORKERS=4`.

| arm | peak trail | best-30 | `sef` | close-out pooled | best row |
|---|---|---|---|---|---|
| `b22d` | **94.94** | **88.3%** | 25.6% | 76.3% | 95.0% @2275k (n=100) |
| `b22a` | 94.88 | 87.0% | **46.3%** | **78.9%** | **96.0% @1075k (n=100)** |
| `b22c` | 94.92 | 86.7% | 27.5% | 75.0% | 80.0% @866k (n=20) |
| `b22b` | 94.78 | 83.0% | 22.5% | 72.5% | — (no full-length row) |
| **mean — b22 IS off** | 94.88 | 86.2% | 30.5% | **75.7%** | — |
| **mean — b23 β→0.1** | 94.90 | 85.8% | 33.1% | 75.7% | — |
| **mean — b18 no IS** | — | 87.3% | 34.6% | ~78.8% | — |

**`b22a` and `b22d` cleared gate 95 at full length; `b22b`/`b22c` did not** (best rows are screens or
bounds), so their `pooled` is the figure to read.

![b22d](charts/b22d-noisseed4.png)
**b22d-noisseed4**

![b22a](charts/b22a-noisseed1.png)
**b22a-noisseed1**

![b22c](charts/b22c-noisseed3.png)
**b22c-noisseed3**

![b22b](charts/b22b-noisseed2.png)
**b22b-noisseed2**

## Batch 21 — partial IS (β 0.4→0.5), `td_error` priority, fc 50,100,50

One knob off batch 20's control: IS β annealed to **0.5** not 1.0, so the gradient keeps `|δ|^{0.30}` of
prioritisation (**ESS/N ≈0.86** on the four end-of-run buffers) instead of the near-uniform ≈1.0 that β=1
gives. The aim: keep IS's anti-forgetting effect while letting end-game transitions pull harder.

**Verdict: better than the β→1.0 control, well short of no-IS.** Best-30 74.1 vs the control's 64.0 (+10 pp)
and `sef` 14.3 vs 11.2, 3 of 4 seeds favouring β→0.5 — directional but n=4 cannot resolve it (p 0.375-0.625).
The desktop close-out confirms the direction: **pooled (eq-effort, gate 95) 64.3 vs 55.0 (+9.3), 3 of 4
seeds.** Still far below batch 18 (no IS: best-30 87.3, `sef` 34.6, ESS/N 0.21). Peak trailing 94.69 is flat
with every batch since 11, so the ceiling is unmoved — only consolidation differs. **Batch 22** (`td_error`,
IS off, ESS/N ≈0.38) tests the next point down. Trained on the laptop, close-out on the desktop.

All at 3M, sorted by best-30. Close-out under gate 95, `EVAL_WORKERS=4`.

| arm | peak trail | best-30 | `sef` | close-out pooled | best row |
|---|---|---|---|---|---|
| `b21b` | 94.78 | **80.7%** | **21.6%** | **68.8%** | 89.3% @2525k (n=56) |
| `b21d` | **94.80** | 76.3% | 16.5% | 66.0% | 84.1% @2392k (n=44) |
| `b21c` | 94.64 | 70.7% | 11.1% | 62.8% | 82.5% @2203k (n=40) |
| `b21a` | 94.52 | 68.7% | 8.2% | 59.5% | 81.6% @1263k (n=38) |
| **mean — b21 β→0.5** | 94.69 | 74.1% | 14.3% | 64.3% | — |
| **mean — control β→1.0** | 94.44 | 64.0% | 11.2% | 55.0% | — |

**No full-length rows** (deepest 38-56 of 100 under gate 95), so `best row` is a bound and `pooled` is exact.

![b21b](charts/b21b-beta05seed2.png)
**b21b-beta05seed2**

![b21d](charts/b21d-beta05seed4.png)
**b21d-beta05seed4**

![b21c](charts/b21c-beta05seed3.png)
**b21c-beta05seed3**

![b21a](charts/b21a-beta05seed1.png)
**b21a-beta05seed1**
