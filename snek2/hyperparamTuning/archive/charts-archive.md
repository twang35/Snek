# Charts — archived batch sections

Chart sections retired from [`../charts.md`](../charts.md), newest-retired first. **History only —
do not read into context during normal work**, same as everything else in this folder.

`charts.md` keeps the **six most recent batches**. When a batch is stopped and added there, the
oldest section moves here, so the live file stays short enough to actually read. The PNGs are *not*
moved: every image stays in `../charts/`, so the captions here still render.

| retired | batch | why it went |
|---|---|---|
| 2026-08-15 | 25 | batch 32 (Adam `epsilon` on C51) landed; batch 25 became the seventh-newest |
| 2026-08-15 | 24 | batch 31 (the first C51 batch) landed; batch 24 became the seventh-newest |
| 2026-08-15 | 23 | batch 28 landed; batch 23 became the seventh-newest |
| 2026-08-14 | 22 | batch 30 launched on the laptop; batch 22 became the seventh-newest |
| 2026-08-14 | 21 | batch 27 launched; batch 21 became the seventh-newest |
| 2026-08-14 | 20 | batch 26 landed; batch 20 became the seventh-newest |
| 2026-08-12 | 19 | batch 20's `100,50,50` closed the nine-shape sweep; batch 19 became the seventh-newest |
| 2026-08-12 | 18 | retired with 19 in the same pass — `charts.md` was holding seven batch sections |
| 2026-08-12 | 17 | batch 22 landed; batch 17 became the seventh-newest |
| 2026-08-12 | 16 | batch 23 landed; batch 16 became the seventh-newest |
| 2026-08-11 | 15 | batch 21 landed; batch 15 became the seventh-newest |
| 2026-08-09 | 14 | batch 20 landed; batch 14 became the seventh-newest |
| 2026-08-08 | 13 | batch 19 landed; batch 13 became the seventh-newest |
| 2026-08-08 | 12 | batch 18 landed; batch 12 became the seventh-newest |

---

## Batch 25 — FC `200,100,100` under IS-off (`SNEK_IS_WEIGHTS=0`), `td_error`, seeds 1-4

b22's exact IS-off config with a 3-layer `200,100,100` net (36,804 params, **3.09×** the control) — the second
shape in the width follow-up after b24's `320` result. It asks whether b24's consolidation lift is width
itself or just more parameters, and whether it survives at a shape other than one wide layer. Seed-matched
control is b22 (`50,100,50`, IS off). Trained on the desktop.

**Fully evaluated — and the first auto-HOF chain ran end to end (training → close-out → HOF-500).** The
close-out (gate 95) pools a mean **86.0** — **+10.3 over the b22 control's 75.7, within 1.9 of b24's
87.9** — so the consolidation lift **replicates at a 3-layer `200,100,100` shape**. (This section first
read that as "capacity rather than width"; **b26 falsified it** — `100,100` carries more parameters than
b24's `320` and gets +3.5. The shapes order by widest layer, and `200,100,100` costs 3.09× the control's
parameters to land *below* `320`'s 0.94×.) Peak is unmoved at 95.0. **But the HOF-500 (gate 98) held nothing: every
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

![b25c](../charts/b25c-fc200x100x100noisseed3-r2.png)
**b25c-fc200x100x100noisseed3-r2**

![b25a](../charts/b25a-fc200x100x100noisseed1-r2.png)
**b25a-fc200x100x100noisseed1-r2**

![b25b](../charts/b25b-fc200x100x100noisseed2-r2.png)
**b25b-fc200x100x100noisseed2-r2**

![b25d](../charts/b25d-fc200x100x100noisseed4-r2.png)
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

![b24b](../charts/b24b-fc320noisseed2.png)
**b24b-fc320noisseed2**

![b24d](../charts/b24d-fc320noisseed4.png)
**b24d-fc320noisseed4**

![b24c](../charts/b24c-fc320noisseed3.png)
**b24c-fc320noisseed3**

![b24a](../charts/b24a-fc320noisseed1.png)
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

![b23b](../charts/b23b-beta01seed2.png)
**b23b-beta01seed2**

![b23a](../charts/b23a-beta01seed1.png)
**b23a-beta01seed1**

![b23c](../charts/b23c-beta01seed3.png)
**b23c-beta01seed3**

![b23d](../charts/b23d-beta01seed4.png)
**b23d-beta01seed4**

---

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

![b22d](../charts/b22d-noisseed4.png)
**b22d-noisseed4**

![b22a](../charts/b22a-noisseed1.png)
**b22a-noisseed1**

![b22c](../charts/b22c-noisseed3.png)
**b22c-noisseed3**

![b22b](../charts/b22b-noisseed2.png)
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

![b21b](../charts/b21b-beta05seed2.png)
**b21b-beta05seed2**

![b21d](../charts/b21d-beta05seed4.png)
**b21d-beta05seed4**

![b21c](../charts/b21c-beta05seed3.png)
**b21c-beta05seed3**

![b21a](../charts/b21a-beta05seed1.png)
**b21a-beta05seed1**

## Batch 20 — matched-capacity reshuffle (`100,50,50`) against the control, both to 3M

The last of the nine shapes: 10,853 params, **0.92×** the control, depth 3 — the same depth and
effectively the same capacity, with the width moved to the first layer. It was pre-registered as an
expected null, and it closes the sweep.

**Verdict: null on the ceiling, as designed.** Peak trailing **94.16** vs the control's 94.44 (−0.28,
**2 of 4** seeds, p=0.750) — inside the flat band on three seeds; only `b20aj` at 93.56 falls below it.
Drawdown **6.35** vs 5.42 is a match, so the base's anti-forgetting property is intact. Every
consolidation column reads *lower* (`sef` 1.9 vs 11.2, best-30 51.2 vs 64.0, pooled 46.3 vs 55.0), but
at **1 of 4** seeds and p=0.375 that is the same seed-driven noise the batch has produced in both
directions — here it is `b20aj` alone (`sef` 0.1%, pooled 30.8 against its control's 62.7).

**‡ The finding is what this shape does to the *rest* of the sweep.** `100,50,50` (10,853 params) and
`320` (10,883 params) differ by 0.3% in capacity, and they land on **pooled 46.3 and 65.1** — an 18.8 pp
spread that **brackets the control from both sides** at effectively identical parameter count. Whatever
the consolidation columns are measuring across batch 20's shapes, it is not capacity: two nets the same
size disagree by more than any shape disagrees with the control. That retroactively settles `320`'s
apparent +10.1 pooled edge as noise, which is where the ‡ downgrades in this batch had already landed it
by the per-seed test.

All at the 3M cap, sorted by best-30. Close-out under gate 95, `EVAL_WORKERS=4`,
`SNEK_FC_LAYERS=100,50,50`.

| arm | peak trail | best-30 | `sef` | max drawdown | close-out pooled | best row |
|---|---|---|---|---|---|---|
| `b20ai` | 94.34 | **58.7%** | **4.5%** | 5.58 | **52.1%** | **72.0% @2647k (n=25)** |
| `b20al` | **94.36** | 57.0% | 1.9% | 5.74 | 51.8% | 64.3% @2836k (n=28) |
| `b20ak` | **94.36** | 53.3% | 1.0% | **5.32** | 50.8% | 68.0% @2692k (n=25) |
| `b20aj` | 93.56 | 36.0% | 0.1% | 8.74 | 30.8% | 52.0% @1026k (n=25) |
| **mean — `100,50,50`** | 94.16 | 51.2% | 1.9% | 6.35 | 46.3% | — |
| **mean — control** | 94.44 | 64.0% | 11.2% | 5.42 | 55.0% | — |
| **mean — `320`** (same params) | 94.67 | 74.7% | 16.5% | 7.44 | 65.1% | — |

**No full-length rows** (deepest 25-28 of 100 under gate 95), so `best row` is a bound and `pooled` is
the exact column. Nothing near 95%, so no hall-of-fame candidate.

![b20ai](../charts/b20ai-fc100x50x50seed1.png)
**b20ai-fc100x50x50seed1**

![b20al](../charts/b20al-fc100x50x50seed4.png)
**b20al-fc100x50x50seed4**

![b20ak](../charts/b20ak-fc100x50x50seed3.png)
**b20ak-fc100x50x50seed3**

![b20aj](../charts/b20aj-fc100x50x50seed2.png)
**b20aj-fc100x50x50seed2**

## Batch 20 wave 3 — small-capacity (`25,50,25`) against the control, both to 3M

The smallest net in the sweep: 3,428 params, **0.29×** the control, depth 3. The pre-registered question
was whether capacity is binding at all — if under a third of the parameters holds the ceiling, it is not.

**Verdict: it does not hold the ceiling — capacity finally binds.** Peak trailing **93.75** vs the
control's 94.44, the first shape in batch 20 to move peak at all: all four seeds lower, two of them
(`b20q` 93.08, `b20s` 93.36) below the flat 94.4-94.9 band every batch since 11 has held. Every
consolidation column drops too (`sef` 2.1 vs 11.2, best-30 52.8 vs 64.1, pooled 43.1 vs 55.0), and
drawdown roughly doubles (11.1 vs 5.4). All four seeds favour the control on every column.

**‡ The first directional result in the batch, at the n=4 floor.** Seed-matched paired differences
(`25,50,25` − control):

| metric | by seed | mean | p (exact paired, 16 flips) | seeds favouring `25,50,25` |
|---|---|---|---|---|
| peak trailing | −0.72 / −0.36 / −0.98 / −0.70 | −0.69 | 0.125 | 0 of 4 |
| `sef` | −0.2 / −9.9 / −2.0 / −24.3 | −9.10 | 0.125 | 0 of 4 |
| best-30 | 0.0 / −12.0 / −9.6 / −23.3 | −11.23 | 0.250 | 0 of 4 |
| pooled | −3.0 / −5.2 / −13.3 / −26.1 | −11.89 | 0.125 | 0 of 4 |
| max drawdown | +8.6 / +6.6 / +6.1 / +1.5 | +5.71 | 0.125 | 0 of 4 (all worse) |

p bottoms out at the n=4 floor (0.125) on four of five metrics, every seed pointing the same way — the
cleanest signal batch 20 has produced. Unlike the shapes at or above the control's capacity, cutting to
0.29× lowers the ceiling *and* worsens consolidation.

All at 3M, sorted by best-30. Close-out under gate 95, `EVAL_WORKERS=4`, `SNEK_FC_LAYERS=25,50,25`.

| arm | peak trail | best-30 | `sef` | max drawdown | close-out pooled | best row |
|---|---|---|---|---|---|---|
| `b20r` | **94.48** | **66.3%** | **6.3%** | 10.84 | **57.4%** | 76.0% @1874k (n=25) |
| `b20t` | 94.06 | 57.0% | 2.0% | **6.22** | 45.3% | 60.0% @1530k (n=25) |
| `b20s` | 93.36 | 46.7% | 0.2% | 11.16 | 39.5% | 56.0% @739k (n=25) |
| `b20q` | 93.08 | 41.3% | 0.0% | 16.30 | 30.3% | 44.0% @897k (n=25) |
| **mean — `25,50,25`** | 93.75 | 52.8% | 2.1% | 11.13 | 43.1% | — |
| **mean — control** | 94.44 | 64.0% | 11.2% | 5.41 | 55.0% | — |

**No full-length rows** (deepest 25-29 of 100 under gate 95), so `best row` is a bound and `pooled` is the
exact column. Nothing near 95%. Drawdown mean 11.1 is the batch's worst so far, but still batch-19
territory, far from batch 18's ~57.

![b20r](../charts/b20r-fc25x50x25seed2.png)
**b20r-fc25x50x25seed2**

![b20t](../charts/b20t-fc25x50x25seed4.png)
**b20t-fc25x50x25seed4**

![b20s](../charts/b20s-fc25x50x25seed3.png)
**b20s-fc25x50x25seed3**

![b20q](../charts/b20q-fc25x50x25seed1.png)
**b20q-fc25x50x25seed1**

## Batch 20 wave 3 — deep-narrow (`60,30,30,30,30`) against the control, both to 3M

Five narrow layers: 6,573 params, **0.55×** the control, depth **5** — the deepest shape in the sweep,
testing whether depth helps once width is small.

**Verdict: it matches the ceiling but forgets more.** Peak trailing **94.25** vs 94.44 — inside the band,
essentially a match (−0.18, p 0.375). Consolidation is a wash (pooled 51.6 vs 55.0, p 0.5; one seed
favours it). What moves cleanly is drawdown: **12.3 vs 5.4, all four seeds worse** — deeper-and-narrower
holds the level but is markedly less steady.

**‡ Only drawdown separates from noise.** Seed-matched paired differences (`60,30,30,30,30` − control):

| metric | by seed | mean | p (exact paired, 16 flips) | seeds favouring the shape |
|---|---|---|---|---|
| peak trailing | −0.20 / −0.44 / +0.32 / −0.40 | −0.18 | 0.375 | 1 of 4 |
| `sef` | −0.1 / −14.0 / +9.7 / −21.7 | −6.53 | 0.375 | 1 of 4 |
| best-30 | −0.6 / −18.0 / +17.0 / −17.3 | −4.73 | 0.375 | 1 of 4 |
| pooled | −2.5 / −8.9 / +9.9 / −12.1 | −3.41 | 0.500 | 1 of 4 |
| max drawdown | +5.1 / +9.3 / +5.8 / +7.4 | +6.89 | 0.125 | 0 of 4 (all worse) |

The ceiling and consolidation gaps are seed-3-carried noise (p ≥ 0.375); drawdown is worse on all four
seeds at p=0.125.

All at 3M, sorted by best-30. Close-out under gate 95, `EVAL_WORKERS=4`, `SNEK_FC_LAYERS=60,30,30,30,30`.
Trained on the laptop; checkpoints rsynced to the desktop for the close-out.

| arm | peak trail | best-30 | `sef` | max drawdown | close-out pooled | best row |
|---|---|---|---|---|---|---|
| `b20w` | **94.66** | **73.3%** | **11.9%** | **10.88** | **62.7%** | 78.6% @1945k (n=28) |
| `b20x` | 94.36 | 63.0% | 4.6% | 12.04 | 59.2% | 75.0% @1139k (n=28) |
| `b20v` | 94.40 | 60.3% | 2.2% | 13.58 | 53.8% | 77.4% @2436k (n=31) |
| `b20u` | 93.60 | 40.7% | 0.1% | 12.74 | 30.8% | 56.0% @2534k (n=25) |
| **mean — `60,30,30,30,30`** | 94.25 | 59.3% | 4.7% | 12.31 | 51.6% | — |
| **mean — control** | 94.44 | 64.0% | 11.2% | 5.41 | 55.0% | — |

**No full-length rows** (deepest 25-36 of 100), so `best row` is a bound. Nothing near 95%.

![b20w](../charts/b20w-fc60x30x30x30x30seed3.png)
**b20w-fc60x30x30x30x30seed3**

![b20x](../charts/b20x-fc60x30x30x30x30seed4.png)
**b20x-fc60x30x30x30x30seed4**

![b20v](../charts/b20v-fc60x30x30x30x30seed2.png)
**b20v-fc60x30x30x30x30seed2**

![b20u](../charts/b20u-fc60x30x30x30x30seed1.png)
**b20u-fc60x30x30x30x30seed1**

## Batch 20 — capacity escalation (`100,200,100`, 3.69×) against the control, both to 3M

The largest net in the sweep: 43,703 params, **3.69×** the control, depth 3. It escalates above the 2.66×
(`200,100,50`) arm to ask whether *any* amount of extra capacity moves the ceiling, given 2.66× did not.
Ran on the desktop `the-claw-den`, seeds 1-4.

**Verdict: null on the ceiling, the seed-1 pattern again.** Peak trailing **94.61** vs the control's 94.44
(+0.17, p 0.875) — inside the flat 94.4-94.9 band. Consolidation reads up (best-30 71.3 vs 64.0, pooled
63.4 vs 55.0), but the rise is one seed: `b20ae` (+36 best-30, +35 pooled) against control arm `b20a`, the
batch's weak seed (41.3 best-30, 33.2 pooled). The other three seeds are flat-to-mixed (best-30 −9 / −1 /
+3) and no p clears the 0.125 floor. **3.69× behaves like the control**, the same reading `200,100,50`
(2.66×) gave — closing the capacity question: across 0.92× / 1.00× / 1.38× / 2.66× / 3.69× nothing moved the
ceiling, only cutting to 0.29× (`25,50,25`) did, downward.

All at 3M, sorted by best-30. Close-out under gate 95, `EVAL_WORKERS=4`, `SNEK_FC_LAYERS=100,200,100`.

| arm | peak trail | best-30 | `sef` | close-out pooled | best row |
|---|---|---|---|---|---|
| `b20ah` | **94.88** | **83.0%** | **32.9%** | **74.6%** | 91.8% @1470k (n=73) |
| `b20ae` | 94.76 | 77.7% | 21.8% | 67.8% | 89.1% @1561k (n=55) |
| `b20af` | 94.60 | 69.3% | 10.2% | 63.1% | 82.9% @1918k (n=35) |
| `b20ag` | 94.18 | 55.3% | 1.4% | 48.3% | 64.0% @551k (n=25) |
| **mean — `100,200,100`** | 94.61 | 71.3% | 16.6% | 63.4% | — |
| **mean — control** | 94.44 | 64.0% | 11.2% | 55.0% | — |

**No full-length rows** (deepest 25-82 of 100 under gate 95), so `best row` is a bound and `pooled` is exact.
Nothing near 95%.

![b20ah](../charts/b20ah-fc100x200x100seed4.png)
**b20ah-fc100x200x100seed4**

![b20ae](../charts/b20ae-fc100x200x100seed1.png)
**b20ae-fc100x200x100seed1**

![b20af](../charts/b20af-fc100x200x100seed2.png)
**b20af-fc100x200x100seed2**

![b20ag](../charts/b20ag-fc100x200x100seed3.png)
**b20ag-fc100x200x100seed3**

## Batch 20 wave 3 — iso-param depth-2 (`93,93`) against the control, both to 3M

Two layers of 93: 11,907 params, **1.00×** the control, at depth 2 against the control's depth 3. The
iso-param depth rung between the control (depth 3) and `320` (depth 1) — hold capacity, cut a layer.

**Verdict: null — matches the control.** Peak trailing 94.41 vs 94.44 (−0.03), the ceiling unmoved, and no
metric separates at n=4 (p 0.375-1.0, 1 of 4 seeds favouring the shape). Seed 1's big best-30 gap (+33) is a
control weakness — `b20a` is the batch's weak arm — not a shape strength. Depth 2 behaves like the control,
exactly as depth-1 `320` did; only `25,50,25`'s 0.29× cut has moved the ceiling.

All at 3M, sorted by best-30. Close-out under gate 95, `EVAL_WORKERS=4`, `SNEK_FC_LAYERS=93,93`.

| arm | peak trail | best-30 | `sef` | close-out pooled | best row |
|---|---|---|---|---|---|
| `b20aa` | **94.78** | **74.3%** | **15.6%** | **65.6%** | 82% @2908k (n=38) |
| `b20ad` | 94.44 | 66.7% | 10.0% | 56.1% | 79% @2912k (n=28) |
| `b20ab` | 94.32 | 54.0% | 0.7% | 42.0% | 56% @2661k (n=25) |
| `b20ac` | 94.08 | 50.7% | 0.6% | 42.2% | 56% @2663k (n=25) |
| **mean — `93,93`** | 94.41 | 61.4% | 6.7% | 51.5% | — |
| **mean — control** | 94.44 | 64.0% | 11.2% | 55.0% | — |

**No full-length rows** (deepest 25-38 of 100 under gate 95), so `best row` is a bound and `pooled` is exact.
Nothing near 95%.

![b20aa](../charts/b20aa-fc93x93seed1.png)
**b20aa-fc93x93seed1**

![b20ad](../charts/b20ad-fc93x93seed4.png)
**b20ad-fc93x93seed4**

![b20ab](../charts/b20ab-fc93x93seed2.png)
**b20ab-fc93x93seed2**

![b20ac](../charts/b20ac-fc93x93seed3.png)
**b20ac-fc93x93seed3**

## Batch 20 wave 2 — depth-1 (`320`) against the control, both to 3M

A single hidden layer of 320 units: 10,883 params, **0.92×** the control, and depth 3 → 1. It holds
capacity roughly constant and strips out all depth, so it is the cleanest architecture test available —
if it matches the control, depth is contributing nothing.

**Verdict: it matches the control — depth buys nothing here.** Peak trailing **94.67** against 94.44, all
four seeds inside the flat 94.4-94.9 band. The consolidation columns all read higher (`sef` 16.5 vs 11.2,
best-30 74.7 vs 64.1, pooled 65.1 vs 55.0) and, unlike `200,50`, they go the *right* way — but none of it
reaches significance.

**‡ 3 of 4 seeds favour the shape on every column, but the magnitude is one seed.** Seed-matched paired
differences (`320` − control):

| metric | by seed | mean | p (exact paired, 16 flips) | seeds favouring `320` |
|---|---|---|---|---|
| peak trailing | +0.82 / +0.02 / +0.10 / +0.00 | +0.23 | 0.250 | 4 of 4 |
| `sef` | +12.7 / +6.9 / +1.7 / −0.1 | +5.30 | 0.250 | 3 of 4 |
| best-30 | +31.7 / +3.0 / +8.0 / −0.3 | +10.60 | 0.250 | 3 of 4 |
| pooled | +32.1 / +7.6 / −1.0 / +1.7 | +10.10 | 0.250 | 3 of 4 |

The consistency is better than `200,50`'s (1 of 4), but `p` bottoms out at 0.250 at n=4, and seed 1 —
where the control `b20a` is the batch's weakest arm — supplies most of every mean (excluding it, the
pooled edge is +2.8).

All at 3M, sorted by best-30. Close-out under gate 95, `EVAL_WORKERS=4`, `SNEK_FC_LAYERS=320`.

| arm | peak trail | best-30 | `sef` | max drawdown | close-out pooled | best row |
|---|---|---|---|---|---|---|
| `b20n` | **94.86** | **81.3%** | 23.1% | 6.10 | 70.3% | 87.0% @2896k (n=54) |
| `b20p` | 94.76 | 80.0% | **26.2%** | 6.44 | **73.0%** | 89.3% @2321k (n=56) |
| `b20m` | 94.62 | 73.0% | 12.9% | **5.24** | 65.3% | 82.4% @2935k (n=34) |
| `b20o` | 94.44 | 64.3% | 3.9% | 11.96 | 51.8% | 69.2% @2807k (n=26) |
| **mean — `320`** | 94.67 | 74.7% | 16.5% | 7.44 | 65.1% | — |
| **mean — control** | 94.44 | 64.0% | 11.2% | 5.41 | 55.0% | — |

**No arm produced a full-length row** (deepest 26-56 of 100), so `best row` is a bound — `pooled` is the
exact column. Nothing reached 95%. Drawdown rose to 7.44, driven by `b20o` (11.96), still batch-19
territory and far from batch 18's ~57 — the base's anti-forgetting property holds under this shape too.

![b20n](../charts/b20n-fc320seed2.png)
**b20n-fc320seed2**

![b20p](../charts/b20p-fc320seed4.png)
**b20p-fc320seed4**

![b20m](../charts/b20m-fc320seed1.png)
**b20m-fc320seed1**

![b20o](../charts/b20o-fc320seed3.png)
**b20o-fc320seed3**

## Batch 20 wave 2 — wide-early (`200,50`) against the control, both to 3M

The wide-early shape: two layers, 200 units first, 16,403 params (1.38× the control). The mechanism
was that the observation is already high-level, so what the net needs is *conjunctions* of engineered
features rather than a deep hierarchy — and conjunctions want width in the first layer.

**Verdict: it did not move the ceiling either, and it is behind the control on the primary metric.**
Peak trailing **94.55** against the control's 94.44 — a 0.11 gap inside a band nine batches have held.
`sef` **8.60%** against 11.2%, i.e. *worse* on the metric with the lowest between-seed variance.

**‡ The apparent best-30 and pooled gains are one seed, and that seed is a control weakness.**
Seed-matched paired differences (`200,50` − control) are what settles this:

| metric | by seed | mean | p (exact paired, 16 flips) | seeds favouring `200,50` |
|---|---|---|---|---|
| `sef` | +8.5 / −6.3 / −0.3 / −12.4 | −2.62 | 0.625 | **1 of 4** |
| best-30 | +31.4 / −9.0 / −1.0 / −8.0 | +3.35 | 1.000 | **1 of 4** |
| pooled | +31.3 / −3.4 / −4.3 / −6.5 | +4.27 | 1.000 | **1 of 4** |

Every positive mean is carried by seed 1 alone, where the control arm `b20a` is the batch's weakest
(`sef` 0.2%, pooled 33.2%). Three of four seeds favour the control on all three metrics.

All at 3M, sorted by best-30. Close-out under gate 95, `EVAL_WORKERS=4`, `SNEK_FC_LAYERS=200,50`.

| arm | peak trail | best-30 | `sef` | max drawdown | close-out pooled | best row |
|---|---|---|---|---|---|---|
| `b20i` | 94.64 | **72.7%** | 8.7% | 9.48 | 64.5% | 87.0% @2818k (n=46) |
| `b20l` | **94.82** | 72.3% | **13.9%** | **7.18** | **64.8%** | 85.7% @2940k (n=42) |
| `b20j` | 94.58 | 69.3% | 9.9% | 9.76 | 59.3% | 83.3% @2001k (n=42) |
| `b20k` | 94.18 | 55.3% | 1.9% | 7.84 | 48.5% | 72.0% @1073k (n=25) |
| **mean — `200,50`** | 94.55 | 67.4% | 8.60% | 8.56 | 59.3% | — |
| **mean — control** | 94.44 | 64.0% | 11.2% | 5.41 | 55.0% | — |

**No arm produced a full-length row** (deepest 25-46 of 100), so `best row` is a bound, not a
measurement — `pooled` is the exact column. Nothing reached 95%.

Drawdown rose from the control's 5.41 to 8.56 but stayed in batch 19 territory (8.76), nowhere near
batch 18's ~57 — so the base's anti-forgetting property survives this shape too.

![b20i](../charts/b20i-fc200x50seed1.png)
**b20i-fc200x50seed1**

![b20l](../charts/b20l-fc200x50seed4.png)
**b20l-fc200x50seed4**

![b20j](../charts/b20j-fc200x50seed2.png)
**b20j-fc200x50seed2**

![b20k](../charts/b20k-fc200x50seed3.png)
**b20k-fc200x50seed3**

## Batch 20 wave 1 — FC-layer capacity (`200,100,50`) vs control (`50,100,50`), both to 3M

The first time the network shape has been varied in the project, aimed at the one quantity nine batches
of optimiser knobs never moved — the ceiling. Control `50,100,50` re-baselined at β=300k (`b20a-d`) and
capacity `200,100,50`, 2.66× the parameters (`b20e-h`). Both ran to **3M** at a matched horizon — the
capacity arms crashed once at ~1.75M on a since-fixed matplotlib chart-writer leak, then resumed to 3M.

**Verdict: 2.66× capacity did not move the ceiling.** Peak trailing **94.44** (control) vs **94.70**
(capacity) — both inside the flat 94.7-95.0 band every batch has held since 11, for a net 2.66× wider.
There is an apparent consolidation edge (best-30 64.0 → 71.4, close-out pooled 55.0 → 64.6) which the
original write-up called "a hint, not a result".

**‡ Downgraded 2026-08-10: it is not even a directional hint.** The paired per-seed differences are
**+28.7 / −4.7 / +19.4 / −5.0** on pooled — **2 of 4** seeds, exact paired p=**0.500** — and the two
seeds carrying it are seeds 1 and 3, exactly where the *control* is weakest (`b20a` `sef` 0.2%, `b20c`
2.2%). The same check on wave 2's `200,50` shows the same shape at 1 of 4. So the edge is control seed
variance, not capacity: the control's four seeds span `sef` 0.2-26.3% and pooled 33.2-71.3%, which is
wider than any between-shape gap in the batch. Capacity-up is not the route to a higher ceiling.

All at 3M. Sorted by best-30. `sef` and close-out pooled (equal-effort, gate 95) are now comparable — same horizon.

| arm | shape | peak trail | best-30 | `sef` | close-out pooled |
|---|---|---|---|---|---|
| `b20g` | 200,100,50 | 94.76 | **81.7%** | 25.6% | **72.2%** |
| `b20d` | 50,100,50 | 94.76 | 80.3% | 26.3% | 71.3% |
| `b20b` | 50,100,50 | 94.84 | 78.3% | 16.2% | 62.7% |
| `b20h` | 200,100,50 | **94.82** | 76.3% | 16.7% | 66.3% |
| `b20e` | 200,100,50 | 94.56 | 68.3% | 7.4% | 61.9% |
| `b20f` | 200,100,50 | 94.64 | 59.3% | 1.7% | 58.0% |
| `b20c` | 50,100,50 | 94.34 | 56.3% | 2.2% | 52.8% |
| `b20a` | 50,100,50 | 93.80 | 41.3% | 0.2% | 33.2% |
| **mean — control** | 50,100,50 | 94.44 | 64.0% | 11.2% | 55.0% |
| **mean — capacity** | 200,100,50 | 94.70 | 71.4% | 12.9% | 64.6% |

Per-arm numbers are in the table above; charts (3M) best-first within each shape.

### Control `50,100,50`

![b20d](../charts/b20d-fc50seed4.png)
**b20d-fc50seed4**

![b20b](../charts/b20b-fc50seed2.png)
**b20b-fc50seed2**

![b20c](../charts/b20c-fc50seed3.png)
**b20c-fc50seed3**

![b20a](../charts/b20a-fc50seed1.png)
**b20a-fc50seed1**

### Treatment `200,100,50` (2.66× params)

Resumed from the ~1.75M crash (a since-fixed chart-writer leak, not the arms) to 3M. `b20g` is the
strongest arm of the batch (best-30 81.7%, close-out 72.2%); `b20f` is the laggard, mirroring `b20a`.

![b20g](../charts/b20g-fc200seed3.png)
**b20g-fc200seed3**

![b20h](../charts/b20h-fc200seed4.png)
**b20h-fc200seed4**

![b20e](../charts/b20e-fc200seed1.png)
**b20e-fc200seed1**

![b20f](../charts/b20f-fc200seed2.png)
**b20f-fc200seed2**

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

![b19d](../charts/b19d-stdperseed4.png)

**The seed that escaped, and it is why the batch is a falsification rather than a catastrophe.** It is
level with its batch-18 control on every column (`sef` 40.2 against 41.6, peak 94.86 against 94.92),
so whatever standard PER costs, one seed in four does not pay it. Read against the other three, this
arm is the evidence that the damage is a *distribution* shifting left rather than a mechanism that
cannot work.

### b19a-stdperseed1 — standard PER, seed 1

2192k steps, peak trailing 94.66 @1299k, best-30 71.0% @1301k, `sef` 8.0%, recent-30 43.3%,
max drawdown **4.94** — the smallest in the batch.

![b19a](../charts/b19a-stdperseed1.png)

Peaked early at 1.30M and then held a visibly flat, slightly declining band for 900k steps without
ever collapsing. Its control `b18a` reached `sef` 41.2% against this arm's 8.0% while suffering a 56.6
drawdown, so the pair is the whole batch in miniature: far steadier, far worse.

### b19b-stdperseed2 — standard PER, seed 2

2116k steps, peak trailing 94.40 @1889k, best-30 66.7% @1914k, `sef` 4.6%, recent-30 51.3%,
max drawdown 10.20.

![b19b](../charts/b19b-stdperseed2.png)

**The slowest consolidator of the four:** it did not reach pf30 ≥ 60% until 1861k where its control
did so at 310k, a 1.55M gap. It was still improving when stopped — peak trailing and best-30 both land
in its final 250k — so this is an arm whose curve had not finished, and the one place the batch's
horizon is a genuine caveat rather than a formality.

### b19c-stdperseed3 — standard PER, seed 3

2004k steps, peak trailing 92.72 @1500k, best-30 **29.7%** @195k, `sef` **0.0%**, recent-30 14.3%,
max drawdown 7.04.

![b19c](../charts/b19c-stdperseed3.png)

**Not one eval at ≥80% perfect in 2005 evals**, and its best 30-eval window came at **195k** — the
opening of the run. It never reached pf30 ≥ 40%, and peak trailing 92.72 is more than two points below
anything else in batches 18-19. It was never dead (`zero_since` null, trailing 88.7 at the end) and it
never collapsed — so it is **none of the four failure modes** in
[`failureModes.md`](../failureModes.md): it learned to play well, plateaued just short of finishing, and
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

![b18d](../charts/b18d-tgt1000seed4.png)

Step 2.60M · peak trailing 94.92 (at 2346k) · **best 30-eval perfect 91.0%** (at 2369k) · `strong_eval_fraction` **47.9%** · recent-30 79.0%

**The steadiest arm on record.** 47.9% of its evals at >=80% perfect is the highest ever, and its max
drawdown of **32.62** is less than half the batch-17 mean — the red trace stays high instead of
collapsing and recovering. Reached pf30 >= 40% at 216k against its control's 329k.

### b18a-tgt1000seed1 — period 1000, forking on, seed 1

![b18a](../charts/b18a-tgt1000seed1.png)

Step 2.61M · peak trailing 94.92 (at 1605k) · best 30-eval perfect 88.0% (at **2480k**) · `strong_eval_fraction` 41.4% · recent-30 64.0%

Second-highest `sef` ever, and still finding its best window at 2480k — near the end of the run. Its
seed-1 counterpart in batch 17 was the arm that made that batch a null, so **the pairing that looked
worst going in came out +40 pp on `sef`**. One seed, but worth noting against the temptation to read
seed identity as arm quality.

### b18c-tgt1000seed3 — period 1000, forking on, seed 3

![b18c](../charts/b18c-tgt1000seed3.png)

Step 2.51M · peak trailing 94.94 (at 2161k) · best 30-eval perfect 84.3% (at 2168k) · `strong_eval_fraction` 24.3% · recent-30 **79.3%**

**Highest recent-30 of the batch at the moment it was stopped**, with peak and best window both inside
its last 350k — the arm most plausibly cut short.

### b18b-tgt1000seed2 — period 1000, forking on, seed 2

![b18b](../charts/b18b-tgt1000seed2.png)

Step 2.40M · peak trailing 94.96 (at 1575k) · best 30-eval perfect 86.0% (at 1600k) · `strong_eval_fraction` 25.0% · recent-30 56.3%

**Fastest start on record: pf30 >= 40% at 180k.** Also the batch's worst drawdown at 85.08, and the
weakest recent-30 — it peaked around 1.6M and gave a lot back. The two facts together are the case for
reading this batch on speed rather than on stability.

## Batch 17 — forked endgame collection (`SNEK_FORK_*`), a null, stopped at 1.41-1.57M

Four seeds, batch 16's config exactly plus forking at length ≥ 85 — one variable, an exact control.
**A null**: `strong_eval_fraction` -1.67 pp at a matched 1.245M (p=0.875). Full write-up in
[`completedRuns.md`](../completedRuns.md#batch-17--forked-endgame-collection-a-null-that-produced-the-project-record).

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

![b17b](../charts/b17b-forkseed2.png)

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
[`completedRuns.md`](../completedRuns.md#-the-close-out-b17b-is-the-best-policy-this-project-has-measured-by-a-clear-margin).

### b17d-forkseed4 — forking on, disc 0.9975 + shield 0.8, seed 4

![b17d](../charts/b17d-forkseed4.png)

Step 1.51M · peak trailing 94.60 (at 1007k) · best 30-eval perfect 75.7% (at **679k**) · `strong_eval_fraction` 20.7% · recent-30 68.3%

Fastest starter of the batch (pf30 ≥ 40% at 329k) and its best window came earliest of any arm here, at
679k — then 800k steps without beating it. A flat-after-early-peak shape, like `b16a`.

### b17c-forkseed3 — forking on, disc 0.9975 + shield 0.8, seed 3

![b17c](../charts/b17c-forkseed3.png)

Step 1.52M · peak trailing 94.78 (at **1386k**) · best 30-eval perfect 82.0% (at **1388k**) · `strong_eval_fraction` 16.4% · recent-30 73.3%

**Still climbing when it was stopped** — both its peak and its best window land in its final 140k
steps, and its `sef` went 6.7% → 16.4% over the last 500k. Also carries the batch's worst drawdown
(85.96). The arm most likely to have been cut short.

### b17a-forkseed1 — forking on, disc 0.9975 + shield 0.8, seed 1

![b17a](../charts/b17a-forkseed1.png)

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
[`completedRuns.md`](../completedRuns.md#batch-16--the-food-distance-shaping-ablated-the-first-non-null-in-six-batches).

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

![b16b](../charts/b16b-noshapeseed2.png)

Step 1.26M · **peak trailing 94.98** (at 816k) · best 30-eval perfect 85.0% (at 919k) · `strong_eval_fraction` **30.7%** · recent-30 79.0%

**Strongest of the batch and the flattest** — peaked at 816k and held 79% recent perfect at stop. At
this ~1.25M horizon its `sef` matches batch 14's best arms, but the horizons are not comparable; the
close-out is what settles level.

### b16a-noshapeseed1 — shaping off, disc 0.9975 + shield 0.8, seed 1

![b16a](../charts/b16a-noshapeseed1.png)

Step 1.25M · peak trailing 94.82 (at 837k) · **best 30-eval perfect 87.0%** (at 850k) · `strong_eval_fraction` 20.6% · recent-30 77.0%

Highest best-30 window of the batch (87.0%), though both peak and best window land early (~840k) and
it has plateaued since. Solid but unremarkable.

### b16c-noshapeseed3 — shaping off, disc 0.9975 + shield 0.8, seed 3

![b16c](../charts/b16c-noshapeseed3.png)

Step 1.26M · peak trailing 94.36 (at **1198k**) · best 30-eval perfect 72.7% (at 1221k) · `strong_eval_fraction` 10.6% · recent-30 67.0%

**The latest-peaking arm** — best window at 1221k, at the very end — so unlike its siblings it may not
have plateaued when stopped. Recovered from a mid-run dip to ~86% trailing back to 93.7%, which is
what dragged its `sef` down despite a normal peak.

### b16d-noshapeseed4 — shaping off, disc 0.9975 + shield 0.8, seed 4

![b16d](../charts/b16d-noshapeseed4.png)

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
[`completedRuns.md`](batches12-15.md#batch-15--n_step_update3-falsified-on-speed-null-on-level-and-a-97100-that-is-really-93).

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

![b15a](../charts/b15a-nstep3seed1.png)

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

![b15b](../charts/b15b-nstep3seed2.png)

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

![b15d](../charts/b15d-nstep3seed4.png)

Step 5.81M · peak trailing 94.70 (at **5799k**) · best 30-eval perfect 86.3% (at 3687k) · `strong_eval_fraction` 33.8% · final band **75.8%**

**Its peak trailing score is at 5799k — the second-to-last eval it ever ran.** Also still gaining in
its final band (75.8% against a 72.8% previous best). This is the single clearest piece of evidence
in the project that stopping at a round number truncates arms: nothing about this curve suggests it
was finished.

### b15c-nstep3seed3 — n=3, disc 0.995 + shield 0.8, seed 3

![b15c](../charts/b15c-nstep3seed3.png)

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

## Batch 14 — `DISCOUNT=0.9975` at `GUIDED_FRACTION=0.8`, and a third null

Four seeds run to 4.1-4.5M, the longest arms on the current vector. **Null against batch 13 on every
metric** — `pooled_equal_effort` 72.08% against 72.07%, best checkpoint +2.8 pp at p=1.000, and
`strong_eval_fraction` +2.1 pp with per-seed diffs from -16.2 to +24.8. Full write-up:
[`completedRuns.md`](batches12-15.md#batch-14--disc-09975-at-guided-08-and-the-widest-seed-spread-yet).

**The one result worth keeping is a chart-shape result, and it is about horizon.** Two arms produced
their best window past 3.5M and `b14c` was still climbing when stopped — which is why the step cap
moved 5M → 10M. Every earlier batch was killed by hand near 3.5M, so the long-standing "arms peak
between ~1M and ~3.4M" reading was partly describing the stopping habit.

| seed | b14 best30 | b13 best30 | diff | peak window |
|---|---|---|---|---|
| 1 | 79.7% | 78.0% | +1.7 | **3707k** |
| 2 | 76.3% | 82.3% | -6.0 | 2282k |
| 3 | 87.7% | 85.3% | +2.3 | **4135k** |
| 4 | **89.7%** | 83.3% | +6.3 | 2700k |
| **mean** | **83.3%** | **82.2%** | **+1.1** | |

**Do not read the graph-100% tier off these arms.** Batch 14 is the first batch measured under
`EVAL_MIN_ACHIEVABLE=90`, which censors that tier from below and inflates it by ~15 pp — see
[`hyperparamTuning.md`](../hyperparamTuning.md#taking-the-arm-level-pooled-rate).

### b14d-disc9975seed4 — disc 0.9975 + shield 0.8, seed 4

![b14d](../charts/b14d-disc9975seed4.png)

Step 4.46M · peak trailing **94.9** (at 2554k) · **best 30-eval perfect 89.7%** (at 2700k) · `strong_eval_fraction` **39.3%** · trailing-30 at stop 78.0%

**The strongest arm the project has recorded on the primary metric** — 39.3% of its evals at ≥80%,
against a previous best of 30.5% (`b11b`). It is also the flattest good arm here: its mean perfect
rate climbed monotonically through 2.5-3.0M (peaking at 81.1% per 500k band) and never fell below
67% afterwards, for only 11.7 pp of drawdown from peak to stop.

Its best checkpoint is 93% @2559k, below `b14a`'s 96%, which is the usual split between an arm that
holds a high level and an arm that spikes once.

### b14a-disc9975seed1 — disc 0.9975 + shield 0.8, seed 1

![b14a](../charts/b14a-disc9975seed1.png)

Step 4.17M · peak trailing 94.8 (at 3794k) · best 30-eval perfect 79.7% (at 3707k) · `strong_eval_fraction` 20.0% · trailing-30 at stop 54.7%

**Produced a 96/100 checkpoint at 3702000, tying `b11b` for the best selected measurement on
record** — and then gave back 25 pp by the time it stopped. Both facts are the arm: its peak window
is the *latest* of any arm on this vector, and it was already falling apart 400k later.

The 96% does not survive a second look at full strength. An independent 100-episode re-measurement
of the same checkpoint read **91/100**, so the honest pooled estimate is **187/200 = 93.5%** (CI
89.2-96.2). That gap is the winner's curse made visible — this checkpoint was the maximum over 176
attempted full-length measurements in this arm.

### b14c-disc9975seed3 — disc 0.9975 + shield 0.8, seed 3

![b14c](../charts/b14c-disc9975seed3.png)

Step 4.16M · peak trailing 94.8 (at 2105k) · **best 30-eval perfect 87.7%** (at 4135k) · `strong_eval_fraction` 17.8% · trailing-30 at stop 82.0%

**The arm that moved the step cap.** Its best 30-eval window is at 4135k — the last one it ran — and
its final 4.0-4.5M band is its strongest of the whole run at 75.9% mean perfect against a 62.6%
previous best. It was still improving when it was stopped, and a 5M cap would have cut it mid-climb.

Only 5.7 pp of drawdown, the smallest in the batch, for the same reason: it never peaked.

### b14b-disc9975seed2 — disc 0.9975 + shield 0.8, seed 2

![b14b](../charts/b14b-disc9975seed2.png)

Step 4.12M · peak trailing 94.5 (at 2053k) · best 30-eval perfect 76.3% (at 2282k) · `strong_eval_fraction` **9.3%** · trailing-30 at stop 29.3%

Weakest of the batch and the clearest decay curve on the current vector: peaked at 2.05M, then lost
**47 pp** of perfect rate over the next 2M steps. Its epsilon reads 0.0064 at stop against its
siblings' 0.002-0.0036, which is the anti-ratchet buying exploration back in response — working as
designed, and not enough to arrest the slide.

It is also the arm that shows what the 90% gate costs: **one** full-length row survived the whole
close-out, so it has a best checkpoint (90%) and no meaningful top-3.

---

## Batch 13 — the lower handover plus the shield, and an exact null

Four seeds, handover 0.0125 and `GUIDED_FRACTION=0.5`, run to 3.4-3.7M. **The schedule works and
the outcome is unchanged.** Epsilon descended on skill to 0.0023-0.0050, all four passed the
pre-registered 350k check, and `best_perfect30` came out at a mean of **82.2% against batch 11's
82.2%** — an exact null, p = 1.000 on an exact paired permutation test.

Read these charts against batch 11's below and the difference is that there is no difference. What
*is* gone is batch 12's shape: no arm here peaks early and then decays to 55.

| seed | b13 best30 | b11 best30 | diff |
|---|---|---|---|
| 1 | 78.0% | 85.7% | -7.7 |
| 2 | 82.3% | 91.7% | -9.4 |
| 3 | 85.3% | 73.0% | +12.3 |
| 4 | 83.3% | 78.3% | +5.0 |
| **mean** | **82.2%** | **82.2%** | **+0.0** |

Per-seed swings of ±12 pp around a zero mean is what n=4 looks like on this metric, and it is the
clearest statement in this file of why seed count is the binding constraint.

### b13c-shieldseed3 — handover 0.0125 + shield, seed 3

![b13c](../charts/b13c-shieldseed3.png)

Step 3.67M · peak trailing 94.8 (at 3185k) · **best 30-eval perfect 85.3%** (at 2864k) · `strong_eval_fraction` **26.5%** · trailing-30 at stop 72.3%

Best of the batch, and the arm that inverts batch 11's seed ordering: seed 3 was batch 11's weakest
at 73.0% and is batch 13's strongest at 85.3%. Same seed, same config but the schedule — a +12.3 pp
swing that means nothing on its own and is exactly why the batch mean is what gets reported.

Still near its peak at 3.19M when stopped, with the smallest gap in the batch between peak trailing
and where it ended.

### b13d-shieldseed4 — handover 0.0125 + shield, seed 4

![b13d](../charts/b13d-shieldseed4.png)

Step 3.51M · peak trailing 94.5 (at 980k) · best 30-eval perfect 83.3% (at 1005k) · `strong_eval_fraction` 14.5% · trailing-30 at stop **39.0%**

Peaked earliest in the batch at ~1M and gave up **44.3 pp** by 3.5M — the largest drawdown here, and
the reason the shield cannot be credited with fixing the post-peak decline: batch 11's seed 4 was
its *most* stable arm at 5.6 pp. The paired drawdown comparison is -1.0 pp at p = 0.875, i.e.
nothing.

### b13b-shieldseed2 — handover 0.0125 + shield, seed 2

![b13b](../charts/b13b-shieldseed2.png)

Step 3.70M · peak trailing 94.8 (at 1919k) · best 30-eval perfect 82.3% (at 1508k) · `strong_eval_fraction` 25.4% · trailing-30 at stop 67.0%

**The fastest start on record**: trailing 92.4 with a 72.3% perfect rate by step 350k, where batch
12's arms were at 0%. Whatever else the epsilon change did or did not do, that is the deadlock
being decisively absent.

### b13a-shieldseed1 — handover 0.0125 + shield, seed 1

![b13a](../charts/b13a-shieldseed1.png)

Step 3.39M · peak trailing 94.5 (at 2661k) · best 30-eval perfect 78.0% (at 2679k) · `strong_eval_fraction` 11.5% · trailing-30 at stop 70.7%

Weakest of the batch and the slowest to get going — 2.0% perfect at 350k, the only arm that would
have looked marginal against the abandon condition. Its best work came latest of the four, at 2.68M,
and it held most of it: a 7.3 pp drawdown against its own seed's 42.4 pp in batch 11.

---

## Batch 12 — the epsilon rewrite, and the deadlock it found

Four seeds of batch 11's config plus the two-phase epsilon schedule, **stopped at ~1M of a
planned 2.5M** because all four failed the same way: epsilon pinned at the refinement ceiling
0.05 and the perfect rate never left 0. Read these four charts as one shape repeated four times —
a fast climb to 81-87 trailing between 214k and 479k, then a slow decay to 53-63 that never
recovers. Both numbers are greedy evals, so that decay is the learned policy getting worse, not
an exploration tax on the measurement.

**`strong_eval_fraction` is 0.0% in all four arms**, against 25.2 / 30.5 / 0.0 / 8.2% for batch 11
at the same 1M steps. The mechanism, the fix, and the two wrong turns taken diagnosing it are in
[`completedRuns.md`](batches12-15.md#-the-new-schedule-deadlocks-all-four-arms-are-failing-44-at-1m-steps). These
arms are kept as the measured cost of sitting at epsilon 0.05: not a wasted batch, a negative
result with four seeds behind it.

### b12s-shield05seed1 — the exploration shield at handover 0.05, seed 1

![b12s](../charts/b12s-shield05seed1.png)

Step 0.43M (stopped) · trailing 83.1 at stop · best 30-eval perfect 0.3% · max single eval 10% · not measured

**The arm that moved the handover.** A verification run, `SEED=1` so it pairs with `b12a`, with the
one-step exploration shield on and the handover still at 0.05. It **fixed the decay** — `b12a` fell
83.8 → 74.2 between 200k and 400k while this one was still rising — and **did not fix perfect
games**: 2 perfect-game evals in 431, plateauing at trailing ~83 where the perfect rate is ~0,
improving at 4.7 points per 100k against `b11a`'s 11.1.

Kept because it is the whole argument for dropping the handover to 0.0125: a one-step mask prevents
blunders but not self-trapping, so the collect policy still never finishes a board and the buffer
never contains the last ten food. Read against `b12a` below and `b11a` above.

### b12a-eps002seed1 — two-phase epsilon, seed 1

![b12a](../charts/b12a-eps002seed1.png)

Step 1.12M (stopped) · peak score 89.1, peak trailing 87.02 (at 214k) · best 30-eval perfect **6.3%** (at 213k) · max single eval 40% · not measured

The best of a bad batch, and the arm that makes the decay unambiguous. It read trailing **87.0**
with 6.3% best-30 at 214k — a genuinely promising arm — then fell to 59.6 over the next 900k steps
**at exactly the same epsilon**. Same exploration rate, worse policy, so nothing about the
measurement explains it.

41 of its 1122 evals contained a perfect game, which was enough to nudge epsilon to 0.0388 at its
best and never enough to escape: the refinement phase needs 20-40% to reach the floor, and 0.05
makes that unreachable.

### b12d-eps002seed4 — two-phase epsilon, seed 4

![b12d](../charts/b12d-eps002seed4.png)

Step 1.09M (stopped) · peak score 87.2, peak trailing 86.36 (at 479k) · best 30-eval perfect 6.3% (at 29k) · max single eval 30% · not measured

The latest peak in the batch at 479k, and the only arm whose best-30 came in its first 30k steps —
during the bootstrap phase, before the ceiling took hold. Everything after is decline.

### b12c-eps002seed3 — two-phase epsilon, seed 3

![b12c](../charts/b12c-eps002seed3.png)

Step 0.98M (stopped) · peak score 85.3, peak trailing 82.46 (at 360k) · best 30-eval perfect 1.7% (at 369k) · max single eval 10% · not measured

8 perfect games in 977 evals, and a max single eval of 10% — it reached the endgame often enough
to prove the policy was not hopeless and never often enough for the schedule to notice.

### b12b-eps002seed2 — two-phase epsilon, seed 2

![b12b](../charts/b12b-eps002seed2.png)

Step 1.03M (stopped) · peak score 82.8, peak trailing 81.4 (at 259k) · **best 30-eval perfect 0.0%** · max single eval **0%** · not measured

**The cleanest demonstration of the deadlock in the project: zero perfect games in 1032 evals.**
Epsilon reached 0.05 at step 11000 and sat there for the remaining 942k steps, because the signal
that would have lowered it requires finishing a game and 3.3% random actions never let it. An arm
that peaked at 81.4 trailing was never once measured completing the board.

---
