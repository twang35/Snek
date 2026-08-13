# Charts

Progress graphs for the most recent batches — **20 through 23**, a cap of six, newest first. Per-arm
numbers live in
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

**Three PNGs in `charts/` are not arm charts and will always appear in that list** —
`champion-vs-mediocre`, `drawdown-b23b-vs-b18` and `per-b18-vs-b20-priorities` are diagnostic figures
referenced from [`findings.md`](findings.md) and [`perDiagnostics/`](perDiagnostics/README.md), not
training graphs. Anything *else* the check prints is a real gap.

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
every batch since 11 — ceiling unmoved, only consolidation differs. `b23b` holds a dense strong region —
**five full-length checkpoints ≥95/100 around 777k, best 97/100** — a hall-of-fame candidate pending the
re-measurement protocol.

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

![b20ai](charts/b20ai-fc100x50x50seed1.png)
**b20ai-fc100x50x50seed1**

![b20al](charts/b20al-fc100x50x50seed4.png)
**b20al-fc100x50x50seed4**

![b20ak](charts/b20ak-fc100x50x50seed3.png)
**b20ak-fc100x50x50seed3**

![b20aj](charts/b20aj-fc100x50x50seed2.png)
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

![b20r](charts/b20r-fc25x50x25seed2.png)
**b20r-fc25x50x25seed2**

![b20t](charts/b20t-fc25x50x25seed4.png)
**b20t-fc25x50x25seed4**

![b20s](charts/b20s-fc25x50x25seed3.png)
**b20s-fc25x50x25seed3**

![b20q](charts/b20q-fc25x50x25seed1.png)
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

![b20w](charts/b20w-fc60x30x30x30x30seed3.png)
**b20w-fc60x30x30x30x30seed3**

![b20x](charts/b20x-fc60x30x30x30x30seed4.png)
**b20x-fc60x30x30x30x30seed4**

![b20v](charts/b20v-fc60x30x30x30x30seed2.png)
**b20v-fc60x30x30x30x30seed2**

![b20u](charts/b20u-fc60x30x30x30x30seed1.png)
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

![b20ah](charts/b20ah-fc100x200x100seed4.png)
**b20ah-fc100x200x100seed4**

![b20ae](charts/b20ae-fc100x200x100seed1.png)
**b20ae-fc100x200x100seed1**

![b20af](charts/b20af-fc100x200x100seed2.png)
**b20af-fc100x200x100seed2**

![b20ag](charts/b20ag-fc100x200x100seed3.png)
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

![b20aa](charts/b20aa-fc93x93seed1.png)
**b20aa-fc93x93seed1**

![b20ad](charts/b20ad-fc93x93seed4.png)
**b20ad-fc93x93seed4**

![b20ab](charts/b20ab-fc93x93seed2.png)
**b20ab-fc93x93seed2**

![b20ac](charts/b20ac-fc93x93seed3.png)
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

![b20n](charts/b20n-fc320seed2.png)
**b20n-fc320seed2**

![b20p](charts/b20p-fc320seed4.png)
**b20p-fc320seed4**

![b20m](charts/b20m-fc320seed1.png)
**b20m-fc320seed1**

![b20o](charts/b20o-fc320seed3.png)
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

![b20i](charts/b20i-fc200x50seed1.png)
**b20i-fc200x50seed1**

![b20l](charts/b20l-fc200x50seed4.png)
**b20l-fc200x50seed4**

![b20j](charts/b20j-fc200x50seed2.png)
**b20j-fc200x50seed2**

![b20k](charts/b20k-fc200x50seed3.png)
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

![b20d](charts/b20d-fc50seed4.png)
**b20d-fc50seed4**

![b20b](charts/b20b-fc50seed2.png)
**b20b-fc50seed2**

![b20c](charts/b20c-fc50seed3.png)
**b20c-fc50seed3**

![b20a](charts/b20a-fc50seed1.png)
**b20a-fc50seed1**

### Treatment `200,100,50` (2.66× params)

Resumed from the ~1.75M crash (a since-fixed chart-writer leak, not the arms) to 3M. `b20g` is the
strongest arm of the batch (best-30 81.7%, close-out 72.2%); `b20f` is the laggard, mirroring `b20a`.

![b20g](charts/b20g-fc200seed3.png)
**b20g-fc200seed3**

![b20h](charts/b20h-fc200seed4.png)
**b20h-fc200seed4**

![b20e](charts/b20e-fc200seed1.png)
**b20e-fc200seed1**

![b20f](charts/b20f-fc200seed2.png)
**b20f-fc200seed2**
