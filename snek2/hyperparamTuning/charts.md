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
![b21d](charts/b21d-beta05seed4.png)
![b21c](charts/b21c-beta05seed3.png)
![b21a](charts/b21a-beta05seed1.png)

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
![b20t](charts/b20t-fc25x50x25seed4.png)
![b20s](charts/b20s-fc25x50x25seed3.png)
![b20q](charts/b20q-fc25x50x25seed1.png)

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
![b20x](charts/b20x-fc60x30x30x30x30seed4.png)
![b20v](charts/b20v-fc60x30x30x30x30seed2.png)
![b20u](charts/b20u-fc60x30x30x30x30seed1.png)

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
![b20ae](charts/b20ae-fc100x200x100seed1.png)
![b20af](charts/b20af-fc100x200x100seed2.png)
![b20ag](charts/b20ag-fc100x200x100seed3.png)

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
![b20ad](charts/b20ad-fc93x93seed4.png)
![b20ab](charts/b20ab-fc93x93seed2.png)
![b20ac](charts/b20ac-fc93x93seed3.png)

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
![b20p](charts/b20p-fc320seed4.png)
![b20m](charts/b20m-fc320seed1.png)
![b20o](charts/b20o-fc320seed3.png)

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
![b20l](charts/b20l-fc200x50seed4.png)
![b20j](charts/b20j-fc200x50seed2.png)
![b20k](charts/b20k-fc200x50seed3.png)

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
![b20b](charts/b20b-fc50seed2.png)
![b20c](charts/b20c-fc50seed3.png)
![b20a](charts/b20a-fc50seed1.png)

### Treatment `200,100,50` (2.66× params)

Resumed from the ~1.75M crash (a since-fixed chart-writer leak, not the arms) to 3M. `b20g` is the
strongest arm of the batch (best-30 81.7%, close-out 72.2%); `b20f` is the laggard, mirroring `b20a`.

![b20g](charts/b20g-fc200seed3.png)
![b20h](charts/b20h-fc200seed4.png)
![b20e](charts/b20e-fc200seed1.png)
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
