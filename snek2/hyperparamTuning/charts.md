# Charts

Progress graphs for the most recent batches — **28, 36, 37, 40, 42 and 43**, a cap of six, newest first.
Per-arm numbers live in [`completedRuns.md`](completedRuns.md); this file is images plus a short reading of
each. A batch appears here **while it is still running**, with training-only numbers, not just once it has
closed. Batch 27 was retired to [`archive/charts-archive.md`](archive/charts-archive.md) when 36 launched,
**batch 30** followed when 34's results arrived, **batch 31** (a void, stopped C51 arm) when 35's arrived,
**batch 33** when 38 launched, **batch 32** when 39 did, **batches 34 and 35** when 37's and 40's results
landed, and **the C51 pilot plus batch 38** when 42/43 launched, and **batch 39** when 42 got its section. 39 went ahead of the strict-oldest 28-29 for the same reason 28-29 has been held four times now: it is the source of three of the four checkpoints 42/43/44 continue, and it is what they are read against. 39 bears on none of that.

**Everything here except 43 is closed.** `b37` and `b40` are done on the desktop, training,
close-out *and* HOF-500. **`b43` is training on the laptop**, at +541-562k past its seed checkpoints, with a
close-out and HOF-500 chained behind it. **`b42` was stopped early** at +385-421k — its answer had resolved on
4 of 4 seeds — and its close-out is running on the desktop now, HOF-500 queued behind it. **`b44` — the same
four checkpoints at `lr 1e-7`** — is queued on the desktop behind that eval chain and has no chart yet. The gate ladder's two null rungs (34, 70 and 35, 40) were retired earlier,
since the ladder's conclusion is now carried by 28-29, 37 and 40 — the batches that bear on whether gate 75's
record region was real.

**`b37` and `b39` are not a numbering slip:** `b37` was queued from the desktop the same evening `b38` was
launched from the laptop, so the two hosts took adjacent numbers out of order. **`b41` has no section here
either** — it is the b29 same-seed determinism probe, still closing out on the desktop.

**Batch 42 and 43 arms do not start at step 0**, so their curves begin mid-chart at the step of the checkpoint
each one continues. There is no resume line at the left edge because the graph history was deliberately not
carried over: these are fresh policy dirs seeded from one checkpoint, not resumes of the source arm.

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
`scripts/refresh_charts.sh`, which re-copies every `runs/*.png` into `charts/` and prints each one's step.

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

**Six PNGs in `charts/` are not arm charts and will always appear in that list** —
`champion-vs-mediocre`, `drawdown-b23b-vs-b18`, `per-b18-vs-b20-priorities`, `plasticity-metrics`,
`best30-drivers` and `gate-behavior-b27-vs-b29` are diagnostic figures referenced from
[`findings.md`](findings.md) and [`perDiagnostics/`](perDiagnostics/README.md), not training graphs.
Anything *else* the check prints is a real gap.

## Batch 43 — **continuing the four best checkpoints at `lr 1e-6`** — *training on the laptop, +541-562k past seed: it holds where `1e-5` decays, 4 of 4 seeds*

**The question:** every record in this project is a checkpoint some arm *passed through* on its way to a worse
endpoint. No arm has ever been continued from its own best checkpoint. `b43` continues the top four of eight
across b29 and b40 — ranked by their best **500-episode** perfect rate — to a 3M absolute cap under b29's
config (`fc 320`, chase-safe `c=0.10` gate 75, no free-space term), with **`SNEK_LEARNING_RATE=1e-6`** the only
change. The desktop's **`b42`** runs the identical four at the default `1e-5` as the seed-matched control.
Rationale, the pre-registered outcome readings and the selection-bias warning are in
[`runs.md`](runs.md#batches-42-43-and-44--what-happens-if-you-keep-training-a-champion--b42-answered-it-and-was-stopped-b43-running-b44-queued);
launcher [`scripts/launch_b43_lowlr.sh`](scripts/launch_b43_lowlr.sh), seeding
[`scripts/seed_from_checkpoint.sh`](scripts/seed_from_checkpoint.sh).

| arm | continues | from | its 500-ep rate there | training-self-eval so far | `b42` best-30 / `sef` |
|---|---|---|---|---|---|
| `b43b` | `b29a-chase10g75seed1` | @1347k | 98.4% | best-30 **100.0** @1667k, `sef` 99.6, recent-30 95.0 | its `b42` twin: 97.3 / 97.9 |
| `b43a` | `b29b-chase10g75seed2` | @1447k | **99.0%** (the project record) | best-30 98.7 @1527k, `sef` 99.6, recent-30 91.7 | 97.3 / **87.0** |
| `b43c` | `b40b-chasefree10g75seed2` | @1513k | 98.2% | best-30 98.3 @1701k, `sef` **99.8**, recent-30 95.3 | **93.3** / 91.2 |
| `b43d` | `b29c-chase10g75seed3` | @1396k | 97.1% (378 ep, gate-abandoned) | best-30 98.3 @1539k, `sef` 97.9, recent-30 94.7 | 97.7 / 95.9 |

**Reading at +541-562k: all four hold, and one has improved on its start.** Every arm sits at `sef`
**97.9-99.8** with `max_single_eval` 100 and `zero_since` null, and **`b43b` reached a `best_perfect30` of
100.0 at 1667k — 320k steps past its own seed checkpoint**, a 30-eval window with no imperfect game in it. So
`1e-6` is not merely preservation. That nothing fell apart is itself a result: **the same four checkpoints fell
80% → 50% perfect in 5k steps when the replay buffer was *not* carried over**, and holding 95-100% here is what
the buffer copy bought.

**Against `b42`, the seed-matched `1e-5` control, `b43` leads on 4 of 4 seeds on every metric** — `best_perfect30`
98.3-100.0 against 93.3-97.7, `sef` 97.9-99.8 against 87.0-97.9, and banded perfect rate +4.7/+7.1/+6.1 pp over
the 100-200k, 200-300k and 300-385k bands past seed. The full banded table is in
[batch 42's section below](#batch-42--the-same-four-checkpoints-at-the-default-lr-1e-5--stopped-early-on-the-desktop-at-177-191m-it-decays-and-that-is-the-answer),
which is where the pair's result lives since `b42` is the arm that moved. **`b44`** now extends the ladder to
**`1e-7`** on the same four checkpoints, queued on the desktop.

**Do not read these best-30 numbers against the 500-episode column beside them.** A 10-episode self-eval
averaged over 30 evals and a flat 500-episode measurement are different instruments, and the 500-ep column is
additionally **the maximum of a noisy statistic over 8 arms and hundreds of checkpoints** — biased upward. The
comparison that means something is `b43` against `b42`, which starts from byte-identical weights. Everything
here is training self-eval; the close-outs will be the real numbers.

**`sef` is not comparable with any other batch in this file.** It is the share of an arm's *own* evals above
80% perfect, and these arms started at ~98% instead of 0 — so 99.8 says "it never dropped", not "it learned
fast". Only `b42`'s `sef` is a fair reference, and it is in the last column of the table above; `b44`'s will be
the other one. **`b42` and `b43` are also not at a common step horizon** (`b42` was stopped at +385-421k,
`b43` is past +541k) and `sef` is a fraction of each arm's own evals, so the banded comparison — not this
column — is the rigorous one.

### b43b-lowlr-b29a — continues `b29a` @1347k, 98.4%/500

Best-30 **100.0** at 1667k, `sef` **99.6**, recent-30 95.0, `max_single_eval` 100. **The strongest evidence in
the pair that `1e-6` improves rather than merely holds** — a 30-eval window with no imperfect game, 320k steps
past its seed, against 97.3 for its byte-identical `b42` twin. On 10-episode evals, so the close-out is what
decides it; but a clean 30-window is not something the starting checkpoint produced.

![b43b](charts/b43b-lowlr-b29a.png)

### b43a-lowlr-b29b — continues `b29b` @1447k, **99.0%/500, the project record**

Best-30 **98.7** at 1527k, `sef` **99.6**, recent-30 91.7 — against 97.3 / **87.0** for `b42a` from
byte-identical weights, the widest `sef` gap of the four. The arm that matters most: it starts from the head
of b29b's 18-checkpoint band, and [that band is now known to be a
seed rather than a config](findings.md#-corrected-2026-08-18-the-record-region-does-not-replicate--the-98500-count-is-seed-noise-and-pooled-is-the-only-metric-of-this-family-worth-reading),
so whether the region can be *extended by training inside it* is the open question `b42`/`b43` exist to answer.

![b43a](charts/b43a-lowlr-b29b.png)

### b43c-lowlr-b40b — continues `b40b` @1513k, 98.2%/500

Best-30 **98.3** at 1701k, `sef` **99.8**, recent-30 95.3 — its `b42` twin is the weakest arm of the eight at
best-30 **93.3**. **The one arm with a confound worth stating:** `b40b`
was trained *with* the free-space PBRS term and is being continued *without* it, because the batch pins b29's
config for all four. PBRS leaves the optimal policy unchanged but the value function absorbs the potential, so
this arm's restored `Q` is mis-calibrated against its new reward in a way the other three are not. It is the
same change on the `b42` side, so the host comparison stays clean — but do not read `b43c` against `b43a/b/d`
as though only the seed differed.

![b43c](charts/b43c-lowlr-b40b.png)

### b43d-lowlr-b29c — continues `b29c` @1396k, 97.1% over 378 episodes

Best-30 **98.3** at 1539k, `sef` 97.9, recent-30 94.7. The weakest starting point of the four and still the
weakest arm of the `b43` four, which is the expected ordering — though it has closed most of the gap, and its
`b42` twin is the *best* of the `b42` four, making this the narrowest pair (+1.7 pp over the matched window). Its starting figure is **not** a 500-episode number — the row was
abandoned by the 98% gate at 378 episodes — so it is not strictly comparable with the three above it.

![b43d](charts/b43d-lowlr-b29c.png)

## Batch 42 — the **same four checkpoints at the default `lr 1e-5`** — *stopped early on the desktop at 1.77-1.91M: it decays, and that is the answer*

`b43`'s seed-matched control, and the reason the pair exists. Identical in every respect — same four seeded
policy dirs, same source checkpoints, same seeds, byte-identical env on all nine of b29's knobs — except the
learning rate is left at the default **`1e-5`**, the rate these checkpoints were originally trained at.
**Stopped by hand after ~400k steps past resume** rather than run to its 3M cap, because by then the
comparison had resolved 4 of 4 and the arms were spending compute going downhill. Close-out and HOF-500 were
queued explicitly so stopping cost no measurement.

| arm | continues | from | `best_perfect30` | `sef` | recent-30 | steps past seed |
|---|---|---|---|---|---|---|
| `b42d` | `b29c` | @1396k | 97.7 | 95.9 | 90.0 | +410k |
| `b42a` | `b29b` | @1447k | 97.3 | **87.0** | 87.0 | +385k |
| `b42b` | `b29a` | @1347k | 97.3 | 97.9 | 92.7 | +421k |
| `b42c` | `b40b` | @1513k | **93.3** | 91.2 | 86.0 | +398k |

**The result: at `1e-5` a champion loses about 5 pp of perfect rate and settles at a lower plateau.** Banded
mean perfect rate over the **+385k window matched across all eight `b42`/`b43` arms** — the largest window
where every arm has data:

| band past seed | `b42` (1e-5) | `b43` (1e-6) | diff | seeds favouring 1e-6 |
|---|---|---|---|---|
| 0-100k | 93.6 | 95.4 | +1.8 | 3 of 4 |
| 100-200k | 91.6 | 96.3 | **+4.7** | **4 of 4** |
| 200-300k | 89.1 | 96.2 | **+7.1** | **4 of 4** |
| 300-385k | 89.5 | 95.6 | **+6.1** | **4 of 4** |

It is a **one-time drop to a lower level, not an ongoing collapse** — the last two bands are flat at ~89, so
the arm re-equilibrates rather than unravelling. `best_perfect30` and `sef` agree on all four seeds
(93.3-97.7 against 98.3-100.0, and 87.0-97.9 against 97.9-99.8). Read against
[the record region being seed noise](findings.md#-corrected-2026-08-18-the-record-region-does-not-replicate--the-98500-count-is-seed-noise-and-pooled-is-the-only-metric-of-this-family-worth-reading),
the reading is that the default rate is simply too large to *sit still* in a narrow high-performing basin: the
step size that found the basin is bigger than the basin.

**⚠ Do not judge these arms with `peak_trailing`.** It is trailing average *score*, which maxes at 95/95, and
**all eight `b42`/`b43` arms read exactly 95.0 with the peak timestamped at their own seed step** — the metric
is saturated from the first eval, because a policy restored from a 98% checkpoint already fills the board. An
earlier pass here misread that as "no arm ever beat its starting checkpoint", which is false: `b43b` reached a
**100.0** best-30 window 320k steps past its seed. Use `perfect_percent`, `best_perfect30` or `sef`.

### b42a-cont3m-b29b — continues the project record, `b29b` @1447k (99.0%/500)

`sef` **87.0**, the worst of the eight arms in this pair of batches, against **99.6** for `b43a` from
byte-identical weights. The arm that starts highest falls furthest — which is what a basin-too-narrow reading
predicts, since the record checkpoint sits at the head of b29b's 18-checkpoint band.

![b42a](charts/b42a-cont3m-b29b.png)

### b42c-cont3m-b40b — continues `b40b` @1513k (98.2%/500)

`best_perfect30` **93.3**, the lowest of the four. Carries the same free-space confound as `b43c` — trained
*with* the global free-space term and continued *without* it — so its restored value function is
mis-calibrated against its new reward. The confound is identical on the `b43` side, so the pairwise comparison
holds.

![b42c](charts/b42c-cont3m-b40b.png)

### b42b-cont3m-b29a — continues `b29a` @1347k (98.4%/500)

`sef` 97.9, the best-holding of the four and the smallest gap to its `b43` twin (+1.7 pp over the whole matched
window). Went furthest past its seed, +421k.

![b42b](charts/b42b-cont3m-b29a.png)

### b42d-cont3m-b29c — continues `b29c` @1396k (97.1%, 378 ep)

`best_perfect30` **97.7**, the highest of the four, from the weakest starting checkpoint — its start is a
gate-abandoned 378-episode row, not a 500-episode number, so it is not strictly comparable with the others.

![b42d](charts/b42d-cont3m-b29c.png)

## Batch 40 — chase-safe **plus a global free-space term** — *done on the desktop: null, and it makes b29's record region look like seed luck*

**`b29`'s config with `SNEK_FREE_SPACE_SHAPING` added on top** — `Φ = 1 / (number of open regions)`, the tail
cell freed before the count, *added to* chase-safe rather than replacing it (PBRS terms sum). `fc 320`,
`c=0.10`, **gate 75**, IS off, `td_error`, 2M, seeds 1-4. The hypothesis was that an explicit
"don't cut the board in two" signal would carry the last few meals, where the chase-safe potential is
measurably exhausted (98-99 carries 0.00-0.04).

| arm | best-30 | `sef` | pooled/eq | ≥98%/100 | held ≥98%/500 |
|---|---|---|---|---|---|
| `b40b` | **97.7** | **62.7** | 89.52 | **63** | **1 — 98.2% @1513k** |
| `b40c` | 95.3 | 61.5 | 89.11 | 9 | 0 |
| `b40a` | 96.3 | 45.0 | 88.28 | 16 | 0 |
| `b40d` | 94.0 | 56.8 | 85.68 | 2 | 0 |
| **group** | | | **88.15** | **90, 4 of 4 seeds** | **1** |

**Two arms produced a flawless 100.0%/100 checkpoint** — `b40a` @1562k and `b40b` @1424k — and every seed
reached the ≥98%/100 tier, which no unshaped batch has done.

**‡ But against `b29` it is a null, and the /100 agreement is what makes that convincing.** b40's ≥98%/100
counts are **16 / 63 / 9 / 2** against b29's **59 / 64 / 9 / 1** — nearly the same distribution, same
4-of-4 shape — and pooled ties the family (**88.15** vs b29 87.83, b35 88.20, b34 86.43). The free-space term
moved neither metric.

**Where they diverge is the tier that turns out to be noise.** b29 held **21** checkpoints at ≥98%/500; b40
holds **1**. And `b37`, an exact `b29` replication on fresh seeds, holds **0** — see below. So the honest
reading is that **the ≥98%/500 band is seed-dependent, not config-dependent**: three batches with
indistinguishable /100 tiers produced 21, 1 and 0 held checkpoints. **90 ≥98%/100 checkpoints in b40 yielded
one that survived 500 episodes**, which is the attrition rate to expect, and `b40b` @1513k (**98.2%/500**) is
a HOF-promotion candidate — third-best /500 on record behind `b29b`'s 99.0 and `b29a`'s 98.4.

![b40a](charts/b40a-chasefree10g75seed1.png)
**b40a-chasefree10g75seed1** — 100.0%/100 @1562k, 16 at ≥98%/100, none held at 500

![b40b](charts/b40b-chasefree10g75seed2.png)
**b40b-chasefree10g75seed2** — the batch's arm: 100.0%/100 @1424k, 63 at ≥98%/100, **98.2%/500 @1513k**

![b40c](charts/b40c-chasefree10g75seed3.png)
**b40c-chasefree10g75seed3** — pooled 89.11, 9 candidates, none held

![b40d](charts/b40d-chasefree10g75seed4.png)
**b40d-chasefree10g75seed4** — the weak seed, as in every batch of this family

## Batch 37 — **`b29` replicated on fresh seeds 5-8** — *done on the desktop: the /100 band replicates, the /500 record does not*

**Byte-identical to `b29` except `SNEK_SEED`** — `fc 320`, chase-safe `c=0.10`, **gate 75**, IS off,
`td_error`, 2M. Queued because b29's 21-checkpoint ≥98%/500 band rested on **2 of its 4 seeds**, and a
record region that depends on the seed is a different claim from one that depends on the config.

| arm | best-30 | `sef` | pooled/eq | ≥98%/100 | held ≥98%/500 |
|---|---|---|---|---|---|
| `b37b` | **97.7** | 56.0 | **90.50** | **43** | 0 — best 97.0%, abandoned at 361 ep |
| `b37c` | 97.0 | **58.6** | 87.88 | 16 | 0 — best 96.9%, abandoned at 357 ep |
| `b37a` | 91.3 | 49.8 | 82.19 | 0 | 0 |
| `b37d` | 90.0 | 40.8 | 80.72 | 0 | 0 |
| **group** | | | **85.32** | **59, 2 of 4 seeds** | **0 of 4** |

**The 2-of-4 pattern replicates exactly** — two strong seeds carrying the ≥98%/100 tier, two contributing
nothing — and `b37b`'s pooled **90.50** is the highest single arm in the chase-safe family. **The record tier
does not replicate at all:** 59 candidates at ≥98%/100 produced **zero** that held 98% over 500 episodes, with
the two best abandoned around 360 episodes at 96.9-97.0%.

**Read this together with `b40` above.** The same config, fresh seeds, gives 0 where b29 gave 21; a different
config with an added shaping term gives 1. **The ≥98%/500 count is the noisiest metric in this project** and a
single batch's value should not be treated as a property of its config. Pooled equal-effort is what
distinguishes these batches, and on that b37 (85.32) is the *weakest* of the family despite holding its best
single arm.

![b37a](charts/b37a-chase10g75seed5.png)
**b37a-chase10g75seed5** — seed 5, no ≥98%/100 checkpoint at all

![b37b](charts/b37b-chase10g75seed6.png)
**b37b-chase10g75seed6** — strongest arm of the family on pooled (90.50), yet 0 held at 500

![b37c](charts/b37c-chase10g75seed7.png)
**b37c-chase10g75seed7** — highest `sef` of the batch (58.6), 16 candidates, 0 held

![b37d](charts/b37d-chase10g75seed8.png)
**b37d-chase10g75seed8** — the weak seed

## Batch 36 — **C51 on `fc 320`**, one wide layer instead of three narrow — *stopped at 1.87-2.02M, closed out*

**Batch 32's config verbatim at `eps 1.5e-4` with `SNEK_FC_LAYERS=320` the only change**, seeds 1-4, win
reward back at its default 100, `lr 1e-4`, 51 atoms over `[-5, 120]`. Launched 12:43 on 2026-08-16;
launcher [`launch_c51_fc320.sh`](scripts/launch_c51_fc320.sh), rationale and pre-registered hypotheses in
[`runs.md`](runs.md).

**Two controls, both on disk, answering different questions.** `b32a`/`b32b` — same `eps`, `lr` and seeds
at `fc 200,100,100` — is the clean one-variable *architecture* pair, but only 1M deep, so **match at 1M
before quoting anything**. `b24a-d` is **ddqn** at this exact shape, **3M** and closed out at pooled
**85.97-89.03** with two ≥98%/500 records, which is the "is C51 worth it at all" comparison — **and b36
did not clear it**, see the reading below.

**What the shape changes for a categorical head is *where* the parameters sit, not how many.** Obs 30 to
3×51 = 153 outputs:

| shape | first layers | final layer | total | share in the final layer |
|---|---|---|---|---|
| **`fc 320`** | 9,920 | **48,960** | ~58.9k | **83%** |
| `fc 200,100,100` | 36,400 | 15,453 | ~51.9k | 30% |

Only +13% capacity, but the budget moves into the layer feeding the 153-way distribution, and two layers
of gradient compounding disappear. **That second half is why churn is the reading, not level** — C51's
defect here is instability.

**The pre-registered expectation is a null on the ceiling**, because nine shapes have never raised it and
the one direct measurement says the deeper net's penultimate layer was *not* capacity-bound (effective rank
16-20 of 100, head outputs 4-6 of 153). **If that rank comes out at 16-20 of 320 here too, widening bought
nothing** and any gain is optimisation rather than capacity.

**Stopped at 2M rather than 3M** (21:26 on 2026-08-16, the user's call), so **match at 2M**. Close-out
complete at gate 95, 4 parallel processes at `EVAL_WORKERS=4`:

| arm | step | best-30 | at | `sef` | trailing | pooled /eq | best ckpt |
|---|---|---|---|---|---|---|---|
| `b36d` | 1873k | **86.7** | 331k | 17.4 | 92.64 | **80.19** | 95.0 @356k |
| `b36b` | 1972k | 86.0 | 402k | **24.7** | 91.14 | 76.70 | 94.0 @305k |
| `b36c` | 2011k | 84.7 | 247k | 19.5 | **93.14** | 74.77 | 91.6 @471k *[83 ep]* |
| `b36a` | 2023k | 84.0 | 977k | 23.2 | 92.02 | 75.36 | **97.0 @550k** |
| *`b32a`/`b32b` control, `fc 200,100,100`, ≤1M* | 1000k | *77.0 / 63.0* | | *16.1 / 10.3* | | *never closed out* | |
| **`b24a-d` control, `ddqn` at this same `fc 320`, 3M** | 3000k | **95.3-96.7** | | **60.5-73.2** | | **85.97-89.03** | **98.0-100.0** *(≤2M rows)* |

**Hypothesis 2 beat the null against `b32`, and the batch still loses badly to `b24`.** Both readings are
real and they answer different questions:

- **Against `b32` (the C51 architecture question): a clear gain.** Best-30 **84.0-86.7 against 77.0/63.0**,
  `sef` 17.4-24.7 against 10.3/16.1, and the **seed spread collapsed from 14 pp to 2.7 pp**. Wide-shallow
  is the better C51 shape.
- **Against `b24` (the "is C51 worth it" question): no.** Same `fc 320`, same observation era, `ddqn`:
  best-30 **95.3-96.7**, `sef` **60.5-73.2** — a factor of 3, and `sef` is the low-variance metric — and
  **every b24 seed produced a ≥98% checkpoint inside 2M** where no b36 seed did. The seed-agreement gain
  also does not generalise: `b24`'s spread is **1.4 pp**, tighter than b36's. Full comparison and its
  caveats in [`findings.md`](findings.md#-after-four-fixes-c51-is-still-well-behind-the-scalar-head-at-its-own-architecture--and-b24-not-b32-is-the-control).

**The 3M question went unanswered** — every arm peaked best-30 by 402k except `b36a` (977k) while trailing
stayed 91-93 to 2M, so it neither collapsed like b33 nor kept climbing. **`b36c`'s best row is 83 episodes,
not 100** — abandoned under `EVAL_MIN_ACHIEVABLE=95`, so `best_of` relaxed to half-depth and it is not
comparable with the full-length rows.

**Init optimism is now measured and excluded as a C51 explanation.** Every arm started at `V ≈ 57.5`, the
grid midpoint, against a true value of **~34** — but the offset is common-mode (the action gap is at full
scale by 8k), it costs the policy nothing, and the `ddqn` control's init at 0 is *further* from the truth.
[The finding](findings.md#-a-c51-head-starts-at-the-grid-midpoint-not-at-0--and-it-cost-b36-nothing-because-the-ddqn-controls-init-is-further-from-the-truth).

![b36a](charts/b36a-c51fc320seed1.png)
**b36a-c51fc320seed1** — paired with `b32a`

![b36b](charts/b36b-c51fc320seed2.png)
**b36b-c51fc320seed2** — paired with `b32b`

![b36c](charts/b36c-c51fc320seed3.png)
**b36c-c51fc320seed3**

![b36d](charts/b36d-c51fc320seed4.png)
**b36d-c51fc320seed4**

## Batches 28-29 — chase-safe **dose** (`c=0.20`) and **gate** (`75`) on `fc 320` — *done: the gate is the lever (desktop)*

Both extend b27's gate-85 null onto the two axes a single dose could not test. `b28` doubles the coefficient
to **`c=0.20`** at gate 85; `b29` drops the **gate to 75** at `c=0.10`. Everything else is b24's config —
`fc 320`, IS off, `td_error`, target 1000, discount 0.9975, `FORK_BRANCHES=4`, 2M cap, seeds 1-4, the same
`b24a-d` control. All eight closed out and HOF-500'd on the desktop.

**`b28` (`c=0.20`, gate 85) is a null — the dose is not the issue.** Pooled mean **85.4**, ~2.5 under the b24
control's 87.9, and **0 of 4** seeds held ≥98%/500. With b27/b30 this rules out both the net and the dose at
gate 85.

| arm | best-30 | `sef` | pooled (eq) | HOF-500 |
|---|---|---|---|---|
| `b28d-chase20g85seed4` | 96.7 | 68.7 | 89.10 | best 96.8% @1061k (341 ep, ab.) — **0 held** |
| `b28a-chase20g85seed1` | 96.0 | 54.0 | 89.72 | best 95.6% @1727k (275 ep, ab.) — **0 held** |
| `b28c-chase20g85seed3` | 92.7 | 33.4 | 82.67 | none reached the gate — **0 held** |
| `b28b-chase20g85seed2` | 90.7 | 47.9 | 80.15 | best 90.0% @1127k (120 ep, ab.) — **0 held** |

**`b29` (`c=0.10`, gate 75) produced a record region — the positive result of the whole investigation.**
Pooled **87.8** (a dead heat with b24) but **21 checkpoints held ≥98%/500 across two of four seeds**, where
the record-holding control produced only 2 isolated ones. `b29b` carries an **18-checkpoint band**
(1446k-1529k) peaking at **99.0%/500 (495/500) @1447k** — **the new project record** (point estimate above
b24's 98.0%/500; lead inside the CI, but the *region* is not).

| arm | best-30 | `sef` | pooled (eq) | HOF-500 (≥98%/500) |
|---|---|---|---|---|
| `b29a-chase10g75seed1` | 97.7 | 60.3 | 89.76 | **3 held**, best 98.4% @1347k |
| `b29c-chase10g75seed3` | 96.3 | 67.9 | 89.68 | 0 held (best 97.1%, 378 ep ab.) |
| `b29b-chase10g75seed2` | 97.3 | 55.6 | 87.14 | **18 held** (1446k-1529k), best **99.0% @1447k** |
| `b29d-chase10g75seed4` | 94.3 | 59.7 | 84.75 | 0 held (best 90.6%, 127 ep ab.) |

**The gate is the lever, not the dose or the net.** Gate 85 is null on `fc 320` (b27), on `fc 200,100,100`
(b30) and at doubled dose (b28); gate 75 matches the control's pooled *and* produces a record region it never
did. The Φ calibration is why — the potential carries ~0 at lengths 98-99, so a gate-85 term grades the flat
final approach while gate 75 turns it on ten meals earlier, in the packing decisions that decide whether the
endgame is winnable. Full write-up:
[`completedRuns.md`](completedRuns.md#batches-28-29--chase-safe-dose-and-gate-the-gate-is-the-lever-and-gate-75-produces-a-record-region).
`b29b` @1447k was promoted to `hallOfFame/` on 2026-08-16 (copy verified 98/100 on fresh laptop episodes).

![b29b](charts/b29b-chase10g75seed2.png)
**b29b-chase10g75seed2 — 99.0%/500 @1447k, the record-region arm**

![b29a](charts/b29a-chase10g75seed1.png)
**b29a-chase10g75seed1 — 3 held ≥98%/500**

![b29c](charts/b29c-chase10g75seed3.png)
**b29c-chase10g75seed3**

![b29d](charts/b29d-chase10g75seed4.png)
**b29d-chase10g75seed4**

![b28a](charts/b28a-chase20g85seed1.png)
**b28a-chase20g85seed1** (`c=0.20`, null)

![b28b](charts/b28b-chase20g85seed2.png)
**b28b-chase20g85seed2** (`c=0.20`, null)

![b28c](charts/b28c-chase20g85seed3.png)
**b28c-chase20g85seed3** (`c=0.20`, null)

![b28d](charts/b28d-chase20g85seed4.png)
**b28d-chase20g85seed4** (`c=0.20`, null)
