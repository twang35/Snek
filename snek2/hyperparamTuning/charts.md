# Charts

Progress graphs for the most recent batches — **28, 37, 40, 42, 43 and 44**, a cap of six, newest first.
Per-arm numbers live in [`completedRuns.md`](completedRuns.md); this file is images plus a short reading of
each. A batch appears here **while it is still running**, with training-only numbers, not just once it has
closed. Batch 27 was retired to [`archive/charts-archive.md`](archive/charts-archive.md) when 36 launched,
**batch 30** followed when 34's results arrived, **batch 31** (a void, stopped C51 arm) when 35's arrived,
**batch 33** when 38 launched, **batch 32** when 39 did, **batches 34 and 35** when 37's and 40's results
landed, and **the C51 pilot plus batch 38** when 42/43 launched, and **batch 39** when 42 got its section, and **batch 36** when 44 got its. 36 was being held only as the named control for 39, and 39 is now archived itself. 39 went ahead of the strict-oldest 28-29 for the same reason 28-29 has been held four times now: it is the source of three of the four checkpoints 42/43/44 continue, and it is what they are read against. 39 bears on none of that.

**`b37`, `b40` and `b42` are fully closed** — training, close-out *and* HOF-500. **`b43` and `b44` finished
training at the 3M cap and their close-outs are running**, `b43` on the laptop since 01:07 and `b44` on the
desktop since 05:43, with HOF-500 chained behind each. Those close-outs are **slow by nature, not stuck** — see
[the cost warning](#-a-continuation-batchs-close-out-costs-10-20x-a-normal-ones) in batch 44's section. **`b42`
was stopped early** at +385-421k, once its answer had resolved on 4 of 4 seeds. The gate ladder's two null rungs (34, 70 and 35, 40) were retired earlier,
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

## Batch 44 — the **same four checkpoints at `lr 1e-7`** — *done training at the 3M cap: it beats `1e-6` on 4 of 4 seeds, so the ladder is still monotone*

The third rung, and **the rung that falsified its own pre-registration.** `b44` was queued expecting a null
against `b43` (~65% by the estimate written into its specs) on the reasoning that if `1e-6` is already doing
nothing but failing to damage the policy, `1e-7` cannot do better than the same nothing. That was wrong: the
~25% "better than `b43`" branch is what happened, on every seed.

| arm | continues | from | `best_perfect30` | `sef` | recent-30 | `b43` best-30 / `sef` |
|---|---|---|---|---|---|---|
| `b44b` | `b29a` | @1347k | **100.0** @2460k | **99.9** | 96.7 | 100.0 / 99.5 |
| `b44a` | `b29b` | @1447k | **99.7** @2190k | **99.9** | 98.3 | 98.7 / 96.5 |
| `b44c` | `b40b` | @1513k | 99.0 @1814k | 98.7 | 86.7 | 98.7 / 98.5 |
| `b44d` | `b29c` | @1396k | 98.0 @2230k | 98.7 | 96.3 | 98.3 / 96.6 |

**The ladder is monotone across all three rungs**, banded mean self-eval perfect rate over the +385k window
matched across all twelve arms:

| band past seed | `b42` 1e-5 | `b43` 1e-6 | `b44` 1e-7 |
|---|---|---|---|
| 0-100k | 93.6 | 95.4 | **96.3** |
| 100-200k | 91.6 | 96.3 | **96.6** |
| 200-300k | 89.1 | **96.2** | 96.1 |
| 300-385k | 89.5 | 95.6 | **96.8** |

**But `1e-7` vs `1e-6` is a much smaller effect than `1e-6` vs `1e-5`, and it is not uniform in time.** Over
their full common window (+1487k — both reached the 3M cap):

| band past seed | `b43` 1e-6 | `b44` 1e-7 | diff | seeds `1e-7` ahead |
|---|---|---|---|---|
| 0-250k | 95.9 | 96.3 | +0.4 | 3 of 4 |
| 250-500k | 95.4 | 96.5 | +1.1 | **4 of 4** |
| 500-750k | 92.2 | **96.6** | **+4.4** | **4 of 4** |
| 750-1000k | 93.4 | 97.0 | **+3.6** | **4 of 4** |
| 1000-1250k | 94.8 | 96.6 | +1.8 | 3 of 4 |
| 1250-1487k | 94.5 | 95.2 | +0.7 | 2 of 4 |

Per seed over the whole window: **+4.6, +1.0, +1.6, +0.8** — 4 of 4, mean **+2.0 pp**, against **+4.9 pp** for
the `1e-5` → `1e-6` step. So **the ladder is monotone but decelerating**, which is what approaching a plateau
looks like without having arrived. The gap opens widest at 500-1000k because that is where `b43` dips (92.2)
while `b44` does not — `1e-6` still wanders on the timescale of half a million steps, `1e-7` largely does not.

**The mechanism reading:** `b44`'s best-30 peaks arrive **late** — 1814k, 2190k, 2230k, 2460k, i.e. 300-1100k
steps past seed — against 1527k-2803k for `b43` and *at the seed step* for `b42`. A lower rate does not merely
preserve; it improves more slowly and for longer. That is why the "it will just freeze" prediction failed: at
`1e-7` these arms are still moving, just not far enough per step to fall out of the basin.

**Close-out and HOF-500 are running** and are what decide this — everything above is 10-episode self-eval. See
the ⚠ note in [batch 42's section](#batch-42--the-same-four-checkpoints-at-the-default-lr-1e-5--stopped-early-on-the-desktop-at-177-191m-it-decays-and-that-is-the-answer)
on why `peak_trailing` is useless for this family, and the [close-out cost
warning](#-a-continuation-batchs-close-out-costs-10-20x-a-normal-ones) below.

### b44a-lowlr7-b29b — continues the project record, `b29b` @1447k (99.0%/500)

`best_perfect30` **99.7** at 2190k, `sef` **99.9**, recent-30 98.3. **The largest single gain in the ladder:**
its `b42` twin sat at `sef` 87.0 and its `b43` twin at 96.5, so the same weights under three rates give
87.0 / 96.5 / 99.9. This is the arm that carried the project record in, and it is the one the rate matters most
for — consistent with the record checkpoint sitting in a narrow basin the default step size walks out of.

![b44a](charts/b44a-lowlr7-b29b.png)

### b44b-lowlr7-b29a — continues `b29a` @1347k (98.4%/500)

`best_perfect30` **100.0** at 2460k, `sef` **99.9**, recent-30 96.7. Ties `b43b`'s 100.0 best-30 window but
reaches it 793k steps later and holds a higher `sef`, so the two are not the same result. **Its close-out is the
long pole across both hosts** — 1196 of its checkpoints qualify for the mandatory ≥90% tier.

![b44b](charts/b44b-lowlr7-b29a.png)

### b44c-lowlr7-b40b — continues `b40b` @1513k (98.2%/500)

`best_perfect30` 99.0 at 1814k, `sef` 98.7, but **recent-30 86.7 — the one arm ending weak**, and the only one
whose `sef` is not clearly above its `b43` twin (98.7 vs 98.5). Carries the free-space confound: trained *with*
the global free-space term and continued *without* it, so its restored value function is mis-calibrated against
its new reward. The confound is identical on the `b42` and `b43` sides, so the three-way comparison holds.

![b44c](charts/b44c-lowlr7-b40b.png)

### b44d-lowlr7-b29c — continues `b29c` @1396k (97.1%, 378 ep)

`best_perfect30` 98.0 at 2230k, `sef` 98.7, recent-30 96.3. **The one seed where `1e-7` does not beat `1e-6` on
best-30** (98.0 against 98.3) though it does on `sef` (98.7 against 96.6) — the weakest starting checkpoint of
the four and the narrowest pair, as it was at every other rung.

![b44d](charts/b44d-lowlr7-b29c.png)

### ⚠ A continuation batch's close-out costs 10-20x a normal one's

`top20` measured **791, 1196, 803 and 826** checkpoints on `b43`'s four arms, not 20. That is documented
behaviour, not a bug: **"N is a target, not a quota"** — every checkpoint whose 10-episode graph eval reached
**≥90%** is measured, past N. A normal arm climbs from 0, so few checkpoints qualify; **an arm continued from a
98% checkpoint spends its entire run above the mandatory threshold, so nearly every checkpoint qualifies.**
`b42` shows the contrast from the other side — it *decayed*, so only 261-373 of its checkpoints qualified.

Practical consequence: budget a continuation batch's close-out at **8-15 hours per wave**, not one, and do not
read a close-out still running after 7 hours as hung. `b43`'s was launched 01:07 and is still going.

## Batch 43 — **continuing the four best checkpoints at `lr 1e-6`** — *done training at the 3M cap; close-out running: it holds where `1e-5` decays, but `1e-7` beats it*

**The question:** every record in this project is a checkpoint some arm *passed through* on its way to a worse
endpoint. No arm has ever been continued from its own best checkpoint. `b43` continues the top four of eight
across b29 and b40 — ranked by their best **500-episode** perfect rate — to a 3M absolute cap under b29's
config (`fc 320`, chase-safe `c=0.10` gate 75, no free-space term), with **`SNEK_LEARNING_RATE=1e-6`** the only
change. The desktop's **`b42`** runs the identical four at the default `1e-5` as the seed-matched control.
Rationale, the pre-registered outcome readings and the selection-bias warning are in
[`runs.md`](runs.md#batches-42-43-and-44--what-happens-if-you-keep-training-a-champion--the-answer-is-yes-at-a-low-enough-rate-b42-closed-b43b44-in-close-out);
launcher [`scripts/launch_b43_lowlr.sh`](scripts/launch_b43_lowlr.sh), seeding
[`scripts/seed_from_checkpoint.sh`](scripts/seed_from_checkpoint.sh).

| arm | continues | from | its 500-ep rate there | training-self-eval so far | `b42` best-30 / `sef` |
|---|---|---|---|---|---|
| `b43b` | `b29a-chase10g75seed1` | @1347k | 98.4% | best-30 **100.0** @1667k, `sef` 99.6, recent-30 95.0 | its `b42` twin: 97.3 / 97.9 |
| `b43a` | `b29b-chase10g75seed2` | @1447k | **99.0%** (the project record) | best-30 98.7 @1527k, `sef` 99.6, recent-30 91.7 | 97.3 / **87.0** |
| `b43c` | `b40b-chasefree10g75seed2` | @1513k | 98.2% | best-30 98.3 @1701k, `sef` **99.8**, recent-30 95.3 | **93.3** / 91.2 |
| `b43d` | `b29c-chase10g75seed3` | @1396k | 97.1% (378 ep, gate-abandoned) | best-30 98.3 @1539k, `sef` 97.9, recent-30 94.7 | 97.7 / 95.9 |

**All four reached the 3M cap** (+1487-1653k past seed) and the close-out launched at 01:07, still running.
Final training numbers: `best_perfect30` **98.3-100.0**, `sef` **96.5-99.5**, `max_single_eval` 100,
`zero_since` null on all four. **`b43b` reached a `best_perfect30` of 100.0 at 1667k — 320k steps past its own
seed checkpoint**, a 30-eval window with no imperfect game in it. So `1e-6` is not merely preservation.

**`b44` at `1e-7` then beat it on 4 of 4 seeds** (+2.0 pp mean over the full common window), so `b43` is the
*middle* rung, not the answer — see [batch 44 above](#batch-44--the-same-four-checkpoints-at-lr-1e-7--done-training-at-the-3m-cap-it-beats-1e-6-on-4-of-4-seeds-so-the-ladder-is-still-monotone). That nothing fell apart is itself a result: **the same four checkpoints fell
80% → 50% perfect in 5k steps when the replay buffer was *not* carried over**, and holding 95-100% here is what
the buffer copy bought.

**Against `b42`, the seed-matched `1e-5` control, `b43` leads on 4 of 4 seeds on every metric** — `best_perfect30`
98.3-100.0 against 93.3-97.7, `sef` 97.9-99.8 against 87.0-97.9, and banded perfect rate +4.7/+7.1/+6.1 pp over
the 100-200k, 200-300k and 300-385k bands past seed. The full banded table is in
[batch 42's section below](#batch-42--the-same-four-checkpoints-at-the-default-lr-1e-5--stopped-early-on-the-desktop-at-177-191m-it-decays-and-that-is-the-answer),
which is where the pair's result lives since `b42` is the arm that moved. **`b44`** extended the ladder to **`1e-7`** and won on 4 of 4 seeds.

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

## Batch 42 — the **same four checkpoints at the default `lr 1e-5`** — *stopped early at 1.77-1.91M, closed out and HOF-500'd: it decays, and its surviving ≥98% checkpoints are its own starting weights*

`b43`'s seed-matched control, and the reason the pair exists. Identical in every respect — same four seeded
policy dirs, same source checkpoints, same seeds, byte-identical env on all nine of b29's knobs — except the
learning rate is left at the default **`1e-5`**, the rate these checkpoints were originally trained at.
**Stopped by hand after ~400k steps past resume** rather than run to its 3M cap, because by then the
comparison had resolved 4 of 4 and the arms were spending compute going downhill. Close-out and HOF-500 were
queued explicitly so stopping cost no measurement, and both are now done.

**The close-out confirms it, and the HOF-500 is the sharpest form of the result.** Pooled equal-effort
**90.75 / 93.77 / 90.29 / 93.54** (mean **92.1**, gate 96). Every arm found a 99-100%/100 checkpoint, but
**the only checkpoints that survived a flat 500 episodes at ≥98% sit within 75k steps of the seed**: `b42a` one
row, **98.4% at 1453k** — 6k steps past its seed and *below* `b29b`'s own 99.0% start — and `b42d` three rows at
1399k/1457k/1470k (98.2/98.0/98.2%), 3-74k past seed. `b42b` and `b42c` produced **none at all**, from 27 and 6
candidates. So at `1e-5` the arm's ≥98% quality is the checkpoint it was handed, re-measured; nothing it earned
survives the 500-episode instrument.

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
