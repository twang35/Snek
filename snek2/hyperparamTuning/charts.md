# Charts

Progress graphs, **batch 12 onward**, newest first. Per-arm numbers live in
[`completedRuns.md`](completedRuns.md); this file is images plus a short reading of each.
Batch 1-11 captions moved to [`archive/batches1-11.md`](archive/batches1-11.md) — the PNGs are all
still in `charts/`.

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
because a successful `refresh_charts.sh` looked like the charts were handled. Check both this file
and the archive, since captions now live in two places:

```
cd snek2/hyperparamTuning
ls charts/*.png | sed 's|.*/||;s|\.png||' | sort > /tmp/have
grep -ho 'charts/[a-zA-Z0-9-]*\.png' charts.md archive/batches1-11.md \
  | sed 's|charts/||;s|\.png||' | sort -u > /tmp/doc
comm -23 /tmp/have /tmp/doc   # anything listed is an undocumented arm
```

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

## Batch 14 — `DISCOUNT=0.9975` at `GUIDED_FRACTION=0.8`, and a third null

Four seeds run to 4.1-4.5M, the longest arms on the current vector. **Null against batch 13 on every
metric** — `pooled_equal_effort` 72.08% against 72.07%, best checkpoint +2.8 pp at p=1.000, and
`strong_eval_fraction` +2.1 pp with per-seed diffs from -16.2 to +24.8. Full write-up:
[`completedRuns.md`](completedRuns.md#batch-14--disc-09975-at-guided-08-and-the-widest-seed-spread-yet).

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
[`hyperparamTuning.md`](hyperparamTuning.md#taking-the-arm-level-pooled-rate).

### b14d-disc9975seed4 — disc 0.9975 + shield 0.8, seed 4

![b14d](charts/b14d-disc9975seed4.png)

Step 4.46M · peak trailing **94.9** (at 2554k) · **best 30-eval perfect 89.7%** (at 2700k) · `strong_eval_fraction` **39.3%** · trailing-30 at stop 78.0%

**The strongest arm the project has recorded on the primary metric** — 39.3% of its evals at ≥80%,
against a previous best of 30.5% (`b11b`). It is also the flattest good arm here: its mean perfect
rate climbed monotonically through 2.5-3.0M (peaking at 81.1% per 500k band) and never fell below
67% afterwards, for only 11.7 pp of drawdown from peak to stop.

Its best checkpoint is 93% @2559k, below `b14a`'s 96%, which is the usual split between an arm that
holds a high level and an arm that spikes once.

### b14a-disc9975seed1 — disc 0.9975 + shield 0.8, seed 1

![b14a](charts/b14a-disc9975seed1.png)

Step 4.17M · peak trailing 94.8 (at 3794k) · best 30-eval perfect 79.7% (at 3707k) · `strong_eval_fraction` 20.0% · trailing-30 at stop 54.7%

**Produced a 96/100 checkpoint at 3702000, tying `b11b` for the best selected measurement on
record** — and then gave back 25 pp by the time it stopped. Both facts are the arm: its peak window
is the *latest* of any arm on this vector, and it was already falling apart 400k later.

The 96% does not survive a second look at full strength. An independent 100-episode re-measurement
of the same checkpoint read **91/100**, so the honest pooled estimate is **187/200 = 93.5%** (CI
89.2-96.2). That gap is the winner's curse made visible — this checkpoint was the maximum over 176
attempted full-length measurements in this arm.

### b14c-disc9975seed3 — disc 0.9975 + shield 0.8, seed 3

![b14c](charts/b14c-disc9975seed3.png)

Step 4.16M · peak trailing 94.8 (at 2105k) · **best 30-eval perfect 87.7%** (at 4135k) · `strong_eval_fraction` 17.8% · trailing-30 at stop 82.0%

**The arm that moved the step cap.** Its best 30-eval window is at 4135k — the last one it ran — and
its final 4.0-4.5M band is its strongest of the whole run at 75.9% mean perfect against a 62.6%
previous best. It was still improving when it was stopped, and a 5M cap would have cut it mid-climb.

Only 5.7 pp of drawdown, the smallest in the batch, for the same reason: it never peaked.

### b14b-disc9975seed2 — disc 0.9975 + shield 0.8, seed 2

![b14b](charts/b14b-disc9975seed2.png)

Step 4.12M · peak trailing 94.5 (at 2053k) · best 30-eval perfect 76.3% (at 2282k) · `strong_eval_fraction` **9.3%** · trailing-30 at stop 29.3%

Weakest of the batch and the clearest decay curve on the current vector: peaked at 2.05M, then lost
**47 pp** of perfect rate over the next 2M steps. Its epsilon reads 0.0064 at stop against its
siblings' 0.002-0.0036, which is the anti-ratchet buying exploration back in response — working as
designed, and not enough to arrest the slide.

It is also the arm that shows what the 90% gate costs: **one** full-length row survived the whole
close-out, so it has a best checkpoint (90%) and no meaningful top-3.

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

![b13c](charts/b13c-shieldseed3.png)

Step 3.67M · peak trailing 94.8 (at 3185k) · **best 30-eval perfect 85.3%** (at 2864k) · `strong_eval_fraction` **26.5%** · trailing-30 at stop 72.3%

Best of the batch, and the arm that inverts batch 11's seed ordering: seed 3 was batch 11's weakest
at 73.0% and is batch 13's strongest at 85.3%. Same seed, same config but the schedule — a +12.3 pp
swing that means nothing on its own and is exactly why the batch mean is what gets reported.

Still near its peak at 3.19M when stopped, with the smallest gap in the batch between peak trailing
and where it ended.

### b13d-shieldseed4 — handover 0.0125 + shield, seed 4

![b13d](charts/b13d-shieldseed4.png)

Step 3.51M · peak trailing 94.5 (at 980k) · best 30-eval perfect 83.3% (at 1005k) · `strong_eval_fraction` 14.5% · trailing-30 at stop **39.0%**

Peaked earliest in the batch at ~1M and gave up **44.3 pp** by 3.5M — the largest drawdown here, and
the reason the shield cannot be credited with fixing the post-peak decline: batch 11's seed 4 was
its *most* stable arm at 5.6 pp. The paired drawdown comparison is -1.0 pp at p = 0.875, i.e.
nothing.

### b13b-shieldseed2 — handover 0.0125 + shield, seed 2

![b13b](charts/b13b-shieldseed2.png)

Step 3.70M · peak trailing 94.8 (at 1919k) · best 30-eval perfect 82.3% (at 1508k) · `strong_eval_fraction` 25.4% · trailing-30 at stop 67.0%

**The fastest start on record**: trailing 92.4 with a 72.3% perfect rate by step 350k, where batch
12's arms were at 0%. Whatever else the epsilon change did or did not do, that is the deadlock
being decisively absent.

### b13a-shieldseed1 — handover 0.0125 + shield, seed 1

![b13a](charts/b13a-shieldseed1.png)

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
[`completedRuns.md`](completedRuns.md#-the-new-schedule-deadlocks-all-four-arms-are-failing-44-at-1m-steps). These
arms are kept as the measured cost of sitting at epsilon 0.05: not a wasted batch, a negative
result with four seeds behind it.

### b12s-shield05seed1 — the exploration shield at handover 0.05, seed 1

![b12s](charts/b12s-shield05seed1.png)

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

![b12a](charts/b12a-eps002seed1.png)

Step 1.12M (stopped) · peak score 89.1, peak trailing 87.02 (at 214k) · best 30-eval perfect **6.3%** (at 213k) · max single eval 40% · not measured

The best of a bad batch, and the arm that makes the decay unambiguous. It read trailing **87.0**
with 6.3% best-30 at 214k — a genuinely promising arm — then fell to 59.6 over the next 900k steps
**at exactly the same epsilon**. Same exploration rate, worse policy, so nothing about the
measurement explains it.

41 of its 1122 evals contained a perfect game, which was enough to nudge epsilon to 0.0388 at its
best and never enough to escape: the refinement phase needs 20-40% to reach the floor, and 0.05
makes that unreachable.

### b12d-eps002seed4 — two-phase epsilon, seed 4

![b12d](charts/b12d-eps002seed4.png)

Step 1.09M (stopped) · peak score 87.2, peak trailing 86.36 (at 479k) · best 30-eval perfect 6.3% (at 29k) · max single eval 30% · not measured

The latest peak in the batch at 479k, and the only arm whose best-30 came in its first 30k steps —
during the bootstrap phase, before the ceiling took hold. Everything after is decline.

### b12c-eps002seed3 — two-phase epsilon, seed 3

![b12c](charts/b12c-eps002seed3.png)

Step 0.98M (stopped) · peak score 85.3, peak trailing 82.46 (at 360k) · best 30-eval perfect 1.7% (at 369k) · max single eval 10% · not measured

8 perfect games in 977 evals, and a max single eval of 10% — it reached the endgame often enough
to prove the policy was not hopeless and never often enough for the schedule to notice.

### b12b-eps002seed2 — two-phase epsilon, seed 2

![b12b](charts/b12b-eps002seed2.png)

Step 1.03M (stopped) · peak score 82.8, peak trailing 81.4 (at 259k) · **best 30-eval perfect 0.0%** · max single eval **0%** · not measured

**The cleanest demonstration of the deadlock in the project: zero perfect games in 1032 evals.**
Epsilon reached 0.05 at step 11000 and sat there for the remaining 942k steps, because the signal
that would have lowered it requires finishing a game and 3.3% random actions never let it. An arm
that peaked at 81.4 trailing was never once measured completing the board.

---
