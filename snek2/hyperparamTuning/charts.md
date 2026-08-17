# Charts

Progress graphs for the most recent batches — **28, 32, 34, 35, 36 and 38**, a cap of six, newest first,
plus the **C51 pilot** as a temporary seventh while its arms are still a live control. Per-arm numbers live in
[`completedRuns.md`](completedRuns.md); this file is images plus a short reading of each. A batch appears
here **while it is still running**, with training-only numbers, not just once it has closed.
Batch 27 was retired to [`archive/charts-archive.md`](archive/charts-archive.md) when 36 launched, **batch 30**
followed when 34's results arrived, **batch 31** (a void, stopped C51 arm) when 35's arrived, and **batch 33**
when 38 launched.

**`b38` is running and entered below; `b36` is closed out.** The gate-ladder batches (28-29, 34, 35) are kept
contiguous for the incoming `b37` replication rather than retiring the strict-oldest of them — **batch 33** went
instead, being closed and falsified, and `b32` stays because it is still the paired control for 36 and 38.
**There is no batch 37 here:** `b37` is the desktop's b29 replication, queued the same evening as `b38` from the
other host, so the laptop's dose arm took the next free number.

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

**Six PNGs in `charts/` are not arm charts and will always appear in that list** —
`champion-vs-mediocre`, `drawdown-b23b-vs-b18`, `per-b18-vs-b20-priorities`, `plasticity-metrics`,
`best30-drivers` and `gate-behavior-b27-vs-b29` are diagnostic figures referenced from
[`findings.md`](findings.md) and [`perDiagnostics/`](perDiagnostics/README.md), not training graphs.
Anything *else* the check prints is a real gap.

## Batch 38 — **Adam ε `3.125e-4`** on b36's `fc 320` — *running, past 2.27M of 3M*

**b36's config verbatim with `SNEK_ADAM_EPSILON=3.125e-4` the only change**, seeds 1-4, 3M steps. Launcher
[`launch_b38_eps3125.sh`](launch_b38_eps3125.sh); chained behind b36's close-out by
[`chain_after_evals.sh`](chain_after_evals.sh). **The numbering skips 37 deliberately** — `b37` is the
desktop's b29 replication (seeds 5-8), queued the same evening from the other host. Rationale in
[`runs.md`](runs.md).

**What it is for.** b32 showed Adam ε cuts greedy-action churn **−26%** on a shared state set but could not
separate `1.5e-4` from `3.125e-4` at n=2 a side. b36 + b38 is **4 seeds per side on one architecture** —
the first configuration here that can resolve the dose at all. Pre-registered expectation: **a null on the
dose**, since b32's two values came out 0.0865 against 0.0895.

| arm | step | best-30 | at | `sef` | trailing | peak |
|---|---|---|---|---|---|---|
| `b38b` | 2331k | **88.3** | 294k | 17.8 | 81.2 | 94.6 @913k |
| `b38d` | 2313k | 87.3 | 571k | 22.3 | 84.1 | **95.0 @559k** |
| `b38a` | 2356k | 84.3 | 412k | **29.3** | 89.2 | 94.8 @1129k |
| `b38c` | 2266k | 80.0 | 451k | 18.6 | **93.2** | 94.6 @921k |
| *`b36a-d`, ε `1.5e-4`, at 2M* | ~2000k | *84.0-86.7* | | *17.4-24.7* | *91.1-93.1* | *94.7-95.0* |

**Reading so far: the null is holding, and the spread is worse.** Best-30 **80.0-88.3** straddles b36's
84.0-86.7 with a mean within a point, and `sef` 17.8-29.3 straddles 17.4-24.7 — but the **spread is 8.3 pp
against b36's 2.7**, so the dose did not tighten anything and may have loosened it. **Not comparable on
`sef` yet**: these arms are at 2.27-2.36M against b36's 1.87-2.02M, and `sef` is a fraction of each arm's
own evals. Match at a common step before quoting it. The dose verdict needs the close-out and a shared-set
churn reading at a common `--end`.

![b38a](charts/b38a-c51fc320eps3125seed1.png)
**b38a-c51fc320eps3125seed1** — highest `sef` of the four (29.3), peaks latest (1129k)

![b38b](charts/b38b-c51fc320eps3125seed2.png)
**b38b-c51fc320eps3125seed2** — the batch's best-30 (88.3) and its lowest trailing (81.2)

![b38c](charts/b38c-c51fc320eps3125seed3.png)
**b38c-c51fc320eps3125seed3** — the weak seed, best-30 80.0, yet the highest trailing (93.2)

![b38d](charts/b38d-c51fc320eps3125seed4.png)
**b38d-c51fc320eps3125seed4** — only arm to reach peak trailing 95.0

## Batch 36 — **C51 on `fc 320`**, one wide layer instead of three narrow — *stopped at 1.87-2.02M, closed out*

**Batch 32's config verbatim at `eps 1.5e-4` with `SNEK_FC_LAYERS=320` the only change**, seeds 1-4, win
reward back at its default 100, `lr 1e-4`, 51 atoms over `[-5, 120]`. Launched 12:43 on 2026-08-16;
launcher [`launch_c51_fc320.sh`](launch_c51_fc320.sh), rationale and pre-registered hypotheses in
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

## Batch 35 — chase-safe `c=0.10`, **gate 40** — *done on the desktop: null, and the sweet spot at 75 is isolated*

**The gate ladder's deep rung: `b29`'s config with `SNEK_CHASE_SAFE_GATE=40`, everything else identical** —
`fc 320`, IS off, `td_error`, target 1000, discount 0.9975, `FORK_BRANCHES=4`, `c=0.10`, 2M cap, seeds 1-4.
Holds the per-flip dose at 0.10 (the calibration clamp) and moves only the gate, so the total episode dose
rises ~2.5× vs gate 85.

**Null on the record metric — yet the highest pooled of any shaped batch.** Pooled equal-effort **88.2**
(88.6 / 85.9 / 90.7 / 87.6), above b29's 87.8, b34's 86.4 and the b24 control's 87.9, but **0 of the 3 measured
seeds held any ≥98%/500 checkpoint** (best partials abandoned at 96-97% over 310-367 episodes; `b35c`'s HOF-500
was still running at check time). So across four gates — 85, 75, 70, 40 — **only 75 records**; the sweet spot
is a narrow, isolated band, and mid-game shaping (40) lifts the *average* board without buying the record-tier
endgame. **Consolidation and the record tier are decoupled.** All four arms healthy throughout (peak 95.00, no
zero stretch). Full read:
[`findings.md`](findings.md#-chase-safe-reward-shaping-null-at-gate-85-at-any-dose-records-at-gate-75--the-gate-is-the-lever);
per-arm table in [`completedRuns.md`](completedRuns.md#batch-35--chase-safe-c010-gate-40-null--the-sweet-spot-at-75-is-isolated-not-a-plateau).
`sef` is on a 2M horizon, comparable to the b28/b29/b34 waves.

| arm | best-30 | `sef` | pooled (eq) | HOF-500 |
|---|---|---|---|---|
| `b35c-chase10g40seed3` | 97.7 | 59.7 | **90.7** | HOF-500 running (best full 100% @1166k /100) |
| `b35a-chase10g40seed1` | 96.0 | 47.4 | 88.6 | best 96.2% @1409k (319 ep, ab.) — **0 held** |
| `b35d-chase10g40seed4` | 96.7 | 61.3 | 87.6 | best 97.0% @1480k (367 ep, ab.) — **0 held** |
| `b35b-chase10g40seed2` | 94.7 | 62.8 | 85.9 | best 96.5% @1353k (310 ep, ab.) — **0 held** |

![b35c](charts/b35c-chase10g40seed3.png)
**b35c-chase10g40seed3** — highest pooled of any shaped batch; HOF-500 still running

![b35a](charts/b35a-chase10g40seed1.png)
**b35a-chase10g40seed1**

![b35d](charts/b35d-chase10g40seed4.png)
**b35d-chase10g40seed4**

![b35b](charts/b35b-chase10g40seed2.png)
**b35b-chase10g40seed2** — weakest seed

## Batch 34 — chase-safe `c=0.10`, **gate 70** — *done on the desktop: null, gate 75 is a narrow sweet spot*

**One variable off the record region: `b29`'s config with the gate dropped 75 → 70.** Otherwise identical —
`fc 320`, IS off, `td_error`, target 1000, discount 0.9975, `FORK_BRANCHES=4`, `c=0.10`, 2M cap, seeds 1-4,
seed-matched control `b24`. Trained, closed out and HOF-500 re-measured on the desktop.

**Gate 70 is a null — a 5-length step off 75 loses the effect.** Pooled equal-effort **86.4** (~1.5 under
b24's 87.9, just under `b29`'s 87.8) and **0 of 4 seeds held any ≥98%/500 checkpoint**, against `b29`'s 21
across two seeds. The close-out threw off two 100%/100 and two 98%/100 rows, but every one deflated below
gate 98 at 500 episodes. All four arms healthy throughout (peak trailing 95.00, no zero stretch). This
confirms gate 75 is a **band, not a threshold** — 85 is null, 75 records, 70 null again. Full read:
[`findings.md`](findings.md#-chase-safe-reward-shaping-null-at-gate-85-at-any-dose-records-at-gate-75--the-gate-is-the-lever);
per-arm table in [`completedRuns.md`](completedRuns.md#batch-34--chase-safe-c010-gate-70-null--gate-75-is-a-narrow-sweet-spot-not-a-threshold).
The `sef` numbers are on a 2M horizon and comparable to the b28/b29 waves below (also 2M), not to the 3M
batches.

| arm | best-30 | `sef` | pooled (eq) | HOF-500 |
|---|---|---|---|---|
| `b34d-chase10g70seed4` | 95.7 | 68.9 | 89.5 | best 97.2% @1915k (392 ep, ab.) — **0 held** |
| `b34c-chase10g70seed3` | 97.0 | 66.1 | 89.4 | best 96.0% @1532k (321 ep, ab.) — **0 held** |
| `b34a-chase10g70seed1` | 94.3 | 46.7 | 82.9 | best 95.2% @1126k (248 ep, ab.) — **0 held** |
| `b34b-chase10g70seed2` | 94.0 | 54.9 | 83.8 | best 93.8% @1776k (193 ep, ab.) — **0 held** |

![b34d](charts/b34d-chase10g70seed4.png)
**b34d-chase10g70seed4** — highest `sef`; best HOF-500 partial (97.2%, abandoned)

![b34c](charts/b34c-chase10g70seed3.png)
**b34c-chase10g70seed3** — highest best-30 of b34 (97.0)

![b34a](charts/b34a-chase10g70seed1.png)
**b34a-chase10g70seed1**

![b34b](charts/b34b-chase10g70seed2.png)
**b34b-chase10g70seed2** — weakest seed

## Batch 32 — **Adam's `epsilon`** on C51, `lr 1e-4`, two reference values — *closed: `epsilon` works at −26% churn, the dose does not matter*

**Final training numbers, 2026-08-16.** Both `1.5e-4` arms beat the `1e-7` control's ≤364k best-30 of 33.9,
and three of four annealed epsilon to 0.0035 where the control stayed near the ceiling:

| arm | `eps` | best-30 | at | `sef` | final ε |
|---|---|---|---|---|---|
| `b32a` | 1.5e-4 | **77.0** | 724k | **16.1** | 0.0035 |
| `b32c` | 3.125e-4 | 73.3 | 353k | 9.0 | 0.0036 |
| `b32b` | 1.5e-4 | 63.0 | 865k | 10.3 | 0.0035 |
| `b32d` | 3.125e-4 | **10.0** | 326k | 0.0 | 0.0115 |

Group means **70.0** at `1.5e-4` and **41.7** at `3.125e-4`, but `b32d` is the whole difference and n=2
cannot resolve a 2× dose — that was stated before launch and still holds. **The primary readout is churn,
and it has not been re-measured past 360k**; the table below is the 360k reading. Everything under it is
unchanged and still the right way to read this batch.

**Whether `epsilon` can separate C51's learning speed from its churn.** Adam steps by
`lr·m/(√v + ε)`, so `ε` is the gradient magnitude below which the update stops being scale-invariant —
above it a parameter moves ~`lr` per step whatever its gradient, below it a small gradient buys a small
step. We had been at Keras's default **1e-7**, where essentially everything is in the first regime, so a
coordinate carrying nothing but batch noise takes a full-size step. That matters far more for a 3×51 = 153
output categorical head than a 3-output scalar one, and the reference implementations do not use the
framework default. Full argument and the measurements behind it:
[`findings.md`](findings.md#-the-c51-arms-chaos-is-the-learning-rate-not-c51--and-the-rate-is-high-because-c51-needs-it).

| arms | `SNEK_ADAM_EPSILON` | source |
|---|---|---|
| `b32a`, `b32b` (seeds 1, 2) | **1.5e-4** | Dopamine's Rainbow config |
| `b32c`, `b32d` (seeds 1, 2) | **3.125e-4** | Dopamine's C51 config |
| *control, already on disk* | 1e-7 | `c51pilotB-lr1e4seed1/2`, same config, 600k |

`lr 1e-4` throughout, otherwise b25's config plus `ALGO=c51`. **The control is not in this wave** — the two
pilot arms at `lr 1e-4` ran it at the default, so seeds 1 and 2 are reused deliberately and the comparison
is paired at 600k with the extra 400k free. `lr 1e-4` rather than the pilot's chosen `5e-5` because that is
where the defect is largest while the rate still learns: churn **0.117-0.245** against the ddqn control's
0.033-0.058, never settling, yet seed 2 reached best-30 66.3 still rising at 599k.

**The readout is churn and drawdown depth, not `best_perfect30`** — within-rate seed spread at `1e-4` is
54.6 pp, so at n=2 per side the score resolves nothing. Measured with
[`perDiagnostics/c51_stability.py`](perDiagnostics/c51_stability.py) at `--end 600000`.

**Verdict, 2026-08-16: `epsilon` works, at about half the size first reported, and the dose does not
matter.** The re-measure to 600k also found a defect in how churn was being compared, so the numbers below
supersede the 200k/360k table this section used to carry.

**Churn per 5k steps on a *shared* 1500-state set** — drawn from `hallOfFame/b29b…ckpt1447000`, mean length
**50.5**, identical for all six arms, paired against each arm's own seed at `eps 1e-7`:

| `eps` | seed 1 | seed 2 | group | best-30 @1M | change vs own control |
|---|---|---|---|---|---|
| **1e-7** (control, `c51pilotB`) | 0.134 | 0.103 | **0.119** | 33.9 (≤364k) | — |
| **1.5e-4** (`b32a`/`b32b`) | **0.085** | **0.088** | **0.0865** | **77.0 / 63.0** | **−37% / −15%** |
| **3.125e-4** (`b32c`/`b32d`) | **0.092** | **0.087** | **0.0895** | 73.3 / **10.0** | **−31% / −16%** |

**4 of 4 paired comparisons favour `epsilon`, group effect −26%, and it is flat from 600k to 1M** (b32's
shared-set mean 0.0875 → 0.086), so it is not an early-training transient. **The dose is a dead heat**
(0.0865 vs 0.0895), exactly as pre-registered for n=2 a side.

**Why these numbers are lower than the ones this section used to show.** The old table used **per-arm** state
sets, and churn depends on the action gap — ~0.2 reward units early-game against 20-24 in the endgame. The
`eps 1e-7` controls die early, so their sets had mean lengths of **11.9 and 21.2** against the treated arms'
34.9-38.0, and they were scored on near-tied states that flip for free. Churn, gap and `len` came out
rank-correlated across all six arms. On the shared set the effect fell from −47% to −26%.
[`c51_stability.py --states-from`](perDiagnostics/c51_stability.py) is the fix and is now required for any
cross-arm reading; full account in
[`findings.md`](findings.md#-corrected-2026-08-16-every-per-arm-churn-figure-above-is-inflated-2x-and-the-fix-is-a-shared-state-set).

**Two things the shared set settles.** **Gap no longer explains churn** — `c51pilotB` seed 2 now has the
*largest* action gap of the six (16.0 against 11.3-13.9) and still churns more than every treated arm, where
on per-arm sets the controls had the two smallest gaps. And **it is the optimizer, not policy quality**:
`b32d` has best-30 **10.0**, worst of the six by a wide margin, yet churn **0.087**, indistinguishable from
the three good arms and well below both controls. Churn tracked the `epsilon` value, not the score.

**What is still not established.** Four paired comparisons rest on **2 independent seeds**, and the per-seed
effect spans **−15% to −37%** — a 2.5× range in the effect itself; a sign test on 2 seeds is p=0.25.
**Direction consistent 4 of 4, magnitude ~26%, not significant.** `eps 1.5e-4` is a reasonable default on
this evidence and not a demonstrated one, which is what `b36`+`b37` (4 seeds a side) is for.

**The confound to keep in view is reverse causation:** churn falls as a policy converges, and these arms are
also *better*, so "a better policy settles" would produce the same table. The evidence against it is
`b32d` — **lowest churn of all six arms (0.062) and the worst best-30 (10.0)**, epsilon still near the
exploration ceiling. Churn fell in an arm that did not learn better, which is what acting on the optimizer
rather than on performance looks like. Not settled at n=2; the 600k pairing is the one to judge on.

**Unchanged by this:** `aeff` is 28-33 of 51 atoms in every arm and boundary mass stays ~0. That is **not**
a defect — realised returns at γ=0.9975 have pooled **sd 24.89**, implying a calibrated net should read ~41
effective atoms, so ours are slightly *over*confident rather than never sharpening. The n-step candidate
that reading supported is [retracted](findings.md#-the-c51-arms-chaos-is-the-learning-rate-not-c51--and-the-rate-is-high-because-c51-needs-it).

![b32a](charts/b32a-c51eps15e4seed1.png)
**b32a-c51eps15e4seed1** — `eps 1.5e-4`, seed 1

![b32b](charts/b32b-c51eps15e4seed2.png)
**b32b-c51eps15e4seed2** — `eps 1.5e-4`, seed 2

![b32c](charts/b32c-c51eps3125e4seed1.png)
**b32c-c51eps3125e4seed1** — `eps 3.125e-4`, seed 1

![b32d](charts/b32d-c51eps3125e4seed2.png)
**b32d-c51eps3125e4seed2** — `eps 3.125e-4`, seed 2

## C51 pilot — distributional RL, learning-rate screen — *closed at 600k, chose `5e-5`*

**A seventh section on purpose, and temporary.** The cap of six counts numbered batches. This screen
became `b31`, which is now void — but its two `lr 1e-4` arms are **batch 32's control**, so the section
stays until b32 closes rather than being retired with b31.

**The table and the graphs below this paragraph are regenerated by
[`pick_c51_lr.py`](pick_c51_lr.py)** between the `C51-PILOT-STATUS` markers, so they are current as of
whenever it last ran, and the prose outside them is hand-written. **Editing inside the markers is
pointless** — the next run overwrites it.

The first C51 arms in this project, from
[`../plans/distributional-c51.md`](../plans/distributional-c51.md): the scalar head is replaced by a
distribution over the return on **51 atoms over `[-5, 120]`** trained by cross-entropy, with the PER
priority as the KL. Everything else is **b25's config verbatim** — `fc 200,100,100`, IS off, target 1000,
discount 0.9975, `FORK_BRANCHES=4`, no food-distance shaping — so `b25a-d` is the seed-matched control
when this becomes a batch.

**Closed 2026-08-15 at 20:09, and `5e-5` was chosen for `b31` on consistency rather than on being
best.** The two readings worth carrying: at n=2 the **between-seed spread (up to 57.6 pp) is twice the
spread between the rate means (30.5 pp)**, so the ranking is thin; and **time to the first perfect game
predicted nothing** about where an arm finished (Spearman ρ = 0.05 — the 8k starter finished at 11.7, the
141k starter at 85.3). `2.5e-4` is the one clear failure, with one arm collapsing to zero at 599k. Three
arms had not stopped improving at the cap, so for `1e-5` and `1e-4` the **600k horizon** may be what bound
them rather than the rate. Full account:
[`findings.md`](findings.md#-the-c51-learning-rate-screen-the-seed-spread-beat-the-rate-effect-and-time-to-first-win-predicted-nothing).

**It is a learning-rate screen, not a result.** A cross-entropy loss starts at `ln 51 ≈ 3.93` where the
Huber TD loss starts near 0, so b25's `1e-5` is not obviously the same step size for a categorical head.
**Four rates × two seeds, seed-matched across the rates**, launched as two waves an hour and a half apart:
`1e-5` and `5e-5` at 15:06 (`c51pilot-`), then `1e-4` and `2.5e-4` at 16:41 (`c51pilotB-`) — so wave B is
several hundred thousand steps behind and the horizon line in the generated block is what makes the
comparison fair. Eight trainers on a 14-core laptop is deliberate and was measured first: ~2.3 GB per arm
against 36 GB, and a swap-in rate of 244 pages per 20 s, so the cost is throughput and not paging.

<!-- C51-PILOT-STATUS:BEGIN -->
*Generated by `pick_c51_lr.py` at 2026-08-15 20:09, when the last pilot arm stopped — the numbers below are read straight off the eval series, and the prose around this block is hand-written.*

**Compared at a common horizon of 600k steps**, the lowest final step any arm reached, because both metrics accumulate over an arm's own evals and a longer arm would otherwise win on horizon alone.

| lr | seeds | mean best-30 | mean `sef` | mean peak trail |
|---|---|---|---|---|
| 5e-05 **← chosen** | 2 | 69.5 | 12.6 | 92.42 |
| 1e-05 | 2 | 56.5 | 3.6 | 89.89 |
| 0.0001 | 2 | 39.0 | 5.3 | 88.19 |
| 0.00025 | 2 | 4.0 | 0.0 | 68.79 |

| arm | lr | seed | step | best-30 | `sef` | peak trail | first perfect |
|---|---|---|---|---|---|---|---|
| `c51pilot-lr1e5seed1` | 1e-05 | 1 | 600k | 85.3 | 7.3 | 93.56 | 141k |
| `c51pilot-lr5e5seed2` | 5e-05 | 2 | 600k | 71.7 | 13.0 | 93.30 | 20k |
| `c51pilot-lr5e5seed1` | 5e-05 | 1 | 600k | 67.3 | 12.1 | 91.54 | 15k |
| `c51pilotB-lr1e4seed2` | 0.0001 | 2 | 600k | 66.3 | 10.6 | 90.80 | 46k |
| `c51pilot-lr1e5seed2` | 1e-05 | 2 | 600k | 27.7 | 0.0 | 86.22 | 92k |
| `c51pilotB-lr1e4seed1` | 0.0001 | 1 | 600k | 11.7 | 0.0 | 85.58 | 8k |
| `c51pilotB-lr25e4seed1` | 0.00025 | 1 | 600k | 5.7 | 0.0 | 70.82 | 49k |
| `c51pilotB-lr25e4seed2` | 0.00025 | 2 | 600k | 2.3 | 0.0 | 66.76 | 59k |

**Chosen: `5e-05`** — best_perfect30 69.5 against 56.5 for the next rate (1e-05).

**Batch `b31` launched at 2026-08-15 20:09** on this rate, 4 seeds, 2M cap, `fc 200,100,100`, otherwise b25's config — so `b25a-d` is the seed-matched control.

![c51pilot-lr1e5seed1](charts/c51pilot-lr1e5seed1.png)
**c51pilot-lr1e5seed1** — lr 1e-05, best-30 85.3, first perfect 141k

![c51pilot-lr5e5seed2](charts/c51pilot-lr5e5seed2.png)
**c51pilot-lr5e5seed2** — lr 5e-05, best-30 71.7, first perfect 20k

![c51pilot-lr5e5seed1](charts/c51pilot-lr5e5seed1.png)
**c51pilot-lr5e5seed1** — lr 5e-05, best-30 67.3, first perfect 15k

![c51pilotB-lr1e4seed2](charts/c51pilotB-lr1e4seed2.png)
**c51pilotB-lr1e4seed2** — lr 0.0001, best-30 66.3, first perfect 46k

![c51pilot-lr1e5seed2](charts/c51pilot-lr1e5seed2.png)
**c51pilot-lr1e5seed2** — lr 1e-05, best-30 27.7, first perfect 92k

![c51pilotB-lr1e4seed1](charts/c51pilotB-lr1e4seed1.png)
**c51pilotB-lr1e4seed1** — lr 0.0001, best-30 11.7, first perfect 8k

![c51pilotB-lr25e4seed1](charts/c51pilotB-lr25e4seed1.png)
**c51pilotB-lr25e4seed1** — lr 0.00025, best-30 5.7, first perfect 49k

![c51pilotB-lr25e4seed2](charts/c51pilotB-lr25e4seed2.png)
**c51pilotB-lr25e4seed2** — lr 0.00025, best-30 2.3, first perfect 59k
<!-- C51-PILOT-STATUS:END -->

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
