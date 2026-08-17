# Charts

Progress graphs for the most recent batches — **28, 31, 32, 33, 34 and 36**, a cap of six, newest first,
plus the **C51 pilot** as a temporary seventh while its arms are still a live control. Per-arm numbers live in
[`completedRuns.md`](completedRuns.md); this file is images plus a short reading of each. A batch appears
here **while it is still running**, with training-only numbers, not just once it has closed.
Batch 27 was retired to [`archive/charts-archive.md`](archive/charts-archive.md) when 36 launched, and
**batch 30** followed when 34's results arrived.

**`b35` (gate 40) is a desktop batch still in training and has no section here yet** — its PNG lives on the
box and arrives with its results, so it is entered when the results branch is copied in. That is a tracked
gap, not a missed one; see [`runs.md`](runs.md) for its status. **`b34` (gate 70) closed and is entered
below.**

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

## Batch 36 — **C51 on `fc 320`**, one wide layer instead of three narrow — *running on the laptop, 3M cap*

**Batch 32's config verbatim at `eps 1.5e-4` with `SNEK_FC_LAYERS=320` the only change**, seeds 1-4, win
reward back at its default 100, `lr 1e-4`, 51 atoms over `[-5, 120]`. Launched 12:43 on 2026-08-16;
launcher [`launch_c51_fc320.sh`](launch_c51_fc320.sh), rationale and pre-registered hypotheses in
[`runs.md`](runs.md).

**Two controls, both on disk, answering different questions.** `b32a`/`b32b` — same `eps`, `lr` and seeds
at `fc 200,100,100` — is the clean one-variable *architecture* pair, but only 1M deep, so **match at 1M
before quoting anything**. `b24a-d` is **ddqn** at this exact shape, 2M and closed out at pooled 87.9 with
two ≥98%/500 records, which is the "is C51 worth it at all" comparison.

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

**3M is as much of the experiment as the shape.** No C51 arm has ever run past **1M** — the pilot stopped
at 600k, b31 at ~560k, b32 at its cap — while b32's best-30 peaks landed at 353-865k and every b33 arm
declined for 1.4M steps after peaking. If C51 decays past ~1.2M, every future C51 batch can stop there.

First evals only, nothing to read yet.

![b36a](charts/b36a-c51fc320seed1.png)
**b36a-c51fc320seed1** — paired with `b32a`

![b36b](charts/b36b-c51fc320seed2.png)
**b36b-c51fc320seed2** — paired with `b32b`

![b36c](charts/b36c-c51fc320seed3.png)
**b36c-c51fc320seed3**

![b36d](charts/b36d-c51fc320seed4.png)
**b36d-c51fc320seed4**

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

## Batch 33 — a filled board pays **10**, not 100 — *stopped at 1.64-1.77M of 3M: the largest single-knob regression measured here*

**`SNEK_PERFECT_GAME_REWARD=10`, `SNEK_V_MAX=40`, otherwise batch 32's config at `eps 1.5e-4`, seeds
1-4.** `b32a`/`b32b` are an exact paired control differing only in the win reward. Launched 01:32,
stopped 10:04 on 2026-08-16 at the user's call, no close-out — the training curves settle it. Launcher
[`launch_win10.sh`](launch_win10.sh); full write-up in [`completedRuns.md`](completedRuns.md), mechanism in
[`findings.md`](findings.md#-falsified-2026-08-16-shrinking-the-win-reward-100--10-does-not-buy-c51-stability--it-teaches-the-agent-that-winning-is-a-mistake).

| arm | step | best-30 | at | trailing now | `sef` | final ε |
|---|---|---|---|---|---|---|
| `b33b` | 1640k | 25.3 | 238k | 73.6 | 0.0 | 0.0102 |
| `b33c` | 1772k | 22.3 | 150k | 78.1 | 0.1 | 0.0115 |
| `b33a` | 1732k | 20.3 | 179k | 76.4 | 0.1 | 0.0110 |
| `b33d` | 1708k | 18.3 | 353k | 67.9 | 0.1 | 0.0104 |
| **`b32a`** control | 1000k | **77.0** | 724k | 84.3 | 16.1 | 0.0035 |
| **`b32b`** control | 1000k | **63.0** | 865k | 69.4 | 10.3 | 0.0035 |

**Every arm peaks by 353k and declines for the next 1.4M steps.** That shape is the reading: the better it
fits the objective, the worse it plays.

**The motivation was resolution, it was delivered, and it bought nothing.** The win is what forces a
125-unit support, so at 51 atoms a meal is **0.40 atoms**; at `v_max=40` it is **1.11**. Measured on the
trained arm: **1.22 atoms per food against the control's 0.44**, the 2.8× as designed. So **atom spacing is
not what limits C51 here.**

**Why it fails, in one line.** Every meal of progress moves board-fill up a notch and moves `V` **down
1.7-4.4 points** while the meal pays 1 — so `Q(don't eat) > Q(eat)` and the agent correctly avoids
finishing. The control's `V` moves **+4.4 to +12.1** over the same bands. The threshold is
`W > 1/(1 − γ^k)`, which at γ=0.9975 and 7-12 steps per meal is **34-58**: 100 clears it, 10 does not.

**Two things this is *not*.** The board-fill input is **rank 1 of 30** by saliency in both arms, so it is
not being ignored; and the endgame action gap is **19.8-24.3, larger than `V` itself**, so the actions are
not near-tied. Measured with [`perDiagnostics/endgame_gradient.py`](perDiagnostics/endgame_gradient.py) —
which also found that **indices 18-20, "this move wins", are constant zeros in 0.000-0.025% of states**,
so neither arm learns to win from them. An earlier reading of this batch as a *calibration* failure was
withdrawn the same day: against the optimal value the net is 9-16% optimistic, which is ordinary.

**What that looks like on the board**, greedy, 60 episodes each:

| arm | outcomes | median steps/meal at 95+ | starve headroom | `chase_safe` |
|---|---|---|---|---|
| `b33a` @852k | 44 coll / 4 starve / **12 perfect** | **32.5** | 468 | 0.054 |
| `b33b` @1640k | 54 / 4 / 2 | 33.5 | 466 | 0.044 |
| `b33c` @1772k | 44 / 14 / 2 | 42.5 | 458 | 0.051 |
| `b33d` @1708k | 53 / 3 / 4 | 36.0 | 464 | 0.052 |
| **`b32a` @724k** control | **5 / 0 / 55** | **2.0** | 498 | 0.158 |

It **stalls** two meals short and dies of **geometry, not the clock** — 44 of 48 lost episodes still
winnable a median **1 move** before death at median length 96, with 458-468 steps of starve budget in
hand. **The predicted urgency collapse is confirmed and larger than predicted (16-21×, not 10×); the
predicted *symptom* — starvation — is wrong**, and that correction is the transferable part.

![b33a](charts/b33a-c51win10seed1.png)
**b33a-c51win10seed1** — paired with `b32a`. Best-30 20.3 @179k, then 1.55M steps of decline

![b33b](charts/b33b-c51win10seed2.png)
**b33b-c51win10seed2** — paired with `b32b`. The batch's best at 25.3, still a third of its control

![b33c](charts/b33c-c51win10seed3.png)
**b33c-c51win10seed3** — peaks earliest (150k) and starves most (14 of 60)

![b33d](charts/b33d-c51win10seed4.png)
**b33d-c51win10seed4** — the weakest, best-30 18.3

## Batch 32 — **Adam's `epsilon`** on C51, `lr 1e-4`, two reference values — *all four reached the 1M cap; churn re-measure at 1M still owed*

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

**Launched 23:14 on 2026-08-15. Status at 01:06 on 2026-08-16 — 370-403k of 1M, all four alive, no zero
stretch, and the readout is moving the right way.** Churn per 5k steps on a fixed 800-state set, paired
against each arm's own seed at `eps 1e-7`:

| `eps` | churn @200k | churn @360k | best-30 ≤364k | peak trailing |
|---|---|---|---|---|
| **1e-7** (control, `c51pilotB`) | 0.166 / 0.127 → **0.147** | 0.167 / 0.106 → **0.137** | 11.7 / 56.0 → 33.9 | 85.6 / 90.1 |
| **1.5e-4** (`b32a`/`b32b`) | 0.115 / 0.104 → **0.110** | 0.101 / 0.119 → **0.110** | 66.3 / 61.3 → **63.8** | **93.0 / 92.3** |
| **3.125e-4** (`b32c`/`b32d`) | 0.099 / 0.062 → **0.081** | 0.099 / 0.097 → **0.098** | 73.3 / **10.0** → 41.7 | 92.5 / 87.2 |
| *ddqn `b30e`/`b30f` — **`lr 1e-5`, not comparable*** | 0.042 / 0.056 | 0.035 / 0.058 | — | — |

**7 of 8 paired comparisons across two independent horizons churn less than their own control**, and the
group means are monotone in dose at both. **It did not cost learning speed** — the failure mode where
`epsilon` acts as a smaller learning rate in disguise — since both `1.5e-4` arms beat their controls on
best-30 *and* peak trailing, and the treated groups hold the two highest peaks here.

**Quote the paired figures only.** The ddqn row is there for scale and **is not a target**: it was measured
at `lr 1e-5` against these arms' `lr 1e-4`, so treating the distance to it as "how much is left to fix"
re-commits the rate-vs-algorithm confound this whole line of work started by correcting. No ddqn-at-`1e-4`
measurement exists.

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

## Batch 31 — **C51** at `lr 5e-5`, 2M — *stopped at 538-569k, no close-out*

**The first C51 batch, and void.** b25's config verbatim plus `ALGO=c51` (51 atoms over `[-5, 120]`, KL
priority) at the rate [`pick_c51_lr.py`](pick_c51_lr.py) chose from the pilot, seeds 1-4, 2M cap. Killed at
23:10 after `c51_stability.py` showed the chaos these curves show is the **learning rate, not C51** — which
made 2M at a rate chosen under the old reading not worth four slots. **No close-out was run**, by decision.

| arm | step | best-30 | `sef` | peak trail |
|---|---|---|---|---|
| `b31d-c51lr5e5seed4` | 569k | **71.7** | 16.5 | 92.86 |
| `b31a-c51lr5e5seed1` | 555k | 66.7 | 9.5 | **94.86** |
| `b31b-c51lr5e5seed2` | 538k | 53.3 | 1.7 | 92.26 |
| `b31c-c51lr5e5seed3` | 562k | 21.0 | 0.0 | 89.76 |

All four healthy at the kill (no zero stretch). The **50.7 pp best-30 spread at one config** is the n=4
noise problem restated, not a result — which is the other reason not to spend a close-out on it.

![b31a](charts/b31a-c51lr5e5seed1.png)
**b31a-c51lr5e5seed1**

![b31b](charts/b31b-c51lr5e5seed2.png)
**b31b-c51lr5e5seed2**

![b31c](charts/b31c-c51lr5e5seed3.png)
**b31c-c51lr5e5seed3**

![b31d](charts/b31d-c51lr5e5seed4.png)
**b31d-c51lr5e5seed4**

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
