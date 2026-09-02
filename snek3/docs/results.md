# Results — every arm

The canonical arm table. One row per arm, filled in when the arm stops and its stage-B measurement
lands. Config, final numbers, verdict.

**Newest batch first.** A batch closes at the top of this file, under the intro and above the
batch before it, so the newest numbers are the ones you land on. The reference sections —
`Imported policies` and `Reading this table` — stay at the bottom.

**The best single policy is not in this file.** An arm's row here is a *selected* maximum over
hundreds of checkpoints; a record needs a fresh measurement at depth, and those live in
[`../hallOfFame/HOF.md`](../hallOfFame/HOF.md) — two entries as of 2026-09-01, confirmed on 30,000
episodes at a seed no selection pass used, the better of them **98.96%** and the first snek3 policy to
beat the snek2 champion at matched depth.

**‡ The PPO batches were renamed on 2026-08-31: `p0`-`p3` became `b3`-`b6`** — one prefix for
every batch in both eras, because a second one had already cost the desktop's batch grouping a day.
The map is a `+3` offset holding the old order: `p0`->`b3`, `p1`->`b4`, `p2`->`b5`, `p3`->`b6`, so
the b-series is not chronological — b5 and b6 ran before b4. Renamed here, in `runs/` and in
`savedPolicies/`. **Two places still hold the old names and are meant to:** the `results` branch,
whose published artifacts are history, and the daemon's ledger, whose keys are the job ids those
waves actually ran under. Looking for an arm's desktop artifacts, search the old name.

## Batch b8 — the stability knobs, 4 knobs x 4 seeds, closed 2026-09-01

Sixteen arms at 100M transitions on the desktop, holding b4's config fixed (`fc (200,100)`, 8 epochs,
b2's reward) so exactly one knob moves per group, with **b4 itself as the control**. Every comparison
truncates b4 to b8's 100M cap, because 65% of b4's record rows land after it.

| group | arms | drawdown < 50% | ≥98%/500 | best row | best30 | sef |
|---|---|---:|---:|---:|---:|---:|
| `target_KL` 0.02 | `b8i`-`b8l` | 5.9% | **6.0%** | 99.6 | 97.03 (96.4-97.3) | 84.7 |
| entropy 0.01 → 0.001 | `b8e`-`b8h` | 3.5% | 5.0% | 99.6 | 97.15 (96.8-98.0) | 87.0 |
| entropy 0.003 | `b8a`-`b8d` | 3.7% | 4.3% | 99.6 | 97.20 (96.7-97.4) | 86.9 |
| λ 0.95 | `b8m`-`b8p` | **2.2%** | 2.3% | 99.2 | 96.40 (96.0-96.9) | **89.8** |
| **b4 control @100M** | `b4a`-`b4h` | 8.4% | 5.7% | 99.4 | 97.41 (97.0-97.9) | 85.8 |

**Every knob cut the drawdown; none beat the control on record density.** Drawdown is the median share
of post-competence stage-A evals below 50% perfect — b4's defining pathology, 8.4% at this cap — and
all four treatments land between 2.2% and 5.9%. But the pre-registered ≥98%/500 density moves the other
way: only `target_KL` clears the control at all, and λ 0.95, the best arm on stability, is less than
half the control. **b8 fixed the thing it was aimed at and it did not help.**

`sef` ranks the groups backwards here too (λ 0.95 top on `sef`, bottom on density), which is the
[`findings.md`](findings.md) result about `strong_eval_fraction` reproducing on a second batch.

**Both never-exercised knobs were confirmed live**, which is what the pre-queue smoke tests were for:
`target_KL` 0.02 stopped the epoch loop on 1.9-3.3% of recorded updates per arm with `epochs_run`
median still 8, and the anneal ran 0.0100 → 0.0010 and completed exactly at the cap. Contrast b4,
where `epochs_run` was 8 in all 97,656 recorded updates.

### b8 at 5,000 episodes — the laptop `hof5000` pass, 2026-09-01

Every b8 checkpoint at ≥98.5% on its 500-episode close-out, re-measured at 5,000 episodes:
**135 rows, 675k episodes, 3.0 + 9.9 min on the laptop in two waves, exit 0.** Merged rows equal the
≥98.5%/500 count and the shard sum on all sixteen arms.

| group | rows | mean | ≥98 | ≥98.73 | best | 500 → 5,000 |
|---|---:|---:|---:|---:|---:|---:|
| `target_KL` 0.02 | 54 | 97.63 | 29.6% | **1** | **98.80** | −1.08 |
| entropy anneal | 44 | 97.56 | 34.1% | 0 | 98.50 | −1.19 |
| entropy 0.003 | 23 | 97.77 | 26.1% | 0 | 98.70 | −1.08 |
| λ 0.95 | 14 | 97.44 | 14.3% | 0 | 98.10 | −1.33 |
| **pooled** | **135** | **97.61** | **29.6%** | **1** | **98.80** | **−1.15** |

**b8 has no hall-of-fame candidate.** One row of 135 clears the snek2 champion's 98.73%, against 29
for b5 and 20 for b6, and nothing reaches 99%. `b8o-lam95-seed3` is the extreme case: its single
candidate scored 98.6 at 500 episodes and **96.90** at 5,000, a −1.70 pp fall.

## Batch b7 — the fc-layout sweep, 8 layouts x 4 seeds, closed 2026-09-01

**The network-shape test this file had been calling for since b3, and it went to the single layer.**
32 arms, 50M transitions each (3,052 rollouts of 16,384), four waves of two layouts, every wave a
comparison that stands alone. Everything but `fc_layers` at PPO's reference — 4 epochs, lr 3e-4,
γ 0.99, λ 0.98, entropy 0.01, 128x128 rollout, minibatch 256, b2's reward — seeds 1-4 pinned to the
seed in each arm name. Desktop, 2026-09-01 00:01 -> ~11:14, with `auto_stage_b` measuring each wave
before the next trained. **`fc (320,)` wins on the ≥98%/500 density, and every seed of it beats every
seed of five of the seven other layouts.**

| layout | wave | rows | ≥98%/500 | per-seed share | best30 (4 seeds) | sef | best row |
|---|---:|---:|---:|---|---|---:|---:|
| `fc (320,)` | 1 | 4,003 | **17.3%** | 18.5 19.0 16.5 15.1 | 97.8 97.8 97.7 97.7 | 90.9 | 99.6% |
| `fc (200,100)` | 1 | 3,799 | **11.8%** | 11.2 12.3 10.6 13.3 | 97.3 97.8 97.7 98.2 | 94.7 | 99.8% |
| `fc (100,200,100)` | 4 | 3,220 | **11.6%** | 9.9 15.5 6.1 14.8 | 97.4 97.9 97.0 97.9 | 93.3 | 99.4% |
| `fc (100,100)` | 3 | 4,010 | **11.3%** | 4.6 16.8 13.7 10.0 | 97.2 98.1 97.8 98.0 | 94.2 | 99.8% |
| `fc (200,100,50)` | 4 | 3,630 | **10.8%** | 10.3 8.9 12.8 11.2 | 97.8 97.8 98.2 98.0 | 94.4 | 99.6% |
| `fc (160,160)` | 3 | 3,469 | **8.3%** | 7.0 7.2 8.6 10.4 | 96.9 97.4 98.1 97.5 | 94.9 | 99.8% |
| `fc (300,100)` | 2 | 3,144 | **6.8%** | 6.6 4.9 6.8 9.0 | 97.3 96.9 97.2 97.9 | 95.0 | 99.4% |
| `fc (400,200)` | 2 | 2,731 | **5.1%** | 3.7 8.3 5.4 2.9 | 96.8 97.3 97.2 96.8 | 94.8 | 99.2% |

`vs fc 320` is an exact two-sided Mann-Whitney on the four per-seed shares. 0.029 is the floor at
4-vs-4 and means complete separation. Full reading, including what the sweep does **not** settle, in
[`findings.md`](findings.md); the charts are in [`charts.md`](charts.md).

**Read the `sef` column against the density column.** They rank the layouts *backwards* — `fc 320`
is last on `strong_eval_fraction` and first on record density, `fc (300,100)` the reverse. That is a
protocol finding rather than a b7 one: the 80% threshold `strong_eval_fraction` uses sits far below
the region a champion hunt cares about, and the stage-A ≥98% rate (r=+0.80) or `best_perfect30`
(+0.71) is the screen to use instead. [`findings.md`](findings.md) has the numbers.

### b7 at 5,000 episodes — the laptop `hof5000` pass, 2026-09-01

Every b7 checkpoint at ≥98.5% on its 500-episode close-out, re-measured at 5,000 episodes:
**766 rows, 3.83M episodes, 84.5 min on the laptop, exit 0.** Merged rows equal the ≥98.5%/500 count
and the shard sum on all thirty-two arms.

| layout | rows | mean | ≥98 | ≥98.73 | best | 500 → 5,000 |
|---|---:|---:|---:|---:|---:|---:|
| `fc (320,)` | 174 | **97.86** | **43.1%** | **2** | 98.90 | −0.94 |
| `fc (100,100)` | 127 | 97.81 | 41.7% | 1 | **99.20** | −1.03 |
| `fc (100,200,100)` | 103 | 97.80 | 40.8% | 1 | 98.80 | −0.99 |
| `fc (200,100,50)` | 92 | 97.75 | 35.9% | 1 | 98.90 | −1.04 |
| `fc (200,100)` | 109 | 97.71 | 34.9% | 0 | 98.70 | −1.13 |
| `fc (160,160)` | 65 | 97.66 | 27.7% | 1 | 98.90 | −1.20 |
| `fc (300,100)` | 65 | 97.54 | 30.8% | 0 | 98.40 | −1.25 |
| `fc (400,200)` | 31 | 97.47 | 9.7% | 0 | 98.50 | −1.27 |
| **pooled** | **766** | **97.75** | **36.8%** | **6** | **99.20** | **−1.05** |

**`fc (320,)`'s stage-B win survives at depth, but as volume rather than quality.** It keeps the top
pooled mean and the most champion-level rows, and `fc (400,200)` stays last on both — yet the whole
mean spread is **0.39 pp** (97.47 to 97.86) against a candidate-count spread of 31 to 174. Layout
buys more shots, not better ones.

**One inversion worth noting: `fc (100,100)` was mid-pack at stage B (11.3%) and comes second here**,
owning the only ≥99% row in all 766. That row — `b7av-fc100x100-seed2` @4.1M, 99.20% — does not
survive its own basin: neighbours within ±1M average **98.19** and within ±3M **98.13**, so it is a
selected high on an ~98.2 policy, and no b7 checkpoint threatens `b5h`'s confirmed 98.96%.

### b7's arms

| arm | best30 | trailing | sef | rows | ≥98%/500 | density | best row |
|---|---:|---:|---:|---:|---:|---:|---:|
| `b7aa-fc320-seed1` | 97.8 | 94.27 | 94.0 | 1,169 | 216 | 18.5% | 99.6% |
| `b7ab-fc320-seed2` | 97.8 | 94.29 | 87.8 | 879 | 167 | 19.0% | 99.4% |
| `b7ac-fc320-seed3` | 97.7 | 94.33 | 90.0 | 991 | 164 | 16.5% | 99.4% |
| `b7ad-fc320-seed4` | 97.7 | 94.00 | 91.7 | 964 | 146 | 15.1% | 99.6% |
| `b7ae-fc200x100-seed1` | 97.3 | 93.81 | 94.5 | 910 | 102 | 11.2% | 99.4% |
| `b7af-fc200x100-seed2` | 97.8 | 93.53 | 93.7 | 847 | 104 | 12.3% | 99.8% |
| `b7ag-fc200x100-seed3` | 97.7 | 93.31 | 95.1 | 1,017 | 108 | 10.6% | 99.4% |
| `b7ah-fc200x100-seed4` | 98.2 | 94.09 | 95.4 | 1,025 | 136 | 13.3% | 99.8% |
| `b7ai-fc300x100-seed1` | 97.3 | 94.13 | 95.7 | 726 | 48 | 6.6% | 99.2% |
| `b7aj-fc300x100-seed2` | 96.9 | 94.12 | 95.1 | 710 | 35 | 4.9% | 99.2% |
| `b7ak-fc300x100-seed3` | 97.2 | 93.41 | 94.1 | 822 | 56 | 6.8% | 99.2% |
| `b7al-fc300x100-seed4` | 97.9 | 93.85 | 95.0 | 886 | 80 | 9.0% | 99.4% |
| `b7am-fc400x200-seed1` | 96.8 | 93.68 | 95.5 | 641 | 24 | 3.7% | 99.2% |
| `b7an-fc400x200-seed2` | 97.3 | 93.75 | 95.3 | 695 | 58 | 8.3% | 99.2% |
| `b7ao-fc400x200-seed3` | 97.2 | 94.25 | 93.1 | 851 | 46 | 5.4% | 99.0% |
| `b7ap-fc400x200-seed4` | 96.8 | 94.01 | 95.2 | 544 | 16 | 2.9% | 98.8% |
| `b7aq-fc160x160-seed1` | 96.9 | 93.72 | 95.7 | 932 | 65 | 7.0% | 99.2% |
| `b7ar-fc160x160-seed2` | 97.4 | 93.60 | 94.2 | 746 | 54 | 7.2% | 99.2% |
| `b7as-fc160x160-seed3` | 98.1 | 93.78 | 94.6 | 893 | 77 | 8.6% | 99.4% |
| `b7at-fc160x160-seed4` | 97.5 | 93.44 | 95.2 | 898 | 93 | 10.4% | 99.8% |
| `b7au-fc100x100-seed1` | 97.2 | 94.03 | 91.7 | 790 | 36 | 4.6% | 99.4% |
| `b7av-fc100x100-seed2` | 98.1 | 94.49 | 94.5 | 1,049 | 176 | 16.8% | 99.8% |
| `b7aw-fc100x100-seed3` | 97.8 | 94.27 | 96.0 | 1,130 | 155 | 13.7% | 99.6% |
| `b7ax-fc100x100-seed4` | 98.0 | 92.36 | 94.6 | 1,041 | 104 | 10.0% | 99.6% |
| `b7ay-fc100x200x100-seed1` | 97.4 | 93.87 | 89.4 | 708 | 70 | 9.9% | 99.0% |
| `b7az-fc100x200x100-seed2` | 97.9 | 94.18 | 96.0 | 974 | 151 | 15.5% | 99.4% |
| `b7ba-fc100x200x100-seed3` | 97.0 | 91.97 | 93.5 | 705 | 43 | 6.1% | 99.4% |
| `b7bb-fc100x200x100-seed4` | 97.9 | 93.64 | 94.2 | 833 | 123 | 14.8% | 99.2% |
| `b7bc-fc200x100x50-seed1` | 97.8 | 93.55 | 93.2 | 838 | 86 | 10.3% | 99.2% |
| `b7bd-fc200x100x50-seed2` | 97.8 | 93.95 | 95.3 | 935 | 83 | 8.9% | 99.4% |
| `b7be-fc200x100x50-seed3` | 98.2 | 93.44 | 94.4 | 956 | 122 | 12.8% | 99.6% |
| `b7bf-fc200x100x50-seed4` | 98.0 | 93.38 | 94.6 | 901 | 101 | 11.2% | 99.2% |
| **pooled** | | | | **28,006** | **3,045** | **10.9%** | **99.8%** |

**Where the raw rows are.** b7 ran on the desktop, so its `_checkpoint_evals.json` files are on the
`results` branch under `results/b7-stageb{,-w2,-w3,-w4}/` — 8.3 MB an arm, 267 MB for the batch, which
is why `runs/` carries the reports and the PNGs and not those. b5's `hof5000` rows are on the same
branch under the **old** name, `results/p2-hof5000/`.

## Batch b4 — `fc (200,100)` + 8 epochs, eight seeds, closed 2026-08-31

**The clean network-shape test, and it came out against the shape.** 8 seeds, 200M transitions each
(199,999,488 = 12,207 rollouts), b2's reward function, everything else at PPO's defaults. Run on the
desktop 2026-08-30 18:46 -> 2026-08-31 02:34, stage B done 04:34. Numbers read off the `results`
branch 2026-09-01, charts imported and redrawn the same day (the published PNGs carried the
pre-rename `p1` titles).

| arm | seed | best30 | trailing | sef | stage B: rows | ≥98%/500 | density | best row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `b4a-fc200x100ep8-seed1` | 1 | 97.4 | 81.28 | 76.8 | 1,466 | 98 | 6.7% | 99.6% |
| `b4b-fc200x100ep8-seed2` | 2 | 97.5 | 93.45 | 80.8 | 1,575 | 84 | 5.3% | 99.4% |
| `b4c-fc200x100ep8-seed3` | 3 | 97.6 | 94.34 | 83.4 | **2,513** | **260** | **10.3%** | 99.4% |
| `b4d-fc200x100ep8-seed4` | 4 | **97.9** | 94.06 | 79.5 | 1,355 | 77 | 5.7% | 99.4% |
| `b4e-fc200x100ep8-seed5` | 5 | 97.2 | 94.22 | 87.8 | 2,140 | 191 | 8.9% | 99.4% |
| `b4f-fc200x100ep8-seed6` | 6 | 97.4 | 93.32 | 83.1 | 1,815 | 116 | 6.4% | 99.2% |
| `b4g-fc200x100ep8-seed7` | 7 | 97.3 | 93.59 | 82.1 | 1,965 | 159 | 8.1% | 99.4% |
| `b4h-fc200x100ep8-seed8` | 8 | 97.0 | 92.52 | 89.5 | 1,904 | 94 | 4.9% | 99.4% |
| **pooled** | | | | | **14,733** | **1,079** | **7.3%** | **99.6%** |

### b4 at 5,000 episodes — the laptop `hof5000` pass, 2026-09-01

Every b4 checkpoint at ≥98.5% on its 500-episode close-out (`above:98.5`), re-measured at 5,000
episodes: **274 rows, 1.37M episodes, 25.8 min on the laptop, exit 0.**

| arm | rows | mean | ≥98 | best |
|---|---:|---:|---:|---:|
| `b4a` | 37 | 97.74 | 45.9% | 98.70 |
| `b4b` | 20 | 97.75 | 30.0% | **98.80** |
| `b4c` | 64 | 97.76 | 37.5% | 98.60 |
| `b4d` | 19 | 97.79 | 31.6% | 98.50 |
| `b4e` | 49 | 97.68 | 30.6% | 98.40 |
| `b4f` | 30 | 97.58 | 23.3% | 98.50 |
| `b4g` | 38 | 97.70 | 34.2% | 98.70 |
| `b4h` | 17 | 97.61 | 23.5% | 98.70 |
| **pooled** | **274** | **97.71** | **33.6%** | **98.80** |

**One row clears the snek2 champion's 98.73% and none reaches 99%**, against 29 and 20
champion-level rows for b5 and b6. Candidate density says it from the other end: 18.6 `above:98.5`
candidates per 1,000 stage-B rows against b6's 35.8. **b4 has no hall-of-fame candidate** — and
[`findings.md`](findings.md) has why the b5-vs-b6 half of this comparison did *not* survive the same
treatment while the b4 half did. The pass is the
[`hof-remeasure`](../skills/hof-remeasure/SKILL.md) skill.

### ‡ b4 is the weakest of the three 8-seed batches, which retires b3's epochs ranking

All three ran the same reward function and the same protocol, so the pre-registered ≥98%/500 density
is directly comparable — with the caveat that b4's horizon is the shortest of the three:

| batch | network | epochs | transitions | best30 | pooled ≥98%/500 |
|---|---|---:|---:|---:|---:|
| **b6** | `fc (200,100)` | **4** | 215-231M | 97.8-98.5 | **12.8%** |
| **b5** | `fc (320,)` | 8 | 255-271M | 97.8-98.5 | 9.6% |
| **b4** | `fc (200,100)` | 8 | 200M | **97.0-97.9** | **7.3%** |

**The two knobs interact, and negatively.** Holding the network at `fc (200,100)`, 4 epochs beats 8
by 12.8% to 7.3%. Holding epochs at 8, `fc 320` beats `fc (200,100)` by 9.6% to 7.3%. So the arm
carrying *both* of b3's best single knobs is worse than either one alone, and b3's ranking of epochs
8 first — from one arm at 20M — does not survive eight seeds at 200M. b4's best30 range sits **below**
both comparators on every seed.

**What it does not settle.** b4 is 200M against b6's 215-231M and b5's 255-271M, so every comparison
truncates at b4's horizon; the density statistic is also known to be unstable at fixed depth
([`findings.md`](findings.md)), and the three batches differ in run length as well as in knobs. The
sign of the interaction is large enough to act on — [batch b7](runs.md) does — but the size is not.

## Batches b5 and b6 — eight seeds each, closed 2026-08-30

Both closed the same afternoon: b5's stage B in 222.6 min on the desktop, b6's in 226.1 min on the
laptop, both status 0. **The stage-B headline is the >=98%/500 count** — the width of the record
region — per [`protocol.md`](protocol.md).

### b6 — `fc (200,100)`, 4 epochs

| arm | stage-B rows | >=98%/500 | best row | stage-A best30 | strong | transitions |
|---|---:|---:|---:|---:|---:|---:|
| `b6a-fc200x100-seed1` | 4499 | 13.5% | 99.6 | 98.4 | 97.0 | 231M |
| `b6b-fc200x100-seed2` | 4503 | 13.0% | 99.6 | 98.4 | 95.5 | 230M |
| `b6c-fc200x100-seed3` | 5812 | 17.4% | 99.8 | 98.5 | 97.5 | 221M |
| `b6d-fc200x100-seed4` | 5151 | 15.0% | 99.6 | 98.3 | 97.3 | 220M |
| `b6e-fc200x100-seed5` | 3882 | 9.6% | 99.6 | 97.9 | 96.1 | 218M |
| `b6f-fc200x100-seed6` | 4042 | 8.9% | 99.6 | 97.9 | 95.9 | 215M |
| `b6g-fc200x100-seed7` | 4279 | 10.7% | 99.4 | 98.0 | 96.3 | 216M |
| `b6h-fc200x100-seed8` | 4104 | 12.0% | 99.8 | 97.9 | 95.5 | 215M |

### b5 — `fc (320,)`, 8 epochs

| arm | stage-B rows | >=98%/500 | best row | stage-A best30 | strong | transitions |
|---|---:|---:|---:|---:|---:|---:|
| `b5a-ep8-seed1` | 3836 | 7.2% | 99.8 | 97.9 | 97.3 | 271M |
| `b5b-ep8-seed2` | 5552 | 13.3% | 100.0 | 98.4 | 98.3 | 271M |
| `b5c-ep8-seed3` | 3494 | 6.2% | 99.6 | 97.8 | 96.3 | 265M |
| `b5d-ep8-seed4` | 3433 | 7.2% | 99.4 | 97.8 | 95.2 | 264M |
| `b5e-ep8-seed5` | 5312 | 11.2% | 99.8 | 98.1 | 98.0 | 258M |
| `b5f-ep8-seed6` | 4861 | 9.5% | 99.8 | 97.9 | 97.0 | 257M |
| `b5g-ep8-seed7` | 5054 | 10.4% | 99.6 | 97.9 | 98.3 | 257M |
| `b5h-ep8-seed8` | 3039 | 8.8% | 99.8 | 98.5 | 96.9 | 255M |

### ‡ These two batches differ in two knobs, so they are not a network-shape test

**‡ And the headline below did not survive 5,000 episodes (2026-09-01).** Re-measured over every
checkpoint at ≥98.5%/500, b5 and b6 have identical means (97.80 both), ≥98 rates within 0.1 pp, and
b5 is *ahead* on champion-level rows (29 to 20) and on the top checkpoint (99.20 to 99.10). The
+3.29 pp pooled gap in the table below was selection, not policy quality — 500 episodes carries a
0.72 pp sd against 5,000's 0.23. Read the table as what a 500-episode close-out reported, and
[`findings.md`](findings.md) for what replaced it. The b7 sweep since settled the network axis
outright, and it went to `fc (320,)`.

[`runs.md`](runs.md) named the two-hidden-layer result as "the most promising thread b3 turned up".
**b6 does not settle it.** b6 is `fc (200,100)` **and** 4 epochs; b5 is `fc (320,)` **and** 8 epochs —
and b3's one moving axis was gradient steps per transition, which is exactly what the epoch count
changes. The budgets differ too (b6 215-231M, b5 255-271M). So the comparison below confounds the
network shape with the axis already known to matter.

| seed | b5 >=98%/500 | b6 >=98%/500 | b6 − b5 |
|---:|---:|---:|---:|
| 1 | 7.2 | 13.5 | **+6.3** |
| 2 | 13.3 | 13.0 | **-0.3** |
| 3 | 6.2 | 17.4 | **+11.2** |
| 4 | 7.2 | 15.0 | **+7.8** |
| 5 | 11.2 | 9.6 | **-1.6** |
| 6 | 9.5 | 8.9 | **-0.6** |
| 7 | 10.4 | 10.7 | **+0.3** |
| 8 | 8.8 | 12.0 | **+3.2** |

**b6 leads on the pooled headline — 12.8% against 9.6%, mean +3.29 pp — and on ~50M fewer
transitions per arm.** The wins are asymmetric: the three largest are +11.2, +7.8 and +6.3, while the
three losses are −1.6, −0.6 and −0.3.

**The sign test says nothing, and it is the test this project leads with.** 5 of 8 seeds favour b6,
which is p≈0.73 two-tailed — a coin. The pooled gap is carried by three seeds. And **rank 1 of the
ranking is a tie**: peak `best_perfect30` is 98.5 in both batches (b6c, b5h). b5 also holds the single
best stage-B row in either batch, **100.0%/500** at b5b/184M.

**What would settle it is one batch varying only the network**, at matched epochs and matched budget.
That arm has still never been run — and since `dqn/net.py` takes the same `fc_layers` config, it is
also one arm away for DQN.

**Neither best row is a record claim.** Both are selected highs over thousands of rows; a record needs
a fresh 1,000+ episode measurement of the single winner, and all three of b3's highs fell 1.3-2.0 pp
on re-measurement.

## Batch b3 — the PPO tuning sweep

**15 arms, seed 1, 10M transitions each, all on b2's reward function**, each one knob off a reference
of lr 3e-4 / γ 0.99 / λ 0.98 / entropy 0.01 / fc 320 / 128x128 rollout / 4 epochs / minibatch 256.
Closed 2026-08-29. Seven ran on the laptop, eight on the desktop. **A tuning pass, not a gate** — no
arm is seed-matched to anything, so no row here supports a between-config claim on its own.

| arm | knob | best30 | sd30 | ≥95 evals | stage B: n | best | ≥98 |
|---|---|---:|---:|---:|---:|---:|---:|
| `b3q-ep8` | epochs 8 | **97.2** | 3.0 | 217 | 217 | 98.8% | **21** |
| `b3k-fc200x100` | fc 200,100 | **97.1** | 2.5 | 215 | 215 | **99.2%** | 17 |
| `b3g-ent003` | entropy 0.003 | 96.9 | 2.8 | 153 | 153 | 98.6% | 2 |
| `b3n-fc300x100` | fc 300,100 | 96.9 | 2.4 | **233** | 233 | 99.0% | **21** |
| `b3e-lam95` | λ 0.95 | 96.8 | 2.4 | 179 | 179 | 98.2% | **7** |
| `b3o-g995` | γ 0.995 | 96.7 | **1.8** | 166 | 166 | 98.6% | 10 |
| `b3a-lr3e4-g99` | *the reference* | 96.6 | 2.2 | 108 | 108 | 98.4% | 6 |
| `b3j-lr5e4` | lr 5e-4 | 96.5 | 3.6 | 160 | 160 | **99.0%** | **7** |
| `b3m-fc200` | fc 200 | 96.4 | 2.0 | 131 | 131 | 98.0% | 1 |
| `b3i-lr1e4` | lr 1e-4 | 95.0 | 5.0 | 47 | 47 | 97.4% | 0 |
| `b3l-fc500` | fc 500 | 94.7 | 3.2 | 93 | 93 | 98.4% | 3 |
| `b3p-roll64` | rollout 64 | 94.8 | 2.9 | 104 | 133 | 97.6% | 0 |
| `b3f-lam100` | λ 1.0 | 90.8 | 5.2 | 11 | 11 | 96.2% | 0 |
| `b3h-ent03` | entropy 0.03 | 90.6 | 9.1 | 9 | 9 | 94.8% | 0 |
| `b3r-mb1024` | minibatch 1024 | **89.7** | 4.4 | 16 | 16 | 95.6% | 0 |
| `b3b`, `b3c`, `b3d` | lr 1e-3, lr 3e-3, γ 0.9975 | 85.2, 69.9, 81.6 | 7.2, 18.4, 4.7 | 1, 3, 0 | 1, 3, — | 94.4%, 95.8% | 0 |

`b3b`/`b3c`/`b3d` stopped at the 3M cap and are the arms the cap-inversion finding is measured
against; the rest ran 3M and were then resumed to 10M.

### What it establishes

**No winner.** Nine arms inside **0.8 pp** on best30, and three metrics give three orderings of the top
three (best30 → `b3q`; ≥98%/500 count → `b3e`/`b3j`; stage-B peak → `b3j`). At n=1 per config, that is
one number. **b3 hands b4 the reference config unchanged.**

**One axis moved, 7.5 pp, monotonically — gradient steps per transition.** minibatch 1024 (0.25x) 89.7
· reference (1x) 96.6 · epochs 8 (2x) 97.2. **Rollout size is a second axis:** `b3p-roll64` holds the
ratio fixed, halves the rollout, and loses ~2.5 pp.

**Two hidden layers beat every single-layer width tried, and the record region is where it shows.**
Density of ≥98%/500 checkpoints, which is the statistic that matters for a champion hunt:

| network | parameters | best30 | ≥98%/500 | density |
|---|---:|---:|---:|---:|
| `fc 300,100` | 39,703 | 96.9 | **21** of 233 | **9.0%** |
| `fc 200,100` | 26,603 | **97.1** | 17 of 215 | 7.9% |
| `fc 320` *(reference)* | 10,883 | 96.6 | 6 of 108 | 5.6% |
| `fc 500` | 17,003 | 94.7 | 3 of 93 | 3.2% |
| `fc 200` | 6,803 | 96.4 | 1 of 131 | 0.8% |

**Depth is not simply capacity here:** `fc 500` has more parameters than `fc 200` and is worse on
best30, and `fc 200,100` has more than `fc 500` and is much better — so width past 320 actively hurts
while a second layer helps. The two-layer arms also carry the two highest single checkpoints in the
whole sweep (99.2% and 99.0%). **This is the most promising thread b3 turned up**, and it belongs to a
b5 "better agent" batch: b4 must hold the network at 320 to stay seed-matched against b2.

### Against DQN, at the same protocol

| | transitions | stage-B measurements | best | ≥98%/500 | density | wall clock per arm |
|---|---:|---:|---:|---:|---:|---:|
| **PPO b3, all 15 arms pooled** | 10M | 1,862 | **99.2%** | **95** | **5.10%** | **~3 min** (7 sharing 14 cores) |
| **DQN b2, 4 seeds pooled** | 18M | 1,135 | 99.2% | 5 | 0.44% | ~7-8 h (16 cores) |

**PPO's record-region density is 11.6x DQN's** — 95 checkpoints at ≥98%/500 against 5 — which is the
metric [`../plans/ppo.md`](../plans/ppo.md) §10 pre-registered for this comparison. The best *single*
checkpoint is a tie at 99.2%, and PPO's got there on **5.05M** transitions against b2's 18M.

**The honest depth, and it is the number to quote.** `b3j-lr5e4` @9,469,952 measured **99.0%/500** —
equal to snek2's admitted hall-of-fame record at that depth — and re-measured on a fresh seed at 3,000
episodes: **97.7% [97.1, 98.1]**, a 1.3 pp fall. `b3g-ent003` @8,159,232 fell 98.6% → **96.6%**
[95.9, 97.2]. So:

| policy | 3,000-episode measurement | its 500-episode figure | transitions |
|---|---:|---:|---:|
| `b44a-import` @2739000 — snek2's champion, converted | **98.8%** [98.3, 99.1] | — | 2.74M |
| `b3k-fc200x100` @5046272 — PPO's best | **97.9%** [97.3, 98.3] | 99.2% | 5.05M |
| `b3j-lr5e4` @9469952 | **97.7%** [97.1, 98.1] | 99.0% | 9.47M |
| `b3g-ent003` @8159232 | **96.6%** [95.9, 97.2] | 98.6% | 8.16M |

**The champion is still ahead — 98.8% against PPO's best 97.9%**, and the intervals only touch at
98.3. It also got there on 2.74M transitions against 5.05M, so on sample efficiency to a *champion
checkpoint* the snek2 DQN lineage remains ahead. **Every one of the three PPO highs fell on
re-measurement**, by 1.3, 1.3 and 2.0 pp — which is the whole reason this table exists and the 500-episode
column is the one not to quote. Neither number is a verdict on the algorithms — the champion is a selected best across
snek2's whole history and `b3j` is one arm of a first tuning sweep — but quoting PPO's 99.0%/500 without
this table would be quoting a selected high, which
[`../CLAUDE.md`](../CLAUDE.md) explicitly warns against.

## Batch b1 — the DDQN baseline at every default, seeds 1-4, 3M steps

Closed 2026-08-29. No stage-B column, because **no checkpoint in any of the four reached 95/100 in
stage A**: `screen:95` selects nothing and there is nothing to measure at 500 episodes. The stage-A
numbers are the result.

| arm | config | steps | trailing score | peak best30 | best single eval | ≥95/100 | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `b1a-baseline-seed1` | defaults | 3.00M | 92.26 | 42.1% | 49% | 0 | still rising at the cap |
| `b1b-baseline-seed2` | defaults | 3.00M | 92.65 | 58.3% | 70% | 0 | still rising at the cap |
| `b1c-baseline-seed3` | defaults | 3.00M | 92.87 | 56.7% | 68% | 0 | still rising at the cap |
| `b1d-baseline-seed4` | defaults | 3.00M | 94.17 | **81.9%** | **91%** | 0 | still rising at the cap |

**The phase-3 gate (≥90% perfect) is not met, and the batch does not say the learning code is
wrong.** Three separate reasons, in order of how much they matter:

1. **All four arms were cut off mid-climb.** Not one had plateaued: b1a's perfect rate went 20% at
   500k to ~40% at 3M, b1d's 0% to ~80%, both monotonically, and b1d's highest band is its last
   500k. The 3M cap is the binding constraint, not convergence.
2. **The config was never snek2's record config.** snek3's defaults are chase-safe shaping `c=0.0`
   and **IS weights on**; snek2's record is `c=0.10` at **gate 75** with **IS off**, and its own
   batch 28-29 finding is that *the gate is the lever*. b1 is the no-shaping baseline class, which
   in snek2 was also far from records. Gating phase 3 on it was my mistake — the plan's own phase 5
   names the b29/b47-class config, and that is what the gate needed.
3. **The gate's wording does not say which number it means.** snek2's best *pooled equal-effort* was
   90.50% while its headline 98-99% figures are single selected checkpoints at 500 episodes. Read as
   a trailing rate, "≥90% perfect" sits at snek2's absolute ceiling; read as "some checkpoint gets
   there", b1d's 91% single eval already passes.

The perfect-game counter is alive, which is worth confirming rather than assuming
([`invariants.md`](invariants.md) invariant 2 is about exactly this failing silently): b1d's
non-perfect games average **91.5 of 95**, so the arm is dying with three or four squares left, which
is the endgame this task has always been about.

## The PPO gate arm

**`ppo-smoke` — the phase-6b gate, not a batch arm.** 508k transitions at
[`../plans/ppo.md`](../plans/ppo.md) §7's untuned defaults, on the laptop, 2026-08-29. Kept because it
is the first PPO measurement in this project and the DQN comparison below is the reason 6c exists;
it is deliberately outside the p-series and nothing should be seed-matched against it.

| | transitions | avg score | perfect | notes |
|---|---:|---:|---:|---|
| `ppo-smoke`, stage A 100 eps | 508k | 77.6 | 1% | best single eval 3% at 442k |
| `ppo-smoke`, re-measured 500 eps | 508k | **79.55** | **1.2%** [0.6, 2.6] | median 82, **max 95** — perfect games happen |
| `b1a-d`, stage A 100 eps, matched | ~510k | 85.6 - 91.9 | 6 - 34% | b1's step 85,000 x 6 transitions |

**PPO learns this game, and at a matched sample budget it is behind DQN rather than beside it.** One
untuned arm against four tuned-by-nothing DQN seeds, so the gap is a starting point and not a verdict
— but it is the honest headline, and the four diagnostics say where to push:

| diagnostic | at 508k | reading |
|---|---:|---|
| `explained_variance` | **0.90** | the critic is not the problem, which is the risk §8 ranked highest |
| `approx_kl` | 0.002 | tiny |
| `clip_fraction` | 0.03 | **the clip is barely binding at 0.2, so the learning rate is *low*, not high.** The first knob for b3 |
| `entropy` | 1.086 → **0.27** | committing fast against ln 3 = 1.0986. Whether that is premature is b3's second question |

25.7k transitions/s at fc 320 on the laptop with the stage-A queue on, and `step == transitions`
exactly, which is the whole point of PPO's step unit.

![ppo-smoke](../runs/ppo-smoke.png)

## Imported policies

Not arms: snek2 checkpoints converted to torch, kept as reference policies for A/B. They carry
snek2's training, so their numbers say something about **snek3's environment and measurement**, not
about snek3 as a learner.

| policy | source | rows | episodes/row | pooled perfect | snek2's own number |
|---|---|---:|---:|---|---|
| `b44a-import` @2739000 | `../../snek2/hallOfFame/b44a-lowlr7-b29b-ckpt2739000` | 1 | 3,000 | **98.8%** [98.3, 99.1] | 98.73% / 3,000 |
| `b45a-import`, seed 0 | every checkpoint of `../../snek2/savedPolicies/b45a-lowlr8-b29b` | 3,222 | 100 | **97.287%** | 97.291% |
| `b45a-import`, seed 1 | the same, a second food stream | 3,222 | 100 | **97.318%** | 97.291% |

Regenerated rather than committed, in one deterministic command — see
[`../CLAUDE.md`](../CLAUDE.md). The measurements are
[`../runs/b44a-import_phase1.json`](../runs/b44a-import_phase1.json),
[`../runs/b45a-import_checkpoint_evals_ab3222.json`](../runs/b45a-import_checkpoint_evals_ab3222.json)
and `..._ab3222seed1.json`.

**The 3,222-row pass is the phase-2 gate and it is the strongest measurement in the project.** Mean
per-row difference −0.004 pp against a 0.041 pp standard error, and per-row spread 2.30 pp observed
against 2.30 pp predicted by sampling alone — a ratio of 1.00, which leaves nothing for an
implementation difference. The threshold counts are in [`findings.md`](findings.md), along with why
the count of rows at exactly 100/100 disagreed and why that turned out to be a food stream.

**The 0.07 pp gap is two episodes and it is not evidence of anything.** 2964/3000 against
2962/3000, on different food streams, and the two 95% intervals are identical to a tenth of a point.
What *is* evidence is that the conversion is exact upstream of the measurement — see
[`findings.md`](findings.md).

**`avg_reward` is not comparable and `perfect_percent` is.** snek2 trained `b44a` with chase-safe
shaping at `c=0.10` and `FOOD_DISTANCE_REWARD=0`; the measurement above ran under snek3's defaults,
`c=0.0` and `0.001`. A greedy policy's action is an argmax over its own Q-values, so the reward
config cannot change which moves it plays or what it scores — it only changes the number the reward
terms add up to. That is why a reward figure is never the basis of a comparison here.

## Reading this table

- **`best 500-ep`** is the best row of the arm's stage-B file. It is a *selected* high — a record
  claim needs a fresh measurement of the winner at 1,000+ episodes
  ([`invariants.md`](invariants.md) invariant 9).
- **`≥98%/500 count`** is the width of the arm's record region, and it is the more robust number.
  snek2's champions were single lucky rows about as often as they were real plateaus.
- **`sef`** is `strong_eval_fraction`, the share of the arm's stage-A evals at ≥80% perfect.
  **Compare only at a common step horizon.**
- Every snek3 arm runs 100 episodes per stage-A eval and 500 per stage-B row, so nothing in this
  table needs an episode-count correction. A comparison against a **snek2** number does — see
  [`invariants.md`](invariants.md) invariant 8.
