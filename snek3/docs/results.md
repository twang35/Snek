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


<!-- progress_update: batch b29 -->
## Batch b29 — the `knob` sweep, 1 values x 8 seeds, 100M, closed 2026-09-11

Closed on both boxes' feeds; every arm has its stage-B measurement. One knob off the reference cell (`b27q-hist8-seed17, b27r-hist8-seed18, b27s-hist8-seed19, b27t-hist8-seed20, b27u-hist8-seed21, b27v-hist8-seed22, b27w-hist8-seed23, b27x-hist8-seed24`, marked in the table). Numbers by `tools/progress_update.py`.

| knob | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| lrclip0 | 20,880 | 83.1% | 80.8 86.6 86.5 79.3 85.8 79.7 79.7 86.1 | 6126 | 100.0 | 99.39 (99.2-99.6) | 95.1 | 0.0% | 0.8% | 73.8% |
| **fixed** (reference) | 22,192 | 95.4% | 95.7 94.3 95.6 95.6 95.5 95.6 95.1 95.7 | 15168 | 100.0 | 99.79 (99.7-99.9) | 95.5 | 0.0% | 0.68% | 85.4% |

<!-- reading -->
b28's `hist8` config at 100M with the anneal moved from the horizon to the optimiser: lr 2.5e-4 → 0 and clip 0.2 → 0.001 over the whole cap, and γ 0.99, λ 0.95, entropy 0.01 held fixed; the reference is b27's `hist8` cell, the same cap. **It is worse in every window, not only at the end**: 25M-window density 68.6 / 73.7 / 89.2 / 95.0% against b27 `hist8`'s 82.9 / 95.4 / 99.5 / 99.8, whole-run 83.1% (79.3-86.6) against 95.4 with the eight seeds cleanly below the reference's eight, stage-A ≥98 share 73.8 against 85.4, best30 99.39 against 99.79. Onset is unchanged (4-8M). The passes confirm the top is lower: `hof5000` put 3,749 rows in and **none through the 99.6 gate** (best 99.54, `b29f` @13860864 — before the schedule had moved), so `hof30k` had nothing to run. The prediction — a frozen, denser endgame — is falsified on density and held only on the lower top. **The read is confounded by design**: the arm removes the horizon anneal (γ and λ stay at 0.99 / 0.95 for 100M) at the same time as it adds the optimiser anneal, and the 0-25M deficit, where lr has fallen only a quarter, points at the horizon rather than the decay. A clean test of the paper's schedule would keep γ/λ → 0.999 and add lr/clip → 0 on top.
<!-- /reading -->

### Every arm

| arm | knob | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b29a-lrclip0-seed1` | lrclip0 | 2573 | 80.8% | 602 | 100.0 | 99.3 @89.0M | 95.0 | 0.0% |
| `b29b-lrclip0-seed2` | lrclip0 | 2591 | 86.6% | 843 | 100.0 | 99.4 @66.3M | 94.2 | 0.0% |
| `b29c-lrclip0-seed3` | lrclip0 | 2674 | 86.5% | 872 | 100.0 | 99.4 @94.1M | 96.7 | 0.0% |
| `b29d-lrclip0-seed4` | lrclip0 | 2552 | 79.3% | 538 | 100.0 | 99.2 @75.5M | 93.6 | 0.24% |
| `b29e-lrclip0-seed5` | lrclip0 | 2620 | 85.8% | 1049 | 100.0 | 99.5 @99.2M | 93.4 | 0.0% |
| `b29f-lrclip0-seed6` | lrclip0 | 2606 | 79.7% | 612 | 100.0 | 99.6 @14.0M | 95.2 | 0.0% |
| `b29g-lrclip0-seed7` | lrclip0 | 2592 | 79.7% | 673 | 100.0 | 99.2 @71.5M | 95.9 | 0.0% |
| `b29h-lrclip0-seed8` | lrclip0 | 2672 | 86.1% | 937 | 100.0 | 99.5 @88.8M | 96.6 | 0.0% |

<!-- /progress_update: batch b29 -->

<!-- progress_update: batch b28 -->
## Batch b28 — the `knob` sweep, 1 values x 8 seeds, 200M, closed 2026-09-11

Closed on both boxes' feeds; every arm has its stage-B measurement. One knob off the reference cell (`b27q-hist8-seed17, b27r-hist8-seed18, b27s-hist8-seed19, b27t-hist8-seed20, b27u-hist8-seed21, b27v-hist8-seed22, b27w-hist8-seed23, b27x-hist8-seed24`, marked in the table). Numbers by `tools/progress_update.py`.

| knob | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| **0.5** (reference) | 22,192 | 95.4% | 95.7 94.3 95.6 95.6 95.5 95.6 95.1 95.7 | 15168 | 100.0 | 99.79 (99.7-99.9) | 95.5 | 0.0% | 0.68% | 85.4% |
| hist8a25 | 46,471 | 97.5% | 97.1 97.7 98.4 96.7 97.1 98.5 96.4 97.8 | 36550 | 100.0 | 99.88 (99.8-99.9) | 97.9 | 0.0% | 0.19% | 91.2% |

<!-- reading -->
b27's `hist8` config run to 200M with `SNEK_PPO_ANNEAL_FRACTION` 0.25, so the anneal spans the same first 50M as b27 and the last 150M run at the final values; the reference row is b27's `hist8` cell at 100M. Window for window the first 100M is the same run (0-100M 94.9% density against 95.4), so the whole-run 97.5% is the extra hold reading 99.8% density, 0.19% of evals below 80 against 0.68, best30 99.88 against 99.79. **Every pass is in (closed 2026-09-11).** `hof5000`: 7,718 rows through the 99.6 gate against b27 `hist8`'s 1,836, 606 at ≥99.8 against 85, 38 at 99.9 against 5. `hof30k` on seed 7 at stop target 99.8: 4,599 rows ≥99.6, 1,702 ≥99.7, 152 at ≥99.8 against b27 `hist8`'s 937 / 278 / 9; top `b28k` @162856960 at 29,946 /30,000 (99.82) and `b28m` @131497984 at 29,945, one and two games above the Hall of Fame pair's 29,943-29,944, z ≈ 0.3. **Verdict: the hold widens the plateau by an order of magnitude — seventeen times the 30k rows at 99.8 — and does not raise the top.** No row reaches the 99.83 that would separate from the record at z ≥ 1, so nothing is promoted; the 100M cap is enough for this config and the next lever has to be a knob, not more steps.
<!-- /reading -->

### Every arm

| arm | knob | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b28i-hist8a25-seed9` | hist8a25 | 5839 | 97.1% | 4676 | 100.0 | 99.9 @60.7M | 98.3 | 0.0% |
| `b28j-hist8a25-seed10` | hist8a25 | 5769 | 97.7% | 4563 | 100.0 | 99.9 @185.4M | 97.8 | 0.0% |
| `b28k-hist8a25-seed11` | hist8a25 | 5825 | 98.4% | 4779 | 100.0 | 99.9 @143.2M | 97.9 | 0.0% |
| `b28l-hist8a25-seed12` | hist8a25 | 5799 | 96.7% | 4257 | 100.0 | 99.9 @139.9M | 98.4 | 0.0% |
| `b28m-hist8a25-seed13` | hist8a25 | 5754 | 97.1% | 4507 | 100.0 | 99.8 @131.1M | 96.6 | 0.0% |
| `b28n-hist8a25-seed14` | hist8a25 | 5835 | 98.5% | 4813 | 100.0 | 99.9 @184.9M | 98.0 | 0.0% |
| `b28o-hist8a25-seed15` | hist8a25 | 5803 | 96.4% | 4352 | 100.0 | 99.8 @138.3M | 98.2 | 0.0% |
| `b28p-hist8a25-seed16` | hist8a25 | 5847 | 97.8% | 4603 | 100.0 | 99.9 @196.1M | 98.0 | 0.0% |

<!-- /progress_update: batch b28 -->

<!-- progress_update: batch b27 -->
## Batch b27 — the `obs_history` sweep, 3 values x 8 seeds, 100M, closed 2026-09-09

Closed on both boxes' feeds; every arm has its stage-B measurement. One knob off the reference cell (`b26a-pen01-seed1, b26b-pen01-seed2, b26c-pen01-seed3, b26d-pen01-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| obs_history | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| **0** (reference) | 3,693 | 46.3% | 46.4 52.6 43.0 42.3 | 325 | 100.0 | 98.88 (98.7-99.1) | 93.0 | 0.0% | 1.77% | 45.0% |
| 0 | 15,271 | 49.2% | 61.4 52.9 46.5 47.6 64.7 43.4 33.0 39.9 | 1914 | 100.0 | 99.08 (98.6-99.4) | 95.9 | 0.0% | 1.07% | 47.2% |
| 4 | 22,102 | 93.6% | 92.2 93.6 93.8 94.6 95.5 91.0 95.7 92.0 | 14027 | 100.0 | 99.76 (99.6-99.9) | 96.2 | 0.0% | 0.97% | 84.3% |
| 8 | 22,192 | 95.4% | 95.7 94.3 95.6 95.6 95.5 95.6 95.1 95.7 | 15168 | 100.0 | 99.79 (99.7-99.9) | 95.5 | 0.0% | 0.68% | 85.4% |

<!-- reading -->
Read `hist4` and `hist8` against `hist0`, the batch's own control at the same 100M cap; the bold reference is the same config at 50M and 4 seeds and says only that the longer cap by itself did little (46.3 to 49.2). **Move history is the largest single lever this project has found**: `hist4` 93.6% density (91.0-95.7) and `hist8` 95.4 (94.3-95.7) against 49.2 (33.0-64.7), every history seed above every control seed, stage-A ≥98 share 84-85% against 47.2, best30 99.76-99.79 against 99.08, onset unchanged at ~5-10M. The deep passes carry it through: at 30,000 episodes on seed 7 `hist4` puts 605 rows at ≥99.6 and 19 at 99.8, `hist8` 937 and 9, while `hist0` (124 rows under the old 99.2 gate) tops out at 99.4. The top rows — `b27t` @85065728 at 29,944 /30,000 and `b27k` @77594624 at 29,943 — lead the Hall of Fame's first place (99.65, 29,894) by 0.17 pp, z ≈ 3.9. **Verdict per cell**: `hist0` confirms b26's `pen01` and adds nothing; `hist4` is a full-size win; `hist8` is a small, consistent step further on density (worst seed 94.3 above six of `hist4`'s eight, spread 1.4 pp against 4.7) and on rows through every gate, level on the peaks — the default should move to 8. **Not settled**: what the bits do to the starvation orbit that phase 1 found (`plans/obs-history.md`; the prediction of no effect is falsified, the mechanism finding is not), and whether more history (16) or more hold (b28, the same `hist8` config with 100M more at the final anneal values) adds anything at 30,000 episodes. `b27t` @85065728 and `b27k` @77889536 are in `hallOfFame/HOF.md` as a pair, the new record (2026-09-09).
<!-- /reading -->

### Every arm

| arm | obs_history | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b27a-hist0-seed1` | 0 | 1973 | 61.4% | 367 | 100.0 | 99.4 @86.8M | 95.6 | 0.0% |
| `b27b-hist0-seed2` | 0 | 1994 | 52.9% | 239 | 100.0 | 99.2 @69.3M | 95.7 | 0.0% |
| `b27c-hist0-seed3` | 0 | 1949 | 46.5% | 302 | 100.0 | 99.3 @92.6M | 95.8 | 0.0% |
| `b27d-hist0-seed4` | 0 | 1951 | 47.6% | 136 | 99.8 | 98.6 @75.6M | 96.7 | 0.0% |
| `b27e-hist0-seed5` | 0 | 2060 | 64.7% | 590 | 100.0 | 99.4 @92.3M | 95.7 | 0.0% |
| `b27f-hist0-seed6` | 0 | 1780 | 43.4% | 121 | 99.8 | 99.2 @88.7M | 95.8 | 0.0% |
| `b27g-hist0-seed7` | 0 | 1742 | 33.0% | 70 | 99.8 | 98.8 @79.2M | 95.2 | 0.0% |
| `b27h-hist0-seed8` | 0 | 1822 | 39.9% | 89 | 99.8 | 98.7 @77.5M | 96.4 | 0.0% |
| `b27i-hist4-seed9` | 4 | 2739 | 92.2% | 1648 | 100.0 | 99.8 @95.2M | 97.0 | 0.0% |
| `b27j-hist4-seed10` | 4 | 2738 | 93.6% | 1644 | 100.0 | 99.7 @78.4M | 96.1 | 0.0% |
| `b27k-hist4-seed11` | 4 | 2783 | 93.8% | 1792 | 100.0 | 99.8 @83.5M | 96.6 | 0.03% |
| `b27l-hist4-seed12` | 4 | 2774 | 94.6% | 1908 | 100.0 | 99.7 @58.9M | 96.5 | 0.0% |
| `b27m-hist4-seed13` | 4 | 2790 | 95.5% | 1869 | 100.0 | 99.8 @91.9M | 96.3 | 0.03% |
| `b27n-hist4-seed14` | 4 | 2737 | 91.0% | 1462 | 100.0 | 99.6 @74.7M | 96.2 | 0.0% |
| `b27o-hist4-seed15` | 4 | 2779 | 95.7% | 1991 | 100.0 | 99.9 @60.2M | 95.9 | 0.0% |
| `b27p-hist4-seed16` | 4 | 2762 | 92.0% | 1713 | 100.0 | 99.8 @75.7M | 95.2 | 0.0% |
| `b27q-hist8-seed17` | 8 | 2799 | 95.7% | 1938 | 100.0 | 99.8 @71.3M | 96.8 | 0.0% |
| `b27r-hist8-seed18` | 8 | 2698 | 94.3% | 1814 | 100.0 | 99.7 @67.6M | 92.7 | 0.0% |
| `b27s-hist8-seed19` | 8 | 2780 | 95.6% | 1929 | 100.0 | 99.9 @59.3M | 96.4 | 0.0% |
| `b27t-hist8-seed20` | 8 | 2840 | 95.6% | 1916 | 100.0 | 99.8 @92.3M | 96.4 | 0.0% |
| `b27u-hist8-seed21` | 8 | 2753 | 95.5% | 1845 | 100.0 | 99.8 @76.9M | 95.8 | 0.0% |
| `b27v-hist8-seed22` | 8 | 2728 | 95.6% | 1916 | 100.0 | 99.8 @97.0M | 94.8 | 0.0% |
| `b27w-hist8-seed23` | 8 | 2757 | 95.1% | 1893 | 100.0 | 99.8 @99.8M | 94.6 | 0.0% |
| `b27x-hist8-seed24` | 8 | 2837 | 95.7% | 1917 | 100.0 | 99.7 @62.2M | 96.4 | 0.0% |

<!-- /progress_update: batch b27 -->

<!-- progress_update: batch b26 -->
## Batch b26 — the `step_penalty` sweep, 4 values x 4 seeds, 50M, closed 2026-09-07

Closed on both boxes' feeds; every arm has its stage-B measurement. One knob off the reference cell (`b24a-hzanneal50-seed1, b24b-hzanneal50-seed2, b24c-hzanneal50-seed3, b24d-hzanneal50-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| step_penalty | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| **0** (reference) | 17,338 | 62.6% | 54.8 69.9 62.1 63.1 | 3761 | 100.0 | 99.40 (99.3-99.5) | 97.9 | 0.0% | 0.9% | 56.9% |
| 0 | 3,095 | 25.5% | 29.5 33.5 16.3 20.2 | 77 | 99.8 | 98.47 (98.0-98.7) | 92.4 | 0.0% | 2.08% | 34.7% |
| 0.0001 | 3,183 | 32.7% | 35.7 40.8 28.7 23.3 | 117 | 99.8 | 98.42 (98.1-98.7) | 92.8 | 0.0% | 1.33% | 35.8% |
| 0.001 | 3,012 | 26.1% | 24.9 32.0 23.2 23.3 | 69 | 99.8 | 98.33 (98.1-98.5) | 91.8 | 0.0% | 2.16% | 33.1% |
| 0.01 | 3,693 | 46.3% | 46.4 52.6 43.0 42.3 | 325 | 100.0 | 98.88 (98.7-99.1) | 93.0 | 0.0% | 1.77% | 45.0% |

<!-- reading -->
Read against `pen0`, the batch's own control (the b24 horizon-anneal config at 50M under the 26-value observation), not the bold reference — that is b24 itself at 200M in the 30-value era and is there for orientation only. **A step penalty of 0.01 is a real lever at n=4 and the other two values are not**: `pen01` reads 46.3% density (42.3-52.6 per seed) against the control's 25.5 (16.3-33.5), every one of its four seeds above every one of the other twelve arms (Mann-Whitney p = 0.029 against each cell), best30 98.88 (98.7-99.1) against 98.47, stage-A ≥98 share 45.0% against 34.7, and the deep passes follow: 165 `hof5000` rows (at the 99.2 cut) against 42, the batch's only ≥99.2 /5,000 rows (`b26b` @38.9M, 99.2, twice) and its only 30k rows (99.1). `pen0001` (32.7%, 23.3-40.8) and `pen001` (26.1%) sit inside the control's spread. The scale explains the shape: a perfect game is ~1,800 steps, so 0.01 a step is ~18 of return against ~95 from food and 100 for the perfect game — about a tenth of the return — while 0.001 is 1% and 0.0001 is noise; only the top value is large enough to be felt, and **the curve has not turned**, so 0.02 and 0.05 are the next cells. What it did not change: onset (50% at 1.1-1.2M and 80% at 2.7-3.0M on every cell), late entropy (0.035 everywhere) and approx KL — the penalty changes what the converged policy does, not how fast it gets there. The one diagnostic that moved with it is late explained variance, 0.82 against 0.77-0.81 (p = 0.029): a per-step cost makes the return depend on time-to-food, which is a thing the critic can predict, and the policy gradient inherits the signal. The mechanism this fits is today's finding that the best checkpoints fail by orbiting reachable food in a closed loop: a per-step cost is the only reward term that charges for a lap that does not eat, and 0.01 makes a 100-step idle orbit cost a meal. Not settled: whether `pen01`'s failures are in fact fewer starvations (`tools/death_trace.py` on `b26b` @38.9M against a `pen0` checkpoint answers it directly); whether the lever holds on the ladder top (γ 0.999, T 512, `mse`) rather than the anneal; and how much of the control's 25.5% against b24's 60.5 is the 50M cap and how much the new observation — b24's config was never run at 50M in the old era, so this batch cannot separate them.
<!-- /reading -->

### Every arm

| arm | step_penalty | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b26m-pen0-seed13` | 0 | 851 | 29.5% | 23 | 99.8 | 98.5 @42.4M | 92.5 | 0.14% |
| `b26n-pen0-seed14` | 0 | 838 | 33.5% | 38 | 99.8 | 98.7 @49.0M | 93.3 | 0.0% |
| `b26o-pen0-seed15` | 0 | 698 | 16.3% | 5 | 99.2 | 98.0 @41.5M | 92.9 | 0.0% |
| `b26p-pen0-seed16` | 0 | 708 | 20.2% | 11 | 99.8 | 98.7 @49.7M | 91.0 | 0.0% |
| `b26i-pen0001-seed9` | 0.0001 | 838 | 35.7% | 41 | 99.8 | 98.6 @48.0M | 92.3 | 0.0% |
| `b26j-pen0001-seed10` | 0.0001 | 860 | 40.8% | 62 | 99.8 | 98.7 @39.2M | 94.1 | 0.0% |
| `b26k-pen0001-seed11` | 0.0001 | 811 | 28.7% | 10 | 99.8 | 98.3 @35.8M | 92.3 | 0.0% |
| `b26l-pen0001-seed12` | 0.0001 | 674 | 23.3% | 4 | 99.0 | 98.1 @27.9M | 92.7 | 0.07% |
| `b26e-pen001-seed5` | 0.001 | 732 | 24.9% | 13 | 99.4 | 98.1 @42.8M | 91.2 | 0.0% |
| `b26f-pen001-seed6` | 0.001 | 828 | 32.0% | 28 | 99.8 | 98.4 @49.3M | 90.9 | 0.0% |
| `b26g-pen001-seed7` | 0.001 | 659 | 23.2% | 12 | 99.6 | 98.3 @40.4M | 91.3 | 0.0% |
| `b26h-pen001-seed8` | 0.001 | 793 | 23.3% | 16 | 99.2 | 98.5 @47.8M | 93.8 | 0.0% |
| `b26a-pen01-seed1` | 0.01 | 924 | 46.4% | 79 | 99.8 | 98.9 @45.8M | 91.2 | 0.0% |
| `b26b-pen01-seed2` | 0.01 | 997 | 52.6% | 128 | 100.0 | 99.1 @37.1M | 93.8 | 0.0% |
| `b26c-pen01-seed3` | 0.01 | 861 | 43.0% | 53 | 99.8 | 98.7 @48.5M | 91.9 | 0.0% |
| `b26d-pen01-seed4` | 0.01 | 911 | 42.3% | 65 | 99.6 | 98.8 @31.5M | 95.2 | 0.0% |

<!-- /progress_update: batch b26 -->

<!-- progress_update: batch b25 -->
## Batch b25 — the `knob` sweep, 1 values x 8 seeds, 200M, closed 2026-09-07

Closed on both boxes' feeds; every arm has its stage-B measurement. One knob off the reference cell (`b23e-g999roll512msehold-seed1, b23f-g999roll512msehold-seed2, b23g-g999roll512msehold-seed3, b23h-g999roll512msehold-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| knob | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| laddertop | 20,942 | 77.5% | 80.5 64.3 81.0 77.5 72.1 82.8 77.4 83.0 | 5626 | 100.0 | 99.34 (99.1-99.6) | 97.5 | 0.01% | 0.45% | 72.7% |
| **base** (reference) | 2,204 | 64.4% | 65.6 61.2 48.2 80.9 | 395 | 100.0 | 98.95 (98.4-99.3) | 90.3 | 0.0% | 1.4% | 58.2% |

<!-- reading -->
Read against the bold reference, b23's `g999roll512msehold` cell — the same config (γ 0.999, λ 0.99, T 512, `mse`, clip 0.2 → 0.001 held from 80%) at 50M and 4 seeds; b25 is that ladder top at 200M and 8 seeds, the first champion attempt on it. **The longer cap paid on every column**: 77.5% density (64.3-83.0 per seed, seven of eight above the reference's best seed) against 64.4, best30 99.34 (99.1-99.6) against 98.95, stage-A ≥98 share 72.7% against 58.2, `sef` 97.5 against 90.3, and the collapses stayed gone — 0.01% of evals below 50, 0.45% below 80 against 1.4. The `hof5000` pass took 5,626 candidates (at the 99 cut this batch ran under; 721 clear the 99.2 cut set on 2026-09-07) and every arm's best 5,000-episode row is 99.2-99.8: `b25a` @106.2M at 99.8, `b25h` @121.3M at 99.6 with 362 rows at ≥99.2, `b25f` @68.2M at 99.5. **At 30,000 episodes, `b25a` @106168320 reads 99.60 [99.52-99.67]** — level with the HOF's second place (99.55) and inside the record's interval (99.65, [99.6-99.7]), the first checkpoint outside the γ 1.00 cell to get there; `b25c` reads 99.3 and `b25b` 99.1. **The 30k pass covers only three of eight arms**: it failed on `b25d` at 15:03 when the observation changed from 30 to 26 values under it (era `obs26-20260907`, commit `b1259c912`) and the remaining arms' 30-value checkpoints no longer load in the current environment; `b25h`, `b25f` and `b25e` are the unmeasured arms with the most to show, and the pass is the other agent's to finish under the old era. The spec's predictions held: fewer evals below 80 than b24 (0.45 against 0.83), a ≥99.3 /30k check (99.6). Not settled: whether a fresh 30,000 on `b25a` @106168320 confirms 99.6 (the `hof-promote` gate), and what the five unmeasured arms hold at depth.
<!-- /reading -->

### Every arm

| arm | knob | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b25a-laddertop-seed1` | laddertop | 2645 | 80.5% | 811 | 100.0 | 99.5 @184.1M | 96.6 | 1.72% |
| `b25b-laddertop-seed2` | laddertop | 2453 | 64.3% | 324 | 100.0 | 99.1 @36.0M | 96.7 | 1.5% |
| `b25c-laddertop-seed3` | laddertop | 2626 | 81.0% | 939 | 100.0 | 99.5 @188.9M | 96.9 | 0.0% |
| `b25d-laddertop-seed4` | laddertop | 2613 | 77.5% | 570 | 100.0 | 99.3 @141.2M | 98.2 | 0.0% |
| `b25e-laddertop-seed5` | laddertop | 2531 | 72.1% | 527 | 100.0 | 99.3 @127.1M | 97.5 | 0.03% |
| `b25f-laddertop-seed6` | laddertop | 2731 | 82.8% | 770 | 100.0 | 99.2 @156.8M | 98.8 | 0.0% |
| `b25g-laddertop-seed7` | laddertop | 2627 | 77.4% | 589 | 100.0 | 99.2 @109.2M | 96.6 | 1.69% |
| `b25h-laddertop-seed8` | laddertop | 2716 | 83.0% | 1096 | 100.0 | 99.6 @183.0M | 98.4 | 0.0% |

<!-- /progress_update: batch b25 -->

<!-- progress_update: batch b24 -->
## Batch b24 — the `knob` sweep, 1 values x 8 seeds, 200M, closed 2026-09-07

Closed on both boxes' feeds; every arm has its stage-B measurement. One knob off the reference cell (`b23e-g999roll512msehold-seed1, b23f-g999roll512msehold-seed2, b23g-g999roll512msehold-seed3, b23h-g999roll512msehold-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| knob | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| hzanneal50 | 34,789 | 60.5% | 54.8 69.9 62.1 63.1 56.9 55.0 60.1 61.0 | 6277 | 100.0 | 99.30 (99.1-99.5) | 97.8 | 0.0% | 0.83% | 56.3% |
| **base** (reference) | 2,204 | 64.4% | 65.6 61.2 48.2 80.9 | 395 | 100.0 | 98.95 (98.4-99.3) | 90.3 | 0.0% | 1.4% | 58.2% |

<!-- reading -->
Read against the same bold reference as b25 (b23's ladder top at 50M); b24 is the user's horizon-anneal config at 200M and 8 seeds — γ 0.99 → 0.999 and λ 0.95 → 0.999 annealed over the first half on the existing ramp, entropy 0.01 → 0.001, `huber` value loss, minibatch 512, lr 2.5e-4, T 256 — so the two 200M batches are the ladder top against an anneal that reaches the same horizon by a different road. **It arrives lower on density and level on the peak**: 60.5% (54.8-69.9) against b25's 77.5 and the reference's 64.4, best30 99.30 (99.1-99.5) against 99.34, stage-A ≥98 share 56.3% against 72.7, and twice b25's evals below 80 (0.83% against 0.45, still under the reference's 1.4). T 256 screens twice the checkpoints (34,789 rows against 20,942), so its `hof5000` pass took 6,277 candidates — the largest this project has run — with per-arm bests 99.2-99.6 (`b24a` @195.3M, `b24b` @141.1M and `b24d` @166.9M at 99.6; `b24d` has 242 rows at ≥99.2, `b24b` 167). At 30,000 episodes `b24a` @196706304 reads 99.5 and `b24b` @141197312 99.4 — the first ties the HOF's third place, neither reaches 99.65. **The 30k pass covers two of eight arms** for the same reason as b25's: it failed on `b24c` after the observation change and the six remaining arms cannot load in the 26-value environment; `b24d` is the unmeasured arm with the most at ≥99.2. Against the spec: "earlier onset than b25, then a denser but less stable second half" — less stable held (0.83 against 0.45 below 80), denser did not (60.5 against 77.5); "best30 close" held (99.30 against 99.34). Verdict: the anneal is a working 200M config that matches the ladder top's peak and trails it on density and stability; the ladder top is the one to carry forward.
<!-- /reading -->

### Every arm

| arm | knob | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b24a-hzanneal50-seed1` | hzanneal50 | 4200 | 54.8% | 625 | 100.0 | 99.3 @194.5M | 98.0 | 0.0% |
| `b24b-hzanneal50-seed2` | hzanneal50 | 4555 | 69.9% | 1265 | 100.0 | 99.5 @120.1M | 97.9 | 0.0% |
| `b24c-hzanneal50-seed3` | hzanneal50 | 4301 | 62.1% | 806 | 100.0 | 99.3 @128.9M | 97.6 | 0.0% |
| `b24d-hzanneal50-seed4` | hzanneal50 | 4282 | 63.1% | 1065 | 100.0 | 99.5 @183.1M | 98.0 | 0.0% |
| `b24e-hzanneal50-seed5` | hzanneal50 | 4249 | 56.9% | 592 | 100.0 | 99.1 @157.6M | 97.9 | 0.0% |
| `b24f-hzanneal50-seed6` | hzanneal50 | 4205 | 55.0% | 627 | 100.0 | 99.3 @171.5M | 98.1 | 0.0% |
| `b24g-hzanneal50-seed7` | hzanneal50 | 4375 | 60.1% | 613 | 100.0 | 99.3 @160.2M | 96.3 | 0.0% |
| `b24h-hzanneal50-seed8` | hzanneal50 | 4622 | 61.0% | 684 | 100.0 | 99.1 @129.7M | 98.3 | 0.0% |

<!-- /progress_update: batch b24 -->

<!-- progress_update: batch b23 -->
## Batch b23 — the `knob` sweep, 2 values x 4 seeds, 50M, closed 2026-09-06

Closed on both boxes' feeds; every arm has its stage-B measurement. One knob off the reference cell (`b22e-g999roll512-seed1, b22f-g999roll512-seed2, b22g-g999roll512-seed3, b22h-g999roll512-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| knob | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| g999roll512mse | 2,275 | 61.6% | 49.8 60.0 56.4 78.1 | 315 | 100.0 | 98.92 (98.4-99.2) | 93.2 | 0.48% | 1.52% | 59.8% |
| g999roll512msehold | 2,204 | 64.4% | 65.6 61.2 48.2 80.9 | 395 | 100.0 | 98.95 (98.4-99.3) | 90.3 | 0.0% | 1.4% | 58.2% |
| **base** (reference) | 1,805 | 32.7% | 27.6 32.8 36.2 34.1 | 75 | 99.6 | 98.35 (98.1-98.6) | 87.8 | 3.54% | 4.98% | 41.4% |

<!-- reading -->
Read against the bold reference, b22's `g999roll512` cell (32.7% density, best30 98.35, 3.54% of evals below 50, 4.98% below 80), the rung below in the ladder. **Rung 3, + the `mse` value loss, is the largest single step this project has measured**: 61.6% density against 32.7 — every seed (49.8-78.1) above every reference seed (27.6-36.2), Mann-Whitney p = 0.029 — with best30 98.92 (98.4-99.2), stage-A ≥98 share 59.8% against 41.4, and the drawdowns γ 0.999 and λ 0.99 had added gone: 0.48% of evals below 50 against 3.54, 1.52% below 80 against 4.98. At depth it is 315 `hof5000` rows at mean 98.55 (105 at ≥98.73, 47 at ≥99) and 47 rows at 30,000 episodes, 28 of them at ≥99, the best 99.2 three times on `b23d` @24-28M — the reference cell had one 30k row and b9's whole λ 0.99 cell none at 99. **Rung 4, + the clip anneal 0.2 → 0.001 held for the last 10M, adds nothing on density at n=4** (64.4%, 48.2-80.9, inside rung 3's spread) and takes the last of the collapses out: 0.0% below 50 on three of four seeds, 1.4% below 80. Its `hof5000` pass is the richest of any cell measured (395 rows, mean 98.60, 162 at ≥98.73, 78 at ≥99) and its 30k pass has 78 rows, 36 at ≥99, best 99.2 (`b23e` @47.3M). **Nothing here beats the HOF's 99.65 at 30,000**: the two mse cells top out at 99.2, below the fourth-place 99.30 — what they do is produce ≥99 /30k checkpoints in bulk (64 in one 8-arm wave, against a dozen from the entire b9-b21 sweep) rather than a taller peak. Falsified: the spec's "the late stage-B density lift on top of rung 3" for the hold — the lift b17 measured on b7's base does not repeat on a base that already has few late collapses. Not settled: whether `mse`'s gain needs γ 0.999 and rollout 512 under it (it read +5 pp on b7's base in b19, +29 pp here — an interaction, or the horizon's density finally held by a critic that does not collapse it), which is the next rung to run backwards; and 8 seeds at a longer cap for a champion attempt. Both cells ran on the laptop in one wave, 44 minutes of training at T 512's quarter-rate stage A.
<!-- /reading -->

### Every arm

| arm | knob | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b23a-g999roll512mse-seed1` | g999roll512mse | 558 | 49.8% | 36 | 99.6 | 98.4 @26.0M | 94.4 | 0.41% |
| `b23b-g999roll512mse-seed2` | g999roll512mse | 572 | 60.0% | 69 | 99.8 | 99.1 @15.9M | 93.8 | 0.55% |
| `b23c-g999roll512mse-seed3` | g999roll512mse | 528 | 56.4% | 53 | 99.8 | 99.0 @34.8M | 93.2 | 0.0% |
| `b23d-g999roll512mse-seed4` | g999roll512mse | 617 | 78.1% | 157 | 100.0 | 99.2 @46.5M | 91.6 | 1.12% |
| `b23e-g999roll512msehold-seed1` | g999roll512msehold | 524 | 65.6% | 150 | 100.0 | 99.3 @34.7M | 91.1 | 0.0% |
| `b23f-g999roll512msehold-seed2` | g999roll512msehold | 552 | 61.2% | 78 | 99.8 | 98.8 @12.5M | 90.0 | 0.0% |
| `b23g-g999roll512msehold-seed3` | g999roll512msehold | 537 | 48.2% | 27 | 99.4 | 98.4 @44.8M | 93.1 | 0.14% |
| `b23h-g999roll512msehold-seed4` | g999roll512msehold | 591 | 80.9% | 140 | 99.8 | 99.3 @23.1M | 86.9 | 0.0% |

<!-- /progress_update: batch b23 -->

<!-- progress_update: batch b22 -->
## Batch b22 — the `knob` sweep, 2 values x 4 seeds, 50M, closed 2026-09-06

Closed on both boxes' feeds; every arm has its stage-B measurement. One knob off the reference cell (`b9bw-lam99-seed1, b9bx-lam99-seed2, b9by-lam99-seed3, b9bz-lam99-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| knob | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| g999 | 6,530 | 36.3% | 44.6 28.5 25.3 44.3 | 386 | 99.8 | 98.70 (98.3-98.9) | 89.1 | 3.59% | 8.04% | 38.4% |
| g999roll512 | 1,805 | 32.7% | 27.6 32.8 36.2 34.1 | 75 | 99.6 | 98.35 (98.1-98.6) | 87.8 | 3.54% | 4.98% | 41.4% |
| **base** (reference) | 5,173 | 27.3% | 27.2 24.1 31.4 26.0 | 138 | 100.0 | 98.33 (98.3-98.4) | 90.6 | 0.77% | 6.43% | 28.9% |

<!-- reading -->
Read against the bold reference, b9's λ 0.99 cell `b9bw`-`b9bz` (27.3% density, best30 98.33, 0.77% of evals below 50, 6.43% below 80) — the base of the corner-grid ladder (`plans/sweep-analysis.md`, `docs/sweep.md`). **Rung 1, γ 0.999 on λ 0.99, adds density and adds drawdown.** 36.3% against 27.3 (per-seed 25.3-44.6 over 24.1-31.4, a lean, not a separation) — above both parents (b10's γ 0.999 at 30.7% on λ 0.98, b9's λ 0.99 at 27.3 on γ 0.99), so the two horizon gains partly add through the shared 1/(1−γλ) — with best30 98.70 (98.3-98.9) and 386 `hof5000` candidates against 138. The price is the collapse column: 3.59% of evals below 50 against 0.77, 8.0% below 80 against 6.4 — a fifth of the way to γ 1.0's regime, exactly the risk the spec named. At depth the cell holds 26 rows at 30,000 episodes, 11 of them at ≥99 and the best 99.1 (`b22a` @40.5M, `b22d` @44.4M), where the reference had three rows and none at 99. **Rung 2, + rollout 512, does not add on top of rung 1**: 32.7% (27.6-36.2), *below* rung 1 on every readout but stability, where it halves the share below 80 (8.0 → 5.0) and leaves the share below 50 where it was (3.5%). Its 1,805 rows are a quarter of rung 1's because a checkpoint lands every 65,536 transitions at T 512, so the shares are the comparison; 75 `hof5000` candidates, one row at 30,000 (99.1). The prediction "the densest cell measured so far" is falsified: b14's +12 pp for rollout 512 was measured on γ 0.99, and on γ 0.999 the longer rollout trades density for stability instead. Not settled at n=4: whether rung 1's extra density survives at 8 seeds, and whether γ 0.999 alone (no rollout change) is the better base for the stabilisers in b23 — b23 was built on rung 2. Both cells ran on the desktop in one wave.
<!-- /reading -->

### Every arm

| arm | knob | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b22a-g999-seed1` | g999 | 1789 | 44.6% | 177 | 99.8 | 98.9 @36.8M | 89.4 | 4.82% |
| `b22b-g999-seed2` | g999 | 1451 | 28.5% | 33 | 99.6 | 98.7 @40.5M | 90.9 | 2.71% |
| `b22c-g999-seed3` | g999 | 1564 | 25.3% | 34 | 99.6 | 98.3 @29.1M | 87.0 | 3.27% |
| `b22d-g999-seed4` | g999 | 1726 | 44.3% | 142 | 99.8 | 98.9 @43.9M | 89.0 | 3.92% |
| `b22e-g999roll512-seed1` | g999roll512 | 431 | 27.6% | 6 | 99.6 | 98.3 @32.5M | 85.3 | 3.67% |
| `b22f-g999roll512-seed2` | g999roll512 | 470 | 32.8% | 23 | 99.6 | 98.4 @39.8M | 89.1 | 4.17% |
| `b22g-g999roll512-seed3` | g999roll512 | 450 | 36.2% | 28 | 99.4 | 98.6 @40.4M | 86.9 | 3.42% |
| `b22h-g999roll512-seed4` | g999roll512 | 454 | 34.1% | 18 | 99.4 | 98.1 @31.0M | 89.8 | 2.39% |

<!-- /progress_update: batch b22 -->

<!-- progress_update: batch b18 -->
## Batch b18 — the `gradient_clipping` sweep, 6 values x 4 seeds, 50M, closed 2026-09-06

Closed on both boxes' feeds; every arm has its stage-B measurement. One knob off the reference cell (`b7aa-fc320-seed1, b7ab-fc320-seed2, b7ac-fc320-seed3, b7ad-fc320-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| gradient_clipping | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| 0 | 4,370 | 17.3% | 16.4 18.0 14.8 19.7 | 69 | 99.6 | 97.90 (97.7-98.3) | 93.3 | 0.23% | 3.71% | 22.0% |
| 0.1 | 4,241 | 19.1% | 15.9 12.6 12.0 32.4 | 84 | 99.8 | 98.17 (97.8-98.6) | 92.0 | 0.18% | 4.59% | 21.7% |
| 0.25 | 4,285 | 16.7% | 9.7 14.0 18.6 22.8 | 71 | 99.8 | 98.05 (97.6-98.5) | 92.4 | 0.07% | 4.18% | 21.7% |
| **0.5** (reference) | 4,003 | 17.3% | 18.5 19.0 16.5 15.1 | 50 | 99.6 | 97.75 (97.7-97.8) | 90.9 | 0.29% | 6.18% | 20.4% |
| 1.0 | 3,935 | 16.7% | 16.5 14.9 22.3 12.9 | 63 | 99.8 | 97.80 (97.7-98.0) | 91.0 | 0.05% | 5.5% | 19.5% |
| 2.0 | 4,570 | 19.5% | 18.7 20.5 16.0 22.3 | 101 | 99.8 | 98.05 (97.9-98.2) | 93.3 | 0.18% | 3.21% | 24.6% |
| 5.0 | 4,050 | 16.8% | 12.4 23.1 17.8 13.1 | 54 | 99.6 | 98.03 (97.7-98.4) | 89.4 | 0.45% | 7.54% | 20.9% |

<!-- reading -->
Read against the bold reference, b7's λ 0.98 cell `b7aa`-`b7ad` (17.3% density, best30 97.75, 6.2% of evals below 80%), which b15-b21 were generated from. **The gradient-norm clip is a no-op at this base from off to 5.0.** Density is inside the reference's noise at every cell (16.7-19.5% against 17.3), best30 spans 97.80-98.17 with the base at the bottom of the range, and no cell's best row beats 99.8. **The null check answered the spec's question**: clipping *off* (`gc0`) reads 17.3%, the base's density to the decimal, with 3.7% of evals below 80% against 6.2 — so the base's collapses are not rare huge gradients, they are policy-level, the prediction's second branch. 0.1 ("slower, steadier") is neither: its stage-A density is 21.7 against 20.4 and its 19.1% is one seed, `b18h` at 32.4% with the batch's best best30 (98.6), over three at 12-16%. 2.0 is the batch's densest and most stable cell (19.5%, 3.21% below 80%, 101 `hof5000` candidates) and 5.0 its least stable (7.54%, `b18v` 1.75% below 50%) — and since 0 and 5.0 are both effectively off, **their 3.8 pp spread on the below-80% column is that column's noise floor at n=4**, which also brackets b21's "every cell a little more stable" and b20's 256-512 lanes reading as a stability lever: differences of that size are not evidence. At depth the batch's best are `b18t-gc2-seed4` @11452416 and `b18h-gc01-seed4` @11976704, both **98.9 /30,000 [98.8, 99.0]**, below the HOF's 99.30 third place and both from checkpoints near 11-12M, as `b17cl`'s HOF entry was. The clip stays at 0.5 and leaves the grid; with it the one-knob sweeps at λ 0.98 (b15-b21) are done.
<!-- /reading -->

### Every arm

| arm | gradient_clipping | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b18a-gc0-seed1` | 0 | 1074 | 16.4% | 16 | 99.6 | 97.8 @28.9M | 94.2 | 0.17% |
| `b18b-gc0-seed2` | 0 | 1123 | 18.0% | 21 | 99.4 | 97.8 @22.8M | 91.0 | 0.28% |
| `b18c-gc0-seed3` | 0 | 1059 | 14.8% | 6 | 99.2 | 97.7 @42.6M | 94.0 | 0.47% |
| `b18d-gc0-seed4` | 0 | 1114 | 19.7% | 26 | 99.4 | 98.3 @25.3M | 94.1 | 0.0% |
| `b18e-gc01-seed1` | 0.1 | 950 | 15.9% | 14 | 99.8 | 98.1 @25.1M | 93.2 | 0.1% |
| `b18f-gc01-seed2` | 0.1 | 931 | 12.6% | 11 | 99.4 | 98.2 @10.9M | 90.5 | 0.57% |
| `b18g-gc01-seed3` | 0.1 | 1083 | 12.0% | 8 | 99.2 | 97.8 @32.5M | 92.1 | 0.0% |
| `b18h-gc01-seed4` | 0.1 | 1277 | 32.4% | 51 | 99.8 | 98.6 @45.5M | 92.1 | 0.27% |
| `b18i-gc025-seed1` | 0.25 | 949 | 9.7% | 6 | 99.4 | 97.6 @36.3M | 93.1 | 0.34% |
| `b18j-gc025-seed2` | 0.25 | 1035 | 14.0% | 17 | 99.8 | 97.7 @29.8M | 93.4 | 0.1% |
| `b18k-gc025-seed3` | 0.25 | 1049 | 18.6% | 22 | 99.2 | 98.5 @46.2M | 92.3 | 0.03% |
| `b18l-gc025-seed4` | 0.25 | 1252 | 22.8% | 26 | 99.4 | 98.4 @23.9M | 90.7 | 0.03% |
| `b18m-gc1-seed1` | 1.0 | 915 | 16.5% | 19 | 99.8 | 97.7 @37.0M | 91.1 | 0.03% |
| `b18n-gc1-seed2` | 1.0 | 930 | 14.9% | 11 | 99.4 | 97.8 @41.7M | 88.9 | 0.07% |
| `b18o-gc1-seed3` | 1.0 | 1059 | 22.3% | 25 | 99.6 | 98.0 @42.5M | 90.0 | 0.03% |
| `b18p-gc1-seed4` | 1.0 | 1031 | 12.9% | 8 | 99.2 | 97.7 @30.2M | 94.2 | 0.13% |
| `b18q-gc2-seed1` | 2.0 | 1223 | 18.7% | 32 | 99.8 | 98.2 @22.6M | 94.8 | 0.13% |
| `b18r-gc2-seed2` | 2.0 | 1078 | 20.5% | 26 | 99.6 | 97.9 @26.3M | 91.1 | 0.41% |
| `b18s-gc2-seed3` | 2.0 | 1047 | 16.0% | 14 | 99.4 | 97.9 @20.5M | 93.1 | 0.24% |
| `b18t-gc2-seed4` | 2.0 | 1222 | 22.3% | 29 | 99.8 | 98.2 @46.8M | 94.1 | 0.07% |
| `b18u-gc5-seed1` | 5.0 | 872 | 12.4% | 8 | 99.2 | 98.0 @39.5M | 86.6 | 0.49% |
| `b18v-gc5-seed2` | 5.0 | 1060 | 23.1% | 22 | 99.6 | 98.4 @39.2M | 90.4 | 1.75% |
| `b18w-gc5-seed3` | 5.0 | 1096 | 17.8% | 12 | 99.2 | 98.0 @39.5M | 89.6 | 0.4% |
| `b18x-gc5-seed4` | 5.0 | 1022 | 13.1% | 12 | 99.4 | 97.7 @33.8M | 91.1 | 0.2% |

<!-- /progress_update: batch b18 -->

<!-- progress_update: batch b21 -->
## Batch b21 — the `chase_safe_shaping` sweep, 6 values x 4 seeds, 50M, closed 2026-09-06

Closed on both boxes' feeds; every arm has its stage-B measurement. One knob off the reference cell (`b7aa-fc320-seed1, b7ab-fc320-seed2, b7ac-fc320-seed3, b7ad-fc320-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| chase_safe_shaping | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| 0.0 | 4,239 | 17.1% | 11.9 18.7 18.3 18.5 | 58 | 99.6 | 98.08 (97.8-98.3) | 92.7 | 0.07% | 3.56% | 21.7% |
| 0.05 | 3,919 | 16.3% | 12.9 17.0 19.6 14.7 | 51 | 99.6 | 97.80 (97.7-97.9) | 91.4 | 0.22% | 4.57% | 20.1% |
| **0.1** (reference) | 4,003 | 17.3% | 18.5 19.0 16.5 15.1 | 50 | 99.6 | 97.75 (97.7-97.8) | 90.9 | 0.29% | 6.18% | 20.4% |
| 0.2 | 4,291 | 16.6% | 10.5 22.3 19.8 13.2 | 62 | 99.8 | 98.05 (97.5-98.4) | 92.6 | 0.14% | 4.14% | 21.6% |
| gate60 | 4,644 | 17.5% | 15.1 19.5 18.7 16.8 | 42 | 99.8 | 98.20 (97.9-98.5) | 92.5 | 0.33% | 4.93% | 23.9% |
| gate85 | 4,262 | 18.2% | 10.6 11.1 23.3 23.4 | 75 | 100.0 | 98.22 (97.6-98.5) | 91.9 | 0.17% | 4.45% | 21.6% |
| gate0 | 4,002 | 17.6% | 11.1 17.9 16.3 24.0 | 60 | 99.8 | 97.92 (97.3-98.3) | 92.1 | 0.23% | 4.24% | 20.4% |

<!-- reading -->
Read against the bold reference, b7's λ 0.98 cell `b7aa`-`b7ad` (17.3% density, best30 97.75, 6.2% of evals below 80%), which b15-b21 were generated from. **The chase-safe shaping is a no-op for PPO at this base — neither the dose nor the gate moves anything.** Density is inside the reference's noise at every cell (16.3-18.2% against 17.3), best30 spans 97.80-98.22 with the base at the bottom of the range, and every cell is a little *more* stable than the base (3.6-4.9% of evals below 80% against 6.2). **Two predictions are falsified.** Shaping off (0.0) was to have a late onset and possibly never arrive: it matches the base on density (17.1%) and stage-A density (21.7 against 20.4), and is the most stable cell of the seven (3.56% below 80%, 0.07% below 50%). gate85, snek3's own default gate with only the last 10 squares shaped, was to read worse than 75: it is the densest cell (18.2%), has the batch's only 100.0 row, and its best checkpoint `b21s-gate85-seed3` @38748160 reads 99.1 /5,000 and **98.9 /30,000 [98.8, 99.0]**, the batch's best at depth and below the HOF's 99.30 third place; the other 30k row, `b21b-shape0-seed2` @35618816, reads 98.7. gate0 (shaped from the first move) and gate60 read 17.6 and 17.5%, so the gate's position from 0 to 85 does nothing either; the direction snek2 improved in does not carry to PPO. What the stage-A traces add: seed 1 is the weak seed in five of the six cells (10.5-12.9% share against the reference's 18.5), a seed effect across the batch rather than a knob effect anywhere in it. What the batch does not settle is whether the shaping mattered at a weaker base — the 1%-perfect gate arm at 508k was DQN-era — but at b7's base it is a knob PPO does not need. **Shaping off is the simplest config at no cost** and is the setting to carry into the corner grid; nothing here is a lever.
<!-- /reading -->

### Every arm

| arm | chase_safe_shaping | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b21a-shape0-seed1` | 0.0 | 927 | 11.9% | 5 | 99.6 | 97.8 @42.5M | 91.1 | 0.2% |
| `b21b-shape0-seed2` | 0.0 | 1132 | 18.7% | 23 | 99.6 | 98.3 @35.3M | 93.0 | 0.07% |
| `b21c-shape0-seed3` | 0.0 | 1148 | 18.3% | 20 | 99.4 | 98.2 @49.8M | 93.4 | 0.07% |
| `b21d-shape0-seed4` | 0.0 | 1032 | 18.5% | 10 | 99.6 | 98.0 @47.3M | 93.4 | 0.07% |
| `b21e-shape005-seed1` | 0.05 | 876 | 12.9% | 12 | 99.4 | 97.7 @10.7M | 91.5 | 0.03% |
| `b21f-shape005-seed2` | 0.05 | 945 | 17.0% | 17 | 99.6 | 97.8 @31.5M | 89.2 | 0.71% |
| `b21g-shape005-seed3` | 0.05 | 1162 | 19.6% | 11 | 99.2 | 97.8 @45.8M | 92.4 | 0.03% |
| `b21h-shape005-seed4` | 0.05 | 936 | 14.7% | 11 | 99.4 | 97.9 @47.0M | 92.4 | 0.41% |
| `b21i-shape02-seed1` | 0.2 | 1006 | 10.5% | 1 | 99.0 | 97.5 @39.4M | 94.3 | 0.0% |
| `b21j-shape02-seed2` | 0.2 | 1130 | 22.3% | 29 | 99.8 | 98.0 @29.4M | 92.8 | 0.07% |
| `b21k-shape02-seed3` | 0.2 | 1094 | 19.8% | 16 | 99.4 | 98.4 @45.1M | 91.3 | 0.61% |
| `b21l-shape02-seed4` | 0.2 | 1061 | 13.2% | 16 | 99.6 | 98.3 @7.7M | 92.1 | 0.2% |
| `b21m-gate60-seed1` | gate60 | 1099 | 15.1% | 9 | 99.6 | 98.3 @40.8M | 93.7 | 0.57% |
| `b21n-gate60-seed2` | gate60 | 1080 | 19.5% | 13 | 99.8 | 98.1 @26.6M | 91.5 | 0.17% |
| `b21o-gate60-seed3` | gate60 | 1131 | 18.7% | 9 | 99.4 | 97.9 @35.2M | 92.0 | 0.2% |
| `b21p-gate60-seed4` | gate60 | 1334 | 16.8% | 11 | 99.4 | 98.5 @20.2M | 92.9 | 0.45% |
| `b21q-gate85-seed1` | gate85 | 834 | 10.6% | 6 | 99.2 | 97.6 @19.3M | 93.1 | 0.03% |
| `b21r-gate85-seed2` | gate85 | 912 | 11.1% | 4 | 99.4 | 98.3 @36.3M | 90.2 | 0.17% |
| `b21s-gate85-seed3` | gate85 | 1260 | 23.3% | 41 | 100.0 | 98.5 @34.6M | 92.6 | 0.17% |
| `b21t-gate85-seed4` | gate85 | 1256 | 23.4% | 24 | 99.6 | 98.5 @44.4M | 91.7 | 0.21% |
| `b21u-gate0-seed1` | gate0 | 889 | 11.1% | 8 | 99.6 | 97.3 @48.1M | 93.8 | 0.37% |
| `b21v-gate0-seed2` | gate0 | 1016 | 17.9% | 8 | 99.4 | 98.3 @33.0M | 88.8 | 0.44% |
| `b21w-gate0-seed3` | gate0 | 1041 | 16.3% | 13 | 99.6 | 97.9 @23.9M | 93.1 | 0.1% |
| `b21x-gate0-seed4` | gate0 | 1056 | 24.0% | 31 | 99.8 | 98.2 @31.1M | 92.8 | 0.1% |

<!-- /progress_update: batch b21 -->

<!-- progress_update: batch b20 -->
## Batch b20 — the `collect_envs` sweep, 4 values x 4 seeds, 50M, closed 2026-09-06

Closed on both boxes' feeds; every arm has its stage-B measurement. One knob off the reference cell (`b7aa-fc320-seed1, b7ab-fc320-seed2, b7ac-fc320-seed3, b7ad-fc320-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| collect_envs | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| 32 | 15,485 | 19.1% | 12.7 15.2 17.9 27.1 | 331 | 99.8 | 98.35 (98.1-98.8) | 91.1 | 0.04% | 6.06% | 20.2% |
| 64 | 8,408 | 20.5% | 15.8 16.0 24.9 22.9 | 166 | 99.8 | 98.15 (97.7-98.6) | 92.2 | 0.27% | 5.97% | 22.2% |
| **128** (reference) | 4,003 | 17.3% | 18.5 19.0 16.5 15.1 | 50 | 99.6 | 97.75 (97.7-97.8) | 90.9 | 0.29% | 6.18% | 20.4% |
| 256 | 2,329 | 20.7% | 21.7 23.5 22.6 14.8 | 58 | 99.8 | 97.95 (97.6-98.4) | 91.5 | 0.04% | 3.88% | 24.5% |
| 512 | 1,230 | 17.8% | 12.8 20.6 24.3 13.3 | 22 | 99.4 | 97.75 (97.4-98.0) | 87.5 | 0.0% | 2.19% | 25.4% |

<!-- reading -->
Read against the bold reference, b7's λ 0.98 cell `b7aa`-`b7ad` (17.3% density, best30 97.75, 6.2% of evals below 80%), which b15-b21 were generated from. **`collect_envs` is a throughput knob, not a learning knob, from 32 to 512 lanes at a fixed rollout.** Density is within noise of the base at every value (19.1, 20.5, 17.3, 20.7, 17.8%), best30 spans 97.75-98.35 with the base at the bottom of that range, and no cell's best row beats 99.8. The row counts scale with the eval cadence, not with the policy: 32 lanes logs 15,485 stage-B rows to 512's 1,230 because it makes four times the updates per step, and its 331 `hof5000` candidates are the same ~2% of rows the base has. **The spec's prediction for 32 — worse than the equivalent rollout because episode diversity is lower — is falsified**: 32 lanes matches the base on density and beats it on best30 (98.35). **What does move is stability**: 512 lanes has 0.0% of evals below 50% and 2.19% below 80% against the base's 6.18%, 256 lanes 3.88%, the smoothest endgame of any cell trained at this base, and stage-A density rises with lanes (24.5, 25.4% at 256, 512 against 20.4). The comparison the spec asked for — 256 lanes against b14's rollout 256, the same batch size two ways — is confounded by λ (b14 ran at 0.99), but the direction is not close: b14's rollout 512 read +11 pp over *its* base and 512 lanes reads +0.5 pp over this one, so **the rollout's gain came from depth, not from batch size or episode diversity**. At 30,000 episodes the batch's best is `b20g-lanes64-seed3` @13729792 at 99.3, nothing for the HOF. 128 stays the default for speed; 256-512 lanes is a stability lever at no density cost if the corner grid wants one.
<!-- /reading -->

### Every arm

| arm | collect_envs | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b20a-lanes32-seed1` | 32 | 3231 | 12.7% | 31 | 99.4 | 98.1 @21.2M | 88.9 | 0.06% |
| `b20b-lanes32-seed2` | 32 | 3450 | 15.2% | 51 | 99.8 | 98.3 @49.3M | 89.9 | 0.17% |
| `b20c-lanes32-seed3` | 32 | 3866 | 17.9% | 52 | 99.4 | 98.2 @11.4M | 93.4 | 0.02% |
| `b20d-lanes32-seed4` | 32 | 4938 | 27.1% | 197 | 99.8 | 98.8 @17.9M | 92.3 | 0.0% |
| `b20e-lanes64-seed1` | 64 | 1765 | 15.8% | 22 | 99.8 | 97.7 @36.5M | 91.1 | 0.34% |
| `b20f-lanes64-seed2` | 64 | 1824 | 16.0% | 26 | 99.8 | 98.0 @24.5M | 91.4 | 0.2% |
| `b20g-lanes64-seed3` | 64 | 2597 | 24.9% | 80 | 99.8 | 98.6 @34.0M | 94.8 | 0.0% |
| `b20h-lanes64-seed4` | 64 | 2222 | 22.9% | 38 | 99.6 | 98.3 @42.4M | 91.5 | 0.72% |
| `b20i-lanes256-seed1` | 256 | 581 | 21.7% | 13 | 99.8 | 98.4 @42.9M | 92.1 | 1.16% |
| `b20j-lanes256-seed2` | 256 | 588 | 23.5% | 21 | 99.4 | 98.0 @26.3M | 89.3 | 0.0% |
| `b20k-lanes256-seed3` | 256 | 605 | 22.6% | 16 | 99.6 | 97.8 @23.6M | 91.4 | 0.07% |
| `b20l-lanes256-seed4` | 256 | 555 | 14.8% | 8 | 99.6 | 97.6 @13.3M | 93.4 | 0.0% |
| `b20m-lanes512-seed1` | 512 | 274 | 12.8% | 4 | 99.2 | 97.4 @29.4M | 84.3 | 0.0% |
| `b20n-lanes512-seed2` | 512 | 311 | 20.6% | 7 | 99.4 | 97.8 @45.8M | 84.8 | 0.29% |
| `b20o-lanes512-seed3` | 512 | 313 | 24.3% | 9 | 99.2 | 98.0 @44.8M | 92.0 | 0.0% |
| `b20p-lanes512-seed4` | 512 | 332 | 13.3% | 2 | 99.0 | 97.8 @49.7M | 88.7 | 0.0% |

<!-- /progress_update: batch b20 -->

<!-- progress_update: batch b17 -->
## Batch b17 — the `clip` sweep, 16 values x 4 seeds, 50M, closed 2026-09-06

Closed on both boxes' feeds; every arm has its stage-B measurement. One knob off the reference cell (`b7aa-fc320-seed1, b7ab-fc320-seed2, b7ac-fc320-seed3, b7ad-fc320-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| clip | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| 0.05 | 3,837 | 17.4% | 10.5 22.2 12.0 22.7 | 76 | 100.0 | 98.30 (97.9-98.7) | 85.7 | 0.09% | 8.17% | 20.4% |
| 0.1 | 4,061 | 13.8% | 14.7 15.1 13.6 12.1 | 45 | 99.6 | 98.00 (97.8-98.2) | 89.6 | 0.17% | 6.56% | 20.2% |
| 0.15 | 4,078 | 18.6% | 11.0 13.8 28.4 17.8 | 74 | 99.8 | 98.00 (97.6-98.4) | 90.5 | 0.11% | 5.02% | 21.3% |
| **0.2** (reference) | 4,003 | 17.3% | 18.5 19.0 16.5 15.1 | 50 | 99.6 | 97.75 (97.7-97.8) | 90.9 | 0.29% | 6.18% | 20.4% |
| 0.3 | 3,991 | 14.9% | 12.0 9.5 17.6 18.7 | 57 | 99.8 | 97.58 (97.0-98.0) | 93.5 | 0.31% | 4.01% | 20.4% |
| 0.4 | 3,642 | 12.3% | 8.3 11.8 18.6 8.5 | 27 | 99.6 | 97.33 (96.8-97.8) | 92.8 | 0.3% | 4.46% | 17.9% |
| clipanneal | 4,615 | 16.7% | 15.8 15.3 22.0 13.3 | 65 | 99.8 | 98.25 (98.2-98.4) | 92.7 | 0.29% | 4.29% | 24.2% |
| clip01anneal | 4,235 | 20.0% | 14.9 31.0 16.2 18.3 | 86 | 99.8 | 98.20 (98.0-98.3) | 86.9 | 0.16% | 7.54% | 22.7% |
| lranneal | 5,273 | 23.7% | 13.1 34.6 22.4 22.2 | 98 | 99.6 | 98.33 (98.2-98.6) | 91.9 | 0.26% | 4.42% | 28.8% |
| lranneal10 | 5,178 | 18.9% | 13.2 18.9 24.9 17.4 | 70 | 99.6 | 98.17 (97.9-98.6) | 92.4 | 0.05% | 4.22% | 27.6% |
| bothanneal | 5,138 | 18.2% | 20.8 20.9 10.1 20.2 | 67 | 99.8 | 98.15 (98.1-98.2) | 92.8 | 0.12% | 3.25% | 27.5% |
| clipanneal2to1 | 4,258 | 18.1% | 12.7 14.0 15.5 28.3 | 73 | 99.6 | 98.05 (97.8-98.5) | 91.3 | 0.11% | 4.27% | 22.4% |
| clip04anneal | 4,515 | 16.3% | 12.5 13.4 17.4 20.8 | 51 | 100.0 | 98.33 (98.0-98.6) | 93.2 | 0.22% | 3.86% | 23.4% |
| clipanneal001 | 4,398 | 17.4% | 17.1 19.0 11.9 20.1 | 74 | 99.8 | 98.22 (97.7-98.6) | 92.0 | 0.24% | 4.98% | 22.9% |
| clipanneal005 | 4,052 | 18.3% | 15.8 20.4 23.5 13.2 | 66 | 99.6 | 98.33 (97.9-98.5) | 89.8 | 0.09% | 7.3% | 20.9% |
| clipannealhold80 | 4,883 | 23.5% | 13.8 23.6 30.7 25.4 | 115 | 99.8 | 98.48 (97.9-98.9) | 92.5 | 0.24% | 3.89% | 26.8% |
| clipanneal001hold80 | 4,436 | 24.3% | 24.4 18.5 16.7 32.5 | 122 | 99.8 | 98.45 (97.8-98.8) | 91.8 | 0.15% | 4.82% | 23.9% |

<!-- reading -->
Read against the bold reference, b7's λ 0.98 cell `b7aa`-`b7ad` (17.3% density, best30 97.75, 6.2% of evals below 80%), which b15-b21 were generated from. **The clip is flat from 0.05 to 0.2 and worse above it** — 17.4, 13.8, 18.6, 17.3% density, then 14.9 at 0.3 and 12.3 at 0.4 — and best30 runs the other way, highest at the tight end (98.30 at 0.05, one of the batch's two 100/500 rows) and lowest at 0.4 (97.33). Loosening the clip *reduces* collapses (8.2% of evals below 80% at 0.05 down to 4.0-4.5% at 0.3-0.4), the reverse of the spec's "looser: more collapses"; the tight end trades stability for a higher top. **Every anneal sits above the base on best30 (98.05-98.48 against 97.75) and all but two below it on collapses**, so annealing the trust region is worth having, but the plain anneals only match the base on density (16.7-20.0%). **What lifts density is holding at the floor**: `clipannealhold80` and `clipanneal001hold80`, which reach the floor at 40M and train the last 10M there, read 23.5 and 24.3% at best30 98.48 and 98.45 — +6-7 pp on the base, the batch's two highest `hof5000` candidate counts (115, 122), at 3.9-4.8% below 80%. The floor's value barely matters (0.02, 0.005 and 0.001 without a hold: 16.7, 18.3, 17.4%); the hold does. `lranneal` does the same thing through the parameters (23.7%, best30 98.33, 4.4%), and annealing the lr to zero beats annealing to a tenth (18.9%): the near-frozen last stretch is what the endgame wants. `bothanneal` is the most stable cell in the batch (3.25% below 80%) at base density. Per-seed spread is wide in the winning cells (13.8-30.7, 16.7-32.5), so the +6 pp is an n=4 reading. **At 30,000 episodes: `b17cl-clipanneal001hold80-seed4` @11386880 read 99.5 [99.4, 99.6]**, its neighbours 99.4 and 99.3 — **promoted to the HOF in third place on 2026-09-06**, above `b9ch`'s 99.30 and not distinguishable from the lower `b10ck` entry, and from a checkpoint at 11.4M steps, a fifth of the run. A hold-at-floor anneal enters the corner grid; which of the two floors is the user's pick, since they do not separate.
<!-- /reading -->

### Every arm

| arm | clip | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b17aa-clip005-seed1` | 0.05 | 811 | 10.5% | 4 | 99.4 | 98.4 @41.2M | 85.3 | 0.11% |
| `b17ab-clip005-seed2` | 0.05 | 1016 | 22.2% | 32 | 100.0 | 98.7 @28.9M | 87.4 | 0.17% |
| `b17ac-clip005-seed3` | 0.05 | 930 | 12.0% | 9 | 99.4 | 97.9 @16.4M | 80.8 | 0.07% |
| `b17ad-clip005-seed4` | 0.05 | 1080 | 22.7% | 31 | 99.4 | 98.2 @30.3M | 89.1 | 0.0% |
| `b17ae-clip01-seed1` | 0.1 | 932 | 14.7% | 15 | 99.4 | 97.8 @39.1M | 90.8 | 0.0% |
| `b17af-clip01-seed2` | 0.1 | 1080 | 15.1% | 10 | 99.2 | 98.2 @28.3M | 88.8 | 0.2% |
| `b17ag-clip01-seed3` | 0.1 | 959 | 13.6% | 13 | 99.2 | 97.8 @40.5M | 88.1 | 0.17% |
| `b17ah-clip01-seed4` | 0.1 | 1090 | 12.1% | 7 | 99.6 | 98.2 @47.5M | 90.6 | 0.17% |
| `b17ai-clip015-seed1` | 0.15 | 865 | 11.0% | 4 | 99.2 | 97.6 @29.3M | 90.1 | 0.2% |
| `b17aj-clip015-seed2` | 0.15 | 993 | 13.8% | 12 | 99.4 | 97.9 @42.9M | 91.3 | 0.0% |
| `b17ak-clip015-seed3` | 0.15 | 1239 | 28.4% | 47 | 99.8 | 98.4 @43.5M | 90.4 | 0.14% |
| `b17al-clip015-seed4` | 0.15 | 981 | 17.8% | 11 | 99.2 | 98.1 @15.6M | 90.1 | 0.07% |
| `b17am-clip03-seed1` | 0.3 | 951 | 12.0% | 11 | 99.6 | 97.0 @10.6M | 92.4 | 0.27% |
| `b17an-clip03-seed2` | 0.3 | 846 | 9.5% | 2 | 99.2 | 97.3 @23.8M | 93.3 | 0.34% |
| `b17ao-clip03-seed3` | 0.3 | 1063 | 17.6% | 22 | 99.8 | 98.0 @12.3M | 94.6 | 0.0% |
| `b17ap-clip03-seed4` | 0.3 | 1131 | 18.7% | 22 | 99.4 | 98.0 @32.2M | 93.5 | 0.37% |
| `b17aq-clip04-seed1` | 0.4 | 744 | 8.3% | 2 | 99.0 | 96.8 @16.4M | 92.4 | 0.4% |
| `b17ar-clip04-seed2` | 0.4 | 932 | 11.8% | 6 | 99.4 | 97.5 @8.5M | 92.7 | 0.67% |
| `b17as-clip04-seed3` | 0.4 | 1082 | 18.6% | 17 | 99.6 | 97.8 @10.5M | 92.5 | 0.1% |
| `b17at-clip04-seed4` | 0.4 | 884 | 8.5% | 2 | 99.2 | 97.2 @40.6M | 93.6 | 0.2% |
| `b17au-clipanneal-seed1` | clipanneal | 1011 | 15.8% | 9 | 99.8 | 98.2 @34.8M | 90.1 | 0.24% |
| `b17av-clipanneal-seed2` | clipanneal | 1108 | 15.3% | 19 | 99.6 | 98.2 @46.4M | 93.4 | 0.4% |
| `b17aw-clipanneal-seed3` | clipanneal | 1258 | 22.0% | 25 | 99.4 | 98.4 @28.2M | 93.2 | 0.23% |
| `b17ax-clipanneal-seed4` | clipanneal | 1238 | 13.3% | 12 | 99.4 | 98.2 @43.8M | 93.9 | 0.34% |
| `b17ay-clip01anneal-seed1` | clip01anneal | 1125 | 14.9% | 15 | 99.2 | 98.0 @31.5M | 88.7 | 0.46% |
| `b17az-clip01anneal-seed2` | clip01anneal | 1036 | 31.0% | 43 | 99.8 | 98.3 @36.7M | 85.6 | 0.28% |
| `b17ba-clip01anneal-seed3` | clip01anneal | 1102 | 16.2% | 17 | 99.4 | 98.2 @35.8M | 88.9 | 0.03% |
| `b17bb-clip01anneal-seed4` | clip01anneal | 972 | 18.3% | 11 | 99.4 | 98.3 @32.8M | 84.3 | 0.03% |
| `b17bc-lranneal-seed1` | lranneal | 1214 | 13.1% | 4 | 99.2 | 98.2 @49.9M | 92.2 | 0.34% |
| `b17bd-lranneal-seed2` | lranneal | 1521 | 34.6% | 56 | 99.6 | 98.6 @15.7M | 93.1 | 0.17% |
| `b17be-lranneal-seed3` | lranneal | 1354 | 22.4% | 22 | 99.4 | 98.2 @25.8M | 89.7 | 0.44% |
| `b17bf-lranneal-seed4` | lranneal | 1184 | 22.2% | 16 | 99.6 | 98.3 @47.1M | 92.6 | 0.03% |
| `b17bg-lranneal10-seed1` | lranneal10 | 1159 | 13.2% | 6 | 99.2 | 97.9 @14.3M | 93.0 | 0.14% |
| `b17bh-lranneal10-seed2` | lranneal10 | 1262 | 18.9% | 23 | 99.4 | 98.1 @45.3M | 92.0 | 0.07% |
| `b17bi-lranneal10-seed3` | lranneal10 | 1411 | 24.9% | 26 | 99.6 | 98.6 @47.8M | 92.0 | 0.03% |
| `b17bj-lranneal10-seed4` | lranneal10 | 1346 | 17.4% | 15 | 99.6 | 98.1 @48.3M | 92.6 | 0.0% |
| `b17bk-bothanneal-seed1` | bothanneal | 1457 | 20.8% | 19 | 99.6 | 98.2 @37.8M | 94.6 | 0.1% |
| `b17bl-bothanneal-seed2` | bothanneal | 1330 | 20.9% | 29 | 99.8 | 98.1 @24.4M | 93.3 | 0.03% |
| `b17bm-bothanneal-seed3` | bothanneal | 1181 | 10.1% | 4 | 99.2 | 98.1 @39.4M | 90.5 | 0.14% |
| `b17bn-bothanneal-seed4` | bothanneal | 1170 | 20.2% | 15 | 99.4 | 98.2 @41.6M | 92.9 | 0.14% |
| `b17bo-clipanneal2to1-seed1` | clipanneal2to1 | 1083 | 12.7% | 8 | 99.6 | 97.9 @38.6M | 91.1 | 0.07% |
| `b17bp-clipanneal2to1-seed2` | clipanneal2to1 | 935 | 14.0% | 9 | 99.4 | 98.0 @30.1M | 89.0 | 0.14% |
| `b17bq-clipanneal2to1-seed3` | clipanneal2to1 | 1027 | 15.5% | 14 | 99.2 | 97.8 @40.0M | 92.6 | 0.3% |
| `b17br-clipanneal2to1-seed4` | clipanneal2to1 | 1213 | 28.3% | 42 | 99.6 | 98.5 @43.7M | 92.7 | 0.03% |
| `b17bs-clip04anneal-seed1` | clip04anneal | 1070 | 12.5% | 7 | 99.2 | 98.0 @46.9M | 93.5 | 0.44% |
| `b17bt-clip04anneal-seed2` | clip04anneal | 988 | 13.4% | 7 | 99.4 | 98.4 @47.5M | 93.5 | 0.03% |
| `b17bu-clip04anneal-seed3` | clip04anneal | 1273 | 17.4% | 16 | 99.4 | 98.3 @47.3M | 94.5 | 0.1% |
| `b17bv-clip04anneal-seed4` | clip04anneal | 1184 | 20.8% | 21 | 100.0 | 98.6 @48.4M | 91.3 | 0.34% |
| `b17bw-clipanneal001-seed1` | clipanneal001 | 957 | 17.1% | 15 | 99.4 | 98.2 @38.3M | 92.7 | 0.27% |
| `b17bx-clipanneal001-seed2` | clipanneal001 | 1298 | 19.0% | 24 | 99.6 | 98.4 @15.0M | 92.6 | 0.14% |
| `b17by-clipanneal001-seed3` | clipanneal001 | 950 | 11.9% | 9 | 99.4 | 97.7 @18.5M | 91.3 | 0.2% |
| `b17bz-clipanneal001-seed4` | clipanneal001 | 1193 | 20.1% | 26 | 99.8 | 98.6 @37.5M | 91.3 | 0.54% |
| `b17ca-clipanneal005-seed1` | clipanneal005 | 1001 | 15.8% | 13 | 99.6 | 97.9 @37.3M | 89.4 | 0.5% |
| `b17cb-clipanneal005-seed2` | clipanneal005 | 918 | 20.4% | 23 | 99.4 | 98.5 @29.4M | 89.8 | 0.03% |
| `b17cc-clipanneal005-seed3` | clipanneal005 | 1102 | 23.5% | 22 | 99.6 | 98.5 @39.2M | 89.4 | 0.14% |
| `b17cd-clipanneal005-seed4` | clipanneal005 | 1031 | 13.2% | 8 | 99.4 | 98.4 @29.9M | 90.8 | 0.0% |
| `b17ce-clipannealhold80-seed1` | clipannealhold80 | 1172 | 13.8% | 6 | 99.4 | 97.9 @42.4M | 93.2 | 0.2% |
| `b17cf-clipannealhold80-seed2` | clipannealhold80 | 1231 | 23.6% | 22 | 99.8 | 98.4 @41.8M | 90.6 | 0.27% |
| `b17cg-clipannealhold80-seed3` | clipannealhold80 | 1243 | 30.7% | 49 | 99.8 | 98.9 @45.8M | 95.0 | 0.07% |
| `b17ch-clipannealhold80-seed4` | clipannealhold80 | 1237 | 25.4% | 38 | 99.8 | 98.7 @26.3M | 91.3 | 0.55% |
| `b17ci-clipanneal001hold80-seed1` | clipanneal001hold80 | 1401 | 24.4% | 33 | 99.6 | 98.7 @41.9M | 91.5 | 0.07% |
| `b17cj-clipanneal001hold80-seed2` | clipanneal001hold80 | 892 | 18.5% | 10 | 99.2 | 97.8 @44.1M | 90.2 | 0.27% |
| `b17ck-clipanneal001hold80-seed3` | clipanneal001hold80 | 801 | 16.7% | 6 | 99.4 | 98.5 @50.0M | 92.2 | 0.24% |
| `b17cl-clipanneal001hold80-seed4` | clipanneal001hold80 | 1342 | 32.5% | 73 | 99.8 | 98.8 @9.9M | 93.3 | 0.0% |

<!-- /progress_update: batch b17 -->

<!-- progress_update: batch b19 -->
## Batch b19 — the switches batch (advantage normalisation off, mse value loss, Adam ε 1e-5 / 1e-8, vf coef 0.1 / 1.0), 6 cells x 4 seeds, 50M, closed 2026-09-05

Closed on the laptop under the old per-box queue, before the shared queue; every arm has its stage-B, hof5000 and hof30k files in `runs/`. One knob off the reference cell (`b7aa-fc320-seed1, b7ab-fc320-seed2, b7ac-fc320-seed3, b7ad-fc320-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| knob | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| noadvnorm | 3,826 | 12.9% | 18.4 13.0 11.7 5.2 | 34 | 99.4 | 97.85 (97.2-98.4) | 93.1 | 0.0% | 1.18% | 19.0% |
| mse | 6,188 | 22.2% | 27.6 23.5 21.2 14.6 | 109 | 99.8 | 98.35 (98.2-98.6) | 96.0 | 0.26% | 0.97% | 33.9% |
| adameps1e5 | 4,439 | 20.5% | 20.2 17.7 25.8 18.5 | 91 | 99.6 | 98.20 (98.1-98.3) | 92.2 | 0.11% | 4.24% | 23.3% |
| adameps1e8 | 4,050 | 16.1% | 16.2 7.9 21.0 17.9 | 59 | 99.8 | 97.97 (97.4-98.6) | 92.5 | 0.1% | 4.39% | 20.5% |
| vf01 | 4,486 | 20.5% | 19.3 20.5 16.1 26.0 | 81 | 99.8 | 98.17 (97.9-98.3) | 91.8 | 0.26% | 4.76% | 23.4% |
| vf10 | 4,258 | 19.5% | 18.1 25.3 13.2 20.5 | 82 | 99.8 | 97.88 (97.7-98.2) | 91.8 | 0.14% | 4.03% | 21.7% |
| **base** (reference) | 4,003 | 17.3% | 18.5 19.0 16.5 15.1 | 50 | 99.6 | 97.75 (97.7-97.8) | 90.9 | 0.29% | 6.18% | 20.4% |

<!-- reading -->
Read against the bold reference, b7's λ 0.98 cell `b7aa`-`b7ad` (17.3% density, best30 97.75, 6.2% of evals below 80%), which b15-b21 were generated from. A switches batch, so each cell is named rather than valued. **`mse` is the row to look at**: 22.2% density against 17.3% (three seeds at 21-28%, one at 14.6%), best30 98.35 against 97.75, stage-A density 33.9% against 20.4, and **0.97% of evals below 80% against 6.2%** at sef 96 — the most stable cell trained at this base and the densest in the batch, where the spec predicted a noisier critic and more collapses. `noadvnorm` is as stable (1.18% below 80%, 0.0% below 50%) and 4 pp short on density, with one seed at 5.2%: turning advantage normalisation off removes the collapses and costs some of the top. Both switches were predicted to add collapses; both removed them, which says the base's collapses come through the critic and the advantage scale, not the policy step. **The Adam ε and vf-coefficient cells are within noise** (16.1-20.5% density, best30 97.88-98.20, 4.0-4.8% below 80%): the value loss's weight does not matter between 0.1 and 1.0, but its form does. At 30,000 episodes the batch's best rows are `b19u-vf10-seed1` @8175616 and `b19o-adameps1e8-seed3` @30261248 at 99.2, nothing for the HOF. `mse` enters the corner grid.
<!-- /reading -->

### Every arm

| arm | knob | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b19a-noadvnorm-seed1` | noadvnorm | 1121 | 18.4% | 20 | 99.4 | 97.9 @25.7M | 94.1 | 0.0% |
| `b19b-noadvnorm-seed2` | noadvnorm | 1059 | 13.0% | 9 | 99.4 | 98.4 @34.4M | 92.5 | 0.0% |
| `b19c-noadvnorm-seed3` | noadvnorm | 997 | 11.7% | 3 | 99.4 | 97.9 @46.4M | 93.1 | 0.0% |
| `b19d-noadvnorm-seed4` | noadvnorm | 649 | 5.2% | 2 | 99.4 | 97.2 @23.2M | 92.7 | 0.0% |
| `b19e-mse-seed1` | mse | 1733 | 27.6% | 36 | 99.6 | 98.2 @13.5M | 95.3 | 0.27% |
| `b19f-mse-seed2` | mse | 1564 | 23.5% | 42 | 99.8 | 98.6 @39.9M | 96.7 | 0.07% |
| `b19g-mse-seed3` | mse | 1568 | 21.2% | 23 | 99.6 | 98.4 @26.3M | 96.8 | 0.24% |
| `b19h-mse-seed4` | mse | 1323 | 14.6% | 8 | 99.4 | 98.2 @34.5M | 95.2 | 0.77% |
| `b19i-adameps1e5-seed1` | adameps1e5 | 1209 | 20.2% | 23 | 99.4 | 98.3 @30.7M | 91.0 | 0.57% |
| `b19j-adameps1e5-seed2` | adameps1e5 | 1038 | 17.7% | 22 | 99.4 | 98.1 @14.3M | 91.3 | 0.07% |
| `b19k-adameps1e5-seed3` | adameps1e5 | 1063 | 25.8% | 25 | 99.6 | 98.1 @47.2M | 92.3 | 0.03% |
| `b19l-adameps1e5-seed4` | adameps1e5 | 1129 | 18.5% | 21 | 99.4 | 98.3 @48.9M | 94.3 | 0.14% |
| `b19m-adameps1e8-seed1` | adameps1e8 | 1049 | 16.2% | 11 | 99.8 | 97.9 @10.3M | 93.2 | 0.1% |
| `b19n-adameps1e8-seed2` | adameps1e8 | 906 | 7.9% | 4 | 99.2 | 97.4 @45.4M | 93.7 | 0.1% |
| `b19o-adameps1e8-seed3` | adameps1e8 | 1085 | 21.0% | 27 | 99.6 | 98.6 @15.0M | 91.5 | 0.24% |
| `b19p-adameps1e8-seed4` | adameps1e8 | 1010 | 17.9% | 17 | 99.4 | 98.0 @24.4M | 91.7 | 0.0% |
| `b19q-vf01-seed1` | vf01 | 1134 | 19.3% | 11 | 99.6 | 97.9 @31.0M | 91.8 | 0.28% |
| `b19r-vf01-seed2` | vf01 | 1074 | 20.5% | 21 | 99.8 | 98.3 @29.6M | 90.7 | 0.24% |
| `b19s-vf01-seed3` | vf01 | 1125 | 16.1% | 17 | 99.6 | 98.3 @14.1M | 93.7 | 0.14% |
| `b19t-vf01-seed4` | vf01 | 1153 | 26.0% | 32 | 99.6 | 98.2 @26.2M | 91.1 | 0.81% |
| `b19u-vf10-seed1` | vf10 | 965 | 18.1% | 22 | 99.8 | 97.7 @30.4M | 91.3 | 0.0% |
| `b19v-vf10-seed2` | vf10 | 1153 | 25.3% | 32 | 99.6 | 98.2 @26.3M | 91.1 | 0.24% |
| `b19w-vf10-seed3` | vf10 | 1025 | 13.2% | 8 | 99.2 | 97.8 @41.8M | 92.1 | 0.14% |
| `b19x-vf10-seed4` | vf10 | 1115 | 20.5% | 20 | 99.6 | 97.8 @37.8M | 92.8 | 0.14% |

<!-- /progress_update: batch b19 -->

<!-- progress_update: batch b16 -->
## Batch b16 — the `target_kl` sweep, 10 values x 4 seeds, 50M, closed 2026-09-05

Closed on the desktop; every arm has its stage-B measurement. One knob off the reference cell (`b7aa-fc320-seed1, b7ab-fc320-seed2, b7ac-fc320-seed3, b7ad-fc320-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| target_kl | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| **0** (reference) | 4,003 | 17.3% | 18.5 19.0 16.5 15.1 | 50 | 99.6 | 97.75 (97.7-97.8) | 90.9 | 0.29% | 6.18% | 20.4% |
| 0.003 | 3,923 | 16.1% | 14.7 15.2 18.7 15.7 | 46 | 99.6 | 97.90 (97.6-98.2) | 90.0 | 0.68% | 6.86% | 19.9% |
| 0.005 | 4,403 | 20.3% | 13.9 11.2 28.9 23.0 | 85 | 99.6 | 98.08 (97.7-98.5) | 91.3 | 0.29% | 6.5% | 23.4% |
| 0.008 | 4,017 | 13.6% | 11.4 13.1 17.0 12.1 | 42 | 99.6 | 97.72 (97.3-98.1) | 90.6 | 0.15% | 4.8% | 20.3% |
| 0.01 | 4,209 | 18.2% | 16.8 18.9 14.1 23.1 | 70 | 99.6 | 98.00 (97.9-98.1) | 91.7 | 0.07% | 4.63% | 22.1% |
| 0.013 | 4,008 | 14.9% | 11.7 18.0 14.1 15.6 | 49 | 99.6 | 97.83 (97.5-98.1) | 91.8 | 0.24% | 5.64% | 20.8% |
| 0.015 | 4,317 | 16.8% | 13.2 15.0 20.6 18.4 | 48 | 100.0 | 97.97 (97.6-98.3) | 92.1 | 0.15% | 4.37% | 22.4% |
| 0.02 | 3,889 | 13.9% | 10.8 17.0 12.4 14.6 | 39 | 99.8 | 97.78 (97.6-97.9) | 92.0 | 0.14% | 5.13% | 19.7% |
| 0.03 | 4,617 | 19.0% | 18.3 20.0 19.0 18.7 | 70 | 99.8 | 98.03 (97.9-98.2) | 92.3 | 0.39% | 3.89% | 24.5% |
| 0.04 | 4,170 | 16.9% | 17.8 11.2 15.4 21.5 | 54 | 99.8 | 98.00 (97.9-98.2) | 91.8 | 0.24% | 4.71% | 21.2% |
| 0.05 | 4,039 | 17.2% | 18.4 10.7 17.7 20.3 | 70 | 99.8 | 97.75 (97.1-98.1) | 91.8 | 0.22% | 4.99% | 20.8% |

<!-- reading -->
**Verdict: target KL does nothing at 4 epochs — every value from 0.003 to 0.05 is within noise of the base on every
column, and the base stays off.** Run on the laptop. Read against the bold reference — **b7's λ 0.98 cell `b7aa`-`b7ad`, not b9's λ 0.99 arms**: b15-b21 were generated from b7's base (every arm's config table says `ppo_gae_lambda 0.98`, and the spec notes name `b7aa-b7ad` as the control), and until 2026-09-05 15:30 this table was read against the λ 0.99 cell, which made every cell look 8-14 pp short. The reference row is 17.3% density, best30 97.75, 6.2% of evals below 80%. Density 13.6-20.3% against 17.3% with no trend along the
sweep (16.1, 20.3, 13.6, 18.2, 14.9, 16.8, 13.9, 19.0, 16.9, 17.2); best30 97.7-98.1 against 97.75; the share of evals
below 80% 3.9-6.9% against 6.2%, with the upper half of the sweep at 3.9-5.6% — a hint at n=4, not a result. The
null-check cells 0.03-0.05 read 16.9-19.0%, the reference's 17.3%: the prediction "identical to the control" holds
statistically but not literally, since same-seed arms at 0.03, 0.04 and 0.05 diverge from one another at 2.9-4.5M
steps — the stop fired during onset in every cell, where approx_kl is large, and it did not matter. 0.005, predicted
the stability candidate, is the densest cell (20.3%, one seed at 28.9%) and no more stable than the base. `hof30k`
gave four rows at 99.3 /30,000 (`b16ac` at 0.003 @13.1M; the seed-1 arms of 0.03, 0.04 and 0.05, each @10.7M), none
above the HOF's third place. Nothing to carry into the corner grid.
<!-- /reading -->

### Every arm

| arm | target_kl | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b16aa-kl003-seed1` | 0.003 | 1106 | 14.7% | 8 | 99.4 | 98.0 @31.0M | 91.7 | 0.41% |
| `b16ab-kl003-seed2` | 0.003 | 791 | 15.2% | 7 | 99.2 | 97.6 @41.7M | 89.1 | 0.71% |
| `b16ac-kl003-seed3` | 0.003 | 1031 | 18.7% | 19 | 99.6 | 98.2 @40.9M | 89.1 | 1.29% |
| `b16ad-kl003-seed4` | 0.003 | 995 | 15.7% | 12 | 99.4 | 97.8 @24.6M | 90.3 | 0.65% |
| `b16ae-kl005-seed1` | 0.005 | 1028 | 13.9% | 10 | 99.2 | 97.7 @13.1M | 93.6 | 0.17% |
| `b16af-kl005-seed2` | 0.005 | 859 | 11.2% | 9 | 99.6 | 97.7 @45.4M | 91.2 | 0.17% |
| `b16ag-kl005-seed3` | 0.005 | 1263 | 28.9% | 40 | 99.6 | 98.5 @36.7M | 89.8 | 0.4% |
| `b16ah-kl005-seed4` | 0.005 | 1253 | 23.0% | 26 | 99.6 | 98.4 @37.4M | 90.8 | 0.94% |
| `b16ai-kl008-seed1` | 0.008 | 923 | 11.4% | 5 | 99.2 | 97.3 @28.1M | 90.0 | 0.24% |
| `b16aj-kl008-seed2` | 0.008 | 992 | 13.1% | 11 | 99.6 | 97.4 @24.6M | 91.0 | 0.2% |
| `b16ak-kl008-seed3` | 0.008 | 1135 | 17.0% | 17 | 99.4 | 98.1 @34.6M | 90.2 | 0.1% |
| `b16al-kl008-seed4` | 0.008 | 967 | 12.1% | 9 | 99.4 | 98.1 @13.2M | 91.3 | 0.0% |
| `b16am-kl01-seed1` | 0.01 | 1039 | 16.8% | 12 | 99.4 | 98.1 @30.5M | 92.1 | 0.1% |
| `b16an-kl01-seed2` | 0.01 | 1101 | 18.9% | 16 | 99.6 | 98.0 @18.6M | 92.3 | 0.0% |
| `b16ao-kl01-seed3` | 0.01 | 1039 | 14.1% | 11 | 99.4 | 97.9 @22.7M | 92.9 | 0.03% |
| `b16ap-kl01-seed4` | 0.01 | 1030 | 23.1% | 31 | 99.6 | 98.0 @36.0M | 89.3 | 0.14% |
| `b16aq-kl013-seed1` | 0.013 | 980 | 11.7% | 8 | 99.2 | 97.7 @44.6M | 91.6 | 0.44% |
| `b16ar-kl013-seed2` | 0.013 | 990 | 18.0% | 18 | 99.4 | 98.1 @22.2M | 92.7 | 0.34% |
| `b16as-kl013-seed3` | 0.013 | 981 | 14.1% | 10 | 99.6 | 97.5 @22.9M | 90.8 | 0.14% |
| `b16at-kl013-seed4` | 0.013 | 1057 | 15.6% | 13 | 99.4 | 98.0 @38.6M | 92.0 | 0.1% |
| `b16au-kl015-seed1` | 0.015 | 1104 | 13.2% | 8 | 99.6 | 97.6 @16.1M | 93.9 | 0.34% |
| `b16av-kl015-seed2` | 0.015 | 1070 | 15.0% | 5 | 99.4 | 98.0 @44.9M | 90.0 | 0.1% |
| `b16aw-kl015-seed3` | 0.015 | 1098 | 20.6% | 24 | 99.4 | 98.3 @10.7M | 91.5 | 0.1% |
| `b16ax-kl015-seed4` | 0.015 | 1045 | 18.4% | 11 | 100.0 | 98.0 @17.9M | 93.0 | 0.2% |
| `b16ay-kl02-seed1` | 0.02 | 931 | 10.8% | 6 | 99.8 | 97.9 @49.8M | 91.9 | 0.37% |
| `b16az-kl02-seed2` | 0.02 | 1107 | 17.0% | 12 | 99.4 | 97.8 @32.0M | 94.6 | 0.1% |
| `b16ba-kl02-seed3` | 0.02 | 878 | 12.4% | 7 | 99.4 | 97.6 @49.2M | 90.9 | 0.17% |
| `b16bb-kl02-seed4` | 0.02 | 973 | 14.6% | 14 | 99.6 | 97.8 @17.2M | 90.6 | 0.0% |
| `b16bc-kl03-seed1` | 0.03 | 1194 | 18.3% | 19 | 99.4 | 98.2 @11.1M | 93.6 | 0.57% |
| `b16bd-kl03-seed2` | 0.03 | 1037 | 20.0% | 15 | 99.4 | 98.0 @29.9M | 88.6 | 0.27% |
| `b16be-kl03-seed3` | 0.03 | 1251 | 19.0% | 17 | 99.4 | 98.0 @21.0M | 94.0 | 0.5% |
| `b16bf-kl03-seed4` | 0.03 | 1135 | 18.7% | 19 | 99.8 | 97.9 @38.2M | 92.9 | 0.0% |
| `b16bg-kl04-seed1` | 0.04 | 1170 | 17.8% | 13 | 99.6 | 98.0 @16.6M | 93.1 | 0.61% |
| `b16bh-kl04-seed2` | 0.04 | 863 | 11.2% | 6 | 99.8 | 97.9 @15.5M | 92.6 | 0.4% |
| `b16bi-kl04-seed3` | 0.04 | 981 | 15.4% | 13 | 99.6 | 97.9 @18.8M | 91.2 | 0.07% |
| `b16bj-kl04-seed4` | 0.04 | 1156 | 21.5% | 22 | 99.6 | 98.2 @23.4M | 90.3 | 0.07% |
| `b16bk-kl05-seed1` | 0.05 | 1187 | 18.4% | 29 | 99.6 | 97.9 @17.0M | 93.7 | 0.57% |
| `b16bl-kl05-seed2` | 0.05 | 844 | 10.7% | 8 | 99.8 | 97.1 @15.3M | 92.7 | 0.34% |
| `b16bm-kl05-seed3` | 0.05 | 888 | 17.7% | 14 | 99.6 | 97.9 @18.8M | 91.0 | 0.1% |
| `b16bn-kl05-seed4` | 0.05 | 1120 | 20.3% | 19 | 99.6 | 98.1 @42.1M | 90.0 | 0.0% |

<!-- /progress_update: batch b16 -->

<!-- progress_update: batch b15 -->
## Batch b15 — the `entropy_coef` sweep, 10 values x 4 seeds, 50M, closed 2026-09-05

Closed on the desktop; every arm has its stage-B measurement. One knob off the reference cell (`b7aa-fc320-seed1, b7ab-fc320-seed2, b7ac-fc320-seed3, b7ad-fc320-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| entropy_coef | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| 0.0 | 4,626 | 11.7% | 11.9 7.7 8.5 17.7 | 35 | 99.6 | 97.75 (97.5-98.0) | 94.8 | 0.22% | 1.67% | 22.4% |
| 0.001 | 3,869 | 8.7% | 8.9 7.2 10.8 7.5 | 18 | 99.4 | 97.90 (97.7-98.2) | 94.8 | 0.01% | 1.9% | 18.5% |
| 0.003 | 4,109 | 8.7% | 6.7 9.3 9.9 8.6 | 15 | 99.4 | 97.65 (97.5-97.9) | 95.0 | 0.0% | 1.5% | 19.2% |
| 0.005 | 4,041 | 11.8% | 10.3 15.0 9.7 12.2 | 24 | 99.8 | 97.50 (97.4-97.7) | 94.2 | 0.22% | 2.7% | 19.9% |
| **0.01** (reference) | 4,003 | 17.3% | 18.5 19.0 16.5 15.1 | 50 | 99.6 | 97.75 (97.7-97.8) | 90.9 | 0.29% | 6.18% | 20.4% |
| 0.02 | 3,121 | 13.9% | 11.2 15.5 12.9 16.0 | 42 | 99.6 | 97.58 (97.3-97.8) | 86.9 | 0.79% | 9.46% | 15.6% |
| 0.03 | 2,815 | 24.1% | 18.1 16.5 34.5 19.8 | 122 | 99.6 | 97.88 (97.3-98.3) | 82.5 | 0.74% | 14.22% | 15.2% |
| entanneal10 | 2,034 | 16.5% | 22.6 12.3 15.6 15.4 | 36 | 99.6 | 97.55 (97.2-97.9) | 70.5 | 2.65% | 23.8% | 10.3% |
| entanneal03 | 3,185 | 12.3% | 8.7 11.3 10.2 17.7 | 35 | 99.6 | 97.50 (97.4-97.7) | 85.8 | 0.95% | 12.29% | 15.4% |
| entanneal01 | 3,883 | 13.0% | 10.5 14.5 13.5 13.2 | 39 | 99.6 | 97.78 (97.5-98.2) | 91.4 | 0.18% | 4.39% | 18.9% |
| entanneal01to0 | 4,023 | 10.8% | 12.7 12.3 7.6 10.3 | 27 | 99.6 | 97.55 (97.5-97.6) | 92.6 | 0.23% | 4.21% | 19.5% |

<!-- reading -->
**Verdict: the entropy coefficient trades density for stability monotonically, an annealed coefficient lands between
its two endpoints rather than combining them, and the base stays 0.01.** Read against the bold reference — **b7's λ 0.98 cell `b7aa`-`b7ad`, not b9's λ 0.99 arms**: b15-b21 were generated from b7's base (every arm's config table says `ppo_gae_lambda 0.98`, and the spec notes name `b7aa-b7ad` as the control), and until 2026-09-05 15:30 this table was read against the λ 0.99 cell, which made every cell look 8-14 pp short. The reference row is 17.3% density, best30 97.75, 6.2% of evals below 80%. Density 8.7-11.8% at 0-0.005, 13.9% at
0.02, **24.1% at 0.03** (one seed at 34.5%; 122 candidates from 2,815 rows, the richest top in the batch), 16.5% for
0.1→0.001, 12.3% for 0.03→0.001, 13.0% for 0.01→0.001, 10.8% for 0.01→0. Stability runs the other way without an
exception: 1.5-2.7% of evals below 80% at 0-0.005 (sef 94-95), 6.2% at the reference, 9.5% at 0.02, 14.2% at 0.03,
**23.8% for the anneal from 0.1** (sef 70). The two anneals from 0.01 sit between fixed 0.01 and fixed 0.001 on both
columns: the average of the endpoints, not the top of one with the tail of the other. Predictions: **0** "may go
deterministic early and stall" — no; best row 99.6, onset on time. **0.001** "fewest collapses; the default moves if
there is no density loss" — fewest collapses, half the density; the default stays. **0.003** fewer collapses — confirmed.
**0.02** "locates the cliff" — it is a slope. **0.03** "b3h's catastrophe" — not here: the worst stability of the fixed
cells and the best density. **0.1 anneal** "possibly the best late stability" — the worst in the batch. Not settled:
0.03 at n=4 with one seed carrying the density, and whether its richer top (three 99.3 /30,000 rows in `b15aw`, and
the batch's `hof30k` best, 99.4 in `b15ay` at 0.1→0.001) is worth 14% of evals below 80%.
<!-- /reading -->

### Every arm

| arm | entropy_coef | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b15aa-ent0-seed1` | 0.0 | 1208 | 11.9% | 10 | 99.4 | 98.0 @46.0M | 96.1 | 0.6% |
| `b15ab-ent0-seed2` | 0.0 | 1104 | 7.7% | 3 | 99.4 | 97.5 @39.3M | 95.0 | 0.03% |
| `b15ac-ent0-seed3` | 0.0 | 1053 | 8.5% | 2 | 99.4 | 97.6 @18.0M | 94.5 | 0.21% |
| `b15ad-ent0-seed4` | 0.0 | 1261 | 17.7% | 20 | 99.6 | 97.9 @33.8M | 93.4 | 0.24% |
| `b15ae-ent001-seed1` | 0.001 | 986 | 8.9% | 5 | 99.2 | 97.9 @22.2M | 95.3 | 0.03% |
| `b15af-ent001-seed2` | 0.001 | 907 | 7.2% | 2 | 99.0 | 97.7 @9.3M | 94.8 | 0.0% |
| `b15ag-ent001-seed3` | 0.001 | 1051 | 10.8% | 9 | 99.4 | 98.2 @6.3M | 94.1 | 0.03% |
| `b15ah-ent001-seed4` | 0.001 | 925 | 7.5% | 2 | 99.2 | 97.8 @36.7M | 94.9 | 0.0% |
| `b15ai-ent003-seed1` | 0.003 | 955 | 6.7% | 1 | 99.2 | 97.5 @30.5M | 94.6 | 0.0% |
| `b15aj-ent003-seed2` | 0.003 | 1059 | 9.3% | 5 | 99.4 | 97.6 @33.6M | 95.8 | 0.0% |
| `b15ak-ent003-seed3` | 0.003 | 1071 | 9.9% | 4 | 99.2 | 97.9 @27.5M | 94.4 | 0.13% |
| `b15al-ent003-seed4` | 0.003 | 1024 | 8.6% | 5 | 99.4 | 97.6 @44.3M | 95.1 | 0.0% |
| `b15am-ent005-seed1` | 0.005 | 1026 | 10.3% | 2 | 99.4 | 97.7 @43.8M | 95.0 | 0.14% |
| `b15an-ent005-seed2` | 0.005 | 1029 | 15.0% | 10 | 99.6 | 97.4 @45.4M | 94.6 | 0.34% |
| `b15ao-ent005-seed3` | 0.005 | 1024 | 9.7% | 6 | 99.8 | 97.5 @32.9M | 94.3 | 0.24% |
| `b15ap-ent005-seed4` | 0.005 | 962 | 12.2% | 6 | 99.6 | 97.4 @37.7M | 93.0 | 0.2% |
| `b15aq-ent02-seed1` | 0.02 | 722 | 11.2% | 9 | 99.4 | 97.3 @25.9M | 87.3 | 0.95% |
| `b15ar-ent02-seed2` | 0.02 | 747 | 15.5% | 17 | 99.4 | 97.8 @39.9M | 86.0 | 0.64% |
| `b15as-ent02-seed3` | 0.02 | 902 | 12.9% | 6 | 99.6 | 97.8 @49.5M | 88.2 | 0.27% |
| `b15at-ent02-seed4` | 0.02 | 750 | 16.0% | 10 | 99.4 | 97.4 @42.5M | 86.1 | 1.41% |
| `b15au-ent03-seed1` | 0.03 | 658 | 18.1% | 24 | 99.6 | 97.3 @18.3M | 85.0 | 0.48% |
| `b15av-ent03-seed2` | 0.03 | 541 | 16.5% | 8 | 99.4 | 97.9 @35.0M | 79.7 | 1.34% |
| `b15aw-ent03-seed3` | 0.03 | 1029 | 34.5% | 74 | 99.6 | 98.3 @46.6M | 85.1 | 1.01% |
| `b15ax-ent03-seed4` | 0.03 | 587 | 19.8% | 16 | 99.4 | 98.0 @49.0M | 80.0 | 0.31% |
| `b15ay-entanneal10-seed1` | entanneal10 | 513 | 22.6% | 16 | 99.6 | 97.7 @48.7M | 66.9 | 4.47% |
| `b15az-entanneal10-seed2` | entanneal10 | 489 | 12.3% | 7 | 99.6 | 97.4 @48.1M | 71.0 | 2.74% |
| `b15ba-entanneal10-seed3` | entanneal10 | 591 | 15.6% | 9 | 99.2 | 97.9 @46.4M | 72.0 | 1.95% |
| `b15bb-entanneal10-seed4` | entanneal10 | 441 | 15.4% | 4 | 99.2 | 97.2 @46.4M | 72.1 | 2.56% |
| `b15bc-entanneal03-seed1` | entanneal03 | 725 | 8.7% | 4 | 99.4 | 97.5 @25.0M | 84.4 | 0.38% |
| `b15bd-entanneal03-seed2` | entanneal03 | 761 | 11.3% | 4 | 99.4 | 97.4 @35.0M | 83.9 | 2.0% |
| `b15be-entanneal03-seed3` | entanneal03 | 786 | 10.2% | 8 | 99.4 | 97.4 @31.9M | 84.9 | 1.51% |
| `b15bf-entanneal03-seed4` | entanneal03 | 913 | 17.7% | 19 | 99.6 | 97.7 @31.7M | 90.0 | 0.24% |
| `b15bg-entanneal01-seed1` | entanneal01 | 904 | 10.5% | 5 | 99.2 | 97.5 @11.8M | 91.6 | 0.0% |
| `b15bh-entanneal01-seed2` | entanneal01 | 1055 | 14.5% | 7 | 99.2 | 97.7 @40.0M | 93.2 | 0.17% |
| `b15bi-entanneal01-seed3` | entanneal01 | 872 | 13.5% | 15 | 99.6 | 97.7 @16.5M | 87.8 | 0.25% |
| `b15bj-entanneal01-seed4` | entanneal01 | 1052 | 13.2% | 12 | 99.6 | 98.2 @37.1M | 93.1 | 0.2% |
| `b15bk-entanneal01to0-seed1` | entanneal01to0 | 1016 | 12.7% | 12 | 99.4 | 97.5 @41.4M | 90.6 | 0.0% |
| `b15bl-entanneal01to0-seed2` | entanneal01to0 | 1056 | 12.3% | 9 | 99.2 | 97.6 @24.2M | 93.3 | 0.5% |
| `b15bm-entanneal01to0-seed3` | entanneal01to0 | 952 | 7.6% | 2 | 99.6 | 97.5 @38.9M | 92.1 | 0.47% |
| `b15bn-entanneal01to0-seed4` | entanneal01to0 | 999 | 10.3% | 4 | 99.4 | 97.6 @48.4M | 94.5 | 0.0% |

<!-- /progress_update: batch b15 -->

<!-- progress_update: batch b14 -->
## Batch b14 — the `rollout` sweep, 6 values x 4 seeds, 50M, closed 2026-09-05

Closed on the desktop; every arm has its stage-B measurement. One knob off the reference cell (`b9bw-lam99-seed1, b9bx-lam99-seed2, b9by-lam99-seed3, b9bz-lam99-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| rollout | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| 32 | 7,470 | 13.0% | 11.3 9.9 10.8 18.0 | 78 | 99.8 | 97.88 (97.5-98.1) | 78.8 | 0.22% | 17.14% | 8.9% |
| 64 | 7,123 | 22.9% | 23.3 21.4 25.9 21.6 | 246 | 100.0 | 98.30 (98.1-98.7) | 88.7 | 0.18% | 6.59% | 18.6% |
| **128** (reference) | 5,173 | 27.3% | 27.2 24.1 31.4 26.0 | 138 | 100.0 | 98.33 (98.3-98.4) | 90.6 | 0.77% | 6.43% | 28.9% |
| 192 | 3,461 | 22.5% | 18.9 16.7 26.5 26.6 | 73 | 99.8 | 98.30 (98.1-98.5) | 90.4 | 1.29% | 6.18% | 28.1% |
| 256 | 2,859 | 29.4% | 31.5 33.9 18.9 32.2 | 108 | 99.8 | 98.40 (98.1-98.7) | 90.6 | 0.85% | 4.71% | 32.0% |
| 512 | 1,778 | 38.3% | 45.8 34.2 43.0 28.9 | 94 | 100.0 | 98.38 (98.2-98.5) | 89.8 | 0.96% | 4.45% | 43.4% |
| 1024 | 759 | 31.2% | 27.4 24.6 52.3 19.3 | 30 | 99.6 | 98.22 (98.0-98.5) | 87.2 | 0.29% | 2.89% | 34.5% |

<!-- reading -->
**Verdict: density rises with rollout to 512 and stability rises with it all the way to 1024; the base 128 is the
short side of the plateau, and 512 joins the corner grid.** Run on the laptop (dequeued 2026-09-03). Read against the
bold reference at 128 — b9's λ 0.99 cell, which is right for this batch: b14 was generated at λ 0.99, unlike b15-b21.
Density 13.0% at 32, 22.9% at 64, **27.3% at 128**, 22.5% at 192, 29.4% at 256, **38.3% at 512** (seeds 28.9-45.8, a
100/500 row), 31.2% at 1024 (one seed at 52.3%, one at 19.3%). The share of evals below 80% falls without a reversal:
17.1% at 32, 6.6 → 6.4 → 6.2% at 64-192, 4.7% at 256, 4.5% at 512, **2.9% at 1024**; best30 is flat at 98.2-98.4 from
64 up. Rows and candidate counts are not comparable along the row — a checkpoint per update means 32 writes ten times
the checkpoints of 1024 — so read shares: ≥99 rows per stage-B row are 2.7% at the reference, 3.5% at 64, 5.3% at 512,
4.0% at 1024. Predictions: **32** worse than 64 — confirmed on every column. **64** slightly worse everywhere — worse on
density only; best30, stability and best row match. **192** within noise — 5 pp below on density, matched elsewhere;
wave 2's noisy stage A did not survive stage B. **256** equal-or-better stability at similar density — confirmed.
**512** later onset, good stability if it arrives — arrived, with the batch's best density and second-best stability.
**1024** "381 update rounds is too few by 50M" — falsified: it arrives, at the best stability in the batch and the
widest seed spread. Not settled: 512 against 256 at n=4 (9 pp, with one 256 seed at 18.9).
<!-- /reading -->

### Every arm

| arm | rollout | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b14a-roll32-seed1` | 32 | 2374 | 11.3% | 15 | 99.4 | 97.9 @16.8M | 86.0 | 0.07% |
| `b14b-roll32-seed2` | 32 | 1267 | 9.9% | 5 | 99.4 | 98.0 @33.4M | 73.1 | 0.33% |
| `b14c-roll32-seed3` | 32 | 1513 | 10.8% | 13 | 99.4 | 97.5 @40.3M | 75.7 | 0.45% |
| `b14d-roll32-seed4` | 32 | 2316 | 18.0% | 45 | 99.8 | 98.1 @25.6M | 80.3 | 0.11% |
| `b14e-roll64-seed1` | 64 | 1920 | 23.3% | 70 | 99.8 | 98.1 @44.3M | 90.5 | 0.19% |
| `b14f-roll64-seed2` | 64 | 1922 | 21.4% | 66 | 99.8 | 98.3 @17.6M | 92.0 | 0.17% |
| `b14g-roll64-seed3` | 64 | 1459 | 25.9% | 67 | 99.8 | 98.7 @42.1M | 81.0 | 1.55% |
| `b14h-roll64-seed4` | 64 | 1822 | 21.6% | 43 | 100.0 | 98.1 @40.9M | 91.3 | 0.17% |
| `b14i-roll192-seed1` | 192 | 798 | 18.9% | 13 | 99.4 | 98.3 @27.8M | 91.8 | 1.27% |
| `b14j-roll192-seed2` | 192 | 804 | 16.7% | 11 | 99.2 | 98.1 @35.8M | 88.6 | 0.92% |
| `b14k-roll192-seed3` | 192 | 942 | 26.5% | 26 | 99.8 | 98.3 @36.8M | 92.3 | 1.32% |
| `b14l-roll192-seed4` | 192 | 917 | 26.6% | 23 | 99.6 | 98.5 @43.9M | 89.0 | 2.05% |
| `b14m-roll256-seed1` | 256 | 806 | 31.5% | 33 | 99.8 | 98.6 @42.0M | 93.1 | 0.81% |
| `b14n-roll256-seed2` | 256 | 722 | 33.9% | 27 | 99.6 | 98.2 @41.8M | 90.1 | 0.63% |
| `b14o-roll256-seed3` | 256 | 644 | 18.9% | 10 | 99.6 | 98.1 @27.6M | 90.2 | 0.89% |
| `b14p-roll256-seed4` | 256 | 687 | 32.2% | 38 | 99.6 | 98.7 @45.5M | 89.1 | 1.11% |
| `b14q-roll512-seed1` | 512 | 480 | 45.8% | 40 | 99.8 | 98.5 @28.5M | 87.5 | 0.44% |
| `b14r-roll512-seed2` | 512 | 438 | 34.2% | 12 | 99.4 | 98.2 @38.5M | 91.3 | 1.09% |
| `b14s-roll512-seed3` | 512 | 442 | 43.0% | 30 | 100.0 | 98.4 @40.5M | 91.5 | 0.82% |
| `b14t-roll512-seed4` | 512 | 418 | 28.9% | 12 | 99.4 | 98.4 @26.6M | 88.9 | 1.11% |
| `b14u-roll1024-seed1` | 1024 | 186 | 27.4% | 5 | 99.6 | 98.5 @48.9M | 89.5 | 0.0% |
| `b14v-roll1024-seed2` | 1024 | 187 | 24.6% | 3 | 99.0 | 98.1 @46.1M | 90.1 | 0.0% |
| `b14w-roll1024-seed3` | 1024 | 199 | 52.3% | 21 | 99.4 | 98.3 @34.6M | 83.0 | 2.09% |
| `b14x-roll1024-seed4` | 1024 | 187 | 19.3% | 1 | 99.2 | 98.0 @44.7M | 86.4 | 0.58% |

<!-- /progress_update: batch b14 -->

<!-- progress_update: batch b13 -->
## Batch b13 — the `minibatch` sweep, 8 values x 4 seeds, 50M, closed 2026-09-04

Closed on the desktop; every arm has its stage-B measurement. One knob off the reference cell (`b9bw-lam99-seed1, b9bx-lam99-seed2, b9by-lam99-seed3, b9bz-lam99-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| minibatch | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| 32 | 1,740 | 6.8% | 2.6 6.7 6.9 9.6 | 5 | 99.4 | 96.78 (95.9-97.2) | 83.4 | 0.97% | 15.07% | 7.8% |
| 64 | 3,308 | 13.3% | 16.5 14.5 12.7 9.4 | 42 | 99.8 | 97.60 (97.3-97.9) | 91.9 | 0.34% | 3.72% | 16.1% |
| 128 | 4,566 | 19.1% | 16.5 17.6 20.5 21.4 | 83 | 99.6 | 98.12 (97.8-98.3) | 92.4 | 0.32% | 4.68% | 23.9% |
| 192 | 4,271 | 16.8% | 14.6 8.8 21.3 20.2 | 54 | 99.6 | 97.95 (97.6-98.2) | 91.8 | 0.47% | 4.36% | 21.4% |
| **256** (reference) | 5,173 | 27.3% | 27.2 24.1 31.4 26.0 | 138 | 100.0 | 98.33 (98.3-98.4) | 90.6 | 0.77% | 6.43% | 28.9% |
| 384 | 4,934 | 26.9% | 24.7 20.5 15.9 41.0 | 166 | 99.6 | 98.28 (98.1-98.5) | 90.6 | 1.08% | 6.81% | 27.2% |
| 512 | 5,062 | 28.2% | 22.5 25.8 23.7 37.8 | 188 | 100.0 | 98.47 (98.3-98.8) | 88.9 | 0.93% | 8.02% | 28.5% |
| 1024 | 4,239 | 19.2% | 19.9 21.4 17.4 18.1 | 75 | 99.6 | 98.05 (97.9-98.2) | 84.3 | 1.42% | 12.07% | 22.4% |
| 2048 | 3,215 | 18.3% | 6.4 7.5 40.1 11.8 | 67 | 99.8 | 98.03 (97.6-99.0) | 78.7 | 1.6% | 14.32% | 16.8% |

<!-- reading -->
**Verdict: minibatch is a plateau at 256-512 with a soft low end and a noisy high end; the base stays
256.** Run on the laptop (dequeued from the desktop 2026-09-03). Density 6.8 → 13.3 → 19.1 → 16.8% at 32-192,
**27.3% at 256** (the reference), 26.9% at 384, **28.2% at 512** (the batch's 100/500 row, 188 candidates,
best30 98.47), 19.2% at 1024, 18.3% at 2048 (one seed at 40%, three at 6-12%). Stability is U-shaped the
other way: below-80% share 15% at 32, **3.7-4.7% at 64-192**, 6.4-8.0% at 256-512, 12-14% at 1024-2048;
`sef` 92 at 64-128 falling to 79 at 2048. In updates per epoch (16,384 / minibatch) the density maximum is
at 32-64 updates and stability rises with the update count until the minibatch is too small to estimate
the advantage scale. Predictions: **32** fast-onset-high-collapse confirmed, and it is not worse than 16
epochs, so reuse is the mechanism. **64** milder 32, confirmed. **128** "likeliest alternative default"
falsified — 8 pp below on all four seeds. **192** "within noise" falsified. **384** within noise confirmed.
**512** "slower, smoother" — equal-or-better and *less* smooth. **1024** "may not reach the record region"
falsified (19.2%, 75 candidates). **2048** arrives, on one seed. Not settled: 512 vs 256 (0.9 pp at n=4);
512 goes into the corner grid as a second value.
<!-- /reading -->

### Every arm

| arm | minibatch | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b13aa-mb32-seed1` | 32 | 384 | 2.6% | 0 | 98.6 | 95.9 @5.4M | 82.5 | 5.65% |
| `b13ab-mb32-seed2` | 32 | 328 | 6.7% | 1 | 99.0 | 96.9 @17.5M | 81.2 | 1.0% |
| `b13ac-mb32-seed3` | 32 | 432 | 6.9% | 1 | 99.0 | 97.2 @12.2M | 82.7 | 0.94% |
| `b13ad-mb32-seed4` | 32 | 596 | 9.6% | 3 | 99.4 | 97.1 @7.9M | 87.1 | 0.44% |
| `b13ae-mb64-seed1` | 64 | 788 | 16.5% | 15 | 99.6 | 97.9 @34.6M | 91.0 | 0.97% |
| `b13af-mb64-seed2` | 64 | 958 | 14.5% | 13 | 99.8 | 97.6 @44.9M | 91.3 | 0.24% |
| `b13ag-mb64-seed3` | 64 | 722 | 12.7% | 4 | 99.4 | 97.3 @20.5M | 94.3 | 0.44% |
| `b13ah-mb64-seed4` | 64 | 840 | 9.4% | 10 | 99.8 | 97.6 @5.3M | 91.1 | 0.07% |
| `b13ai-mb128-seed1` | 128 | 1089 | 16.5% | 10 | 99.4 | 97.8 @38.5M | 90.2 | 0.37% |
| `b13aj-mb128-seed2` | 128 | 1059 | 17.6% | 17 | 99.4 | 98.1 @16.9M | 93.0 | 0.3% |
| `b13ak-mb128-seed3` | 128 | 1164 | 20.5% | 24 | 99.4 | 98.3 @19.1M | 91.8 | 0.34% |
| `b13al-mb128-seed4` | 128 | 1254 | 21.4% | 32 | 99.6 | 98.3 @29.3M | 94.5 | 0.07% |
| `b13am-mb192-seed1` | 192 | 1065 | 14.6% | 13 | 99.6 | 98.1 @33.7M | 92.8 | 0.61% |
| `b13an-mb192-seed2` | 192 | 857 | 8.8% | 5 | 99.2 | 97.6 @40.3M | 92.5 | 0.23% |
| `b13ao-mb192-seed3` | 192 | 1161 | 21.3% | 16 | 99.4 | 98.2 @16.2M | 91.8 | 0.34% |
| `b13ap-mb192-seed4` | 192 | 1188 | 20.2% | 20 | 99.6 | 97.9 @18.1M | 90.2 | 0.66% |
| `b13aq-mb384-seed1` | 384 | 1224 | 24.7% | 32 | 99.6 | 98.4 @22.5M | 90.0 | 1.52% |
| `b13ar-mb384-seed2` | 384 | 1145 | 20.5% | 22 | 99.6 | 98.1 @33.9M | 91.5 | 0.47% |
| `b13as-mb384-seed3` | 384 | 1036 | 15.9% | 16 | 99.4 | 98.1 @48.3M | 91.7 | 0.64% |
| `b13at-mb384-seed4` | 384 | 1529 | 41.0% | 96 | 99.6 | 98.5 @25.8M | 89.2 | 2.4% |
| `b13au-mb512-seed1` | 512 | 1198 | 22.5% | 27 | 99.8 | 98.3 @40.7M | 87.6 | 2.26% |
| `b13av-mb512-seed2` | 512 | 1223 | 25.8% | 40 | 99.6 | 98.5 @34.3M | 89.1 | 0.27% |
| `b13aw-mb512-seed3` | 512 | 1121 | 23.7% | 36 | 100.0 | 98.3 @31.6M | 88.5 | 0.27% |
| `b13ax-mb512-seed4` | 512 | 1520 | 37.8% | 85 | 99.8 | 98.8 @48.6M | 90.4 | 1.59% |
| `b13ay-mb1024-seed1` | 1024 | 1154 | 19.9% | 32 | 99.6 | 98.2 @24.6M | 83.9 | 1.42% |
| `b13az-mb1024-seed2` | 1024 | 1038 | 21.4% | 17 | 99.6 | 98.1 @48.6M | 82.7 | 1.21% |
| `b13ba-mb1024-seed3` | 1024 | 1061 | 17.4% | 14 | 99.4 | 98.0 @36.7M | 86.5 | 1.42% |
| `b13bb-mb1024-seed4` | 1024 | 986 | 18.1% | 12 | 99.4 | 97.9 @28.7M | 84.0 | 1.52% |
| `b13bc-mb2048-seed1` | 2048 | 780 | 6.4% | 0 | 98.8 | 97.7 @49.1M | 70.5 | 4.09% |
| `b13bd-mb2048-seed2` | 2048 | 657 | 7.5% | 2 | 99.4 | 97.6 @48.0M | 82.7 | 0.67% |
| `b13be-mb2048-seed3` | 2048 | 982 | 40.1% | 62 | 99.8 | 99.0 @46.3M | 79.7 | 2.02% |
| `b13bf-mb2048-seed4` | 2048 | 796 | 11.8% | 3 | 99.2 | 97.8 @46.0M | 81.7 | 1.18% |

<!-- /progress_update: batch b13 -->

<!-- progress_update: batch b12 -->
## Batch b12 — the `epochs` sweep, 10 values x 4 seeds, 50M, closed 2026-09-04

Closed on the desktop; every arm has its stage-B measurement. One knob off the reference cell (`b9bw-lam99-seed1, b9bx-lam99-seed2, b9by-lam99-seed3, b9bz-lam99-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| epochs | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | 2,425 | 10.8% | 7.4 4.2 9.4 20.8 | 15 | 99.4 | 97.78 (97.5-98.3) | 73.8 | 4.2% | 19.31% | 11.9% |
| 2 | 4,171 | 15.7% | 13.3 8.4 21.8 17.8 | 35 | 99.4 | 98.07 (98.0-98.1) | 84.3 | 2.58% | 11.04% | 21.1% |
| 3 | 4,890 | 26.5% | 27.9 23.2 32.0 22.4 | 165 | 99.8 | 98.30 (97.9-98.6) | 89.5 | 0.66% | 6.56% | 26.5% |
| **4** (reference) | 5,173 | 27.3% | 27.2 24.1 31.4 26.0 | 138 | 100.0 | 98.33 (98.3-98.4) | 90.6 | 0.77% | 6.43% | 28.9% |
| 5 | 4,707 | 24.6% | 23.9 14.1 23.5 33.2 | 139 | 99.8 | 98.22 (97.5-98.8) | 90.5 | 0.49% | 6.18% | 25.4% |
| 6 | 4,598 | 23.1% | 17.1 28.4 22.2 23.0 | 136 | 99.8 | 98.22 (97.8-98.5) | 91.0 | 0.54% | 4.48% | 24.3% |
| 7 | 4,439 | 19.3% | 10.7 25.5 22.0 16.7 | 92 | 99.8 | 98.20 (97.6-98.6) | 92.2 | 0.51% | 4.27% | 23.3% |
| 8 | 3,344 | 14.7% | 8.0 19.7 16.6 10.1 | 43 | 100.0 | 97.78 (97.5-98.2) | 92.4 | 0.17% | 4.99% | 16.5% |
| 10 | 2,323 | 8.7% | 10.1 6.2 9.3 8.5 | 15 | 99.2 | 97.53 (96.6-97.9) | 91.2 | 0.27% | 6.61% | 10.4% |
| 12 | 1,485 | 6.9% | 7.9 7.8 7.2 4.4 | 10 | 99.4 | 97.10 (96.4-97.7) | 83.8 | 1.31% | 12.93% | 6.5% |
| 16 | 419 | 6.0% | 1.6 7.6 7.8 4.8 | 2 | 99.6 | 96.25 (95.0-97.9) | 56.7 | 11.88% | 42.28% | 1.7% |

<!-- reading -->
**Verdict: 3-4 epochs is the top; density falls monotonically past 4 and stability breaks at 12-16. Base
stays 4.** Density 26.5% at 3 and 27.3% at 4 (the reference), then 24.6 → 23.1 → 19.3 → 14.7 → 8.7 → 6.9 →
6.0% at 5 through 16; best30 holds ~98.2 through 7 and slides to 96.25 at 16. Stability improves with epochs
until it breaks: below-80% share 19.3% at 1, 11.0% at 2, 6.2-6.6% at 3-5, **4.3-5.0% at 6-8** (8 has the
lowest drawdown in the batch, 0.17%, and the one 100/500 row, `b12ay` @8.3M), then 6.6% at 10, 12.9% at 12,
**42% at 16** with `sef` 57. Cell by cell: **1** predicted "very stable, slow" — slow yes, stable no: the
least stable cell short of 16. **2** stable-but-slower, as predicted, at a real cost (15.7%). **3, 5** within
noise of the base. **6, 7** slightly below on density, best on stability so far — the "first sign of b4's
collapse pattern" did not appear. **8** predicted to collapse far above the base; it is instead the most
stable cell — b4's collapses at 8 epochs were its `fc (200,100)` net's. **10, 12** the slide: density 8.7%
and 6.9%, collapses visible at 12. **16** the cliff, as predicted. Not settled: whether 3 is *better* than 4
on stability (6.6% vs 6.4% is nothing at n=4). The 16 × lr 1e-4 and 16 × clip 0.1 cells stay in the factorial.
<!-- /reading -->

### Every arm

| arm | epochs | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b12aa-ep1-seed1` | 1 | 618 | 7.4% | 2 | 99.4 | 97.5 @46.3M | 71.8 | 5.15% |
| `b12ab-ep1-seed2` | 1 | 613 | 4.2% | 1 | 99.2 | 97.6 @35.9M | 74.9 | 4.04% |
| `b12ac-ep1-seed3` | 1 | 510 | 9.4% | 5 | 99.0 | 97.7 @48.9M | 73.1 | 3.09% |
| `b12ad-ep1-seed4` | 1 | 684 | 20.8% | 7 | 99.2 | 98.3 @44.2M | 75.5 | 4.36% |
| `b12ae-ep2-seed1` | 2 | 1087 | 13.3% | 12 | 99.4 | 98.1 @27.0M | 87.0 | 1.37% |
| `b12af-ep2-seed2` | 2 | 850 | 8.4% | 3 | 99.2 | 98.0 @32.7M | 83.1 | 3.11% |
| `b12ag-ep2-seed3` | 2 | 1067 | 21.8% | 12 | 99.4 | 98.1 @48.6M | 80.4 | 2.59% |
| `b12ah-ep2-seed4` | 2 | 1167 | 17.8% | 8 | 99.4 | 98.1 @31.1M | 86.7 | 2.56% |
| `b12ai-ep3-seed1` | 3 | 1332 | 27.9% | 50 | 99.8 | 98.4 @48.4M | 89.0 | 0.51% |
| `b12aj-ep3-seed2` | 3 | 1149 | 23.2% | 36 | 99.6 | 97.9 @41.7M | 90.7 | 0.81% |
| `b12ak-ep3-seed3` | 3 | 1226 | 32.0% | 59 | 99.8 | 98.6 @33.4M | 89.3 | 1.27% |
| `b12al-ep3-seed4` | 3 | 1183 | 22.4% | 20 | 99.6 | 98.3 @48.4M | 89.0 | 0.31% |
| `b12am-ep5-seed1` | 5 | 1013 | 23.9% | 27 | 99.4 | 98.3 @36.0M | 88.0 | 0.45% |
| `b12an-ep5-seed2` | 5 | 978 | 14.1% | 11 | 99.6 | 97.5 @14.8M | 92.8 | 0.54% |
| `b12ao-ep5-seed3` | 5 | 1257 | 23.5% | 27 | 99.8 | 98.3 @47.3M | 89.6 | 1.28% |
| `b12ap-ep5-seed4` | 5 | 1459 | 33.2% | 74 | 99.6 | 98.8 @48.7M | 91.8 | 0.41% |
| `b12aq-ep6-seed1` | 6 | 924 | 17.1% | 14 | 99.6 | 97.8 @47.8M | 89.3 | 0.57% |
| `b12ar-ep6-seed2` | 6 | 1282 | 28.4% | 56 | 99.8 | 98.2 @8.8M | 93.0 | 0.64% |
| `b12as-ep6-seed3` | 6 | 1237 | 22.2% | 32 | 99.8 | 98.4 @31.3M | 90.1 | 0.52% |
| `b12at-ep6-seed4` | 6 | 1155 | 23.0% | 34 | 99.6 | 98.5 @35.7M | 91.6 | 0.21% |
| `b12au-ep7-seed1` | 7 | 897 | 10.7% | 6 | 99.4 | 97.6 @36.7M | 92.4 | 0.64% |
| `b12av-ep7-seed2` | 7 | 1180 | 25.5% | 38 | 99.8 | 98.6 @32.3M | 92.1 | 0.14% |
| `b12aw-ep7-seed3` | 7 | 1256 | 22.0% | 30 | 99.6 | 98.5 @45.4M | 93.7 | 0.37% |
| `b12ax-ep7-seed4` | 7 | 1106 | 16.7% | 18 | 99.6 | 98.1 @35.2M | 90.8 | 0.84% |
| `b12ay-ep8-seed1` | 8 | 614 | 8.0% | 3 | 100.0 | 97.8 @23.4M | 91.5 | 1.58% |
| `b12az-ep8-seed2` | 8 | 1132 | 19.7% | 17 | 99.6 | 98.2 @39.5M | 94.0 | 0.17% |
| `b12ba-ep8-seed3` | 8 | 913 | 16.6% | 20 | 99.6 | 97.6 @30.6M | 92.8 | 0.17% |
| `b12bb-ep8-seed4` | 8 | 685 | 10.1% | 3 | 99.8 | 97.5 @12.0M | 91.3 | 0.1% |
| `b12bc-ep10-seed1` | 10 | 586 | 10.1% | 5 | 99.2 | 97.8 @11.8M | 88.8 | 0.44% |
| `b12bd-ep10-seed2` | 10 | 486 | 6.2% | 2 | 99.2 | 96.6 @22.9M | 90.8 | 1.01% |
| `b12be-ep10-seed3` | 10 | 701 | 9.3% | 3 | 99.2 | 97.9 @5.7M | 93.4 | 0.1% |
| `b12bf-ep10-seed4` | 10 | 550 | 8.5% | 5 | 99.2 | 97.8 @13.3M | 91.8 | 0.07% |
| `b12bg-ep12-seed1` | 12 | 228 | 7.9% | 1 | 99.0 | 97.1 @4.7M | 74.5 | 2.47% |
| `b12bh-ep12-seed2` | 12 | 501 | 7.8% | 5 | 99.2 | 97.2 @5.7M | 90.0 | 0.34% |
| `b12bi-ep12-seed3` | 12 | 416 | 7.2% | 1 | 99.0 | 97.7 @10.2M | 89.9 | 0.77% |
| `b12bj-ep12-seed4` | 12 | 340 | 4.4% | 3 | 99.4 | 96.4 @7.1M | 80.9 | 1.85% |
| `b12bk-ep16-seed1` | 16 | 61 | 1.6% | 0 | 98.0 | 95.6 @9.1M | 49.8 | 15.14% |
| `b12bl-ep16-seed2` | 16 | 79 | 7.6% | 0 | 98.4 | 96.5 @3.9M | 50.2 | 12.49% |
| `b12bm-ep16-seed3` | 16 | 153 | 7.8% | 2 | 99.6 | 97.9 @3.6M | 63.1 | 7.25% |
| `b12bn-ep16-seed4` | 16 | 126 | 4.8% | 0 | 98.8 | 95.0 @7.2M | 63.8 | 11.28% |

<!-- /progress_update: batch b12 -->

<!-- progress_update: batch b11 -->
## Batch b11 — the `learning_rate` sweep, 8 values x 4 seeds, 50M, closed 2026-09-04

Closed on the desktop; every arm has its stage-B measurement. One knob off the reference cell (`b9bw-lam99-seed1, b9bx-lam99-seed2, b9by-lam99-seed3, b9bz-lam99-seed4`, marked in the table). Numbers by `tools/progress_update.py`.

| learning_rate | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) | sef | drawdown < 50% | < 80% | stage-A ≥98% |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| 4e-5 | 1,411 | 2.2% | 3.4 1.9 2.8 1.3 | 0 | 98.6 | 97.17 (97.0-97.6) | 73.4 | 0.83% | 14.28% | 5.8% |
| 1e-4 | 4,496 | 24.8% | 17.5 13.0 43.7 17.6 | 135 | 100.0 | 98.17 (97.8-98.6) | 85.1 | 1.41% | 9.49% | 24.8% |
| 1.5e-4 | 4,836 | 20.8% | 15.1 19.0 21.8 26.5 | 81 | 99.8 | 98.22 (98.0-98.4) | 87.4 | 0.53% | 7.34% | 25.9% |
| 2.5e-4 | 4,964 | 30.9% | 34.8 27.8 28.8 32.0 | 210 | 100.0 | 98.45 (98.2-98.7) | 89.4 | 0.71% | 6.29% | 27.9% |
| **3e-4** (reference) | 5,173 | 27.3% | 27.2 24.1 31.4 26.0 | 138 | 100.0 | 98.33 (98.3-98.4) | 90.6 | 0.77% | 6.43% | 28.9% |
| 5e-4 | 4,636 | 21.6% | 20.5 23.2 21.7 21.0 | 96 | 99.8 | 98.15 (98.0-98.4) | 91.5 | 0.59% | 5.41% | 24.6% |
| 8e-4 | 4,031 | 16.3% | 13.3 7.7 26.6 14.8 | 73 | 99.6 | 98.08 (97.8-98.2) | 90.2 | 0.45% | 4.56% | 20.6% |
| 1e-3 | 3,653 | 15.8% | 8.9 9.4 24.6 16.1 | 61 | 99.6 | 97.70 (97.1-98.2) | 92.2 | 0.25% | 4.91% | 18.6% |
| 2e-3 | 2,298 | 6.7% | 8.1 3.6 7.4 6.7 | 5 | 99.4 | 96.62 (95.9-97.2) | 88.1 | 1.68% | 9.97% | 10.2% |

<!-- reading -->
**Verdict: the learning rate is a plateau from 1e-4 to 5e-4 and a cliff on both sides; the base stays at
3e-4.** Density 21-31% and best30 98.15-98.45 across the five middle cells, with per-seed spread inside a cell
(13-44% at 1e-4) wider than the differences between cells. 2.5e-4 is the best cell (30.9%, 210 ≥99 rows,
two 100/500 rows) but n=4 cannot separate it from the reference's 27.3%. Cell by cell: **4e-5** slow onset
as predicted, still climbing at 50M, zero ≥99 rows. **1e-4** falsified its prediction of "zero ≥98 rows" with
24.8% and 135 candidates, though it does spend more evals below 80% (9.5% vs 6.4%). **1.5e-4, 2.5e-4, 5e-4**
within noise of the base. **8e-4 and 1e-3** were predicted to collapse visibly and are instead the two most
stable cells (4.6-4.9% below 80%, `sef` 90-92) with a lower ceiling (16% density, best30 97.7-98.1).
**2e-3** locates the cliff: 6.7%, best30 96.6, drawdown 1.68%. `sef` rises monotonically with lr (73 → 92)
while density peaks in the middle — the two metrics rank the sweep differently, as b7 found. Not settled:
whether 2.5e-4 is a real 3-pp gain; it goes into the γ × λ corner grid as a second lr value.
<!-- /reading -->

### Every arm

| arm | learning_rate | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b11aa-lr4e5-seed1` | 4e-5 | 264 | 3.4% | 0 | 98.6 | 97.0 @49.6M | 65.2 | 5.26% |
| `b11ab-lr4e5-seed2` | 4e-5 | 419 | 1.9% | 0 | 98.4 | 97.6 @48.1M | 78.9 | 0.11% |
| `b11ac-lr4e5-seed3` | 4e-5 | 351 | 2.8% | 0 | 98.6 | 97.1 @47.3M | 74.1 | 1.26% |
| `b11ad-lr4e5-seed4` | 4e-5 | 377 | 1.3% | 0 | 98.4 | 97.0 @42.5M | 75.5 | 0.41% |
| `b11ae-lr1e4-seed1` | 1e-4 | 989 | 17.5% | 12 | 99.2 | 98.2 @38.2M | 81.9 | 2.3% |
| `b11af-lr1e4-seed2` | 1e-4 | 953 | 13.0% | 4 | 99.0 | 97.8 @46.6M | 87.3 | 0.52% |
| `b11ag-lr1e4-seed3` | 1e-4 | 1419 | 43.7% | 106 | 100.0 | 98.6 @22.0M | 87.7 | 0.45% |
| `b11ah-lr1e4-seed4` | 1e-4 | 1135 | 17.6% | 13 | 99.2 | 98.1 @36.7M | 83.4 | 2.54% |
| `b11ai-lr1.5e4-seed1` | 1.5e-4 | 1143 | 15.1% | 11 | 99.4 | 98.0 @39.2M | 83.4 | 0.48% |
| `b11aj-lr1.5e4-seed2` | 1.5e-4 | 1180 | 19.0% | 15 | 99.6 | 98.1 @26.7M | 90.2 | 0.41% |
| `b11ak-lr1.5e4-seed3` | 1.5e-4 | 1186 | 21.8% | 25 | 99.8 | 98.4 @42.0M | 88.3 | 0.59% |
| `b11al-lr1.5e4-seed4` | 1.5e-4 | 1327 | 26.5% | 30 | 99.6 | 98.4 @31.4M | 87.7 | 1.77% |
| `b11am-lr2.5e4-seed1` | 2.5e-4 | 1265 | 34.8% | 81 | 99.8 | 98.7 @48.1M | 88.4 | 0.78% |
| `b11an-lr2.5e4-seed2` | 2.5e-4 | 1336 | 27.8% | 45 | 100.0 | 98.6 @19.2M | 87.9 | 0.63% |
| `b11ao-lr2.5e4-seed3` | 2.5e-4 | 1067 | 28.8% | 39 | 99.8 | 98.3 @33.0M | 90.8 | 0.98% |
| `b11ap-lr2.5e4-seed4` | 2.5e-4 | 1296 | 32.0% | 45 | 100.0 | 98.2 @12.5M | 90.5 | 0.1% |
| `b11aq-lr5e4-seed1` | 5e-4 | 1154 | 20.5% | 18 | 99.2 | 98.0 @39.9M | 91.8 | 0.68% |
| `b11ar-lr5e4-seed2` | 5e-4 | 1139 | 23.2% | 23 | 99.6 | 98.1 @22.4M | 91.5 | 0.74% |
| `b11as-lr5e4-seed3` | 5e-4 | 1141 | 21.7% | 31 | 99.8 | 98.1 @48.6M | 92.2 | 0.51% |
| `b11at-lr5e4-seed4` | 5e-4 | 1202 | 21.0% | 24 | 99.4 | 98.4 @27.6M | 90.4 | 0.34% |
| `b11au-lr8e4-seed1` | 8e-4 | 1012 | 13.3% | 18 | 99.4 | 98.1 @24.4M | 87.5 | 0.27% |
| `b11av-lr8e4-seed2` | 8e-4 | 805 | 7.7% | 0 | 98.8 | 97.8 @27.9M | 91.5 | 1.29% |
| `b11aw-lr8e4-seed3` | 8e-4 | 1138 | 26.6% | 41 | 99.6 | 98.2 @9.0M | 87.7 | 0.36% |
| `b11ax-lr8e4-seed4` | 8e-4 | 1076 | 14.8% | 14 | 99.4 | 98.2 @48.5M | 94.3 | 0.54% |
| `b11ay-lr1e3-seed1` | 1e-3 | 754 | 8.9% | 4 | 99.0 | 97.6 @13.2M | 90.4 | 0.95% |
| `b11az-lr1e3-seed2` | 1e-3 | 709 | 9.4% | 3 | 99.4 | 97.1 @25.7M | 92.0 | 0.27% |
| `b11ba-lr1e3-seed3` | 1e-3 | 1088 | 24.6% | 34 | 99.6 | 97.9 @14.5M | 91.8 | 0.21% |
| `b11bb-lr1e3-seed4` | 1e-3 | 1102 | 16.1% | 20 | 99.6 | 98.2 @7.9M | 94.5 | 0.23% |
| `b11bc-lr2e3-seed1` | 2e-3 | 683 | 8.1% | 3 | 99.4 | 97.1 @28.5M | 85.7 | 2.04% |
| `b11bd-lr2e3-seed2` | 2e-3 | 442 | 3.6% | 2 | 99.0 | 95.9 @9.5M | 87.5 | 2.48% |
| `b11be-lr2e3-seed3` | 2e-3 | 528 | 7.4% | 0 | 98.8 | 96.3 @14.4M | 87.2 | 1.31% |
| `b11bf-lr2e3-seed4` | 2e-3 | 645 | 6.7% | 0 | 98.8 | 97.2 @43.5M | 91.9 | 0.7% |

<!-- /progress_update: batch b11 -->

## Batch b10 — the discount γ sweep, 16 values x 4 seeds, closed 2026-09-03

**One knob off b7's winning cell, and again the default sat mid-ramp.** 64 arms at 50M transitions on the
desktop, eight waves of two γ values, `auto_stage_b` measuring each wave before the next trained,
2026-09-02 ~16:50 → 2026-09-03 16:08. Everything but `discount` is b7's reference — `fc (320,)`, 4
epochs, lr 3e-4, **λ 0.98**, entropy 0.01, clip 0.2, 128x128 rollout, minibatch 256, b2's reward — so
**`b7aa`-`b7ad` are the γ 0.99 row.** `SNEK_DISCOUNT` also sets the shaping discount, so the dense reward
moves with γ, correctly. Drawdown is the median share of post-competence stage-A evals (onset = first
eval ≥80% perfect) below the threshold, as for b8 and b9.

| γ | value horizon 1/(1−γ) | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (4 seeds) | sef | drawdown < 50% | < 80% |
|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|
| 0.70 | 3.3 | 0 | – | never screened | 0 | – | 2.7 5.2 5.1 2.8 | 0.0 | never competent | |
| 0.80 | 5 | 0 | – | never screened | 0 | – | 49.7 24.4 41.8 45.2 | 0.0 | never competent | |
| 0.85 | 6.7 | 0 | – | never screened | 0 | – | 67.2 64.1 66.6 68.0 | 0.0 | 54.4% (2 of 4 reached 80%) | 99.7% |
| 0.90 | 10 | 0 | – | never screened | 0 | – | 81.6 75.8 79.4 81.4 | 3.6 | 38.7% | 95.5% |
| 0.91 | 11.1 | 0 | – | never screened | 0 | – | 85.5 79.9 77.5 84.7 | 5.8 | 26.5% | 92.6% |
| 0.92 | 12.5 | 0 | – | never screened | 0 | – | 85.1 86.6 84.2 83.1 | 12.1 | 18.4% | 85.3% |
| 0.93 | 14.3 | 0 | – | never screened | 0 | – | 86.9 88.5 87.1 87.7 | 21.0 | 13.9% | 74.0% |
| 0.94 | 16.7 | 11 | 0.0% | 0 0 0 0 | 0 | 93.2 | 90.4 89.0 90.2 89.8 | 34.5 | 4.85% | 56.8% |
| 0.95 | 20 | 95 | 0.0% | 0 0 0 0 | 0 | 97.8 | 93.6 91.6 93.2 93.0 | 51.5 | 3.35% | 44.7% |
| 0.96 | 25 | 319 | 0.9% | 4.3 0 0 0 | 0 | 98.4 | 95.6 93.4 94.9 94.7 | 63.9 | 2.60% | 30.0% |
| 0.97 | 33 | 756 | 3.6% | 0.8 5.1 2.0 7.1 | 2 | 99.4 | 96.4 94.7 95.4 95.6 | 75.6 | 0.28% | 19.8% |
| 0.98 | 50 | 2,540 | 10.4% | 6.0 14.1 8.1 13.1 | 20 | 99.6 | 97.5 97.1 97.3 97.3 | 86.5 | 0.14% | 8.3% |
| 0.99 (b7) | 100 | 4,003 | 17.3% | 18.5 19.0 16.5 15.1 | 50 | 99.6 | 97.8 97.8 97.7 97.7 | 90.9 | 0.29% | 6.2% |
| 0.995 | 200 | 5,027 | 19.6% | 19.0 17.8 20.5 20.7 | 85 | **100.0** | 98.1 97.7 98.5 98.1 | 92.5 | 0.78% | **4.5%** |
| **0.9975** | 400 | 5,844 | 25.6% | 24.3 29.0 20.3 28.5 | 164 | 99.8 | **98.6 98.4 98.7 98.7** | **92.6** | 1.30% | 4.7% |
| **0.999** | 1,000 | **6,614** | **30.7%** | 21.7 22.9 39.2 35.4 | **277** | 99.8 | 98.5 98.2 98.7 98.6 | 91.7 | 1.55% | 4.9% |
| 1.00 | ∞ | 1,535 | 38.6% | 36.4 45.8 39.4 33.9 | 178 | **100.0** | 98.4 97.3 98.7 98.3 | **39.2** | **44.4%** | 63.6% |

**Density is zero through γ 0.93, appears at 0.96 and climbs monotonically to 30.7% at γ 0.999; γ 1.00
collapses.** 28 arms across the seven lowest values did not produce one screened checkpoint. From 0.96
up every step adds density and the base at 0.99 is mid-ramp; drawdown climbs from 0.29% at 0.99 to 1.55%
at 0.999. γ 1.00's deployed policy spends 44% of its post-competence evals below 50% perfect while its
surviving checkpoints are the densest in the batch. `sef` ranks the top backwards, fourth batch running.
Reading in [`findings.md`](findings.md), charts in [`charts.md`](charts.md). **No `hof5000` pass yet** —
726 candidates at ≥99/500, most at γ 0.999 (277) and 1.00 (178).

### Every arm

`rows` is the stage-B checkpoint count, `≥99` the `hof5000` candidates at the current cut, `best30 @step`
the peak 30-eval trailing perfect rate and where it landed. γ ≤ 0.93 arms screened zero checkpoints.

| arm | γ | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b10aa-g70-seed1` | 0.7 | 0 | 0.0% | 0 | 0.0 | 2.7 @1.0M | 0.0 | – |
| `b10ab-g70-seed2` | 0.7 | 0 | 0.0% | 0 | 0.0 | 5.2 @3.5M | 0.0 | – |
| `b10ac-g70-seed3` | 0.7 | 0 | 0.0% | 0 | 0.0 | 5.1 @2.1M | 0.0 | – |
| `b10ad-g70-seed4` | 0.7 | 0 | 0.0% | 0 | 0.0 | 2.8 @0.9M | 0.0 | – |
| `b10ae-g80-seed1` | 0.8 | 0 | 0.0% | 0 | 0.0 | 49.7 @14.3M | 0.0 | – |
| `b10af-g80-seed2` | 0.8 | 0 | 0.0% | 0 | 0.0 | 24.4 @14.8M | 0.0 | – |
| `b10ag-g80-seed3` | 0.8 | 0 | 0.0% | 0 | 0.0 | 41.8 @17.5M | 0.0 | – |
| `b10ah-g80-seed4` | 0.8 | 0 | 0.0% | 0 | 0.0 | 45.2 @36.2M | 0.0 | – |
| `b10ai-g85-seed1` | 0.85 | 0 | 0.0% | 0 | 0.0 | 67.2 @22.6M | 0.0 | – |
| `b10aj-g85-seed2` | 0.85 | 0 | 0.0% | 0 | 0.0 | 64.1 @40.4M | 0.0 | 62.8% |
| `b10ak-g85-seed3` | 0.85 | 0 | 0.0% | 0 | 0.0 | 66.6 @34.4M | 0.0 | – |
| `b10al-g85-seed4` | 0.85 | 0 | 0.0% | 0 | 0.0 | 68.0 @38.3M | 0.1 | 46.0% |
| `b10am-g90-seed1` | 0.9 | 0 | 0.0% | 0 | 0.0 | 81.6 @26.3M | 3.9 | 37.8% |
| `b10an-g90-seed2` | 0.9 | 0 | 0.0% | 0 | 0.0 | 75.8 @49.6M | 2.4 | 39.5% |
| `b10ao-g90-seed3` | 0.9 | 0 | 0.0% | 0 | 0.0 | 79.4 @26.4M | 2.1 | 40.9% |
| `b10ap-g90-seed4` | 0.9 | 0 | 0.0% | 0 | 0.0 | 81.4 @41.2M | 6.1 | 30.1% |
| `b10aq-g91-seed1` | 0.91 | 0 | 0.0% | 0 | 0.0 | 85.5 @30.4M | 9.4 | 15.8% |
| `b10ar-g91-seed2` | 0.91 | 0 | 0.0% | 0 | 0.0 | 79.9 @49.9M | 2.5 | 44.7% |
| `b10as-g91-seed3` | 0.91 | 0 | 0.0% | 0 | 0.0 | 77.5 @37.1M | 4.1 | 28.1% |
| `b10at-g91-seed4` | 0.91 | 0 | 0.0% | 0 | 0.0 | 84.7 @32.7M | 7.3 | 25.0% |
| `b10au-g92-seed1` | 0.92 | 0 | 0.0% | 0 | 0.0 | 85.1 @24.5M | 15.4 | 15.9% |
| `b10av-g92-seed2` | 0.92 | 0 | 0.0% | 0 | 0.0 | 86.6 @38.4M | 10.8 | 21.4% |
| `b10aw-g92-seed3` | 0.92 | 0 | 0.0% | 0 | 0.0 | 84.2 @38.9M | 9.4 | 20.9% |
| `b10ax-g92-seed4` | 0.92 | 0 | 0.0% | 0 | 0.0 | 83.1 @34.6M | 12.9 | 15.0% |
| `b10ay-g93-seed1` | 0.93 | 0 | 0.0% | 0 | 0.0 | 87.7 @17.1M | 22.5 | 9.4% |
| `b10az-g93-seed2` | 0.93 | 0 | 0.0% | 0 | 0.0 | 86.9 @48.3M | 16.1 | 18.3% |
| `b10ba-g93-seed3` | 0.93 | 0 | 0.0% | 0 | 0.0 | 88.5 @24.1M | 26.7 | 10.0% |
| `b10bb-g93-seed4` | 0.93 | 0 | 0.0% | 0 | 0.0 | 87.1 @49.7M | 18.7 | 17.8% |
| `b10bc-g94-seed1` | 0.94 | 2 | 0.0% | 0 | 90.8 | 90.2 @26.0M | 37.7 | 3.1% |
| `b10bd-g94-seed2` | 0.94 | 4 | 0.0% | 0 | 93.2 | 89.0 @32.0M | 33.3 | 6.4% |
| `b10be-g94-seed3` | 0.94 | 2 | 0.0% | 0 | 91.6 | 89.8 @46.1M | 40.9 | 3.3% |
| `b10bf-g94-seed4` | 0.94 | 3 | 0.0% | 0 | 91.0 | 90.4 @33.1M | 26.3 | 13.7% |
| `b10bg-g95-seed1` | 0.95 | 30 | 0.0% | 0 | 95.8 | 93.6 @15.2M | 57.1 | 2.5% |
| `b10bh-g95-seed2` | 0.95 | 23 | 0.0% | 0 | 97.8 | 92.7 @43.2M | 49.4 | 3.7% |
| `b10bi-g95-seed3` | 0.95 | 15 | 0.0% | 0 | 95.6 | 91.6 @39.2M | 46.7 | 9.1% |
| `b10bj-g95-seed4` | 0.95 | 27 | 0.0% | 0 | 95.2 | 93.5 @33.9M | 53.0 | 3.0% |
| `b10bk-g96-seed1` | 0.96 | 70 | 4.3% | 0 | 98.4 | 95.0 @21.6M | 63.4 | 3.2% |
| `b10bl-g96-seed2` | 0.96 | 154 | 0.0% | 0 | 97.6 | 95.6 @35.8M | 68.8 | 2.0% |
| `b10bm-g96-seed3` | 0.96 | 41 | 0.0% | 0 | 97.0 | 94.6 @21.4M | 56.3 | 3.5% |
| `b10bn-g96-seed4` | 0.96 | 54 | 0.0% | 0 | 96.4 | 93.4 @29.5M | 67.1 | 1.1% |
| `b10bo-g97-seed1` | 0.97 | 123 | 0.8% | 0 | 98.0 | 94.8 @46.7M | 75.0 | 0.4% |
| `b10bp-g97-seed2` | 0.97 | 294 | 5.1% | 2 | 99.4 | 96.4 @11.9M | 77.0 | 0.4% |
| `b10bq-g97-seed3` | 0.97 | 254 | 2.0% | 0 | 98.4 | 96.2 @48.4M | 79.3 | 0.1% |
| `b10br-g97-seed4` | 0.97 | 85 | 7.1% | 0 | 98.8 | 94.7 @45.2M | 71.0 | 0.2% |
| `b10bs-g98-seed1` | 0.98 | 580 | 6.0% | 1 | 99.2 | 97.1 @28.1M | 89.6 | 0.0% |
| `b10bt-g98-seed2` | 0.98 | 653 | 14.1% | 8 | 99.6 | 97.5 @23.9M | 84.7 | 0.7% |
| `b10bu-g98-seed3` | 0.98 | 664 | 8.1% | 3 | 99.4 | 97.4 @23.7M | 87.3 | 0.0% |
| `b10bv-g98-seed4` | 0.98 | 643 | 13.1% | 8 | 99.4 | 97.2 @37.0M | 84.4 | 0.2% |
| `b10bw-g995-seed1` | 0.995 | 1,234 | 19.0% | 18 | 100.0 | 97.7 @49.4M | 91.7 | 1.0% |
| `b10bx-g995-seed2` | 0.995 | 1,195 | 17.8% | 20 | 99.6 | 97.9 @36.8M | 92.8 | 0.5% |
| `b10by-g995-seed3` | 0.995 | 1,329 | 20.5% | 22 | 99.6 | 98.5 @39.5M | 94.0 | 0.1% |
| `b10bz-g995-seed4` | 0.995 | 1,269 | 20.7% | 25 | 99.6 | 98.3 @45.2M | 91.5 | 1.5% |
| `b10ca-g9975-seed1` | 0.9975 | 1,454 | 24.3% | 49 | 99.8 | 98.6 @33.7M | 91.5 | 1.3% |
| `b10cb-g9975-seed2` | 0.9975 | 1,461 | 29.0% | 44 | 99.8 | 98.7 @40.8M | 92.5 | 0.9% |
| `b10cc-g9975-seed3` | 0.9975 | 1,440 | 20.3% | 26 | 99.6 | 98.4 @40.4M | 93.7 | 1.4% |
| `b10cd-g9975-seed4` | 0.9975 | 1,489 | 28.5% | 45 | 99.8 | 98.7 @36.9M | 92.6 | 1.3% |
| `b10ce-g999-seed1` | 0.999 | 1,494 | 21.7% | 17 | 99.4 | 98.2 @21.8M | 90.7 | 2.5% |
| `b10cf-g999-seed2` | 0.999 | 1,459 | 22.9% | 34 | 99.4 | 98.6 @47.8M | 91.1 | 1.7% |
| `b10cg-g999-seed3` | 0.999 | 1,924 | 39.2% | 133 | 99.8 | 98.7 @48.9M | 91.8 | 1.2% |
| `b10ch-g999-seed4` | 0.999 | 1,737 | 35.4% | 93 | 99.6 | 98.7 @35.4M | 93.3 | 1.4% |
| `b10ci-g100-seed1` | 1.00 | 129 | 36.4% | 15 | 99.4 | 97.3 @49.9M | 18.8 | 68.9% |
| `b10cj-g100-seed2` | 1.00 | 511 | 45.8% | 73 | 99.8 | 98.7 @42.5M | 44.2 | 37.7% |
| `b10ck-g100-seed3` | 1.00 | 142 | 39.4% | 12 | 100.0 | 98.4 @30.7M | 25.6 | 51.0% |
| `b10cl-g100-seed4` | 1.00 | 753 | 33.9% | 78 | 99.8 | 98.3 @34.9M | 68.1 | 16.7% |

## Batch b9 — the GAE λ sweep, 16 values x 4 seeds, closed 2026-09-02

**One knob off b7's winning cell, and the default was not the top.** 64 arms at 50M transitions on the
desktop, eight waves of two λ values, `auto_stage_b` measuring each wave before the next trained,
2026-09-01 ~21:50 → 2026-09-02 16:48. Everything but `ppo_gae_lambda` is b7's reference — `fc (320,)`,
4 epochs, lr 3e-4, γ 0.99, entropy 0.01, clip 0.2, 128x128 rollout, minibatch 256, b2's reward — so
**`b7aa`-`b7ad` are the λ 0.98 row.** Drawdown is the median share of post-competence stage-A evals
(onset = first eval ≥80% perfect) below the threshold, as defined for b8.

| λ | GAE horizon | rows | ≥98%/500 | per-seed share | ≥98.5 | best row | best30 (4 seeds) | sef | drawdown < 50% | < 80% |
|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|
| 0.00 | 1.0 | 8 | 0.0% | 0 0 0 0 | 0 | 94.0 | 88.7 85.1 90.8 88.5 | 24.1 | 11.29% | 68.6% |
| 0.50 | 2.0 | 392 | 0.0% | 0 0 0 0 | 0 | 96.6 | 94.2 94.2 94.9 93.7 | 81.8 | 0.05% | 11.1% |
| 0.80 | 4.8 | 1,133 | 0.8% | 0.4 1.7 0.4 0.7 | 3 | 98.6 | 96.6 95.6 96.1 96.2 | 85.9 | 0.02% | 9.3% |
| 0.85 | 6.3 | 1,279 | 0.8% | 0.3 1.1 1.7 0.0 | 1 | 98.8 | 95.2 96.3 96.1 96.0 | 87.3 | 0.00% | 8.8% |
| 0.90 | 9.2 | 2,130 | 3.9% | 3.7 5.3 5.1 1.1 | 16 | 99.2 | 96.5 97.1 96.9 96.5 | 89.4 | 0.07% | 6.5% |
| 0.91 | 10.1 | 2,168 | 4.3% | 2.3 7.8 3.6 3.7 | 14 | 99.2 | 96.5 97.4 96.9 96.7 | 89.4 | 0.02% | 6.7% |
| 0.92 | 11.2 | 2,363 | 4.8% | 2.0 6.9 3.8 5.7 | 18 | 99.2 | 96.5 97.6 96.8 96.8 | 89.6 | 0.02% | 6.1% |
| 0.93 | 12.6 | 2,545 | 5.4% | 4.1 9.9 3.3 4.2 | 22 | 99.2 | 96.9 97.5 97.1 97.2 | 90.6 | 0.02% | 4.9% |
| 0.94 | 14.4 | 2,359 | 4.6% | 4.5 5.6 2.8 5.2 | 21 | 99.2 | 97.0 96.6 96.7 96.9 | 90.8 | 0.00% | 5.6% |
| 0.95 | 16.8 | 2,665 | 5.3% | 3.7 9.0 4.6 3.4 | 23 | 99.6 | 97.1 97.0 96.9 96.9 | 92.3 | 0.02% | 3.9% |
| 0.96 | 20.2 | 2,882 | 8.3% | 4.6 12.7 8.1 7.0 | 67 | 99.6 | 96.8 97.8 97.2 97.4 | 92.0 | 0.03% | 5.0% |
| 0.97 | 25.2 | 3,613 | 11.7% | 9.0 17.4 9.2 10.8 | 130 | 99.6 | 98.1 98.0 97.6 97.8 | **93.0** | 0.03% | **4.1%** |
| 0.98 (b7) | 33.6 | 4,003 | 17.3% | 18.5 19.0 16.5 15.1 | 174 | 99.6 | 97.8 97.8 97.7 97.7 | 90.9 | 0.29% | 6.2% |
| **0.99** | 50.3 | 5,173 | **27.3%** | 27.2 24.1 31.4 26.0 | 451 | **100.0** | 98.3 98.3 98.4 98.3 | 90.6 | 0.77% | 6.4% |
| 0.995 | 66.9 | 4,998 | 27.3% | 24.8 27.4 34.3 22.3 | 496 | 99.8 | 98.2 98.5 98.8 98.1 | 89.1 | 1.03% | 7.7% |
| 0.999 | 91.0 | 4,897 | 25.6% | 27.8 17.4 24.2 31.3 | 453 | **100.0** | 98.5 98.2 98.1 **99.0** | 87.9 | 1.26% | 8.1% |
| **1.00** | 100 | 5,364 | **29.5%** | 25.2 32.7 29.6 30.8 | **574** | 99.8 | 98.4 98.5 98.4 98.3 | 88.4 | 2.10% | 8.7% |

**Record density climbs monotonically to λ 0.99 and plateaus to 1.00; drawdown climbs with it.**
17.3% at 0.98 → 27.3% at 0.99 is complete seed separation (Mann-Whitney p=0.029, the 4-vs-4 floor),
and the four groups from 0.99 up are indistinguishable at four seeds. Three rows scored 100/500 —
`b9ch-lam999-seed4` at 47,235,072 and 47,316,992 and `b9bw-lam99-seed1` at 48,414,720 — the first in
this project. `sef` peaks at λ 0.97 and falls thereafter, ranking the sweep backwards for the third
time; the stage-A ≥98% share (20.4 → 28.9 → 30.5% at 0.98, 0.99, 1.00) tracks density. Reading in
[`findings.md`](findings.md), charts in [`charts.md`](charts.md). The `hof5000` pass, at the new ≥99/500
cut, is below the arm table.

### Every arm

`rows` is the stage-B checkpoint count, `≥98.5` the count at that 500-episode score, `best30 @step` the peak
30-eval trailing perfect rate and where it landed. `b9ab-lam0-seed2` screened zero checkpoints.

| arm | λ | rows | ≥98%/500 | ≥98.5 | best row | best30 @step | sef | drawdown < 50% |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `b9aa-lam0-seed1` | 0.00 | 1 | 0.0% | 0 | 89.6 | 88.7 @35.7M | 27.1 | 2.0% |
| `b9ab-lam0-seed2` | 0.00 | 0 | 0.0% | 0 | 0.0 | 85.1 @28.9M | 11.4 | 27.9% |
| `b9ac-lam0-seed3` | 0.00 | 5 | 0.0% | 0 | 94.0 | 90.8 @35.5M | 28.1 | 10.9% |
| `b9ad-lam0-seed4` | 0.00 | 2 | 0.0% | 0 | 93.2 | 88.5 @29.0M | 29.9 | 11.7% |
| `b9ae-lam50-seed1` | 0.50 | 101 | 0.0% | 0 | 96.4 | 94.2 @18.5M | 82.7 | 0.1% |
| `b9af-lam50-seed2` | 0.50 | 110 | 0.0% | 0 | 96.6 | 94.2 @14.0M | 78.1 | 0.1% |
| `b9ag-lam50-seed3` | 0.50 | 73 | 0.0% | 0 | 96.4 | 93.7 @38.6M | 83.7 | 0.0% |
| `b9ah-lam50-seed4` | 0.50 | 108 | 0.0% | 0 | 96.6 | 94.9 @27.1M | 82.9 | 0.0% |
| `b9ai-lam80-seed1` | 0.8 | 276 | 0.4% | 0 | 98.0 | 95.7 @9.9M | 87.1 | 0.1% |
| `b9aj-lam80-seed2` | 0.8 | 291 | 1.7% | 2 | 98.6 | 96.6 @34.2M | 83.7 | 0.0% |
| `b9ak-lam80-seed3` | 0.8 | 275 | 0.4% | 0 | 98.0 | 96.6 @47.3M | 86.0 | 0.0% |
| `b9al-lam80-seed4` | 0.8 | 291 | 0.7% | 1 | 98.6 | 95.6 @34.2M | 86.8 | 0.0% |
| `b9am-lam85-seed1` | 0.85 | 305 | 0.3% | 0 | 98.4 | 95.2 @32.2M | 86.0 | 0.0% |
| `b9an-lam85-seed2` | 0.85 | 371 | 1.1% | 0 | 98.4 | 96.2 @7.1M | 88.3 | 0.0% |
| `b9ao-lam85-seed3` | 0.85 | 300 | 1.7% | 1 | 98.8 | 96.3 @22.4M | 85.6 | 0.0% |
| `b9ap-lam85-seed4` | 0.85 | 303 | 0.0% | 0 | 97.4 | 95.9 @38.7M | 89.4 | 0.0% |
| `b9aq-lam90-seed1` | 0.9 | 562 | 3.7% | 1 | 98.6 | 96.6 @11.4M | 89.8 | 0.0% |
| `b9ar-lam90-seed2` | 0.9 | 605 | 5.3% | 7 | 98.8 | 97.1 @26.6M | 89.0 | 0.1% |
| `b9as-lam90-seed3` | 0.9 | 510 | 5.1% | 8 | 99.2 | 96.8 @14.5M | 88.8 | 0.1% |
| `b9at-lam90-seed4` | 0.9 | 453 | 1.1% | 0 | 98.4 | 96.5 @10.5M | 90.0 | 0.0% |
| `b9au-lam91-seed1` | 0.91 | 440 | 2.3% | 1 | 98.6 | 96.5 @13.4M | 89.2 | 0.0% |
| `b9av-lam91-seed2` | 0.91 | 500 | 7.8% | 7 | 99.2 | 97.4 @8.6M | 87.5 | 0.0% |
| `b9aw-lam91-seed3` | 0.91 | 555 | 3.6% | 2 | 98.8 | 96.7 @10.5M | 89.8 | 0.0% |
| `b9ax-lam91-seed4` | 0.91 | 673 | 3.7% | 4 | 99.0 | 96.9 @25.2M | 91.1 | 0.0% |
| `b9ay-lam92-seed1` | 0.92 | 492 | 2.0% | 0 | 98.4 | 96.5 @45.8M | 90.0 | 0.0% |
| `b9az-lam92-seed2` | 0.92 | 670 | 6.9% | 8 | 99.2 | 96.9 @15.0M | 89.3 | 0.0% |
| `b9ba-lam92-seed3` | 0.92 | 574 | 3.8% | 3 | 99.0 | 96.7 @22.9M | 91.2 | 0.0% |
| `b9bb-lam92-seed4` | 0.92 | 627 | 5.7% | 7 | 99.0 | 97.6 @48.5M | 88.0 | 0.0% |
| `b9bc-lam93-seed1` | 0.93 | 641 | 4.1% | 4 | 99.0 | 97.3 @9.6M | 91.3 | 0.1% |
| `b9bd-lam93-seed2` | 0.93 | 659 | 9.9% | 17 | 99.2 | 97.5 @9.4M | 89.7 | 0.0% |
| `b9be-lam93-seed3` | 0.93 | 599 | 3.3% | 1 | 98.6 | 96.9 @15.2M | 91.2 | 0.0% |
| `b9bf-lam93-seed4` | 0.93 | 646 | 4.2% | 0 | 98.4 | 97.0 @14.7M | 90.1 | 0.0% |
| `b9bg-lam94-seed1` | 0.94 | 629 | 4.5% | 4 | 99.2 | 97.0 @11.3M | 92.7 | 0.0% |
| `b9bh-lam94-seed2` | 0.94 | 569 | 5.6% | 7 | 98.8 | 96.6 @19.8M | 89.1 | 0.0% |
| `b9bi-lam94-seed3` | 0.94 | 505 | 2.8% | 1 | 98.6 | 96.7 @24.5M | 91.8 | 0.0% |
| `b9bj-lam94-seed4` | 0.94 | 656 | 5.2% | 9 | 99.2 | 96.9 @21.2M | 89.8 | 0.0% |
| `b9bk-lam95-seed1` | 0.95 | 619 | 3.7% | 2 | 98.8 | 97.1 @31.2M | 93.7 | 0.0% |
| `b9bl-lam95-seed2` | 0.95 | 732 | 9.0% | 15 | 99.6 | 97.0 @14.6M | 89.6 | 0.3% |
| `b9bm-lam95-seed3` | 0.95 | 669 | 4.6% | 6 | 99.2 | 96.9 @13.7M | 92.1 | 0.0% |
| `b9bn-lam95-seed4` | 0.95 | 645 | 3.4% | 0 | 98.4 | 96.9 @11.2M | 93.8 | 0.0% |
| `b9bo-lam96-seed1` | 0.96 | 615 | 4.6% | 5 | 99.6 | 96.8 @12.9M | 91.8 | 0.0% |
| `b9bp-lam96-seed2` | 0.96 | 790 | 12.7% | 29 | 99.2 | 97.8 @12.7M | 91.7 | 0.0% |
| `b9bq-lam96-seed3` | 0.96 | 695 | 8.1% | 17 | 99.2 | 97.2 @17.7M | 92.0 | 0.0% |
| `b9br-lam96-seed4` | 0.96 | 782 | 7.0% | 16 | 99.4 | 97.4 @19.0M | 92.3 | 0.0% |
| `b9bs-lam97-seed1` | 0.97 | 848 | 9.0% | 22 | 99.4 | 98.1 @14.1M | 92.4 | 0.4% |
| `b9bt-lam97-seed2` | 0.97 | 908 | 17.4% | 50 | 99.4 | 98.0 @28.8M | 91.5 | 0.0% |
| `b9bu-lam97-seed3` | 0.97 | 869 | 9.2% | 27 | 99.6 | 97.6 @34.6M | 94.0 | 0.0% |
| `b9bv-lam97-seed4` | 0.97 | 988 | 10.8% | 31 | 99.6 | 97.8 @30.2M | 94.0 | 0.0% |
| `b9bw-lam99-seed1` | 0.99 | 1,342 | 27.2% | 115 | 100.0 | 98.3 @44.8M | 91.4 | 0.5% |
| `b9bx-lam99-seed2` | 0.99 | 1,163 | 24.1% | 80 | 99.2 | 98.3 @43.5M | 91.9 | 0.3% |
| `b9by-lam99-seed3` | 0.99 | 1,339 | 31.4% | 146 | 99.6 | 98.4 @46.3M | 88.8 | 1.0% |
| `b9bz-lam99-seed4` | 0.99 | 1,329 | 26.0% | 110 | 99.6 | 98.3 @47.8M | 90.4 | 1.2% |
| `b9ca-lam995-seed1` | 0.995 | 1,214 | 24.8% | 101 | 99.6 | 98.2 @18.1M | 86.8 | 1.2% |
| `b9cb-lam995-seed2` | 0.995 | 1,227 | 27.4% | 131 | 99.8 | 98.5 @18.6M | 87.5 | 1.3% |
| `b9cc-lam995-seed3` | 0.995 | 1,308 | 34.3% | 175 | 99.6 | 98.8 @40.4M | 89.7 | 0.8% |
| `b9cd-lam995-seed4` | 0.995 | 1,249 | 22.3% | 89 | 99.8 | 98.1 @41.8M | 92.4 | 0.2% |
| `b9ce-lam999-seed1` | 0.999 | 1,261 | 27.8% | 127 | 99.6 | 98.5 @26.2M | 87.5 | 1.5% |
| `b9cf-lam999-seed2` | 0.999 | 1,003 | 17.4% | 48 | 99.4 | 98.2 @41.6M | 88.4 | 1.3% |
| `b9cg-lam999-seed3` | 0.999 | 1,375 | 24.2% | 104 | 99.8 | 98.1 @25.8M | 89.3 | 1.2% |
| `b9ch-lam999-seed4` | 0.999 | 1,258 | 31.3% | 174 | 100.0 | 99.0 @47.5M | 86.5 | 1.2% |
| `b9ci-lam100-seed1` | 1.00 | 1,333 | 25.2% | 101 | 99.6 | 98.4 @49.8M | 88.3 | 3.2% |
| `b9cj-lam100-seed2` | 1.00 | 1,293 | 32.7% | 161 | 99.8 | 98.5 @42.7M | 87.8 | 2.1% |
| `b9ck-lam100-seed3` | 1.00 | 1,453 | 29.6% | 171 | 99.6 | 98.4 @24.8M | 89.3 | 1.3% |
| `b9cl-lam100-seed4` | 1.00 | 1,285 | 30.8% | 141 | 99.6 | 98.3 @33.6M | 88.2 | 2.1% |


### b9 at 30,000 episodes — the desktop `hof30k` pass, 2026-09-03: a new record

Every b9 checkpoint at **≥99 /5,000** — 33 from 9 arms — re-measured at **30,000 episodes on seed 7**,
the seed no selecting pass used, so each row is a confirmed rate. 16 shards on the desktop, 10:21 →
~10:55, exit 0, 990k episodes. **18 of the 33 beat `b5h`'s 98.96, the standing record; the drop from
5,000 was −0.14 pp on average**, against −0.24 for `b5h` and −1.45 for snek2's entries.

| checkpoint | λ | /30,000 | 95% CI | /5,000 |
|---|---:|---:|---|---:|
| **`b9ch-lam999-seed4` @47251456** | 0.999 | **99.30** | [99.2, 99.4] | 99.4 |
| `b9ch-lam999-seed4` @47267840, @47235072, @47218688, @47611904 | 0.999 | 99.20 | [99.1, 99.3] | 99.0-99.3 |
| `b9cl-lam100-seed4` @19316736, @19251200 | 1.00 | 99.10 | [99.0, 99.2] | 99.0-99.2 |
| `b9cc-lam995-seed3` @18350080, @17350656, @17334272 | 0.995 | 99.10 | [99.0, 99.2] | 99.1 |
| `b9ch-lam999-seed4` @47284224, @47349760 | 0.999 | 99.10 | [98.9, 99.2] | 99.0-99.3 |
| 6 more rows | 0.995-1.00 | 99.00 | | 99.0-99.2 |
| 15 more rows | 0.99-1.00 | 98.40-98.90 | | 99.0-99.2 |

**`b9ch-lam999-seed4` @47251456 is promoted** — [`../hallOfFame/HOF.md`](../hallOfFame/HOF.md) — the
first entry above 99% at depth. Its four neighbours at 99.20 make it a region rather than a pixel. Six
arms of the λ ≥ 0.99 plateau hold a checkpoint above the old record; one is promoted, per the rule.


### b9 at 5,000 episodes — the laptop `hof5000` pass, 2026-09-02

Every b9 checkpoint at **≥99%** on its 500-episode close-out (the cut was raised from 98.5 the same day),
re-measured at 5,000 episodes: **727 rows from 36 arms, 3.64M episodes, 76.5 min on the laptop, 8
shards, exit 0.** Merged rows equal the ≥99/500 count on every arm. Groups below λ 0.90 had no
candidate; b7's `fc 320` pass (λ 0.98, 174 rows at the old 98.5 cut) is the reference row:

| λ | rows | mean | ≥98 | ≥98.73 | ≥99 | best | 500 → 5,000 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.90-0.95 | 20 | 97.68 | 35.0% | 1 | 0 | 98.90 | −1.44 |
| 0.96 | 17 | 97.91 | 52.9% | 0 | 0 | 98.60 | −1.21 |
| 0.97 | 36 | 97.92 | 55.6% | 0 | 0 | 98.60 | −1.21 |
| 0.98 (b7, cut 98.5) | 174 | 97.86 | 43.1% | 2 | 0 | 98.90 | −0.94 |
| **0.99** | 138 | 98.21 | **80.4%** | 10 | 3 | 99.10 | −0.93 |
| 0.995 | 153 | 98.24 | 76.5% | 18 | 7 | 99.20 | −0.94 |
| **0.999** | 176 | **98.27** | 72.7% | **31** | **16** | **99.40** | −0.90 |
| 1.00 | 187 | 98.23 | 72.2% | 17 | 7 | 99.20 | −0.93 |
| **pooled** | **727** | **98.20** | **72.5%** | **77** | **33** | **99.40** | **−0.96** |

**The plateau holds at depth, and this is the first batch with a real hall-of-fame candidate.** b7's
whole pass produced 6 rows at or above the snek2 champion's 98.73; b9's four plateau groups produce
**76**, and 33 rows are at ≥99 where b7 and b8 had none. The 500 → 5,000 regression is the usual
−0.9 pp, so the 500-episode numbers were not flattering these arms more than earlier batches.

**`b9ch-lam999-seed4` is the candidate.** Its checkpoints at 47.25M, 47.27M and 47.35M measure
**99.40 [99.1, 99.5]**, 99.30 and 99.30 at 5,000, and the arm's 27 measured neighbours within ±1M
average **98.70** — against 99.20 own and 98.54 basin for `b5h @9027584`, the current record at
98.96 /30,000. The three 100/500 rows came back 99.2, 98.6 and 98.4, which is what a selected high
does. Promotion is [`hof-promote`](../skills/hof-promote/SKILL.md): a fresh 30,000-episode
measurement at seed 7, not yet run.

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
metric [`../plans/archive/ppo.md`](../plans/archive/ppo.md) §10 pre-registered for this comparison. The best *single*
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
[`../plans/archive/ppo.md`](../plans/archive/ppo.md) §7's untuned defaults, on the laptop, 2026-08-29. Kept because it
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
