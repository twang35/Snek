# Findings

What is established, what is falsified, and what is still open. Organized by topic rather than
by when it was discovered.

**Read this before proposing an experiment** — several questions here are closed, and a few were
closed *narrowly* in ways worth checking before reopening them.

| | |
|---|---|
| what is running, what is next | [`runs.md`](runs.md) |
| per-arm numbers and verdicts | [`completedRuns.md`](completedRuns.md) |
| how to measure and judge | [`hyperparamTuning.md`](hyperparamTuning.md) |
| the four degradation patterns | [`failureModes.md`](failureModes.md) |
| superseded findings, batches 1-10 | [`archive/`](archive/) — history only, don't load it |

## Status at a glance

Only current and load-bearing findings. Results about observation vectors this project has
replaced (20, 21, 23, 26 values) and per-batch config results that later batches settled live in
[`archive/findings-superseded.md`](archive/findings-superseded.md).

**Environment and checkpoints**

| finding | status |
|---|---|
| The vector is **30 values**; only batch 11+ checkpoints load on `master` (`450e66e` = 26, `e4514a8` = 20) | **breaking** |
| A same-width observation change loads silently and plays like a beginner — 90.3% → scoring 0 | **standing hazard** |
| Index 29 (food-space) reads 1 in **99.95%** of states, so its weights are barely trained | **hazard**, don't repurpose it |
| ~~Nothing in the vector distinguishes snake lengths 50 to 99~~ | **fixed 2026-08-02** — index 22 is linear board-fill, so 50 and 99 differ |
| The 2026-08-03 observations gave +4 to +5 pp on three metrics, none significant | **open**, n=4, p 0.14-0.24 |

**Records and the horizon**

| finding | status |
|---|---|
| **The record is 97.6%** — `b18b-tgt1000seed2` @1588k, **683/700 fresh episodes** (CI 96.1-98.5) | **measured 2026-08-09**. Beats `b17b`'s 94.24%/5120 by **+3.33 pp, p=0.0002**, intervals **non-overlapping** — the first move in the ceiling that is a different class, not a better sample |
| **A selected high can survive re-measurement** — @1588k was selected at 98/100 and re-measures at 97.4%/500, a **0.6 pp** change | **first instance**, 2026-08-09. Every prior one shrank (99→94.2, 97→93.0, 96→93.5, 96→~94); across nine batch-18 checkpoints >95% the mean shrinkage was **−5.2 pp**, so this is an outlier, not a new norm |
| The record is a **narrow peak, not a region** — @1578k is 10k steps away and reads **91.6%/500** | **standing caveat** — a position-chosen grid is still the only way to claim a region |
| ~~The record is ~95%, `b17b` @1190k at 95.17%/600~~ | **superseded 2026-08-09** by `b18b` @1588k. The `b17b` figure itself was later refined to 94.24% over 5,120 |
| ~~The record is ~96%, `b17b` @1205k reads 99/100~~ | **falsified by re-measurement the same day** — 99/100 → **92.4% over 500**; all four ≥98% rows shrank a mean of **5.05 pp** |
| The previous record was ~93-94% — `b15b` @3245k (93.0% /300), `b14a` @3702k (93.5% /200), `b11b` @855k | **narrowly superseded 2026-08-08** |
| **The frontier is reachable at ~1.2-1.6M steps**, not millions | **measured** — `b17b` @1190k for 94.2% and `b18b` @1588k for 97.6%, against `b15b`'s 3.2M and `b14a`'s 3.7M for ~93.5%. The higher record did **not** cost more steps |
| Two of the four record jumps came from the **horizon** and the **env audit**, not hyperparameters | **established** |
| An arm has a lifetime: peak ~2.5-3M steps, dead by ~7M | **established**, 2 arms to the end |
| Arms peak by ~3.4M | **falsified** — 2 of batch 14's 4 peaked past 3.5M, and 2 of batch 15's were still gaining at **5.5-6.0M**; the old rule tracked where humans stopped arms |
| The horizon was the binding constraint — records live past 2.5M, early arms stopped at ~1.06M | **established** |
| Degradation after 236-312k is systemic across configs | **established**, 5 arms |
| Arms recover from long zero stretches — `b8g` came back from 1.2M steps at zero | **established** |

**Config**

| finding | status |
|---|---|
| **Architecture does not raise the ceiling** — **all 9 shapes** against a seed-matched control at 3M, depths 1-5, **12.7× param range** | **complete 2026-08-12**, batch 20. Peak trailing spans 93.75-94.69 across the whole range and **no shape produced a single full-length row under gate 95**. `FC_LAYERS` is closed as a tuning direction. See below |
| **Capacity binds only *below* the control** — knee between 0.29× and 0.55× | **established** — `25,50,25` at 0.29× is the first shape to move the ceiling, and **down**: peak −0.69, pooled −11.9, **4/4 seeds worse, p 0.125**. `60,30,30,30,30` at 0.55× still holds it (−0.18, p 0.375) |
| A wider or wide-early net raises consolidation (`best-30`, pooled) | **not supported under β→1.0** (batch 20) — the apparent edges are **1 of 4** seeds for `200,50` (p=1.000) and **2-3 of 4** for `200,100,50`/`320` (p ≥ 0.25), carried by the control's weak seeds. Sub-capacity nets also forget ~2× more (drawdown 11-12 vs 5.4). **But reopened under IS-off** — batch 24's `320` reads **+12.2 pooled higher on all 4 seed-matched controls** (b22), cleaner than batch 20's within-batch confound. **Provisional** (n=4, p=0.0625); the HOF-500 confirmed 9 genuine ≥97%/500 checkpoints and a new record (`b24d` @1342k, 98.0%/500), so the gain is real consolidation not /100 inflation — but **4 more `320` seeds** are still owed to move it off the n=4 floor |
| **‡ Two nets of the same size straddle the control by 18.8 pp on pooled** | **established 2026-08-12** — `100,50,50` 46.3% and `320` 65.1% differ by 0.3% in params. The consolidation columns in batch 20 measure seed draw, not architecture; a ~10 pp pooled gap at n=4 is indistinguishable from an iso-capacity relabelling |
| **‡ Batch 20's control seed spread is wider than any between-shape gap it measured** | **established** — control `sef` spans 0.2-26.3%, pooled 33.2-71.3%. At n=4 this design cannot see an architecture effect smaller than that |
| **Removing the food-distance shaping raises how long an arm stays good** | **the first non-null in six batches** — batch 16 `sef` +11.35 pp at a matched 1.25M (p=0.250) and `best_perfect30` +12.58 pp with 4/4 seeds (p=0.125). **Needs replication**; see below |
| `DISCOUNT=0.995` matches the best ceiling and survives 3 of 3 seeds | **measured**, ~2.3x expected value |
| Higher discount is monotonically better | **falsified** — 0.999 died 2 of 2 |
| `0.995` vs `0.9975` on the current environment | **falsified as a difference** — batch 14 null vs 13, `pooled_equal_effort` +0.01 pp, n=4 paired |
| `td_loss` + alpha 0.8 + no IS is effectively alpha 1.6 | **established** — arithmetic, and now **measured**: the log-log slope of Huber against `\|δ\|` is 1.92-1.99 on 8 arms, so alpha 0.6 + `td_loss` is an effective **1.15-1.20** |
| **The two priority signals prioritize the *same* transitions** | **established** — Huber is monotone in `\|δ\|`, so the ranking is identical; top-1000 Jaccard is **1.0000** on 8 of 8 arms. The signal changes how much mass the top gets, never which rows are at the top |
| **‡ IS at β=1.0 cancels prioritization outright, so batches 19-20 were uniform replay past their anneal** | **measured 2026-08-10** — expected update `∝ raw^(α(1−β))`, flat at β=1.0. Realised ESS/N **0.951** against a 0.975 same-effort noise floor, versus **0.213** for batch 18. See below |
| **‡‡ Batch 20 never learned to read observation 15-17, "is it safe to chase the food"** | **measured 2026-08-10** and still correct. Counterfactual ΔQ **+11.70 vs +0.228** (4/4, p=0.125); **6.9x** on the conservative safe-actions-only version; `b20a`'s weight is *negative* |
| ~~Reading observation 15-17 is the mechanism behind batch 18's perfect rate~~ | **‡ demoted 2026-08-11** — the reading rises with steps in every arm and *anti*-correlates with skill inside batch 18 (worst two arms hold the highest ratios); corr with `sef` across 8 arms **+0.04**. `b23b` reads it like batch 20 and scores like batch 18. It marks how much prioritisation survives IS, not skill |
| **‡‡ The elite-vs-mediocre difference is endgame hunting speed, not blunders** | **measured 2026-08-11**, 12 checkpoints × the same 100 games. p90 steps per meal at length 95-99: **5-13 for the records, 86-226 for batch 20's peaks**; `steps_per_food` at 85-94 correlates **−0.967** with perfect rate. Packing, fragmentation and straightness all move with it — one factor, not five |
| **‡‡ A perfect game is 95 consecutive meals, so 99% needs a 5× cut in per-meal error** | **arithmetic on measured rates** — the record checkpoint already plays **1,850 meals per mistake**; 99% needs one per **9,450**. Reframes the objective: the remaining gap is per-meal reliability in the ~5 meals played at length 95+ |
| **‡‡ Free space in one piece at length 90-94 separates the records from a dud by 87 points** | **measured 2026-08-14** — one-piece share **92% / 77% / 5%** for `b24d` / `b18b` / `b20d`, per meal, identical food, exact (one flood fill, no search). The gap opens **ten meals before the end**, and all three reach those lengths equally often. Largest per-policy separation on record here |
| **‡‡ Realised chase-safety is the only marker that still separates the top seven** | **best available lead**, n=7, ~18 tests — pearson **+0.860** (85-94) and **+0.822** (95-99). The *behaviour*, not the Q-sensitivity to obs 15-17 that this file demotes below |
| **‡‡ An arm's best checkpoint is set by its median (r=+0.971) — there is no lucky checkpoint** | **established** on 3,712 full-depth rows. `b10b` measured **624** and never cleared 90%; `b18b` measured 9 and all 9 cleared it. **Screening more checkpoints is not a route to a better policy** |
| **‡ Checkpoints under 20k steps apart are indistinguishable at 100 episodes** | **measured** — mean \|Δperfect\| **5.90 pp** against a **6.48 pp** noise floor. Selecting the max of 20-50 such reads inflates by **5-6 pp**, which fully accounts for the project's documented −5.05 to −5.2 pp shrinkage |
| **‡‡ A drawdown is not how a policy escapes a local minimum** | **falsified 2026-08-11** on `b23b`'s 217-242k collapse plus four batch-18 windows. Endgame value structure, input rankings and churn are all unchanged through it, and the sibling with **no** drawdown gained **more** (+48.6 vs +40.9 pp). A drawdown is a *mid-game* failure: median death length **30** inside it, 96-97 either side |
| **‡‡ The seed decides which arm in a wave wins, and it does not wash out** | **measured 2026-08-11** — seed 2 or 4 is the best arm in **18 of 18** config waves at 550k, mean `sef` gap **+5.41 pp**, exact paired **p=0.00005**; still +8.73 pp at 2M. Comparable to the largest config effect on record. Paired designs difference it out; nothing else does |
| Batch 20's low endgame Q means the terminal reward propagates slowly | **superseded the same day** — it is not lagging, it is **undiscriminating**: ~2-3 for winnable and doomed boards alike, against batch 18's 34-66 vs 18-35 |
| ~~**‡‡ Losses are never trapped positions — the food is reachable until the last 0-2 moves, 75/75**~~ | **retracted 2026-08-14** — `geom` asked only whether a path to the food exists, never whether eating it is survivable. Eating leaves the head **no legal move in 54%** of losses, and the food cell has **no open neighbour in 86%**. The positions are trapped; the test could not see it. See below |
| **‡‡ Starvation is now the modal failure: 55% of losses in both batches, at median length 98** | **measured 2026-08-10**, and **reinterpreted 2026-08-14** — it is not dithering. In **22 of 38** starvations eating the reachable meal would have killed the snake, so there was no safe meal to go and get. **The binding constraint is finishing from length 96-98 inside the starve budget** |
| Removing the food-distance shaping may have bought `sef` and paid in starvations | **untested, motivated** — the modal failure is now failing to go get reachable food, and every arm since batch 16 has the shaping off. Confounded by era; see below |
| `td_error` + `IS_WEIGHTS=0` sits halfway up the concentration ladder | **predicted, untested** — ESS/N 0.454 between batch 18's 0.213 and uniform's 1.0. The one PER cell with a live hypothesis |
| No prioritization setting tested so far survives reliably | **established**, 7 seeds |
| `GRADIENT_CLIPPING=10` on 0.995 helps | **falsified** — 1 of 3 seeds, no ceiling gain |
| n-step returns help | **falsified on speed** — batch 15 at n=3 reached pf30 >= 40% **128k later** than its control, 3 of 4 seeds slower; evals null too (best ckpt +0.05 pp, p=1.000) |
| Forking the collect line at endgame decision points helps | **the batch failed to measure it** — `sef` -1.67 pp and eq-effort -5.02 pp, both entirely from one arm at -30.64; the other **3 of 3 seeds read +3.34 / +3.66 / +3.56 pp on eq-effort**, and the batch holds the project record. Dose was ~60% of design. **Neither established nor falsified**; see below |
| **The replay buffer holds no endgame experience at eps ~0.003** | **falsified** — 20-34% of every current-era buffer is at length >= 80, and 12-81% of collected episodes end perfect. True only of batch 12; see below |
| A larger replay buffer prevents the collapse | **not settled** — opposite results twice |
| Epsilon reaching 0.0 causes the collapse | **falsified**, *but only at 0.001 vs 0.0* |
| **96.8% of batches 10-11's steps ran at epsilon exactly 0.0**, the ladder bottoming out at ~15k | **measured**, 8 arms, 31.1M steps |
| Elevated exploration (handover 0.05) helps | **falsified** — batch 12 deadlocked, 0% perfect 4 of 4 |
| Elevated exploration (handover 0.0125) helps | **falsified** — batch 13 null on **five** metrics, n=4 paired |
| A one-step exploration shield helps | **open**, confounded twice — nothing to fix at 0.0125, and `GUIDED_FRACTION` 0.8 moved with the discount in batch 14 |
| A seed number is a stable unit of quality across configs | **falsified** — batch 11's best seed became batch 13's worst |
| The same `SNEK_SEED` reproduces a run | **falsified** — same seed and config diverge in weights inside 1000 steps; `cpprb`'s sampling RNG is unseeded and unseedable |
| The epsilon *ratchet* was a real defect | **standing**, on mechanism: no recovery from a collapse |

**Measurement**

| finding | status |
|---|---|
| **`fraction of evals >= 80%` has the lowest between-seed variance** of the candidate metrics | **measured**, sd 5.8 vs 8.6 for best-30 |
| Abandoning a checkpoint eval early is not worth it | **falsified for an arithmetic rule** — full-length work falls to 71 / 52 / 31% at gates of 85 / 90 / 95; only the *predictive* version was a mere 14% |
| **n=4 cannot resolve an effect below ~10 pp**; 5 pp needs n≈17-37 depending on the metric | **established** |
| 100-episode measurement reproduces within binomial noise | **established**, 51 repeats |
| The max of N noisy measurements is upward-biased — **re-measure** before quoting it | **established, twice** — 96/100 → 93.5%, 97/100 → 93.0% |
| A high selected reading means a near-perfect policy | **falsified twice** — `b15b`'s 97/100 → 93.0%, `b17b`'s 99/100 → **92.4% over 500**. No exceptions found yet |
| ~~The distribution of an arm's full-length rows tests whether its max is real~~ | **falsified, and it was my argument** — those rows reached full depth *because* they screened well, so their mean is inflated by the same mechanism as the max. `b17b`'s selected rows read **96.2%**; a position-chosen grid over the same region reads **84.06%** |
| **Only a sample chosen by position can describe a region** | **established** — the 12 pp gap above is the selection effect, measured directly on 1,700 fresh episodes |
| The graph-100% tier is comparable across arms and batches | **falsified under a gate** — `EVAL_MIN_ACHIEVABLE` censors it from below; reads +15.6 pp on batch 14 as pure artifact |
| A 100% single graph eval is the only graph value with a usable floor | **measured**, 9 of 9 above 64% |
| A high single 10-episode eval predicts a good checkpoint; smoothing is anti-predictive | **established**, +0.64 vs −0.40 |
| Policy quality changes materially within 1000 training steps | **established**, up to 27 points |
| Checkpoint-to-checkpoint variance is large, and it is not sampling noise | **established** |
| The graph misranks arms badly — `b5c` is 2nd by graph, last by measurement | **established** |
| This domain is very noisy: the same config has produced 62.5 and 18.0 | **established** |

---

## Network shape: the sweep is complete — nine shapes, and architecture never raises the ceiling

`FC_LAYERS` sat at `(50, 100, 50)` from batch 1 to batch 19 with no measurement behind it. Batch 20 is
the first test, and it is now **finished**: nine shapes spanning a **12.7× parameter range and depths
1-5**, each against the same seed-matched control at a matched 3M under β=300k, closed out under gate 95:

| shape | params | vs control | depth | peak trailing | `sef` | best-30 | pooled | drawdown |
|---|---|---|---|---|---|---|---|---|
| `25,50,25` (small) | 3,428 | **0.29×** | 3 | **93.75** | 2.1% | 52.8% | 43.1% | 11.13 |
| `60,30,30,30,30` (deep-narrow) | 6,573 | 0.55× | **5** | 94.25 | 4.7% | 59.3% | 51.6% | 12.31 |
| `100,50,50` (reshuffle) | 10,853 | 0.92× | 3 | 94.16 | 1.9% | 51.2% | 46.3% | 6.35 |
| `320` (depth-1) | 10,883 | 0.92× | **1** | 94.67 | 16.5% | 74.7% | 65.1% | 7.44 |
| `50,100,50` (control) | 11,853 | 1.00× | 3 | 94.44 | 11.2% | 64.0% | 55.0% | 5.42 |
| `93,93` (iso-param depth-2) | 11,907 | 1.00× | 2 | 94.41 | 6.7% | 61.4% | 51.5% | 6.60 |
| `200,50` (wide-early) | 16,403 | 1.38× | 2 | 94.56 | 8.6% | 67.4% | 59.3% | 8.56 |
| `200,100,50` (capacity) | 31,503 | **2.66×** | 3 | 94.69 | 12.8% | 71.4% | 64.6% | 8.31 |
| `100,200,100` (escalation) | 43,703 | **3.69×** | 3 | 94.61 | 16.6% | 71.3% | 63.4% | 7.83 |

**Peak trailing spans 93.75-94.69 across a 12.7× parameter range.** Every shape at or above 0.55× sits
inside the 94.16-94.69 band; the sole shape to leave it does so downward. That is the whole architecture
result in one line.

**Three conclusions, all firm across the sweep:**

1. **Architecture does not raise the ceiling.** At and above the control's capacity — `320` (depth 1),
   `93,93` (depth 2), `200,50` (depth 2), `200,100,50` (2.66×), `100,200,100` (3.69×) — peak trailing stays
   inside the band every batch since 11 has held, and **not one of the nine shapes produced a single
   full-length row under gate 95**. Removing all depth cost nothing; 3.69× capacity bought nothing. Where
   the consolidation columns tick up the paired per-seed differences are seed-driven noise straddling zero
   (p ≥ 0.25, 2-3 of 4 seeds, carried by the control's weak seeds 1 and 3), never a real effect.

2. **Capacity binds only *below* the control, with a knee between 0.29× and 0.55×.** `25,50,25` at 0.29×
   is the first shape in the batch to move the ceiling, and it moves it **down** — peak −0.69, `sef` −9.1,
   pooled −11.9, drawdown +5.7, **all four seeds worse on every column, p at the n=4 floor of 0.125**. It
   is the cleanest directional result batch 20 produced. `60,30,30,30,30` at 0.55× still holds the ceiling
   (peak −0.18, p 0.375), so the knee sits between the two — the net stops being able to reach the control's
   ceiling somewhere under 0.55× the parameters.

3. **‡ The consolidation columns are noise, and the sweep now prices that directly.** `100,50,50` and
   `320` differ in capacity by **0.3%** (10,853 vs 10,883 params) and land on **pooled 46.3% and 65.1%** —
   an 18.8 pp spread that **brackets the control from both sides**. Two nets of the same size disagree by
   more than any shape disagrees with the control, so `sef`/best-30/pooled across this batch are measuring
   seed draw, not architecture. This is the independent confirmation of the per-seed downgrades already
   applied to `320` (+10.1) and `200,100,50` (+9.6), and it is why the batch's verdict rests on peak
   trailing and drawdown. **The corollary is a warning for future batches: a ~10 pp pooled gap at n=4 in
   this design is indistinguishable from an iso-capacity relabelling.**

**Depth costs steadiness below capacity.** Both sub-capacity shapes forget about twice as much as the
control (drawdown 11-12 vs 5.4), and the deep-narrow `60,30,30,30,30` is worst — worse on all four seeds,
+6.9, p 0.125 — the only column that separates it from the control. So the higher-drawdown signature tracks
narrowness/depth below capacity, not capacity alone. Above capacity, drawdown stays in batch-19 territory
(7.4-8.6), far from batch 18's ~57: the base's anti-forgetting property is intact everywhere the net has
enough capacity.

**The transferable lesson is still about the design, not the architecture.** The control's own four seeds
span `sef` **0.2-26.3%** and pooled **33.2-71.3%** — a spread larger than any between-shape gap among the
shapes at or above the control's capacity. An architecture effect there has to exceed the control's seed
variance before n=4 can see it, and none does. The only shapes that cleared that bar are the two that
*under*-provision capacity, and they clear it by getting worse. **Keep `50,100,50`**: nine shapes, none
raised the ceiling, and the smaller nets lowered it. **`FC_LAYERS` is closed as a tuning direction** —
the constraint is elsewhere, which is what the β ladder (batches 21-23) went after next and where it
found real movement in consolidation.

**Reopened for consolidation under IS-off (batch 24).** This whole sweep ran under the β→1.0 control
(β=300k anneal), the weakest base on the ladder. Batch 24 re-ran the `320` shape under IS-off — the
strongest base — and it reads **pooled 87.9, +12.2 over the b22 control and higher on all four
seed-matched seeds** (p=0.0625). The **ceiling conclusion is untouched** (peak 95.00, unmoved), but "the
consolidation columns are pure seed noise" and "`FC_LAYERS` is closed" were established under β→1.0 and do
**not** carry to IS-off unchanged: width and prioritisation appear to interact, so width pays only when the
gradient is prioritised. This stays **provisional** — n=4 at the sign-test floor — until 4 more `320` seeds
move it off. The HOF-500 has settled the *peak* question, though: 9 of the batch's 199 ≥97%/100 checkpoints
held ≥97%/500, and `b24d` @1342k took the record at **98.0%/500** — so the consolidation is real, deeply
measured, not /100 selection inflation.
Full result: [`completedRuns.md`](completedRuns.md#batch-24--fc-width-320-under-is-off-the-first-architecture-result-and-a-new-record).

## The food-distance shaping was a drag on consistency — the first signal in six batches

**Batch 16 removed `FOOD_DISTANCE_REWARD` (0.001 subtracted on every ordinary move that increases the
distance to food) and beat its control on every level metric.** Against batch 14 — the same four seeds
and the same config in every other respect — at a matched 1.25M horizon:

| metric | shaping on | shaping off | delta | p |
|---|---|---|---|---|
| `strong_eval_fraction` | 5.80% | **17.15%** | **+11.35 pp** | 0.250 |
| `best_perfect30` | 66.83% | **79.42%** | **+12.58 pp** | **0.125** (4/4 seeds) |
| mean perfect, back half | 51.15% | 62.27% | +11.13 pp | 0.250 |
| steps to pf30 ≥ 40% | 429k | 424k | -6k | 0.875 |
| peak trailing | 94.31 | 94.71 | +0.41 | 0.250 |

**The mechanism is consolidation, and the two nulls are what pin it down.** Speed is unchanged (the
pf30 crossing, and the epsilon handover at 11.5k against 12.5k) and the ceiling is unchanged (peak
trailing, which batches 11-16 hold inside 0.3 points). What roughly **tripled** is the share of evals
at ≥80% perfect. So an arm without the shaping learns to win at the same moment and then *stays*
winning, where a shaping-on arm oscillates back down.

That is the predicted failure of a hand-designed reward: 0.001 per retreating move is a small
permanent tax on exactly the detours a 93% endgame requires. It never stopped an arm learning to win —
it stopped it holding the win.

**‡ Not a horizon artifact.** 1.25M is where batch 16 stopped, so the obvious objection is a
cherry-picked slice. The `sef` delta is **-0.25 pp at 400k, +1.37 at 600k, +3.68 at 800k, +8.17 at
1.0M, +11.35 at 1.25M** — absent early and growing monotonically, which is arms consolidating rather
than a lucky window.

**Why this is a lead and not established.** One batch at n=4, so p bottoms out at 0.125 even with every
seed agreeing. **The arms were stopped at 1.25M**, so nothing is known past that horizon — batch 14's
own curve was still climbing there, and its full-run `sef` (21.53%) exceeds batch 16's 17.15% at 1.25M.
A replication at a longer horizon is what would settle it. In the meantime the shaping stays **off**,
since nothing here argues for restoring it, and batch 17 onward runs with `FOOD_DISTANCE_REWARD=0`.

**Also corrected by this batch:** an interim read at 500k called it a sixth null, on the grounds that
the pf30 crossing was flat (424k vs 429k) and the ceiling had not moved. Both facts were right and the
conclusion was wrong — they are the two metrics this effect does *not* touch. **"Read crossings early,
read levels late" cuts both ways: an early crossing read is trustworthy and says nothing about level.**

## Forked endgame collection: null at 60% of the intended dose, and the premise it was built on is false

**Batch 17 forked the collect trajectory at endgame decision points** — at snake length ≥ 85, with
more than one non-fatal action available, it snapshotted the game and played the untaken action out on
a branch for up to 60 steps, up to 3 branches alongside the main line, all feeding the replay buffer.
Against its seed-matched batch-16 control at a matched 1.245M: **`sef` -1.67 pp, p=0.875**, every level
metric slightly negative, speed 21.8k better at p=0.750. Full numbers in
[`completedRuns.md`](completedRuns.md#batch-17--forked-endgame-collection-a-null-that-produced-the-project-record).

**‡ The premise it was proposed on is false, and this is the more reusable finding.** The idea was
that the endgame is never explored at epsilon ~0.003, so the buffer holds no endgame experience.
Measured across all 20 current-era arms' saved replay buffers:

| arm group | buffer at len ≥ 80 | at len ≥ 90 | collected episodes ending perfect |
|---|---|---|---|
| batches 11, 13, 14, 15, 16 (eps ~0.003) | **20-34%** | 9-21% | **12-81%** |
| batch 12 (eps 0.05, the deadlock) | **0.0%** | 0.0% | **0 of 3142** |

Batch 12 is the calibration: the metric reads exactly zero when the endgame really is missing, so
20-34% is not a floor artifact. **The old claim that "the buffer holds no trajectories that eat the
last ~10 food" is a true description of batch 12 and false of every arm since.** Any future proposal
that starts "the agent never sees the endgame" has to clear this table first.

**What survived the correction, and what the batch actually tested.** At an endgame decision point the
buffer holds the consequence of the action taken and **never** that of the alternative — mean **2.06
safe actions** per eligible state at length ≥ 85, so ~1.06 per state are never tried. An arm that dies
from state `s` learns `Q(s, a_bad)` is low, but nothing raises `Q(s, a_good)` for the action it did not
take, so the argmax has no reason to flip. That is **counterfactual coverage**, and naming it that way
is what made the null closable. Branch points are also **not rare**, contrary to the original design:
≥ 2 safe actions on **42-45%** of steps at length 80-84, falling to **9-11%** at 95-100 — about
**74-104 eligible states per episode** at length ≥ 85. "Only a few points" holds above ~95, not from 85.

**Why this is not a falsification.** The delivered dose was **24-29% branch share against a predicted
~46%**, with ~30% of eligible fork points skipped because the 4-branch cap was full — so the cap bound,
not the fork probability. And one seed carries the whole result: dropping `b17a` turns `sef` from
-1.67 to **+4.23 pp**. The honest reading is "no effect at ~60% of the intended dose, measured at a
sample size that cannot resolve one bad seed." `SNEK_FORK_BRANCHES=6-8` is the experiment that would
settle it.

**‡ The close-out points the other way, and consistently.** On `pooled_equal_effort` the three
non-outlier seeds read **+3.34 / +3.66 / +3.56 pp** — a 0.32 pp spread on a metric with several points
of between-seed sd, which is the most consistent signal any batch has produced — while `b17a` reads
**-30.64** and drags the mean to -5.02 (p=1.000). **And the batch produced the project record**:
`b17b-forkseed2` @1205k at 99/100 with a 96.2% region, at 1.2M steps. None of that makes forking
established; all of it makes "forking is null" the wrong summary. **The accurate summary is that the
batch failed to measure its own effect**, because at n=4 one arm at -30 pp is larger than the effect
being looked for.

**‡ Also a correction to how the interim read was made.** A 900k read called it a null and was right,
but a 500-700k read would have called it a **win** (`sef` +2.30 to +3.67 pp) — the sweep rises to
~700k and then reverses. Batch 16's signal grew monotonically to its horizon; this one did not.
**A non-monotone sweep is the tell for a noise effect**, and it is worth reading the shape rather than
any single truncation.

## The discount: an optimum near 0.995-0.9975, and now a closed question

The one hyperparameter that has reliably helped. Not monotone — the sweep has a peak and falls off
hard on both sides:

| discount | eff horizon | outcome | verdict |
|---|---|---|---|
| 0.99 | ~100 | 12.0% measured, dies 2 of 4 seeds | too short |
| **0.995** | ~200 | 38.8% measured, **survived 3 of 3**; the current default | best expected value |
| **0.9975** | ~400 | held the record twice (92%, then 69.3% best-30), but 1 of 2 on survival | best ceiling |
| 0.999 | ~1000 | **dead 2 of 2**, at 452k and 398k, peak trailing 63.1 / 31.8 | too long |

**`0.995`'s gain is reliability, not ceiling.** Priced for survival it is ~2.3x the best previous
config: 28.2% mean level at 3 of 3 surviving, against `b4c`'s 37.1% at 1 of 3 (expected value 12.4%).
The ceiling claim made for it originally would have been wrong — `0.9975` beats it there.

**`0.995` vs `0.9975` is settled on the current vector, and the answer is "no difference".**
Batch 14 ran 0.9975 against batch 13's 0.995 at n=4 paired and came back null on every metric that
survives its abandonment gate — `pooled_equal_effort` **72.08% against 72.07%**, best checkpoint
+2.75 pp at p=1.000, `best_perfect30` +1.08 pp at p=0.625. Write-up:
[`completedRuns.md`](archive/batches12-15.md#batch-14--disc-09975-at-guided-08-and-the-widest-seed-spread-yet).

That closes the question batch 9 left open at n=2, and it closes it against a specific hypothesis
worth recording as dead: **the 2026-08-03 endgame observations do not need a longer horizon to be
usable.** The argument for re-asking was that following-tail (26-28), food-space (29) and
reachable-tail (9-14) all describe structure 300+ steps out, while 0.995's effective horizon is ~200
against a perfect game's ~1780. Measured, it makes no difference. **Do not re-ask at n=4.**

Batch 9's reason for staying open still describes the shape of the problem: the two values won
*different* things there (0.995 better expected value, one 0.9975 seed dead at 328k; 0.9975 the
steadier single arm), and **the seed spread exceeded the effect** — batch 9's two 0.995 seeds were 18
points apart on best checkpoint. Batch 14 is the same story with more arms: -16.2 to +24.8 pp per
seed on the primary around a +2.05 pp mean.

**Stop sweeping the discount.** Above 0.9975 is measured dead, 0.995 and 0.9975 are measured
indistinguishable, and an interior point like 0.996 cannot plausibly differ from either by more than
the ~10 pp n=4 can resolve. This is now a closed question, not a narrower one.

Two process notes that came out of batch 9 and still apply:

- **A partial close-out is not a small version of a complete one.** `b9d` at 12 of 17 checkpoints
  reported numbers that moved materially once the rest landed.
- **The best checkpoint and the trailing peak are in different places** — `b9a`'s best is at 1735k
  while its trailing peaked at 3277k. Confirmed again across batches 10-11, where the gap ran to 3M
  steps on `b10b`. Do not use a graph peak to decide where to look for a checkpoint.

Per-batch tables: [`archive/findings-superseded.md`](archive/findings-superseded.md) and
[`archive/batches1-11.md`](archive/batches1-11.md).

## Graph evals are a filter, not a ranker

A graph point is 10 episodes, so `perfect_percent` only takes values 0, 10, … 100. What that
signal can and cannot do, from 88 checkpoints measured on 2026-07-30 — the largest sample here:

| question | answer | evidence |
|---|---|---|
| Does a high single eval beat a smoothed one for *selecting*? | Yes, decisively | +0.64 vs **−0.40** correlation; outlier picks measured 41.3% vs 27.1%, CIs disjoint |
| Among already-high checkpoints, does the graph value rank them? | No | +0.10; 90% and 80% points are indistinguishable (57.9% vs 58.6%) and both span ~50 points |
| Does the *surrounding* rate rank them? | Yes | **+0.48** |
| Is any graph value trustworthy on its own? | Only 100% | 9 of 9 measured at ≥64%, mean 72.5%; 90%/80% reach into the 20s |

**The −0.40 and the +0.48 are not a contradiction**, and it took a while to see why: the first
compares *selecting* on smoothed vs raw across a wide range, the second asks whether — among
checkpoints that already spiked — the region rate picks the best. Both are true. Range
restriction attenuates every correlation in the second column.

So: **measure 100% points first** (rare and reliably top-decile), measure the whole ≥90% tier
because there is no way to tell the 88% from the 33% in advance, and use the surrounding rate as
the tiebreak. That is what `eval_checkpoints.py` implements.

**A 100% graph point is not a shortlist of champions**, though — 6 of the 8 arms measured across
batches 10-11 found their best checkpoint in the 90% tier, which is ~4x larger. See
[`hyperparamTuning.md`](hyperparamTuning.md).

**`b6b-alpha06` and `b6a-alpha04` were measured with the old smoothed selector and are
underestimates.** They cannot be fixed by re-measuring — `b6a`'s best graph point in 1415 evals is
50%, so the current thresholds yield nothing from it. The alpha comparison needs new seeds.

## Three measurement caveats

**‡ An abandonment gate silently invalidates every pooled figure except `pooled_equal_effort`.**
Learned on batch 14, the first batch measured under `EVAL_MIN_ACHIEVABLE=90`. The gate stops a
checkpoint once its ceiling drops below 90%, which means every surviving full-length row is a
*winner* — so any statistic pooled over "the rows that reached full length" is censored from below.
Two casualties:

| statistic | what the gate does to it |
|---|---|
| **graph-100% tier** | reads 90.3% on batch 14 against batch 13's 74.6%, a +15.6 pp artifact. Tier sizes fell from 31-114 checkpoints per arm to 1-28. **Unusable.** |
| **winner's-curse shrinkage** | was fitted on each arm's *unselected* graph-100% rows; under a gate those rows are abandoned and biased low by optional stopping. **Not computable.** |
| `pooled_equal_effort` | **exact at any gate** — it truncates every checkpoint to the 20-episode screen depth, and abandonment cannot fire before the floor |

The reason this was easy to miss is that the gate's safety argument is about *rankings* — an
abandoned row can never outrank a kept one, which is true — and says nothing about *pooling*. Check
`min_achievable` in a payload before pooling anything from it, and never compare a gated figure to an
ungated one. The substitute for shrinkage is a second independent 100-episode run on the champion,
which is stronger evidence anyway: `b14a`'s 96/100 re-measured **91/100**, pooling to 93.5%.

**Pooled rates only compare when the selection rule matches.** The rule has changed several times
and the checkpoint count now varies per arm (1 to 660), so pooling over 16 checkpoints and over 1
are not the same statistic. **Use best checkpoint for cross-arm comparison**, and read pooled as a
within-arm consistency check: a config whose best and pooled figures are close is producing a
strong *region*, which is what the project is actually chasing.

**A single 100-episode figure is usable at ±10.** 51 checkpoints were measured twice on the same
day: mean spread **4.8** points, 47 of 51 within ±10, no systematic direction — comfortably inside
binomial expectation. An earlier warning here, built on `b4c` @869000 reading 51 / 42 / 32 across
three runs, should be read as "one checkpoint once behaved strangely" rather than a property of
the instrument. Pooling over many checkpoints is still what shrinks the interval (±1.3 at 6300
episodes).

## Policy quality changes materially within 1000 training steps

Evaluating each high-single-eval checkpoint **together with the checkpoints immediately
either side of it** — 100 episodes each — settles whether "this checkpoint is good" can be
distinguished from "this part of the run is good". It can:

| cluster | centre | neighbours at +/-1000 | centre advantage |
|---|---|---|---|
| 851000 (`b4c`) | **40.0%** | 28.5% | **+11.5 points** |
| 869000 (`b4c`) | **32.0%** | 23.0% | **+9.0 points** |
| 970000 (`b4c`) | **35.0%** | 7.5% | **+27.5 points** |
| 2806000 (`b8f`) | **80.0%** | 74.0% | **+6.0 points** |

Pooled over the first three, centres measure 35.7% (CI 30.5-41.2) against neighbours' 19.7%
(CI 16.7-23.0) — non-overlapping, and the effect is in the same direction in **4 of 4** clusters.

The `b8f` cluster is the weakest confirmation and the most informative one. Its graph values read
80% / **100%** / 70% and measured 74% / **80%** / 74%, so the centre still won — but by 6 points
with overlapping intervals, on an arm where *every* checkpoint in the region is strong. **The
advantage shrinks as the surrounding region improves**, which is what you would expect if the
spike reflects a genuinely better policy rather than a measurement artefact: there is less room
above a 74% neighbourhood than above a 7.5% one.

The 970000 cluster is the extreme case: **969000 measures 8%, 970000 measures 35%, 971000
measures 7%.** Those are 100-episode measurements, so **1000 training steps can gain or
lose 27 points of perfect-game rate.** Training is far more non-stationary at the
checkpoint level than this investigation previously assumed, and adjacent checkpoints are
not interchangeable samples of one policy.

## Checkpoint-to-checkpoint variance is large, and it is not sampling noise

Within `b6b`'s 1455-1464k cluster — 9000 train steps end to end, checkpoints that should
be nearly identical policies — measured rates at 100 episodes each:

| ckpt | 1455k | 1456k | 1461k | 1462k | 1463k | 1464k |
|---|---|---|---|---|---|---|
| perfect % | **36** | 25 | 24 | **16** | 24 | 31 |

**A 20-point spread across 9000 steps.** At 100 episodes each these are real differences,
not sampling error. Consequences:

- **One checkpoint does not characterise a policy region.** Evaluating a single checkpoint
  from this cluster would have yielded anywhere from 16% to 36% depending on the draw.
- **Pool across several checkpoints** for any number that gets compared across arms. This
  is why `top20` deliberately allows adjacent picks: spacing them out hides exactly this.
- The published **51% for `b4c` at 869000 is one checkpoint**, so it is the top of a
  distribution like this one, not the config's level. `b4c`'s pooled 31.8% is the fairer
  figure.
## The mechanism: `td_loss` doubles the effective priority exponent

`common.element_wise_huber_loss` uses delta 1.0, so for `|td_error| < 1` — most transitions once
a policy is decent — `td_loss = 0.5 * td_error^2`. Priorities are then raised to alpha, so
squaring inside and exponentiating outside **compounds**: `PRIORITY_SIGNAL=td_loss` with alpha
0.8 is really an exponent of ~1.6.

**So alpha 0.8 was never the config under test**, and every `td_loss` arm is incomparable to its
alpha label. The three "PER changes" recovered from `theSchlong` were also never independent:
`td_loss` and alpha 0.8 multiply into an extreme exponent, and dropping IS weights removes the
only thing correcting the resulting bias.

The current config is alpha 0.6 + `td_loss` = **effective ~1.2**.

**Sharpness is a variance dial, not a quality dial.** Among surviving arms the ceiling rises
monotonically with the effective exponent (~1.6 → 34%, ~1.2 → 21.7%, ~0.8 → 14.3%), and so does
the death risk (2 of 3 arms at ~1.6 died outright). The two best arms were also the two with the
most near-death excursions.

Two caveats that weakened the original version of this finding:

- **Lower sharpness may only delay death.** Seeding eff ~1.2 four times gave 2 deaths of 4, at
  573k and 1162k, against eff ~1.6's 2 of 3 at 246k and 272k. At these sample sizes 50% and 33%
  are not different, so "safer" has no support — but "later" does. **Any survival rate quoted
  here needs a fixed step horizon attached**, or it is partly an artefact of run length.
- **The "stability cliff" between eff 0.8 and 1.2 is retracted** — `b6b` crossed it and thrived.

Full per-arm tables and the batch-by-batch derivation:
[`archive/findings-superseded.md`](archive/findings-superseded.md).

## ‡ Measured: batches 19-20 compared aggressive PER against *uniform replay*

Measured 2026-08-10 from the saved replay buffers and final checkpoints of batch 18 and batch 20
wave 1 — eight arms, all `(50, 100, 50)`, all alpha 0.6, differing only in the two PER knobs. The
script is
[`perDiagnostics/per_priorities.py`](perDiagnostics/per_priorities.py); the chart is
[`charts/per-b18-vs-b20-priorities.png`](charts/per-b18-vs-b20-priorities.png).

The question was whether `td_loss` priorities put *different states* in the buffer's top than
`td_error` does. **They cannot, and the real difference is elsewhere.**

| claim | verdict |
|---|---|
| The signals rank different transitions | **false, by construction** — Huber is monotone in `\|δ\|`, top-1000 Jaccard **1.0000** on 8/8 arms |
| The signals concentrate the update differently | **true, and large** — realised ESS/N **0.213** vs **0.951** |
| Batch 19/20's IS correction left prioritization partly intact | **false past the anneal** — β=1.0 is uniform in expectation |
| The buffers held different states | **only mildly**, and downstream of policy quality |
| Batch 18's value function is better fit | **false — it is worse fit and shaped differently**, 4/4 seeds |

### The effect being explained is real, and it is the largest config effect on record

`sef` is a share of each arm's own evals, so the two batches have to be truncated to a common
horizon — batch 20 ran 400-600k steps longer. At **2.401M**, exact paired permutation over 16 sign
flips:

| metric | b18 (`td_loss`, no IS) | b20 (`td_error`, IS) | delta | p |
|---|---|---|---|---|
| **`strong_eval_fraction`** | **33.84%** | 12.45% | **+21.39 pp** | **0.125** (4/4) |
| `best_perfect30` | 87.25% | 64.08% | **+23.17 pp** | **0.125** (4/4) |
| peak trailing | 94.94 | 94.41 | +0.52 | **0.125** (4/4) |
| max drawdown | 55.52 | 5.41 | +50.11 | **0.125** (4/4) |

0.125 is the floor at n=4. This reproduces the batch 18 vs 19 table almost exactly (`sef` −17.78
there, −21.39 here) on a **different** control batch, so the two PER knobs now have eight seeds
behind them rather than four.

### What the signal actually changes: mass, not membership

`element_wise_huber_loss` is strictly increasing in `|td_error|`, so both signals induce the
**identical ordering** — verified as a top-1000 Jaccard of exactly 1.0000 on every arm. The
log-log slope of Huber against `|δ|` measures **1.92-1.99**, confirming the effective-exponent
arithmetic empirically for the first time: alpha 0.6 + `td_loss` is an effective **1.15-1.20**.

What differs is the share of the update the top rows receive. The quantity that matters is
sampling probability **times** the IS weight, because that product is what reaches the gradient:
`p ∝ raw^α` and cpprb's mean-normalised weights give `w ∝ p^-β`, so

    exposure  ∝  raw^(α(1 − β))

**At β=1.0 the exponent is zero and prioritization cancels exactly.** Not "weakens" — cancels, in
expectation. Realised exposure over 768,000 actual cpprb draws, against a same-effort uniform
noise floor:

| config | top 1% of the update | ESS/N |
|---|---|---|
| uniform noise floor (flat priorities) | 1.46% | 0.975 |
| **b18**: `td_loss`, no IS | **14.98%** | **0.213** |
| `td_error`, no IS — **never run** | 8.67% | 0.454 |
| b19/b20 early: `td_error`, IS β=0.4 | 3.94% | 0.764 |
| **b19/b20 past the anneal**: `td_error`, IS β=1.0 | 1.83% | **0.951** |

Batch 20 reached β=1.0 at **300k** steps and batch 19 at **1M**, and every arm in both peaked after
its own anneal completed — batch 19 at 1299-1932k, batch 20 at 332-2493k, with `b20d`'s 332k the only
close call. **So neither batch tested "standard PER" against "aggressive PER" — past the anneal they
tested uniform replay against an effective-alpha-1.2 prioritized buffer**, and batch 19's
"standard PER falsified" is better read as *uniform replay is worse here, 8 seeds*. Batch 18's
update behaves as though the buffer were **21%** of its size; batch 20's uses all of it evenly.

One residue the algebra misses: `normalize_is_weights` divides by the **batch** mean rather than a
global constant, so cancellation is per-batch and imperfect. `td_loss` + IS at β=1.0 reads ESS/N
0.868, clearly above the floor, and the gap grows with priority skew. It does not change the
reading above, where `td_error` at β=1.0 sits within noise of uniform.

### Where the concentrated mass goes

The top of the buffer is **the last few moves of a nearly-finished game**. Top-100 by priority has
mean snake length **94.6-96.3** against a buffer mean of 66.5-70.1, and the largest `|δ|` rows are
almost all "ate food at length 97-98" — one or two foods from the 100-point payoff. Share of the
expected update by category, on batch 18's buffers:

| category | in buffer | b18 signal | `td_error` α=.6 | β=1.0 | mean `\|δ\|` |
|---|---|---|---|---|---|
| ate food | 4.48% | **11.22%** | 8.15% | 4.48% | 1.645 |
| ordinary, len ≥ 80 | 46.75% | **56.59%** | 52.13% | 46.75% | 0.730 |
| ordinary, len < 80 | 48.51% | **31.69%** | 39.33% | 48.51% | 0.360 |
| died (wall/body) | 0.194% | 0.405% | 0.313% | 0.194% | 1.403 |
| won the game | 0.055% | **0.046%** | 0.051% | 0.055% | 0.410 |
| starved | 0.005% | 0.053% | 0.029% | 0.005% | 7.398 |

**Prioritization does not chase wins — it deprioritizes them.** A won game is the one outcome the
network predicts *well* (mean `|δ|` 0.410, below the 0.573 arm mean), so it gets slightly less than
its buffer share. What the sharper signal buys is food transitions at 2.5x and late-game ordinary
moves at 1.2x, paid for by early-game moves dropping to 0.65x. Starvation is the most mispredicted
event in the buffer by a wide margin and is far too rare to matter.

### The buffers are similar; the value functions are opposites

Buffer composition differs about as much as two policies of different quality would predict, and
no more — batch 18's last 100k transitions hold **3.4x** the wins (0.055% vs 0.016%) and **half**
the deaths (0.194% vs 0.404%). Endgame share is comparable (46.8% vs 50.8% at length ≥ 80), so the
"buffer holds no endgame experience" idea stays falsified for both.

The networks are the opposite of similar. Batch 18's arms show **4.4x** the mean `|td_error|`
(0.52-0.64 vs 0.12-0.17), and **crossing the arms settles that this lives in the network rather than
the data**: run each seed's two checkpoints over both seeds' buffers and every network keeps its own
level, 4/4 seeds. Each fits its own buffer slightly better, as expected, but the batch gap survives
the swap intact. Some of the gap is scale — batch 18's Q values are ~3x larger — and normalising by
mean max-Q leaves batch 18 still worse fit, 0.0173 vs 0.0124, 4/4.

**The shape is the finding, and it is scale-free.** Mean max-Q against snake length:

| network | len 10 | len 50 | len 85 | len 95 | len 97 |
|---|---|---|---|---|---|
| `b18a` | 29.15 | 37.34 | 39.63 | 42.71 | **43.32** |
| `b18c` | 19.98 | 35.43 | 39.84 | 42.71 | **38.95** |
| `b18d` | 20.25 | 34.27 | 43.20 | 43.09 | **48.55** |
| `b18b` | 26.72 | 30.40 | 26.44 | 22.37 | 21.72 |
| `b20a` | 28.90 | 16.65 | 6.33 | 3.64 | **3.09** |
| `b20b` | 30.76 | 18.58 | 8.92 | 5.09 | **4.16** |
| `b20c` | 29.42 | 16.76 | 6.56 | 4.02 | **3.09** |
| `b20d` | 30.33 | 18.45 | 8.04 | 4.76 | **3.92** |

Three of four batch-18 arms **rise** with length — the value function says a longer snake is closer
to the payoff, which can only be true if the terminal 100 is being counted, because the remaining
*food* is worth less at length 90 than at length 10. All four batch-20 arms instead **decline
steadily from length ~10 to 98**, by **6.04-7.93x** between lengths 10 and 95, reaching 0.12, −0.04,
0.10 and 0.20 at length 98. The separation is total: at length 95 batch 18 spans 22.4-43.1 and batch
20 spans 3.6-5.1, with no overlap in either direction. Both then dip at 98 and spike at 99-100, so
the terminal reward is represented in the states that *collect* it and the difference is how far back
it has propagated.

**‡ But batch 20's low endgame values are not obviously an error — they may be correct pessimism**, and
this is the reading to prefer. The grid holds 100 cells and the snake starts at 5, so length 98 means
93 food eaten with two to go, and `b20a`'s final eval averages **93.2** food at a 20% perfect rate.
Length 98 is *literally where these arms stop*. A value of 0.12 for arriving there is close to right
for a policy that is about to fail, and batch 18's 12-29 is close to right for one that often
finishes. So the profile is a faithful readout of each policy's own endgame competence rather than an
independent defect.

**What survives that, and makes it more than a restatement of the score:** batch 20 is not failing to
*reach* the endgame. It spends **more** time at length ≥ 98 than batch 18 does — 3.34%, 5.31%, 6.25%,
8.17% against 3.64%, 3.91%, 3.73%, 3.11% — so 3 of 4 batch-20 arms have more endgame experience in
the buffer than any batch-18 arm, and still value it at zero. So **endgame coverage is not the
mechanism** — the transitions are there and are being sampled in proportion.

What the update concentration plausibly buys instead is gradient on the *rare* endgame states, and the
section below identifies which ones: the 5-7% where observation 15-17 fires. Under β=1.0 those get
exactly their population share, which turns out not to be enough to train the weight that reads them.
This paragraph originally proposed "propagation speed through the endgame"; that reading is
**superseded** by the counterfactual measurement below, which finds the value function undiscriminating
rather than lagging.

`b18b` is the honest exception — its profile declines like batch 20's, though only by 1.19x and from
a level three times higher, and it is the batch-18 arm with the worst drawdown (85.08).

### ‡ The gap is decided before the last move, not at it

**Retracted 2026-08-10, within a day of being written: this section first claimed "the whole gap is
one move wide."** That was wrong, and the measurement that refutes it is in the same buffers.

Transitions at snake length **99** are the last move of the game: the board is one cell short, so the
action either wins 100 or dies. They are directly countable, and the conversion rates separate the
batches completely.

| arm | attempts | wins | deaths | converted |
|---|---|---|---|---|
| `b18d` | 83 | 56 | 27 | **67.5%** |
| `b18a` | 89 | 59 | 30 | **66.3%** |
| `b18c` | 104 | 65 | 39 | **62.5%** |
| `b18b` | 74 | 41 | 33 | **55.4%** |
| `b20b` | 55 | 22 | 33 | 40.0% |
| `b20c` | 79 | 22 | 57 | 27.8% |
| `b20a` | 45 | 7 | 38 | 15.6% |
| `b20d` | 106 | 14 | 92 | 13.2% |
| **pooled** | | | | **63.1%** (221/350) vs **22.8%** (65/285) |

**4/4 with no overlap**, and the attempt counts are comparable — 350 against 285 — so batch 20 arrives
at the final move nearly as often and converts it a third as well. `b20d` gets there the most times of
any arm in either batch, 106, and converts 14.

**But splitting those attempts by whether a winning move was even legal shows there is no decision
left to make.** Observation indices 18-20 flag, per action, "does this move win the game", and they
fire only when the snake is exactly one food short:

| | states with a winning move | won | states with none | won |
|---|---|---|---|---|
| batch 18, all four arms | 221 | **221** | 132 | **0** |
| batch 20, all four arms | 65 | **65** | 224 | **0** |

**286 of 286 winnable positions were won and 285 of 285 unwinnable ones were lost, in both batches.**
Play at length 99 is already perfect and identical; the conversion rate is not measuring last-move
skill at all, it is measuring **whether the arm arrived in a position that was still winnable.** By
length 99 the game is decided.

So the failure is upstream, and the observations locate roughly where. Tail reachability — at least one
move that is safe and keeps the tail reachable — sits at **98.5-99.8% for every arm of both batches at
every length from 85 to 98**, so neither batch is trapping itself in the sense that signal measures.
What does differ is whether the food can be chased *safely*, head, food and tail in one region:

| length | 85 | 90 | 95 | 98 |
|---|---|---|---|---|
| batch 18 mean | **27.9%** | 20.9% | 12.8% | 6.9% |
| batch 20 mean | 17.5% | 15.2% | 9.6% | 5.1% |

Batch 18's boards more often admit a safe route to the food, and the gap is already open at length 85,
roughly ten food from the end.

**That gap is suggestive, not decisive, and the reason is in the same table.** At length 98 both
batches sit at 5-7% — the food is usually *not* safely chaseable for either of them — yet batch 18
still converts 63%. So a low food-chase rate plainly does not prevent winning, and the 85-98 gap
cannot be read as the mechanism. What is established is narrower and still useful:

- **The last move is not a target.** Play there is already perfect and identical in both batches.
- **What differs is the rate of arriving winnable**, which is a property of everything upstream.
- **Where upstream is unknown.** These proxies do not pin it, and the length-99 result shows how
  easily a downstream readout can look like a cause.

**That measurement has now been run** — see the next section. It read as "there is no trapped position
to find", and **that reading was retracted 2026-08-14**: the test could not see the traps.

### ‡‡ Ran it: the modal loss is starvation at length 98 — and the "never a dead end" reading is retracted below

Measured 2026-08-10 with [`perDiagnostics/point_of_no_return.py`](perDiagnostics/point_of_no_return.py),
360 greedy episodes over six shards, **0 simulator mismatches** against the live game. For every lost
episode it walks back from the death and asks, by exact breadth-first search over real game states,
the last point at which the current food could still be eaten. Three criteria, because they separate
different causes:

| criterion | question | last held, moves before death |
|---|---|---|
| **`geom`** | is the food reachable at all, ignoring the starve clock | **median 0, max 2, 100% of 75 losses** |
| `reach` | reachable *within* the remaining starve budget | median 9-15 |
| `safe` | `reach`, and the tail still reachable after eating | median 20-82, over half censored |

**`geom` holds until the very last move in every single loss, for both checkpoints.** A *path* to the
food is always there. **This was read as killing the trapped-position hypothesis, and that reading is
retracted** — a path existing says nothing about whether following it is survivable, and in the
majority of these losses it is not. See the retraction at the end of this section.

What there is instead, and it splits almost evenly:

| | b18b HoF @1588k | b20d final @3000k |
|---|---|---|
| episodes | 240 | 120 |
| perfect | **229 (95.4%)** | 56 (46.7%) |
| **starved** | 6 | **35** |
| collision | 5 | 29 |
| loss rate | **4.6%** | **53.3%** |
| starvations as a share of losses | **55%** | **55%** |
| median length at death, starvations | 82 | **98** |
| median length at death, collisions | 98 | 98 |

- **Starvation, 55% of losses in both.** Dies at median length 98 for `b20d` — two food short — with the
  food geometrically reachable at **every step including the last**. The snake burns its entire starve
  budget without going to get a reachable meal. That is dithering, not entrapment.
- **Collision, 45%.** Dies at median length 98 with a food-reaching sequence available **1-2 moves**
  earlier, so the fatal move had a non-fatal food-reaching alternative.

`b18b`'s 95.4% perfect over 240 fresh episodes is a useful side-check on the record: consistent with
the recorded 97.6% (CI 96.1-98.5) to within sampling, on different food.

**‡ Starvation being the modal failure is new and is recorded nowhere.** The 2026-08-02 diagnostics on
`b8f-disc9975seed2` @3149000 measured 360 episodes and found **288 perfect, 72 collisions, 0
starvations** ([`diagnostics/README.md`](diagnostics/README.md)). The starvation *rule* is unchanged —
`533556c` split the observation from the rule and its docstring records that the rule fires at the same
moment — so the failure mode has genuinely shifted. The likely reason is that these policies now reach
length 96-98 routinely, which `b8f` did not (its fatal decisions sat at median length 83), so they now
have somewhere to get stuck. **The binding constraint is finishing from length 96-98 inside the starve
budget**, which is a narrower target than "the endgame".

**A candidate that falls straight out, with its confound named.** Every arm in batches 16 onward runs
`SNEK_FOOD_DISTANCE_REWARD=0`; removing that shaping is [the one non-null in six
batches](#status-at-a-glance). Its job was to pull the snake towards food, and the modal failure is now
*not going to get reachable food*. So the shaping removal may have bought `sef` and paid for it in
starvations, which nobody measured either way. `b8f` had the shaping and starved zero times in 360
episodes — but it is a different environment era and never reached length 96, so that is a motivating
coincidence, not evidence. The clean test is this same script on a batch-16 arm against a
shaping-enabled control, which is cheap and needs no new training.

**What this does not establish.** The full point of no return is still unpinned. `geom` asks only "can
this food be eaten", not "can the game still be won", so a state can pass it and be doomed two food
later. `safe` is the criterion strong enough to answer that, and it is **not trustworthy here**: it
tests tail reachability on the static body, and [`diagnostics/README.md`](diagnostics/README.md) already
records that the static test flags a fatal move only **22.1%** of the time against **94.1%** for the
advanced-tail variant. So `safe` is biased pessimistic, which is consistent with it being censored in
over half the losses, and its 20-82 move figure should not be quoted as the distance.

### ‡‡ Retracted 2026-08-14: the positions *are* trapped — `geom` counts routes that eat and die

**`geom` returns success on the first move sequence that reaches the food and asks nothing about the
board it leaves behind.** Eating is the one move that does not vacate the tail (`add_segment` refills
the tile), so arriving on a food with no open neighbour fills the pocket's last cell with a head that
has nowhere to go. Measured with
[`perDiagnostics/eat_and_survive.py`](perDiagnostics/eat_and_survive.py), same two checkpoints and
protocol, **0 simulator mismatches**, 70 losses (the shard seeds behind the original 75 were never
recorded, so this is a fresh food draw at matching loss rates), enumerating **every** eating route
rather than the shortest:

| at the last state where the food was reachable | pooled n=69 | b18b @1588k | b20d @3000k |
|---|---|---|---|
| head has **no legal move** after eating | **37 (54%)** | 2 of 7 | 35 of 62 |
| dies within 5 moves of eating | **39 (57%)** | 2 of 7 | 37 of 62 |
| can eat and survive ≥100 moves | 30 (43%) | 5 of 7 | 25 of 62 |
| food cell had **no open neighbour** | **59 (86%)** | 4 of 7 | 55 of 62 |

The split is sharp, not graded: 37 sealed instantly, 2 dead within five moves, 30 fine for 100+.

**Two consequences.** The starvation reading above becomes entrapment one food earlier — in 22 of 38
starvations there was no survivable meal, so the snake was not declining one. And **"no routing
mistake tens of moves earlier" does not hold**: walking back 40 moves and asking whether *any* state
offered a survivable meal, **36 of 70 losses had none in the whole window**, while the other 34 had one
up to the last 0-2 moves.

**What it points at.** At length 98 the board holds two free cells and whether eating is survivable
depends on whether they are arranged so the head can enter one and still move — set by packing many
moves earlier, not by the move being chosen. That is a computable form of this file's open
"what differs is the rate of arriving winnable", and it is the argument for
[`CHASE_SAFE_SHAPING`](../plans/chase-safe-reward-shaping.md) rather than against it: the flag reading
0 through 95-99 is the board having no safe meal, not the flag going blind.

### ‡‡ The packing property: the records keep their free space in one piece, and it separates them by 87 points

Measured 2026-08-14 with [`perDiagnostics/endgame_packing.py`](perDiagnostics/endgame_packing.py), 60
episodes per checkpoint on **identical food streams** (seeds 201/202), **one sample per meal** so a
dithering policy cannot weight its own statistics, 0 simulator mismatches. `regions` counts connected
components of free space with the food counted as free, so **1 means the remaining space is a single
pocket**. It is one bitwise flood fill — no search, exact.

| checkpoint | perfect | one-piece @90-94 | @95-97 | @98 | mean regions @90-94 | safe meal at spawn @90-94 |
|---|---|---|---|---|---|---|
| `b24d` @1342k | 98.0% | **92%** | 96% | 98% | 1.09 | 97% |
| `b18b` @1588k | 97.6% | **77%** | 88% | 100% | 1.31 | 93% |
| `b20d` @3000k | ~47% | **5%** | 16% | 58% | 3.75 | 64% |

- **The gap opens at length 90-94, about ten meals before the end** — upstream of every loss this
  project has pinned. 87 points between `b24d` and `b20d`, against the ~10 pp effect this folder can
  resolve at n=4.
- **Not a "reaches the endgame less often" artifact.** All three reach 90-94 about equally often (290,
  299, 292 meals in 60 episodes). `b20d` arrives at the same lengths *fragmented*, it does not arrive
  less.
- **Fragmentation is what puts food where eating it kills.** Food spawns uniformly on a free cell, so
  a shredded board is a board where the food lands in a pocket: food with no open neighbour is **0-1%**
  for the records and **24%** for `b20d` at 90-94. That is the upstream cause of the trapped positions
  retracted above.
- **The two records order correctly** — `b24d` (98.0%) is better packed than `b18b` (97.6%) at every
  band. n=2, so this is an observation, not a result.
- **This does not contradict "`lg(num_groups)` points the wrong way".** That finding compares the
  *three actions* at one fatal decision, where splitting space is often the correct move — cleaning a
  pocket while the tail still adjoins it. This compares *policies* at meal spawns. Locally splitting is
  fine; chronically fragmented is fatal.
- **Read within a band only.** With two free cells at length 98, adjacency is far likelier than 6-10
  cells forming one region, so the baseline moves with length. The length-99 row is uninformative by
  construction: one free cell, always one region, always no open neighbour.

**Why it matters beyond diagnosis.** "Free space in one piece" is a bounded state function that is
already computed every step — `count_groups` runs for the observation — so it is available as the
graded potential [`chase-safe-reward-shaping.md`](../plans/chase-safe-reward-shaping.md) holds in
reserve, and as a candidate observation. It is also the first quantity measured here that separates
elite from mediocre **before** the endgame it decides.

### ‡‡ Batch 20 never learned to read "is it safe to chase the food" — but reading it does not make an arm good

> **‡ Demoted 2026-08-11, and the demotion is the useful part.** This was written up as "the best
> mechanism found" for the batch-18 gap. Tracking the same counterfactual *over training* on eight
> arms breaks that reading. The chase/is-safe ratio **rises monotonically with steps in every arm
> measured**, and **inside batch 18 it is anti-correlated with skill**: at a matched 1.0M the two
> worst arms carry the two highest ratios (`b18c` 0.635 at sef 7.4, `b18a` 0.577 at 14.8) and the two
> best carry the lowest (`b18d` 0.189 at 27.2, `b18b` 0.273 at 26.5). Across all eight arms the
> correlation with sef is **+0.04** — nothing. `b23b` is the clincher: at 540k it reads the flag like
> a batch-20 arm (ratio 0.154) and performs like a batch-18 one (sef 26.6 at 550k, against `b18b`'s
> 26.5 at 1.0M).
>
> So the batch-18-vs-batch-20 contrast below is real and correctly measured, but it is a **marker of
> how much prioritisation survives the IS correction, not a cause of the perfect rate**. The ratio at
> a matched ~500k walks down the β ladder exactly as the concentration figures do — b18 (no IS) mean
> 0.205, b23 (β→0.1) mean 0.110, b20 (β→1.0) ~0.02. Measured with
> `perDiagnostics/input_sensitivity_over_time.py`; chart in
> [`charts/drawdown-b23b-vs-b18.png`](charts/drawdown-b23b-vs-b18.png) panel C.

Observation indices **15-17** are, per action, "head, food and tail all end up in one region" — the
signal added specifically so a policy could tell a reachable meal from one that seals it in. The
question is whether a board where it fires is valued higher, and it separates the batches more
sharply than anything else measured. Mean max-Q over lengths 95-98, split by whether any action is
chase-safe:

| arm | n | Q, chase-safe available | Q, none | delta |
|---|---|---|---|---|
| `b18a` | 12,265 | **65.95** | 34.25 | **+31.71** |
| `b18d` | 9,502 | **63.78** | 35.27 | **+28.51** |
| `b18c` | 11,680 | **53.89** | 34.44 | **+19.44** |
| `b18b` | 16,491 | **33.53** | 18.39 | **+15.14** |
| `b20b` | 20,406 | 5.03 | 2.98 | +2.05 |
| `b20c` | 22,068 | 3.30 | 1.89 | +1.41 |
| `b20a` | 9,006 | 2.59 | 1.27 | +1.33 |
| `b20d` | 19,452 | 3.26 | 2.03 | +1.23 |
| **mean** | | | | **+23.70 vs +1.50**, 4/4, p=0.125 |

**A correlational split cannot show the network reads the input**, so the load-bearing measurement is
a counterfactual: flip index 15+a on the real board, hold everything else at its measured value, and
re-read that action's Q. Index 6+a (is the move survivable) gets the same treatment as a positive
control, because every network must weigh that one.

| | idx 15-17 (chase-safe) | idx 6-8 (is safe) | ratio |
|---|---|---|---|
| batch 18 mean | **+11.70** | +12.39 | **1.051** |
| batch 20 mean | **+0.228** | +5.587 | **0.045** |
| | 4/4, p=0.125 | 4/4, p=0.125 | 4/4, p=0.125 |

**Batch 18's networks weigh "the food is safely reachable" about as heavily as "this move will not
kill me". Batch 20's weigh it at 4.5% of that** — and `b20a`'s weight is **negative** (−0.97), so it
treats a safely reachable meal as marginally bad. That is a wrong weight, not merely an untrained one.

The ratio matters because batch 20's Q values are ~3x smaller overall, which shrinks every derivative.
The control absorbs that: the is-safe sensitivity differs by only 2.2x, tracking the scale gap, while
the chase-safe sensitivity differs by 51x.

**Robustness.** Setting `chase=1` on a move the board says is fatal is a contradictory input, so the
whole measurement was repeated on **safe actions only**, where the flag is meaningful. The effect
shrinks but survives: **+3.708 vs +0.541, 4/4, p=0.125** — a 6.9x gap rather than 51x. Take 6.9x as
the conservative figure. Do **not** normalise this restricted version by the wall-hug flag at 23-25 as
a control: its sensitivity is near zero (−0.50 to +1.15), so the ratio is unstable and reads p=0.625
purely from dividing by noise.

**This supersedes the slow-propagation reading of the length-98 dip.** Batch 20's Q at 98 is not low
because the terminal reward has not arrived; it is low *and flat* because the network cannot tell a
winnable board from a doomed one and assigns ~2-3 to both. Batch 18 assigns 34-66 against 18-35. The
value function is not lagging, it is **undiscriminating** — and routing through the endgame is exactly
the decision that needs that discrimination.

**Why prioritization is the plausible cause.** Index 15-17 fires in only **5-7%** of endgame states.
Under β=1.0 those transitions receive exactly their population share of the update, so the weights on
a rare-but-decisive input stay weakly determined. This is the same hazard the root `CLAUDE.md` records
for index 29 (1 in 99.95% of states) and for the `game_over` input whose unconstrained weights turned a
90.3% champion into one scoring 0 — **rare rather than constant, so a milder form, but pointed at the
one input that predicts the 100-point reward.** It also reframes the next experiment: what
`IS_WEIGHTS=0` would be buying is not faster backups but enough gradient on the rare informative
states to fix a weight. **That last sentence is the part the demotion above bites**: more gradient on
the rare states does reliably raise this reading, and raising this reading does not raise the perfect
rate, so `IS_WEIGHTS=0` has to be justified by the concentration ladder rather than by this input.

### The record checkpoint, specifically

`hallOfFame/b18b-tgt1000seed2-ckpt1588000`, the 97.6% record, restored and confirmed at
`global_step 1588000`:

| length | n | mean max-Q | Q chase-safe | Q none |
|---|---|---|---|---|
| 95 | 4,695 | 19.71 | **26.40** | 18.92 |
| 98 | 3,815 | 9.85 | **32.94** | 8.42 |
| 99 | 74 | 53.10 | — | — |

**Its mean Q at length 98 is 9.85, and that average is misleading** — 222 of 3,815 boards offer a safe
chase and are valued at 32.94, while the other 94% are valued at 8.42, correctly, because they are
losing positions. So "the value function goes flat at the endgame" was partly an artefact of averaging
over a state distribution that is overwhelmingly unfavourable.

At length 99, **41 of 74 states have a winning move and their mean max-Q is 99.73** — the terminal 100
is learned essentially exactly, with no propagation deficit at the final step. Note also that the
chase-safe flag is **structurally 0 at length 99** for every arm: with one cell free there is no region
containing head, food and tail, so indices 18-20 take over the job. The two blocks are complementary
rather than redundant, which is worth knowing before anyone prunes either.

Measured with `perDiagnostics/per_priorities.py`'s sibling probes; the buffer boards come from ~2.40M
while this checkpoint is from 1.588M, so they are real length-98 boards but not the ones it would
generate itself.

### What this does not establish

- **Two knobs moved together.** Batch 18 changed the signal *and* dropped IS, so nothing here
  attributes the outcome to one. The concentration ladder prices them separately (0.213 → 0.454 →
  0.951), which is what makes `td_error` + `IS_WEIGHTS=0` worth running: it is a **pre-registered
  midpoint**, and the docs already wanted it for the drawdown result.
- **Priorities were recomputed, not recovered.** `save_transitions()` resets them to the max, so
  these are fresh priorities under each arm's final network. Real in-buffer priorities were
  **staler** and therefore flatter, so treat the concentration figures as the sharpest the config
  could be.
- **One snapshot per arm, after its peak** — the final 100k transitions at 2.4-2.6M (b18) and 3.0M
  (b20). The direction is 4/4 on every comparison, but nothing here tracks how the picture evolves,
  and the value-shape gap could as easily be a *consequence* of batch 20 winning less as a cause.
- **Sample sizes per cell of the Q table run 251 to 8,056, and the thin end is the *early* game,
  not the endgame** — length 10 rests on 251-493 rows against length 98's 3,003-8,056. The buffer is
  the last 100k transitions of a policy that spends most of its time long, so the early game is a
  few dozen steps per episode and the endgame is hundreds. Length 99 holds 45-107 real decision
  points. **Length 100 is 25-32 rows and every one is a boundary frame** — `step_type` LAST, reward
  0, `next_step_type` FIRST — so the policy never acts there and that column's Q is unconstrained,
  the same shape as the `game_over` trap in the root `CLAUDE.md`. Read the spike at the end of the
  curve as length **99** only.

## ‡‡ What the record checkpoints do differently: they find food in the endgame, and that is nearly all of it

Twelve checkpoints spanning 30-95% perfect, each played over **the same 100 greedy games** so the
comparison is paired and the game set cancels. `perDiagnostics/behaviour_profile.py` logs, per step,
the observation values of the action the policy *actually chose*. Chart:
[`charts/champion-vs-mediocre.png`](charts/champion-vs-mediocre.png).

The separation is almost entirely one quantity — **how long a meal takes at length 95-99**:

| checkpoint | perfect % | p90 steps/meal at 95-99 | meals over 200 steps | budget left, worst tenth |
|---|---|---|---|---|
| `b17b` @1190k | 95 | **5.0** | 0.2% | **495** |
| `b18b` @1588k | 93 | **5.5** | 1.0% | **495** |
| `b11b` @855k | 92 | **5.5** | 0.2% | **495** |
| `b13d` @986k | 89 | **5.4** | 0.2% | **495** |
| `b15b` @3245k | 87 | 7.9 | 0.6% | 492 |
| `b14a` @3702k | 86 | 13.4 | 2.5% | 487 |
| `b23b` @549k | 80 | 45.2 | 0.2% | 455 |
| `b20d` peak | 78 | **85.8** | 2.2% | 414 |
| `b20b` peak | 72 | **92.5** | 5.0% | 408 |
| `b20c` peak | 36 | **226.2** | **11.3%** | **274** |
| `b20a` peak | 30 | **175.4** | 9.1% | 325 |

The records reach the food in **2 moves at the median and 5-6 at the 90th percentile**. The batch-20
peaks take **86-226 moves at the 90th percentile**, and one meal in ten costs `b20c` more than 200
moves against a 500-step budget. That is the starvation finding made mechanical: the mediocre
policies are not trapped and are not making fatal blunders, they are **wandering**, and at length 96+
wandering runs out the clock.

Across all twelve, `steps_per_food` at 85-94 correlates **−0.967** with perfect rate. Every other
marker moves with it: packing (`hug` +0.944), fragmentation (`regions` −0.942), straight-line movement
(`forward` +0.917), tail reachability (+0.899). These are one factor, not five.

**Within the top seven** (all 85-95%, so the two failures cannot drive it) most of those markers wash
out — and the one that survives is **realised chase-safety**: the share of chosen moves that keep
head, food and tail in one region. Pearson **+0.860** at 85-94 and **+0.822** at 95-99, the only
marker strong in both bands. Note what this is *not*: it is not the Q-sensitivity to observation
15-17, which the section below demotes for failing to predict skill. **The network does not need to
value the flag; the policy needs to keep the property true.** With n=7 across ~18 tested
correlations, treat it as the best available lead rather than as established.

**The arithmetic of what is left.** A perfect game is **95 consecutive meals**, so the per-meal error
rate is the quantity that matters and it compounds brutally:

| checkpoint | perfect % | per-meal failure | needed for 99% | reduction |
|---|---|---|---|---|
| `b17b` @1190k | 95 | 0.054% | 0.011% | **5.1×** |
| `b18b` @1588k | 93 | 0.076% | 0.011% | 7.2× |
| `b20d` peak | 78 | 0.261% | 0.011% | 24.7× |
| `b20a` peak | 30 | 1.259% | 0.011% | 119× |

So the best checkpoint on record already plays 1,850 meals per mistake, and a 99% perfect rate needs
one per 9,450. **Chasing perfect-game percentage points understates how good the policy already is
and how much is left**: 5× on per-meal reliability, concentrated in the ~5 meals per game played at
length 95+. The reason those meals are special is not skill but geometry — the free space is a thin
corridor, so a wrong turn costs a long detour under a clock that no longer scales with length (the
budget caps at 500 from length 50 up).

## ‡‡ There is no lucky checkpoint: an arm's best is set by its median

From the project's own close-out measurements — 3,712 full-depth rows (≥100 episodes, not abandoned)
across 68 files, no new compute:

| arm | rows | best | median | ≥90% | ≥95% |
|---|---|---|---|---|---|
| `b18b-tgt1000seed2` | 9 | 97.4 | **95.0** | **100%** | **67%** |
| `b17b-forkseed2` | 39 | 97.0 | **95.0** | 77% | 59% |
| `b15b-nstep3seed2` | 94 | 97.0 | 90.0 | 69% | 9% |
| `b11b-obs30seed2` | 204 | 96.0 | 83.0 | 19% | 1% |
| `b13d-shieldseed4` | 148 | 95.0 | 79.0 | 3% | 1% |
| `b10d-disc995seed4` | 660 | 93.0 | 75.5 | **0%** | 0% |
| `b10b-disc995seed2` | 624 | 90.0 | 72.0 | **0%** | 0% |

**An arm's best measured checkpoint is predicted by its median at r=+0.971.** `b10b` had 624
checkpoints measured at full depth and **not one cleared 90%**; `b18b` had nine and **all nine did**.
There is no lottery to win inside a mediocre arm — the high checkpoints live in arms that are high
everywhere, and `b18b` held ≥90% across **1.2M steps** of measured rows.

The practical consequence is that **screening more checkpoints is not a route to a better policy**,
which matters because that is where a lot of close-out compute goes. Raising an arm's median is the
only thing that moves its best.

### ‡ And the corollary: most of a selected high is selection

Two measured checkpoints of the same arm **less than 20k steps apart differ by no more than binomial
noise** — mean |Δperfect| 5.90 pp against a 6.48 pp noise floor at these episode counts. Only past
~50k does real signal exceed noise. So a 100-episode read cannot resolve neighbouring checkpoints at
all, and picking the maximum of many such reads buys mostly luck:

| checkpoints screened | pp the max reads high, all truly at 90% |
|---|---|
| 5 | +3.4 |
| 20 | +5.2 |
| 50 | +6.1 |
| 100 | +6.7 |

This **fully accounts for the shrinkage this project has documented four times** — `b17b` 99/100 →
94.2%, `b15b` 97/100 → 93.0%, `b14a` 96/100 → 93.5%, mean **−5.05 to −5.2 pp**. No extra mechanism is
needed, and in particular the "the record is a narrow peak, not a region" reading is not required by
the data: neighbouring checkpoints are indistinguishable at the depth used to call them different.

**Confirmed against fresh games, with one caveat that cuts the other way.** Re-measuring the two most
heavily measured entries on games neither had seen:

| checkpoint | recorded | fresh /200 | delta |
|---|---|---|---|
| `b17b` @1190k | 94.24% /5120 | **95.5%** | +1.3 |
| `b18b` @1588k | 97.57% /700 | **94.0%** | −3.6 |

`b17b`'s 5,120-episode figure reproduces within noise, which is the check that says the protocol here
matches the project's. `b18b`'s reads 3.6 pp low, about 2σ — **suggestive that even the 700-episode
record figure is a little optimistic, not conclusive.** The five remaining hall-of-fame entries also
read 2.0-7.5 pp below their recorded values, but those share one game set that the two cross-checks
show runs ~1-2 pp hard, and they are therefore **not seven independent observations** — an earlier
draft of this section quoted a paired p-value over them, which was wrong for exactly that reason.

## ‡‡ Falsified: a drawdown is not how a policy escapes a local minimum

`b23b` collapsed from score 94 to 4 over 217-242k and came out the other side with a much higher
perfect rate — trailing-30 perfect ran ~35% before, bottomed at 9.7%, and reached 55% by 300k and
75% by 550k. The hypothesis that suggested itself is that the collapse *is* the escape: a forced
excursion that breaks a mediocre optimum. **It is not.** Five measurements, one direction.

| question | measurement | answer |
|---|---|---|
| Did the endgame value structure reorganise? | dQ from flipping "safe to chase food" (obs 15-17) on 5,973 fixed boards | **no** — 0.993 → 0.990 → 1.020 across 218k/223k/236k, on a smooth curve from 0.90 at 180k to 1.11 at 300k |
| Did the network start reading different inputs? | mean gradient per input, all 30 | **no** — the same rank order before and after, everything scaling together (total mass 96 → 170) |
| Did the greedy policy churn faster? | share of the 5,973 boards changing argmax per 2,000 steps | **mildly** — 0.034 inside vs 0.020 before, but the run's two *largest* churn events (0.055, 0.048) happen at score 93 |
| Was the rise the drawdown's doing? | the three sibling seeds, same config, no collapse | **no** — `b23d` gained **+48.6 pp** with no drawdown at all, against `b23b`'s +40.9 |
| Is it a batch-18 pattern too? | four dip windows, 2,000-step ladders | **no** — churn 1.10-1.33× in dips, and the chase-reading slope is *unchanged* (faster in 2 windows, slower in 1, flat in 1) |

The sibling control is the decisive one. All four `b23` arms make the same level shift over the same
steps — `b23a` +27.1, `b23c` +25.8, `b23d` +48.6, `b23b` +40.9 pp between 150-215k and 450-550k —
so the rise is ordinary continued learning that `b23b`'s collapse briefly interrupted, not a
consequence of it. Chart: [`charts/drawdown-b23b-vs-b18.png`](charts/drawdown-b23b-vs-b18.png).

**What a drawdown actually is: the mid-game breaks, not the endgame.** Fresh greedy episodes per
checkpoint, `perDiagnostics/point_of_no_return.py`, 0 simulator mismatches:

| checkpoint | episodes | perfect % | starved | collision | median length at death | starve share of losses |
|---|---|---|---|---|---|---|
| 200k, healthy, before | 160 | 46.9 | 73 | 12 | **96** | 86% |
| 216k, entering the collapse | 80 | 20.0 | 62 | 2 | **30** | 97% |
| 300k, recovered | 80 | 61.2 | 30 | 1 | **97** | 97% |
| 549k, now | 160 | 76.2 | 19 | 19 | **98** | 50% |

Outside the collapse the policy reaches length 96-98 and loses to the starve clock at the very end,
which is the failure mode already on record. Inside it, the median death length is **30** and 97% of
losses are starvation — the snake stops finishing the *mid* game and starves early. So a drawdown is
not a reorganisation of endgame skill; it is a temporary loss of the ability to keep eating, and the
endgame machinery sits untouched underneath it, which is exactly why recovery is fast.

The one thing that did improve monotonically is the **failure mix**: starvation falls from 86% of
losses at 200k to 50% at 549k, while collisions rise from 12 to 19 in the same number of episodes.
Progress toward a high perfect rate looks like trading starvations for collisions, not like escaping
an optimum — and it puts a number on how much headroom the starve clock still holds.

**Two caveats.** `SNEK_MIN_CHECKPOINT_SCORE=40` gates on `max(avg_score, trailing_avg_score)`, so the
entry and exit of a collapse are always checkpointed but a deep trough is thin: `b23b`'s 26-eval
collapse left only 218k, 223k and 236k. The three agree with each other and with the smooth curve
through them, but a transient excursion *between* them cannot be excluded — and pre-gate arms with
full trough coverage (`b8d`, `b10c`) are the 20- and 26-value observation eras, where indices 15-17
do not exist at all. Second, the rollout rows are **one checkpoint each**, and checkpoint-to-
checkpoint variance is large in this project; the 200k row reads 46.9% where the training graph's
eval at that step read 50%, which is agreement, but the before/after gap should be read against the
training curve rather than these two numbers.

## ‡‡ The seed, not the config, decides which arm in a wave wins — and it holds to 2M steps

Grouping every four-arm wave since the observation era froze by its config tag, and scoring
`strong_eval_fraction` at a matched horizon:

| horizon | waves | mean sef(seed 2,4) − sef(seed 1,3) | positive | exact paired p |
|---|---|---|---|---|
| 550k | 18 | **+5.41 pp** | 16/18 (one tie at 0.00, one at −0.09) | **0.00005** |
| 1.0M | 16 | **+7.70 pp** | 15/16 | 0.00040 |
| 1.5M | 14 | **+8.82 pp** | 13/14 | 0.00122 |
| 2.0M | 14 | **+8.73 pp** | 12/14 | 0.00122 |

**Seed 2 or seed 4 was the best arm in 18 of 18 waves at 550k**, across `disc995`, `obs30`,
`shield`, `disc9975`, `nstep3`, `noshape`, `fork`, `tgt1000`, `stdper`, five FC shapes, `beta05` and
`beta01` — every config change of the last thirteen batches. Mean sef at 550k by seed: **0.6, 7.1,
0.6, 6.5**. Exact two-sided paired permutation over all 2^n sign flips.

This matters for how results here are read:

- **The effect is comparable to the largest config effect on record.** Batch 18's signal change is
  ~+21 pp of sef; the seed is worth ~+9 pp and it never fails to show up.
- **It is a ceiling effect, not just a slow start.** It *grows* from 550k to 1.5M rather than washing
  out, so "give the bad seeds more steps" does not fix it.
- **It does not invalidate the config comparisons**, which is the good news: every batch since 10 has
  used the same four seeds, so the paired designs already difference it out. What it does invalidate
  is any comparison across a *different* seed subset, and it explains part of what this project has
  been calling domain noise — "the same config produced 62.5 and 18.0" is partly which seed ran.
- **The mechanism is open, and it is genuinely odd** that it exists at all: a seed does not reproduce
  a run here, because `cpprb`'s sampling RNG is unseeded and two same-seed arms diverge inside 1,000
  steps (see the section below). Whatever seeds 2 and 4 confer therefore survives that divergence.
  Initial weights are the obvious candidate — identical per seed for every 30-value `50,100,50` arm —
  but `b10` is 26 values with a differently shaped first layer and still shows +6.7, which an
  init-only story does not explain.

The cheap follow-up is **seeds 5-12 on one fixed config**: it prices the seed distribution properly,
says whether 2 and 4 are unusually good or 1 and 3 unusually bad, and costs no new code.

## Falsified: epsilon reaching 0.0 does not cause the collapse

The hypothesis was that the epsilon ladder's last rung (`epsilon.assign(0.0)` once
`avg_reward > 100`) makes the collect policy fully greedy, turning the replay buffer into
a closed loop on the policy's own behaviour. The evidence looked strong: `b1a-base` was
the only arm that reached 0.0 and the only arm that collapsed, with a mechanism and a
timing that fit (0.0 at 92k, collapse at 265k, ~173k apart — about the time to flush a
100k buffer at ~800-step episodes).

Three arms settled it, all past the judgeable horizon:

| arm | epsilon regime | outcome |
|---|---|---|
| `b3b-epsfloor2` | floored at 0.001 from 147k | peaked 305k, declined 71 → 52 and 7.0% → 3.3% |
| `b3a-epsfloor` | floored at 0.001 from 267k | peaked ~300k, declined 74 → 61 and 8.6% → 1.3% |
| `b3c-buf500k` | **fully greedy at 0.0** from 282k | did not break in the predicted window; later died at 750k for unrelated reasons |
| `b4c-schlongper` | fully greedy from 121k | the best arm in the investigation |

The prediction failed in both directions. **The caveat recorded when the hypothesis was
proposed turned out to be the entire signal:** reaching 0.0 requires `avg_reward > 100`,
so only *strong* runs get there — "reached 0.0" and "was good enough to collapse from a
height" are entangled and the correlation cannot separate them.

Two things worth carrying forward:

- **That correlation was as strong as this domain produces** — one arm at 0.0, one arm
  collapsed, same arm, specific mechanism, timing that fit. It was still wrong. With n=1
  arms and a stated confound, a mechanism that "fits the timing" adds no evidence.
- **The test was still worth running.** It cost one knob and three arms that were going to
  run anyway, closed the question, and incidentally produced batch 3's best arm and the
  natural experiment that settled it.

`MIN_EPSILON` stays in the code, and knowing epsilon 0.0 is *safe* relative to 0.001 is a
useful result.

### Scope of that falsification, added 2026-08-04: it was never about the descent rate

**This result stands, and it is narrower than its own heading.** What batch 3 compared was a
floor of **0.001 against 0.0** — and 0.001 is ~1.2 forced non-greedy moves in a 1780-step game,
so both arms were playing essentially greedily. The test established that those two are
indistinguishable. It did not test exploration, because neither condition had any.

What went unexamined for four months is how fast the ladder got there. Measured across batches
10 and 11: **96.8% of all 31.1M training steps ran at epsilon exactly 0.0**, 99.6% at ≤0.001,
and the ladder bottomed out at median step 15000 of runs 3.2-4.7M long — while 7 of 8 arms were
still at 0% perfect games. So the correct reading of batch 3 is:

| tested | not tested |
|---|---|
| floor 0.001 vs 0.0, both effectively greedy | any floor large enough to change behaviour |
| whether the *last rung's value* matters | whether reaching the last rung at step 15k matters |

The schedule was rewritten 2026-08-04 for that reason — two phases, no ratchet, floor 0.002,
and exactly 0 rejected. See
[`hyperparamTuning.md`](hyperparamTuning.md#the-epsilon-schedule--rewritten-2026-08-04-and-it-breaks-curve-comparability).
**That rewrite is not evidence against anything on this page.** It is a change to an
untested part of the design.

#### Now measured, 2026-08-05: exploration was tested in both directions, and neither helped

Two batches closed the gap this section described, and the answer is that the untested part of the
design was untested because there was nothing there.

| batch | handover epsilon | result |
|---|---|---|
| 12 | 0.05 | **deadlock.** 4 arms, 0% perfect games to ~1M, greedy trailing 53-63 vs 84-88 |
| 12s | 0.05 + exploration shield | decay fixed, still plateaus at trailing ~83 with 0.3% perfect |
| 13 | 0.0125 + shield | works, and is **null on five metrics** vs batch 11, n=4 paired |

Batch 13's five, now that its checkpoints are measured: best ckpt -1.8 pp (p=0.875), top-3 -1.4 pp
(p=0.875), graph-100% tier **-0.1 pp** (p=1.000), `best_perfect30` **+0.0 pp** (p=1.000),
`strong_eval_fraction` +2.0 pp. Two independently-computed metrics landing within 0.1 pp is what a
genuinely zero effect looks like measured five ways.

So the honest closing position on epsilon: **too much exploration is actively harmful and the right
amount is indistinguishable from none.** 0.05 is fatal to sit at, because a collect policy at 3.3%
random actions never finishes a board and the buffer never holds the last ten food. 0.0125 is
harmless and buys nothing measurable.

What survives is only what was defensible on mechanism in the first place: the **ratchet** was a
real defect (`b11b` sat at 0.001 through a collapse from 64.6 to 8.8 with no way to recover), and
exactly 0.0 makes the buffer a closed loop. A 0.002 floor plus a stateless schedule fixes both
without needing elevated exploration. The original framing — "96.8% of steps at exactly 0 is a
defect" — was **cosmetic**, and it cost two batches to find that out.

**Do not re-open this at n=4.** Three batches have now failed to separate any epsilon regime from
batch 11's on a metric whose between-seed spread is ±12 pp.

The lesson worth keeping is about how the original hypothesis was framed. "Epsilon 0.0 causes
the collapse" named a *value*, so the experiment tested a value and closed the question at that
value — while the schedule that produced the value went unexamined. A hypothesis about a knob
should say which property of the knob is doing the work.

### Related: when each arm's epsilon treatment actually started

`MIN_EPSILON` only changes behaviour at the last rung, and crossing `avg_reward > 100` is
uncommon and late. **One crossing is all it takes**, because the ladder is a one-way
ratchet — a single eval over 100 pins epsilon permanently, and a later score drop never
raises it back.

| policy | first `avg_reward > 100` | epsilon after |
|---|---|---|
| `b4c-schlongper` | 121k | 0.0 |
| `b1a-base` | 92k (18 evals over) | 0.0 |
| `b3b-epsfloor2` | 147k | 0.001 (floored) |
| `b3a-epsfloor` | 267k | 0.001 (floored) |
| `b3c-buf500k` | 282k | 0.0 |
| `b4a-uniform` | 425k | 0.0 |
| `b4b-unifbuf500k` | 290k | 0.0 |
| `b2a-base2` | never (peaked 99.1) | 0.001 |

This is why a floored arm can be indistinguishable from an unfloored one for its first
few hundred thousand steps, and why `b3a-epsfloor` spent 267k steps as an accidental
baseline repeat.

## Not settled: whether a larger replay buffer helps

`REPLAY_BUFFER_MAX_LENGTH=500000` was tested twice with opposite results:

| arm | buffer | sampling | outcome |
|---|---|---|---|
| `b3c-buf500k` | 500k | PER alpha 0.6 | flattest curve in the investigation, then **died completely at 750k** |
| `b4b-unifbuf500k` | 500k | uniform | steadiest arm, healthy at 1.23M, but only 9.3% |

The difference between them is prioritization, not buffer size, which points the same way
as the `b4c` result. A 500k buffer with uniform sampling is stable-but-low; with PER it
died. Neither is evidence that buffer size is the lever, so **the diversity-squeeze
mechanism described in [`completedRuns.md`](completedRuns.md) is still an untested
hypothesis**, not a finding. `REPLAY_BUFFER_MAX_LENGTH=1000000` is in the backlog at low
priority.

## A seed does not reproduce a run, and the reason is the replay buffer

Measured 2026-08-07, because two docs disagreed: `seed_process`'s docstring said a seed buys "reduced
variance, not reproducibility" while the hall-of-fame README claimed a fresh run of the same config
*was* reproducible. The docstring was right, and the cause is now identified.

**Two arms, same `SNEK_SEED=7`, byte-identical config.** What is the same and what is not:

| layer | same seed → same? | how it was tested |
|---|---|---|
| `random` / `numpy` / `tf` draws | **yes**, bit-identical | direct draws in fresh processes |
| network initialisation | **yes**, identical weight hash | hashed all 8 variables after `create_variables` |
| environment: food, episode lengths, observations | **yes**, identical stream | fixed action sequence, hashed 6 episodes of observations |
| 200 gradient steps on a **fixed** batch | **yes**, identical weight hash | same batch fed to both |
| **trained weights at step 1000** | **no** | hashed every checkpoint from two real arms |
| eval trajectory | **no** — diverges then amplifies | 0.001 apart at first, 1.8 vs 7.5 avg score by 5000 |

**The nondeterminism is in the data, not the math.** Gradient steps on identical batches reproduce
exactly, so what differs is *which transitions get sampled*. Prioritized sampling runs inside
`cpprb`'s C++ sum tree, and nothing in this project seeds it — `seed_process` covers `random`,
`numpy` and `tf` only. Sampling the same buffer in fresh processes with all three seeded returns
different indexes every time. **cpprb's constructor accepts a `seed=` kwarg and silently ignores it**,
verified directly, so this is not a one-line fix.

Three plausible fixes were tested and none works: `TF_DETERMINISTIC_OPS=1`, single-threaded TF
(`TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1`) and `PYTHONHASHSEED=0`. Each only moves the
divergence point by a step or two of eval. Running the arms sequentially instead of in parallel also
does not help, which rules out CPU contention.

**What this means in practice, and it is mostly reassuring:**

- **Seeds still do their job.** Every arm in a batch starts from the same weights and meets the same
  environment, which is exactly the between-arm variance a seed is meant to remove. What it cannot
  remove is trajectory divergence once training starts.
- **Paired-by-seed statistics remain valid.** Pairing `b14a` with `b15a` is still pairing two arms
  that began identically; it was never an assumption that they would stay identical. This is the
  mechanism behind the already-established finding that a seed is not a stable unit of quality.
- **Never report a re-run as confirming an exact number**, and never expect a resume to continue the
  original trajectory — the RNG state is not checkpointed.
- **‡ The replay buffer *is* checkpointed, corrected 2026-08-09.** This line previously said the
  buffer was not saved either, which is wrong: `training.py` calls `replay_buffer.save()` every
  `10 * eval_interval` steps and `snek2.py` calls `replay_buffer.restore()` at startup, printing
  `restored replay buffer: N transitions`. **A resume is warm-started, not cold**, which matters
  whenever a resume's early behaviour is being interpreted. Two caveats that remain true: the save is
  gated on the same `MIN_CHECKPOINT_SCORE` condition as the policy checkpoint, so an arm that never
  clears the bar has no buffer to restore; and because it saves 10x less often than the policy, the
  restored buffer can be **up to 10k steps older** than the restored weights.
- **Bit-reproducibility would require replacing cpprb** with a seedable buffer. Not worth it for
  tuning; it would matter only for debugging a specific divergence.

## Re-opened 2026-08-02: n-step returns were never cleanly tested

| policy | steps | peak score (at) | best perfect-30 | 1st perfect |
|---|---|---|---|---|
| `b1c-nstep3` | 1.14M | 76.0 (255k) | 1.7% | 206k |
| `b2b-nstep2` | 580k | 74.6 (140k) | 0.7% | 121k |

Both peaked *below* every baseline, both then declined for hundreds of thousands of steps,
and both sat at zero perfect games in their trailing windows. Two arms ordered by n giving
the same shape looked like a trend rather than noise, and this overturned a still earlier read
that n=3 had "the best trajectory of the batch" — true through 200k, false afterwards.

**That conclusion is withdrawn, because the mechanism it tested was broken.** Terminal steps
carried a non-zero discount until 2026-08-02, and `to_n_step_transition` composes

```
r_t + g*d_t*r_{t+1} + g^2*d_t*d_{t+1}*r_{t+2} + ...
```

where those per-step `d` values are the **only** thing that truncates the sum at an episode
boundary. At `d = 0.9975` on a terminal step, an n-step return keeps accumulating past the end of
the episode into whatever sits next to it in the replay buffer. Both arms above were therefore
trained on returns that mix episodes together, which is a fair explanation for peaking below every
1-step baseline.

This is **not** evidence that n-step helps. It is a retraction of the evidence that it does not.

**Batch 15 tested `n=3` on 2026-08-06 and the predicted mechanism did not appear.** n-step's claim
here was faster credit propagation — the +100 perfect-game reward needs ~890 sequential backups to
cross a ~1780-step game at n=2 and ~593 at n=3 — so the pre-registered read was **steps to
pf30 ≥ 40%**. It came out **128k slower** than the batch-14 control, 3 of 4 seeds slower (p=0.250).
Level is a null: `strong_eval_fraction` +4.05 pp at p=0.625 equal-effort, peak trailing -0.10.

So the honest status is **falsified on speed, null on level**, at n=4. Contamination was never the
issue — at the measured epsilon of 0.0034-0.0039 an uncorrected n=3 return is exact for 99.5% of
targets — so the absence of an effect is not explained by the one cost the theory predicts. What
remains unexplained is *why* propagation did not speed up, and the most likely answer is that credit
propagation was never the binding constraint. Do not try `n=5`: the reason for preferring larger n
was that the effect scales with n, and the effect is absent at n=3 in the direction it should be
largest.

**‡ 2026-08-10 supports "never the binding constraint", from the other end.** The +100 is the reward
n-step exists to move, and at the final decision **286 of 286 winnable positions were won and 285 of
285 unwinnable ones lost, identically in both PER families** — so there is no terminal-value error at
the end of the chain for a faster backup to correct, and what separates arms is arriving winnable.
Note also that batch 15 ran n=3 on the `td_loss` + no-IS family, which converts the last move at
55-67%; n=3 has **never** been tried on the uniform-replay family that converts at 13-40%. That is a
real gap in coverage, but it is not a reason to expect a win, and n=3 there would confound the
propagation change with a priority change, since larger n-step errors feed a sharper effective
exponent. See
[above](#-the-gap-is-decided-before-the-last-move-not-at-it).

Design and full numbers in
[`completedRuns.md`](archive/batches12-15.md#batch-15--n_step_update3-falsified-on-speed-null-on-level-and-a-97100-that-is-really-93).
The checkpoint evals agree: best checkpoint +0.05 pp at p=1.000, `pooled_equal_effort` +2.24 pp at
p=0.625.

## The record across four environments: 51% → 92% → 93% → **96%**

Each environment change resets the comparison set, so the record is really four records. The
progression is still worth reading as one line, because every step came from a different cause:

| record | arm / checkpoint | environment | what moved it |
|---|---|---|---|
| 51% | `b7f-disc995seed3` @860k | pre-audit | `DISCOUNT=0.995` |
| 92% | `b8f-disc9975seed2` @2816k | pre-audit | **the horizon** — 2.8M steps instead of 1.06M |
| 93% | `b10d-disc995seed4` @1695k | 26-value (2026-08-02's seven fixes) | the environment fixes, not a config change |
| **96%** | `b11b-obs30seed2` @855k | **30-value (current)** | unattributable — see the caveat below |

**The 96% cannot be credited to the two new observations.** Batch 11 differs from batch 10 only by
those observations, and its close-out came out +4 to +5 pp on three metrics with p between 0.14 and
0.24 — consistent with a real effect and equally consistent with seed luck. A single record
checkpoint is the weakest possible evidence for a config change, being the max of 204 noisy
measurements in one arm; corrected for the winner's curse it is ~94%, against `b10d`'s ~87%.
Write-up: [`archive/batches1-11.md`](archive/batches1-11.md#batch-11--the-same-config-on-the-30-value-vector-no-significant-difference).

**Two of the four steps came from something other than hyperparameters** — the horizon, and the
environment audit. That is the most useful pattern in this table, and it is why the standing backlog
in [`runs.md`](runs.md) is ordered below the design fix rather than above it.

### The 92% of the batch-8 era, and why the horizon was the binding constraint

Final close-out measurement 2026-08-01, with the same arms' earlier measurements below for the
trajectory — which is the whole story of this section:

| arm | when | ckpts | best ckpt | top-3 | pooled | 95% CI |
|---|---|---|---|---|---|---|
| `b8f-disc9975seed2` | **close-out, 5.47M** | 52 | **92.0%** @2816k | **86.7%** | **66.3%** /5200 | 65.1-67.6 |
| `b8d-disc995clip` | close-out, 11.64M | 20 | 76.0% @5027k | 72.7% | 60.4% /2000 | 58.2-62.5 |
| `b8f` | mid-run, 2.65M | 63 | 88.0% @2581k | 82.7% | 59.2% /6300 | 57.9-60.4 |
| `b8d` | mid-run, 2.93M | 25 | **80.0%** @2538k | 74.7% | 58.4% /2500 | 56.5-60.3 |
| `b8f` | mid-run, 1.78M | 16 | 63.0% @1618k | 60.3% | 46.5% /1600 | 44.1-48.9 |
| `b8d` | mid-run, 2.08M | 10 | 62.0% @1688k | 58.7% | 48.3% /1000 | 45.2-51.4 |
| `b7f-disc995seed3` | final, 1.06M | 10 | 51% @860k | 48.0% | 38.8% /1000 | — |
| `b4c-schlongper` | final, 1.06M | 10 | 50% @869k | 46.7% | 37.1% /1000 | — |

**Pooled figures are only comparable within one selector.** The close-out rows used the current rule
(all >=90%, fill to 20 from >=60%); the mid-run rows used the earlier >=80% rule. A more selective
set has a higher pooled rate by construction, so `b8f`'s 59.2% → 66.3% is partly the selector. The
**best-checkpoint column is comparable throughout**, and there the record went 51% → 88% → **92%**.

**The pooled column carries the claim.** 59.2% over 6300 episodes has a ±1.3 interval, so this is
not a best-of-N artefact: it is 20 points above the pooled figure that stood the same morning and
non-overlapping with it. `b8f` has 35 of 63 checkpoints at >=60%.

**The two configs stay tied on pooled** (overlapping intervals) with `b8f` ahead on best. The
champion checkpoint is preserved in [`../hallOfFame/`](../hallOfFame/README.md).

### The late-checkpoint hypothesis: confirmed for supply, mixed for quality

The previous version of this section flagged as speculative that "the horizon may have been
truncating the best checkpoints of good arms". Re-measurement supports it, but not uniformly, and
the distinction matters:

| | corr(step, measured) | 1.0-1.8M | 2.2-2.6M | 2.6-3.0M |
|---|---|---|---|---|
| `b8f` | **+0.61** | ~45% | **64.5%** | 63.6% |
| `b8d` | **-0.11** | 59.5% | 60.3% | 54.0% |

**What is solid is the supply of good checkpoints, not per-checkpoint quality.** In thirteen
hours `b8f` went from 16 checkpoints at >=80% to **63**, and `b8d` from 4 to 25. Both arms' best
checkpoints sit at ~2.55M, and every previous record-holder was stopped at 1.06M — before that
region existed.

**Per-checkpoint quality rises with steps for `b8f` (+0.61) and not for `b8d` (-0.11)**, whose
late band is slightly worse. So "train longer" is not a law. Note also that this correlation is
computed only over checkpoints that already cleared the 80% filter, which restricts the range and
understates any true relationship.

The counter-evidence from before still stands: `b7d` ran to 1.60M at 0.995 and peaked at 26%,
`b7a` reached 2.00M with a 19% ceiling. Long runs do not rescue a mediocre arm.

**Practical rule: do not stop a healthy arm at ~1M steps.** Both records came from territory the
old horizon forbade.

### The horizon has an upper bound too: peak ~2.5-3M, dead by ~7M

Followed to the end, both arms traced the same four-phase arc. `b8d` ran to **11.6M steps** — the
longest run in the project by more than 2x — and died:

| phase | steps | `b8f` perfect (per 1M) | `b8d` perfect (per 1M) |
|---|---|---|---|
| climb | 0-2M | 17.2% → 30.1% | 6.8% → 15.4% |
| **peak** | **~2.5-3M** | **40.9%** | **27.4%** |
| decline | 3-6M | 18.6% → 7.4% → 10.1% | 14.6% → 11.9% → 0.3% |
| death | 7M+ | — | **0.0%** for 4.5M steps |

Both arms' best measured checkpoints (2581k, 2538k) and best 30-eval windows (2828k, 2671k) fall in
the peak band. `b8d`'s last perfect game was at 5496k, 6.1M steps before it was still running.

**So the practical horizon is ~3-3.5M steps**, not 1M and not unlimited. The ~8.5M steps `b8d` spent
after its peak produced nothing measurable. That the decline ends in death rather than a plateau
also means a past-peak arm is not merely unproductive — it is on its way to zero.

#### Corollary: a sudden jump in step rate is a symptom of death

`b8d` advanced **7.3M steps in ~24 hours** while `b8f` managed 1.9M on the same machine. Almost all
of that gap is that **a dead policy plays very short episodes** — the snake dies immediately — so it
burns training steps several times faster than a competent one.

Never read step rate as progress. This is the same confound that once made eval cost look like a
config difference, and it now has a second use: an arm that suddenly starts advancing much faster
than its sibling is probably dying, not accelerating.

## Falsified: `GRADIENT_CLIPPING=10` does not buy stability

Clipping went in as a cheap independent stability aid on top of `DISCOUNT=0.995`, on the
reasoning that the 10.0 terminal reward produces occasional huge gradients and that clipping
them would prevent the catastrophic drops. After three seeds it is **1 of 3**, against **3 of
3** for plain 0.995:

| arm | peak trailing | best 30-eval pf | best measured | outcome |
|---|---|---|---|---|
| `b8d-disc995clip` | **86.9** | **50.0%** | **80.0%** (58.4% pooled) | peaked ~2.7M, declining at 3.48M |
| `b8e-clipseed2` | 85.9 | 21.3% | 32.0% (1 ckpt) | faded; stopped at 1.16M |
| `b8g-clipseed3` | 77.0 | 30.0% | **none >50%** | dead; stopped at 3.43M |

**It was briefly this file's headline, off `b8d` at 163k steps.** That reading — "the fastest
riser on record", 36.0% best-30 by 163k against `b7f`'s 699k — was wrong twice over. `b8d`'s
own early window was followed by a near-total collapse (0.4% mean perfect across 300-600k) and
everything durable came after 600k, so it was not a head start. And the two seeds that followed
did not reproduce it.

**The "raises the ceiling" escape hatch is now closed too.** `b8d` measured 62.0% best / 48.3%
pooled, which looked like a unique ceiling gain — until `b8f` measured **63.0% / 46.5% without
clipping**, with overlapping intervals. Re-measurement 13 hours later widened the gap the other
way: **`b8f` 88.0% / 59.2% against `b8d` 80.0% / 58.4%**, still tied on pooled but with the
non-clipped arm ahead on ceiling. Clipping shows **no measured benefit and a worse survival
record**. Do not adopt it.

Recording the process error, because it is the recurring one: that ceiling claim was written
while `b8d` was measured and `b8f` was not, off the arm that happened to finish first. A
two-arm comparison graded from one arm is not a comparison. Wait for both.

## An arm recovered from 1.2M steps at zero — and then died anyway

`b8g-clipseed3` sets both records at once, which is why it is worth its own section:

| block | mean trailing | mean perfect |
|---|---|---|
| 0-300k | 52.7 | 8.7% |
| 600-1800k | **1.7 - 14.7** | **0.0%** |
| **2100-2400k** | **63.7** | **4.3%** |
| 2700-3600k | **0.0** | 0.0% |

**The recovery.** 1.2M steps near zero, then back to 63.7 trailing and a 4.3% perfect rate. The
previous record was ~400k steps. Any stop rule that would have killed this arm at 1M steps —
including the one this project used for most of its life — was wrong on this case.

**The death.** It then collapsed and spent its final 900k pinned at 0.0. So the recovery bought
nothing in the end, and an arm that has completed a recovery arc can still be finished.

The rule that survives both halves: **read `zero_since` against the current step, and require
both a long pinned stretch and no recovery in progress.** `b8g` would satisfy that at 2625k
onward and would not at 1M. Two prior errors in this file — calling `b6b` permanently damaged
and calling `b7b` merely oscillating — were the two directions of getting this wrong, and
`b8g` is the case that contains both.

## Engineering facts worth not rediscovering

- **Importance-sampling weights must stay mean-normalized.** cpprb normalizes by the
  largest weight in the whole buffer, so raw batch weights average 0.087 at beta=0.4 and
  0.0027 at beta=1.0 — a silent 11x-370x cut to the learning rate that worsens as beta
  anneals. `normalize_is_weights()` fixes this; don't remove it. (Applies only when
  `IS_WEIGHTS=1`.)
- **`legacy.Adam` is not faster here** despite TF's M1/M2 warning: 0.809 ms/step vs 0.721
  ms for the modern optimizer. Ignore the warning.
- **Throughput is ~230-240 steps/s** for one run on an idle machine, and roughly holds up
  with 4 runs sharing 14 cores. That affects wall-clock only, not learning per step.
- **cpprb is ~2.4x faster than `PyUniformReplayBuffer`** with no measured learning cost.
- **The "upgrade to Gymnasium" warning is inert.** It costs a few log lines and the
  upgrade is unavailable; do not propose it.
