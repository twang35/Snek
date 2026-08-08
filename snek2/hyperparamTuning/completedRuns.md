# Completed runs

Every arm that has finished: config, final numbers, verdict. The table below is **canonical** —
it covers every arm ever run, including the batches whose narratives moved to
[`archive/batches1-11.md`](archive/batches1-11.md).

Companion to [`runs.md`](runs.md) (what is running), [`findings.md`](findings.md) (conclusions) and
[`charts.md`](charts.md) (graphs, batch 12 onward). Nothing here should be re-run without a reason.

## All arms, ranked by best sustained perfect rate

**Use `best ckpt` for cross-arm comparison.** The `measured` column is a pooled figure whose
selection rule changed several times, so the episode count is spelled out per row (`/6300`, `/100`)
and rows with different counts are different statistics — `b8e` has 1 checkpoint behind its figure,
`b8f` 63. Read `measured` as a within-arm consistency check instead: best and pooled close together
means a strong *region*, which is the property being chased.

**Every graph-derived column misranks arms** — `b5c` is 2nd by best perfect-30 and last by
measurement. Read `best perfect-30` as description, not ranking.

**`best perfect-30` is no longer the primary metric.** From 2026-08-04 that is
`strong_eval_fraction`, the share of an arm's evals at >=80%, which has ~40% lower between-seed
variance; see [`hyperparamTuning.md`](hyperparamTuning.md#the-primary-metric-strong_eval_fraction-the-share-of-an-arms-evals-at-80).
Best perfect-30 stays in this table because every arm through batch 11 is recorded on it.

**‡ marks arms measured on the post-audit environment (2026-08-02 onward)**, whose numbers are
not comparable to any row without the mark. The audit changed two observation components and the
reward; the same checkpoint that scored 92% before reads 73% after. Compare ‡ rows only to each
other, or to `b8f`'s `3149000` re-measured at 82%.

**‡‡ marks arms trained on the environment after 2026-08-02's seven further fixes**
(fatal-move zeroing, wall/body hugging, normalized group count, the corrected starve/length
split, the terminal-discount fix, safe-to-chase-food, and the audit that started the day) — a
third, later environment than ‡. Batch 9 (‡) predates all seven; batch 10 (‡‡) is the first to
train on the result. Compare ‡‡ rows only to each other; they are not comparable to ‡ or
unmarked rows.

**‡‡‡ marks arms trained on the 30-value vector (2026-08-03 onward)** — a fourth environment,
adding the following-tail block (26-28) and food-space (29). Batches 11-15 all train on it,
and it is still current. ‡‡‡ rows compare to each other and, with care, to ‡‡ rows: batch 11 is
byte-identical to batch 10 in config, so that one cross-era comparison is *designed* and is written
up in [`archive/batches1-11.md`](archive/batches1-11.md). Everything else stays within its own era.
Batch 10's checkpoints stopped loading on `master` when this landed.

**§§ in the `measured` column marks a run measured under an abandonment gate**, whose figure is
`pooled_equal_effort` rather than the graph-100% tier — a gate censors the tier, and
`pooled_equal_effort` is exact regardless. Gates by batch: none through 13, **90** for 14 and 15,
**95** from 16. ***trunc*** in `best ckpt` means no full-length row survived the gate, so the figure
comes from a shorter row and reads noisier.

**§ marks arms trained after the epsilon rewrite of 2026-08-04.** Not an environment change —
the observation vector is untouched and every ‡‡‡ checkpoint still loads — but a config change,
and a large one: every arm above ran at epsilon exactly 0.0 for 96.8% of its steps, and § arms
explore for a meaningful fraction of the run.

**Every metric here is a greedy eval, so § rows are comparable to ‡‡‡ rows in exactly the way
batch 11 was comparable to batch 10 — as a controlled test of the one thing that changed.** An
earlier draft of this note claimed best-checkpoint survived the change while best-30 did not;
that was wrong, and the distinction does not exist. Epsilon only shapes the data, so it changes
*which policy you get at step N*, and every column then differs — but all of them still measure
the greedy policy's quality, which is what makes the comparison meaningful rather than broken.
What is not legitimate is treating a § arm as another seed of the ‡‡‡ baseline: it is a different
config, and pooling the two would hide the change instead of measuring it.

**Four environments in two days is the real cost being paid here**, and it is worth stating
plainly: every environment change resets the comparison set, so a batch's numbers are only ever
readable against its own siblings. That is why seed count inside a batch matters more than the
number of knobs tried across batches — see the note at the end of [`runs.md`](runs.md).

| policy | config change | final steps | best ckpt | top-3 | **measured** | best perfect-30 | verdict |
|---|---|---|---|---|---|---|---|
| `b17b-forkseed2` ‡‡‡ § | **forking on**, disc 0.9975, shaping off | 1.57M | **95.2%** @1190k /600 | 93.5% /500 | **82.42%** /eq §§ | **92.7%** | ‡‡‡ § **the record**, re-measured: selected reads of 99/100 fell to 92.4-95.0 over 500. Reached ~95% at **1.19M** |
| `b15b-nstep3seed2` ‡‡‡ § | **n=3**, disc 0.995, shield 0.8 | 5.75M | 97% @3245k | 96.3% | 79.7% /eq §§ | 89.3% | ‡‡‡ § previous best selected ckpt, but **93.0% over 300** |
| `b11b-obs30seed2` ‡‡‡ | disc 0.995, **fourth env** | 3.56M | **96%** @855k | **95.3%** | **81.0%** /10400 † | **91.7%** | ‡‡‡ 96/100 selected, ~94% shrunk |
| `b14a-disc9975seed1` ‡‡‡ § | disc **0.9975**, shield 0.8 | 4.17M | 96% @3702k | 93.0% | 72.4% /eq §§ | 79.7% | ‡‡‡ § 96/100 selected; 91/100 on re-measure, **93.5% over 200** |
| `b15a-nstep3seed1` ‡‡‡ § | **n=3**, disc 0.995, shield 0.8 | 5.79M | 95% @4697k | 94.3% | 77.7% /eq §§ | **89.7%** | ‡‡‡ § **peak trailing 95.00, highest on record**; still gaining at 5.8M |
| `b13d-shieldseed4` ‡‡‡ § | + eps handover 0.0125, shield 0.5 | 3.51M | **95%** @986k | 93.3% | 77.2% /4800 † | 83.3% | ‡‡‡ § **2nd best ckpt on record**; peaked ~1M then lost 44 pp |
| `b14d-disc9975seed4` ‡‡‡ § | disc **0.9975**, shield 0.8 | 4.46M | 93% @2559k | 93.0% | **77.6%** /eq §§ | **89.7%** | ‡‡‡ § **best arm on record for `strong_eval_fraction`, 39.3%**; only 11.7 pp drawdown |
| `b17c-forkseed3` ‡‡‡ § | **forking on**, disc 0.9975, shaping off | 1.52M | 93% @1424k *trunc* | 92.6% | 73.33% /eq §§ | 82.0% | ‡‡‡ § **still climbing when stopped** — peak and best window in its last 140k |
| `b13c-shieldseed3` ‡‡‡ § | + eps handover 0.0125, shield 0.5 | 3.67M | 92% @3367k | 90.7% | 75.9% /11400 † | **85.3%** | ‡‡‡ § best of batch 13 on the graph; best ckpt in its final 300k |
| `b13b-shieldseed2` ‡‡‡ § | + eps handover 0.0125, shield 0.5 | 3.70M | 91% @1166k | 90.3% | 74.8% /7800 † | 82.3% | ‡‡‡ § fastest start on record: 72.3% pf30 by 350k |
| `b14b-disc9975seed2` ‡‡‡ § | disc **0.9975**, shield 0.8 | 4.12M | 90% @2261k | — | 67.1% /eq §§ | 76.3% | ‡‡‡ § weakest of batch 14; **one full-length row survived the gate**; lost 47 pp |
| `b14c-disc9975seed3` ‡‡‡ § | disc **0.9975**, shield 0.8 | 4.16M | 90% @2099k | 89.3% | 71.3% /eq §§ | 87.7% | ‡‡‡ § **still improving at 4.5M** — the arm that moved the step cap to 10M |
| `b15d-nstep3seed4` ‡‡‡ § | **n=3**, disc 0.995, shield 0.8 | 5.81M | 91% @3671k | 90.3% | 73.5% /eq §§ | 86.3% | ‡‡‡ § **peak trailing at 5799k, its 2nd-to-last eval** — stopped mid-climb |
| `b13a-shieldseed1` ‡‡‡ § | + eps handover 0.0125, shield 0.5 | 3.39M | 80% @2044k | 80.0% | 70.5% /3100 † | 78.0% | ‡‡‡ § weakest of batch 13, and the batch's slowest starter |
| `b17d-forkseed4` ‡‡‡ § | **forking on**, disc 0.9975, shaping off | 1.51M | 86% @1260k *trunc* | 84.5% | 66.83% /eq §§ | 75.7% | ‡‡‡ § fastest starter of batch 17; best window at 679k then flat |
| `b15c-nstep3seed3` ‡‡‡ § | **n=3**, disc 0.995, shield 0.8 | 5.46M | 86% @3823k *trunc* | — | 66.4% /eq §§ | 75.7% | ‡‡‡ § weakest of batch 15; **no full-length row survived the gate** — best is a truncated 69/80 |
| `b11a-obs30seed1` ‡‡‡ | disc 0.995, **fourth env** | 3.19M | 94% @671k | 93.3% | 79.5% /4800 † | 85.7% | ‡‡‡ 2nd of batch 11; peaked at 678k, then lost 42 pp |
| `b11d-obs30seed4` ‡‡‡ | disc 0.995, **fourth env** | 3.59M | 88% @3507k | 86.7% | 69.3% /4000 † | 78.3% | ‡‡‡ only arm still near peak when stopped |
| `b11c-obs30seed3` ‡‡‡ | disc 0.995, **fourth env** | 3.23M | 87% @1706k | 84.7% | 69.0% /2300 † | 73.0% | ‡‡‡ weakest of batch 11 |
| `b17a-forkseed1` ‡‡‡ § | **forking on**, disc 0.9975, shaping off | 1.41M | 68% @916k *trunc* | 66.7% | 46.25% /eq §§ | 54.0% | ‡‡‡ § **the arm that made batch 17 a null**; never reached ε ≤ 0.003; 22 episodes is its deepest row |
| `b12d-eps002seed4` ‡‡‡ § | eps handover **0.05**, no shield | 1.09M | not measured | — | — | 6.3% | ‡‡‡ § **deadlocked**; abandoned at 1M of 2.5M |
| `b12a-eps002seed1` ‡‡‡ § | eps handover **0.05**, no shield | 1.12M | not measured | — | — | 6.3% | ‡‡‡ § deadlocked; peaked 87.0 trailing @214k then decayed to 55 |
| `b12c-eps002seed3` ‡‡‡ § | eps handover **0.05**, no shield | 0.98M | not measured | — | — | 1.7% | ‡‡‡ § deadlocked; 8 perfect games in 977 evals |
| `b12b-eps002seed2` ‡‡‡ § | eps handover **0.05**, no shield | 1.03M | not measured | — | — | **0.0%** | ‡‡‡ § deadlocked; **zero perfect games in 1032 evals** |
| `b12s-shield05seed1` ‡‡‡ § | eps handover **0.05** + shield 0.5 | 0.43M | not measured | — | — | 0.3% | ‡‡‡ § the probe that moved the handover; decay fixed, plateau not |
| `b10d-disc995seed4` ‡‡ | disc **0.995**, third env | 4.45M | **93%** @1695k | **93.3%** | **74.9%** /66000 | 84.3% | ‡‡ best of batch 10; held the record until `b11b` |
| `b10b-disc995seed2` ‡‡ | disc **0.995**, third env | 4.65M | 90% @1501k | 86.3% | 71.8% /62400 | 85.0% | ‡‡ 2nd of batch 10 |
| `b10a-disc995seed1` ‡‡ | disc **0.995**, third env | 4.29M | 85% @2344k | — | 67.2% /27200 | 78.3% | ‡‡ stopped healthy |
| `b10c-disc995seed3` ‡‡ | disc **0.995**, third env | 4.12M | 79% @3965k | — | 63.0% /4700 | 72.7% | ‡‡ weakest of batch 10 |
| `b9d-disc995b` ‡ | disc **0.995**, new env | 3.45M | **70%** @2544k | **66.3%** | 42.4% /1700 | 30.3% | ‡ best ceiling of batch 9 |
| `b9a-disc9975a` ‡ | disc **0.9975**, new env | 3.68M | 65% @1735k | 64.3% | **54.9%** /2000 | 56.0% | ‡ most consistent of batch 9 |
| `b9c-disc995a` ‡ | disc **0.995**, new env | 3.71M | 52% @2603k | 51.3% | 38.0% /2000 | 37.3% | ‡ weakest survivor |
| `b9b-disc9975b` ‡ | disc **0.9975**, new env | **10.47M** | not measured | — | — | 5.0% | ‡ **dead**, peaked at 328k |
| `b8f-disc9975seed2` | alpha 0.6, `td_loss`, no IS, **disc 0.9975** | 5.47M | **92%** | **86.7%** | **66.3%** /5200 | **69.3%** | **project record**; declining when stopped |
| `b8d-disc995clip` | disc 0.995 + **`GRADIENT_CLIPPING=10`** | **11.64M** | **80%** | 74.7% | 60.4% /2000 | 50.0% | 2nd by measurement, then **died at ~7M** |
| `b7f-disc995seed3` | alpha 0.6, `td_loss`, no IS, **disc 0.995** | 1.06M | 51% | 48.0% | 38.8% /1000 | 44.0% | best of batch 7, and survived |
| `b4c-schlongper` | alpha 0.8, `td_loss`, no IS | 1.06M | 50% | 46.7% | 37.1% /1000 | 34.0% | ties `b7f` on ceiling, **1 of 3 seeds survive** |
| `b7e-disc995seed2` | alpha 0.6, `td_loss`, no IS, **disc 0.995** | 1.28M | 39% | 34.7% | 29.5% /1000 | 32.3% | strong, survived |
| `b6b-alpha06` | alpha 0.6, `td_loss`, no IS | 1.80M | — | — | 24.5% /1000 | 21.7% | old selector, **underestimate**; re-measure |
| `b7d-discount995` | alpha 0.6, `td_loss`, no IS, **disc 0.995** | 1.60M | 26% | 22.7% | 16.4% /1000 | 17.7% | survived, weakest of the three discount seeds |
| `b8e-clipseed2` | disc 0.995 + **`GRADIENT_CLIPPING=10`** | 1.16M | **32%** | — | 32% /100 | 21.3% | one good ckpt, **no good region** — 1 above the floor |
| `b7a-a06seed2` | alpha 0.6, `td_loss`, no IS | 2.00M | 19% | 18.3% | 12.0% /1000 | 15.0% | survived to 2M, low ceiling |
| `b6a-alpha04` | alpha 0.4, `td_loss`, no IS | 1.41M | — | — | 8.1% /1000 | 14.3% | stable, never near death, low ceiling |
| `b5d-schlongTDE` | alpha 0.8, `td_error`, no IS | 2.07M | — | — | 6.6% /1000 | 10.7% | stable, low ceiling |
| `b5c-schlongIS` | alpha 0.8, `td_loss`, **IS on** | 2.31M | — | — | 2.1% /1000 | 17.0% | IS correction cancels the benefit; peak ckpt evicted |
| `b8c-disc9975` | alpha 0.6, `td_loss`, no IS, **disc 0.9975** | 1.75M | — | — | not measured | 14.7% | monotone decline to a stop; `b8f`'s sibling |
| `b8g-clipseed3` | disc 0.995 + **`GRADIENT_CLIPPING=10`** | 3.43M | — | — | **none >50%** | 30.0% | **died, recovered after 1.2M, died again** |
| `b7b-a06seed3` | alpha 0.6, `td_loss`, no IS | 1.78M | — | — | 0% | 7.7% | **died at 1162k** |
| `b7c-a06seed4` | alpha 0.6, `td_loss`, no IS | 1.74M | — | — | 0% | 9.7% | **died at 573k** |
| `b8a-disc999` | alpha 0.6, `td_loss`, no IS, **disc 0.999** | 1.11M | — | — | 0% | 0.7% | **died at 452k**, peak trailing only 63.1 |
| `b8b-disc999seed2` | alpha 0.6, `td_loss`, no IS, **disc 0.999** | 1.41M | — | — | 0% | 0.0% | **zero perfect games in 1.41M steps** |
| `b1a-base` | none (control) | 503k | — | — | — | 16.7% | collapsed at 265k; score recovered, skill did not |
| `b3a-epsfloor` | `MIN_EPSILON=0.001` | 545k | — | — | — | 11.0% | best of batch 3, degraded anyway |
| `b4b-unifbuf500k` | alpha 0 + 500k buffer | 1.23M | — | — | — | 9.3% | steadiest arm, but a low ceiling |
| `b4a-uniform` | alpha 0 | 1.25M | — | — | — | 8.7% | peaked ~575k, drifted down |
| `b3b-epsfloor2` | `MIN_EPSILON=0.001` | 549k | — | — | — | 8.3% | declined despite the floor — falsified hypothesis A |
| `b2a-base2` | none (repeat) | 999k | — | — | — | 7.0% | the 1M-step reference: no collapse, long oscillation, 1.1% at the end |
| `b3c-buf500k` | 500k buffer, alpha 0.6 | 4.81M | — | — | — | 5.7% | **died at ~750k**, score 0.0 for 4M steps |
| `b5a-schlong` | alpha 0.8, `td_loss`, no IS | 2.05M | — | — | — | 10.0% | **died at 272k**, `b4c` repeat |
| `b5b-schlong2` | alpha 0.8, `td_loss`, no IS | 1.92M | — | — | — | 7.7% | **died at 246k**, `b4c` repeat |
| `b1c-nstep3` | `N_STEP_UPDATE=3` | 1.14M | — | — | — | 1.7% | dead end |
| `b1b-tgt200` | `TARGET_UPDATE_PERIOD=200` | 106k | — | — | — | 1.0% | stopped early, verdict weak |
| `b2b-nstep2` | `N_STEP_UPDATE=2` | 580k | — | — | — | 0.7% | dead end |

**† batches 11 and 13's `measured` column is the graph-100% tier only, not the whole arm.** Batch 10 was
measured flat — every selected checkpoint at 100 episodes — so pooling all its rows is already an
equal-effort figure. Batch 11 ran the three-stage protocol, where the rows have different depths and
the deep ones were chosen *because* they screened well, so pooling them reads high by construction.
The equal-effort fix is to truncate every checkpoint to its first 20 episodes, and that is not
recoverable here: the output file stores per-checkpoint totals, not per-episode results, and these
four runs predate the `pooled_equal_effort` field that computes it in-process. What *is* clean in
both batches is the graph-100% tier — 100 episodes each, no screening applied in either — so that is
what these four rows report, and it is the column the batch 11 vs batch 10 comparison below uses.
The episode counts are correspondingly smaller (`/4800` against batch 10's `/27200`).

Batch 13 ran the same three-stage protocol and *does* have `pooled_equal_effort` (68.7 / 71.6 / 73.2
/ 74.8 across seeds 1-4). Its rows still report the graph-100% tier so the column means one thing
down the table and so the batch 11 comparison is like-for-like; the equal-effort figures are in the
batch 13 write-up below.

**§§ batch 14's `measured` column is `pooled_equal_effort` (`/eq`), because the graph-100% tier does
not exist for it.** Batch 14 is the first batch measured under `EVAL_MIN_ACHIEVABLE=90`, and the gate
truncates every tier member that falls below 90%, leaving the tier's own top slice — it reads 90.3%
against batch 13's 74.6% and means nothing. `pooled_equal_effort` truncates to the 20-episode screen
depth, which is at or below the abandon floor, so it is exact at any gate and is the **only** column
here that compares cleanly to batch 13 (72.07% against 72.08%). When comparing `measured` across the
† and §§ rows, don't — compare the batch write-ups, which state both figures.

**§§ Batch 17's gate is 95, the same as batch 16's**, so those two batches' `best ckpt` columns are
comparable to each other and to nothing above them. `b17b` is the exception in both directions: it is
the only arm in either batch with full 100-episode rows, and it has 26 of them.

**‡ `b17b-forkseed2` displaces `b15b` at the top of this table on every column at once** — best ckpt
99% vs 97%, top-3 98.7% vs 96.3%, eq-effort 82.42% vs 79.7%, best-30 92.7% vs 89.3% — and it did it at
**1.57M steps against 5.75M**. Its re-measurement is outstanding, so read the 99% as provisional; the
eq-effort figure is not provisional, and it is a record on its own.

`train` was a human-started run on committed defaults, stopped by the human. Never
touch `snek2/savedPolicies/train*`.

Four things this ranking makes visible that per-batch reading did not:

- **`DISCOUNT=0.995` ties the best ceiling and keeps it.** `b7f` (51%) and `b4c` (50%) are
  a dead heat, but `b4c`'s config dies in 2 of 3 seeds while all three discount seeds lived.
  Priced for survival that is 28.2% against 12.4% expected.
- **Everything above ~12% shares one config family**: alpha 0.6-0.8, `td_loss` priorities,
  no IS weights. The discount is the only addition that has helped on top of it.
- **`b1a-base`, a plain baseline, outranks four deliberate interventions**
  (`MIN_EPSILON`, both n-step values, the 500k buffer with PER). Most changes tried in
  this investigation made things worse.
- **The graph misranks arms badly.** `b5c-schlongIS` is 2nd of the batch-5/6 arms by best
  perfect-30 (17.0%) and **last by measurement** (2.1%). Any ranking built on 10-episode
  graph evals is unreliable; see [`hyperparamTuning.md`](hyperparamTuning.md).
- **Dying late is not the same as dying.** `b8g-clipseed3` sits below arms with a third of its
  best perfect-30 because it ended dead, but it spent 1.2M steps at zero and *recovered* to
  63.7 trailing before collapsing again. A ranking by endpoint hides that entirely — see
  [`findings.md`](findings.md).

**The top two rows were measured mid-run at ~2.6-2.9M steps**, not at the final steps shown, because
both arms kept training after measurement and then declined. Those are their best figures, and their
record checkpoints are preserved in [`../hallOfFame/`](../hallOfFame/README.md).

**Read the `final steps` column with care — it is not a quality signal.** `b8d`'s 11.64M is the
largest number in this table and it died; the same arm's best checkpoint came at 2538k. Both top
arms peaked at ~2.5-3M and were stopped well past it. Everything below them was stopped before
~2.1M, and the four next-best at ~1.06M, so **this ranking compares most configs at a horizon where
they had not finished improving** — see [`findings.md`](findings.md).

## Batch 17 — forked endgame collection: **a null that produced the project record**

**Ran 2026-08-07 19:55 to 2026-08-08 00:45, four arms stopped by hand at 1.41-1.57M, measured
2026-08-08.** `SNEK_FORK_BRANCHES=4 SNEK_FORK_PROB=0.5 SNEK_FORK_MIN_LENGTH=85
SNEK_FORK_MAX_STEPS=60` on batch 16's config exactly — same four seeds, same discount 0.9975, same
`GUIDED_FRACTION=0.8`, shaping off in both, `N_STEP_UPDATE=1`. **Batch 16 is the seed-matched control
and differs only by `FORK_BRANCHES=1`.** The first change in this project aimed at the *collect
distribution* rather than the optimiser or the reward. Graphs in
[`charts.md`](charts.md#batch-17--forked-endgame-collection-snek_fork_-a-null-stopped-at-141-157m).

| seed | step | peak trailing | best-30 | `sef` | best ckpt | eq-effort |
|---|---|---|---|---|---|---|
| 1 | 1245k | 93.86 @1144k | 54.0% @1166k | **1.3%** | 68.2% @916k †22ep | **46.25%** |
| 2 | 1245k | 94.92 @1068k | **92.7%** @1223k | **32.0%** | **99.0%** @1205k **100ep** | **82.42%** |
| 3 | 1245k | 94.38 @1230k | 71.3% @954k | 9.9% | 93.5% @1424k †92ep | 73.33% |
| 4 | 1245k | 94.60 @1007k | 75.7% @679k | 18.6% | 86.0% @1260k †50ep | 66.83% |
| **mean** | | **94.44** | **73.4%** | **15.4%** | | **67.21%** |

**Closed out 2026-08-08 at `EVAL_MIN_ACHIEVABLE=95`, matching batch 16's protocol exactly.** An
earlier launch at 90 was stopped after ~4 minutes with no completed rows and restarted at 95: when the
whole point is a paired comparison against batch 16, running the *same* protocol as the control beats
running a better one. † marks a truncated row; `b17b`'s is the only full 100 in the batch, and there
are 26 of them.

### The pre-registered comparison, at a matched 1.245M horizon

`b16a` stopped at 1245k, so that is the common horizon — 1246 evals per arm on both sides. Exact
paired permutation over all 16 sign flips.

| metric | b16 (fork off) | b17 (fork on) | delta | p |
|---|---|---|---|---|
| **`strong_eval_fraction`** (primary) | **17.11%** | **15.45%** | **-1.67 pp** | 0.875 |
| `best_perfect30` | 79.42% | 73.42% | -6.00 pp | 0.875 |
| mean perfect, back half | 62.22% | 56.37% | -5.85 pp | 0.750 |
| mean perfect | 42.80% | 42.07% | -0.73 pp | 1.000 |
| peak trailing | 94.71 | 94.44 | -0.27 | 0.250 |
| mean trailing | 85.13 | 80.16 | -4.97 | 0.250 |
| max drawdown | 66.92 pp | 73.97 pp | +7.04 pp | 0.500 |
| steps to pf30 ≥ 40% | 424k | 402k | **-21.8k** | 0.750 |

**The primary metric points the wrong way and nothing is significant.** The only direction worth
anything is speed — 22k earlier to pf30 ≥ 40%, 2 of 4 seeds — which is the metric with the largest
per-seed spread here (+110k, -68k, +7k, -136k).

### ‡ Unlike batch 16, the sweep is flat, so this is not a horizon artifact either way

| horizon | `sef` delta | `best30` delta |
|---|---|---|
| 300k | +0.00 pp | +4.67 pp |
| 500k | +2.30 pp | +4.17 pp |
| 700k | +3.67 pp | +6.83 pp |
| 900k | +0.97 pp | -6.00 pp |
| 1.0M | -1.15 pp | -9.75 pp |
| **1.245M** | **-1.67 pp** | **-6.00 pp** |

Positive and rising to ~700k, then negative and stable. Batch 16's signal grew monotonically to its
horizon; this one reverses, which is what a noise effect looks like. **The 900k interim read called it
a null for the same reason it later stayed a null, by luck** — the 500-700k slice would have called it
a win.

### ‡ One seed is the whole result, and that has to be said in both directions

| metric | all four seeds | dropping `b17a` |
|---|---|---|
| `sef` | **-1.67 pp** | **+4.23 pp** |
| `best_perfect30` | -6.00 pp | +3.03 pp |
| mean perfect, back half | -5.85 pp | +4.07 pp |

`b17a-forkseed1` is -19.3 pp on `sef` and -33.0 on best-30; the other three are +1.7 / -0.6 / +11.6.
**This is not a licence to drop it** — it is the honest statement that n=4 cannot separate "forking
destabilises some seeds" from "seed 1 drew badly." Two facts argue for the second: `b13a` failed the
same way with no forking at all, and `b17a`'s fork counters are entirely normal (9683 forks, 27%
share, retired == created). It is also the only arm of the eight that **never reached ε ≤ 0.003**,
where all seven others did between 373k and 974k — its schedule is gated on a trailing-30 perfect
rate it never sustained, so the instability and the high epsilon feed each other. That is a
[failure mode](failureModes.md), not a forking mechanism.

### The mechanism engaged and held for 1.5M steps — this is a real null, not a null test

| arm | forks | branch share | truncated / ended | eligible skipped, slots full |
|---|---|---|---|---|
| `b17a` | 9,683 | 27% | 5,725 / 3,958 | 11,670 |
| `b17b` | 8,614 | 24% | 5,730 / 2,881 | 12,228 |
| `b17c` | 10,268 | 29% | 6,537 / 3,731 | 13,772 |
| `b17d` | 10,544 | 29% | 6,691 / 3,853 | 12,921 |

Inside the pre-registered 25-60% band, `forks == retired` on all four (no leaked branch envs, RSS flat
at 496-659 MB), `main_steps + branch_steps == global_step` exactly, and 0 violations of the buffer
integrity invariant.

**But the delivered dose was below design, and that is the one loose end.** Branch share landed at
**24-29% against the predicted ~46%**, and 12-14k eligible fork points per arm were skipped for want
of a slot — roughly **30% of all eligible points**. The **cap of 4 is what binds, not `FORK_PROB=0.5`**.
So the finding is "counterfactual coverage at endgame decision points does not help *at this dose*",
and `SNEK_FORK_BRANCHES=6-8` is the version of the experiment that would actually test the hypothesis.
The batch nonetheless landed inside its own pre-registered band, so by its own terms it is a fair null.

### ‡ The close-out: `b17b` is the best policy this project has measured, by a clear margin

| arm | rows | episodes | saved | eq-effort | deepest | rows ≥90% | rows ≥95% | best row |
|---|---|---|---|---|---|---|---|---|
| `b17a` | 20 | 440 | 1,560 | **46.25%** | **22 ep** | 0 | 0 | 68.2% @916k |
| `b17b` | **306** | **13,078** | 10,642 | **82.42%** | 100 ep (**26 of them**) | **69** | **25** | **99.0% @1205k** |
| `b17c` | 117 | 3,662 | 8,038 | 73.33% | 92 ep | 6 | 0 | 93.5% @1424k |
| `b17d` | 131 | 3,096 | 9,524 | 66.83% | 50 ep | 0 | 0 | 86.0% @1260k |

**`pooled_equal_effort` 82.42% is the highest ever recorded here**, against `b15b`'s 79.7%, and it is
the one column that cannot be inflated by selection — it truncates every checkpoint to the 20-episode
screen depth, so it is exact at any gate.

### ‡ Re-measured the same day over 6,600 fresh episodes: the 99/100 was mostly selection

The close-out produced 26 rows at the full 100 episodes, **25 of them ≥95%**, mean 96.2% — which looked
like a ~96% region rather than a lucky max. **It was not.** Four processes, `EVAL_MIN_ACHIEVABLE=0`,
no screening:

| step | close-out | re-measured /500 | change |
|---|---|---|---|
| 1190k | 98/100 | **95.0%** (92.7-96.6) | -3.0 |
| 1205k | **99/100** | **92.4%** (89.7-94.4) | **-6.6** |
| 1231k | 98/100 | 92.6% (90.0-94.6) | -5.4 |
| 1248k | **99/100** | 93.8% (91.3-95.6) | -5.2 |
| **mean of the ≥98% group** | **98.5%** | **93.45%** | **-5.05** |
| mean of the six 97% rows, /200 | 97.0% | **95.25%** | -1.75 |

**The ranking inside the top rows was noise**: the ≥98% group re-measures *below* the 97% group. And a
**blind grid** — 17 checkpoints every 10k across 1110-1270k, chosen by position, 100 fresh episodes
each — prices the selection effect directly:

| sample of the same region | mean | pooled | CI |
|---|---|---|---|
| position-chosen grid | **84.06%** | 1429/1700 | 82.2-85.7 |
| the close-out's selected full-length rows | **96.2%** | — | — |
| `b16b` control grid, 840-1000k | **80.41%** | 1367/1700 | 78.5-82.2 |

**Three results, in order of how much they matter:**

1. **The reasoning that defended the 99 was circular** — the 26 deep rows reached full depth *because*
   they screened well, so their mean is inflated by the same mechanism as the max. Full account in
   [`runs.md`](runs.md#-the-reasoning-error-a-selected-sample-cannot-defend-itself-against-selection).
2. **The record is real but small.** `b17b` @1190k pools to **95.17% over 600 fresh episodes** (CI
   93.1-96.6) against `b14a`'s 93.5%/200 and `b15b`'s 93.0%/300 — better, overlapping intervals, and
   reached at **1.19M steps** against 3.7M and 3.2M. **The speed is the unambiguous part.**
3. **The region beats its control by +3.65 pp** (84.06% vs 80.41%) on position-chosen samples — which
   independently reproduces this arm's `pooled_equal_effort` delta of **+3.34 pp**. Two different
   protocols, same answer, and it is the same ~+3.5 pp that seeds 3 and 4 showed.

**And the region is not a plateau.** `1140k` reads **12.0%** on the blind grid — a complete collapse
30k steps from a 95% checkpoint, invisible to the close-out because it never screened well enough to be
selected. Excluding it the grid mean is 88.56%, and quoting that instead would be the same selection
error one level up, so the 84.06% stands as the region figure.

**And `b17a`'s close-out is nearly worthless, which is worth recording as a protocol result**: 20 rows,
deepest 22 episodes, 440 episodes total. The gate is arithmetic and correct — that arm has no
checkpoint that could reach 95% — but it means the batch's mean eq-effort (67.21%) is one real
measurement and three shallow ones plus one near-empty one.

### The paired close-out comparison, and why it says the opposite of the mean

| seed | b16 eq-effort | b17 eq-effort | delta |
|---|---|---|---|
| 1 | 76.89% | 46.25% | **-30.64** |
| 2 | 79.08% | **82.42%** | **+3.34** |
| 3 | 69.67% | 73.33% | **+3.66** |
| 4 | 63.27% | 66.83% | **+3.56** |
| **mean** | **72.23%** | **67.21%** | **-5.02** (p=1.000) |

**Seeds 2, 3 and 4 move the same way by nearly the same amount** — +3.34, +3.66, +3.56, a spread of
0.32 pp on a metric whose between-seed sd is several points. That is a more consistent signal than any
graph metric in this batch produced. And seed 1 reads -30.64, which swamps it.

**Do not resolve this by dropping seed 1.** Two readings survive the data: forking gives ~+3.5 pp and
seed 1 failed for unrelated reasons, or forking destabilises some fraction of seeds and seed 1 is that
fraction. **n=4 cannot separate them**, and that — not the sign of the mean — is the batch's real
finding. The re-run that could separate them is `SNEK_FORK_BRANCHES=6-8` at n=8.

### What it establishes, and what it does not

- **Do not adopt forking on this evidence.** The primary metric is negative at p=0.875.
- **Do not call the hypothesis falsified either.** At 30% of eligible points skipped, the dose is
  ~60% of design, and three of four seeds moved the right way.
- **`b17b-forkseed2` is the most interesting single arm this project has produced on the graph**:
  `best_perfect30` **92.7%**, the highest of any arm ever run here, at **1.22M steps** against
  `b11b`'s 91.7% at 3.56M and `b15a`'s 89.7% at 5.79M — and its full-length peak trailing reads
  **95.00**, tying `b15a` for the highest on record. `best_perfect30` is a max statistic with sd 8.6
  across identical seeds, which is exactly why this file demoted it, so **one arm is a curiosity and
  not evidence**. Whether it holds a champion checkpoint is the close-out's job.
- **The ceiling still has not moved.** Peak trailing 94.44 against 94.71 / 95.00 / 94.90 / 94.80 for
  batches 16 / 15 / 14 / 13 — seven batches, everything inside 0.6 points.

## Batch 16 — the food-distance shaping ablated: **the first non-null in six batches**

**Ran 2026-08-07 09:5x to ~19:00, four arms stopped by hand at ~1.25M, measured 2026-08-07.**
`SNEK_FOOD_DISTANCE_REWARD=0` on batch 14's config exactly — same four seeds, same discount 0.9975,
same `GUIDED_FRACTION=0.8`, same `td_loss` + `IS_WEIGHTS=0`, `N_STEP_UPDATE=1` in both. **One live
variable and no reliance on a previous null**, which no batch since 13 has managed. Graphs in
[`charts.md`](charts.md#batch-16--food_distance_reward0-the-shaping-term-ablated-stopped-at-125m).

| seed | step | peak trailing | best-30 | `sef` | best ckpt | eq-effort |
|---|---|---|---|---|---|---|
| 1 | 1245k | 94.82 @837k | 87.0% @850k | 20.6% | 93.3% @838k †90ep | 76.89% |
| 2 | 1261k | **94.98** @816k | 85.0% @919k | **30.5%** | 93.3% @913k †90ep | **79.08%** |
| 3 | 1257k | 94.36 @1198k | 72.7% @1221k | 10.5% | 85.0% @979k †40ep | 69.67% |
| 4 | 1256k | 94.68 @946k | 73.0% @1032k | 7.0% | 80.0% @1010k †30ep | 63.27% |
| **mean** | | **94.71** | **79.4%** | **17.2%** | | **72.23%** |

**† Every "best ckpt" here is a truncated measurement, not a full 100 episodes** — see the gate
section below, which is a protocol result in its own right.

### The paired comparison, at a matched 1.25M horizon

Batch 16 stopped at ~1.25M while batch 14 ran to 4.2M, and `strong_eval_fraction` is a fraction of an
arm's *own* evals — so **every figure below truncates both batches to steps ≤ 1.25M**, giving 1251
evals per arm on both sides. Exact paired permutation over all 16 sign flips.

| metric | b14 (shaping on) | b16 (shaping off) | delta | p |
|---|---|---|---|---|
| **`strong_eval_fraction`** (primary) | **5.80%** | **17.15%** | **+11.35 pp** | 0.250 |
| `best_perfect30` | 66.83% | 79.42% | **+12.58 pp** | **0.125** (4/4 seeds) |
| mean perfect, back half | 51.15% | 62.27% | +11.13 pp | 0.250 |
| mean perfect | 36.52% | 42.87% | +6.35 pp | 0.250 |
| peak trailing | 94.31 | 94.71 | +0.41 | 0.250 |
| mean trailing | 85.96 | 85.15 | -0.82 | 0.625 |
| max drawdown | 65.75 pp | 66.92 pp | +1.18 pp | 0.750 |
| steps to pf30 ≥ 40% | 429k | 424k | **-6k** | 0.875 |

Per-seed `sef`: b14 `5.6 / 6.9 / 2.8 / 7.9` against b16 `20.6 / 30.5 / 10.5 / 7.0`. 0.125 is the
**floor** at n=4 — it means all four seeds moved the same way, which is what `best_perfect30` did.

### ‡ It is not a horizon artifact: the gap widens monotonically

The obvious objection is that 1.25M was chosen because that is where batch 16 stopped. Swept:

| horizon | `sef` b14 | `sef` b16 | delta | `best30` delta |
|---|---|---|---|---|
| 400k | 0.62% | 0.37% | -0.25 pp | +6.33 pp |
| 600k | 1.41% | 2.79% | +1.37 pp | +11.00 pp |
| 800k | 2.81% | 6.49% | +3.68 pp | +6.58 pp |
| 1.0M | 4.45% | 12.61% | +8.17 pp | +13.50 pp |
| **1.25M** | **5.80%** | **17.15%** | **+11.35 pp** | **+12.58 pp** |

A noise effect would wander; this one is absent at 400k and grows at every step after. That is the
signature of arms consolidating earlier, not of a lucky slice.

### What it establishes, and what it does not

**The mechanism is consolidation, not speed and not the ceiling** — and that is exactly the
pre-registered "handover unchanged, level up" outcome:

- **Not speed.** Steps to pf30 ≥ 40% is flat (-6k, p=0.875), and the epsilon handover was flat too
  (11.5k against 12.5k, three of four seeds in the same eval).
- **Not the ceiling.** Peak trailing +0.41 with batches 11-15 holding 94.8-95.0 anyway.
- **It is time spent near the top.** `sef` is literally the share of evals at ≥80% perfect, and it
  roughly **tripled**. Arms without the shaping reach a high perfect rate at the same moment and then
  *hold* it, where the shaping-on arms oscillate back down.

That reading fits the mechanism proposed before launch: 0.001 per retreating move is a small
permanent tax on exactly the detours a 93% endgame requires, so it does not stop an arm learning to
win — it stops it staying there.

**What this is not:** established. One batch at n=4, p=0.125-0.250, and **the arms were stopped at
1.25M**, so whether the advantage holds, closes or reverses past that point is untested — batch 14's
own curve was still climbing there and its full-run `sef` is 21.53% against batch 16's 17.15% at 1.25M.
It needs a replication at a longer horizon before it goes into [`findings.md`](findings.md) as more
than a lead. **Do not restore the shaping** in the meantime: nothing here argues for it, and batch 17
onward runs with it off.

### ‡ The 95% gate saved 65% of the work and cost the ability to rank checkpoints

A protocol result, and the first close-out run at `EVAL_MIN_ACHIEVABLE=95`:

| arm | rows | episodes | vs flat 100 | deepest row | rows ≥90% | rows ≥95% |
|---|---|---|---|---|---|---|
| b16a | 127 | 4,680 | 36.9% | 100 ep (one row) | 9 | **0** |
| b16b | 196 | 7,080 | 36.1% | 90 ep | 15 | **0** |
| b16c | 45 | 1,390 | 30.9% | **40 ep** | 0 | **0** |
| b16d | 26 | 760 | 29.2% | **30 ep** | 0 | **0** |
| **total** | 394 | **13,910** | **35.3%** | | 24 | **0** |

The saving beat the predicted 31%. The cost is that **no arm produced a full-length measurement worth
ranking**: b16c's best is 34/40 and b16d's 24/30, so their headline numbers carry ±15 pp intervals, and
`best_full_length_row`'s half-depth fallback is now the normal path rather than the exception it was
written as. Two consequences:

- **Cross-batch best-checkpoint comparison is dead at this gate.** Batch 14's bests are 100-episode
  measurements; batch 16's are 30-90 episode ones. Use `pooled_equal_effort` (exact at any gate) or the
  training-graph metrics, which do not touch the eval protocol at all.
- **The gate is calibrated above the population.** Zero of 394 rows reached 95% and only 24 reached
  90%, so at this skill level a 95 gate is not selecting good checkpoints — it is declining to measure
  almost all of them. See
  [`hyperparamTuning.md`](hyperparamTuning.md#measuring-a-policy-properly-eval_checkpointspy) for the
  standing recommendation that follows.

## Batch 15 — `N_STEP_UPDATE=3`: falsified on speed, null on level, and a 97/100 that is really 93%

**Ran 2026-08-06 16:26 to 2026-08-07 08:22 (15.9 h), four arms to 5.79M / 5.75M / 5.46M / 5.81M,
measured 2026-08-07.** `N_STEP_UPDATE=3` on batch 13's config with `GUIDED_FRACTION=0.8`; control is
batch 14, which shares the 0.8 and differs only by a discount that is a measured null. Graphs in
[`charts.md`](charts.md#batch-15--n_step_update3-falsified-on-speed-and-the-longest-arms-on-record).

| seed | best ckpt | eq-effort | pf30≥40% at | best30 | SEF | full rows | ≥95% rows |
|---|---|---|---|---|---|---|---|
| 1 | 95% @4697k | 77.67% | 620k | 89.7% | 39.9% | 63 | 1 |
| 2 | **97%** @3245k | **79.68%** | 524k | 89.3% | 39.0% | 94 | **8** |
| 3 | 86% @3823k † | 66.40% | 707k | 75.7% | 9.4% | **0** | 0 |
| 4 | 91% @3671k | 73.53% | 378k | 86.3% | 33.8% | 6 | 0 |
| **mean** | **92.3%** | **74.32%** | **557k** | **85.3%** | **30.5%** | | **9** |

**† `b15c` produced no full-length row at all** and is the first arm to exercise
`best_full_length_row`'s half-depth fallback in production — its best is a 69/80 row abandoned by the
90% gate, printed `[truncated]`. Working as designed, and a preview of what the 95% gate makes normal.

### The design: why n=3, and which control

Both earlier n-step arms (`b1c-nstep3`, `b2b-nstep2`) are **retracted rather than negative** — they
trained on returns that summed straight through episode boundaries, because the per-step discount is
the only truncation in `r_t + g·d_t·r_{t+1} + …` and terminal steps carried 0.9975 until 2026-08-02.
Fixed at `snake_environment.py:126`.

n=3 over n=2 because the usual reason to keep n small — an uncorrected n-step return is only exact
if the intermediate actions were greedy — is negligible at this project's epsilon. Measured mean
epsilon over the back half of each arm: **0.0034** (batch 13), **0.0039** (batch 14).

| epsilon | P(non-greedy)/step | contaminated at n=2 | at n=3 |
|---|---|---|---|
| 0.0125 (refinement ceiling) | 0.83% | 0.83% | 1.66% |
| **~0.004 (what arms actually run at)** | **0.27%** | **0.27%** | **0.53%** |
| 0.002 (floor) | 0.13% | 0.13% | 0.27% |

So n=2 → n=3 moves contamination from ~0.3% to ~0.5% of targets while the upside scales with n:
propagating the +100 perfect-game reward back across a ~1780-step perfect game takes ~890 sequential
backups at n=2 against ~593 at n=3. Add that n=4 arms resolve only a clear win, and n=3 is Rainbow's
value and so the only choice with outside evidence behind it. The unresolved counter: priorities come
from the n-step TD error, so larger n feeds larger-magnitude errors into `td_loss` + alpha 0.6 — it
argues both ways, since bigger errors push Huber into its linear region.

Config, and the control it implies:

```
SNEK_SEED=1..4  SNEK_DISCOUNT=0.995  SNEK_GUIDED_FRACTION=0.8  SNEK_N_STEP_UPDATE=3
SNEK_PRIORITY_EXPONENT=0.6  SNEK_PRIORITY_SIGNAL=td_loss  SNEK_IS_WEIGHTS=0
```

| control | differs from batch 15 by | usable? |
|---|---|---|
| **batch 14** | discount (0.9975 → 0.995) **and** n | **yes** — the discount is measured null, so n is the only live variable |
| batch 13 | `GUIDED_FRACTION` (0.5 → 0.8) **and** n | weaker — `GUIDED_FRACTION` has never been isolated |
| batches 13+14 pooled, n=8 | both | for the primary, since 13 and 14 are indistinguishable |

**‡ The residual risk this rests on**, stated because it still applies to any batch using batch 14 as
a control: the discount null was measured *at* `GUIDED_FRACTION` 0.8 vs 0.5, so it is a null on the
pair, not on each knob independently. Nothing suggests they interact — at epsilon ~0.003 the shield
touches ~0.1% of steps — but "0.995 ≡ 0.9975" is not separately established at fixed
`GUIDED_FRACTION=0.8`.

### The pre-registered primary failed; the evals are a null

Pre-registered before the batch ran: n-step's predicted effect is faster credit propagation to the
same asymptote, so the primary is **steps to pf30 ≥ 40%**, `strong_eval_fraction` secondary, and a
higher ceiling *without* an earlier arrival would be a surprise worth writing up rather than a win.

| seed | b14 (control) | b15 | delta |
|---|---|---|---|
| 1 | 639k | 620k | -19k |
| 2 | 227k | **524k** | **+297k** |
| 3 | 530k | **707k** | **+177k** |
| 4 | 320k | 378k | +58k |
| **mean** | **429k** | **557k** | **+128k slower**, p=0.250 |

Note how little room seeds 2 and 4 had: 227k and 320k are close to the floor the bootstrap ladder
itself sets, so a real speed gain would have shown up on the slow seeds. Batch 13's per-seed figures
on the same metric, for the pooled n=8 baseline: 1525k / 246k / 807k / 179k, mean **689k**.

| metric | b14 | b15 | delta | p |
|---|---|---|---|---|
| **steps to pf30 ≥ 40%** (primary) | **429k** | **557k** | **+128k slower** | 0.250 |
| best ckpt | 92.25% | 92.30% | +0.05 pp | 1.000 |
| `pooled_equal_effort` | 72.08% | 74.32% | +2.24 pp | 0.625 |
| `strong_eval_fraction` (eq. effort) | 21.53% | 25.59% | +4.05 pp | 0.625 |
| mean perfect | 53.64% | 55.76% | +2.13 pp | 0.625 |
| mean perfect, back half | 62.39% | 63.01% | +0.62 pp | 1.000 |
| `best_perfect30` | 82.50% | 83.17% | +0.67 pp | 0.875 |
| mean trailing | 90.15 | 89.55 | -0.60 | 0.875 |
| peak trailing | 94.74 | 94.64 | -0.10 | 0.875 |
| drawdown | 23.75 pp | 20.42 pp | -3.33 pp | 1.000 |

n-step's whole predicted mechanism was faster credit propagation, and it came out **slower**, 3 of 4
seeds. Best checkpoint lands within 0.05 pp. **`N_STEP_UPDATE=5` is closed**, not queued: the case for
a larger n was that the effect scales with n, so its absence at n=3 is the worst possible sign.
Contamination cannot be blamed either — at the measured epsilon an n=3 return is exact for 99.5% of
targets — so there is no predicted cost to attribute the null to.

**‡ The 280k interim read was half wrong, in the same direction batch 14's was.** It reported the
perfect rate down 11.9 pp; at equal effort the batch finished **up** 2.13 pp. What survived is the
early deficit it was actually measuring — that is the +128k on the primary. So: **an early snapshot of
level reverses, an early snapshot of timing does not. Read crossings early, read levels late.**

**Full-length numbers look much better than the equal-effort ones, and the gap is the point.** `b15a`
set a peak trailing score of 95.00 and a `strong_eval_fraction` of 39.9%, both records — but `sef` is
a fraction of an arm's *own* evals and these arms ran 1.3-1.7M steps longer than batch 14's while
still playing at 70-80% perfect. At matched steps the advantage is +4.05 pp at p=0.625. **The arms
sustained a high level for 5.8M steps; they did not reach a better one.**

### ‡ The 97/100 does not survive re-measurement, and the ≥95% count is mostly noise

`b15b` @3245000 read **97/100**, which would have been a project record. Re-measured over **200 fresh
episodes it read 182/200 = 91.0%**, pooling to **279/300 = 93.0%** (CI 89.5-95.4). Every champion this
project has selected behaves the same way:

| checkpoint | selected | corrected | how |
|---|---|---|---|
| `b11b` @855k | 96/100 | ~94.0% | shrunk against a Beta prior |
| `b14a` @3702k | 96/100 | 93.5% (CI 89.2-96.2) | re-measured, 187/200 |
| `b15b` @3245k | **97/100** | **93.0%** (CI 89.5-95.4) | re-measured, 279/300 |

**So the record has not moved. Three batches have produced the same ~93-94% policy.**

Batch 15 has **9 rows at ≥95%** against batch 11's 3, batch 13's 1 and batch 14's 1, and it is
tempting to read that as more near-perfect checkpoints. It is mostly the right tail of a population
that is not near-perfect. `b15b`'s 94 full-length rows have **mean 90.7% and median 90.0%**, and for a
population centred at 90% the expected number of 100-episode measurements reading ≥95 by noise alone
is ~5.4 per 94 rows. Observed: 8.

The practical consequence, which matters for the gate: **`EVAL_MIN_ACHIEVABLE=95` does not find 95%
policies, it finds ~90% policies caught on a lucky 100 episodes.** Only re-measurement separates the
two, and it should be run on any checkpoint before it is called a record.

### What batch 15 did establish

- **`b15b` is the strongest arm ever measured on the arm-level figure** — `pooled_equal_effort`
  79.68%, against a previous best of 77.55% (`b14d`). That is a real result and it is not a
  selection artifact, since the figure truncates every checkpoint to 20 episodes.
- **The horizon is longer than three batches of docs assumed.** Final 500k band against each arm's
  best prior band:

  | arm | final band (5.5-6.0M) | best prior | |
  |---|---|---|---|
  | `b15a` | **80.6%** | 80.4% | still gaining |
  | `b15d` | **75.8%** | 72.8% | still gaining |
  | `b15b` | 62.3% | 78.5% | past peak |
  | `b15c` | 53.7% | 59.7% | past peak |

  `b15d`'s peak trailing score is at **5799k — its second-to-last eval**. Stopping at 5.8M truncated
  two arms mid-climb; the 10M cap was not the wrong setting, the habit of stopping at a round number
  is. `b15a` and `b15d` are the top resume candidates on record.
- **The ceiling has not moved.** Peak trailing reads 94.92 / 94.80 / 94.90 / 95.00 across batches
  11 / 13 / 14 / 15 — flat inside 0.2 points while run length went 3.2M to 5.8M. Full-length
  `strong_eval_fraction` rising 19.5 → 21.6 → 30.5% is arms spending longer at a good level, not
  reaching a better one.
- **The seed spread is again larger than any effect.** `b15a` at 39.9% `sef` against `b15c` at 9.4%,
  same config, adjacent seeds; and `b15c` is the only arm in five batches to produce no full-length
  row at all.

## Batch 14 — disc 0.9975 at guided 0.8, and the widest seed spread yet

**Ran 2026-08-05 19:34 to 2026-08-06 08:12 (12.6 h), four arms to 4.17M / 4.13M / 4.16M / 4.46M,
stopped by hand ~2 h short of the 5M cap, measured 2026-08-06.** `DISCOUNT=0.9975`,
`GUIDED_FRACTION=0.8`, otherwise batch 13's config. Graphs in
[`charts.md`](charts.md#batch-14--discount09975-at-guided_fraction08-and-a-third-null).

| seed | best ckpt | top-3 | eq-effort | best30 | SEF | drawdown | final eps |
|---|---|---|---|---|---|---|---|
| 1 | **96%** @3702k | 93.0% | 72.4% | 79.7% | 20.0% | 25.0 pp | 0.0036 |
| 2 | 90% @2261k | — † | 67.1% | 76.3% | 9.3% | **47.0 pp** | 0.0064 |
| 3 | 90% @2099k | 89.3% | 71.3% | 87.7% | 17.8% | **5.7 pp** | 0.0020 |
| 4 | 93% @2559k | 93.0% | **77.6%** | **89.7%** | **39.3%** | 11.7 pp | 0.0021 |
| **mean** | **92.3%** | **91.3%** | **72.1%** | **83.3%** | **21.6%** | **22.3 pp** | |

**† `b14b` produced exactly one full-length row**, so it has no top-3. That is the 90% abandonment
gate, not the arm: every other checkpoint was truncated once 90% became unreachable. Batch 14 is the
first batch measured under a gate, and the counts are 9 / 1 / 4 / 47 full-length rows against batch
13's 131 / 178 / 214 / 148.

### Null against batch 13 on every metric that survives the gate

Paired by seed, exact permutation over all 16 sign flips:

| metric | batch 13 | batch 14 | diff | p | per-seed diffs |
|---|---|---|---|---|---|
| `pooled_equal_effort` | 72.07% | 72.08% | **+0.01 pp** | **1.000** | +3.7, -4.5, -1.9, +2.8 |
| best ckpt | 89.5% | 92.3% | +2.75 pp | 1.000 | +16.0, -1.0, -2.0, -2.0 |
| top-3 mean | 88.6% | 91.3% | +2.75 pp | 1.000 | +13.0, -0.3, -1.3, -0.3 |
| `best_perfect30` | 82.2% | 83.3% | +1.08 pp | 0.625 | +1.7, -6.0, +2.3, +6.3 |
| `strong_eval_fraction` | 19.5% | 21.6% | +2.05 pp | 1.000 | +8.3, -16.2, -8.7, +24.8 |
| drawdown | 19.3 pp | 22.3 pp | +3.00 pp | 1.000 | +18.7, +31.3, -6.3, -31.7 |

**`pooled_equal_effort` landing within 0.01 pp is the cleanest null this project has measured.** It
is also the metric to trust here: it truncates every checkpoint to its first 20 episodes, and
abandonment cannot fire before the 20-episode floor, so it is exact under the gate where the others
are not.

**The best-ckpt gain is one seed and mostly winner's curse.** +2.75 pp is +16.0 on seed 1 and
-1 to -2 on the other three. `b14a`'s 96/100 @3702k tied the record on paper; re-measured
independently at 100 episodes it read **91/100**, pooling to **187/200 = 93.5%** (CI 89.2-96.2). It
was the maximum over 176 attempted full-length measurements in its own arm, and this is what that
costs.

**‡ The graph-100% tier is unusable for this batch and is deliberately absent from the table above.**
It reads 90.3% against batch 13's 74.6%, a +15.6 pp "win" that is pure artifact: the gate truncates
every tier member below 90%, so what remains is the tier's own top slice. Tier sizes went from 31-114
checkpoints per arm to 1-28. See
[`hyperparamTuning.md`](hyperparamTuning.md#taking-the-arm-level-pooled-rate).

### What batch 14 actually established

- **`DISCOUNT=0.9975` is not better than 0.995 on this vector.** The pre-registered argument was that
  the 2026-08-03 endgame observations need a longer horizon than 0.995's ~200 steps. Measured, it
  makes no difference, and this is now the second batch to say so (batch 9 said it on the old vector,
  which was the reason to re-ask). **Do not re-ask at n=4.**
- **The interim read at 1.3M was wrong, and the mechanism it proposed is falsified.** At 1.3M
  `strong_eval_fraction` was -7.8 pp and the story was variance compression — 0.9975 raising the floor
  and lowering the ceiling. Back-half within-arm sd of the perfect rate is **18.4 in both batches**
  (13: 18.7/18.5/16.9/19.3, 14: 18.9/20.1/19.4/15.4). There is no compression. The early number was a
  peak-counting metric read before the peaks had arrived.
- **Arms do not reliably peak by 3.4M**, which moved `SNEK_MAX_STEPS` to 10M. `b14a`'s best trailing
  window is at 3.79M, `b14c`'s best perfect window at 4.14M, and `b14c`'s final 4.0-4.5M band is its
  best of the run. Every earlier batch was stopped by hand near 3.5M, so the old rule of thumb partly
  measured the stopping habit.
- **`GUIDED_FRACTION=0.8` is still unmeasured.** It moved with the discount, so this batch cannot
  attribute anything to it. Since the discount result is a null, the confound is now moot rather than
  resolved — there is nothing to attribute.
- **Seed instability is worse than previously stated.** `strong_eval_fraction` per-seed diffs run
  -16.2 to +24.8 pp around a +2.05 pp mean. `b14d` is the best arm on the primary the project has
  recorded (39.3%, previous best 30.5%) and `b14b` among the worst (9.3%) — same config, adjacent
  seeds.

## Batch 13 — the epsilon rewrite plus the exploration shield: an exact null

**Ran 2026-08-05, four arms to 3.4-3.7M, stopped healthy, measured 2026-08-05.** Same environment
and config as batch 11 (‡‡‡, 30-value vector) except the epsilon schedule: handover 0.0125,
`GUIDED_FRACTION=0.5`. Graphs in
[`charts.md`](charts.md#batch-13--the-lower-handover-plus-the-shield-and-an-exact-null).

| seed | best ckpt | top-3 | eq-effort | graph-100% tier | best30 | SEF | final eps |
|---|---|---|---|---|---|---|---|
| 1 | 80% @2044k | 80.0% | 68.7% | 70.5% /3100 | 78.0% | 11.5% | 0.0024 |
| 2 | 91% @1166k | 90.3% | 71.6% | 74.8% /7800 | 82.3% | 25.4% | 0.0027 |
| 3 | 92% @3367k | 90.7% | 73.2% | 75.9% /11400 | **85.3%** | **26.5%** | 0.0023 |
| 4 | **95%** @986k | 93.3% | **74.8%** | 77.2% /4800 | 83.3% | 14.5% | 0.0050 |
| **mean** | **89.5%** | **88.6%** | **72.1%** | **74.6%** | **82.2%** | **19.5%** | |

### The pre-registered comparison: null on all five metrics

Paired by seed against `b11a`-`b11d`, exact permutation test over all 16 sign flips:

| metric | batch 13 | batch 11 | diff | p | per-seed diffs |
|---|---|---|---|---|---|
| best ckpt | 89.5% | 91.2% | -1.8 pp | 0.875 | -14.0, -5.0, +5.0, +7.0 |
| top-3 mean | 88.6% | 90.0% | -1.4 pp | 0.875 | -13.3, -5.0, +6.0, +6.7 |
| graph-100% tier | 74.6% | 74.7% | **-0.1 pp** | **1.000** | -9.0, -6.1, +6.9, +7.9 |
| `best_perfect30` | 82.2% | 82.2% | **+0.0 pp** | **1.000** | -7.7, -9.4, +12.3, +5.0 |
| `strong_eval_fraction` | 19.5% | 17.5% | +2.0 pp | — | -6.7, +2.5, +16.9, -4.7 |

**Two independently-computed metrics landing on a mean difference of 0.1 pp or less is not
coincidence, it is what a genuinely zero effect looks like measured five ways.** The per-seed spread
is what to read instead: every metric has one seed at roughly -10 pp and another at roughly +7 pp.
Post-peak drawdown is the same story, 20.0 vs 21.0 pp at p = 0.875.

This confirms the null hypothesis stated in advance: batch 11's near-zero epsilon regime was not
costing anything, and no version of this schedule has beaten it.

**The seeds swapped ranks wholesale**, which is worth internalising before designing another n=4
batch. Seed 3 went from batch 11's weakest arm (87% best ckpt) to 92%; seed 1 went from 94% to 80%.
Same config, same environment, same seed number — the only difference is which random stream the
epsilon draws consumed. **A seed is not a stable unit of quality**, so pairing by seed removes less
variance than it looks like it should.

### What the rewrite did buy, which is not nothing

**The schedule now works as designed.** Epsilon descended on skill to 0.0023-0.0050 in all four
arms, so the refinement phase reached its intended range instead of pinning at a ceiling. The two
real defects are fixed and stay fixed: exactly-0 is unreachable, and the one-way ratchet is gone, so
a collapsing arm buys exploration back — `b13d` reads 0.0050 at stop against its siblings' 0.0024,
which is the anti-ratchet responding to its 44 pp drawdown exactly as intended.

**The 350k abandon condition passed 4/4** — `b13b` was at trailing 92.4 with a 72.3% perfect rate by
step 350k, where every batch-12 arm was at 0%. Whatever the rewrite is worth, the deadlock is gone.

### What it did not buy, stated plainly

- **No outcome gain, on five metrics.** Three batches of work on epsilon have produced one deadlock
  and one null measured five independent ways. The `min_epsilon` floor and the anti-ratchet are worth
  keeping on mechanism; the elevated exploration is not worth pursuing further at n=4.
- **No new record.** `b13d`'s 95% @986k is the second-best checkpoint on record and does not beat
  `b11b`'s 96%. It is in [`../hallOfFame/`](../hallOfFame/README.md) because a 95% on the current
  environment is worth keeping, not because it changed the ceiling.
- **The shield is unproven.** It fixed batch 12's decay at handover 0.05 (see
  `b12s-shield05seed1`), but at 0.0125 there is no decay to fix and nothing separates it from batch
  11. It is confounded with the handover change here and cannot be isolated at this sample size.
- **The drawdown is not improved.** -1.0 pp at p = 0.875, with per-seed diffs of -35.1 and +38.7.
  `b13d` gave up 44.3 pp where batch 11's seed 4 gave up 5.6, so the seeds simply swapped places.

## Batch 12 — the deadlock, abandoned at ~1M of 2.5M

`b12a-eps002seed1`, `b12b-eps002seed2`, `b12c-eps002seed3`, `b12d-eps002seed4`, stopped
2026-08-04. All four cleared the pre-registered abandon condition together, so the batch was
called early rather than run to its horizon. Charts and per-arm readings in
[`charts.md`](archive/charts-archive.md#batch-12--the-epsilon-rewrite-and-the-deadlock-it-found).

Note for anyone stopping an arm: `SIGTERM` and `SIGINT` are both swallowed by the trainer — there
is no signal handler in `training.py` — so it takes `SIGKILL`. Checkpoints and `_evals.json` are
rewritten every 1000 steps, so at most a partial interval is lost, but copy the four
`_evals.json` files aside first if they hold the only record of something.

### ‡ The new schedule deadlocks. All four arms are failing, 4/4, at ~1M steps

Epsilon descends out of bootstrap correctly and then **pins at the refine ceiling 0.05 forever**,
because the refine phase descends on mastery that the level of exploration it is holding prevents
the agent from acquiring.

| arm | step | trailing | pf30 now | eps | pinned at 0.05 for | evals with a perfect game |
|---|---|---|---|---|---|---|
| `b12a` | 1.02M | 55.5 | 0.0% | 0.05 | 686k steps | 41 / 1022 |
| `b12b` | 0.95M | 57.8 | 0.0% | 0.05 | **942k steps** | **0 / 954** |
| `b12c` | 0.92M | 60.4 | 0.0% | 0.05 | 409k steps | 8 / 918 |
| `b12d` | 1.02M | 61.7 | 0.0% | 0.05 | 455k steps | 32 / 1021 |

Against the control at the same step, on the pre-registered primary metric:

| | `strong_eval_fraction` @1.01M | trailing @1.01M | pf30 @1.01M |
|---|---|---|---|
| batch 11 | 25.2 / 30.5 / 0.0 / 8.2% | 84.5-88.5 | 14.0-82.3% |
| batch 12 | **0.0% ×4** | **54.9-61.7** | **0.0% ×4** |

**The numbers above are greedy, and the comparison is clean.** Evals run `agent.policy`, which
TF-Agents builds as `GreedyPolicy`; only `agent.collect_policy` is the `EpsilonGreedyPolicy` that
`epsilon` feeds. Verified two ways — `_setup_policy` in the installed `dqn_agent`, and empirically
with `epsilon=1.0`, where `agent.policy` returns one action on a fixed observation across 60 calls
while `collect_policy` returns all three. So there is **no exploration tax on the metric**: pf30 = 0
is a real property of the greedy policy, and batch 11 and batch 12 are measured the same way.

**The mechanism is a learning deadlock, not a measurement one.** At eps 0.05, 3.3% of *collected*
actions are random, and a random move with a long snake is usually fatal — so the replay buffer
fills with trajectories that die before the endgame and the agent never sees the states a perfect
game is made of. The greedy policy therefore never masters the endgame, greedy pf30 stays 0,
`refine_epsilon(0, top=0.05, floor=0.002)` returns exactly `top`, and the collection distribution
never improves. The loop closes through the *policy*:

```
eps 0.05 → 3.3% random collected actions → buffer lacks endgame states
        → greedy policy cannot finish → pf30 = 0 → refine returns the ceiling → (repeat)
```

Batch 11's crude ladder always escaped because it descended on step count, which no policy can
suppress.

**The descent is also far too shallow to escape even with luck.** 0% → 6.3% pf30 (`b12a`'s
best-ever window) moves epsilon only 0.0500 → 0.0388. Meaningful relief needs 20-40% pf30, which
is exactly what 0.05 makes impossible: `pf30=10%` → 0.0334, `20%` → 0.0224, `40%` → 0.0100.

**Sustained high epsilon degrades a policy that was already working.** `b12a` read greedy trailing
**87.0** and pf30 6.3% at step 214k, then decayed to 55.5 over the next 800k steps at the same
epsilon. All four peak at 81-87 by step 214-479k and then decline. Both numbers are greedy, so this
is the learned policy getting worse, not a measurement artefact.

This clears the pre-registered abandon condition (a >10 pp drop on the primary metric) 4/4 with a
mechanism understood analytically, at 1M of the 2.5M horizon. Running to 2.5M would spend ~5 h
confirming a deadlock that is provable from the code.

The design flaw is general: *any* purely mastery-gated schedule deadlocks if its ceiling sits above
the exploration level at which mastery is achievable.

### The fix: shield the exploration move, not the schedule

Rather than making epsilon decay faster, attack what makes exploration expensive. In a *guided*
episode the epsilon coin's random move is drawn from the moves that do not kill the snake this
step, instead of uniformly from all three. `shielded_policy.py`, wired in as the collect policy.

| decision | value | why |
|---|---|---|
| what is shielded | **the epsilon draw only** | see below — this is the whole design |
| the greedy argmax | **never shielded** | it must eat the -5 and learn |
| `SNEK_GUIDED_FRACTION` | 0.5 | half of refinement-phase episodes |
| when it engages | at the bootstrap handover | nothing to protect while the snake is short |
| `INITIAL_EPSILON` | **unchanged**, 0.4 | the early ladder is the part that works |
| handover | **0.05 → 0.0125** | two rungs added below; see the smoke result |
| guaranteed-descent envelope | **not added** | judge the lower ceiling on its own first |

**Only exploration is shielded, never the greedy action.** Overriding a fatal *greedy* move would
mean `Q(s, a_fatal)` never gets updated toward `DEATH_REWARD` in the states where the network is
wrong, so those values would drift on generalisation alone — and evals run unshielded, so the arm
would walk into walls it was never allowed to learn about. Shielding exploration only removes the
tax while keeping every death the policy earns itself.

**The mask was already in the observation.** Indices 6-8 are "is the move safe (not body or wall)",
per action, and `state_helpers.body_and_wall_collisions` already handles the case a naive check gets
wrong: the cell the tail is vacating is safe to enter. So the shield needs **no environment change
and no new game logic**, just `obs[6:9]`.

**It is one step deep, deliberately.** Snake's hard problem is sealing itself into a region it
cannot escape, and that is untouched — an arm still has to learn it. All this removes is "the coin
flipped and the snake drove into its own body".

**The shield turns off if an arm collapses**, because `guided_fraction_for` is stateless in the same
way `epsilon_for` is: one rule, "shielded iff refining". An arm back in the bootstrap band is
relearning to survive, which is where dying is informative.

**Verified before launch.** 19 tests in `tests/test_shielded_policy.py`, all 9 mutants of the mask
and schedule logic caught; 237 tests total, 0 failures.

### ‡ The shield alone is not enough — this is why the handover moved too

A smoke run at the batch-12 config, `SEED=1` so it pairs with `b12a`, shield on, handover still at
0.05. Mean trailing score per 50k band:

| band | smoke shielded | `b12a` unshielded | `b11a` near-zero eps |
|---|---|---|---|
| 200-250k | 79.6 | 83.8 | 84.9 |
| 300-350k | 83.3 | 77.5 | 89.5 |
| 350-400k | **82.8** | **74.2** | **90.9** |
| pf30 @350k | 0.3% | 0.0% | **19.0%** |
| trailing gained per 100k | 4.7 | negative | 11.1 |

**What the shield fixed:** the decay. `b12a` peaked at 214k and fell 83.8 → 74.2; the shielded arm
was still rising at the same step. That is the failure mode that killed batch 12, and it is gone.

**What it did not fix:** perfect games. 2 perfect-game evals in 355 against `b12a`'s 41 by the same
step, and it plateaus at trailing ~83 where the perfect rate is ~0. The curve is steeply nonlinear —
trailing 83 → ~0% perfect, trailing 91 → 19% — so plateauing 8 points low costs everything.

**Why.** A one-step mask prevents blunders, not *self-trapping*, and in a near-full board almost any
deviation seals a region a few moves later. So the collect policy still never finishes a board, the
buffer still holds no trajectories that eat the last ~10 food, and the greedy policy still cannot
learn them. Perfect games are measured greedy, so this was never exploration killing the eval — it
is the buffer missing completed endgames. **The shield makes exploration survivable without making
the endgame completable.**

Hence the handover drop to 0.0125: 0.83% forced non-greedy per step against 3.3%, close to the
regime batch 11 proved. Keep the shield anyway — it costs nothing and removes the decay.

Three things this write-up got wrong on the way, all worth remembering. The perfect rate was
believed to be measured under epsilon, making the controller read its own noise — it is not, evals
are greedy, so the proposed greedy probe episodes were **rejected as solving nothing**. A `-1e9`
masked logit made the boxed-in fallback look redundant, because `tf.random.categorical` shifts each
row by its maximum and so samples an all-masked row uniformly by accident; `-inf` makes the fallback
load-bearing and testable. And the shield's one-step depth was flagged as an acceptable limitation
when it is in fact the binding one.
