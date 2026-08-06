# Completed runs

Every arm that has finished: config, final numbers, verdict. The table below is **canonical** —
it covers every arm ever run, including the batches whose narratives moved to
[`archive/batches1-10.md`](archive/batches1-10.md).

Companion to [`runs.md`](runs.md) (what is running), [`findings.md`](findings.md) (conclusions) and
[`charts.md`](charts.md) (graphs, batch 11 onward). Nothing here should be re-run without a reason.

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
adding the following-tail block (26-28) and food-space (29). Batches 11, 12 and 13 all train on it,
and it is still current. ‡‡‡ rows compare to each other and, with care, to ‡‡ rows: batch 11 is
byte-identical to batch 10 in config, so that one cross-era comparison is *designed* and is written
up in [`archive/batches1-10.md`](archive/batches1-10.md). Everything else stays within its own era.
Batch 10's checkpoints stopped loading on `master` when this landed.

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
| `b11b-obs30seed2` ‡‡‡ | disc 0.995, **fourth env** | 3.56M | **96%** @855k | **95.3%** | **81.0%** /10400 † | **91.7%** | ‡‡‡ **best measured ckpt on record** |
| `b13d-shieldseed4` ‡‡‡ § | + eps handover 0.0125, shield 0.5 | 3.51M | **95%** @986k | 93.3% | 77.2% /4800 † | 83.3% | ‡‡‡ § **2nd best ckpt on record**; peaked ~1M then lost 44 pp |
| `b13c-shieldseed3` ‡‡‡ § | + eps handover 0.0125, shield 0.5 | 3.67M | 92% @3367k | 90.7% | 75.9% /11400 † | **85.3%** | ‡‡‡ § best of batch 13 on the graph; best ckpt in its final 300k |
| `b13b-shieldseed2` ‡‡‡ § | + eps handover 0.0125, shield 0.5 | 3.70M | 91% @1166k | 90.3% | 74.8% /7800 † | 82.3% | ‡‡‡ § fastest start on record: 72.3% pf30 by 350k |
| `b13a-shieldseed1` ‡‡‡ § | + eps handover 0.0125, shield 0.5 | 3.39M | 80% @2044k | 80.0% | 70.5% /3100 † | 78.0% | ‡‡‡ § weakest of batch 13, and the batch's slowest starter |
| `b11a-obs30seed1` ‡‡‡ | disc 0.995, **fourth env** | 3.19M | 94% @671k | 93.3% | 79.5% /4800 † | 85.7% | ‡‡‡ 2nd of batch 11; peaked at 678k, then lost 42 pp |
| `b11d-obs30seed4` ‡‡‡ | disc 0.995, **fourth env** | 3.59M | 88% @3507k | 86.7% | 69.3% /4000 † | 78.3% | ‡‡‡ only arm still near peak when stopped |
| `b11c-obs30seed3` ‡‡‡ | disc 0.995, **fourth env** | 3.23M | 87% @1706k | 84.7% | 69.0% /2300 † | 73.0% | ‡‡‡ weakest of batch 11 |
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

## Batch 11 — the same config on the 30-value vector: no significant difference

**Launched 2026-08-03 22:55, all four stopped 2026-08-04 09:09** by request — 10h14m, 3.19-3.59M
steps per arm (~330k/arm/hour). Four seeds of batch 10's config, byte for byte
(`SNEK_DISCOUNT=0.995 SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0`),
on the 30-value observation vector. The **first seeded batch** in this project — `SNEK_SEED=1..4`,
recorded in each `runs/<policy>.md`. The only difference from batch 10 is the two observations added
2026-08-03 (following-tail at 26-28, food-space at 29), verified by checking that no other
training-relevant commit landed in between.

| policy | final step | peak trailing | best-30 perfect | best-30 @3.185M | drawdown to final |
|---|---|---|---|---|---|
| `b11b-obs30seed2` ‡‡‡ | 3.56M | 94.92 @855k | **91.7%** @873k | **91.7%** | 18.0 pp |
| `b11a-obs30seed1` ‡‡‡ | 3.19M | 94.82 @653k | 85.7% @678k | 85.7% | **42.4 pp** |
| `b11d-obs30seed4` ‡‡‡ | 3.59M | 94.18 @3468k | 78.3% @3468k | 76.3% | 5.6 pp |
| `b11c-obs30seed3` ‡‡‡ | 3.23M | 94.26 @2452k | 73.0% @1718k | 73.0% | 18.0 pp |

### The pre-registered result: not significant

Fixed before launch: best-30 perfect at a **common horizon**, not at each arm's final step. Batch 11
turned out to be the shorter batch, so the horizon is **3.185M**.

| batch | best-30 @3.185M | mean | sd |
|---|---|---|---|
| 10 (26-value) | 74.7 / 84.0 / 62.0 / 84.3 | 76.2% | 10.5 |
| 11 (30-value) | 85.7 / 91.7 / 73.0 / 76.3 | **81.7%** | 8.6 |

**+5.4 pp, SE 6.8, t=0.80 — not significant**, against a pre-registered threshold of ~10 pp. This is
the outcome the design predicted, and it is recorded as a non-result rather than a lean. The two
observations are kept under the pre-registered decision rule: keep unless clearly worse.

`b11b`'s **91.7%** best-30 is the highest any arm in either batch reached, but that is n=1 out of
eight arms and the batch means overlap heavily. Do not promote it to a finding.

### The close-out agrees with the pre-registration, on two more metrics

All four close-outs finished 2026-08-04 — 1.7 to 2.9 hours of eval per arm, 9.1 hours of work in
total across four parallel processes, 67,140 episodes. They add two *measured* comparisons to the
graph-based one, and all three land in the same place: batch 11 ahead by 4-5 pp, none of it
significant at n=4.

| metric | batch 10 | batch 11 | difference | exact p (one-sided) |
|---|---|---|---|---|
| best-30 @3.185M horizon (pre-registered) | 76.2% | 81.7% | **+5.4 pp** | 0.243 |
| graph-100% tier rate, 100 episodes each | 70.6% | 74.7% | **+4.1 pp** | 0.143 |
| best checkpoint, 100 episodes | 86.8% | 91.2% | **+4.5 pp** | 0.157 |

Per arm, paired by rank within its batch:

| rank | batch 11 arm | best ckpt | tier rate | tier n | batch 10 arm | best ckpt | tier rate | tier n |
|---|---|---|---|---|---|---|---|---|
| 1 | `b11b` | **96%** @855k | **81.0%** | 104 | `b10d` | 93% @1695k | 75.7% | 146 |
| 2 | `b11a` | 94% @671k | 79.5% | 48 | `b10b` | 90% @1501k | 72.9% | 142 |
| 3 | `b11d` | 88% @3507k | 69.3% | 40 | `b10a` | 85% @2344k | 68.7% | 47 |
| 4 | `b11c` | 87% @1706k | 69.0% | 23 | `b10c` | 79% @3965k | 65.1% | 7 |

Batch 11 is ahead at every rank on both columns, which shows the +4 to +5 pp more usefully than the
means do: it is not one lucky arm carrying the batch. The pairing is presentational — nothing links
`b11b` to `b10d` beyond each being its batch's best — so this is not a paired test, and the
permutation p-values above are the unpaired ones.

The p-values are exact permutation tests over all 70 ways to split eight arms 4-and-4, which is the
right test here: the unit of analysis is the arm, not the checkpoint, because checkpoints within an
arm share a training trajectory and are nowhere near independent. Pooling 2205 checkpoint
measurements would give a tiny p-value answering a question nobody asked.

**Three metrics agreeing on +4 to +5 pp is worth more than any one of them, and still is not
significance.** They are *not* independent — all three read the same eight training runs, and the
measured pair comes from the same eval data — so this is one piece of evidence looked at three ways,
not three. What the agreement does buy is that the sign is not an artefact of which metric was
chosen, since the graph metric and the measured ones disagree badly in general (`b5c` is 2nd of its
batch by graph and last by measurement). It raises the odds the effect is real and positive rather
than zero; it cannot separate +5 pp from +1 pp, which is the resolution the design lacked from the
start. The pre-registered decision rule stands: the observations are kept.

#### A new best measured checkpoint: `b11b` @855000, 96/100

The highest single-checkpoint measurement in the project, ahead of `b10d`'s 93% @1695000, and
preserved in [`../hallOfFame/`](../hallOfFame/README.md).

**Corrected for the winner's curse it is ~94%.** It is the maximum of 204 full-length measurements
in its own arm, so some of the 96 is luck. Fitting a Beta prior by moment-matching on `b11b`'s 104
*unselected* graph-100% rows — Beta(12.4, 2.9), mean 81.0%, true-rate sd 9.7 pp — shrinks 96% to
**94.0%**. That is a much smaller correction than `b10d`'s 95% → ~87%, for two reasons: the
measurement is 100 episodes rather than 30, and this arm's true-rate spread is wide enough that a
high observation is more plausibly real.

Fitting the prior on the *selected* full-length rows instead would have given 93.5%, and that
number would have been too generous — those rows were chosen for screening well, so their mean is
inflated and the prior pulls less. The unselected tier is the only clean prior available.

**Two of the four arms' best checkpoints came from the graph-90% tier**, `b11b` @855k among them
(its graph point read 90, not 100). In batch 10 it was all four. The pattern in
[`hyperparamTuning.md`](hyperparamTuning.md) holds: the 100% tier is a coverage guarantee, not a
shortlist of champions, and the confirm stage is what actually finds the best checkpoint.

### Two things that did differ, and neither was predicted

**Peaks came far earlier.** Best-30 peak step: 678k / 873k / 1718k / 3468k against batch 10's
3402k / 4547k / 4064k / 1666k — mean **1.68M vs 3.42M**. Three of four batch-11 arms had already
peaked before 1.8M, and then spent 1.5-2.5M further steps declining.

This is not a speed advantage, and an earlier read of mine that said so was wrong. Measured properly
— steps for the rolling best-30 to first reach 70% — batch 10 is if anything *faster*: median 653k
against 913k. The "peaks came earlier" number was conflating *when the best window occurred* with
*how fast the arm learned*, and `b10c` reaching 70% only at 4052k drags batch 10's mean while
leaving its median untouched.

**The decline was worse.** Mean drawdown from best-30 to the final window: **21.0 pp against
16.9 pp**, and `b11a` gave up **42.4 pp** — from 85.7% at 678k to 43.3% at the end. Every arm in
both batches declined, so this is the known post-peak failure mode rather than a new one, but batch
11 hit it sooner and harder. With n=4 and this spread that is a hypothesis, not a finding.

### What this suggests for the next batch

**Three of four arms peaked before 1.8M, so most of a 10-hour run was spent watching arms get
worse.** If that holds, a batch capped near 2M costs a third as much per arm and buys 3x the seeds
in the same wall time — which is the only thing that fixes the ~10 pp resolution problem that made
this batch's result unreadable. n=12 at 2M would detect ~5 pp. That is worth testing before
spending another four slots on a knob whose effect is probably smaller than 10 pp.

