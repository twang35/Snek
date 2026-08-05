# Completed runs

Every arm that has finished: config, final numbers, verdict. The table below is **canonical** —
it covers every arm ever run, including the batches whose narratives moved to
[`archive/batches1-8.md`](archive/batches1-8.md).

Companion to [`runs.md`](runs.md) (what is running), [`findings.md`](findings.md) (conclusions) and
[`charts.md`](charts.md) (graphs, batch 9 onward). Nothing here should be re-run without a reason.

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
adding the following-tail block (26-28) and food-space (29). Batch 11 is the first and only batch
to train on it. ‡‡‡ rows compare to each other and, with care, to ‡‡ rows: batch 11 is byte-identical
to batch 10 in config, so that one cross-era comparison is *designed* and is written up under
batch 11 below. Everything else stays within its own era. Batch 10's checkpoints stopped loading on
`master` when this landed.

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
| `b11a-obs30seed1` ‡‡‡ | disc 0.995, **fourth env** | 3.19M | 94% @671k | 93.3% | 79.5% /4800 † | 85.7% | ‡‡‡ 2nd of batch 11; peaked at 678k, then lost 42 pp |
| `b11d-obs30seed4` ‡‡‡ | disc 0.995, **fourth env** | 3.59M | 88% @3507k | 86.7% | 69.3% /4000 † | 78.3% | ‡‡‡ only arm still near peak when stopped |
| `b11c-obs30seed3` ‡‡‡ | disc 0.995, **fourth env** | 3.23M | 87% @1706k | 84.7% | 69.0% /2300 † | 73.0% | ‡‡‡ weakest of batch 11 |
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

**† batch 11's `measured` column is the graph-100% tier only, not the whole arm.** Batch 10 was
measured flat — every selected checkpoint at 100 episodes — so pooling all its rows is already an
equal-effort figure. Batch 11 ran the three-stage protocol, where the rows have different depths and
the deep ones were chosen *because* they screened well, so pooling them reads high by construction.
The equal-effort fix is to truncate every checkpoint to its first 20 episodes, and that is not
recoverable here: the output file stores per-checkpoint totals, not per-episode results, and these
four runs predate the `pooled_equal_effort` field that computes it in-process. What *is* clean in
both batches is the graph-100% tier — 100 episodes each, no screening applied in either — so that is
what these four rows report, and it is the column the batch 11 vs batch 10 comparison below uses.
The episode counts are correspondingly smaller (`/4800` against batch 10's `/27200`).

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

## Batch 10 — a fresh baseline, and a new project record

**Launched 2026-08-02, all four stopped 2026-08-03** by request, healthy, to make room for
further changes — not because any of them died or declined. Seven observation/reward changes
had landed the same day batch 10 launched (fatal-move zeroing, wall/body hugging, normalized
group count, the corrected starve/length split, the terminal-discount fix, safe-to-chase-food,
and the audit that started the day), on top of the audit that already made batch 9 incomparable
to batch 8. Nothing had trained on the resulting environment at all, so batch 10 was deliberately
four seeds of **one** config — `DISCOUNT=0.995`, the most reliably-surviving value on record (3 of
3 in batch 7, 2 of 2 in batch 9) — rather than another comparison, on the reasoning that comparing
two things before either has a baseline on the actual environment being measured just repeats
batch 9's own lesson at one remove. Shared base for all four:
`SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0`.

| policy | final step | peak trailing | best-30 perfect | best ckpt (mid-run eval) | top-3 | pooled | outcome |
|---|---|---|---|---|---|---|---|
| `b10a-disc995seed1` | 4.29M | 94.4 @3402k | 78.3% @3402k | close-out eval pending | — | — | stopped healthy |
| `b10b-disc995seed2` | 4.65M | 94.96 @4545k | 85.0% @4547k | 87% @1157k | 86.3% | 70.4%/12000 | stopped healthy, 2nd-best |
| `b10c-disc995seed3` | 4.12M | 93.8 @4021k | 72.7% @4064k | close-out eval pending | — | — | stopped healthy |
| `b10d-disc995seed4` | 4.45M | 94.7 @3978k | 84.3% @1666k | **95%** @1815k | **93.3%** | **74.5%/24600** | stopped healthy, **project record** |

**All four were still healthy when stopped** — no `dead_since`/`zero_since` on any of them, and
two (`b10b`, `b10c`) were at or near their own peak trailing score *at the moment they were
stopped*, not declining from an earlier peak the way every record-setting arm before this batch
was. That is a first for this project: every previous best-checkpoint arm (`b8f`, `b8d`, `b9d`)
had already peaked and was declining by the time it was measured or stopped.

**`b10d`'s 95% (CI 88.8-97.8, 246 checkpoints cleared the ≥90% tier) is the best figure this
project has produced on any environment**, ahead of the pre-audit record of 92% (`b8f`
`ckpt2816000`, no longer reproducible — see the hall of fame) and far ahead of the previous
post-audit best of 82% (`b8f` `ckpt3149000`, also no longer reproducible after this session's
seven fixes). Both `b10b`'s and `b10d`'s measurements were taken **mid-run** — each arm kept
training for another ~2.5-2.9M steps afterward, up to a final step neither eval's checkpoint
range covered — so a full close-out re-measurement across each arm's complete checkpoint history
may move these numbers further; see [`runs.md`](runs.md) for whether that has happened yet.

**Not yet established: whether this is the config or the environment.** Batch 10 deliberately
ran one config, so it cannot separate "the seven 2026-08-02 fixes made training better" from
"this seed cluster happened to be strong" — n=4 on one value is enough to trust as a baseline,
not enough to attribute the gain. Both `b10d` (95%) and `b10b` (87%) comfortably clear the old
82% ceiling, and `b10a`/`b10c` graph-eval no worse than batch 9's arms did at a comparable
horizon, which is suggestive but not the isolating comparison a future batch would need to make
the claim safely.

Both champion checkpoints are preserved in
[`../hallOfFame/`](../hallOfFame/README.md#the-current-record-95-trained-end-to-end-on-todays-environment-2026-08-03).

## Batch 9 — 0.995 against 0.9975 on the post-audit environment

**Launched 2026-08-02 00:41, all four stopped the same day.** The first batch trained on the
environment left by the observation/reward audit, so **none of its numbers are comparable to any
batch above it** — the same checkpoint that scored 92% pre-audit reads 73% here. Shared base
`SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0`.

| policy | discount | final step | peak trailing | best30 | **best eval'd ckpt** | top-3 | pooled | outcome |
|---|---|---|---|---|---|---|---|---|
| `b9a-disc9975a` | 0.9975 | 3.68M | **89.8** @3277k | **56.0%** @1738k | 65.0% @1735k | 64.3% | **54.9%** /2000 | survived |
| `b9b-disc9975b` | 0.9975 | **10.47M** | 72.1 @**328k** | 5.0% @221k | not measured | — | — | **dead** |
| `b9c-disc995a` | 0.995 | 3.71M | 86.5 @3573k | 37.3% @2608k | 52.0% @2603k | 51.3% | 38.0% /2000 | survived |
| `b9d-disc995b` | 0.995 | 3.45M | 86.7 @1232k | 30.3% @1324k | **70.0%** @2544k | **66.3%** | 42.4% /1700 | survived |

**Why the batch was shaped this way.** The queued plan had two 0.9975 seeds plus `0.996` and
`LEARNING_RATE=1e-4`, resting on 0.995 having survived 3 of 3 — but that was measured pre-audit and
was void, so as written the batch had no 0.995 arm and could not settle the question it existed for.
Four arms on two values answers one question properly instead of three partially. `0.996` and the
learning rate stayed deferred.

**Outcome: the candidates win different things and the question is still open.** Ceiling and top-3
go to 0.995 (`b9d` at 70% / 66.3%), consistency to 0.9975 (`b9a` pools 54.9% with 18 of 20
checkpoints above 40%), and survival to 0.995 at 2 of 2 against 1 of 2. Survival dominates expected
value — mean top-3 across seeds is **58.8% for 0.995** against **32.2% for 0.9975** — but the two
0.995 seeds are **18 points apart** on best checkpoint, so seed spread still exceeds the effect.

**`b9b` is the clearest overrun failure in the file.** It peaked at step **328k**, before any sibling
had warmed up, then ran a further 10.1M steps producing nothing, `zero_since` 9.92M. It did that
overnight with nobody watching, which is the entire argument for the 3-3.5M stop rule rather than
stopping by inspection.

**The three survivors were measured while still training**, at 3.4-3.7M, so their close-outs are
mid-run. `b9c` was still climbing when stopped — peak trailing at 3573k, its last few hundred
thousand steps — so it is the one arm here that might have had more to give.

**Batch 9's best checkpoint (70%) is below `b8f`'s re-measured 82%.** Nothing trained after the audit
yet beats something trained before it, at a comparable horizon. One batch, and not a verdict, but
recorded because it is the opposite of what the fixes were meant to buy.
