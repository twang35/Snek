# Completed runs

Every arm that has finished, with its config, final numbers and verdict. Companion to
[`runs.md`](runs.md) (what is running now), [`findings.md`](findings.md) (what was
learned), and [`charts.md`](charts.md) (the graphs).

Nothing here should be re-run without a reason. Configs that are closed are marked as
such; the resume commands are in [`runs.md`](runs.md).

## All arms, ranked by best sustained perfect rate

"Best perfect-30" is the highest 30-eval trailing average the arm reached. "Cumulative"
is the mean perfect rate over its whole run. Both come from
`runs/<policy>_evals.json`; see [`hyperparamTuning.md`](hyperparamTuning.md) for why a
single eval is not usable as a measurement.

**Sort by the measured column where it exists.** It is a pooled 100-episode-per-checkpoint
measurement; every other column is derived from 10-episode graph evals and systematically
misranks arms (`b5c` is 2nd by graph, last by measurement).

Arms measured with the **outlier-top10** rule carry a best-checkpoint and top-3 figure and
are mutually comparable; older `measured` figures used other selection rules and are not.

**From 2026-07-30 the `measured` column stops being comparable at all**, so the episode count
is now spelled out per row (`/6300`, `/1000`, `/100`). The selector measures every checkpoint at
>=90% and nothing below 60%, capped at 20, so the checkpoint count varies per arm — `b8e` has 1,
`b8f` has 63. Use **best ckpt**. `b8e` is the illustration: 32% best checkpoint reads mid-table, but
having one checkpoint above the floor where the batch's best arm has dozens is the actual result.

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

**Four environments in two days is the real cost being paid here**, and it is worth stating
plainly: every environment change resets the comparison set, so a batch's numbers are only ever
readable against its own siblings. That is why seed count inside a batch matters more than the
number of knobs tried across batches — see the note at the end of [`runs.md`](runs.md).

| policy | config change | final steps | best ckpt | top-3 | **measured** | best perfect-30 | verdict |
|---|---|---|---|---|---|---|---|
| `b11b-obs30seed2` ‡‡‡ | disc 0.995, **fourth env** | 3.56M | close-out running | pending | pending | **91.7%** | ‡‡‡ highest best-30 of any arm on record |
| `b11a-obs30seed1` ‡‡‡ | disc 0.995, **fourth env** | 3.19M | close-out running | pending | pending | 85.7% | ‡‡‡ peaked at 678k, then lost 42 pp |
| `b11d-obs30seed4` ‡‡‡ | disc 0.995, **fourth env** | 3.59M | close-out running | pending | pending | 78.3% | ‡‡‡ only arm still near peak when stopped |
| `b11c-obs30seed3` ‡‡‡ | disc 0.995, **fourth env** | 3.23M | close-out running | pending | pending | 73.0% | ‡‡‡ weakest of batch 11 |
| `b10d-disc995seed4` ‡‡ | disc **0.995**, third env | 4.45M | **93%** @1695k | **93.3%** | **74.9%** /66000 | 84.3% | ‡‡ best measured checkpoint on record |
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

## Batch 8 — the discount optimum, gradient clipping, and the arm lifetime

**Started 2026-07-29, all arms stopped 2026-08-01.** Seven arms in four slots. It produced the
**project record (92% perfect games)**, falsified `DISCOUNT=0.999` and gradient clipping, and — by
running two arms far longer than any before — established that an arm has a *lifetime*.

| policy | extra override | final step | best 30-eval pf | **best measured ckpt** | pooled | verdict |
|---|---|---|---|---|---|---|
| `b8f-disc9975seed2` | `DISCOUNT=0.9975` | 5.47M | **69.3%** @2828k | **92.0%** @2816k | **66.3%** /5200 | **project record**; declining when stopped |
| `b8d-disc995clip` | `0.995` + `CLIPPING=10` | **11.64M** | 50.0% @2671k | **80.0%** @2538k | 60.4% /2000 | 2nd by measurement, then **died at ~7M** |
| `b8g-clipseed3` | `0.995` + `CLIPPING=10` | 3.43M | 30.0% @253k | none >50% | — | died, recovered after 1.2M, died again |
| `b8e-clipseed2` | `0.995` + `CLIPPING=10` | 1.16M | 21.3% @515k | 32.0% @500k | 32% /100 | flat; one good ckpt, no good region |
| `b8c-disc9975` | `DISCOUNT=0.9975` | 1.75M | 14.7% @343k | not measured | — | monotone decline to a stop |
| `b8a-disc999` | `DISCOUNT=0.999` | 1.11M | 0.7% @82k | 0% (dead) | — | dead at 452k |
| `b8b-disc999seed2` | `DISCOUNT=0.999` | 1.41M | 0.0% | 0% (dead) | — | dead; zero perfect games in 1.41M |

**Four results, in descending order of how much they change what to do next:**

1. **The horizon, not a hyperparameter, was the binding constraint.** `b8f` and `b8d` measured 63%
   and 62% at ~1.8M steps and **92% and 80%** at their best. Every previous record-holder had been
   stopped at ~1.06M. But followed further both peaked at **~2.5-3M** and then declined, and `b8d`
   died. The horizon has both a floor and a ceiling: **stop around 3-3.5M**.
2. **`DISCOUNT=0.9975` holds the record but is 1 of 2 on survival.** `b8f` produced the 92%
   champion; its sibling `b8c` ran the identical config and declined monotonically to a stop. The
   discount optimum sits somewhere in 0.995-0.9975 and is not yet resolved.
3. **`DISCOUNT=0.999` is falsified, 2 of 2 dead** — and it failed differently from other deaths,
   never learning at all (peak trailing 63.1 and 31.8; `b8b` produced zero perfect games in 1.41M
   steps). That gives the discount an optimum rather than a monotone benefit.
4. **`GRADIENT_CLIPPING=10` is falsified at 1 of 3**, against 3 of 3 for plain 0.995. It briefly
   looked like the batch's headline off `b8d` at 163k steps, then off `b8d`'s 62% measurement; both
   readings were premature, and `b8f` matched or beat that ceiling without clipping.

### Original design rationale

The obvious next question: if 0.995 helped this much, does more help more? A perfect game
runs several hundred steps, so even 0.995 (~200-step horizon) may still under-weight the
terminal bonus. All arms keep the winning base
`SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0`:

| policy | extra override | effective horizon | role |
|---|---|---|---|
| `b8a-disc999` | `SNEK_DISCOUNT=0.999` | ~1000 steps | is more better, or unstable? |
| `b8b-disc999seed2` | `SNEK_DISCOUNT=0.999` | ~1000 steps | second seed, since n=1 proves nothing here |
| `b8c-disc9975` | `SNEK_DISCOUNT=0.9975` | ~400 steps | the midpoint, if 0.999 breaks |
| `b8d-disc995clip` | `SNEK_DISCOUNT=0.995 SNEK_GRADIENT_CLIPPING=10` | ~200 steps | clipping on the known-good setting |

Higher discounts are a **known source of instability** — bootstrapped targets grow as the
horizon lengthens — so 0.999 may well be worse rather than better. That is why 0.9975 sits
in the batch as a fallback midpoint and why `b8d` tests a stability aid on the setting that
already works rather than on a riskier one.

### What the batch cost, and the step-rate trap

`b8d` spent ~8.5M steps after its peak and produced nothing measurable in them. It also advanced
**7.3M steps in ~24 hours** while `b8f` managed 1.9M on the same machine — almost entirely because a
dead policy plays very short episodes and therefore burns training steps far faster. A sudden jump
in step rate is a symptom of death, not speed.

## Batch 7 — seeding `b6b`, and finding `DISCOUNT=0.995`

**Started 2026-07-28, all arms stopped 2026-07-29.** Six arms ran in four slots: three
seeds of `b6b`'s config, then `DISCOUNT=0.995` seeded three times as the seeds died.

All shared `SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0`.

| policy | extra | final step | best ckpt | top-3 pooled | outcome |
|---|---|---|---|---|---|
| `b7f-disc995seed3` | `DISCOUNT=0.995` | 1.06M | **51%** @860k | **48.0%** | **best arm on record** |
| `b7e-disc995seed2` | `DISCOUNT=0.995` | 1.28M | 39% @334k | 34.7% | strong, survived |
| `b7d-discount995` | `DISCOUNT=0.995` | 1.60M | 26% @1330k | 22.7% | survived, weakest of the three |
| `b7a-a06seed2` | none | 2.00M | 19% @1822k | 18.3% | survived to 2M |
| `b7b-a06seed3` | none | 1.78M | — | — | **died at 1162k** |
| `b7c-a06seed4` | none | 1.74M | — | — | **died at 573k** |

**The batch was designed to answer a different question than the one it answered.** Its
purpose was seeding `b6b`'s config to n=3, because every single-seed result in this project
had failed to replicate. That question got answered — **2 of 4 seeds survive**, so eff ~1.2
is not the reliable setting it looked like, and the "lower sharpness is safer" reading was
weakened rather than confirmed.

The useful result came from the fourth slot. `DISCOUNT=0.995` was included as the one
untested high-prior knob, and after the first arm looked strong the two freed slots went to
seeding it instead of `b6b`. That decision — replacing dead seeds of a mediocre config with
seeds of a promising one, rather than completing the original design — is what produced the
result. See [`findings.md`](findings.md).

Both `b6b`-config deaths came **late** (573k, 1162k) compared to the eff ~1.6 deaths (246k,
272k), which suggests lower sharpness delays death rather than preventing it. Any survival
rate from here needs a step horizon attached.

**The gain from `DISCOUNT=0.995` is reliability, not ceiling.** `b7f` (51%) and `b4c` (50%,
eff exponent ~1.6) are a dead heat on level, and their intervals nearly coincide. What changed
is survival: `b4c`'s config threw away two runs in three to reach that level, and the three
`0.995` seeds reached a comparable level three times out of three. Weighting level by survival
makes the difference concrete:

| config | mean level across seeds | survival | expected value |
|---|---|---|---|
| `DISCOUNT=0.995` | 28.2% | **3 of 3** | **28.2%** |
| `b4c` config, eff ~1.6 | 37.1% | 1 of 3 | 12.4% |
| same config at 0.99 (`b7a`) | 12.0% | 2 of 4 | 6.0% |

**~2.3x the expected value of the best previous config.** This was the first change in the
investigation to *remove* the ceiling/reliability tradeoff rather than move along it, and the
mechanism was predicted in advance: at 0.99 the effective horizon is ~100 steps while a perfect
game runs several hundred, so the terminal bonus was discounted into irrelevance.

## Batch 6 — the effective-exponent sweep

**Started 2026-07-27, both arms stopped 2026-07-28.** Both arms keep the `b4c` signature (`td_loss`, no IS) and dial
alpha down, testing whether the effective exponent is what governs stability:

| policy | alpha | effective exponent | prediction |
|---|---|---|---|
| `b6a-alpha04` | 0.4 | ~0.8 — matches live `b5d` | survives |
| `b6b-alpha06` | 0.6 | ~1.2 — between `b5d` and the dead arms | marginal |

Because `td_loss` squares the error before alpha is applied, `alpha=0.8` with `td_loss`
is really ~1.6 on the `td_error` scale — see
[`findings.md`](findings.md). So the alpha *label* has never matched what was tested, and
these two arms are the first honest points on that axis.

Note `b6b`'s alpha 0.6 **is** the committed default, so that override is a no-op on that
knob; `b6b` is precisely "committed alpha, `theSchlong`'s other two PER changes."

Why this rather than more seeds of `b5c`: `b5c` was still running at the time, and the alpha sweep is the only experiment that
could recover `b4c`'s 51% without its 2-in-3 death rate. If both new arms survive the
200-270k window, the lottery becomes a dial.

**What would falsify the mechanism:** `b6a` dying anyway (something other than exponent
sharpness kills these arms), or both surviving *and* scoring no better than baseline
(sharpness was never where the gain came from either).

**Outcome: the prediction half held and the correction mattered more.** `b6a` (eff ~0.8)
stayed stable throughout and measured 8.1% — safe but a low ceiling. `b6b` (eff ~1.2) was
called "marginal", crashed to near-zero twice, recovered both times, and measured 24.5% —
the best arm at the time. That produced the "sharpness is a variance dial" reading in
[`findings.md`](findings.md), later weakened when batch 7 seeded `b6b`'s config and lost 2
of 4 to late deaths.

Both arms were measured with the old smoothed-first selector, so **both figures are
underestimates** and are due a re-measure.

## Batch 5 — `b4c` repeat plus factor isolation

**Started 10:05 2026-07-27, restarted 17:09 for visible windows, all arms stopped
2026-07-28.** `b5a`/`b5b` stopped as dead; `b5c`/`b5d` stopped past peak.

All four arms shared `PRIORITY_EXPONENT=0.8` and differed only in the other two PER
factors, making the batch a replication attempt *and* a factor isolation at once:

| policy | differs from `b4c` by | role | outcome |
|---|---|---|---|
| `b5a-schlong` | nothing | `b4c` repeat, seed 1 | **died** 272k |
| `b5b-schlong2` | nothing | `b4c` repeat, seed 2 | **died** 246k |
| `b5c-schlongIS` | IS weights back on | isolates IS weights | survived, 17.0% peak then 2M-step decline |
| `b5d-schlongTDE` | `abs(td_error)` signal | isolates the priority signal | survived, stable, 10.7% ceiling |

`b5c` and `b5d` each differ from the repeats by **exactly one factor**, so the design was
meant to read three ways: all four high would give four seeds for the config plus the
factor isolation free; only the repeats high would name the factor carrying the gain; none
high would mean `b4c` was a lucky seed. That is the same information two sequential
batches would give, in half the wall clock.

**Outcome: the third reading, and the batch inverted its own premise.** Both exact repeats
died permanently in the 200-270k window and stayed flat at 0.0 for 1.7-1.9M steps. Both
single-factor arms survived. This retracted the "restoring `theSchlong`'s PER triples the
perfect rate" finding and led to the effective-exponent mechanism — see
[`findings.md`](findings.md).

### Restarted at ~36-47k to add visible windows

All four were killed and relaunched from their checkpoints so every arm renders a game
as it trains. Their graphs continue rather than restarting, with a resume marker at the
restart step. **Caveat: cpprb does not persist replay-buffer priorities across
save/restore**, so the restart reset all priorities to uniform and they rebuilt as
transitions were resampled. For a batch whose subject *is* prioritization that is a real
perturbation, but it landed at ~5% of the planned run and hit all four arms about
equally, so it should not bias the between-arm comparison. It is a reason not to restart
an arm deep into a run.

## Batch 4 — the sampling machinery

Three arms spanning the whole prioritization axis: none (`b4a`), none-plus-diversity
(`b4b`), and `theSchlong`'s original maximum (`b4c`). Spanning the axis paid off,
because the answer was at the end nobody expected — uniform sampling was the prior
favourite and came in at a third of `b4c`'s rate.

| arm | expected | actual |
|---|---|---|
| `b4a-uniform` | best of the three; PER already lost at 30k | 8.7% — better than the committed default, far below `b4c` |
| `b4b-unifbuf500k` | uniform plus diversity should stack | 9.3% — steadiest curve, marginal gain over `b4a` |
| `b4c-schlongper` | a long shot; "less correct" than the default | **34.0%** — the breakthrough |

Detail on why `b4c` won is in [`findings.md`](findings.md). The one caveat carried
forward: it is **n=1**, and its own history contains a collapse at 150-300k deep enough
to have ended the run if it had been judged at 300k.

## Batch 3 — the epsilon hypothesis

Led with interventions that plausibly prevented the ~265k collapse. **Result: A
falsified, B ambiguous and later retracted.**

| item | change | outcome |
|---|---|---|
| A | floor epsilon at 0.001 | **FALSIFIED** — `b3a-epsfloor`, `b3b-epsfloor2` both degraded anyway |
| B | `REPLAY_BUFFER_MAX_LENGTH=500000` | looked supported, then `b3c-buf500k` died at 750k. **Not settled** |
| C | `PRIORITY_EXPONENT=0.0` | promoted to batch 4, run as `b4a-uniform` |
| D | `GRADIENT_CLIPPING=10` | never run; still queued |
| E | third baseline repeat | effectively done — `b3a` spent its first 267k steps as one |
| F | LR schedule | dropped; no evidence of optimization instability |

The reasoning behind A and B at the time, kept because A was wrong and the reasoning
was still sound:

- **A — floor epsilon.** The one arm that reached epsilon 0.0 was the one arm that
  collapsed, and at 0.0 the buffer becomes a closed policy-data feedback loop. Expected
  the same 14%-class rate as `b1a-base` but held past 300k.
- **B — bigger buffer.** The buffer holds *transitions*, not episodes, and episode
  length grows with skill: at score 5 an episode is ~50 steps so 100k transitions span
  ~2000 episodes, but at score 80 it is ~800+ steps and spans only ~125. Experience
  diversity therefore *shrinks as the policy improves*, which fits the timing of a
  collapse that arrives well after the policy gets good.
- A and B were considered complementary rather than redundant: **A keeps exploratory
  data being generated, B keeps it from being evicted.**

## Batch 2 — baseline repeats and n=2

| arm | purpose | outcome |
|---|---|---|
| `b2a-base2` | repeat the baseline to learn its spread | ran to 999k as the 1M reference. No collapse, but a long oscillation ending at 1.1% |
| `b2b-nstep2` | test whether n-step's steadiness survives at n=2 | same shape as n=3, one step milder. Closed the n-step direction |

`b2a-base2` mattered more than expected: it is the same config as `b1a-base` and did
*not* collapse, which is what established that collapse is stochastic rather than a
property of a config.

## Batch 1 — the original plan, and how wrong it was

| policy | change | expected | actual |
|---|---|---|---|
| `b1a-base` | none (control) | a *reference*, not a winner | became the key run: collapsed at 265k, and still finished 2nd overall |
| `b1b-tgt200` | `TARGET_UPDATE_PERIOD=200` | smoother curve, smaller drawdown | faster early, *larger* drawdown |
| `b1c-nstep3` | `N_STEP_UPDATE=3` | earlier learning onset, better late game | slowest to rise, then declined for 850k steps |

The reasoning at the time, kept because two of three predictions were wrong:

- **`b1a-base`** — a control under the same machine load as its batch mates, since
  throughput and contention vary between batches and make cross-batch comparison weaker
  than within-batch.
- **`b1b-tgt200`** — the highest-prior change of the batch. 8 gradient steps between
  target-network syncs is extremely frequent where standard DQN uses hundreds to
  thousands, and a target chasing the online network is a classic cause of oscillation
  and forgetting, which was exactly the reported symptom.
- **`b1c-nstep3`** — the food reward is immediate but the perfect-game bonus is terminal
  and extremely sparse, so credit has to crawl back one step per gradient update; n-step
  returns propagate it ~3x faster.

### The interim read at 83k, and why it was worthless

Batch 1 was first compared at matched step 83000:

| policy | trailing-20 perfect % | 1st perfect | last-5 score | curve mean | max drawdown | peak |
|---|---|---|---|---|---|---|
| `b1a-base` | **1.5** | 44000 | **66.0** | 52.4 | 19.2 | 78.6 |
| `b1b-tgt200` | 0.5 | **33000** | 59.2 | **53.4** | 27.4 | 76.9 |
| `b1c-nstep3` | 0.0 | never | 26.3 | 16.5 | 18.6 | 38.1 |

Three conclusions were drawn from this table and **all three were later overturned**:

1. That no arm showed catastrophic forgetting. `b1a-base` collapsed at 265k.
2. That `b1c-nstep3` had the batch's best trajectory. It peaked at 255k and declined
   for 850k steps.
3. That perfect games arrive ~20x earlier than the premise suggested, since `b1a-base`
   hit a 10% eval at 44k. A single 10-episode eval reading 10% is one perfect game in
   ten, which is noise, not an arrival.

This is the origin of the rule that **nothing can be judged below ~250k steps** — see
[`hyperparamTuning.md`](hyperparamTuning.md).

### `N_STEP_UPDATE=3`: the momentum that fooled everyone

Judged on trajectory rather than level, n=3 looked like batch 1's most interesting arm.
Score by 40k-step block:

| steps | mean score | max |
|---|---|---|
| 0-40k | 5.1 | 16.2 |
| 40-80k | 26.9 | 38.1 |
| 80-120k | 26.8 | 35.4 |
| 120-160k | 37.4 | 66.3 |
| 160-200k | 49.7 | 70.1 |
| 200-240k | 57.3 | 67.5 |

Near-monotonic, and at 201k it was still gaining +10.3 per 20 evals while `b1a-base` had
gone flat at ~69. That looked like exactly what the investigation wanted: slower but
consistently improving.

It peaked at 255k and declined for the next 850k steps. **The momentum was real and it
still meant nothing** — which is the strongest available argument for the ~350k minimum
horizon, since every signal available at 240k pointed the wrong way.

### `TARGET_UPDATE_PERIOD=200`: hypothesis not supported, but interesting

The prediction was smoother curves and smaller drawdowns. It got the **opposite** on
drawdown (27.4 vs 19.2) and a lower last-5. What it did do is learn **much faster
early** — roughly score 55 by 15k steps where the baseline needed ~25k — and reach its
first perfect game sooner.

So the frequent-target-update theory of forgetting looks wrong, while "less frequent
target updates learn faster early" is worth pursuing. `=50` and `=500` are in the
backlog. Note this arm was stopped at 104k, well short of the judgeable horizon, so
both readings are weak.
