# What is running, and what is next

The live file. Everything here is current state and forward plan — results and
conclusions live elsewhere so this stays short enough to actually keep accurate.

| file | contents |
|---|---|
| `runs.md` (this file) | what is running, what to run next |
| [`completedRuns.md`](completedRuns.md) | every arm that has finished: config, final numbers, verdict |
| [`findings.md`](findings.md) | what is established, what has been falsified |
| [`failureModes.md`](failureModes.md) | the four ways a policy degrades, and how to tell them apart |
| [`hyperparamTuning.md`](hyperparamTuning.md) | the protocol: metrics, how to judge, how to launch |
| [`charts.md`](charts.md) | progress graph per arm |

## Current best: **88% perfect games**, measured 2026-07-30 late

Re-measured after both arms trained another ~860k steps. The record moved by **25 points** in
thirteen hours:

| arm | config | ckpts | best ckpt | top-3 | pooled | 95% CI |
|---|---|---|---|---|---|---|
| `b8f-disc9975seed2` | **disc 0.9975** | 63 | **88.0%** @2581k | **82.7%** | **59.2%** /6300 | 57.9-60.4 |
| `b8d-disc995clip` | disc 0.995 + clip 10 | 25 | **80.0%** @2538k | 74.7% | 58.4% /2500 | 56.5-60.3 |
| same two arms, 13h earlier | — | 16 / 10 | 63.0% / 62.0% | 60.3% | 46.5% / 48.3% | — |
| `b7f-disc995seed3` | disc 0.995 | 10 | 51% @860k | 48.0% | 38.8% /1000 | — |
| `b4c-schlongper` | disc 0.99 | 10 | 50% @869k | 46.7% | 37.1% /1000 | — |

**One checkpoint won 88 of 100 games.** It is preserved in
[`../hallOfFame/`](../hallOfFame/README.md) along with `b8d`'s 80% checkpoint, outside the
`max_to_keep` rotation that would eventually delete them.

**Pooled rate is up 20 points on the record that stood the same morning** — 59.2% over 6300
episodes with a ±1.3 interval, so this is not a lucky best-of-N. `b8f` has 35 of 63 checkpoints
at >=60%.

**The two configs remain statistically tied on pooled** (56.5-60.3 vs 57.9-60.4, overlapping)
while `b8f` is clearly ahead on best checkpoint. Gradient clipping still buys nothing; see below.

**Both arms have since been stopped** (2026-08-01), so these mid-run measurements at 2.93M (`b8d`)
and 2.65M (`b8f`) are their final figures. Files are suffixed `_midrun2`; the earlier `_midrun` set
is kept for the repeat-measurement analysis in [`findings.md`](findings.md), and `_100pct` holds the
targeted spot check.

### Older `DISCOUNT=0.995` context

```
SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0 SNEK_DISCOUNT=0.995
```

It **matched the best ceiling then known while surviving 3 of 3 seeds** instead of 1 of
3. Every measurement below uses the same outlier-top10 selection rule, so the columns are
comparable with each other — but **not with the two arms above**, measured after the
threshold-tier selector made the checkpoint count vary per arm. The `best ckpt` column stays
comparable across both rules; the pooled columns do not.

| arm | discount | best ckpt | top-3 pooled | all-10 pooled | survived |
|---|---|---|---|---|---|
| `b7f-disc995seed3` | **0.995** | **51%** @860k | **48.0%** | 38.8% | yes |
| `b4c-schlongper` | 0.99 | 50% @869k | 46.7% | 37.1% | 1 of 3 seeds |
| `b7e-disc995seed2` | **0.995** | 39% @334k | 34.7% | 29.5% | yes |
| `b7d-discount995` | **0.995** | 26% @1330k | 22.7% | 16.4% | yes |
| `b7a-a06seed2` | 0.99 | 19% @1822k | 18.3% | 12.0% | yes |

**The gain is reliability, not ceiling.** `b7f` at 51% and `b4c` at 50% are a dead heat, and
their intervals nearly coincide. What changed is that `b4c`'s config threw away two runs in
three to reach that level, and the discount arms reached it three times out of three.
Weighting level by survival:

| config | mean level across seeds | survival | expected |
|---|---|---|---|
| `DISCOUNT=0.995` | 28.2% | **3 of 3** | **28.2%** |
| `b4c` config, eff ~1.6 | 37.1% | 1 of 3 | 12.4% |
| same config at 0.99 (`b7a`) | 12.0% | 2 of 4 | 6.0% |

**~2.3x the expected value of the best previous config.** This is the first change in the
investigation to *remove* the ceiling/reliability tradeoff rather than move along it — and
the mechanism was predicted in advance: at 0.99 the effective horizon is ~100 steps while a
perfect game runs several hundred, so the terminal bonus was discounted into irrelevance.

**Prefer top-3 pooled to best-checkpoint** when comparing arms. A best-of-10 is the maximum
of a noisy statistic, upward-biased, and its ±10-point interval is wider than the gaps
between arms. Top-3 pools 300 episodes, halves the interval, and still answers "how good
does this config get". `b7f`'s top three are 51/47/46, so its peak is a real region rather
than a lone spike.

**Caveat on the 3-of-3:** `b7e` and `b7f` were stopped at 1.28M and 1.06M, while their 0.99
siblings died at 1162k and 573k. Survival is therefore established only out to ~1.1M steps.

**Superseded** by the measurements at the top of this file: `b8f` and `b8d` measure **88% and
80%** best checkpoint and ~59% pooled, against `b7f`'s 51% and 38.8%.

### The `b6a`/`b6b` re-measurement is retired, not pending

It was queued as "do this first" because both arms predate the selection fix and were biased
low. The **60% floor closes it instead of answering it**:

| arm | best graph point | checkpoints the selector now picks |
|---|---|---|
| `b6a-alpha04` | 50% @510k | **none** — selector exits with a message |
| `b6b-alpha06` | 60% (2 points, both ~1.74M) | **2** |

Neither has anything at >=90%, and `b6a` has nothing above the 60% floor at all. Re-measuring
`b6b` would produce a 2-checkpoint figure that is not comparable to a 10- or 16-checkpoint
one, so the honest read is that **these two arms cannot be placed on the ranking table** and
the alpha question stays where batch 7 left it. Do not re-add this task; if the alpha
comparison matters, it needs new seeds, not new measurements of old ones.

## Nothing is running — all four slots are free

**Batch 8 finished 2026-08-01.** Its last two arms, `b8f-disc9975seed2` and `b8d-disc995clip`,
were stopped at 5.47M and 11.64M steps. Full per-arm results and the batch's design rationale
have moved to
[`completedRuns.md`](completedRuns.md#batch-8--the-discount-optimum-gradient-clipping-and-the-arm-lifetime).

| policy | extra override | final step | best 30-eval pf | **best measured ckpt** | pooled | why stopped |
|---|---|---|---|---|---|---|
| `b8f-disc9975seed2` | `DISCOUNT=0.9975` | 5.47M | **69.3%** @2828k | **88.0%** @2581k | **59.2%** /6300 | 2.5M past peak, ~3% of best rate |
| `b8d-disc995clip` | `0.995 + CLIPPING=10` | **11.64M** | **50.0%** @2671k | **80.0%** @2538k | 58.4% /2500 | **dead** — no perfect game in 6.1M steps |
| `b8g-clipseed3` | `0.995 + CLIPPING=10` | 3.43M | 30.0% @253k | none >50% | — | died, recovered after 1.2M, died again |
| `b8e-clipseed2` | `0.995 + CLIPPING=10` | 1.16M | 21.3% @515k | 32.0% @500k | 32% /100 | flat — never above 6.9% in any 300k block |
| `b8c-disc9975` | `DISCOUNT=0.9975` | 1.75M | 14.7% @343k | not measured | — | monotone decline, no perfect game in 1.26M |
| `b8a-disc999` | `DISCOUNT=0.999` | 1.11M | 0.7% @82k | 0% (dead) | — | dead at 452k |
| `b8b-disc999seed2` | `DISCOUNT=0.999` | 1.41M | 0.0% | 0% (dead) | — | dead — zero perfect games in 1.41M |

**Both were stopped for running past their useful life.** Per-1M-step means tell the whole arc:

| block | `b8f` trailing / perfect | `b8d` trailing / perfect |
|---|---|---|
| 1-2M | 80.6 / 30.1% | 73.4 / 15.4% |
| **2-3M** | **82.1 / 40.9%** | **72.2 / 27.4%** |
| 3-4M | 65.8 / 18.6% | 56.8 / 14.6% |
| 4-5M | 44.7 / 7.4% | 47.8 / 11.9% |
| 5-6M | 38.9 / 10.1% | 25.5 / 0.3% |
| 6-7M | — | 12.0 / 0.0% |
| 7-11.6M | — | **0.0-2.0 / 0.0%** |

**`b8d` died.** Its last perfect game was at step 5496k — **6.1M steps** before it was stopped — and
it sat at trailing ~0-2 from 7M onward. Its final step of **11.64M** is the longest run in the
project by more than 2x.

**`b8f` was declining but not dead** when stopped: last perfect game 12k steps before the end,
final trailing 59.6, and it still threw the occasional 100% single eval. It was stopped because its
last two 100k blocks averaged 2.7% perfect against 40.9% at its 2-3M peak — 2.5M steps past peak at
~3% of its best rate, with its record checkpoint already banked.

**This supersedes the "oscillating, not declining" read from 2026-07-31**, which was correct on the
evidence then — high and low blocks were genuinely alternating with 100% evals in the high ones.
Three further declining blocks have settled it. The lesson stands in both directions: that read
was not wrong to withhold judgement, and withholding judgement forever is not free either.

**Both record checkpoints are in [`../hallOfFame/`](../hallOfFame/README.md)**, so nothing was lost
by stopping either arm.

### New finding: these arms have a lifetime, and the horizon has an upper bound

`b8d` ran to 11.6M steps and died. Together with `b8f`'s decline this bounds the "do not stop a
healthy arm at 1M" finding from the other side:

| phase | steps | what happens |
|---|---|---|
| climb | 0 to ~2.5M | rate rises; the best checkpoints appear late in this phase |
| **peak** | **~2.5-3M** | both arms' best measured checkpoints (2581k, 2538k) and best windows (2828k, 2671k) |
| decline | ~3-6M | perfect rate falls steadily; occasional elite checkpoints persist |
| death | ~7M+ (`b8d`) | trailing ~0, no perfect games at all |

So the practical horizon is **stop around 3-3.5M**, not 1M and not indefinitely. Everything after
the peak cost ~8.5M steps of machine time on `b8d` and produced nothing measurable.

**A step-count trap this exposes:** `b8d` advanced 7.3M steps in ~24 hours while `b8f` managed
1.9M, and the difference is almost entirely that **a dead policy plays very short episodes**, so it
burns training steps far faster. Never compare two arms' progress by wall-clock step rate — a
sudden acceleration is a symptom of death, not of speed.

### Targeted re-measurement of `b8f`, 2026-07-31: the 88% record stands

A 6-checkpoint spot check of the four unmeasured 100% graph points plus the two neighbours of
2806000 — 30 minutes instead of the ~3 hours a full 47-checkpoint re-measurement would cost:

| step | graph | measured | 95% CI | avg score |
|---|---|---|---|---|
| 3145000 | 100% | **83.0%** | 74.5-89.1 | 92.4 |
| 3149000 | 100% | 81.0% | 72.2-87.5 | 90.6 |
| 2806000 | **100%** | 80.0% | 71.1-86.7 | 88.1 |
| 2805000 | 80% | 74.0% | 64.6-81.6 | 88.0 |
| 2807000 | 70% | 74.0% | 64.6-81.6 | 90.4 |
| 3386000 | 100% | 73.0% | 63.6-80.7 | 89.0 |

**Nothing beat the 88% champion at 2581000**, so the hall-of-fame entry stands. This file predicted
2806000 was "the most promising unmeasured checkpoint in the project" because it was a 100% point
inside the arm's peak window; it measured **80%**, third of the six. The prediction was directionally
right — all six are top-decile — and wrong about the ranking.

**Pooled 77.5% over 600 episodes**, which is *not* comparable to the 59.2% from the 63-checkpoint
run: this set was chosen to be exceptional, so it measures how good `b8f`'s best region is rather
than how good the arm is on average.

Two findings sharpened, both in [`findings.md`](findings.md): 100% graph points are now **9 for 9
above 64%** with a mean of 72.5%, the only graph value with a usable floor; and the
outlier-beats-neighbours result is **4 of 4**, though the margin shrank to +6 points here because
the whole neighbourhood is strong.

**Still unmeasured: 26 checkpoints at 90%** (the four 100% points are now done). A full
re-measurement would select 47 and take ~3 hours. Given that the best of nine 100% points reached
83% and the record came from an 80% point, more sampling of this arm has diminishing returns —
prefer spending the machine on new seeds.

**Do not judge either on a single trailing reading.** `b8d` read trailing 42.6 at 2336k, which
looked alarming and was not — its surrounding 50k blocks ran 66.7-80.6 and it went on to a 50.0%
best-30 window at 2671k and an 80% measured checkpoint. `b8f` dipped to 9.0% mean perfect around
1965-2015k and recovered to 56.8%. **Read blocks, not the latest eval.**

**The two arms stopped on 2026-07-30 were both clipping seeds** — see below. Two slots are free.

#### `b8c-disc9975`: stopped after a long monotone decline

Stopped at 1.75M with trailing 10.3, **13% of its 79.8 peak**, and **no perfect game for
1.26M steps** (none since 491k). Its sibling `b8f` runs the identical config and is
thriving, so this is seed variance rather than the config:

| block | mean trailing | mean perfect |
|---|---|---|
| 200-400k | **71.1** | 8.8% |
| 400-600k | 50.7 | 2.5% |
| 600-800k | 20.4 | 0.0% |
| 1000-1200k | 14.8 | 0.0% |
| 1600-1800k | **11.2** | 0.0% |

Never technically dead — trailing stayed near 10, not 0.0, and `zero_since` was null
throughout — which is why it needed a different criterion from the `0.999` arms. **Every
200k block was lower than the last, with no recovery in any of them.** That distinguishes
the monotonic-decline mode from the oscillations `b6b` and `b7b` showed, where low blocks
alternated with high ones. An arm that has produced nothing for 1.26M steps and is at 13% of
peak is finished even without hitting zero.

Replaced with `b8g-clipseed3` rather than a third 0.9975 seed: 0.9975 now stands at 1 of 2
and is already beaten by 0.995 on measured rate, while clipping had two healthy seeds and
n=3 is this project's bar for calling anything established.

#### Field caveat: `zero_since` is missing on `b8d`, and `.get()` hides it

`b8c` and `b8d` were launched before `zero_since` was added to the summary block, so their
running processes hold the old `run_report` and overwrite the backfill on every eval. Verified
2026-07-30: `b8d`'s summary has **no** `zero_since` key, while `b8f`'s has it. **Editing
`run_report.py` does not affect already-running arms** — a general point, not specific to this
field.

**The trap:** reading it with `summary.get('zero_since')` returns `None` for a missing key,
which is indistinguishable from a computed `None` meaning "alive right now". A status check on
`b8d` therefore reports "not dead" whatever the arm is actually doing. It happens to be
correct here — recomputing from the eval series gives `None`, and trailing is 81.4 — but the
answer was luck, not measurement. **Check the key is present before trusting it**, or recompute
with `build_summary(rows['evals'])`, which agrees with the stored block on every other field
for both arms.

#### `DISCOUNT=0.999` is falsified, 2 of 2 dead

Both seeds died, and neither got far first — peak trailing **63.1** and **31.8** against
85.4 for `b8d`. `b8b` never produced a single perfect game in 1.41M steps. The hedge
recorded before launch was that longer horizons grow bootstrapped targets and 0.999 might
destabilise rather than help; that is what happened, and unusually for this project it was
clean at n=2 with no ambiguity to wait out.

So the discount has an optimum rather than a monotone benefit. Known points:

| discount | eff horizon | outcome |
|---|---|---|
| 0.99 | ~100 | 12.0% measured, dies 2 of 4 |
| 0.995 | ~200 | **38.8% measured, 3 of 3 survived** |
| 0.9975 | ~400 | **best-ever 47.7% best-30 (`b8f`), but 1 of 2** |
| 0.999 | ~1000 | **dead 2 of 2** |

**0.9975 is now the open question rather than a footnote.** `b8c` declined monotonically to a
stop while `b8f` measured **63.0% best / 46.5% pooled**, the joint-best result in the project. So
the config is 1 of 2 on survival and tied-best on ceiling. Whether the optimum sits at 0.995 or
0.9975 is undecided and is the obvious target for the two free slots: a third and fourth 0.9975
seed would settle it, where more 0.995 seeds would only re-confirm something already at 3 of 3.

#### `GRADIENT_CLIPPING=10` does not deliver the stability it was added for

This reverses the earlier reading in this file, which called clipping "the most promising
thing in the batch" off `b8d` alone at 163k steps. At n=3 it is **1 of 3**:

| arm | peak trailing | best 30-eval pf | best measured ckpt | outcome |
|---|---|---|---|---|
| `b8d-disc995clip` | **86.9** | **38.3%** | **62.0%** (10 ckpts, 48.3% pooled) | thriving at 2.34M |
| `b8e-clipseed2` | 85.9 | 21.3% | **32.0%** (1 ckpt) | faded, stopped at 1.16M |
| `b8g-clipseed3` | 77.0 | 30.0% | **none measurable** | dead, stopped at 3.43M |

**Plain `0.995` without clipping was 3 of 3** (`b7d`/`b7e`/`b7f`), so on survival the
clipping variant is worse than the config it was meant to stabilise, not better. The
hypothesis was that clipping the 10.0 terminal reward's gradient would prevent the
catastrophic drops; it did not prevent them in `b8e` or `b8g`.

**`b8d`'s 62% does not rescue clipping, and an interim read that said otherwise was wrong.**
Partway through the measurement, with `b8d` at 62% and `b8f` unmeasured, the live hypothesis
was "clipping raises the ceiling while lowering survival". `b8f` then measured **63.0%
without clipping**, with overlapping pooled intervals. So clipping has **no ceiling advantage**
either, and the verdict is simply: no measured benefit on ceiling, worse record on survival.
**Do not adopt it.** The lesson for next time is not to grade a two-arm comparison off the arm
that finished first.

#### `b8g-clipseed3`: died, recovered after 1.2M steps, then died permanently

The most instructive failure in the batch. Its 300k block means:

| block | mean trailing | mean perfect |
|---|---|---|
| 0-300k | 52.7 | 8.7% |
| 600-900k | 1.7 | 0.0% |
| 1200-1500k | 8.4 | 0.0% |
| 1800-2100k | 46.9 | 2.5% |
| **2100-2400k** | **63.7** | **4.3%** |
| 2700-3000k | **0.0** | 0.0% |
| 3300-3600k | 0.1 | 0.0% |

**It was near zero from 600k to 1800k — 1.2M steps — and came back to 63.7 trailing.** That
is by far the longest recovery on record and it stretches the "no arm recovers from sustained
zero" rule further than any previous case. Then it collapsed again and stayed at 0.0 for its
final 900k (`zero_since` 2625k).

Both halves matter. A long dead stretch is **not** proof an arm is finished, which argues for
patience. And a recovery is **not** proof of durability — the same lesson `b7b` taught, now
with a much larger swing. The practical rule that survives both: judge on `zero_since` against
current step, and 800k+ steps pinned at zero after a completed recovery arc is terminal.

#### `b8e-clipseed2`: stopped flat rather than dead

Stopped at 1.16M with trailing 57.8 — not dead, and it never was (`dead_since` and
`zero_since` both null for the whole run). It was stopped because **it never got good**: no
300k block averaged above 6.9% perfect, its best graph point in 1165 evals was a single 60%,
and its recent-30 perfect had fallen to 1.7%.

Its one measurable checkpoint (step 500k) came in at **32.0% (CI 23.7-41.7)**, which is
*better* than its 21.3% graph window implied and comparable to `b7e`'s 39%. So the config can
find a good policy; what it could not do is find more than one. Under the new >=80%/<=50%
thresholds it yielded exactly 1 checkpoint against `b8f`'s 16 — the clearest illustration yet
that **checkpoint count above the floor is itself the consistency metric** the project is
after.

**First batch to run under quiet logging and the fixed perfect-game pause.** Two things
follow. Logs are ~1 line per 10 evals, so take status from the `summary` block in
`runs/<policy>_evals.json` rather than by tailing. And these arms are **not wall-clock
comparable to batches 1-7**, which paid a 5-second stall per winning eval — step-indexed
comparisons are fine.

Batch 7's arms were all stopped and measured; rationale and results in
[`completedRuns.md`](completedRuns.md#batch-7--seeding-b6b-and-finding-discount0995).

Verify with `pgrep -fl "python -u snek2.py"`. Not `grep "[s]nek2.py"` — git telemetry
`curl` processes carry `snek2/snek2.py` in their payload and inflate the count.

### Batch 5/6 arms: stopped, resumable

All four were stopped deliberately at 1.4M-2.3M steps. None died; none were finished.
Checkpoints and replay buffers are intact, so any of them can be resumed.

### Which to resume, in priority order

| arm | eff exp | resume? | why |
|---|---|---|---|
| `b6b-alpha06` | ~1.2 | **yes, first** | best active arm (21.7%), perfect trend still rising across oscillations, best checkpoint intact |
| `b6a-alpha04` | ~0.8 | **yes** | stable 73 trailing for 1M steps, never near death; the low-variance control worth having at n>1 |
| `b5d-schlongTDE` | ~0.8 | only if a slot is idle | healthy but 10.7% ceiling, and `b6a` covers this exponent better |
| `b5c-schlongIS` | ~1.6 corr | **no** | declining for 2M steps, last-30 perfect 0.3%, and its 17.0% peak checkpoint is already deleted |

**Resuming now costs more than the batch-5 restart did.** cpprb does not persist
priorities, so a resume resets them to uniform. At 40k steps that was ~5% of a run and
harmless; at 1.8M steps, mid-oscillation, it perturbs exactly the mechanism under study.
Prefer starting a fresh seed over resuming a deep arm unless the specific late-run
trajectory is the thing being continued.

Note the `max_to_keep` increase to 10000 only takes effect on the next launch, so a
resumed arm keeps its existing 1000-deep history and starts extending from there.

All per-arm results now live in
[`completedRuns.md`](completedRuns.md#all-arms-ranked-by-best-sustained-perfect-rate),
including the cross-batch ranking. Batch 5, 6 and 7 descriptions moved there when their
last arms stopped, per the bookkeeping rule below.

### Do not judge before ~850k steps

`b4c-schlongper` did not reach its best level (32% perfect) until the 850-900k block,
and it was **mid-collapse at 300k** — the horizon this protocol previously used would
have killed it. Budget **~8 hours per arm**. Expect to check in rather than watch.

The exception is **total death**: trailing score pinned at 0.0 for hundreds of thousands
of steps is not a dip, and no arm has ever recovered from it (`b3c-buf500k` stayed dead
for 4M steps). `b5a` and `b5b` qualify at 1.6M+ steps dead. Everything short of that
gets the full horizon.

### Finish the batch with 100-episode evals

Comparing arms by their graph peaks would be the winner's curse (see
[`hyperparamTuning.md`](hyperparamTuning.md)). Use `top20`, which ranks *surviving*
checkpoints by their single 10-episode eval and measures each over 100 episodes:

```
cd /Users/tony_wang/Projects/Snek/snek2
PYTHONPATH=. EVAL_OUT_SUFFIX=_top20 \
  /opt/miniconda3/envs/snek/bin/python -u eval_checkpoints.py b8d-disc995clip top20
```

Spelled `top20`, not `--top 20`: `handle_main` routes argv through absl, which rejects
unregistered `--flags` before `main()` runs.

**The count is a target, not a quota** (thresholds revised 2026-07-31). Everything at **>=90%**
on its graph point is measured even if that exceeds twenty, remaining slots are filled from
**>=60%** best-first, and nothing below 60% is measured. Expect a variable number per arm — 32
for `b8f`, 20 for `b8d`, 1 for `b8e`, and an outright refusal for an arm that never cleared the
floor. **Compare arms on best checkpoint, not pooled rate**, because pooled averages over
different checkpoint counts and a truncated population.

**Results are written after every checkpoint**, so an interrupted run keeps its measurements.
Check `complete` in the JSON before treating a file as an arm's full result.

Budget **~4 minutes per checkpoint**. Throughput is core-bound, so raising `EVAL_WORKERS` past
~10 does not speed a run up — running two arms in parallel does.

Budget **~50 minutes for four arms in parallel**, not the ~8 minutes a single arm takes.
Good policies play long episodes and 40 eval workers oversubscribe 14 cores, so the
parallel speedup is much less than 4x. Dead arms need no evals at all, and the selector now
says so itself rather than spending 100 episodes proving it.

#### Long runs used to delete their own best checkpoints

**Fixed: `max_to_keep` is now 10000**, a rolling 10M-step window at ~188 KB per checkpoint
(~1.8 GB per policy at full depth). At the old value of 1000 it was a 1M-step window, and it
cost real evidence — three of batch 5/6's four arms outran it:

| arm | best 30-eval pf | that checkpoint | oldest surviving | best *surviving* smoothed pf |
|---|---|---|---|---|
| `b6b-alpha06` | 21.7% @1467k | **kept** | 780k | 28.0% |
| `b6a-alpha04` | 14.3% @372k | **gone** (missed by 24k) | 396k | 15.0% |
| `b5d-schlongTDE` | 10.7% @410k | **gone** | 1052k | 14.0% |
| `b5c-schlongIS` | 17.0% @211k | **gone** | 1282k | 7.0% |

`b5c` is the painful one: its 17.0% peak is permanently unmeasurable, and its best surviving
region is worth only 7.0%. Those four rows cannot be recovered by raising the setting.

Two habits still apply. Close an arm out at its horizon — past peak, the marginal training
step is worth less than the disk it consumes. And `top20` filters to surviving checkpoints
automatically, so it degrades gracefully rather than failing on a deleted step. The legacy
`train*/` dirs run 9.7 MB per checkpoint, so do not resume those at depth 10000 without
checking disk first.

### Batch bookkeeping

Each batch keeps its **description** — why it is shaped that way, what each arm isolates,
what outcome would mean what — in this file for as long as any of its arms is running.
When the last arm of a batch stops, move that description and its results to
[`completedRuns.md`](completedRuns.md) and delete it here. Batch 5 is mid-batch: two arms
stopped, two still running, so it stays.

The reason to keep the description live rather than only the status table: the design
rationale is what tells a future session whether a surprising result is informative or
just an arm that was never going to answer anything.

## Resuming a stopped arm

Relaunch with the same policy name **and the same `SNEK_*` overrides**. The overrides
are *not* persisted in the checkpoint, so relaunching without them silently changes
the config mid-run and invalidates the arm.

| policy | overrides needed to resume |
|---|---|
| `b4c-schlongper` | `SNEK_PRIORITY_EXPONENT=0.8 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0` |
| `b4b-unifbuf500k` | `SNEK_PRIORITY_EXPONENT=0.0 SNEK_REPLAY_BUFFER_MAX_LENGTH=500000` |
| `b4a-uniform` | `SNEK_PRIORITY_EXPONENT=0.0` |
| `b3a-epsfloor` / `b3b-epsfloor2` | `SNEK_MIN_EPSILON=0.001` |
| `b3c-buf500k` | `SNEK_REPLAY_BUFFER_MAX_LENGTH=500000` (dead; not worth resuming) |
| `b1b-tgt200` | `SNEK_TARGET_UPDATE_PERIOD=200` |
| `b1a-base` / `b2a-base2` | none — committed defaults |
| `b5a-schlong` / `b5b-schlong2` | `SNEK_PRIORITY_EXPONENT=0.8 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0` |
| `b5c-schlongIS` | `SNEK_PRIORITY_EXPONENT=0.8 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=1` |
| `b5d-schlongTDE` | `SNEK_PRIORITY_EXPONENT=0.8 SNEK_PRIORITY_SIGNAL=td_error SNEK_IS_WEIGHTS=0` |
| `b6a-alpha04` | `SNEK_PRIORITY_EXPONENT=0.4 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0` |
| `b6b-alpha06` | `SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0` |

Per-run logs written to `$CLAUDE_JOB_DIR/tmp` are job-scoped and do not survive.
The durable record is `runs/<policy>_evals.json`; analyse from there.

## Next up: batch 9 — settle the discount optimum

**All four slots are free.** The one unresolved question from batch 8 is whether the optimum sits at
`0.995` or `0.9975`: 0.9975 holds the record (88%) but is **1 of 2** on survival, while 0.995 is 3
of 3 with a lower ceiling. Two more 0.9975 seeds decide it.

| policy | override on top of the shared base | role |
|---|---|---|
| `b9a-disc9975seed3` | `SNEK_DISCOUNT=0.9975` | third 0.9975 seed — takes it to n=3 |
| `b9b-disc9975seed4` | `SNEK_DISCOUNT=0.9975` | fourth seed; 0.9975 is 1 of 2, so n=4 is not excessive |
| `b9c-disc996` | `SNEK_DISCOUNT=0.996` | between the two known-good values, in case the optimum is interior |
| `b9d-lr1e4` | `SNEK_DISCOUNT=0.9975 SNEK_LEARNING_RATE=1e-4` | the highest-value untested knob, on the record config |

Shared base for every arm:

```
SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0
```

**Run each to ~3-3.5M steps and stop.** That is the new horizon finding: both batch-8 arms peaked at
~2.5-3M and `b8d` then died at ~7M, spending 8.5M steps producing nothing. Do not repeat that.

**Close out with `top20`**, which now measures every checkpoint at >=90% and fills to 20 from >=60%.
Measure the **100% graph points first** — 9 of 9 have come in above 64%.

### Later candidates

| change | why | gate |
|---|---|---|
| eff exponent ~1.4 (`td_loss` alpha 0.7) at 0.9975 | `b4c` and `b7f` tie on ceiling; sharpness may still add on top of the discount | after batch 9 |
| best config + `REPLAY_BUFFER_MAX_LENGTH=500000` | `b4b` beat `b4a` slightly, so diversity may stack | after batch 9 |
| partial IS correction (beta < 1) | full correction cost `b5c` almost everything (2.1%); partial may keep stability without the cost | needs a new knob |
| anything aimed at the post-peak decline | both batch-8 arms peaked at ~2.5-3M and fell away; nothing tried so far addresses *why* | needs a mechanism first |

**Seed count is the binding constraint, not the number of knobs tried.** Six single-seed conclusions
in this document have been overturned or weakened — most recently gradient clipping, which looked
like batch 8's headline twice before failing at 1 of 3. Nothing goes in
[`findings.md`](findings.md) as established without n=3, which is why batch 9 spends two of four
slots on the same value.

Launch commands are in
[`hyperparamTuning.md`](hyperparamTuning.md#launching-a-run).

## Standing backlog

Untested, ordered by expected value. Rationale for the ones that need it follows the
table.

| change | targets | prior |
|---|---|---|
| `DISCOUNT=0.995` / `0.999` | perfect-game reward being reachable at all | **high** |
| `LEARNING_RATE=1e-4` | training speed | high, but order it after a stability fix |
| `TARGET_UPDATE_PERIOD=50` / `500` | early learning speed | medium — 2 points to test a hinted trend |
| `TARGET_UPDATE_TAU=0.005`, period 1 | smoothness (soft target updates) | medium |
| `FC_LAYERS=128,128` | capacity | low |
| epsilon ladder *shape* (not floor) | exploration schedule | low |
| `REPLAY_BUFFER_MAX_LENGTH=1000000` | experience diversity | low — the 500k result was ambiguous |

**`DISCOUNT=0.995` or `0.999` — the most under-rated item here.** At 0.99 the
effective horizon is ~100 steps, but a perfect game is several hundred steps long, so
the terminal bonus is discounted into near-irrelevance. Raising it should make the
perfect-game reward actually reachable by the value function — plausibly the single
most relevant change for the end goal. It is also a known source of instability, so
pair it with a stability fix rather than running it first.

**`LEARNING_RATE=1e-4` — only after a stability fix.** 1e-5 is very conservative and
the in-code comment already suggests 1e-4. With a stable target it may train several
times faster; on its own with `TARGET_UPDATE_PERIOD=8` it would probably make
instability worse. The order matters.

**`TARGET_UPDATE_PERIOD=50` and `=500`.** Batch 1 hinted that longer periods learn
faster early even though they didn't reduce drawdown. Two more points establish
whether that is a trend or noise. Note `b1b-tgt200` was stopped at 104k, well short of
the ~250k horizon, so that hint is weak evidence.

**Epsilon ladder shape.** The floor was tested and the hypothesis falsified. What
remains untested is the *shape*: the ladder is driven by reward thresholds and steps
down once per eval, so it is coupled to `eval_interval` — a latent confound if that
interval is ever changed, and a reason a slower or step-count-based decay is worth
trying.

## Explicitly not planned

- **Reward changes** — they would break comparability of `avg_score` with every run
  recorded so far.
- **Reverting to `PyUniformReplayBuffer`** — cpprb is ~2.4x faster with no measured
  learning cost, so cheaper experiments come from keeping it.
- **An LR schedule** — no evidence of optimization instability; degradation is gradual
  in every arm, not spiky.
- **Making the epsilon last-rung threshold tunable** — the ladder is no longer a
  suspect, see [`findings.md`](findings.md).
- **`N_STEP_UPDATE=5`** — n=2 and n=3 both peak below baseline and then decline, so
  the trend already points the wrong way.
