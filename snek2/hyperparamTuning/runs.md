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

## Current best: a two-way tie at ~62%, measured 2026-07-30

Two arms measured mid-run, both **~10 points above the previous record on pooled rate**:

| arm | config | ckpts | best ckpt | top-3 | pooled | 95% CI |
|---|---|---|---|---|---|---|
| `b8f-disc9975seed2` | **disc 0.9975** | 16 | **63.0%** @1618k | **60.3%** | 46.5% /1600 | 44.1-48.9 |
| `b8d-disc995clip` | disc 0.995 + clip 10 | 10 | **62.0%** @1688k | 58.7% | **48.3%** /1000 | 45.2-51.4 |
| `b7f-disc995seed3` | disc 0.995 | 10 | 51% @860k | 48.0% | 38.8% /1000 | — |
| `b4c-schlongper` | disc 0.99 | 10 | 50% @869k | 46.7% | 37.1% /1000 | — |

**Read the pooled column, not the best.** At 1000-1600 episodes its interval is ±3 points where
a single best-checkpoint reading is ±9, and repeat measurements of one frozen checkpoint have
spread 19 points here. On pooled, `b8d` and `b8f` overlap each other and both clear `b7f`
decisively.

**The two configs are statistically indistinguishable.** 63.0 vs 62.0 on best, overlapping
pooled intervals. Do not read a winner out of this pair.

**Both arms are still running**, so these are mid-run snapshots taken at 2.08M (`b8d`) and 1.78M
(`b8f`). Files are suffixed `_midrun`; a close-out `_top10` run comes later.

### Older `DISCOUNT=0.995` context

```
SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0 SNEK_DISCOUNT=0.995
```

It **matched the best ceiling then known while surviving 3 of 3 seeds** instead of 1 of
3. Every measurement below uses the same outlier-top10 selection rule, so the columns are
comparable with each other — but **not with the two arms above**, measured after the
>=80%/<=50% thresholds made the checkpoint count vary per arm. The `best ckpt` column stays
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

**Superseded 2026-07-30** by the measurements at the top of this file: `b8f` and `b8d` both
measure ~62% best and ~47% pooled, against `b7f`'s 51% and 38.8%.

### The `b6a`/`b6b` re-measurement is retired, not pending

It was queued as "do this first" because both arms predate the selection fix and were biased
low. The **50% floor added on 2026-07-30 closes it instead of answering it**:

| arm | best graph point | checkpoints the selector now picks |
|---|---|---|
| `b6a-alpha04` | 50% @510k | **none** — selector exits with a message |
| `b6b-alpha06` | 60% (2 points, both ~1.74M) | **2** |

Neither has anything at >=80%, and `b6a` has nothing above the floor at all. Re-measuring
`b6b` would produce a 2-checkpoint figure that is not comparable to a 10- or 16-checkpoint
one, so the honest read is that **these two arms cannot be placed on the ranking table** and
the alpha question stays where batch 7 left it. Do not re-add this task; if the alpha
comparison matters, it needs new seeds, not new measurements of old ones.

## Currently running: batch 8

**Launched 2026-07-29.** All arms share
`SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0`; overrides
verified in every log. Design rationale is in the Batch 8 section below.

Status as of **2026-07-30 11:45**, two arms running:

| policy | extra override | step | best 30-eval pf | **best measured ckpt** | pooled | max pt | status |
|---|---|---|---|---|---|---|---|
| `b8f-disc9975seed2` | `DISCOUNT=0.9975` | 2.12M | 47.7% @1625k | **63.0%** @1618k | 46.5% /1600 | **90%** | running, healthy |
| `b8d-disc995clip` | `0.995 + CLIPPING=10` | 2.34M | 38.3% @1910k | **62.0%** @1688k | 48.3% /1000 | 80% | running, healthy |
| `b8g-clipseed3` | `0.995 + CLIPPING=10` | 3.43M | **0.1** | 30.0% @253k | 0.0% | 50% | **stopped — died, recovered, died** |
| `b8e-clipseed2` | `0.995 + CLIPPING=10` | 1.16M | 57.8 | 21.3% @515k | 1.7% | 60% | **stopped — flat, then faded** |
| `b8c-disc9975` | `DISCOUNT=0.9975` | 1.75M | 10.3 | 14.7% @343k | 0.0% | 40% | **stopped — monotone decline** |
| `b8a-disc999` | `DISCOUNT=0.999` | 1.11M | **0.0** | 0.7% @82k | 0.0% | 10% | **stopped — dead** |
| `b8b-disc999seed2` | `DISCOUNT=0.999` | 1.41M | **0.0** | 0.0% | 0.0% | 0% | **stopped — dead** |

**Both arms are measured and effectively tied at the top of the project** — 63.0% and 62.0%
best checkpoint, overlapping pooled intervals, both ~10 points above `b7f` on pooled. See the
table at the top of this file.

**Both are healthy and neither should be stopped on a single trailing reading.** `b8d` read
trailing 42.6 at 2336k, which looks alarming and is not: its 50k block means over the last
400k run 66.7-80.6, and its most recent block (2286-2336k) has the **highest perfect rate of
its last 400k at 33.2%** with an 80% point. `b8f` similarly dipped to 9.0% mean perfect around
1965-2015k and recovered to 29.6%. **Read blocks, not the latest eval.**

**Both stopped arms were the clipping seeds** — see below. Two slots are free.

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
[`hyperparamTuning.md`](hyperparamTuning.md)). Use `top10`, which ranks *surviving*
checkpoints by their single 10-episode eval and measures each over 100 episodes:

```
cd /Users/tony_wang/Projects/Snek/snek2
PYTHONPATH=. EVAL_OUT_SUFFIX=_top10 \
  /opt/miniconda3/envs/snek/bin/python -u eval_checkpoints.py b8d-disc995clip top10
```

Spelled `top10`, not `--top 10`: `handle_main` routes argv through absl, which rejects
unregistered `--flags` before `main()` runs.

**The count is a target, not a quota** (added 2026-07-30). Everything at >=80% on its graph
point is measured even if that exceeds ten, nothing at <=50% is measured at all, and the
remaining slots come from the 60-70% band. Expect a variable number per arm — 16 for `b8f`,
1 for `b8e`, and an outright refusal for an arm that never cleared 50%. **Compare arms on
best checkpoint, not pooled rate**, because pooled now averages over different checkpoint
counts and a truncated population.

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
step is worth less than the disk it consumes. And `top10` filters to surviving checkpoints
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

## Batch 8 — push the discount further

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

### Later candidates

| change | why | gate |
|---|---|---|
| eff exponent ~1.4 (`td_loss` alpha 0.7) at 0.995 | `b4c` and `b7f` tie on ceiling; sharpness may still add on top of the discount | after batch 8 |
| best config + `REPLAY_BUFFER_MAX_LENGTH=500000` | `b4b` beat `b4a` slightly, so diversity may stack | after a stable base exists |
| partial IS correction (beta < 1) | full correction cost `b5c` almost everything (2.1%); partial may keep stability without the cost | needs a new knob |
| `LEARNING_RATE=1e-4` | 1e-5 is very conservative; worth trying now that a stable base exists | after batch 8 |

**Seed count is the binding constraint, not the number of knobs tried.** Five single-seed
conclusions in this document have been overturned or weakened. Nothing goes in
[`findings.md`](findings.md) as established without n=3 — which is why batch 8 spends two of
four slots on the same value.

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
