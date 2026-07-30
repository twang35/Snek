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

## Current best: `DISCOUNT=0.995`

```
SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0 SNEK_DISCOUNT=0.995
```

It **matches the best ceiling ever measured while surviving 3 of 3 seeds** instead of 1 of
3. Every measurement below uses the same outlier-top10 selection rule, so the columns are
comparable:

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

### Do this first

Re-measure `b6b-alpha06` and `b6a-alpha04` with the current selector before comparing them
to anything. Both predate the selection fix and are biased low:

```
cd /Users/tony_wang/Projects/Snek/snek2
PYTHONPATH=. EVAL_OUT_SUFFIX=_outlier10 \
  /opt/miniconda3/envs/snek/bin/python -u eval_checkpoints.py b6b-alpha06 top10
```

## Currently running: batch 8

**Launched 2026-07-29.** All four arms share
`SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0`; overrides
verified in every log. Design rationale is in the Batch 8 section below.

| policy | extra override | step | trailing | best 30-eval pf | recent-30 pf | status |
|---|---|---|---|---|---|---|
| `b8d-disc995clip` | `0.995 + CLIPPING=10` | 931k | **71.3** | **36.0%** @163k | 14.0% | running, healthy |
| `b8e-clipseed2` | `0.995 + CLIPPING=10` | 412k | **76.2** | 19.0% @245k | 4.3% | running, healthy |
| `b8f-disc9975seed2` | `DISCOUNT=0.9975` | 478k | **73.1** | 22.0% @473k | **21.7%** | running, rising |
| `b8g-clipseed3` | `0.995 + CLIPPING=10` | new | — | — | — | running, takes clipping to n=3 |
| `b8c-disc9975` | `DISCOUNT=0.9975` | 1.75M | 10.3 | 14.7% @343k | 0.0% | **stopped — monotone decline** |
| `b8a-disc999` | `DISCOUNT=0.999` | 1.11M | **0.0** | 0.7% @82k | 0.0% | **stopped — dead** |
| `b8b-disc999seed2` | `DISCOUNT=0.999` | 1.41M | **0.0** | 0.0% | 0.0% | **stopped — dead** |

**Clipping now has two healthy seeds.** `b8d` at 931k and `b8e` at 412k are both at their
strongest trailing scores yet (71.3 and 76.2), neither has been near death, and `b8d` has
cleared the 200-600k window where most arms in this project break.

**`b8f` is the arm to watch** — its recent-30 perfect rate of 21.7% is the highest of any
live arm and is *higher* than its own best window's neighbourhood, so it is still climbing
at 478k.

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

#### Field caveat: `zero_since` is missing on pre-change arms

`b8c` and `b8d` were launched before `zero_since` was added to the summary block, so their
running processes hold the old `run_report` and overwrite the backfill on every eval. Their
`zero_since` has to be computed from the eval series until they restart. **Editing
`run_report.py` does not affect already-running arms** — a general point, not specific to
this field.

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
| 0.9975 | ~400 | alive at 359k, 14.7% best-30 so far |
| 0.999 | ~1000 | **dead 2 of 2** |

#### `GRADIENT_CLIPPING=10` is the most promising thing in the batch

`b8d` reached a **36.0% best-30 window by 163k steps** with a 70% single eval — for
comparison `b7f`, the best arm on record, needed 699k to reach 44.0%. Far too early to
call on n=1 and a graph window, which is why `b8e` now seeds it.

The two replacements go to the two configs still alive rather than to another 0.999 seed:
0.999 is answered, while `b8c` and `b8d` were both n=1.

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
[`hyperparamTuning.md`](hyperparamTuning.md)). Use `top10`, which picks the ten most
promising *surviving* checkpoints by smoothed perfect rate and measures each over 100
episodes — the only apples-to-apples comparison available:

```
cd /Users/tony_wang/Projects/Snek/snek2
PYTHONPATH=. EVAL_OUT_SUFFIX=_top10 \
  /opt/miniconda3/envs/snek/bin/python -u eval_checkpoints.py b6b-alpha06 top10
```

Spelled `top10`, not `--top 10`: `handle_main` routes argv through absl, which rejects
unregistered `--flags` before `main()` runs.

Budget **~50 minutes for four arms in parallel**, not the ~8 minutes a single arm takes.
Good policies play long episodes and 40 eval workers oversubscribe 14 cores, so the
parallel speedup is much less than 4x. `b5a`/`b5b` need no evals — a dead policy scores 0.

#### Long runs delete their own best checkpoints

`max_to_keep=1000` with a checkpoint every 1000 steps is a **rolling 1M-step window**.
Three of the four live arms have already lost the checkpoint behind their best number:

| arm | best 30-eval pf | that checkpoint | oldest surviving | best *surviving* smoothed pf |
|---|---|---|---|---|
| `b6b-alpha06` | 21.7% @1467k | **kept** | 780k | 28.0% |
| `b6a-alpha04` | 14.3% @372k | **gone** (missed by 24k) | 396k | 15.0% |
| `b5d-schlongTDE` | 10.7% @410k | **gone** | 1052k | 14.0% |
| `b5c-schlongIS` | 17.0% @211k | **gone** | 1282k | 7.0% |

`b5c` is the painful one: its 17.0% peak is unmeasurable, and its best surviving region is
worth only 7.0%. **Every additional 1000 steps on a past-peak arm destroys evidence.**

Two consequences. Close an arm out at its horizon instead of letting it run — the marginal
step is worth less than the checkpoint it evicts. And `top10` filters to surviving
checkpoints automatically, so it degrades gracefully rather than failing on a deleted step.
Raising `max_to_keep` would also work if long runs stay the norm.

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
