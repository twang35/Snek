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

**Best bet: `b6b-alpha06`** (`PRIORITY_EXPONENT=0.6`, `PRIORITY_SIGNAL=td_loss`,
`IS_WEIGHTS=0`) — 24.5% over 1000 episodes, and that is an **underestimate**; it was
measured with the old smoothed-first selector, since shown to pick worse checkpoints.
**Best ceiling: `b4c-schlongper` at ~31%**, but it survives only 1 seed in 3, so its
expected value is ~10.6%.

Its best checkpoint is **851000 at ~40%**, not the widely-quoted 869000 — that checkpoint
pools **41.7%** over 300 episodes, and the famous "51%" was the high draw of three
measurements. See [`findings.md`](findings.md).

### Do this first

Re-measure `b6b-alpha06` and `b6a-alpha04` with the current selector before comparing them
to anything. Both were measured before the selection rule was fixed, so both are biased low:

```
cd /Users/tony_wang/Projects/Snek/snek2
PYTHONPATH=. EVAL_OUT_SUFFIX=_clusters \
  /opt/miniconda3/envs/snek/bin/python -u eval_checkpoints.py b6b-alpha06 top10
```

## Currently running: batch 7

| policy | extra override | step | trailing | peak | best 30-eval pf | state |
|---|---|---|---|---|---|---|
| `b7a-a06seed2` | none | 692k | 66.2 | **80.4** @516k | 14.0% @251k | healthy |
| `b7d-discount995` | `SNEK_DISCOUNT=0.995` | 629k | 63.5 | 78.3 @331k | **17.3%** @143k | healthy |
| `b7b-a06seed3` | none | 1.03M | 15.8 | 77.8 @118k | 7.7% @127k | alive, in a low phase |
| `b7e-disc995seed2` | `SNEK_DISCOUNT=0.995` | just started | — | — | — | replaces `b7c` |
| `b7c-a06seed4` | none | 1.74M | **0.0** | 76.6 @169k | 9.7% @186k | **stopped — dead** |

Shared by all: `SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss
SNEK_IS_WEIGHTS=0`. Step counts are not comparable — a dead policy ends episodes instantly
and burns steps several times faster.

#### `b7c` is dead, confirmed by waiting rather than guessing

Last check it had been at 0.0 for 162k steps and the call was to leave it, because `b6b`
itself spent 140-600k crashed and recovered to 76.8. That caution is now resolved: `b7c`
sat at **exactly 0.0 for 363 consecutive evals — 1.17M steps** — which clears the death
criterion by a wide margin. Stopped and replaced.

This is worth noting as the process working. The same restraint that produced four
retractions here would have produced a fifth; waiting cost ~1M steps of a doomed arm and
bought a verdict that needs no hedging.

#### `b7b` is not dying, it is oscillating

At trailing 15.8 it looks like `b7c` did, but its history says otherwise:

| block | mean trailing |
|---|---|
| 0-200k | 52.6 |
| 200-400k | **19.1** |
| 400-600k | 61.9 |
| 600-800k | 50.9 |
| 800-1000k | **14.3** |
| 1000-1200k | 13.8 |

It has already recovered from a 19.1 trough once. Same very-long-period oscillation as
`b6b`. Leave it.

#### Seed tally so far for `b6b`'s config

| seed | outcome |
|---|---|
| `b6b-alpha06` | survived, measured 24.5% |
| `b7a-a06seed2` | healthy, new peak 80.4 |
| `b7b-a06seed3` | alive, oscillating |
| `b7c-a06seed4` | **dead at 573k** |

**3 of 4 survive.** Against eff ~1.6's 1 of 3, that supports the risk/return reading:
lower sharpness survives more often. Not yet enough for a rate estimate — 3/4 and 1/3 are
not far apart at these sample sizes.

#### Why the replacement is a second discount arm

`b6b`'s config now has four seeds, so survival is reasonably characterised. `DISCOUNT=0.995`
has **one** (`b7d`), and it is currently the joint-best arm in the batch on perfect rate. A
second seed of the most promising untested lever is worth more than a fifth seed of a config
already at n=4.

Shared by all four: `SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss
SNEK_IS_WEIGHTS=0`. All four render a visible window. Launched from scratch, not resumed,
so cpprb priorities start clean.

**Launch trap, cost one failed launch:** in zsh an unquoted `$VAR` holding
`A=1 B=2 C=3` is **not** word-split, so `env $VAR cmd` passes it as a single malformed
assignment. All four arms crashed with
`ValueError: could not convert string to float: '0.6 SNEK_PRIORITY_SIGNAL=...'`. Write the
assignments literally on each command line. The crash happened inside `tuned()` before any
checkpoint was written, so nothing was corrupted — that validation earns its keep.

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

### Where every arm ended up, with measured rates

Every arm was closed out with `eval_checkpoints.py <arm> top10` — ten best surviving
checkpoints, 100 greedy episodes each. The measured column is the pooled 1000-episode rate
and is the only number worth comparing across arms:

| policy | batch | alpha | signal | IS | eff exp | final step | best 30-eval pf | **measured** |
|---|---|---|---|---|---|---|---|---|
| `b6b-alpha06` | 6 | **0.6** | `td_loss` | 0 | ~1.2 | 1.80M | 21.7% @1467k | **24.5%** (CI 21.9-27.3) |
| `b6a-alpha04` | 6 | **0.4** | `td_loss` | 0 | ~0.8 | 1.41M | 14.3% @372k | 8.1% (CI 6.6-10.0) |
| `b5d-schlongTDE` | 5 | 0.8 | **`td_error`** | 0 | ~0.8 | 2.07M | 10.7% @410k | 6.6% (CI 5.2-8.3) |
| `b5c-schlongIS` | 5 | 0.8 | `td_loss` | **1** | ~1.6 corr | 2.31M | 17.0% @211k | 2.1% (CI 1.4-3.2) |
| `b5a-schlong` | 5 | 0.8 | `td_loss` | 0 | ~1.6 | 2.05M | 10.0% @84k | not measured — dead, scores 0 |
| `b5b-schlong2` | 5 | 0.8 | `td_loss` | 0 | ~1.6 | 1.92M | 7.7% @129k | not measured — dead, scores 0 |

**`b6b` is 3x the next best arm with non-overlapping intervals.** Note how badly the graph
ranked these: it put `b5c` second at 17.0%, and `b5c` measured **last at 2.1%**. Never rank
arms by graph windows.

**All four numbers are underestimates** — they were produced by the smoothed-first
selector, which has since been shown to pick systematically worse checkpoints than raw
single-eval selection. The ordering is probably still right, since every arm was
disadvantaged the same way, but the levels are low. Re-measure before drawing on them.

Raw results in `runs/<arm>_checkpoint_evals_top10.json`.

All arms rendered a visible window. Step counts are **not** comparable across arms — a
degraded policy ends episodes instantly and burns steps several times faster, so
`b5a`/`b5b`'s 2M totals reflect death, not progress.

Verify nothing is running with `pgrep -fl "python -u snek2.py"`. Not
`grep "[s]nek2.py"` — git telemetry `curl` processes carry `snek2/snek2.py` in their
payload and inflate the count for a few seconds at a time.

Update this section whenever runs start or stop — a future session reads it to know
what is in flight and might have been terminated.

### Batch 5 — moved to completedRuns.md

All four arms stopped. Design rationale, per-arm roles and the outcome are in
[`completedRuns.md`](completedRuns.md#batch-5--b4c-repeat-plus-factor-isolation). Only
`b5d` is a resume candidate, and only if a slot is otherwise idle.

### Batch 6 — the effective-exponent sweep

**Both arms stopped, both worth resuming.** Started in the slots freed by stopping
`b5a`/`b5b`. This description stays here rather than moving to `completedRuns.md` because
the batch is paused, not finished.

Both keep the `b4c` signature (`td_loss`, no IS) and dial alpha down, testing whether
*effective* exponent governs stability. Because `td_loss` squares the error before alpha
is applied, `alpha=0.8` with `td_loss` is really ~1.6 on the `td_error` scale — see
[`findings.md`](findings.md) — so the alpha label has never matched what was tested, and
these are the first honest points on that axis:

| policy | alpha | eff exponent | prediction made before launch | outcome |
|---|---|---|---|---|
| `b6a-alpha04` | 0.4 | ~0.8, matches live `b5d` | survives | **held** — stable throughout |
| `b6b-alpha06` | 0.6 | ~1.2, between `b5d` and the dead arms | marginal | **wrong** — see below |

`b6b`'s alpha 0.6 **is** the committed default, so that override is a no-op on that knob;
`b6b` is precisely "committed alpha, `theSchlong`'s other two PER changes."

**`b6b` falsified the "marginal" call and is now the best active arm.** It crashed to
trailing 0.3 early, which I read as permanent capability loss. It then recovered, exceeded
its old peak, crashed to 0.9 a second time near 1.2M, recovered again, and now holds a
**21.7% best 30-eval window — second-best on record behind only `b4c`'s 34.0%** — with a
rising perfect-game trend (13.3% mean over its most recent 200k block). It is a very
long-period oscillator, not a casualty.

`b6a` is the mirror image: stable at ~73 trailing for over a million steps, never near
death, and stuck at a 14.3% ceiling.

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

## Batch 6 — the effective-exponent sweep

**Launched, running.** Both arms keep the `b4c` signature (`td_loss`, no IS) and dial
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

Why this rather than more seeds of `b5c`: `b5c` is still running and unfinished, so
replicating it now would be premature, and the alpha sweep is the only experiment that
could recover `b4c`'s 51% without its 2-in-3 death rate. If both new arms survive the
200-270k window, the lottery becomes a dial.

**What would falsify the mechanism:** `b6a` dying anyway (something other than exponent
sharpness kills these arms), or both surviving *and* scoring no better than baseline
(sharpness was never where the gain came from either).

## Batch 7 — seeding the winner (running, see above)

**Why three seeds of one config rather than three new knobs.** `b6b-alpha06` measured 24.5%
and survived, but on **n=1**, and every single-seed result in this document has failed to
replicate — four have been overturned outright. Seed count, not knob count, is the binding
constraint, so three of the four slots go to repeats.

What the batch can show:

| outcome | reading |
|---|---|
| all 3 seeds survive and land ~20-25% | 24.5% is the config's level. First reliable result in the project |
| all 3 survive but scatter widely | the config is stable but its *quality* is seed-dependent; needs more seeds still |
| 1-2 die | eff ~1.2 is also a lottery, just a better-odds one than eff ~1.6's 1-in-3 |
| all 3 die | `b6b` was the fluke, and nothing here beats the baseline reliably |

`b7d` tests `DISCOUNT=0.995` on the best base. At 0.99 the effective horizon is ~100 steps
while a perfect game runs several hundred, so the terminal bonus is discounted to
near-irrelevance — plausibly the most relevant untested knob for the actual objective.

**Judge with the outlier selector**, and note `b7d` is not comparable to the others on
`avg_reward` since changing the discount changes the reward scale; compare perfect rates.

### Later candidates

| change | why | gate |
|---|---|---|
| eff exponent ~1.4 (`td_loss` alpha 0.7) | between `b6b` (24.5%, survived) and `b4c` (31.8%, 1-of-3) — the ceiling/risk frontier | after batch 7 |
| `GRADIENT_CLIPPING=10` | cheap, independent, and variance is exactly what makes eff ~1.6 fatal | anytime |
| best config + `REPLAY_BUFFER_MAX_LENGTH=500000` | `b4b` beat `b4a` slightly, so diversity may stack | after a stable base exists |
| lower `IS_WEIGHTS` partially (beta < 1) | full IS correction cost `b5c` almost everything (2.1%); partial may keep stability without the cost | needs a new knob |

**Seed count is the binding constraint, not the number of knobs tried.** Four single-seed
conclusions in this document have been overturned. Nothing goes in
[`findings.md`](findings.md) as established without n=3.

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
