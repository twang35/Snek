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

**Current best: 51% perfect games**, measured over 100 episodes from
`b4c-schlongper` checkpoint 869000. Config: `PRIORITY_EXPONENT=0.8`,
`PRIORITY_SIGNAL=td_loss`, `IS_WEIGHTS=0`. **That config has since failed to replicate
twice** — it is a ~1-in-3 lottery, not a better policy. See
[`findings.md`](findings.md).

## Nothing is running

All four arms were stopped deliberately at 1.4M-2.3M steps to free the machine. None
died; none were finished. Checkpoints and replay buffers are intact, so any of them can
be resumed.

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

### Where every arm ended up

| policy | batch | alpha | signal | IS | eff exp | final step | final trailing | best 30-eval pf |
|---|---|---|---|---|---|---|---|---|
| `b6b-alpha06` | 6 | **0.6** | `td_loss` | 0 | ~1.2 | 1.80M | 30.6 (mid-dip) | **21.7%** @1467k |
| `b5c-schlongIS` | 5 | 0.8 | `td_loss` | **1** | ~1.6 corr | 2.31M | 57.7 | 17.0% @211k |
| `b6a-alpha04` | 6 | **0.4** | `td_loss` | 0 | ~0.8 | 1.41M | 73.3 | 14.3% @372k |
| `b5d-schlongTDE` | 5 | 0.8 | **`td_error`** | 0 | ~0.8 | 2.07M | 72.4 | 10.7% @410k |
| `b5a-schlong` | 5 | 0.8 | `td_loss` | 0 | ~1.6 | 2.05M | **0.0** | 10.0% @84k |
| `b5b-schlong2` | 5 | 0.8 | `td_loss` | 0 | ~1.6 | 1.92M | **0.0** | 7.7% @129k |

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

### Batch 7 candidates

| change | why | gate |
|---|---|---|
| 2 more seeds of the best batch-5/6 config | every single-seed result here has failed to replicate | after batch 6 |
| `DISCOUNT=0.995` | untested, high-prior; the perfect-game bonus is discounted to near-nothing at 0.99 | anytime |
| `GRADIENT_CLIPPING=10` | cheap, independent, and variance is what needs taming | anytime |
| best config + `REPLAY_BUFFER_MAX_LENGTH=500000` | `b4b` beat `b4a` slightly, so diversity may stack | after a stable base exists |

**Seed count is now the binding constraint, not the number of knobs tried.** Three
single-seed conclusions in this document have been overturned. Any config that looks good
from here needs n=3 before it goes in `findings.md` as established.

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
