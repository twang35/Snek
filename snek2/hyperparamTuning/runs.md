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

## Currently running

**Batch 5, started 10:05, restarted 17:09.** All four arms share
`PRIORITY_EXPONENT=0.8`; they differ only in the other two PER factors, so the batch is
a repeat *and* a factor isolation at the same time. All four render a visible window.

| policy | alpha | signal | IS | eff exp | step | trailing now | best 30-eval pf | state |
|---|---|---|---|---|---|---|---|---|
| `b5c-schlongIS` | 0.8 | `td_loss` | **1** | ~1.6 corrected | 821k | 72.3 | **17.0%** @211k | **running**, healthy, past peak |
| `b5d-schlongTDE` | 0.8 | **`td_error`** | 0 | ~0.8 | 763k | 63.4 | 10.7% @410k | **running**, healthy, easing off peak |
| `b6a-alpha04` | **0.4** | `td_loss` | 0 | ~0.8 | 256k | 72.4 | 13.7% @243k | **running**, healthy, still climbing |
| `b6b-alpha06` | **0.6** | `td_loss` | 0 | ~1.2 | 539k | 16.3 | 9.7% @101k | **running**, crashed then stuck low |
| `b5a-schlong` | 0.8 | `td_loss` | 0 | ~1.6 | 2.05M | **0.0** | 10.0% @84k | **stopped** — dead since 272k |
| `b5b-schlong2` | 0.8 | `td_loss` | 0 | ~1.6 | 1.92M | **0.0** | 7.7% @129k | **stopped** — dead since 246k |

**The batch inverted its own premise.** Both exact `b4c` repeats died permanently in the
200-270k window and have been flat at 0.0 for 1.7-1.9M steps since. Both arms that
reverted *one* factor are alive and healthy past that window, and `b5c`'s 17.0% is the
second-best 30-eval window on record.

### Matched-step comparison at 250k

The only fair way to read arms that are at wildly different step counts. Every arm's
state at ~250k, with what it eventually did:

| arm | eff exp | trailing @250k | 30-eval pf @220-250k | eventual fate |
|---|---|---|---|---|
| `b5c-schlongIS` | ~1.6 corrected | 73.7 | 10.0% | alive at 821k |
| `b6a-alpha04` | ~0.8 | 66.6 | **11.0%** | running, 256k |
| `b5d-schlongTDE` | ~0.8 | 37.7 | 0.0% | alive at 763k |
| `b4c-schlongper` | ~1.6 | 28.5 | 3.9% | survived, 51% measured |
| `b6b-alpha06` | ~1.2 | 16.5 | 0.0% | crashed at ~140k, stuck low |
| `b5a-schlong` | ~1.6 | 3.3 | 0.0% | died 272k |
| `b5b-schlong2` | ~1.6 | 0.5 | 0.0% | died 246k |

`b6a` has the **highest perfect rate of any arm at this step count** and has already
cleared the step where `b5b` died. Note how little the 250k reading predicts the ending:
`b4c` was at 28.5 here and went on to the best result on record, while `b5a` at 3.3 never
came back. Read this table as "where each arm was," not as a leaderboard.

The step-count spread is the eval-cost confound, not progress: a dead policy ends every
episode instantly, so `b5a`/`b5b` burned ~4x the steps of the live arms in the same wall
clock. High step count on a dead arm means nothing.

`b5c` and `b5d` are **not finished** — do not call them yet. `b3c-buf500k` looked like the
batch's best arm and then died at 750k.

#### Restarted at ~36-47k to add visible windows

All four were killed and relaunched from their checkpoints so every arm renders a game
as it trains. Their graphs continue rather than restarting, with a resume marker at the
restart step. **Caveat: cpprb does not persist replay-buffer priorities across
save/restore**, so the restart reset all priorities to uniform and they rebuilt as
transitions were resampled. For a batch whose subject *is* prioritization that is a real
perturbation, but it landed at ~5% of the planned run and hit all four arms about
equally, so it should not bias the between-arm comparison. It is a reason not to restart
an arm deep into a run.

Verify with:

```
pgrep -fl "python -u snek2.py"
```

Not `grep "[s]nek2.py"` — git telemetry `curl` processes carry `snek2/snek2.py` in their
payload and inflate the count for a few seconds at a time.

Update this section whenever runs start or stop — a future session reads it to know
what is in flight and might have been terminated.

### Do not judge before ~850k steps

`b4c-schlongper` did not reach its best level (32% perfect) until the 850-900k block,
and it was **mid-collapse at 300k** — the horizon this protocol previously used would
have killed it. Budget **~8 hours per arm**. Expect to check in rather than watch.

The exception is **total death**: trailing score pinned at 0.0 for hundreds of thousands
of steps is not a dip, and no arm has ever recovered from it (`b3c-buf500k` stayed dead
for 4M steps). `b5a` and `b5b` qualify at 1.6M+ steps dead. Everything short of that
gets the full horizon.

### Finish the batch with 100-episode evals

Comparing these arms by their graph peaks would be the winner's curse (see
[`hyperparamTuning.md`](hyperparamTuning.md)). When the batch ends, run
`eval_checkpoints.py` on each arm's best few checkpoints and compare *those* numbers.
~3 minutes per arm, and it is the only apples-to-apples comparison available.

For batch 5 that means `b5c` around 211k and `b5d` around 410k, plus whatever late peak
each reaches. `b5a`/`b5b` need no evals — a dead policy scores 0.

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
