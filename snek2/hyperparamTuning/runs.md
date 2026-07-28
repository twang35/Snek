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
`PRIORITY_SIGNAL=td_loss`, `IS_WEIGHTS=0`. That is the number to beat — see
[`findings.md`](findings.md).

## Currently running

**Batch 5, started 10:05.** All four arms share `PRIORITY_EXPONENT=0.8`; they differ
only in the other two PER factors, so the batch is a repeat *and* a factor isolation at
the same time.

| policy | `PRIORITY_SIGNAL` | `IS_WEIGHTS` | display | role |
|---|---|---|---|---|
| `b5a-schlong` | `td_loss` | 0 | **visible** | `b4c` repeat, seed 1 |
| `b5b-schlong2` | `td_loss` | 0 | headless | `b4c` repeat, seed 2 |
| `b5c-schlongIS` | `td_loss` | **1** | headless | isolates IS weights |
| `b5d-schlongTDE` | **`td_error`** | 0 | headless | isolates the priority signal |

Verify with:

```
ps -eo pid,etime,command | grep "[s]nek2.py" | grep -v spawn_main
```

Update this section whenever runs start or stop — a future session reads it to know
what is in flight and might have been terminated.

### Why this batch is shaped like this

`b5c` and `b5d` each differ from `b5a`/`b5b` by **exactly one factor**, so they are
also near-repeats of `b4c`. That means:

- If `b4c`'s result is real, **all four should land high** — giving 4 seeds of evidence
  for the config plus the factor isolation for free.
- If only `b5a`/`b5b` land high, the factor that the low arm changed is the one
  carrying the gain.
- If none land high, `b4c` was a lucky seed and that is worth knowing before building
  anything else on it.

This is the same information two sequential batches would give, in half the wall-clock.

### Do not judge before ~850k steps

`b4c-schlongper` did not reach its best level (32% perfect) until the 850-900k block,
and it was **mid-collapse at 300k** — the horizon this protocol previously used would
have killed it. Budget **~8 hours per arm**. Expect to check in rather than watch.

### Finish the batch with 100-episode evals

Comparing these arms by their graph peaks would be the winner's curse (see
[`hyperparamTuning.md`](hyperparamTuning.md)). When the batch ends, run
`eval_checkpoints.py` on each arm's best few checkpoints and compare *those* numbers.
~3 minutes per arm, and it is the only apples-to-apples comparison available.

Only `b5a-schlong` has a visible window; the other three run with
`SDL_VIDEODRIVER=dummy`.

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

Per-run logs written to `$CLAUDE_JOB_DIR/tmp` are job-scoped and do not survive.
The durable record is `runs/<policy>_evals.json`; analyse from there.

## Batch 6 — after batch 5 reports

Shape depends on what batch 5 says, so decide then rather than now:

| if batch 5 shows | then batch 6 should |
|---|---|
| all four high | the config is solid — move to stacking: `b4c` config + 500k buffer, and `DISCOUNT=0.995` |
| only `td_loss` + no-IS high | the two factors interact; try alpha 0.6 and 1.0 with that pairing to find the peak |
| one factor clearly responsible | vary that factor alone across 3-4 values |
| none high | `b4c` was a lucky seed. Re-examine whether *anything* here beats the baseline reliably, and consider that arm-to-arm noise dominates every result in this document |

Standing candidates regardless:

| change | why |
|---|---|
| `b4c` config + `REPLAY_BUFFER_MAX_LENGTH=500000` | `b4b` beat `b4a` slightly, so diversity may stack on the winner |
| `DISCOUNT=0.995` | untested, high-prior; the perfect-game bonus is discounted to near-nothing at 0.99 |
| `GRADIENT_CLIPPING=10` | cheap, independent, and variance is what needs taming |

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
