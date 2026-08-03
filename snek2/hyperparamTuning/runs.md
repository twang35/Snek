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

## The best figure on record cannot be reproduced today

`b8f-disc9975seed2` @3149000 scored 82/100 and is preserved in
[`../hallOfFame/`](../hallOfFame/README.md). The observation vector changed from 20 to 26
values on 2026-08-02 — fatal-move zeroing, `lg(num_groups)` normalization, and a new
wall/body-hugging observation, on top of four earlier same-day changes (starve/length split,
terminal-discount fix, safe-to-chase-food, the audit itself). **No checkpoint on record loads
on the current environment, and no arm has ever trained on it.** 82% is the best figure that
exists, not a floor or a ceiling for what batch 10 should produce. Full history and the
82%-vs-73%-vs-92% picture: [`findings.md`](findings.md#what-this-cost-about-10-points-measured-properly).

## Nothing is running — all four slots are free

**Batch 9 finished 2026-08-02**, the day it launched, and settled nothing between the two
discounts (0.995 and 0.9975 each won on a different axis). Full results:
[`completedRuns.md`](completedRuns.md#batch-9--0995-against-09975-on-the-post-audit-environment).
Batch 9 also predates the seven observation/reward changes above, so its numbers are not
comparable to anything trained from here on either.

## Batch 10 — a fresh baseline on the changed environment

**Why a baseline batch, not another comparison.** Seven observation/reward changes landed on
2026-08-02, on top of the audit that already made batch 9 incomparable to batch 8. Nothing has
trained on the resulting environment at all. Launching straight into another discount
comparison — the plan queued before this session's changes — would repeat batch 9's own
lesson at one remove: comparing two things before either has a baseline on the environment
actually being measured. So batch 10 is four seeds of **one** config, chosen for reliability
rather than for testing anything new:

```
SNEK_DISCOUNT=0.995 SNEK_PRIORITY_EXPONENT=0.6 SNEK_PRIORITY_SIGNAL=td_loss SNEK_IS_WEIGHTS=0
```

| policy | role |
|---|---|
| `b10a-disc995seed1` | baseline seed 1 |
| `b10b-disc995seed2` | baseline seed 2 |
| `b10c-disc995seed3` | baseline seed 3 |
| `b10d-disc995seed4` | baseline seed 4 |

**Why `DISCOUNT=0.995` and not `0.9975`.** It is the single most reliably-surviving config on
record — 3 of 3 in batch 7, 2 of 2 in batch 9, 5 of 5 overall — against `0.9975`'s 1 of 2. The
discount question is deliberately not what this batch is for; a baseline should be boring, and
if the new environment reopens the discount question, that is a batch 11 problem with its own
seeds.

**What this answers.** Where "normal" sits on the current environment, at n=4 instead of n=1 or
n=0. Concretely: do this session's observation-space changes net out to better, worse, or
unchanged training relative to batch 9's two `0.995` seeds (52-70% best ckpt, 38.0-42.4%
pooled)? Every write-up from this session ended with "effect on the perfect rate is unmeasured
and needs an arm" — this is that arm, four times over.

**Run each to ~3-3.5M steps and stop**, per the horizon batch 8 established — do not leave
these unattended overnight (`b9b` reached 10.47M and spent 10.1M steps past its peak producing
nothing).

**Close out with `top20`** on each, then compare **best checkpoint** and **top-3 pooled**
across the four seeds — the same protocol as every batch since 7. Launch commands:
[`hyperparamTuning.md`](hyperparamTuning.md#launching-a-run). Watch with
`./watch_when_ready.sh <arm>`, launched right alongside the trainer — it waits out the
no-checkpoint-yet period itself rather than needing a timed retry by hand.

### Later candidates

Deferred until batch 10 gives the new environment a baseline to compare against:

| change | why | gate |
|---|---|---|
| second discount value (`0.9975`, or an interior point like `0.996`) | batch 9 left 0.995 vs 0.9975 unsettled; whether that survives the new environment is unknown | after batch 10 |
| `SNEK_LEARNING_RATE=1e-4` | the highest-value untested knob | after batch 10 |
| eff exponent ~1.4 (`td_loss` alpha 0.7) at 0.995 | `b4c` and `b7f` tie on ceiling; sharpness may still add on top of the discount | after batch 10 |
| best config + `REPLAY_BUFFER_MAX_LENGTH=500000` | `b4b` beat `b4a` slightly, so diversity may stack | after batch 10 |
| partial IS correction (beta < 1) | full correction cost `b5c` almost everything (2.1%); partial may keep stability without the cost | needs a new knob |
| anything aimed at the post-peak decline | both batch-8 arms peaked at ~2.5-3M and fell away; nothing tried so far addresses *why* | needs a mechanism first |

**Seed count is the binding constraint, not the number of knobs tried.** Six single-seed
conclusions in this document have been overturned or weakened — most recently gradient
clipping, which looked like batch 8's headline twice before failing at 1 of 3. Nothing goes in
[`findings.md`](findings.md) as established without n=3, which is why batch 10 spends all four
slots on one value rather than splitting them.

## Standing backlog

Untested, ordered by expected value. Rationale for the ones that need it follows the
table.

| change | targets | prior |
|---|---|---|
| `LEARNING_RATE=1e-4` | training speed | high, but order it after a stability fix |
| `TARGET_UPDATE_PERIOD=50` / `500` | early learning speed | medium — 2 points to test a hinted trend |
| `TARGET_UPDATE_TAU=0.005`, period 1 | smoothness (soft target updates) | medium |
| `FC_LAYERS=128,128` | capacity | low |
| epsilon ladder *shape* (not floor) | exploration schedule | low |
| `REPLAY_BUFFER_MAX_LENGTH=1000000` | experience diversity | low — the 500k result was ambiguous |

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
- **Resuming any arm from batch 9 or earlier** — every checkpoint on record was trained
  on an observation vector this session changed (20 or 23 values against today's 26), so
  none of them load; see [`../hallOfFame/README.md`](../hallOfFame/README.md#these-checkpoints-do-not-run-on-master-2026-08-02).
  A fresh seed is the only option now regardless of how promising an old arm looked.

### Batch bookkeeping

Each batch keeps its **description** — why it is shaped that way, what each arm isolates,
what outcome would mean what — in this file for as long as any of its arms is running.
When the last arm of a batch stops, move that description and its results to
[`completedRuns.md`](completedRuns.md) and delete it here.

The reason to keep the description live rather than only the status table: the design
rationale is what tells a future session whether a surprising result is informative or
just an arm that was never going to answer anything.

Verify what's actually running with `pgrep -fl "python -u snek2.py"`. Not
`grep "[s]nek2.py"` — git telemetry `curl` processes carry `snek2/snek2.py` in their
payload and inflate the count.
