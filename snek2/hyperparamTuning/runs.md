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

## The current record: 95%, and it runs on `master` today

`b10d-disc995seed4` @1815000 scored 95/100 (CI 88.8-97.8), measured mid-run, and is preserved in
[`../hallOfFame/`](../hallOfFame/README.md#the-current-record-95-trained-end-to-end-on-todays-environment-2026-08-03).
Unlike every earlier record in this project, **this one needs no checkout** — it trained on the
observation vector that is on `master` right now. Full batch results and the "config vs
environment" open question: [`completedRuns.md`](completedRuns.md#batch-10--a-fresh-baseline-and-a-new-project-record).

## No training is running — all four slots are free (two close-out evals are)

**Batch 10 was stopped 2026-08-03 by request**, all four arms healthy (no `dead_since` on any
of them), to make room for further changes. It was launched as a fresh baseline after seven
observation/reward changes landed 2026-08-02 (fatal-move zeroing, wall/body hugging, normalized
group count, the corrected starve/length split, the terminal-discount fix, safe-to-chase-food,
and the audit that started the day) — nothing had trained on the resulting environment before
it. All four seeds beat every prior post-audit result; full design and results:
[`completedRuns.md`](completedRuns.md#batch-10--a-fresh-baseline-and-a-new-project-record).

**Close-out evals: two done, two running.** `b10a` (272 checkpoints) and `b10c` (47) are
complete. `b10b` and `b10d` are still going, resumed 2026-08-03 at `EVAL_WORKERS=10` after the
worker-count measurement below — no training slots are involved either way.

| arm | checkpoints | best (100 episodes) | pooled | state |
|---|---|---|---|---|
| `b10a-disc995seed1` | 272 | 85.0% @2344000 | 67.2% | complete |
| `b10b-disc995seed2` | 624 | 90.0% @1501000 so far | 71.8% | ~58%, ~2h left |
| `b10c-disc995seed3` | 47 | 79.0% @3965000 | 63.0% | complete |
| `b10d-disc995seed4` | 660 | 93.0% @1695000 so far | 74.1% | ~53%, ~3h left |

`b10d`'s 93% @1695000 now beats the 95% @1815000 in the hall of fame on point estimate order
— except it doesn't, quite: the two intervals overlap almost entirely, and the 95% was itself
the max of ~300 noisy measurements. Both are the same policy family measured twice. See the
winner's-curse note in [`hyperparamTuning.md`](hyperparamTuning.md#why-not-abandon-weak-checkpoints-early)
before quoting either as a record.

### Evals got ~10x cheaper on 2026-08-03

Two changes, both measured rather than assumed:

| change | effect |
|---|---|
| `EVAL_WORKERS` 2 → 10 | **2.8x** faster per checkpoint (103s → 37s) at *lower* CPU per episode |
| `EVAL_SCREEN_EPISODES=20` | **3.6x** fewer episodes for a statistically indistinguishable answer |

The worker-count finding is a correction: this project had concluded eval throughput was
core-bound past ~10 workers and dropped batch 10's close-out to 2 workers to hit a ~50% CPU
target. That made it 2.8x slower *and* cost more CPU per episode, because TensorFlow's thread
pool costs about a core whether its batch has 2 rows or 20. **To be gentler on the machine, run
fewer arms, not fewer workers.**

The early-abandonment idea — cut a checkpoint once its running rate looks weak — was simulated
against all 937 batch-10 measurements and **rejected**: safe thresholds save only 14%, because
the selected population is a tight blob between 60% and 80% rather than a few good runs among
junk. Screening wins by economising on the many mediocre checkpoints instead of the few bad
ones. Full numbers in [`hyperparamTuning.md`](hyperparamTuning.md#screening-eval_screen_episodes).

A close-out is also resumable now (`EVAL_RESUME=1`), which is what made switching the worker
count mid-run cost only the checkpoint in flight rather than the 333 already measured.

### Next batch: pending the user's planned changes

The user wants to make additional changes before the next batch — unspecified as of this
close-out. Whatever they are, they will presumably retrain the environment (again) and reopen
the "config vs environment" question `completedRuns.md` flags for batch 10: nothing yet
isolates whether the 2026-08-02 fixes or just this seed cluster produced the 95%. Update this
section once the next batch is actually designed.

### Later candidates

Deferred pending the above. Still relevant to whatever comes after it:

| change | why | gate |
|---|---|---|
| second discount value (`0.9975`, or an interior point like `0.996`) | batch 9 left 0.995 vs 0.9975 unsettled; whether that survives the third environment is unknown | after the user's pending changes |
| `SNEK_LEARNING_RATE=1e-4` | the highest-value untested knob | after the user's pending changes |
| eff exponent ~1.4 (`td_loss` alpha 0.7) at 0.995 | `b4c` and `b7f` tie on ceiling; sharpness may still add on top of the discount | after the user's pending changes |
| best config + `REPLAY_BUFFER_MAX_LENGTH=500000` | `b4b` beat `b4a` slightly, so diversity may stack | after the user's pending changes |
| partial IS correction (beta < 1) | full correction cost `b5c` almost everything (2.1%); partial may keep stability without the cost | needs a new knob |
| anything aimed at the post-peak decline | both batch-8 arms peaked at ~2.5-3M and fell away; nothing tried so far addresses *why* | needs a mechanism first |

**Seed count is the binding constraint, not the number of knobs tried.** Six single-seed
conclusions in this document have been overturned or weakened — most recently gradient
clipping, which looked like batch 8's headline twice before failing at 1 of 3. Nothing goes in
[`findings.md`](findings.md) as established without n=3, which is why batch 10 spent all four
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
- **Resuming any arm from batch 9 or earlier** — every checkpoint on record from before
  batch 10 was trained on an observation vector this project has since changed (20 or 23
  values against the 26 batch 10 trained on), so none of them load; see
  [`../hallOfFame/README.md`](../hallOfFame/README.md#the-entries-below-predate-2026-08-02-and-do-not-run-on-master).
  Batch 10's own checkpoints *do* still load on `master` as of this close-out — see the
  note above about the user's pending changes for whether that keeps being true.

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
