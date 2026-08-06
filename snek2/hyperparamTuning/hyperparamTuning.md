# Hyperparameter tuning protocol

Long-running, resumable investigation into what makes snek2 reach the **highest
possible perfect-game percentage**, and get there by **learning consistently**.
Those two are the objective. Catastrophic forgetting matters as one of the things
that breaks consistency, not as a target in its own right.

These files are the handoff. A fresh Claude Code session should be able to read
them and continue with no other context. Keep them current: they are worth more
than any single run.

| file | contents | changes |
|---|---|---|
| `hyperparamTuning.md` (this file) | the protocol: goals, metrics, stop criteria, how to judge, how to launch, available knobs | rarely |
| [`runs.md`](runs.md) | what is running, what to run next | constantly |
| [`completedRuns.md`](completedRuns.md) | every finished arm: config, final numbers, verdict | when an arm finishes |
| [`findings.md`](findings.md) | what is established, what has been falsified | when something is learned |
| [`failureModes.md`](failureModes.md) | the four ways a policy degrades, and how to tell them apart | rarely |
| [`charts.md`](charts.md) | progress graph for every arm, with captions | when charts are refreshed |
| `charts/` | snapshot copies of the graphs | via `refresh_charts.sh` |

**Start with [`runs.md`](runs.md)** if you are picking this up mid-flight — it says
what is in progress and what to do next. Read this file for how the machinery works,
and [`findings.md`](findings.md) before proposing an experiment, so a closed question
doesn't get reopened.

---

## The goal, and why it's hard

The end target is perfect-game percentage, reached by learning **consistently** — a rate
that rises and keeps rising, not a lucky spike.

**Best config level: ~31%** (`b4c-schlongper`, alpha 0.8 / `td_loss` / no IS, over 1400
episodes), whose best checkpoint is **851000 at ~40%**. But that config **dies in 2 of 3
seeds**, so its expected value is ~10.6%; `b6b-alpha06` (alpha 0.6, same otherwise) is the
better bet at 24.5% and survived. The committed config reaches ~1%, and the whole gap was
three PER changes made during the cpprb port. See [`findings.md`](findings.md).

The often-quoted "51% at checkpoint 869000" is **superseded**: that checkpoint pools 41.7%
over 300 episodes, and 51% was the high draw of three separate measurements.

Three things make tuning awkward, and each has cost this investigation a wrong
conclusion:

- **High noise.** Two runs of the identical config reached final avg_score 62.5
  and 18.0 at 30k steps. Any single-run comparison is meaningless. Promising
  configs must be repeated **2-3 times**. The noise extends to *whether a run
  collapses at all* — see [`failureModes.md`](failureModes.md).
- **Non-linear trajectories.** Sometimes nothing much happens until ~40k
  iterations and then the slope turns steep. A config that looks dead at 20k may
  not be. Do not judge a config on a short run alone.
- **Nothing is judgeable below ~250k steps, and the best arm peaked at 875k.**
  Degradation begins somewhere between 236k and 312k in nearly every arm observed,
  so a verdict formed earlier is measuring the pre-degradation phase only. The
  investigation's own stop criteria would have killed its best arm at 300k.

### When to keep a run going, and when to stop it

Judge on **trajectory, not absolute level**. A run is worth continuing while
*either* its perfect-game rate *or* its average score is still climbing — both are
valuable, and a config that is behind but rising may overtake one that plateaued.
Check both: they can disagree, and have.

Concretely, compare a trailing window against the window before it (last 20 evals
vs the previous 20 for score, last 30 vs previous 30 for the coarser perfect-game
rate). Stop an arm when neither is rising, or when its remaining question is
already answered.

**The 4-hour cap this protocol used to recommend was too short.** `b4c-schlongper` was
mid-collapse at 300k (~2.5h) and did not reach its best level until the 850-900k block
(~8h). A 4-hour cap would have killed the best arm in the investigation. Budget **~8
hours** for any arm whose config has a plausible shot, and treat a mid-run collapse as
uninformative rather than terminal.

That cost is real, so spend it deliberately: run arms overnight, prefer batches where
every arm answers something even if it loses, and don't burn 8 hours on a config with no
mechanism behind it. If an arm must be cut early, record it as **"promising but too
slow"** rather than as a failure — that is a genuinely different verdict from "does not
learn".

Stopping is cheap and reversible: relaunching the same policy name continues its
graph and checkpoint. Always re-supply the same `SNEK_*` overrides when resuming,
or the config silently changes mid-run.

Because of that, avoid the final eval as a metric. Prefer, in rough order of
usefulness:

| rank | metric | what it tells you | when it lies |
|---|---|---|---|
| 1 | **perfect % over last N evals** | the objective itself | very coarse early — one perfect game reads as 10% |
| 2 | **steps to first perfect game** | how fast a config gets into the region that matters | nothing about the ceiling |
| 3 | **last-5-eval avg score** | leading indicator; the workhorse for comparisons | **decouples from the objective past ~250k** |
| 4 | **mean over the whole curve** | rewards learning early *and* holding on | punishes slow starters that end high |
| 5 | **steps to first reach score N** | learning speed | — |
| 6 | **max drawdown** | diagnostic for *why* a config is erratic | can't tell noisy-but-high from collapsed |

### Measuring a policy properly: `eval_checkpoints.py`

Everything above is for comparing *runs*. To measure what a specific **policy** actually
does, reload its checkpoint and evaluate it over hundreds of episodes:

```
cd snek2
EVAL_WORKERS=10 EVAL_OUT_SUFFIX=_top20 \
  PYTHONPATH=. python -u eval_checkpoints.py b4c-schlongper top20
```

**Screening is on by default** and is **3.6x cheaper than measuring every selected checkpoint at
100 episodes, for the same answer** — see "Screening" below. `EVAL_SCREEN_EPISODES=0` gets the flat
one-pass protocol every arm before batch 10 was measured under.

**Early abandonment is also on by default, from 2026-08-05** (`EVAL_MIN_ACHIEVABLE`, **90 since
2026-08-06**, 85 before that): a checkpoint stops being measured the moment it cannot reach the gate
even if every remaining episode is perfect. Measured on batch 13's first 505 full-length rows, the
85% gate cuts full-length work to **70%** and the 90% gate to **52%**. It cannot change any ranking
among rows that reach the gate — the test is arithmetic, so a checkpoint that would have reached it is
never stopped, and an abandoned row's own rate is always below it. `pooled_equal_effort` truncates to
the screen depth, so it is untouched at any gate. `EVAL_MIN_ACHIEVABLE=0` restores the old behaviour.

**Why 90, and the one thing it costs.** The project is chasing 95%+, so a checkpoint in the 85-89%
band is not a candidate and does not need 100 episodes to be ruled out. The cost is that
**best ckpt degrades on a weak arm**: an arm that never clears 90% has no full-length row at all, and
3 of the 8 arms measured across batches 11 and 13 peaked below 90% (`b11c` 87%, `b11d` 88%, `b13a`
80%). `best_full_length_row` then relaxes to **half-depth** rows — never to all rows, which would
crown a 20-episode screen on a lucky 20/20 — and prints `[truncated]` when it does.

**Check `min_achievable` in the payload before comparing best-checkpoint across batches.** Batches 11
and 13 were measured with no gate at all, so their sub-90% rows are full length where batch 14's are
truncated. Above the gate the comparison is exact; below it, it is not a comparison.

`top20` (or `top`, `top:N`) is the normal way to close out an arm. It ranks on the **single
10-episode eval** from the graph, using the surrounding perfect rate to order within an
equal-eval tier, and applies two thresholds:

| rule | threshold | effect |
|---|---|---|
| always measure | single eval **>=90%** | every such checkpoint runs, even past N |
| fill remaining slots | **>=60%**, best first | at 10-episode granularity this is {60, 70, 80} |
| never measure | below **60%** | skipped entirely, however few slots are filled |

**N is a target, not a quota.** A graph point is 10 episodes, so `perfect_percent` only takes
values 0, 10, … 100 — which makes those thresholds coarser than they look. `>=90%` is the set
{90, 100} and the fill band is exactly {60, 70, 80}. `b8f-disc9975seed2` has 32 checkpoints at
>=90% and runs all 32; `b8e-clipseed2` has one point above the floor in 1165 evals and runs one.

Explicit steps still work (`... b4c-schlongper 869000 871000`) when a specific checkpoint is
the question, and they bypass both thresholds.

#### Screening: `EVAL_SCREEN_EPISODES` (on by default)

Three stages instead of one:

1. every checkpoint whose **graph point is 100%** (ten perfect games out of ten) gets the full
   **100 episodes** immediately. Uncapped, and large on a strong arm — 47/142/7/146 across batch
   10's four arms. Explicitly named steps join this tier, since naming one is a request to measure it.
2. everything else selected gets **20 episodes**
3. the best **100** (`EVAL_CONFIRM_COUNT`) **of those screened** get **80 more**, reaching 100

A promoted checkpoint ends with exactly 100 episodes — the screen counts toward the total — so its
number is directly comparable with every arm measured under the flat protocol. Checkpoints that
miss the cut keep their 20-episode row, whose much wider Wilson interval says how little it is worth.
Confirmation slots exclude the 100% tier, which already has the measurement a slot would buy.

**Cost on a batch-10-shaped arm**, against measuring all 660 selected checkpoints at 100:

| arm | selected | at 100% | screened | episodes | vs flat |
|---|---|---|---|---|---|
| `b10a` | 272 | 47 | 225 | 11,600 | 2.34x |
| `b10b` | 624 | 142 | 482 | 26,240 | 2.38x |
| `b10c` | 47 | 7 | 40 | 3,900 | 1.21x |
| `b10d` | 660 | 146 | 514 | 27,280 | 2.42x |

Those figures are at the old `EVAL_CONFIRM_COUNT=30`; at 100 the b10d plan is 29,980 episodes and
2.20x.

`EVAL_SCREEN_EPISODES=0` returns to the flat protocol.

**The 100% tier is coverage, not a shortlist of likely champions.** Measured across batch 10:

| training graph point | n | mean measured | max measured |
|---|---|---|---|
| 90% | 1007 | 71.2% | **93%** |
| 100% | 270 | **73.0%** | 89% |

1.8pp better on average, but a *lower* maximum, and **all four arms' best checkpoint came from the
90% tier** — only 1-2 of each arm's top 10 were graph-100%. The 90% tier is ~4x larger, so the max
of the bigger group wins. Stage 3 is what looks for the champion.

Batch 11 weakens the "all four" but not the conclusion: **2 of its 4 best checkpoints came from the
90% tier**, including the project record (`b11b` @855000 at 96%, whose graph point read 90). Across
the eight arms now measured that is 6 of 8, so the rule to act on stays "the 100% tier is coverage
and the confirm stage finds the champion" — which is also the argument for keeping
`EVAL_CONFIRM_COUNT` high.

**`EVAL_CONFIRM_COUNT` is 100, raised from 30 on 2026-08-03**, because 30 was losing the champion
outright. Simulated on b10d, the best non-100% checkpoint reaches the confirm set only 57% of the
time at 30:

| confirm count | champion recall | episodes | vs flat |
|---|---|---|---|
| 30 (the old default) | 57% | 24,380 | 2.71x |
| 50 | 85% | 25,980 | 2.54x |
| **100 (current)** | **97%** | 29,980 | **2.20x** |
| 150 | 99% | 33,980 | 1.94x |

100 is the knee: a coin flip on the headline number becomes near-certainty for ~23% more episodes.
There is a floor on how far this can go — below about 2x a flat pass is simpler and gives *every*
checkpoint a real measurement, and `test_default_confirm_count_is_100_and_still_pays_for_itself`
fails if a future increase crosses it.

**A fixed count of 100 degenerates on a small arm, and batch 11 hit it.** The saving comes entirely
from checkpoints that screen and are then *dropped*, so once the screened pool is smaller than
`EVAL_CONFIRM_COUNT` every checkpoint is confirmed anyway and the 20-episode screen becomes pure
overhead — the run pays 120 episodes per checkpoint to measure 100. Measured across batch 11's
close-outs:

| arm | at 100% | screened | confirmed | episodes | vs flat |
|---|---|---|---|---|---|
| `b11b` | 104 | 272 | 100 | 23,840 | 1.58x |
| `b11a` | 48 | 181 | 100 | 16,420 | 1.39x |
| `b11d` | 40 | 194 | 100 | 15,880 | 1.47x |
| `b11c` | 23 | **87** | **87** | 11,000 | **1.00x** |

`b11c` screened 87 checkpoints against a confirm count of 100, so all 87 were promoted and it ran
**exactly** a flat pass — 11,000 episodes either way. That is 1.00x, not worse than flat: the screen's
20 episodes count toward the promoted checkpoint's 100, so a fully-promoted arm wastes no episodes.
What it does waste is *measurements* — 197 instead of 110, so 87 extra checkpoint restores and round
set-ups for nothing. Small next to the episodes, but not free.

None of the four reached the 2.2x the b10d projection implied, because batch 11's arms had smaller
selected pools. The projection was taken from the largest arm of the previous batch and should have
been quoted as a range.

**The knob would be better expressed as a fraction of the screened pool** (`min(100, 0.4 * screened)`
or similar), which would have put `b11c` at ~1.6x while leaving the large arms untouched. Not
implemented. The fixed count is still right for *choosing* the champion — that is what the 97% recall
figure above buys — it is only the saving that collapses.

#### Taking the arm-level pooled rate

**Take it from the equal-effort figure the run prints.** Do not pool the
rows in the output file: they have different episode counts, and the deep ones are by construction
the arm's best, so pooling them weights the winners 5x and reads high however good the policy is.
The printed figure truncates every checkpoint to its first 20 episodes, which is a valid sample of
each and lets the 100% tier count too. Best-checkpoint is taken over full-length rows only, since
across hundreds of 20-episode screens some will read 19/20 on luck.

**If the run did not print it, the equal-effort rate is gone — use the graph-100% tier instead,
but only for a run measured without an abandonment gate.** The output file stores per-checkpoint
totals, not per-episode results, so the first-20 prefix of a 100-episode row cannot be reconstructed
afterwards; `pooled_equal_effort` has to be computed in the process that ran the episodes. Batch 11's
four close-outs predate that field and have no arm-level rate for exactly this reason. The fallback is
to pool the **graph-100% tier only**: with no gate, every checkpoint in it gets 100 episodes with no
screening and no selection applied, so it is unbiased within itself and comparable across arms. It
answers a narrower question — how good is a checkpoint the graph called perfect — and its episode
counts are much smaller, but it is a real number rather than a biased one.

**‡ `EVAL_MIN_ACHIEVABLE` destroys the graph-100% tier, and this was missed when the gate shipped.**
The tier's whole claim is "no selection applied", and a gate *is* a selection: only tier members that
clear the gate stay at 100 episodes, so the pooled figure is censored from below. Batch 14 makes the
size of the artifact obvious:

| arm | graph-100% tier, gated | batch 13's, ungated |
|---|---|---|
| seed 1 | 91.0% over 500 ep / 5 ckpts | 70.5% over 3100 ep / 31 ckpts |
| seed 4 | 90.5% over 2800 ep / 28 ckpts | 77.2% over 4800 ep / 48 ckpts |

That reads as +15.6 pp across the batch and means nothing — the tier shrank from 31-114 checkpoints
per arm to 1-28, because the rest were abandoned. **Never compare a gated tier figure to an ungated
one, and never quote a gated one at all.** `pooled_equal_effort` is the metric to use: it truncates
to the screen depth, abandonment cannot fire before the floor, so it is exact at any gate. Check
`min_achievable` in the payload before using the tier for anything.

**The winner's-curse shrinkage is collateral damage from the same cause.** It was fitted on each
arm's *unselected* graph-100% rows, and under a gate those rows are abandoned and downward-biased by
optional stopping. A shrunk figure cannot be computed for a gated arm by the existing method. The
substitute is a second independent 100-episode measurement of the champion, which is worth more
anyway — see `b14a` in [`../hallOfFame/README.md`](../hallOfFame/README.md).

**Ranking uses the screen rate, ties broken on the surrounding graph rate.** 20 episodes admit
only 21 distinct values, so ties are the common case and the tie-break does real work — the
surrounding rate correlates +0.48 with the true rate inside the high band where the graph point
itself manages +0.10.

#### Watching a run in progress: `eval_progress.py`

A close-out is 20-50 checkpoints at ~4 minutes each, usually split across several parallel
processes, so "how is it going and how long is left" is not answerable from the logs without
reading all of them. This renders one consolidated view instead:

```
cd snek2
PYTHONPATH=. python -u eval_progress.py b8f-disc9975seed2
EVAL_PROGRESS_WATCH=30 PYTHONPATH=. python -u eval_progress.py b8f-disc9975seed2
```

It prints a text summary and writes `evals/<policy>_eval_progress.png`. **Prefer the text output
when checking programmatically** — it carries the same numbers without opening an image. Watch
mode exits by itself once every run is complete, so it does not become a process to remember.

**`evals/` holds only the current eval or batch.** `eval_checkpoints.py` moves whatever is
already there into a timestamped folder under `evals/archive/` before it writes anything new,
so glancing at the top-level folder always shows exactly what last finished — no need to check
step numbers or timestamps to tell current from stale. Nothing is lost; look in `evals/archive/`
for a previous batch's charts.

**Launching a batch of evals together does not make them archive each other.** The archive
step happens before any process has written a chart of its own, so a simultaneous batch only
ever clears the *previous* batch, verified by actually running four at once. Starting one more
eval while an earlier one is still running is the one case that visibly does something: the
new process archives the still-running one's current chart along with everything else, since
it cannot tell "mid-run" from "finished". That arm's chart reappears at the top level within
one round regardless, so nothing is lost — but a chart can look briefly archived while its arm
is still very much alive.

| part | shows |
|---|---|
| in-flight chart | running perfect rate vs round, one line per process — is the current checkpoint any good, and how many rounds left |
| completed chart | every finished measurement by checkpoint step, with best and pooled marked |
| text block | progress bar, top 5 checkpoints, pace, ETA |

Three things worth knowing about it:

- **It reports the current *job*, not the arm's lifetime.** Result files written within
  `EVAL_PROGRESS_WINDOW` seconds (default 3600) of the newest one are grouped as one job, which
  captures the parallel chunks and excludes earlier sessions. `EVAL_PROGRESS_ALL=1` pools
  everything, which double-counts any checkpoint measured twice.
- **The ETA comes from this run's own pace**, not a constant. Strong policies play longer
  episodes and measure slower, so a fixed estimate is wrong in both directions.
- **It flags a run as STALE** if its file has not been written for 180s while incomplete. A
  killed process leaves its last in-flight state behind and would otherwise be drawn forever as
  though still working.

It is a separate script rather than a chart inside `eval_checkpoints.py` because six processes
each drawing their own window would be six partial pictures of one job, and writing one shared
PNG would have them overwrite each other.

### The primary metric: `strong_eval_fraction`, the share of an arm's evals at >=80%

**Changed 2026-08-04, from `best_perfect30`.** Both are in `runs/<policy>_evals.json`'s summary
block; the old one is kept because every arm through batch 11 is recorded on it.

The reason is variance, not taste. Measured across batch 11's four *identical* configs, the
between-seed spread of each candidate metric — and what that spread implies a batch can resolve:

| candidate primary metric | sd across 4 identical seeds | detects at n=4 | n=8 | n=12 |
|---|---|---|---|---|
| **fraction of evals >=80%** | **5.8** | 10.2 pp | **7.2 pp** | **5.9 pp** |
| mean perfect over the last half of the run | 6.3 | 11.2 pp | 7.9 pp | 6.4 pp |
| best checkpoint, 100 episodes | 7.3 | 12.9 pp | 9.1 pp | 7.4 pp |
| `best_perfect30` (the old primary) | 8.6 | 15.1 pp | 10.7 pp | 8.7 pp |
| graph-100% tier rate | 9.3 | 16.4 pp | 11.6 pp | 9.5 pp |

**Switching buys a ~40% tighter detectable effect for zero extra compute.** Two reasons it behaves
better: `best_perfect30` is a **max statistic**, and maxima inflate variance — it reports the
single luckiest 30-eval window an arm ever had. And a share-of-strong-evals measures *sustained*
competence, which is closer to the stated goal ("the highest perfect rate while learning
consistently") than one good window is.

**It is a fraction of each arm's own evals, so a common step horizon is mandatory**, not just good
practice — the denominator grows with run length and a long declining tail drags it down. That is
intended behaviour (the decline is real and should cost the arm something), but it means the
horizon has to be fixed before comparing, exactly as the pre-registered best-30 comparison already
required.

**The honest ceiling.** At these variances, detecting a **5 pp** effect needs n≈17 arms per group
on the new metric and n≈37 on the old one. Nothing feasible here resolves 5 pp on `best_perfect30`,
and an earlier claim in [`runs.md`](runs.md) that "n=12 at 2M would detect ~5 pp" was true only for
a low-variance metric — on best-30 it is 8.7 pp. Choosing the metric is the only lever that moves
this materially; adding arms runs into a square root.

### Comparing arms fairly

**Always compare at matched step counts, never by wall-clock or by "where they got
to".** Two confounds make elapsed time useless:

- **Eval cost scales with policy quality.** A better policy eats more food, so its
  episodes are longer, so its 10-episode eval takes longer. `b1c-nstep3` reached 161k
  steps while its batch mates were at ~68k in the same elapsed time — purely because it
  was worse and its evals were cheap.
- **The same confound runs in reverse for a dead policy.** Score 0 means episodes end
  instantly, so evals become nearly free: `b3c-buf500k` raced to 4.81M steps while its
  batch mates did ~1.2M, because it had died. **A step count far ahead of its batch
  mates is a symptom to investigate, not progress.**

Matched-step comparison is right for fairness but **blind to momentum**. An arm that is
behind at matched steps may still be the one improving. Check the trajectory as well as
the level — this cost one wrong verdict, on `b1c-nstep3`, in both directions.

**Never call a trend from the most recent window.** `b2a-base2` oscillates with a
~100k-step wavelength, which produced two contradictory write-ups 80k steps apart. A
20-30 eval trailing window cannot see that; only the 50k-block table can. See
[`failureModes.md`](failureModes.md).

---

## Running the experiments

### Slot budget

**Never exceed 4 snek trainers at once, including any the human started.** Each
one spawns 9 headless eval workers on top of its main process, so 4 runs is ~40
processes on 14 cores. Check first:

```
pgrep -fl "python -u snek2.py"
```

Do **not** count with `ps ... | grep "[s]nek2.py"`: git telemetry `curl` processes embed
`snek2/snek2.py` in their JSON payload as a git argument and match it, inflating the
count for the few seconds each curl lives. That pattern reported 6 trainers once when
only 4 were running, which would have looked like the cap was already blown.

A human-started `python snek2.py train` counts against the budget. Leave it
alone; never touch `snek2/savedPolicies/train*`.

### Launching a run

Hyperparameters are overridden through `SNEK_*` environment variables, read by
`tuned()` in `snek2.py`. No file edits are needed to vary a config, so several
policies can run side by side from the same code.

```
cd snek2
SNEK_TARGET_UPDATE_PERIOD=200 \
  PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u snek2.py b1b-tgt200 \
  > /tmp/b1b.log 2>&1 &
```

Notes that matter:

- Use the env's python binary directly, **not `conda run`** — it buffers stdout
  and makes a healthy run look hung. See `CLAUDE.md`.
- **Runs log quietly.** One compact line per 10 evals; `SNEK_DEBUG=1` restores the full
  original output for debugging. Take status from the `summary` block in
  `runs/<policy>_evals.json` rather than from the log — it precomputes step, trailing,
  peak, best 30-eval perfect window and `dead_since`, which is everything a progress
  check needs. `dead_since` is an onset, not a verdict; compare it to `step` for duration.
- **Every arm gets a progress-chart window**, titled `<policy_name> results` — the score and
  perfect-rate graph, redrawn each eval. Name arms so they are identifiable at a glance.
- **No arm draws a game.** `display_eval` and `SNEK_DISPLAY_EVAL` are gone, and `snek2.main()`
  sets `SDL_VIDEODRIVER=dummy` itself, so there is nothing to pass and nothing to switch on.
  A game window cost ~5.2ms per frame and the game flips once per game step, which is why it
  left the training loop entirely. **To watch a policy play, run `watch.py <arm>`** — it
  follows the arm's newest checkpoint in its own process and costs training nothing
  (measured: 4 watchers alongside 4 arms had no detectable effect on throughput).
- **A freshly-launched arm has no checkpoint yet**, since training skips writing one until
  the score clears `SNEK_MIN_CHECKPOINT_SCORE` (default 40) — `watch.py` run immediately
  after the trainer just exits with "no checkpoints in …". Use
  `./watch_when_ready.sh <arm>` instead when starting a watcher alongside a fresh arm: it
  polls every 30s (`WATCH_WHEN_READY_POLL` to change that) and execs into `watch.py` the
  moment the first checkpoint appears, so it can be launched in the same breath as the
  trainer rather than timed by hand. In practice the wait is short — batch 10's four arms
  all had a checkpoint within ~12k steps.
- Every override prints a `hyperparameter override:` line at startup. Grep for it
  to confirm a run really got the config you intended — this has already caught
  one silently-misconfigured control arm.

### Available knobs

| env var | default | notes |
|---|---|---|
| `SNEK_LEARNING_RATE` | 1e-5 | in-code comment suggests trying 1e-4 |
| `SNEK_BATCH_SIZE` | 128 | |
| `SNEK_DISCOUNT` | 0.99 | horizon ~100 steps; perfect games are much longer |
| `SNEK_TARGET_UPDATE_PERIOD` | 8 | very frequent for DQN; prime suspect for forgetting |
| `SNEK_TARGET_UPDATE_TAU` | 1.0 | 1.0 = hard copy; <1 = soft/Polyak updates |
| `SNEK_GRADIENT_CLIPPING` | 0.0 (off) | norm clip; 0 disables |
| `SNEK_N_STEP_UPDATE` | 1 | n-step returns; buffer window is n+1 automatically |
| `SNEK_INITIAL_EPSILON` | 0.4 | where the bootstrap phase starts; the refinement ceiling is this / 32 |
| `SNEK_MIN_EPSILON` | 0.002 | floor. **0 is rejected**, as is any value at or above `INITIAL_EPSILON / 32` |
| `SNEK_GUIDED_FRACTION` | 0.5 | share of refinement-phase episodes whose epsilon move avoids fatal actions; 0 disables the shield |
| `SNEK_MAX_STEPS` | 10000000 | **absolute** step at which training stops, so a wave self-terminates |
| `SNEK_FC_LAYERS` | 50,100,50 | comma separated |
| `SNEK_REPLAY_BUFFER_MAX_LENGTH` | 100000 | |
| `SNEK_PRIORITY_EXPONENT` | 0.6 | alpha; 0.0 disables prioritization |
| `SNEK_PRIORITY_SIGNAL` | `td_error` | or `td_loss` (element-wise Huber), which is what `theSchlong` used |
| `SNEK_IS_WEIGHTS` | 1 | 0 disables importance sampling entirely, as `theSchlong` did |
| `SNEK_IS_BETA` | 0.4 | ignored when `IS_WEIGHTS=0` |
| `SNEK_BETA_ANNEAL_STEPS` | 1000000 | |
| `SNEK_INITIAL_POPULATE_STEPS` | 1000 | |
| `SNEK_MIN_CHECKPOINT_SCORE` | 40.0 | below this a checkpoint is not written at all |

**`SNEK_MAX_STEPS` is absolute, and it is what makes an unattended wave safe.** Added
2026-08-05; before it, `num_iterations` was hardcoded to 1e9 and every batch had to be stopped by
hand. `b9b-disc9975b` is why it exists: it ran **10.1M steps past its peak** overnight with nobody
watching. The default is deliberately generous, so it is a backstop rather than a planned horizon.

**The default was raised 5M → 10M on 2026-08-06**, because 5M is closer to the useful range than it
looked. In batch 14, two of four arms produced their best window past 3.5M — `b14a`'s best trailing
score at 3.79M, `b14c`'s best 30-eval perfect rate at 4.14M — and `b14c` was still gaining in its
final 4.0-4.5M band (75.9% perfect, its best of the run, against a 62.6% previous best). The other
two were past peak. So one arm in four would have been truncated mid-climb by a 5M stop.

The old "best checkpoint lands between ~1M and ~3.4M" rule of thumb came from batches a human
killed around 3.5M, so it partly described the stopping habit rather than the learning curve.

Absolute means "stop when `global_step` reaches this", not "run this many more steps", because
`global_step` is restored on resume; a relative count would let an arm resumed at 4M run to 9M. An
arm already at its cap prints `already at or past the N-step cap` and exits after its opening eval.

One interaction worth knowing: an arm whose score never clears `SNEK_MIN_CHECKPOINT_SCORE` never
writes a checkpoint, so it cannot resume at all and its cap counts from 0 again. That is
pre-existing behaviour, but it means a *cap* on such an arm is per-launch in practice.

Rewards (`snake_constants.py`) are deliberately **held fixed** so `avg_score` stays
comparable across every run. Changing them invalidates comparison with everything
already recorded here.

### The epsilon schedule — rewritten 2026-08-04, and it breaks curve comparability

`training.epsilon_for()` has two phases. Neither is a ratchet: epsilon is a pure function of
the current eval, so it can rise again.

| phase | driven by | range | window |
|---|---|---|---|
| bootstrap | `avg_reward` past 5 / 10 / 20 | `INITIAL_EPSILON` → /4, then stands down | trailing 5 evals |
| refinement | trailing perfect rate, 0 → 80% | `INITIAL_EPSILON`/8 → `MIN_EPSILON`, geometric | trailing 30 evals |

The two are combined with `max()`, so whichever phase is live controls epsilon. Each eval row
records the value in `runs/<policy>_evals.json`.

**Arms from 2026-08-04 onward are a different config, not a different environment.** Checkpoints
still load; the observation vector is untouched. Epsilon only shapes the collected data, so it
changes *which policy you get at step N* and therefore moves every column — but every metric here
is a greedy eval either way, so a post-rewrite batch compares to batch 11 as a controlled test of
the schedule, in exactly the way batch 11 compared to batch 10. **What is not legitimate is
pooling the two as seeds of one baseline.**

#### What was wrong with the ladder it replaced

The old version ratcheted down fixed rungs (0.2, 0.1, 0.05, 0.01, 0.001, then 0.0) on
`avg_reward`, one rung per eval, and never back up. Measured across batches 10 and 11 — 8 arms,
31.1M steps:

| epsilon | share of all training steps | first reached (median) |
|---|---|---|
| 0.4 → 0.01 | 0.37% | steps 0 → 13k |
| 0.001 | 2.79% | 15k |
| **exactly 0.0** | **96.83%** | 86k |

**The rungs were calibrated to sub-beginner skill.** `avg_reward > 60` → 0.001 translates to
"eats 65 of 95 food, then dies, never wins". At the moment epsilon hit 0.001 the arms were at
`avg_score` 57-76 with **0% perfect games in 7 of 8**. The whole perfect-game learning phase,
which is 99% of a run, happened at epsilon ≈ 0. **33 of the 42 arms on record reached exactly
0.0**, and the median arm spent 2.38% of its life above 0.001.

Three specific defects, each fixed:

- **Score saturates where the ladder spends its rungs.** `avg_score` goes 0 → 70 in 13k steps
  and 70 → 95 over the next 3M, so every rung fired inside the first 2% of the journey.
  Perfect rate spans 0 → 96% across the whole run, which is why the refinement phase uses it.
- **One-way ratchet on a 10-episode signal.** A single eval crossing a threshold pinned epsilon
  permanently: `b11b` sat at 0.001 while its score collapsed 64.6 → 8.8. Both phases are now
  pure functions of the current estimate.
- **Exactly 0 was reachable, and was where almost all training happened.** Now rejected at
  startup rather than clamped, so a `SNEK_MIN_EPSILON=0` override fails loudly.

#### Why geometric, and why the floor is small

The useful range spans more than an order of magnitude, so equal *ratios* matter rather than
equal differences: 0.05 → 0.025 changes behaviour as much as 0.004 → 0.002. A linear ramp sits
above 0.02 for more than half its length — a forced random move every ~40 steps — then crosses
the entire low range in its last few percent.

The floor is small because a random action is nearly free early in an episode and usually fatal
late, and the cost scales with length: at epsilon 0.01 a 1780-step perfect game absorbs ~12
forced non-greedy moves. 0.002 is ~2.4.

#### The handover moved from 0.05 to 0.0125 on 2026-08-05, and why

The original handover was 0.05, on the reading that `b11d` reached a 10% perfect rate at step 13000
while collecting at epsilon 0.05. **Batch 12 falsified that**: 0.05 is fine to pass *through* and
fatal to *sit at*.

| evidence | at epsilon 0.05 |
|---|---|
| batch 12, 4 arms to ~1M | 0% perfect games, greedy trailing 53-63 vs batch 11's 84-88 |
| worst case | `b12b` pinned 942k steps, **0 perfect games in 1032 evals** |
| with the exploration shield | decay fixed, but plateaus at trailing ~83 with 0.3% perfect |
| improvement rate, shielded | 4.7 trailing points per 100k vs `b11a`'s 11.1 |

The mechanism is that a collect policy at 3.3% random actions per step essentially never finishes
a board, so the buffer holds no trajectories that eat the last ~10 food and the greedy policy
cannot learn them. The shield makes exploration *survivable* without making the endgame
*completable*, because a one-step mask cannot prevent self-trapping.

Two rungs were added **below** the existing ladder rather than above — thresholds are now
`(2, 5, 10, 15, 20)` — so the handover falls to `0.4 / 2**5` while every threshold that drops
epsilon still sits in the pre-winning regime. Pinned at the new ceiling, an arm forces a
non-greedy move 0.83% of steps against 3.3% before.

#### What is still unknown

**Whether any elevated exploration helps at all.** Every record this project holds was set at
near-zero epsilon, and the rewrite's original premise — that batches 10-11 running 96.8% of steps
at exactly 0 was itself the defect — has no evidence behind it and one falsified prediction against
it. What survives on evidence is narrower: the *ratchet* was a real defect, because `b11b` sat at
0.001 through a collapse from 64.6 to 8.8 with no way to buy exploration back. A floor of 0.002
plus a stateless schedule fixes that without needing a high ceiling.

So the honest position is that 0.0125 is a bet, not a finding, and the null hypothesis — that
b11's near-zero regime is simply correct — is still live. A **guaranteed-descent envelope**
(`max(pf30 / target, step / S)`) is the fallback if 0.0125 also deadlocks; it was deliberately
left out so the ceiling can be judged on its own.

### Naming

`b<batch><letter>-<change>`, e.g. `b1b-tgt200`. The batch prefix keeps runs from
the same machine-load conditions grouped, which matters because throughput and
therefore contention differ between batches.

### Artifacts, and resuming

Each run continuously writes to `snek2/runs/`:

```
<policy>.png                     graph, whole history across all runs of that policy
<policy>.md                      graph + config table + recent evals
<policy>_evals.json              full eval series; this is what later sessions analyse
<policy>_checkpoint_evals.json   100-episode measurements, written by eval_checkpoints.py
```

Those are the **live** files, rewritten on every eval. The tuning docs must never
link to them directly: if `runs/` is ever cleaned out, every chart in `charts.md`
would silently vanish. Instead `charts/` holds snapshot **copies**, refreshed with:

```
snek2/hyperparamTuning/refresh_charts.sh
```

which re-copies each graph and prints the step it is at, so captions in
`charts.md` can be updated to match. Add new charts there, not in this file or
`runs.md`.

**Never delete anything in `snek2/runs/`** (see `CLAUDE.md`). Re-running the same
policy name continues its graph and draws a dashed line at the resume step, so a
promising run can be extended later just by launching it again with the same name.
That is the intended way to answer "does the slope keep going up?".

Runs are expected to be killed and restarted. `kill -9` is safe: the graph, report
and evals json are all written via write-then-rename, and the agent checkpoints
every eval.

---

