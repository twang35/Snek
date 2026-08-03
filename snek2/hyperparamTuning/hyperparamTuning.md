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
EVAL_OUT_SUFFIX=_top20 \
  PYTHONPATH=. python -u eval_checkpoints.py b4c-schlongper top20
```

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

#### Why the thresholds moved from >=80%/cap 10 to >=90%/cap 20

The first version measured **everything** at >=80% with no upper bound, which does not scale
on a good arm. Once `b8f-disc9975seed2` reached 3M steps it presented **109** such checkpoints —
about seven hours of evaluation — and it was still training:

| arm | checkpoints picked now | under the old rule |
|---|---|---|
| `b8f-disc9975seed2` | **32** | 109 |
| `b8d-disc995clip` | **20** | 33 |
| `b7f-disc995seed3` | 20 | 10 |

Measurement justified the narrowing: across 88 checkpoints, 90% and 80% graph points had
**indistinguishable** mean true rates (57.9% vs 58.6%), so an unbounded 80% tier was buying
volume rather than information. The 100% points were the only genuinely distinct group.

Note the change is not uniformly cheaper. A **weak** arm now gets *more* attention — `b7f` goes
from 10 to 20 — because the cap doubled and the fill band widened to include 80%. The saving is
concentrated where the cost was: strong arms with hundreds of high checkpoints.

**Most arms have nothing above the floor,** so the selector refuses them outright with a message
naming their best graph point. That is the intended outcome: 22 of the 30 arms run so far never
produced a single eval above 50%. It also means `b6a-alpha04` cannot be re-measured at all and
`b6b-alpha06` yields only 2 checkpoints.

**These rule changes break pooled-rate comparability.** Pooled now averages over a checkpoint
count that varies by arm (32 vs 20 vs 1) and over a population truncated at 60%, so a strong
arm's pooled figure is not computed the same way as a weak one's. Compare **best checkpoint**
across arms, and treat pooled as a within-arm consistency read.

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

#### Results are saved incrementally

The output JSON is rewritten after **every** checkpoint, via `.partial` + `os.replace` so a
reader never sees a half-written file. An interrupted run therefore keeps everything measured up
to that point, and the payload carries `complete: false` until the final checkpoint lands —
check that field before treating a file as an arm's full measurement.

This matters because these runs are long: 20 checkpoints is over an hour, and a 63-checkpoint
run took four. The earlier version wrote once at the end, so any interruption discarded all of
it — the numbers would survive in the log, but nothing machine-readable.

**Measure the whole >=80% tier; do not trust its order.** 26 high-eval checkpoints measured on
2026-07-30 show the graph value carries no ranking signal once it is high — correlation with the
true rate is **-0.09** across both arms, and it flips sign between them (+0.66 / -0.57). The
three 90% points in that sample measured 39%, 21% and 44%, the worst three of `b8f`'s sixteen.
Range restriction explains part of it (everything here is 70-90%), so this coexists with the
+0.64 below rather than replacing it. The practical consequence is the important part: **there is
no way to tell in advance which >=80% checkpoint is the 63% and which is the 21%**, so measure
all of them rather than truncating the list.

**Outlier evals are not luck — those checkpoints really are better.** This was measured,
not assumed, and it reversed an earlier version of this protocol that ranked by smoothed
rate on the theory that a 70-80% single eval had to be a fluke:

| evidence | result |
|---|---|
| correlation with true 100-episode rate | raw single eval **+0.64**, smoothed **-0.40** |
| pooled measurement by selection rule | raw **41.3%**, smoothed 27.1% (non-overlapping) |
| outlier vs the checkpoints 1000 steps either side | outlier won **3 of 3**, by 9.0 / 11.5 / 27.5 points |
| P(10-episode eval shows 7+ perfect \| true rate 27%) | **0.006** |

So a spike is evidence about *that checkpoint*, and smoothing averages it into a statement
about the *region* — which is a different and less useful thing. Ranking on the surrounding
rate systematically picked worse checkpoints.

**Adjacent steps are allowed through on purpose.** 1000 training steps is enough to move the
perfect rate by tens of points — one measured triple reads 8% / 35% / 7% — so neighbouring
checkpoints are separate policies rather than repeat samples of one.

**Budget ~4 minutes per checkpoint**, so a 20-checkpoint run is ~80 minutes. That rate held for
both a single arm at 10 workers and two arms in parallel at 20 workers each — **throughput is
core-bound**, so raising `EVAL_WORKERS` past ~10 does not make a run faster, it only lets a
second arm run alongside. Two arms in parallel is the real speedup; more workers is not.

An arm whose mandatory tier exceeds the cap costs proportionally more: `b8f` at 32 is ~2 hours.

Results land in `runs/<policy>_checkpoint_evals<suffix>.json` with a Wilson 95%
confidence interval. Several copies can run at once on different arms — give each its own
`EVAL_OUT_SUFFIX` or they overwrite each other, then merge.

**Checkpoint retention bounds all of this.** A checkpoint is written every 1000 steps, so
`max_to_keep` is a rolling window measured in millions of steps, and an arm run past that
window **deletes the checkpoint behind its best number**.

`max_to_keep` was 1000 — a 1M-step window — which cost real evidence: three of batch 5/6's
four arms outran it, and `b5c-schlongIS`'s 17.0% peak at 211k became unmeasurable once the
arm passed 1.28M steps, leaving a best surviving region worth only 7.0%. **It is now
10000**, a 10M-step window, at ~188 KB per checkpoint (~1.8 GB per policy at full depth).
The legacy `train*/` dirs run 9.7 MB per checkpoint because they predate moving the replay
buffer out of the checkpointer, so they would be ~97 GB at this depth — do not resume those
under the new setting without checking disk.

Two habits still apply. Close an arm out at its horizon: past peak, the marginal training
step is worth less than the checkpoint it evicts. And `top20` filters to surviving
checkpoints, so it degrades gracefully instead of failing on a deleted step.

**Each eval process opens one visible window** (worker 0 renders, the rest are
headless), so four parallel evals give four games to watch. The rendering worker is
slower, which only means its round takes longer — episodes are i.i.d. across workers, so
which worker produced one carries no information.

**Episodes are collected in whole rounds**, one per worker, rather than by stopping at
the Nth finished episode. Stopping mid-flight discards the episodes still running, and
**perfect games are the longest episodes there are**, so truncation drops them
preferentially and biases the measured rate *downward*. `EVAL_EPISODES` is rounded up to
a whole number of rounds for this reason. The first published measurement (51% at
`b4c-schlongper` 869000) used the truncating version, so it is if anything an
underestimate.

**Never quote a graph peak as a policy's perfect-game rate**, but do use it to *choose*
what to measure. A graph point is only 10 episodes, so the number itself is unusable — a
70% point measured 40%, and a 40-50% point measured 12-17%. What the spike reliably tells
you is *which checkpoint is worth 100 episodes*, and there it beats every smoothed
alternative (+0.64 vs -0.40 correlation). Use the graph to select, `eval_checkpoints.py`
to quote.

**Every arm ends with checkpoint evals.** Run `eval_checkpoints.py <arm> top20` and compare
*those* numbers across arms. Comparing graph peaks across arms compounds the error once per
arm, and it demonstrably misranks: `b5c-schlongIS` is 2nd of its batch by graph window and
**last by measurement** (17.0% vs 2.1%).

**Compare pooled rates only when the selection rule matches.** `b4c-schlongper` pools 31.4%
under outlier+smoothed selection and 26.2% under cluster selection — not a contradiction,
because 6 of the 10 cluster picks are deliberately the weaker neighbours. Its level is
~31%. A pooled number is only meaningful alongside the rule that produced it.

**Repeat a measurement before trusting it.** Checkpoint 869000, frozen weights and a greedy
policy, measured 51%, 42% and 32% on three separate 100-episode runs — a 19-point spread,
about 2.8 sigma, wider than binomial noise comfortably explains. Its pooled figure over 300
episodes is 41.7%. Treat a lone 100-episode result as provisional even though its Wilson
interval looks tight.
Choose the candidates by trailing-window rate rather than by single-eval peak, and expect
the measured value to come in below the window that selected it.

Three of the run-comparison metrics above need care:

- **Always use a trailing window for perfect %, never a single eval.** One eval is
  10 episodes, so the metric moves in 10-point jumps. Use the last 30 for a coarse
  read, and compare it against the previous 30 rather than an absolute threshold.
- **Score stops proxying the objective late in a run.** `b1a-base` recovered from a
  collapse to its old score while its perfect rate kept falling to a fifth of its
  peak. Score is right early, when perfect % sits at 0 for tens of thousands of
  steps and cannot separate two configs at all. Past ~250k steps, or after any
  collapse, read perfect % directly and treat score as context only.
- **A large drawdown is not disqualifying.** A config that swings wildly but reaches
  a high perfect-game rate beats a placid one that plateaus low — `b1a-base` versus
  `b2a-base2` is exactly that contrast. `b4c-schlongper` is the extreme case: it
  collapsed to score ~19 and went on to be the best arm in the investigation.

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
| `SNEK_INITIAL_EPSILON` | 0.4 | where the decay ladder starts |
| `SNEK_MIN_EPSILON` | 0.0 | floor for the ladder. 0.0 = the historical behaviour, a fully greedy collect policy at the end |
| `SNEK_FC_LAYERS` | 50,100,50 | comma separated |
| `SNEK_REPLAY_BUFFER_MAX_LENGTH` | 100000 | |
| `SNEK_PRIORITY_EXPONENT` | 0.6 | alpha; 0.0 disables prioritization |
| `SNEK_PRIORITY_SIGNAL` | `td_error` | or `td_loss` (element-wise Huber), which is what `theSchlong` used |
| `SNEK_IS_WEIGHTS` | 1 | 0 disables importance sampling entirely, as `theSchlong` did |
| `SNEK_IS_BETA` | 0.4 | ignored when `IS_WEIGHTS=0` |
| `SNEK_BETA_ANNEAL_STEPS` | 1000000 | |
| `SNEK_INITIAL_POPULATE_STEPS` | 1000 | |

Rewards (`snake_constants.py`) are deliberately **held fixed** so `avg_score` stays
comparable across every run. Changing them invalidates comparison with everything
already recorded here.

One behaviour worth understanding before reading any run: `maybe_update_epsilon()`
in `training.py` ratchets epsilon down a fixed ladder (0.2, 0.1, 0.05, 0.01, 0.001,
then `SNEK_MIN_EPSILON`) as `avg_reward` rises, one rung per eval, and never back
up. The rungs are driven by reward thresholds, so **only strong runs reach the
bottom** — the last rung needs `avg_reward > 100`. That makes "which rung did it
reach" a proxy for how good a run got, and a confound when comparing arms.

Historically the last rung was hard-coded to 0.0, i.e. a fully greedy collect
policy; `SNEK_MIN_EPSILON` now floors it and defaults to 0.0, preserving that.
Each eval row records the current epsilon in `runs/<policy>_evals.json`, which is
how to tell after the fact which rung a run was on — worth checking, because
reaching the last rung is the leading suspect for the collapse (see
[`runs.md`](runs.md)).

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

