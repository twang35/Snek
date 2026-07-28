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

**Current best: 51% measured over 100 episodes**, from `b4c-schlongper` checkpoint 869000
(config: alpha 0.8, `td_loss` priorities, no IS weights). That is the number to beat, and
it matches the ~50%-at-1M figure the investigation started from — the committed config
reaches ~1%, and the gap was three PER changes made during the cpprb port. See
[`findings.md`](findings.md).

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
EVAL_EPISODES=100 EVAL_WORKERS=7 EVAL_OUT_SUFFIX=_869000 \
  PYTHONPATH=. python -u eval_checkpoints.py b4c-schlongper 869000
```

Results land in `runs/<policy>_checkpoint_evals<suffix>.json` with a Wilson 95%
confidence interval. Several copies can run at once on different checkpoints — give each
its own `EVAL_OUT_SUFFIX` or they overwrite each other, then merge. Four 100-episode
evals in parallel take a few minutes.

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

**Never quote a graph peak as a policy's perfect-game rate.** A graph point is 10
episodes, and picking the highest one selects for luck: of four checkpoints chosen by
best smoothed graph rate, three measured *worse* than the graph implied, by up to 24
points (the winner's curse). Single evals understate just as badly in the other
direction — two checkpoints reading 10% on the graph measured 25% and 29%. Use the graph
for trajectory and `eval_checkpoints.py` for numbers.

**Every batch ends with checkpoint evals.** When a batch finishes, pick each arm's best
few checkpoints and measure them at 100 episodes, then compare *those* numbers across
arms. It costs ~3 minutes per arm and it is the only apples-to-apples comparison
available — comparing graph peaks across arms compounds the winner's curse, once per arm.
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
ps -eo pid,etime,command | grep "[s]nek2.py" | grep -v spawn_main
```

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
- **Every arm gets a visible window.** `display_eval` defaults to `True`, so launch with
  no `SDL_VIDEODRIVER` override at all. The user wants to see each arm as it runs; do
  not add `SDL_VIDEODRIVER=dummy` to a tuning arm unless asked. (Earlier guidance here
  suggested one visible arm per batch with the rest headless — superseded.)
- The policy name is the graph window title, so name it so it's identifiable at a
  glance while running.
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

