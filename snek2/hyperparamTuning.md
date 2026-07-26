# Hyperparameter tuning protocol

Long-running, resumable investigation into what makes snek2 learn **consistently**
with **minimal catastrophic forgetting**, and eventually reach a higher
perfect-game percentage.

This file is the handoff document. A fresh Claude Code session should be able to
read it and continue without any other context. Keep it current: it is more
valuable than any single run.

---

## The goal, and why it's hard

The end target is perfect-game percentage. With the config as committed that
needs roughly **1M iterations / 4-5 hours** of training and then plateaus around
**50%**. Two things make tuning awkward:

- **High noise.** Two runs of the identical config reached final avg_score 62.5
  and 18.0 at 30k steps. Any single-run comparison is meaningless. Promising
  configs must be repeated **2-3 times**.
- **Non-linear trajectories.** Sometimes nothing much happens until ~40k
  iterations and then the slope turns steep. A config that looks dead at 20k may
  not be. Do not judge a config on a short run alone.

Because of that, avoid the final eval as a metric. Prefer, in rough order of
usefulness:

1. **last-5-eval average** — much lower variance than a single eval.
2. **mean over the whole curve** — rewards learning early and holding on.
3. **max drawdown** — biggest drop from a running peak. This is the direct
   measurement of catastrophic forgetting and the thing this investigation is
   mainly chasing.
4. **steps to first reach score N** — for comparing learning speed.

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
- Do **not** set `SDL_VIDEODRIVER=dummy` for real tuning runs. The human wants
  one visible eval env plus the headless parallel ones so training can be
  watched. `display_eval` is already `True` by default; just leave it.
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
| `SNEK_INITIAL_EPSILON` | 0.4 | |
| `SNEK_FC_LAYERS` | 50,100,50 | comma separated |
| `SNEK_REPLAY_BUFFER_MAX_LENGTH` | 100000 | |
| `SNEK_PRIORITY_EXPONENT` | 0.6 | 0.0 disables prioritization |
| `SNEK_IS_BETA` | 0.4 | |
| `SNEK_BETA_ANNEAL_STEPS` | 1000000 | |
| `SNEK_INITIAL_POPULATE_STEPS` | 1000 | |

Rewards (`snake_constants.py`) are deliberately **held fixed** so `avg_score` stays
comparable across every run. Changing them invalidates comparison with everything
already recorded here.

### Naming

`b<batch><letter>-<change>`, e.g. `b1b-tgt200`. The batch prefix keeps runs from
the same machine-load conditions grouped, which matters because throughput and
therefore contention differ between batches.

### Artifacts, and resuming

Each run continuously writes to `snek2/runs/`:

```
<policy>.png          graph, whole history across all runs of that policy
<policy>.md           graph + config table + recent evals
<policy>_evals.json   full eval series; this is what later sessions analyse
```

**Never delete anything in `snek2/runs/`** (see `CLAUDE.md`). Re-running the same
policy name continues its graph and draws a dashed line at the resume step, so a
promising run can be extended later just by launching it again with the same name.
That is the intended way to answer "does the slope keep going up?".

Runs are expected to be killed and restarted. `kill -9` is safe: the graph, report
and evals json are all written via write-then-rename, and the agent checkpoints
every eval.

---

## What is currently running

Update this section whenever runs start or stop — a future session reads it to
know what is in flight and might have been terminated.

| policy | config change | started | status |
|---|---|---|---|
| `train` | human-started, committed defaults | before this work | **running, do not touch** |
| `b1a-base` | control: committed defaults | 2026-07-26 14:15 | running, target ~120k steps |
| `b1b-tgt200` | `TARGET_UPDATE_PERIOD=200` | 2026-07-26 14:15 | running, target ~120k steps |
| `b1c-nstep3` | `N_STEP_UPDATE=3` | 2026-07-26 14:15 | running, target ~120k steps |

That is 4 trainers, i.e. the budget is full — do not launch more until one stops.
Logs for this batch are in `/Users/tony_wang/.claude/jobs/f3cb1855/tmp/b1{a,b,c}.log`,
which is job-scoped and will not survive; the durable record is
`runs/<policy>_evals.json`, so analyse from there.

If these were killed before ~120k steps, they can be resumed simply by relaunching
with the same policy name and the same `SNEK_*` overrides (the overrides are *not*
persisted in the checkpoint — relaunching without them silently changes the config
mid-run, which would invalidate the arm).

---

## Prior findings carried in from earlier sessions

These are already established; don't re-litigate them without new evidence.

- **Prioritized replay at alpha=0.6 measured *worse* than uniform** over 3 seeds
  at 30k steps: last-5 avg 46.7 (sd 10.6) vs **60.1 (sd 4.0)** for `alpha=0`.
  Uniform was also far more consistent. At alpha=0.8 with Huber-loss priorities it
  was worse still. Plausible reason: the reward is already dense and shaped
  (`FOOD_DISTANCE_REWARD` on every step), and PER's advantage is largest with
  sparse rewards, so over-sampling high-error transitions mostly adds variance.
  **Unresolved:** that was only 30k steps. PER may pay off closer to 1M. Worth one
  long-horizon retest.
- `alpha=0.6` is nevertheless the committed default, chosen deliberately. Treat it
  as the baseline to beat, not as known-good.
- **Importance-sampling weights must stay mean-normalized.** cpprb normalizes by
  the largest weight in the whole buffer, so raw batch weights average 0.087 at
  beta=0.4 and 0.0027 at beta=1.0 — a silent 11x-370x cut to the learning rate
  that worsens as beta anneals. `normalize_is_weights()` fixes this; don't remove
  it.
- **Priorities come from `|td_error|`, not `td_loss`.** Huber is quadratic below
  |e|=1 so it shrinks small errors, widening its spread; feeding it in gave an
  effective exponent near |e|^1.6 instead of |e|^0.6 and measurably hurt learning.
- **`legacy.Adam` is not faster here** despite TF's M1/M2 warning: measured 0.809
  ms/step vs 0.721 ms for the modern optimizer. Ignore that warning.
- **Throughput is ~230-240 steps/s** for a single run on an idle machine. Expect
  substantially less with 4 runs sharing 14 cores; that affects wall-clock only,
  not learning per step.

---

## Completed runs

Nothing yet under this protocol. Batch 1 is the first. Earlier ad-hoc results are
summarized under prior findings above; their raw logs were not kept.

| policy | key change | steps | last-5 avg | max drawdown | verdict |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

---

## Planned queue, and what each is expected to show

Ordered by expected value. Revise freely as results land — this is a plan, not a
commitment.

### Batch 1 (first, 3 slots, target ~120k steps)

1. **`b1a-base`** — committed defaults, as a control under the same machine load
   as its batch mates. Not expected to be good; expected to be the *reference*.
   Needed because throughput and contention vary between batches, so cross-batch
   comparison is weaker than within-batch.
2. **`b1b-tgt200`** — `TARGET_UPDATE_PERIOD=200`. **Highest-prior change.** The
   default of 8 gradient steps between target-network syncs is extremely frequent;
   standard DQN uses hundreds to thousands. A target that chases the online
   network is a classic cause of oscillation and forgetting, which is exactly the
   reported symptom. Expect: visibly smoother curve and smaller drawdowns, even if
   peak score is similar.
3. **`b1c-nstep3`** — `N_STEP_UPDATE=3`. The food reward is immediate but the
   perfect-game bonus is terminal and extremely sparse, so credit has to crawl
   back one step per gradient update. n-step returns propagate it ~3x faster.
   Expect: earlier onset of learning (relevant to the "nothing until 40k" effect)
   and better late-game behaviour.

### Batch 2 candidates (pick based on batch 1)

4. **Winner of batch 1, repeated 2-3x.** Mandatory before believing anything,
   given the 62.5-vs-18.0 noise.
5. **`tgt200 + nstep3` combined**, if both help individually.
6. **`LEARNING_RATE=1e-4`** *only after* the target-update fix. Currently 1e-5 is
   very conservative; the in-code comment already suggests 1e-4. With a stable
   target it may train several times faster. On its own with `TARGET_UPDATE_PERIOD=8`
   it would probably make instability worse, so the order matters.
7. **`DISCOUNT=0.995` or `0.999`.** At 0.99 the effective horizon is ~100 steps,
   but a perfect game is several hundred steps long, so the terminal bonus is
   discounted into near-irrelevance. Raising it should make the perfect-game reward
   actually reachable by the value function — plausibly the single most relevant
   change for the *stated* end goal, though also a known source of instability, so
   it pairs naturally with the target-update fix.
8. **Soft target updates**, `TARGET_UPDATE_TAU=0.005` with
   `TARGET_UPDATE_PERIOD=1`. An alternative to (2) that some find smoother still.
9. **`PRIORITY_EXPONENT=0.0` at long horizon.** Confirms or overturns the 30k-step
   finding at the scale that actually matters.
10. **`GRADIENT_CLIPPING=10`.** Cheap insurance against the loss spikes that tend
    to accompany forgetting events.
11. **`FC_LAYERS=128,128`.** Capacity. Lower prior: 20 inputs and 3 actions is a
    small problem, so capacity is unlikely to be the binding constraint before the
    stability issues above are fixed.
12. **Longer epsilon exploration.** The decay ladder in `maybe_update_epsilon()` is
    driven by reward thresholds and steps down once per eval, reaching 0.1 fairly
    early. If forgetting correlates with epsilon getting small, a floor is worth
    testing. Note this schedule is coupled to `eval_interval`, which is a latent
    confound if that interval is ever changed.

### Explicitly not planned

- Reward changes — they'd break comparability of `avg_score` with every recorded
  run.
- Reverting to `PyUniformReplayBuffer` — cpprb is ~2.4x faster with no measured
  learning cost, so cheaper experiments come from keeping it.
