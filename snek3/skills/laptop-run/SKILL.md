---
name: laptop-run
description: Start a snek3 training, a batch of arms, or a stage-B close-out on the laptop. Use for "train X here", "launch this batch locally", "run stage B on the laptop", "smoke test the trainer". For the desktop box use desktop-batch instead.
---

# Launch on the laptop

Run from `snek3/`.

## Rules that break a run if missed

- **Call the env python directly. Never `conda run`** for anything backgrounded — it buffers stdout,
  so the log stays empty for 90+ seconds while the process is fine, and `kill -9` then discards the
  buffer for good.
- **Never more than 8 trainers at once**, counting anything the user started:
  `ps -Ao pid=,command= | grep '[t]rain.py' | wc -l`. Use `ps | grep`, not `pgrep -f` — a `pgrep`
  pattern matches the shell that runs it.
- **Do not launch a chart window.** The scheduler owns the box's one window: it opens it when it
  launches a wave or a pass and closes it when it exits. `python -m tools.scheduler --reopen-window`
  replaces it if it was closed. No training and no close-out opens one.
- **Verification runs use the policy name `smoke`**, so output lands in `savedPolicies/smoke/` and is
  safe to delete.

## Launch through the queue, not by hand

**Arms are launched by dropping their specs into `logs/laptop-queue/<batch>/` and starting the
scheduler** ("Queueing batches here" below) — for a batch dequeued from the desktop *and* for arms written
for the laptop: write the specs (`queue/examples/` on `ops` has the shape; `tools.sweep_specs` writes a
batch's) and queue them. The scheduler is what publishes the laptop's state to the `laptop-status` branch,
which the desktop folds into `ops-status` as `laptop_running` / `laptop_queued`, so an arm started with
a bare `train.py` is invisible to every status read (user, 2026-09-04). The bare command below is for a
smoke or a one-off check, not for an arm anyone will read about later.

## Train (one-off only)

Hyperparameters are `SNEK_*` env vars, so variants run side by side with no file edits.
`docs/running.md` is the knob list.

```
PYTHONPATH=. SNEK_SEED=1 /opt/miniconda3/envs/snek3/bin/python -u train.py <policy> \
    > logs/<policy>.log 2>&1 &
```

`SNEK_MAX_STEPS` is **absolute**, not "this many more" — `global_step` is restored on resume, and an
arm already at its cap exits after one eval. A batch is four seed-matched arms; one arm answers
nothing, and n=4 cannot resolve an effect below ~10 pp.

**Then confirm each arm got its config. Two greps, and neither is optional:**

```
grep -E 'hyperparameter override:|reward config:' logs/<policy>.log
```

`hyperparameter override:` covers everything read through `tuned()`. It is **silent on the shaping
set** — `SNEK_CHASE_SAFE_*`, `SNEK_FREE_SPACE_*`, `SNEK_FOOD_DISTANCE_REWARD`,
`SNEK_PERFECT_GAME_REWARD`, `SNEK_ZERO_OBS` — because `env/constants.py` reads those at import,
before the trainer's config exists. That is exactly the set a shaping experiment is about.
`reward config:` is the line that covers them.

## A whole batch of desktop specs

To run a batch here that was written for the desktop -- dequeued from `ops` to shorten the box's
queue, as b13 was on 2026-09-03 -- do not launch the arms by hand. `tools.scheduler` takes the
daemon's own `queue/pending/*.json` specs and runs them the way the daemon would: waves of 8, each
arm with the spec's `env` and `max_steps`, each wave followed by its own `tools.closeout` named
`<batch>-stageb`, `-w2`, ... and then -- as the daemon also does since 2026-09-04 -- the wave's
`hof5000` and `hof30k` passes over the same arms (`<batch>-hof5000`, `<batch>-hof30k`, `-w2`, ...),
each only if the pass before it exited 0. It skips an arm already at its cap, waits for an arm
already live here instead of relaunching it, and never puts a ninth trainer on the box.
`--no-hof` stops after stage B; `--no-stage-b` trains only.

```
mkdir -p logs/<batch>specs && for f in $(git ls-tree --name-only origin/ops snek3/desktop/queue/pending/ | grep '/<batch>[a-z]'); do
    git show "origin/ops:$f" > logs/<batch>specs/$(basename $f); done
PYTHONPATH=. nohup /opt/miniconda3/envs/snek3/bin/python -u -m tools.scheduler logs/<batch>specs/ \
    > logs/<batch>-batch.log 2>&1 &
```

Rerunning the same command after a kill or a reboot is the recovery procedure: finished arms are
skipped, and so is any pass whose merged file every arm of the wave already has -- a shard resumes
only from its own shard files and the merge deletes them, so without that skip a rerun would
re-measure a finished wave's stage B from scratch. Then check the config of one arm per wave with the
two greps above, as for any launch.

### Queueing batches here: `--queue`, not a daemon

```
mkdir -p logs/laptop-queue/<batch> && for f in $(git ls-tree --name-only origin/ops snek3/desktop/queue/pending/ | grep '/<batch>[a-z]'); do
    git show "origin/ops:$f" > logs/laptop-queue/<batch>/$(basename $f); done
ps -Ao pid=,command= | grep '[t]ools.scheduler' | grep -v 'zsh -c'    # a scheduler already up? then you are done
PYTHONPATH=. nohup /opt/miniconda3/envs/snek3/bin/python -u -m tools.scheduler --queue logs/laptop-queue/ \
    > logs/laptop-queue.log 2>&1 &                                       # only if none is
```

Each subdirectory of `logs/laptop-queue/` is a batch. The scheduler publishes what it is doing to the
`laptop-status` branch on every launch, exit and pass, every ten minutes while it waits, and once more,
empty, as it exits (`--no-status` turns that off, for a smoke); read it with the desktop's
`git fetch origin ops-status && git show origin/ops-status:status.json` under `laptop_running` /
`laptop_queued`. The scheduler runs the batches in name order, one at a
time with waves and all three passes, **rescans the directory between batches** so a batch dropped in
while another runs is picked up next, and exits when nothing there has work left. So queueing a batch
while the scheduler is up is just the first two lines; while it is down, all four. It is not a daemon:
nothing runs while there is no work. A batch that still reports work after it has run once -- a failed
pass -- is left alone rather than looped on, and the log says so.

`--after <pid>` still works on either form: the running process keeps its batch, the queue starts when
it exits. **A killed scheduler leaves its arms training** (their own session) and the next scheduler
adopts them through `runs/.live/`; a pass it was running finishes and merges on its own.

**An eval spec in a batch directory runs too**, once, after the batch's waves — a hand hof pass, a
`one` re-measure — as the `tools.closeout` command it spells, and with the window on its charts. That
is how a hand pass gets a window now; a `tools.closeout` typed at the shell gets none.

## Stage B

Under the scheduler: every wave gets its stage B, hof5000 and hof30k without anyone asking. By hand, one
process per batch, in arm order, **no window** (queue an eval spec for one):

```
PYTHONPATH=. /opt/miniconda3/envs/snek3/bin/python -u -m tools.closeout <policy...> --shards 12 \
    > logs/<batch>-closeout.log 2>&1 &
```

Single arm, or a re-measure:

```
PYTHONPATH=. python -u evaluate.py <policy>                      # screen:97, 500 eps, 4 shards
PYTHONPATH=. python -u evaluate.py <policy> one --episodes 1000   # one checkpoint, this process
```

Ground truth for a wave's progress is `logs/<pass>-s<i>of<n>.log`. **A killed wave loses nothing** —
each shard rewrites its own file after every measurement and the same command resumes it.

Defaults are the protocol: `--episodes 500`, selector `screen`. Pass `--label` when an A/B would
otherwise overwrite the file it is compared against.

## What a launch starts besides the trainers

The scheduler starts the wave's **shared `tools.eval_worker` processes** first (the count the specs'
`SNEK_EVAL_WORKERS` agree on, else 6; they exit after 300 s idle) and one chart viewer. The trainers
still ask for workers and find the slots held. So an 8-arm wave is 8 trainers, the workers and one
viewer. A bare `train.py` starts its own workers, as before.

## Smoke test

```
PYTHONPATH=. SNEK_MAX_STEPS=5000 SNEK_EVAL_INTERVAL=500 SNEK_GRAPH_EVAL_EPISODES=20 \
    SNEK_MIN_CHECKPOINT_SCORE=0 SNEK_EVAL_QUEUE=0 SNEK_CHART_WINDOW=0 \
    /opt/miniconda3/envs/snek3/bin/python -u train.py smoke > logs/smoke.log 2>&1 &
```

- `SNEK_MIN_CHECKPOINT_SCORE=0` — a smoke scores ~0, so at the default 40 it writes no checkpoint and
  cannot resume.
- `SNEK_EVAL_QUEUE=0` keeps a 5,000-step check from starting six workers. `SNEK_CHART_WINDOW=0` is
  harmless here (a trainer opens no window) and stays for the scheduler's sake.
