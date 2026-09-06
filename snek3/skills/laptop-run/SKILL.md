---
name: laptop-run
description: Start a snek3 training, a batch of arms, or a stage-B close-out on the laptop. Use for "train X here", "launch this batch locally", "run stage B on the laptop", "smoke test the trainer". To queue a batch for either box use queue-batch.
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

**Arms are queued on the shared queue (`queue-batch`: specs on `ops`, pinned `"box": "laptop"` if they
must run here) and the laptop's scheduler claims and runs them** ("The laptop's scheduler" below). The
scheduler is what publishes the laptop's state to the `laptop-status` branch, which the desktop folds
into `ops-status` as `laptop_running` / `laptop_queued`, so an arm started with a bare `train.py` is
invisible to every status read and to the other box's claims. The bare command below is for a smoke or
a one-off check, not for an arm anyone will read about later.

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

## The laptop's scheduler

```
ps -Ao pid=,command= | grep '[t]ools.scheduler' | grep -v 'zsh -c'    # one up already? then you are done
PYTHONPATH=. nohup /opt/miniconda3/envs/snek3/bin/python -u -m tools.scheduler --shared --queue logs/laptop-queue/ \
    > logs/laptop-queue.log 2>&1 &                                       # only if none is
```

**`--shared` is the shared queue** (`tools/claims.py`, `plans/archive/shared-queue.md`): the scheduler
fetches `ops` and `claims`, mirrors the waves this box holds into `logs/laptop-queue/<batch>/` (each spec
carrying its `_wave`), runs them the way the desktop does -- waves of 8, each followed by its own
`tools.closeout` as `<batch>-stageb`, `-w2`, ... and then the wave's `hof5000` and `hof30k` over the same
arms, each only if the pass before it exited 0 -- and **claims the next free wave, or eval spec, only when
nothing it holds is left to run**. It exits when the pool has nothing this box may take. **The laptop takes
part exactly while its scheduler is up**: it is not a daemon, nothing runs while there is no work, and a
batch pushed to `ops` while it is down waits for the desktop or for the next start of this command.
`--no-hof` stops after stage B; `--no-stage-b` trains only.

Rerunning the same command after a kill or a reboot is the recovery procedure: the mirror is rewritten
from the claims, finished arms are skipped, and so is any pass whose merged file every arm of the wave
already has -- a shard resumes only from its own shard files and the merge deletes them, so without that
skip a rerun would re-measure a finished wave's stage B from scratch. **A killed scheduler leaves its arms
training** (their own session) and the next scheduler adopts them through `runs/.live/`; a pass it was
running finishes and merges on its own. Then check the config of one arm per wave with the two greps
above, as for any launch.

The scheduler publishes what it is doing to the `laptop-status` branch on every launch, exit, claim and
pass, every ten minutes while it waits, and once more, empty, as it exits (`--no-status` turns that off,
for a smoke); read it with `git fetch origin ops-status && git show origin/ops-status:status.json` under
`laptop_running` / `laptop_queued`, and the shared queue under `at_a_glance.pool`. `--after <pid>` still
works: the running process keeps its batch, the queue starts when it exits. A batch that still reports
work after it has run once -- a failed pass -- is left alone rather than looped on, and the log says so.

**An arm training here that no claim covers** (a released wave, a bare launch of a spec on `ops`) is named
under `attention` and left alone: a sync never kills a trainer, and it is never published from here.

**`tools.scheduler <spec files or dirs>` and `--queue` without `--shared` still work** for a directory of
specs nobody else should see -- a smoke batch -- and run it as one box's own queue, nothing claimed.

**An eval spec runs too**, once, after its batch's waves — a hand hof pass, a `one` re-measure — as the
`tools.closeout` command it spells, and with the window on its charts. That is how a hand pass gets a
window; a `tools.closeout` typed at the shell gets none. Queue it on `ops` (`queue-batch`); this box
claims it only if it holds every one of its policies' checkpoints.

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
