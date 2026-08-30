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
- **Do not launch a chart window.** Every training opens the box's one window itself. Eight arms
  spawn eight viewers and seven exit in ~0.3 s; that is the design, not a fault.
- **Verification runs use the policy name `smoke`**, so output lands in `savedPolicies/smoke/` and is
  safe to delete.

## Train

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

## Stage B

One process per batch, in arm order, with its own chart window:

```
PYTHONPATH=. /opt/miniconda3/envs/snek3/bin/python -u -m tools.closeout <policy...> --shards 8 \
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

## What a trainer starts besides itself

Each arm spawns a chart viewer and asks for the box's **6 shared `tools.eval_worker` processes**
(`SNEK_EVAL_QUEUE=1` is the default; the workers are per box, not per arm, and exit after 300 s
idle). So a "one process" launch is 8 processes on an idle box. Count that against a close-out or a
batch already running here.

## Smoke test

```
PYTHONPATH=. SNEK_MAX_STEPS=5000 SNEK_EVAL_INTERVAL=500 SNEK_GRAPH_EVAL_EPISODES=20 \
    SNEK_MIN_CHECKPOINT_SCORE=0 SNEK_EVAL_QUEUE=0 SNEK_CHART_WINDOW=0 \
    /opt/miniconda3/envs/snek3/bin/python -u train.py smoke > logs/smoke.log 2>&1 &
```

- `SNEK_MIN_CHECKPOINT_SCORE=0` — a smoke scores ~0, so at the default 40 it writes no checkpoint and
  cannot resume.
- `SNEK_EVAL_QUEUE=0` and `SNEK_CHART_WINDOW=0` keep a 5,000-step check from starting six workers and
  claiming the box's window slot from whatever is really training.
