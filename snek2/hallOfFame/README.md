# Hall of Fame

The best Snake policies this project has produced, preserved as standalone checkpoints so
they survive whatever happens to `savedPolicies/`.

**Why this folder exists.** Training writes a checkpoint every 1000 steps and keeps the most
recent `max_to_keep` (currently 10000), so a long run eventually **deletes its own best
checkpoint**. That has already cost real evidence — `b5c-schlongIS`'s 17.0% peak became
permanently unmeasurable once the arm passed 1.28M steps. Copies here are outside that
rotation and are not deleted by anything.

## Entries

| checkpoint | measured perfect rate | 95% CI | avg score | config |
|---|---|---|---|---|
| **`b8f-disc9975seed2-ckpt2816000`** | **92.0%** (92/100) | 84.9-95.9 | 94.7 | `DISCOUNT=0.9975` |
| `b8f-disc9975seed2-ckpt2581000` | 88.0% (88/100) | 80.2-93.0 | 94.2 | `DISCOUNT=0.9975` |
| `b8d-disc995clip-ckpt2538000` | 80.0% (80/100) | 71.1-86.7 | 90.6 | `DISCOUNT=0.995` + `GRADIENT_CLIPPING=10` |

"Perfect rate" is the share of episodes where the snake fills the board, over 100 greedy episodes
each; see [`../hyperparamTuning/findings.md`](../hyperparamTuning/findings.md).

> **These rates were measured on the pre-2026-08-01 environment and will not reproduce.** An env
> audit that day fixed six bugs, two of which changed the observation (the tail is now freed when
> the connectivity features are computed, and `reset()` builds the same bordered grid as `step()`)
> and one of which changed the reward. These policies were trained to read the old features, so
> they are now off-distribution. `ckpt2816000` re-measured **21/30** afterwards against 27/30
> before. The entries are kept because they are the record *as achieved*, not because they still
> score that — see the audit section in `findings.md`. Anything measured from here on belongs to
> the new baseline, so do not put a new row in the table above next to an old one without
> labelling which environment produced it.

**`ckpt2816000` won 92 of 100 games** and is the project record, found in the close-out measurement
of `b8f` on 2026-08-01. It re-measured at 25/30 during the restore verification for this folder,
consistent with 92/100.

Three entries, kept deliberately. `2816000` and `2581000` are the top two policies ever measured
and come from the same arm — the second is kept because it was measured twice independently
(88/100 and 90/10) and is the more corroborated of the pair. `b8d`'s entry is the best from a
**different config**, so the clipping-vs-no-clipping comparison stays reproducible if either is
revisited.

## Running a checkpoint manually

These are `tf.train.Checkpoint` files, not a SavedModel, so they need the same network and
agent that produced them. The scripts in `snek2/` already build it.

The eval script locates checkpoints under `savedPolicies/<policy_name>/`, so copy an entry
into a policy directory of its own and evaluate it by explicit step:

```
cd /Users/tony_wang/Projects/Snek/snek2
conda activate snek

mkdir -p savedPolicies/champion
cp hallOfFame/b8f-disc9975seed2-ckpt2816000/* savedPolicies/champion/

EVAL_RENDER=1 PYTHONPATH=. python -u eval_checkpoints.py champion 2816000
```

**`EVAL_RENDER=1` is what opens the window** — worker 0 renders a game while the other nine
run headless. Watching one game is the point of running this by hand, so leave it on here.
Without it every worker is headless and you get numbers and no window.

Rendering is off by default because it is the slowest thing in an eval: 163us per game step
headless against 6050us in a real window, and since all workers step together the rendering
one paces the whole round. A 30-episode eval takes 14s headless and 70s with the window.
That is the right trade for an unattended close-out and the wrong one for watching a game.

Useful environment variables:

| variable | default | effect |
|---|---|---|
| `EVAL_EPISODES` | 100 | episodes to measure, rounded up to whole rounds |
| `EVAL_WORKERS` | 10 | parallel envs; `1` with `EVAL_RENDER=1` gives a single visible game |
| `EVAL_RENDER` | 0 | `1` shows a game in a window, at ~5x the wall clock |
| `EVAL_PERFECT_WAIT_MS` | 400 | pause on a win — raise to ~2000 to actually see it |
| `EVAL_OUT_SUFFIX` | none | appended to the output filename |

To just watch one game end to end, with a visible pause on the win:

```
EVAL_RENDER=1 EVAL_EPISODES=1 EVAL_WORKERS=1 EVAL_PERFECT_WAIT_MS=2000 \
  PYTHONPATH=. python -u eval_checkpoints.py champion 2816000
```

Results are written to `runs/champion_checkpoint_evals.json`.

### Things that look broken and are not

- **The window stops mid-game and closes.** A round runs until every worker finishes one
  episode; workers that finish early are auto-reset into extra episodes that are *not*
  counted, so the visible worker is usually part-way through a throwaway game when the round
  ends. Only each worker's first episode of a round is scored.
- **`OSError: Bad file descriptor` on exit.** Multiprocessing shutdown noise from
  `ParallelPyEnvironment`, printed after the results are written. Harmless.
- **A `Gym has been unmaintained since 2022` notice at startup.** Inert; the upgrade is not
  available for this dependency set.

### What is *not* here

**No replay buffer**, so these cannot be used to resume training — only to evaluate or watch.
Resuming needs `savedPolicies/<arm>/replay_buffer/` from the original run, and note that cpprb
does not persist priorities across save/restore anyway, so a resume starts from uniform
priorities regardless.

Each entry is two files, ~190 KB total: `ckpt-<step>.index` and
`ckpt-<step>.data-00000-of-00001`. Both are required, and the `checkpoint` file from
`savedPolicies/` is not — that one only records which checkpoint is "latest", and restoring by
explicit step does not consult it.

## Adding an entry

When a run produces a checkpoint worth keeping:

```
cd /Users/tony_wang/Projects/Snek/snek2
mkdir -p hallOfFame/<arm>-ckpt<step>
cp savedPolicies/<arm>/ckpt-<step>.index \
   savedPolicies/<arm>/ckpt-<step>.data-00000-of-00001 \
   hallOfFame/<arm>-ckpt<step>/
```

Then add a row to the table above with its **measured** rate over at least 100 episodes — not
a graph point. A graph point is 10 episodes and reads in 10-point jumps; 90% graph points have
measured anywhere from 22% to 82%.

Note that training now **skips writing checkpoints below `SNEK_MIN_CHECKPOINT_SCORE`** (default
40), because `max_to_keep` is a rolling window and a dead arm used to evict good checkpoints
behind it. That reduces the risk of losing a record before it can be copied here, but does not
remove it — copy anything worth keeping as soon as it is measured.
