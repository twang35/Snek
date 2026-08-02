# Hall of Fame

The best Snake policies this project has produced, preserved as standalone checkpoints so
they survive whatever happens to `savedPolicies/`.

**Why this folder exists.** Training writes a checkpoint every 1000 steps and keeps the most
recent `max_to_keep` (currently 10000), so a long run eventually **deletes its own best
checkpoint**. That has already cost real evidence — `b5c-schlongIS`'s 17.0% peak became
permanently unmeasurable once the arm passed 1.28M steps. Copies here are outside that
rotation and are not deleted by anything.

## These checkpoints do not run on `master` (2026-08-02)

The observation vector is **23 values** and these were trained on 20, so the first layer's shape
no longer matches and restoring one fails immediately:

```
ValueError: Received incompatible tensor with shape (20, 50) when attempting to restore
variable with shape (23, 50) and name sequential/dense/kernel:0
```

To run one by hand, check out the last commit whose observation matches them:

```
git checkout e4514a8    # "Fix head_with_tail to advance the tail"
```

**Do not assume a matching width means a working checkpoint.** For part of 2026-08-02 the vector
was coincidentally back at 20 while indices 18 and 19 meant entirely different things, and in that
window these checkpoints restored with no warning at all and played like beginners — the champion
below scored **0, 0, 1** over three episodes against **90.3%** at `e4514a8`. Nothing checks that
values still mean what they meant; only the length is checked. If the count ever returns to 20,
that silent failure comes back.

The `new env` column below was measured at `e4514a8`. The entries stay regardless: they are the
record of what this project achieved, and the weights are still the weights.

## Entries

Every entry is measured on **both** environments, because the 2026-08-01 audit changed two
observation components and the reward. `old env` is what the policy achieved when it was
trained; `new env` is the same weights re-measured on the environment that ran on 2026-08-02,
before the 21-value change. Read `new env` for what a checkpoint does at commit `e4514a8`.

| checkpoint | old env | new env | config |
|---|---|---|---|
| **`b8f-disc9975seed2-ckpt3149000`** | 84.3% pooled /300 | **82.0%** (82/100, CI 73.3-88.3), avg 92.0 | `DISCOUNT=0.9975` |
| `b8f-disc9975seed2-ckpt2816000` | **88.9% pooled /360**, peak 92.0% | 73.0% (73/100, CI 63.6-80.7), avg 86.8 | `DISCOUNT=0.9975` |
| `b8f-disc9975seed2-ckpt2581000` | 87.6% pooled /170 | 63.0% (63/100, CI 53.2-71.8) | `DISCOUNT=0.9975` |
| `b8d-disc995clip-ckpt2538000` | 73.3% pooled /300, peak 80.0% | 61.0% (61/100, CI 51.0-70.2), avg 86.5 | `DISCOUNT=0.995` + `GRADIENT_CLIPPING=10` |

"Perfect rate" is the share of episodes where the snake fills the board, over greedy episodes;
see [`../hyperparamTuning/findings.md`](../hyperparamTuning/findings.md). Old-env figures are
**pooled over every measurement of that step**, which is why `2816000` reads 88.9% rather than
the 92.0% single run it is famous for, and `2538000` reads 73.3% rather than 80.0% — both of
those were the high draw of three (92/88.3/83/92 and 70/80/70). The peak is given alongside.

> **The two columns rank the entries differently, and that is the point.** `3149000` is now the
> best checkpoint this project has, at 82%, despite ranking third on the old environment.
> `2816000`, the famous 92%, reads 73% today. Which checkpoint is "best" depends on the
> environment measuring it, so a record is only ever a record for one of them.
>
> The drop is not evidence the audit was harmful: these policies were trained to read features
> whose meaning then changed, so being off-distribution costs them. Across 72 checkpoints
> measured on both, the loss was ~10 points and 49 of 52 got worse — systematic, and roughly
> equal for both arms. Whether the corrected observations *train* better is what batch 9 tests.

Four entries, kept deliberately.

- **`3149000`** is the best policy that exists on the current environment, 82 of 100. Added
  2026-08-02 after the re-measurement, which is also when it stopped being just b8f's
  third-ranked checkpoint.
- **`2816000`** won 92 of 100 on the old environment and was the project record for a day. Kept
  as the historical high-water mark, and as the clearest single illustration of what the audit
  cost a policy trained before it.
- **`2581000`** is kept because it was independently corroborated on the old environment (88/100
  and 90/100) when reproducibility of the 100-episode measurement was still an open question.
- **`b8d`'s `2538000`** is the best from a **different config**, so the clipping-vs-no-clipping
  comparison stays reproducible if either is revisited.

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
