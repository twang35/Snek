# Hall of Fame

The best Snake policies this project has produced, preserved as standalone checkpoints so
they survive whatever happens to `savedPolicies/`.

**Why this folder exists.** Training writes a checkpoint every 1000 steps and keeps the most
recent `max_to_keep` (currently 10000), so a long run eventually **deletes its own best
checkpoint**. That has already cost real evidence — `b5c-schlongIS`'s 17.0% peak became
permanently unmeasurable once the arm passed 1.28M steps. Copies here are outside that
rotation and are not deleted by anything.

## The current record: 96%, and it runs on `master`

Batch 11 is four seeds of batch 10's config on the **30-value observation vector** — the current one,
after the following-tail block (26-28) and food-space (29) landed 2026-08-03. It is the only batch
trained on it, so **these two entries are the only ones in this folder that load on `master` as it
stands.** Both were found by the batch's close-out on 2026-08-04, not measured mid-run.

| checkpoint | measured | config |
|---|---|---|
| `b11b-obs30seed2-ckpt855000` | **96.0%** (96/100, CI 90.2-98.4), top-3 95.3%, ~94% shrunk | `DISCOUNT=0.995`, `SNEK_SEED=2` |
| `b11a-obs30seed1-ckpt671000` | 94.0% (94/100, CI 87.5-97.2), top-3 93.3%, ~90% shrunk | `DISCOUNT=0.995`, `SNEK_SEED=1` |

**Verified by loading the copies in this folder**, not just the originals: `b11b`'s copy re-measured
19/20 perfect (95.0%, avg score 94.8) from `savedPolicies/champion_b11b/` before that scratch
directory was removed. Worth doing every time, given the silent-failure history below.

The **shrunk** figure is the winner's-curse correction — each is the maximum over 148-204 full-length
measurements in its own arm, so some of the headline is luck. Shrinking against a Beta prior fitted
on each arm's *unselected* graph-100% rows gives 94.0% and 90.2%. Use those when comparing to
anything that was not selected the same way; see
[`../hyperparamTuning/completedRuns.md`](../hyperparamTuning/completedRuns.md#a-new-best-measured-checkpoint-b11b-855000-96100).

Both arms were **still healthy when stopped** at 3.19M and 3.56M steps, but both had long since
peaked — `b11a` gave up 42 pp between 678k and its final window. The checkpoints preserved here are
from the peak, which is the entire reason this folder exists.

**The 30-value era starts at `b09c616`** ("Two new observations, a three-stage close-out, and
`EVAL_CONFIRM_COUNT=100`"), which is the commit to check out if a future change moves the vector
again — the same role `450e66e` plays for the 26-value era and `e4514a8` for the 20-value one. Both
new blocks already carry their final polarity in that commit (1 = good at 26-28 and 29), so there is
no sub-era to worry about, and no checkpoint exists from the few hours the reversed version was live
because batch 11 launched after the flip.

**These two are also the first entries whose arm can still be resumed at all**, since every earlier
batch trained on a narrower vector. `SNEK_SEED` is recorded per arm in `runs/<policy>.md` (a
batch-11 first), which makes a *fresh* run of the same config reproducible — it does **not** make a
resume reproducible, because the replay buffer and RNG state are not checkpointed. A resumed arm
diverges from the original immediately.

## The previous record: 95%, on the environment of 2026-08-02 (superseded 2026-08-03)

Batch 10 was a fresh baseline on the environment left by 2026-08-02's seven observation/reward
changes (fatal-move zeroing, the wall/body-hugging observation, normalized group count, the
corrected starve/length split, the terminal-discount fix, safe-to-chase-food, and the audit that
started the day) — the first arms ever to train end-to-end on it, since batch 9 predates all
seven. All four seeds beat every post-audit result on record before being stopped 2026-08-03 to
make room for further changes; two were measured with `eval_checkpoints.py` first:

| checkpoint | measured | config |
|---|---|---|
| `b10d-disc995seed4-ckpt1815000` | **95.0%** (95/100, CI 88.8-97.8), top-3 93.3%, pooled 74.5%/24600 | `DISCOUNT=0.995` |
| `b10b-disc995seed2-ckpt1157000` | 87.0% (87/100, CI 79.0-92.2), top-3 86.3%, pooled 70.4%/12000 | `DISCOUNT=0.995` |

**These no longer load on `master` either, as of 2026-08-03.** Two observations were added that
day — following-tail (26 -> 29) and food-space (29 -> 30) — so batch 10's checkpoints now fail the
same way every entry below does:

```
ValueError: Shapes (30, 50) and (26, 50) are incompatible
```

They held the "runs on master today" status for one day. To run one, check out a commit before
those landed — `450e66e` is the last one with a 26-value vector.

Both measurements were taken mid-run (the arms kept training for another ~2.5-2.9M steps
afterward) and the close-out re-measurement has since found better checkpoints in both arms —
90% @1501000 in `b10b` and 93% @1695000 in `b10d`, the latter within noise of the 95% above.
See [`../hyperparamTuning/completedRuns.md`](../hyperparamTuning/completedRuns.md) for the
batch's full results and whichever numbers are most current.

## The entries below predate 2026-08-02 and do not run on `master`

The observation vector is **30 values** and these were trained on 20, so the first layer's shape
no longer matches and restoring one fails immediately:

```
ValueError: Shapes (30, 50) and (20, 50) are incompatible
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

**Superseded as the project's best by the batch-10 entries above** — this section's own
analysis of *why* is left as written, since it is still the correct read of these four
relative to each other and to the pre-08-02 environment. Read "best checkpoint this project
has" below as "best of these four", not literally.

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
