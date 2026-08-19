# Launchers, chaining scripts and the chart refresher

Eleven of these moved here from `hyperparamTuning/` on 2026-08-18; the directory held eleven `.sh`
files beside the tuning docs and they crowded out the files a session actually has to read.
`seed_from_checkpoint.sh` was written the same day and started here.

| script | args | what it does | kind |
|---|---|---|---|
| `refresh_charts.sh` | — | re-copies every `runs/*.png` into `../charts/` and prints the step each arm reached, so `charts.md` captions can be written to match. **Copies images only — it never edits `charts.md`** | reusable tool |
| `seed_from_checkpoint.sh` | `<src-policy> <step> <dst-policy> [savedPolicies-root]` | builds a **fresh** policy dir that resumes from **one chosen checkpoint** of an existing arm — `arch.json`, that one `ckpt-*` pair, a copy of the source's `replay_buffer/`, and a `checkpoint` state file naming only that step. Refuses to clobber an existing dir, and refuses if the step or `arch.json` is missing | reusable tool |
| `chain_after_evals.sh` | `<launcher-script> [poll-seconds]` | polls until every `eval_checkpoints.py` has drained, then `exec`s a launcher — queues a *wave* behind a close-out with nobody on the terminal | reusable tool |
| `chain_closeout_after_training.sh` | `<batch-prefix> [poll-seconds]` | the mirror: polls until that batch's trainers drain (they self-terminate at `SNEK_MAX_STEPS`), then runs the close-out as one `eval_checkpoints.py` per arm at `EVAL_WORKERS=4` | reusable tool |
| `launch_c51_wave.sh` | `<prefix> <lr>:<seed> [...]` | the generic C51 launcher — one arm per `<lr>:<seed>` on b25's config plus the c51 knobs, 600k cap, `SNEK_CHART_VIEWER=0` plus one hand-started viewer for the wave | reusable tool |
| `launch_b43_lowlr.sh` | — | batch 43: continues the four best b29/b40 checkpoints to 3M with `SNEK_LEARNING_RATE=1e-6` the only change from b29's config. Expects the policy dirs pre-seeded by `seed_from_checkpoint.sh`; the desktop's control is `b42` on the `ops` queue | record — do not edit |
| `launch_c51_batch.sh` | `[batch-prefix]` (default `b31`) | the pilot→batch handoff, run unattended: waits out the pilot, waits for a free trainer slot, calls `pick_c51_lr.py` to choose the rate, launches 4 seeds at 2M, then regenerates and **commits** the pilot's doc regions | record of b31's launch |
| `launch_c51_pilot.sh` | — | batch `c51pilot` wave A: `{lr 1e-5, 5e-5} × {seed 1, 2}` at a 600k cap, after waiting out b30's close-out | record — do not edit |
| `launch_adam_eps.sh` | — | batch 32: Adam `epsilon` at `1.5e-4` and `3.125e-4`, two seeds each, 1M, on `fc 200,100,100` | record — do not edit |
| `launch_win10.sh` | — | batch 33: `PERFECT_GAME_REWARD=10` with `V_MAX=40`, seeds 1-4, 3M | record — do not edit |
| `launch_c51_fc320.sh` | — | batch 36: b32's config verbatim at `eps 1.5e-4` with `FC_LAYERS=320`, seeds 1-4, 3M | record — do not edit |
| `launch_b38_eps3125.sh` | — | batch 38: b36 with `ADAM_EPSILON=3.125e-4` the only change, seeds 1-4, 3M | record — do not edit |
| `launch_b39_zeroinit.sh` | — | batch 39: b36 with `SNEK_C51_ZERO_INIT=1` the only change, seeds 1-4, 3M | record — do not edit |

## The two kinds, and why the distinction matters

**A reusable tool takes its subject as an argument.** `refresh_charts.sh`, `seed_from_checkpoint.sh`,
both chaining scripts and `launch_c51_wave.sh` are the whole set — nothing in them names a batch, so a
new batch comes through them unchanged. These are the ones to reach for.

**Everything else is the record of one launch, and its header comment is the record.** A per-batch
launcher carries the config *and* the pre-registered hypotheses — what each outcome would mean,
written before the arms started — which is the only thing that lets a later session tell a surprising
result from an arm that was never going to answer anything. Editing one to launch a new batch
destroys that, so **copy it to a new file instead**, and give the copy its own header.

`launch_c51_pilot.sh` says this about itself, from `launch_c51_wave.sh`'s header:

> The wave-A launcher (`launch_c51_pilot.sh`) is kept as the record of that launch and **must not be
> edited while it runs** — bash reads a script incrementally, so editing a running one can make it
> execute garbage.

Both halves apply to every launcher here: never edit one while its wave is up, and don't edit a
finished one at all.

## Why `seed_from_checkpoint.sh` exists — a resume takes the *last* checkpoint, not the best

`common.Checkpointer.initialize_or_restore()` restores whatever the dir's `checkpoint` state file names,
which for a finished arm is its **final** step. So there was no way to continue an arm from its own best
checkpoint, which is what `b42`/`b43` needed. Seeding a fresh dir solves it and buys two more things:
the original arm's checkpoints and graph history are not overwritten by the continuation, and because the
new dir starts with exactly one checkpoint, **every later `ckpt-*` in it belongs to the new run** — so a
close-out cannot mix two arms' weights at the same step.

**It copies the replay buffer on purpose, and that is measured, not cautious.** A fresh dir holds only
the 1000 random transitions `random_play()` writes, and training samples a batch from those on step 1.
Continuing `b29b` @1447000 for 5k steps: **80% → 50% perfect without the buffer, 90-100% with it.** The
buffer comes from the source's *final* step rather than the chosen one, so it is slightly ahead of the
weights being restored — still on-policy data from the same arm, and PER re-prioritises against the
restored net within a few thousand samples.

**Continuing an arm at a different learning rate needs `training.enforce_learning_rate`**, which post-dates
this script by minutes: Adam's rate is a checkpointed Variable, so before that fix a resume silently
restored the saved rate and `SNEK_LEARNING_RATE` was a no-op. Check the run prints its reset line.

## Conventions every script in here follows

**Each one `cd`s to `snek2/` from its own location** — `cd "$(dirname "$0")/../.."`, or
`${0:a:h}/../..` in `refresh_charts.sh`'s zsh — so the caller's cwd is never load-bearing and the
usage lines (`cd snek2` then `bash hyperparamTuning/scripts/<name>.sh`) are a habit rather than a
requirement. **That `../..` is what changed when these moved down a directory**; a launcher copied
from somewhere else needs it checked.

**Every `pgrep` is filtered, and both filters are load-bearing.** `pgrep -f` is a substring match, so
a bare `pgrep -f snek2.py` also matches git pathspecs and the Airbnb telemetry `curl` whose payload
names `snek2/snek2.py` — it once read 6 trainers against 4 running — and a bare
`pgrep -f eval_checkpoints.py` matches the `chart_viewer` that a laptop eval spawns and which
outlives the evals by design, so a wait written on it never ends. Hence `| grep python` and
`| grep -v chart_viewer`. The polls count processes rather than tracking pids, because `kill -0`
succeeds on a zombie.

**A launcher refuses rather than breaking the 4-trainer limit**, and the ones written after batch 36
also refuse while a close-out is running, since a wave launched on top of one gets a third of the
cores.

## Not covered by the diagnostics push authorization

CLAUDE.md lets `perDiagnostics/`, `diagnostics/` and `tests/` go up unreviewed because they only
measure. **Nothing here qualifies** — these scripts start trainings, start evals, and in
`launch_c51_batch.sh`'s case commit and push. They are code, so a change to one waits for the user's
approval like any other.

## Only `refresh_charts.sh` is called from Python

[`../pick_c51_lr.py`](../pick_c51_lr.py)'s `refresh_charts()` runs it via `subprocess` when writing
docs. Nothing else in the repo invokes any of these; the rest are launched by hand or by each other.
