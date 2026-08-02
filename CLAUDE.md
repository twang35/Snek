# Snek

## Work as a collaborator

**Ask questions when a request is ambiguous** in a way that changes the work — which
arms to stop, what "best" means for a selection, how long to let something run. Don't
ask about things already answered by the code, the tuning docs, or an obvious default.

**Volunteer better approaches**, before starting and as they occur mid-task. Give a
recommendation rather than a menu, then proceed with it unless redirected — don't block
on approval for ordinary work. This is explicitly wanted: the largest result in this
project so far came from an unrequested suggestion to diff `snek2` against the old
`theSchlong` implementation instead of continuing to sweep hyperparameters.

**Say so when evidence undercuts the plan**, including a previous conclusion written in
these docs. This investigation has overturned several of its own findings, and each
retraction was more valuable than the section it replaced.

This project uses a dedicated conda environment named `snek` (Python 3.10).

Before running any Python scripts or installing packages in this project, activate it:

```
conda activate snek
```

If running commands non-interactively, use `conda run -n snek <command>` instead.

**Caveat:** `conda run` buffers/relays the wrapped process's stdout internally,
even with `python -u`. If you redirect a background process's output to a log
file and poll that file (e.g. while a training run is going), the log can
stay completely empty for 90+ seconds while the process is actually running
fine — and killing it with `kill -9` at that point discards whatever was
sitting in `conda run`'s buffer, permanently. For any background run where you
need to see live/incremental output, invoke the env's python binary directly
instead, e.g.:

```
/opt/miniconda3/envs/snek/bin/python -u snek2.py smoke > log 2>&1 &
```

Reserve `conda run` for short one-shot commands where only the final
exit/output matters.

## Smoke tests

`policy_name` (the arg to `snek2.py`, e.g. `python snek2.py train`) doubles as
the checkpoint directory name under `snek2/savedPolicies/`. The user runs real
training under `train` — never delete or overwrite `snek2/savedPolicies/train/`.

When running a smoke test (verifying a code change doesn't crash, timing
startup, etc.), always pass `smoke` as the policy name:

```
/opt/miniconda3/envs/snek/bin/python -u snek2.py smoke > log 2>&1 &
```

This keeps smoke-test checkpoints isolated in `snek2/savedPolicies/smoke/`,
which is safe to `rm -rf` before/after a test. Never run `rm -rf
snek2/savedPolicies/` wholesale — target the `smoke/` subdirectory only.

## Run artifacts

Every eval writes `snek2/runs/<policy_name>.{png,md}` plus
`<policy_name>_history.json`. **Keep everything from a real training arm.** The user
reviews these by hand, a deleted graph cannot be recovered without re-running the
training that produced it, and disk is not a constraint. That covers the whole
`b<n><letter>-<name>` family and their `_checkpoint_evals*.json` measurements, which
the tuning docs cite directly.

**Throwaway output is fine to delete.** Smoke tests, speed benchmarks and
verification runs are not results — they are scaffolding from proving a change
works, and they age into noise that makes the directory harder to read. Judge by
what produced it, not by where it lives:

| keep | delete |
|---|---|
| `b8f-disc9975seed2.*`, `b8d-disc995clip_checkpoint_evals.json` | `smoke.*`, `smoke_evals.json` |
| any arm named in `completedRuns.md` or `runs.md` | `bench-*` and similar timing scaffolding |
| `_checkpoint_evals*.json` for a real arm | `champion_*` / `smoke_checkpoint_evals_*` verification evals |

When in doubt, keep it — the asymmetry still holds, since a wrongly kept file costs
a few KB and a wrongly deleted one costs a training run. Check for references first
(`grep -rn <name> --include='*.md' .`): the tuning docs link specific files by name,
and a deletion that breaks one of those links is worse than the clutter.

Letting a later run of the same policy overwrite these files is expected and fine.

## `snek2/hallOfFame/` — record checkpoints, never delete

Copies of the best policies produced so far, with a README covering how to run one by
hand. They live outside `savedPolicies/` precisely so the `max_to_keep` rotation cannot
delete them — a long run otherwise eventually evicts its own best checkpoint, which has
already destroyed evidence once (`b5c-schlongIS`'s 17.0% peak).

**Never delete anything here**, and add an entry whenever a run produces a checkpoint
worth keeping. Each is two files, ~190 KB. The README documents the copy-in-and-evaluate
procedure; both commands in it were verified working when written.

**Matching widths do not mean a checkpoint still works, and this has already bitten.** A checkpoint
restores whenever the vector is the same *length*; nothing checks that the values still mean what
they meant. The observation changed four times on 2026-08-02, and for part of that day it was
coincidentally back at its original 20 values while two indices meant something different — so
every hall-of-fame checkpoint restored with no warning and played like a beginner, the champion
going from **90.3%** to scoring **0, 0, 1** over three episodes.

An input that was constant is the specific trap: `game_over` sat at 0 in every state a policy acts
in, so its weights were never constrained by anything, and the index it occupied now carries
board-fill. Arbitrary weights times a live signal.

So when the observation changes, **record which indices changed meaning in the hall of fame
README, and name the last commit whose observation matches those checkpoints** (currently
`e4514a8`). A width change at least fails loudly — the vector is 23 now, so those checkpoints error
out — but that is luck, not a safeguard.

**Append new per-action values after the existing blocks rather than interleaving them.** The
frozen diagnostics in `hyperparamTuning/diagnostics/` read `head_with_tail` at `obs[9 + 2 * i]`,
and putting the safe-to-chase triple after the group block instead of inside it kept them correct
through every change on 2026-08-02.

Note that deleting `<policy_name>_history.json` also throws away the graph's
history for that policy, so its next run restarts the curve from the current
iteration instead of continuing.

## Hyperparameter tuning

There is an ongoing, resumable investigation into configs that reach the highest
possible perfect-game percentage while learning consistently. Reducing
catastrophic forgetting is a means to that end, not the goal itself. Everything
lives in **`snek2/hyperparamTuning/`**:

- `runs.md` — what is running and what to run next. **Start here** when picking
  the task up mid-flight. Kept short on purpose.
- `hyperparamTuning.md` — the protocol: metrics, stop criteria, how to judge, how
  to launch, available knobs. Read this for how the machinery works.
- `findings.md` — what is established and what has been falsified. Read before
  proposing an experiment, so a closed question doesn't get reopened.
- `completedRuns.md` — every finished arm: config, final numbers, verdict.
- `failureModes.md` — the four ways a policy degrades here and how to tell them
  apart. They look alike in a single trailing window.
- `charts.md` + `charts/` — progress graph per arm; snapshot copies, refreshed
  with `refresh_charts.sh`.

Keep the split clean: `runs.md` is current state and forward plan only. Results go
to `completedRuns.md`, conclusions to `findings.md`, and anything about *how to
measure or judge* to `hyperparamTuning.md`. The reason is that `runs.md` grew to 950
lines of interleaved status, results and conclusions and stopped being usable.

**Every running batch keeps its description in `runs.md`** — why the batch is shaped
that way, what each arm isolates, and what each possible outcome would mean. Keep it
there while *any* arm of the batch is still running, then move it to
`completedRuns.md` when the last arm stops. Without the rationale, a future session
can't tell whether a surprising result is informative or came from an arm that was
never going to answer anything.

Read those before starting or judging any tuning run, and update them as runs
start, finish, or get killed — they are the handoff between sessions.

### "Progress update" means look, don't touch

When the user asks for a progress update or to check on progress, that is
**read-only with respect to running processes**: analyse the evals, refresh the
charts, update the docs, report. **Do not kill, stop, or restart any arm** — not
even one that looks finished, is past the 4-hour cap, or is clearly failing.

A progress update means updating **`charts.md` as well as `runs.md`**. Running
`refresh_charts.sh` only copies the PNGs — it does not add anything to `charts.md`,
so a new arm silently ends up with an image and no entry. Every arm needs a row in
the "Every arm at a glance" table and a `### <policy> — <change>` section with a
stats line, a short reading of what the chart shows, and the `![...]` image. Check
with:

```
ls snek2/hyperparamTuning/charts/*.png | sed 's|.*/||;s|\.png||' | sort > /tmp/have
grep -o 'charts/[a-zA-Z0-9-]*\.png' snek2/hyperparamTuning/charts.md | sed 's|charts/||;s|\.png||' | sort -u > /tmp/doc
comm -23 /tmp/have /tmp/doc   # anything listed is an undocumented arm
```

This drifted once already: batches 5, 6 and 7 accumulated 12 arms with images and no
sections, because `refresh_charts.sh` succeeding looked like the charts were handled.

Deciding a run is done is the user's call, not a side effect of asking how it's
going. A long run may still be producing the late-horizon data that makes it
worth something, and judging an arm dead has already been wrong here more than
once. If a run looks finished and no slot is needed, say so in the report as a
recommendation and let the user decide.

Stop an arm only when the user asks, or when they ask for something that
plainly needs a free slot ("start the next batch" with the budget full) — and
then say which ones are stopping and why.

Two other rules are easy to get wrong:

- **Never run more than 4 snek trainers at once**, counting human-started ones.
  Check with `pgrep -fl "python -u snek2.py"`. Do **not** count with
  `grep "[s]nek2.py"` — Airbnb's git telemetry fires `curl` processes whose JSON
  payload contains `snek2/snek2.py` as a git argument, so that pattern
  intermittently over-counts by several processes for the few seconds each curl
  lives. It read 6 trainers once when only 4 were running.
- **This domain is very noisy** — the same config has produced final scores of
  62.5 and 18.0. Never conclude anything from a single run; repeat promising
  configs 2-3 times.

Hyperparameters are overridden with `SNEK_*` env vars (see `tuned()` in
`snek2.py`), so variants run side by side without editing files.

### Logging: quiet by default, `SNEK_DEBUG=1` for everything

Training runs **quiet**: one compact line per 10 evals, and nothing else after startup.
`SNEK_DEBUG=1` restores the original output verbatim — per-200-step loss lines, the
five-line eval block, `Saved checkpoint` per eval, perfect-game and high-score banners.
Use it when a run is actually being debugged, not for status.

**Read status from `runs/<policy>_evals.json`, not from the log.** Every history file
carries a precomputed `summary` block:

```
step, evals, trailing_now, peak_trailing{value,step}, best_perfect30{value,step},
recent_perfect30, max_single_eval, dead_since, zero_since, epsilon
```

That is exactly what a progress check needs, so reading `summary` from each arm replaces
scanning the eval series.

**Use `zero_since`, not `dead_since`, to ask whether an arm is dead now.** `dead_since` is
the earliest sustained-zero stretch and is history; `zero_since` is the start of the
*current* unbroken stretch and is `null` if the latest eval is above threshold.
`b8d-disc995clip` carried `dead_since=275000` while going on to a 36% best-30 window, so
`dead_since` alone would have condemned the best arm in its batch.

Neither is a verdict. Arms have recovered from trailing 0.3, and `b8g-clipseed3` recovered from
**1.2M steps** near zero — to 63.7 trailing — before collapsing for good. Read `zero_since`
against `step` for the duration, and only call an arm dead after hundreds of thousands of steps
pinned there *and* no recovery arc still in progress.

**Check `zero_since` is actually present before trusting it.** Arms launched before the field
was added keep overwriting the backfill from their old in-memory `run_report`, so the key is
absent — and `summary.get('zero_since')` returns `None` for a missing key, which reads
identically to "alive right now". That made a status check on `b8d-disc995clip` report "not
dead" for a reason unrelated to the data. Use `'zero_since' in summary`, or recompute with
`run_report.build_summary(rows['evals'])`.

The one log line worth grepping is `hyperparameter override:` at startup, which confirms
an arm got the config intended. That prints in both modes.

## Writing the docs

**Lead with a table, then explain underneath.** Any time a section covers a *set* of
comparable things — run statuses, queued experiments, config options, ranked
metrics — open with a compact table, one row per item and only columns that carry
signal, then put the rationale in `####` subsections or paragraphs below it. Keep
cells to a few words and push anything longer down into the prose. The point is
that the state of things should be readable at a glance without reading paragraphs
to reconstruct facts that belong in a grid.

Two markdown traps that look fine in the source and render wrong:

- **`A.`/`a.` are not list markers** — only digits and `-`/`*` are. Lettered items
  collapse into one run-together paragraph. Use `#### A. Thing` headings when the
  letters are cross-referenced elsewhere.
- **Duplicate numbers in one list renumber silently** — two items both written `6.`
  render as 6 and 7, shifting everything after them and breaking any prose that
  refers to "item 12". Cross-reference items by name, not number.

Worth a grep for `^[A-Za-z]\.\s` and a duplicate-number check after editing any md
file here, since the user reads these rendered rather than as source.

## No audio — never call bare `pygame.init()`

`pygame.init()` starts *every* subsystem, including `pygame.mixer`, which opens a real
CoreAudio output stream **per process**. `SDL_VIDEODRIVER=dummy` does not affect audio, so
headless workers were still holding audio streams: 10 idle env processes drove
`coreaudiod` to **15% CPU**, and evals routinely run several 10-worker processes at once,
which is what made it look like the evals were sending `coreaudiod` off the rails.

Nothing in this project plays sound. `Snake.Game.__init__` therefore inits only what it
uses — `pygame.display.init()` and `pygame.font.init()` (`pygame.time`/`sprite`/`draw` need
no init) — and `snek2.py` and `eval_checkpoints.py` both set `SDL_AUDIODRIVER=dummy` before
any pygame import as a second line of defence.

If a future change needs another subsystem, init that subsystem by name. Measure with
`ps -o %cpu= -p $(pgrep -x coreaudiod)` while workers run; it should read 0.0.

## Tests: `snek2/tests/` — write them, and run them

**When a change has logic worth pinning down, add a test for it in the same pass.** Coverage here
is sparse — one file, 26 assertions, all of them on `group_obs` and `body_and_wall_collisions` —
so most of the environment has nothing guarding it, and the sparseness is a reason to add rather
than a precedent to follow. Worth a fixture whenever a change involves:

- a conditional whose branches are easy to collapse later (the eat/no-eat distinction has now
  been got wrong twice, in `update_grid` and again in `group_obs`)
- an index or coordinate convention (the padded grid's `(y + 1) * cols + (x + 1)` bit layout, the
  `SCREENTILES + 1` bounds guards)
- a rule someone could "simplify" without seeing why it exists, like the clause for stepping onto
  the cell the tail is vacating
- an edge case that took thinking to get right — if it needed reasoning, it needs a fixture,
  because the reasoning does not survive in the diff

The bar is not complexity, it is whether a future edit could break the behaviour without anything
noticing. These fixtures are cheap: a hand-written grid, a call, one `assert`.

`tests/test_state_helpers.py` covers `group_obs` and `body_and_wall_collisions` on
hand-written grids. **pytest is not installed in the `snek` env**, so run them directly:

```
cd snek2
PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -c "
import sys; sys.path.insert(0, 'tests')
import test_state_helpers as t
for n in [x for x in dir(t) if x.startswith('test')]:
    try: getattr(t, n)(); print('PASS', n)
    except Exception as e: print('FAIL', n, type(e).__name__, e)
"
```

13 of these were dead for two signature generations — they called `group_obs` with a food
position it had stopped taking, so they raised `TypeError` rather than failing an assertion,
and a `TypeError` looks like noise if nobody is watching. **Every test covering `group_obs`
was dead at the point `group_obs` was changed.** Run the suite before and after touching
`state_helpers.py`, and check the failure *type*: a `TypeError` means the test is stale, not
that the code is fine.

For refactors, also diff observations against a fixed-seed run — byte-identical output over a
few thousand steps catches what 26 assertions do not.

**A passing suite is not coverage of the change you just made.** `group_obs` took a third
signature in 2026-08-02 — `next_tail_pos` was added, because the tail moves on the step the head
does — and all 24 existing tests passed *before and after* the behaviour changed, since every
fixture was an open board where the old and new answers agree. Two tests were added to cover it,
and both were checked by mutating the implementation and confirming a test fails:

| mutation | tests that catch it |
|---|---|
| don't advance the tail at all (the old behaviour) | `test_hwt_enclosed_vacated_tail_still_reaches_the_tail` |
| advance it even on a move that eats | `test_hwt_eating_move_does_not_advance_the_tail` |
| drop the "stepping onto the vacated cell" clause | 7 of the existing tests |

The flip side happened the same day: fixtures for `group_obs` routinely include a fatal action
(a wall or a self-collision) alongside the legal ones, because a real snake facing three choices
usually has one that kills it. When fatal moves were changed to read zero outright instead of a
hypothetical "what if this move survived" value, **15 of the 44 tests broke immediately** —
not because anyone wrote a test for that change in advance, but because the existing fixtures
already exercised the case by accident. Comprehensive fixtures pay for themselves in ways
written for a different purpose; it is the reason a test failure here is worth reading rather
than reflexively rewriting the assertion to match.

That mutation check is worth repeating for any future change here: write the fixture, then break
the code deliberately and confirm the fixture notices.

## Rendering is off by default — use `watch.py` to see a game

A game window costs **~5.2ms per frame**, and the game flips once per game step. That is a
round trip to the macOS window server; it is not our drawing code, and dirty rects do not
help, because the cost is per flip rather than per pixel. Everything else `render()` does is
2-4us per call.

**To watch a policy play, run `watch.py` — never turn a window on inside a run:**

```
cd snek2
PYTHONPATH=. python -u watch.py <policy_name>        # follows the newest checkpoint
PYTHONPATH=. python -u watch.py <policy_name> <step>  # pins one checkpoint
```

It renders in its own process, reloads the newest checkpoint between episodes so a live arm's
progress shows up without a restart, and costs training nothing — it only reads checkpoint
files. `WATCH_FPS` caps the frame rate (default 90; drop to 20-30 to follow the moves).

**Window size is `SNEK_TILE_PIXELS`, and it is cosmetic only.** Every pixel constant in
`snake_constants.py` — `TILE_SIZE`, `SCREENSIZE`, the sprite radii, the HUD — derives from it,
so one number scales the whole window; `watch.py` sets 15 (a 150x150 window) and training keeps
the default 10. Observations are built from tile positions and never pixels, verified by a
fixed-seed hash of every observation and reward over 40 episodes coming out byte-identical at
10, 20, 25 and 40 pixels per tile. **It must be set in the environment before
`snake_constants` is imported**, not assigned afterwards: `from snake_constants import *` binds
a copy, so a later assignment never reaches `Snake.py`.

The window title is `<policy_name> — ckpt <step>`, arm name first because macOS truncates from
the right and the arm is what tells two watchers apart. `Game.reset()` re-applies
`game.caption` every episode, so `watch.py` setting it once per checkpoint load survives — and
follows a live arm forward.

**Training cannot draw at all — there is no switch.** `SNEK_DISPLAY_EVAL` and
`SNEK_DISPLAY_TRAINING` are gone, along with the second environment that existed only to play
one eval episode where it could be drawn. `snek2.main()` selects `SDL_VIDEODRIVER=dummy`
unconditionally. `eval_checkpoints.py` keeps `EVAL_RENDER=1` for debugging a policy by hand
(worker 0 only, ~5x the wall clock), and that plus `watch.py` are the only ways to see a game.

Removing the display path made every eval episode parallel, which is worth ~24% of an eval:
at champion skill the old split shape measured 5.95s (1.67s serial + 4.28s round) against
4.55s for one round of ten. The tenth worker costs ~0.27s, because a round ends with its
slowest episode and slowest-of-ten is barely worse than slowest-of-nine.

**A process that will not draw must select the dummy video driver, not merely skip
drawing.** `Game.__init__` calls `pygame.display.set_mode()` unconditionally and `reset()`
blits the background and flips, so a process with a real driver and `display=False` opens a
window, paints it white once, and never touches it again. That looks like a broken window
rather than an absent one, and it is exactly what happened when training's default first
flipped to headless.

`main()` does that, not module scope: `eval_checkpoints.py` and `watch.py` both import
`snek2` for `build_q_net`, and an import-time `setdefault` would suppress *their* windows too.

**The two paths were slow for different reasons, and it matters.** In `eval_checkpoints.py`
the visible worker sits *inside* the `ParallelPyEnvironment`, which steps every worker
together and waits for the slowest — so one window paced all ten, and a 30-episode eval went
**70.1s → 14.0s** when it was turned off. Training is not like that: `compute_avg_return()`
runs the displayed episode alone and *then* calls the parallel batch, so the window never
gated a worker and the cost was purely additive — **15.6s against 1.3s** for one champion eval
episode, every eval.

**Measure a render change per-episode with a fixed policy, not by timing a training run.**
An earlier "10k steps 83s → 56s" claim here was learning-speed variance, not the flag it was
attributed to: first-eval scores across smoke runs ranged 0.1 to 39.1, episode length tracks
skill, and the flag in question was a no-op at the time anyway.

**When something here seems slow, profile the critical path rather than the hot-looking
code.** An 11x speedup of the observation code and a 6.8x speedup of policy inference both
moved eval wall clock by approximately nothing, because neither was the slowest worker.

## Active development

`snek2/` is the only directory that should be edited going forward. It's a
working copy of `theSchlong/`.

All other directories (`theSchlong/`, `theSchlongCardinalDirs/`, `humanPlayer/`,
etc.) are kept as-is for posterity — do not edit files in them.

## Git workflow

Leave finished work uncommitted. Do not run `git commit` or `git push` until the
user has explicitly approved that specific change — "push" or "commit this" from
the user is the go-ahead, and it applies only to the change in front of you, not
to later ones.

The reason is review: the user reads diffs in their editor, and committing moves
the change out of the working tree where it's no longer visible there. Staging
has the same problem, so don't `git add` either.

So the loop is: make the edit, describe what changed, then stop and wait. It's
fine to run read-only git commands (`status`, `diff`, `log`, `show`) at any
point — the restriction is only on commands that write history or the index.
