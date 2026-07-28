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
`<policy_name>_history.json`. **Never delete anything in `snek2/runs/`** — not
even throwaway smoke output like `snek2/runs/smoke.*`. The user reviews these by
hand, so they have to survive after a run finishes, and a deleted graph can't be
recovered without re-running the training that produced it.

This is the opposite of the checkpoint rule above: `savedPolicies/smoke/` is
scratch space and is fine to clear, `runs/` is output to keep. Letting a later run
of the same policy overwrite these files is expected and fine; removing them is
not.

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

Read those before starting or judging any tuning run, and update them as runs
start, finish, or get killed — they are the handoff between sessions.

### "Progress update" means look, don't touch

When the user asks for a progress update or to check on progress, that is
**read-only with respect to running processes**: analyse the evals, refresh the
charts, update the docs, report. **Do not kill, stop, or restart any arm** — not
even one that looks finished, is past the 4-hour cap, or is clearly failing.

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
  Check with `ps -eo pid,command | grep "[s]nek2.py" | grep -v spawn_main`.
- **This domain is very noisy** — the same config has produced final scores of
  62.5 and 18.0. Never conclude anything from a single run; repeat promising
  configs 2-3 times.

Hyperparameters are overridden with `SNEK_*` env vars (see `tuned()` in
`snek2.py`), so variants run side by side without editing files.

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
