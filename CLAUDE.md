# Snek

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
