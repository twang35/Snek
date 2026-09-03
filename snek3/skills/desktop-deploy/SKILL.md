---
name: desktop-deploy
description: Deploy snek3 code to the desktop box the-claw-den and restart its daemon. Use for "deploy to the desktop", "get the box on the new code", "the box is running old code", or when a queued job needs a change that is only on the laptop.
model: sonnet
---

# Deploy code to the desktop

The box runs from `~/Snek` on `master`. ssh to it needs no separate approval.

## The deploy

```
ssh the-claw-den 'Snek/snek3/desktop/deploy'            # fetch, settle runs/ collisions, ff-merge
ssh the-claw-den 'sudo systemctl restart snek3-runner'   # only if desktop/runner/* changed — see below
```

**Read the output in full. Never pipe it to `tail`** — that hides the failure and its exit code. The
script prints every colliding file with what it did to it, then `HEAD <old> -> <new>`. Exit codes:

| exit | meaning | what to do |
|---:|---|---|
| 0 | merged; HEAD moved | done, restart if the daemon's code changed |
| 3 | **a colliding JSON differs from the commit; nothing was touched** | a live arm's `_evals.json` or `_checkpoint_evals.*` got committed from the laptop. `git rm --cached` those paths on master in their own commit, push, run `deploy` again. Never overwrite the box's copy |
| 4 | the fast-forward itself failed | local commits on the box, or a diverged master — read the git output |

**Why collisions are the normal case, not a mistake.** Since 2026-09-02 every progress update commits
the charts of every arm, including the ones the box is still training, so the box always holds
untracked `runs/*.png` and `.md` that master also carries, and a bare `git merge --ff-only` would abort
on every one of them. `deploy` settles each colliding file by what it is: pictures are deleted (the
next eval redraws them, and a finished arm's committed copy is the same picture), any other file whose
bytes match the incoming blob is staged at that hash (a closed batch imported from `results`), and a
differing JSON stops the run. `--dry-run` prints the plan without doing it.

**Until `desktop/deploy` is on the box, the first deploy that carries it is done by hand** — the same
three moves, typed:

```
ssh the-claw-den 'cd ~/Snek && git fetch origin master && git ls-files --others --exclude-standard -- snek3/runs \
  | while read f; do git cat-file -e origin/master:"$f" 2>/dev/null && case "$f" in *.png|*.md) rm -- "$f";; esac; done; \
  git merge --ff-only origin/master && git rev-parse --short HEAD'
```

If that still aborts, the leftover collisions are JSON: compare each with `git hash-object` against
`git ls-tree -r origin/master`, `git add` the identical ones on the box, and treat any that differ as
the exit-3 case above.

## Restart only when the daemon's own code changed

Jobs are fresh processes — the daemon shells out to `train.py` and `tools/closeout.py` — so a change
under `snek3/ppo/`, `snek3/tools/`, `snek3/train.py` or anything else the *jobs* run is live for the
next job as soon as the merge lands, with no restart. Restart when `snek3/desktop/runner/*` or the
unit file changed. It needs `sudo`, which ssh alone does not carry, so skip it when it is not needed.

## Restarting is safe with arms mid-run, when it is needed

`KillMode=process`, and jobs are launched detached with `setsid`. They self-terminate at
`SNEK_MAX_STEPS` and the daemon re-adopts them by pid on the next poll. So a restart does not stop
work — which also means it is not a way to stop a job (see `stop-run`).

The daemon runs on **base** python so it can start before the conda env exists. `PYTHON_BIN` in
`desktop/config/host.env` is the env python and is what actually needs torch. `host.env` is not in
git — only `host.env.example` is — so a change to machine identity or the hard ceilings is edited on
the box.

## Confirm it came back

```
ssh the-claw-den 'Snek/snek3/desktop/trigger'          # 0 healthy, 2 not polling, 1 unreachable
git fetch origin ops-status && git show origin/ops-status:status.json
```

**The fetch is mandatory.** Without it you read a stale local ref whose embedded timestamp reads like
a dead daemon.

If a git write was interrupted, `index.lock` can outlive the daemon and every later publish fails
while jobs keep running — a frozen heartbeat over healthy work. `clear_stale_locks` handles this now;
if a heartbeat is frozen but `trigger` says the daemon is polling, that is the shape to look for:
`journalctl -u snek3-runner | tail -50`.
