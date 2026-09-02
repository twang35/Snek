---
name: desktop-deploy
description: Deploy snek3 code to the desktop box the-claw-den and restart its daemon. Use for "deploy to the desktop", "get the box on the new code", "the box is running old code", or when a queued job needs a change that is only on the laptop.
---

# Deploy code to the desktop

The box runs from `~/Snek` on `master`. ssh to it needs no separate approval.

## The deploy

```
ssh the-claw-den 'cd ~/Snek && git fetch origin && git merge --ff-only origin/master'
ssh the-claw-den 'sudo systemctl restart snek3-runner'
```

**Read the merge output in full. Never pipe it to `tail`** — that hides the failure, and there is one
failure mode that happens often:

> `error: Untracked working tree files would be overwritten by merge`

**Untracked run artifacts on the box abort the fast-forward.** A live arm rewrites
`runs/<policy>.{md,png}` and `runs/<policy>_evals.json` on every eval, so a committed copy of those
paths collides forever and blocks every deploy for the hours the arm runs. The fix is at the source:
**never commit a live desktop arm's run artifacts** from the laptop. Desktop artifacts arrive on the
`results` branch at close-out; that is what it is for.

**Sort the colliding files by identity, not by name.** Get the hashes the incoming commit carries,
compare each with the box's own copy, and let the comparison decide:

```
# laptop: the blob hashes the commit carries
git ls-tree -r origin/master --format='%(objectname) %(path)' -- snek3/runs | grep <batch> > blobs.txt
scp blobs.txt the-claw-den:/tmp/blobs.txt
# box: which ones differ?
ssh the-claw-den 'cd ~/Snek && while read -r h p; do [ "$(git hash-object "$p")" = "$h" ] \
    || echo "DIFFERS: $p"; done < /tmp/blobs.txt; echo checked'
```

| the box's copy | what it means | what to do |
|---|---|---|
| **identical** | a closed batch imported from `results` and committed — the box holds the same bytes as untracked originals. 96 files on b7, 56 on b8 | stage them on the box at that hash; the merge then has nothing to overwrite and the index ends clean |
| **differs, and the arm or its stage B is still running** | a live arm's `_evals.json` or `_checkpoint_evals.json` got committed — usually by a progress update that swept `runs/` into a charts commit. b8, 2026-09-01: 8 stage-B files mid-pass | **untrack them on master**: `git rm --cached` those paths in their own commit, push, then merge. The laptop keeps its copies as untracked files, and the box's live copies are never touched. Do not wait for the pass to finish — the next eval changes them again |
| **differs, and the batch is closed** | the two copies really are different measurements | stop and look at the difference before touching either; this has not happened yet |

```
# box: stage the identical ones and merge. The grep drops whatever DIFFERS listed.
ssh the-claw-den 'cd ~/Snek && grep -v -e <differing-name-1> -e <differing-name-2> /tmp/blobs.txt \
    | cut -d" " -f2- | tr "\n" "\0" | xargs -0 git add -- && git merge --ff-only origin/master'
```

Both branches of that table follow from one rule in the root `CLAUDE.md`: **a live desktop arm's
artifacts are never committed from the laptop**; they arrive on `results` at close-out. A charts
commit made while a batch's stage B is running is where this goes wrong, so a progress update that
commits `runs/` should `git add` the `.png` and `.md` by name and leave every `*_evals.json` and
`*_checkpoint_evals.*` of a batch the box is still measuring. Deleting the box's copies also works and
is what this skill used to say, but it discards without checking, and for a finished batch nothing
regenerates them.

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
