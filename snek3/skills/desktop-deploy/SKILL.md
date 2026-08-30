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
`results` branch at close-out; that is what it is for. To unblock now, remove the offending untracked
files on the box — they are regenerated on the next eval.

## Restarting is safe with arms mid-run

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
