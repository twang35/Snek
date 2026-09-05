---
name: desktop-deploy
description: Deploy snek3 code to the desktop box the-claw-den and restart its daemon. Use for "deploy to the desktop", "get the box on the new code", "the box is running old code", "restart the daemon", or when a queued job needs a change that is only on the laptop.
model: sonnet
---

# Deploy code to the desktop

The box runs from `~/Snek` on `master`. **The code must be committed and pushed to `origin/master`
first** — a deploy is a fast-forward to that ref, and pushing code needs the user's approval of *that*
change (root `CLAUDE.md`, Git workflow).

## The deploy: one command, no ssh, no sudo

```
cd /Users/tony_wang/Projects/Snek && snek3/desktop/queue_action deploy
```

It commits a `deploy-<stamp>.json` action to `ops`, triggers the daemon, and waits for the ledger: the
box fetches and fast-forwards with its own `desktop/deploy`, and **restarts itself only if the merge
touched `desktop/runner/` or `desktop/systemd/`** (a restart is the daemon recording the action, publishing
and exiting; systemd's `Restart=always` brings it back in 10 s, and it re-adopts the running scheduler by pid --
the scheduler and everything under it are detached and never restart with the daemon).
`--restart` forces the restart, `restart` alone just restarts, `--no-wait` returns after queueing. The
last line printed is the ledger state and the box's `head`; **check `head` is your master sha**.

Exit codes: 0 done, 1 the action failed (the record's `error` and `attention` say why — exit 3 from
`deploy` is a differing JSON and nothing was touched: `git rm --cached` those paths on master, push,
queue a new deploy), 2 not terminal within the timeout (the daemon may be mid-restart: read `ops-status`
again in a minute).

**Why not ssh.** `ssh the-claw-den 'sudo systemctl restart snek3-runner'` is refused by the laptop's
permission classifier (2026-09-05), and the box's passwordless sudo was never the constraint. The action
path runs as `claw` with no sudo at all. Named actions only: nothing on `ops` runs arbitrary shell.

## By hand, if the bus is down

```
ssh the-claw-den 'Snek/snek3/desktop/deploy'            # fetch, settle runs/ collisions, ff-merge; read ALL of it, never | tail
```

then, **typed by the user at the prompt** (an agent's sudo over ssh is refused):
`! ssh the-claw-den 'sudo systemctl restart snek3-runner'` — only if `desktop/runner/*` or the unit changed.
Arms and passes are fresh processes, so a change under `ppo/`, `train.py`, `tools/closeout.py` is live for
the next one with no restart. **A change to `tools/scheduler.py` or `tools/window.py` reaches the box only
when the scheduler is next started**, since the running scheduler is the old code -- and **a pause does not
restart it**: a paused scheduler blocks inside its wait loop and never exits, and lifting the hold starts a
new one only if none is alive (found 2026-09-05; the earlier "pause, wait, unpause" advice restarted nothing).
The procedure is: set `"paused": true` on `ops` (`desktop-batch`), wait until `status.json`'s `running` is
empty (the wave's arms and its pass have finished), then kill the scheduler **by the pid `status.json`
names** -- `ssh the-claw-den 'kill <scheduler.pid>'`, never a pattern -- and set `"paused": false`. The
daemon sees the hold lifted with no scheduler alive and starts one on the new code; its arms are untouched
(their own sessions) and are adopted through `runs/.live/`. Without the pause the daemon also restarts a
dead scheduler on its own once work is pending, after a 10-minute backoff, or at once on `trigger`. `desktop/README.md` ("Deploy over the bus", and the deploy script's exit codes 3 and 4) has the
detail, including the one-time settling of the box's pre-2026-09-03 files.

## Confirm it came back

```
ssh the-claw-den 'Snek/snek3/desktop/trigger'          # 0 healthy, 2 not polling, 1 unreachable; prints both boxes' at_a_glance
git fetch origin ops-status && git show origin/ops-status:status.json | python3 -c "import json,sys; s=json.load(sys.stdin); print(s['iso'], s.get('head'))"
```

**The fetch is mandatory** — without it you read a stale local ref whose embedded timestamp reads like
a dead daemon. If a heartbeat is frozen but `trigger` says the daemon is polling, a stale `index.lock`
is the shape to look for: `journalctl -u snek3-runner | tail -50` on the box (`clear_stale_locks`
handles the known case).
