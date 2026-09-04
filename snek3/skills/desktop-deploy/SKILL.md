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

**Since 2026-09-03 the box writes under `snek3/desktop/runs/` (gitignored), so collisions are history.**
Every job gets `SNEK_RUNS_DIR=<SNEK_DIR>/desktop/runs` from the daemon and nothing new lands in
`snek3/runs/` on the box. The settling below still runs, for whatever was there before the move.

**The one-time move, done on the first deploy that carries this change.** The daemon must be idle
(`paused: true` in `runtime.json`, then wait for the running wave and its stage B to publish), because
a live trainer keeps writing to the path it opened with, and a wave whose trainer files sit in one
directory and whose stage-B files land in the other publishes half of itself. Then, on the box:

```
cd ~/Snek/snek3 && mkdir -p desktop/runs
git ls-files --others --exclude-standard runs/ | xargs -I{} mv {} desktop/{}   # untracked only: live arms, .live/, .evalq/
mv runs/.live runs/.evalq desktop/runs/ 2>/dev/null; true
# tracked pictures the box redrew since they were committed: keep the bytes, then make the tree clean
cd ~/Snek && git diff --name-only | xargs -I{} cp {} snek3/desktop/runs/ && git checkout -- snek3/runs snek2/runs snek2/evals
git status --porcelain | grep -v "^?? snek2/\|host.env"     # must print nothing
```

The tracked files in `runs/` stay: they are master's archive, and after the checkout they are
byte-identical to it, with no writer left on the box to change them. Then `deploy`, restart (this
change is under `desktop/runner/`), unpause, `trigger`.

**What is left that could still refuse a fast-forward, and why it will not recur.** Only snek2's
untracked leftovers on the box (183 files on 2026-09-03: `snek2/runs/`, `snek2/evals/`, from the frozen
era's own daemon). They collide only if the laptop commits those exact paths, as the stragglers commit
did once; snek2 is frozen and its daemon stopped, so nothing adds to them, and `deploy.py` now settles
snek2 paths the same way as snek3's if it happens again.

**Why collisions were the normal case before that, not a mistake.** Since 2026-09-02 every progress update commits
the charts of every arm, including the ones the box is still training, so the box always holds
`runs/*.png` and `.md` that master also carries, and a bare `git merge --ff-only` would abort on every
one of them. `deploy` settles each colliding file by what it is: **the box's pictures are kept** —
saved, merged over, written back — because the box drew every chart the laptop has ever committed, so
the committed copy is always the older snapshot and a finished arm's final chart must not be replaced
by a mid-training one; any other file whose bytes match the incoming blob is staged at that hash (a
closed batch imported from `results`); a differing JSON stops the run. The kept pictures then show as
modified tracked files on the box, which is expected and harmless — the daemon publishes from its own
worktrees, and the next `deploy` handles them the same way. `--dry-run` prints the plan without doing
it.

**`desktop/deploy` has been on the box since 2026-09-02 18:50** (first run: HEAD `13def02e8` ->
`602f48b21`, 215 pictures kept, 64 identical b9 JSON staged, exit 0), so there is no by-hand path any
more. If the script itself is ever broken on the box, run the *incoming* copy from the fetched tree
rather than reconstructing its moves:

```
ssh the-claw-den 'cd ~/Snek && git fetch -q origin master && git show origin/master:snek3/desktop/runner/deploy.py > /tmp/snek_deploy.py && python3 /tmp/snek_deploy.py --no-fetch'
```

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
