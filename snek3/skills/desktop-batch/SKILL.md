---
name: desktop-batch
description: Queue a training batch, eval wave, smoke or benchmark on the desktop box the-claw-den, and confirm it started. Use for "queue this on the desktop", "run this batch on the box", "send these arms to the-claw-den", or to pause/drain/retune the daemon.
---

# Queue work on the desktop

**Pushing to `ops` starts real work on another machine, so it needs the user's approval for *that*
job.** Approval of an earlier job does not carry over. Draft the specs, show them, then push.

`snek3/desktop/README.md` has the full design; this is the procedure.

## 1. Write the specs in an `ops` worktree

Use a worktree, not `git checkout ops` — the user reads diffs in the master working tree and a
branch switch moves their work out from under them.

**`ops` is usually already checked out in a worktree from an earlier session**, often under a
scratchpad path that may or may not still exist, so `git worktree add` fails with
`fatal: 'ops' is already used by worktree at …`. This handles all three states — reuse, stale
registration, none yet:

```
cd /Users/tony_wang/Projects/Snek
git fetch origin ops
git worktree prune          # drop worktrees whose directory is gone
OPS=$(git worktree list --porcelain | awk '/^worktree /{p=$2} /^branch refs\/heads\/ops$/{print p}')
[ -n "$OPS" ] || { git worktree add /tmp/snek-ops-wt ops && OPS=/tmp/snek-ops-wt; }
git -C "$OPS" merge --ff-only origin/ops && echo "ops worktree: $OPS"
```

Then write one JSON file per arm into `$OPS/snek3/desktop/queue/pending/<id>.json`.
`queue/examples/` holds a worked spec per type.

| field | required | means |
|---|---|---|
| `project` | **yes, no default** | must be `"snek3"`. The guard against `ops`'s ~150 retired snek2 specs |
| `id` | yes | unique; ledger key and log name. `b<n><letter>-...` groups arms into a batch |
| `type` | yes | `train`, `smoke`, `benchmark`, `eval` |
| `policy` / `policies` | yes | the checkpoint dir; `policies` is eval-only and makes one wave own every arm |
| `max_steps` | no | `SNEK_MAX_STEPS`. **Absolute**, not "this many more" |
| `env` | no | any `SNEK_*` knob; wins over the runtime defaults. See `docs/running.md` |
| `label` | no | one line for `at_a_glance` |

**Omit `selector` and `episodes` on an eval.** Absent means `tools/closeout.py`'s own defaults, which
*are* the protocol. `eval_shards` and `priority` are in
[`desktop/README.md`](../../desktop/README.md) with the rest.

A malformed spec is recorded against its filename and skipped, never raised into the loop — so a bad
commit cannot stop the box, but it also will not tell you until after the push. **Validate first with
the daemon's own parser**, which is stdlib-only and runs here:

```
cd /Users/tony_wang/Projects/Snek/snek3/desktop && python3 -c "
import sys, glob; sys.path.insert(0, '.')
from runner.job import parse_job, JobError
for p in sorted(glob.glob('$OPS/snek3/desktop/queue/pending/*.json')):
    try: parse_job(open(p).read(), source=p); print('OK  ', p)
    except JobError as e: print('BAD ', e)
"
```

## 2. Push, then trigger

The trigger is what makes it start now rather than within ten minutes.

```
git -C "$OPS" add snek3/desktop/queue/pending/ \
  && git -C "$OPS" commit -m 'queue <batch>' \
  && git -C "$OPS" push origin ops
ssh the-claw-den 'Snek/snek3/desktop/trigger'
```

`trigger` exits 0 healthy, 2 not polling, 1 unreachable. ssh to this box needs no separate approval.

## 3. Confirm it started

```
git fetch origin ops-status && git show origin/ops-status:status.json
```

**The fetch is mandatory** — without it you read an old local ref whose embedded timestamp looks like
a dead daemon. Read `at_a_glance` first, then `attention`.

**Nothing starts while anything is running.** `_dispatch` returns early whenever any job is live, so
waves never overlap and a freed slot is never backfilled mid-wave. A queued batch behind a running
one is normal, not stuck. With `auto_stage_b` on, each finished training queues its own stage B at
priority 10.

## Retune or hold the box

`config/runtime.json` on `ops`, re-read every network cycle (600 s; `trigger` applies it now). Same
worktree, same push. Keys: `max_trainers` 8, `eval_shards` 16, `poll_seconds` 30, `git_seconds` 600,
`torch_threads` 1, `omp_num_threads` 1, `nice` 0, `disk_min_gb` 5, `paused`, `drain`, `auto_stage_b`,
`viewer`.

**A malformed or unknown-key file is rejected whole and the last known-good config kept**, and it says
so in `status.json`. `max_evals`, `HARD_MAX_EVALS` and `clamp_total_shards` were removed on
2026-08-29 — a `runtime.json` still naming one is rejected.

`paused` / `drain`: finish what is running, start nothing new. Set one before killing a desktop job
(see the `stop-run` skill) or the freed slot refills within one poll.
