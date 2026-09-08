---
name: queue-batch
description: Queue a snek3 training batch, eval spec, smoke or benchmark on the shared queue that both boxes pull from, pin it to one box, release a stranded wave, or pause/retune the desktop. Use for "queue this batch", "run these arms", "send this to the desktop", "run this on the laptop only", "release that wave", "pause the desktop".
---

# Queue work on the shared queue

**One queue, both boxes.** Specs live in `queue/pending/` on the `ops` branch; each box's scheduler claims
a wave of the highest-priority batch when it has nothing left, by pushing a claim to the `claims` branch
(`tools/claims.py`; design `plans/archive/shared-queue.md`). **A spec runs on whichever box comes free
first** unless it carries `"box": "desktop"` or `"box": "laptop"`. A batch may therefore span both boxes,
wave by wave; the pass ids (`b21-stageb-w2`) are numbered across both.

**Pushing to `ops` starts real work, on either machine, so it needs the user's approval for *that*
job.** Approval of an earlier job does not carry over. Draft the specs, show them, then push.

## 1. Write the specs in an `ops` worktree

Use a worktree, not `git checkout ops` — the user reads diffs in the master working tree and a branch
switch moves their work out from under them. **`ops` is usually already checked out in a worktree from
an earlier session**, often under a path that may no longer exist, so `git worktree add` fails with
`fatal: 'ops' is already used by worktree at …`. This handles all three states — reuse, stale
registration, none yet:

```
cd /Users/tony_wang/Projects/Snek
git fetch origin ops
git worktree prune          # drop worktrees whose directory is gone
OPS=$(git worktree list --porcelain | awk '/^worktree /{p=substr($0,10)} /^branch refs\/heads\/ops$/{print p}')
[ -n "$OPS" ] || { git worktree add /tmp/snek-ops-wt ops && OPS=/tmp/snek-ops-wt; }
git -C "$OPS" merge --ff-only origin/ops && echo "ops worktree: $OPS"
```

**`substr($0,10)` rather than `$2`, and that is not a style choice.** Invoking a skill with arguments
substitutes `$1`, `$2`, ... inside its body, including inside fenced code — measured when a skill
rendered this line as `p=fc-layout`. **No snippet in any skill may contain a bare `$<digit>`.**

Then one JSON file per arm into `$OPS/snek3/desktop/queue/pending/<id>.json`; `tools.sweep_specs <batch>
--out $OPS/snek3/desktop/queue/pending` writes a manifest batch's. `queue/examples/` holds a worked spec
per type.

| field | required | means |
|---|---|---|
| `project` | **yes, no default** | must be `"snek3"`. The guard against `ops`'s retired snek2 specs |
| `id` | yes | unique; the log name. `b<n><letter>-...` groups arms into a batch |
| `type` | yes | `train`, `smoke`, `benchmark`, `eval`; `deploy` / `restart` are actions — use `desktop/queue_action` (the `desktop-deploy` skill), not a hand spec |
| `policy` / `policies` | yes | the checkpoint dir; `policies` is eval-only and makes one pass own every arm |
| `max_steps` | no | `SNEK_MAX_STEPS`. **Absolute**, not "this many more" |
| `env` | no | any `SNEK_*` knob; wins over the runtime defaults. See `docs/running.md` |
| `priority` | no | lower runs first, **across the whole pool**: batches by their lowest spec, then name; within a batch by priority then id. Default 100. Set it deliberately: a new batch at 200 runs before one waiting at 240 on either box |
| `box` | no | `"desktop"` or `"laptop"`: only that box may claim it. **Absent means either**, which is the normal case |
| `label` | no | one line for `at_a_glance`. **ASCII only** — `--` not an em dash, `>=` not `≥`: `status.json` is read by a human and non-ASCII publishes as `\uXXXX` |
| `notes` | no | why this arm exists, for the reader of the spec. ASCII only |

**Omit `selector` and `episodes` on an eval.** Absent means `tools/closeout.py`'s own defaults, which
*are* the protocol. **An eval spec is claimed only by a box that holds every one of its policies'
checkpoints** (`savedPolicies/<arm>/`), and only once none of its arms is an unclaimed training spec.
A hand eval over a batch that was split across the boxes is therefore claimable by neither until the
missing checkpoints are rsynced to one of them (the `hof-remeasure` skill's step 2). **The pool line does
not say why**: it reads `b21 eval | b21-hof30k-confirm` on both boxes for as long as neither
can take it, so an eval spec sitting unclaimed past a wave boundary on a box that is otherwise idle means
the checkpoints -- `ls savedPolicies/` on each box is the check. Give it an id that is not a chain pass
label (`b7-hof30k-confirm`, not `b7-hof30k`).

A malformed spec is skipped and named under `attention`, never raised into either scheduler — so a bad
commit cannot stop a box, but it also will not tell you until after the push. **Validate first with the
daemon's own parser**, which is stdlib-only and runs here:

```
cd /Users/tony_wang/Projects/Snek/snek3/desktop && python3 -c "
import sys, glob; sys.path.insert(0, '.')
from daemon.job import parse_job, JobError
for p in sorted(glob.glob('$OPS/snek3/desktop/queue/pending/*.json')):
    try:
        j = parse_job(open(p).read(), source=p)
        print('OK  ', p, '|', j.box or 'either', '|', j.label)
    except JobError as e: print('BAD ', e)
"
```

## 1c. Name the batch's reference cell for the chart viewer

Every one-knob batch is read against an earlier cell, and the GitHub-Pages viewer shows that cell at the
end of the batch's own arms, marked with a gold edge, **only if it is listed**. Add the batch to
`snek3/viewer/references.json` (arm names plus a one-line label saying what the cell is and which knob
value it holds) in the same change as the specs. A batch with no entry shows only itself.

## 2. Push, trigger, and make sure a laptop scheduler is up

```
git -C "$OPS" add snek3/desktop/queue/pending/ \
  && git -C "$OPS" commit -m 'queue <batch>' \
  && git -C "$OPS" push origin ops
ssh the-claw-den 'Snek/snek3/desktop/trigger'          # the desktop looks now rather than within ten minutes
```

**The desktop's daemon starts its scheduler whenever `ops` or `claims` moves.** The laptop has no daemon:
its scheduler claims work while it is up and exits when the pool has nothing for it, so **the laptop takes
part exactly while its scheduler runs.** If the laptop should help with this batch, start one if none is
up (the `laptop-run` skill, "The laptop's scheduler"):

```
cd /Users/tony_wang/Projects/Snek/snek3
ps -Ao pid=,command= | grep '[t]ools.scheduler' | grep -v 'zsh -c'    # one up already? then you are done
PYTHONPATH=. nohup /opt/miniconda3/envs/snek3/bin/python -u -m tools.scheduler --shared --queue logs/laptop-queue/ \
    > logs/laptop-queue.log 2>&1 &
```

## 3. Confirm it started

```
git fetch origin ops-status && git show origin/ops-status:status.json
```

**The fetch is mandatory** — without it you read an old local ref whose embedded timestamp looks like a
dead daemon. `at_a_glance.pool` is the shared queue: what is unclaimed, by batch, and what each box holds
(`b18 training | 16/24 arms`; `laptop holds b21-w2 (8 arms)`); `desktop_running`/`desktop_queued` are the desktop's own, `laptop_running`/
`laptop_queued` the laptop's, `attention` anything needing a human. A batch pushed while both boxes are
busy shows under `pool` until a wave boundary on one of them: that is normal, not stuck. Nothing needs
restarting. `PYTHONPATH=. python -m tools.claims show` prints the same pool from the laptop.

## Pin, unpin, release

- **Pin a batch to one box** before it is claimed: set `"box"` on its specs on `ops` and push. A wave already
  claimed stays where it is (its checkpoints are there); to move it, release it (below) and let the other
  box claim it, and the arms retrain from scratch.
- **A stranded claim** — a box that claimed a wave and went quiet (a laptop lid, a killed scheduler) — shows
  under `attention` once that box's status is two hours old. **No automatic expiry**, by decision: an expiry
  that fires while a box is merely slow is two boxes training the same arms. If the box is not coming back,
  stop its arms (`stop-run`) and return the work to the pool:

  ```
  PYTHONPATH=. python -m tools.claims release b21-w3      # or an eval spec's id
  ```

  The arms go back unclaimed; the next claimer numbers a fresh wave (`w4`; a released number is never reused).

## Retune or hold the desktop

`config/runtime.json` on `ops`, re-read every network cycle (600 s; `trigger` applies it now). Same
worktree, same push. Keys: `max_trainers` 8, `eval_shards` 16, `poll_seconds` 30, `git_seconds` 600,
`torch_threads` 1, `omp_num_threads` 1, `nice` 0, `disk_min_gb` 5, `paused`, `drain`, `auto_stage_b`,
`viewer`. **A malformed or unknown-key file is rejected whole and the last known-good config kept**, and
it says so in `status.json`.

`paused` / `drain`: the desktop finishes what is running and claims nothing new. Set one before killing a
desktop job (`stop-run`) or the freed slot refills within one poll. On the laptop the hold is
`touch runs/.live/.paused`, and its scheduler claims nothing while the file exists.

**A pause holds a chain pass *after* its skip checks, so a `.failed-<label>` marker written during the pause
does not stop that pass** (2026-09-08: b27 wave 2's hof30k was marked skipped and moved to the desktop while
the laptop sat paused on it; removing the pause launched it anyway, and the scheduler relaunched it once when
it was killed). The scheduler checks `pass_done` and the failed marker, *then* waits out the pause. To move
a pass another box will run: write the marker, then stop the laptop's scheduler (SIGTERM; it leaves a running
pass behind as an orphan, so kill that close-out and its `pgrep -P` children by pid too) rather than unpausing
it, and start a fresh scheduler when the laptop should work again.
