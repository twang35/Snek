---
name: move-batch
description: Move a queued snek3 batch between the desktop box the-claw-den and the laptop, in either direction. Use for "move b19 to the laptop", "dequeue that from the desktop", "run this one here instead", "put b18 back on the box", "rebalance the two boxes".
---

# Move a batch between the boxes

Both boxes run the same `queue/pending/*.json` specs: the desktop's daemon reads them from the `ops`
branch, the laptop's `tools.scheduler --queue` reads them from `snek3/logs/laptop-queue/<batch>/`.
Moving a batch is moving its spec files from one place to the other. **Move whole batches that have
not started.** A batch with a wave trained already has checkpoints on the box it started on, and
neither side can resume the other's; let it finish where it is (`stop-run` if it must not).

The desktop side is a push to `ops`, which starts or stops real work on another machine: **the user
has asked for *this* move** before any of the below runs. Say which batches and which direction back
to them.

## Desktop -> laptop (as b13, b14, b16, b19, b20 were, 2026-09-03/04)

Run from `/Users/tony_wang/Projects/Snek`. The commit that dequeues is where the specs will be read
from later, so copy them out **before** removing them, from the ref they are still on:

```
git fetch origin ops ops-status
B=b19   # one batch per run of this block; repeat for the next
Q=/Users/tony_wang/Projects/Snek/snek3/logs/laptop-queue
mkdir -p $Q/$B && for f in $(git ls-tree --name-only origin/ops snek3/desktop/queue/pending/ | grep "/${B}[a-z]"); do
    git show "origin/ops:$f" > $Q/$B/$(basename $f); done
ls $Q/$B | wc -l          # must equal the batch's arm count; 0 means the grep or the cwd is wrong (see below)
```

`git ls-tree` paths are **relative to the shell's cwd inside the repo**, so run it from the repo root,
not `snek3/`. From `snek3/` the path resolves to `snek3/snek3/...`, matches nothing, and the loop
writes nothing — 2026-09-04, caught by the count.

Then remove them from `ops` in its worktree (the `desktop-batch` skill's worktree block finds or makes
it; **globs there need the worktree's absolute path**, since a relative glob expands against the cwd):

```
OPS=$(git worktree list --porcelain | awk '/^worktree /{p=substr($0,10)} /^branch refs\/heads\/ops$/{print p}')
git -C "$OPS" merge --ff-only origin/ops
git -C "$OPS" rm -q $OPS/snek3/desktop/queue/pending/${B}[a-z]*.json
git -C "$OPS" commit -m "dequeue $B (<n> arms): moved to the laptop queue driver" && git -C "$OPS" push origin ops
ssh the-claw-den 'Snek/snek3/desktop/trigger'      # prints the queue as the daemon now sees it; the batch must be gone
```

Only the training specs exist on `ops`; the batch's stage B and hof passes are queued by the chain
when a training finishes, so they vanish with the specs and nothing else needs removing.

Then make sure a laptop driver is up, and start one if not — the `laptop-run` skill, "Queueing
batches here". A running driver rescans `logs/laptop-queue/` between batches and picks the new
directory up in name order; nothing else to do, and the batch shows under `laptop_queued` in
`status.json` once the driver next publishes (its next event, or ten minutes). It runs the batch as the daemon would: waves of 8,
stage B, hof5000, hof30k after each wave.

## Laptop -> desktop

Only if the batch's directory has not been started: `grep "starting $B" snek3/logs/laptop-queue.log`
prints nothing. Move the directory out of the queue (the driver rescans, so it is simply not there
next time) and put the specs back on `ops`; they keep their own `priority`, which is what orders the
desktop's queue, so a batch goes back to its place in the run order rather than to the end:

```
mv $Q/$B /Users/tony_wang/Projects/Snek/snek3/logs/laptop-queue-moved-$B
cp /Users/tony_wang/Projects/Snek/snek3/logs/laptop-queue-moved-$B/*.json $OPS/snek3/desktop/queue/pending/
git -C "$OPS" add snek3/desktop/queue/pending/ && git -C "$OPS" commit -m "requeue $B (<n> arms): moved back from the laptop" && git -C "$OPS" push origin ops
ssh the-claw-den 'Snek/snek3/desktop/trigger'
```

Validate the specs with the daemon's parser first if anything in them was edited (`desktop-batch`,
section 1). The batch's chart-viewer reference entry is per batch, not per box, and stays.

## Afterwards

- Update the state column of section 0 in `plans/hyperparam-sweep.md` (which box, ETA) and the `Now`
  table in `docs/runs.md` if a progress update is not about to.
- Laptop cadence for the ETA: ~2.3 h to train a base-config wave, ~1 h stage B, 20-45 min of hof
  passes, **all in series** — call it ~3.5 h per 8-arm wave. Desktop: ~2.6 h per wave (stage B runs
  beside the next wave), plus ~0.3 h per wave of hof passes that hold training back.
