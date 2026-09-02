---
name: progress-update
description: Report what is training or being measured on both boxes, then bring snek3's docs up to date. Use for "progress update", "status", "how are the arms doing", "what is running". Read-only with respect to processes — it never kills, stops or restarts anything.
---

# Progress update

Run from `snek3/`. **Read-only with respect to running processes.** Do not kill, stop or restart any
arm — not even one past its cap, finished-looking, or clearly failing. Deciding a run is done is the
user's call. If a slot looks free, say so as a recommendation.

## 1. Both boxes, and name which is which

"N arms running" is meaningless without the box. Neither check sees the other.

```
ps -Ao pid=,etime=,command= | grep -E '[t]rain\.py|[c]loseout|[t]ools\.shard'   # laptop
git fetch origin ops-status && git show origin/ops-status:status.json          # desktop
```

**`git fetch` is mandatory.** `git show origin/ops-status:…` reads a local tracking ref, so without
the fetch you get an arbitrarily old snapshot with a timestamp in it that reads like a dead daemon.
That has caused three false alarms. Before calling the box down, in order: fetch and re-read;
`ssh the-claw-den 'Snek/snek3/desktop/trigger'`; `ssh the-claw-den -o ConnectTimeout=8 -o BatchMode=yes`.
A `status.json` up to 10 minutes old is healthy (`git_seconds` is 600).

On the home LAN, `trigger` is the better first read: it makes the daemon publish *now* and prints
`counts`, the running batches and `attention` straight back, so it is one round trip instead of
fetch-plus-parse. Off-LAN it exits 1 and the git bus is the only route.

Start from `status.json`'s `at_a_glance` and its `attention` list. Ledger `interrupted` means the box
rebooted under the job; it is non-terminal and gets relaunched.

## 2. Read each arm from its summary block, never from a log

`runs/<policy>_evals.json` → `summary`: `step`, `transitions`, `evals`, `trailing_now`,
`peak_trailing`, `best_perfect30`, `strong_eval_fraction`, `recent_perfect30`, `max_single_eval`,
`zero_since`, `dead_since`, `epsilon`.

- **`strong_eval_fraction` is the primary metric** (share of evals at ≥80% perfect). It is a fraction
  of each arm's own evals, so compare only at a common step horizon.
- **`zero_since` answers "is it dead now"**; `dead_since` is history. Neither is a verdict — a snek2
  arm came back from 1.2M steps near zero.
- **Rank a comparison: peak `best_perfect30` > consistency > speed.** An arm still rising at its cap
  has not reported its best30 yet.
- **Read `transitions`, not `step`,** across any change in `SNEK_COLLECT_ENVS`, `SNEK_FORK_BRANCHES`
  or algorithm. A DQN counted step is four game moves at the default `fork_branches=4`.

## 3. Update the docs in the same pass

| file | takes |
|---|---|
| `docs/runs.md` | current state and forward plan **only** |
| `docs/results.md` | every arm: config, final numbers, verdict |
| `docs/findings.md` | conclusions — established and falsified |
| `docs/charts.md` | **always refresh, finished or not.** A running batch with no entry is a bug |

`charts.md` links `../runs/<policy>.png` directly — no copy step. Keep the split clean; snek2's
equivalent grew to 950 lines of interleaved status and stopped being usable.

**Write new material at the top, never appended to the bottom.** A batch that closed goes above the
batch before it in `results.md` and `charts.md`; a new finding goes directly under `## Established`
in `findings.md`; `runs.md` keeps the order current state -> forward plan -> history, so a stale
`Now` block is moved down rather than left in place above the new one. Reference sections
(`Imported policies`, `Reading this table`) stay at the bottom.

Markdown traps: `A.`/`a.` are not list markers (use `#### A. Thing`), and duplicate numbers in one
list renumber silently. Cross-reference items by name.

## 4. Commit

A **docs-only** change (Markdown, plus any chart PNGs riding along) is committed and pushed with no
confirmation — standing authorization. The moment the same change touches code or config, the whole
thing waits for the user.

**Never commit a live desktop arm's `runs/<policy>.{md,png}` or `runs/<policy>_evals.json`.** The box
rewrites those paths every eval, so a committed copy makes its `git merge --ff-only` abort and blocks
every deploy for hours. Desktop artifacts arrive on the `results` branch at close-out.

**"Live" includes a batch whose stage B is still running**, and it bit on 2026-09-01: b8's charts
commit swept `runs/b8*` in wholesale while the box was mid-pass, and the next deploy aborted on 64
files, 8 of them changing under it. So `git add` a desktop batch's charts **by name**:

```
git add snek3/runs/b<n>*.png snek3/runs/b<n>*.md      # never the _evals.json or _checkpoint_evals.*
```

and add the `*_evals.json` / `*_checkpoint_evals.*` only once the batch's stage B reads `done` on the
ledger. If it has already happened, the `desktop-deploy` skill says how to sort the collision out.
