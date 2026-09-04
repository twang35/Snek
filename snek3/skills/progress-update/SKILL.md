---
name: progress-update
description: Report what is training or being measured on both boxes, then bring snek3's docs up to date. Use for "progress update", "status", "how are the arms doing", "what is running". Read-only with respect to processes — it never kills, stops or restarts anything.
model: sonnet
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

## 1b. Refresh the charts and publish the site **before** any reading or writing

The GitHub-Pages viewer is served from `master`'s top-level `docs/`, and Pages goes live about a
minute after a push — so publish first, and by the time the update's summary is written the site
already shows what it describes. In order, before opening a single results file:

```
git fetch origin results                                        # closed waves' artifacts
# import any batch/wave whose stage B reads `done` on the ledger and is not yet in snek3/runs/:
#   for f in $(git ls-tree -r --name-only origin/results | grep 'results/<job-id>/'); do git show origin/results:$f > snek3/runs/$(basename $f); done
rsync -a --include='b<n>*.png' --include='b<n>*.md' --exclude='*' the-claw-den:Snek/snek3/runs/ snek3/runs/   # live arms' charts
cd snek3 && PYTHONPATH=. /opt/miniconda3/envs/snek3/bin/python -m tools.publish_pages && cd ..           # manifest + docs/
git add docs snek3/viewer/manifest.js snek3/runs/b<n>*.png snek3/runs/b<n>*.md                          # plus the imported closed-wave JSON
git commit -m 'Charts: <what changed>' && git push origin master
```

`tools.publish_pages` rewrites `docs/` completely — the page, a manifest pointing at `charts/`, exactly
the PNGs the manifest refers to — so `docs/` is **never edited by hand**. `rsync` is home-LAN only; off-LAN,
publish without the live charts and say so. The rest of the update's doc edits go up in a second commit.

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

**Every arm's charts, live desktop arms included, were committed in step 1b** (rule changed 2026-09-02).
If step 1b was skipped, do it now rather than adding pictures here by hand — the published site and the
committed manifest must come from the same `publish_pages` run.

**Never commit a live desktop arm's `*_evals.json` or `*_checkpoint_evals.*`.** `_evals.json` is the
trainer's own history, read on resume; the stage-B file is a pass in progress. Add them only once the
batch's stage B reads `done` on the ledger, when they have arrived on the `results` branch. The box's
`desktop/deploy` keeps its own copy of every colliding picture but refuses to merge over a differing
JSON, so committing one blocks every deploy until it is `git rm --cached` on master.
