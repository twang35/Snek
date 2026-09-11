---
name: progress-update
description: Report what is training or being measured on both boxes, then bring snek3's docs up to date. Use for "progress update", "status", "how are the arms doing", "what is running". Read-only with respect to processes — it never kills, stops or restarts anything.
model: sonnet
---

# Progress update

Run from `snek3/`. **Read-only with respect to running processes.** Do not kill, stop or restart any
arm — not even one past its cap, finished-looking, or clearly failing. Deciding a run is done is the
user's call. If a slot looks free, say so as a recommendation.

**The busy work is one command; the reading is yours.** `tools/progress_update.py` syncs, publishes,
computes every table and regenerates the mechanical parts of the docs, then prints a digest. Do not
redo any of that by hand — if a number you need is missing from the digest, add it to the tool.

## 1. Run the tool, and the laptop check

```
PYTHONPATH=. /opt/miniconda3/envs/snek3/bin/python -m tools.progress_update
ps -Ao pid=,etime=,command= | grep -E '[t]rain\.py|[c]loseout|[t]ools\.shard'   # laptop cross-check; status.json's laptop_running / laptop_queued (from the queue driver, as of laptop_iso) is the first read
```

The tool, in order: fetches both results feeds, both statuses, `ops` and `claims`; imports every finished
job's files that `runs/` lacks from either feed (an arm at its cap, a pass's merged files); `rsync`s the
desktop's live batches' charts into
`runs/` and their live `_evals.json` and `_checkpoint_evals.json` into the gitignored `runs/.live/desktop/`,
which the tables read only for an arm with no close-out file yet (off-LAN this fails and the digest says
so — carry on, and say so in the summary); regenerates the `charts.md` sections it owns; inserts a
`results.md` skeleton for a batch that just closed; prints the digest. **The site is not built here**:
the desktop rebuilds the `site` branch from both boxes' results feeds every network cycle, and
`ssh the-claw-den 'Snek/snek3/desktop/trigger'` rebuilds it now. Read the digest top to bottom:

| digest line | what it is |
|---|---|
| `sync:` | what moved. A nonzero "imported" means an arm or a wave finished since the last update |
| `desktop <iso>:` / `laptop <iso>:` | both boxes' `at_a_glance` lines from a **fresh** fetch, each running desktop job's step and %, then `pool:` -- the shared queue's unclaimed work and each box's holdings |
| `=== bN:` … `In flight`/`Closed` | the batch wave by wave across both boxes (`tools/batch_state.py`: which box holds each wave, trained, measuring, unclaimed), with an ETA from its own wave cadence while it trains |
| the table | the canonical per-batch table — knob value from the spec, rows, density, per-seed share, `hof5000` candidates, best row, best30, sef, drawdown, stage-A ≥98%, and the **reference cell's row in bold at its knob value** |
| `top rows:` | the five best stage-B rows in the batch |
| `prediction for <value>:` | what the spec said would happen, for every cell with rows — read each against its row |
| `results.md: skeleton inserted` | a batch closed: its reading is now owed (step 3) |

Every generated table shares its definitions with the Pages viewer (`tools/viewer_manifest.py`), so
quote the digest's numbers rather than recomputing them.

## 2. Commit a closed batch's files; the site takes care of itself

```
ssh the-claw-den 'Snek/snek3/desktop/trigger'      # the box rebuilds the site now, both boxes' latest included
cd .. && git add snek3/docs/charts.md snek3/runs/b<closed>* && git commit -m 'Runs: b<n> closed' && git push origin master && cd snek3
```

A **closed** batch's `runs/` files — its `_evals.json`, `_checkpoint_evals*.json`, `.md` and `.png` — go
to master as the archive once the tool has imported them. **Never add a live batch's `*_evals.json` or
`*_checkpoint_evals.*`**: the trainer rewrites the first on resume and the second is a pass in progress.
A live batch's pictures are on the site, not on master; `charts.md` links them there.

## 3. Write the readings — the only hand-written part

Every `charts.md` section the tool owns sits between `<!-- progress_update: batch bN -->` markers and
carries one `<!-- reading -->` … `<!-- /reading -->` block. **Edit only inside that block**; everything
outside it is rewritten on the next run. A `results.md` skeleton has the same block with a placeholder
sentence in it.

What a reading is: the paragraph a person needs *after* seeing the table — which group to look at
first and against which reference, where the sweep turns, which prediction the row confirmed or
falsified, what the stage-A traces show that the table cannot (drawdowns, late onset, a seed that never
arrived). Not a restatement of the numbers. Three to eight sentences.

| file | takes |
|---|---|
| `docs/runs.md` | a closed batch's **Learned** paragraph (≤100 words: the result, what surprised, what it changed) and its one-line result in `At a glance`; a newly queued batch's entry (base, varies, cells × seeds, **Why**). Update `## Open` if the plan moved. **No state, box, ETA or "superseded" blocks** — that is `status.json` |
| `docs/results.md` | a closed batch's reading, in the skeleton's block; the verdict on each cell |
| `docs/findings.md` | a conclusion, established or falsified, directly under `## Established` |
| `docs/charts.md` | the reading block of every live or just-closed batch |

**Write new material at the top, never at the bottom.** Reference sections (`Imported policies`,
`Reading this table`) stay at the bottom. Markdown traps: `A.`/`a.` are not list markers (use
`#### A. Thing`), and duplicate numbers in one list renumber silently.

Then the second commit — docs only, standing authorization:

```
cd .. && git add snek3/docs && git commit -m 'Progress update: <headline>' && git push origin master && cd snek3
```

## 4. The summary to the user

Both boxes by name, what closed and what is live with an ETA, the one or two rows that matter, and
whether anything undercuts the plan. The tables are on the page and in the docs — link, do not paste.

## When the tool is wrong

A skill that fails is a bug in the skill, and here the skill is mostly the tool. A switches batch with no single
knob (b19) gets `"knob": "name"` in `viewer/references.json`, and each arm's name token is its cell; an anneal
cell that shares its start value with a fixed cell is separated by its `_FINAL` key automatically. A reference row at the wrong place needs `value` set in `viewer/references.json`. A
hand-written `charts.md` section that should become generated: `--adopt bN`, which moves every prose
paragraph into the reading block and drops the rest. `--no-sync` works offline; `--no-docs` prints
tables without touching a file. Tests: `tests/test_progress_update.py`.
