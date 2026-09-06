---
name: hof-promote
description: Promote a snek3 checkpoint into hallOfFame/, on a confirmed fresh measurement. Use for "put that checkpoint in the HOF", "promote this one", "add a hall of fame entry", "is this a record", "save the best checkpoint".
---

# Promote a checkpoint to `hallOfFame/`

`hallOfFame/` is outside checkpoint rotation, so an entry survives whatever happens to
`savedPolicies/`. snek2 lost a 17.0% peak permanently for want of this. The bar is not "it scored
well once" — it is **a number that held up on episodes nobody selected on**.

**A promotion is five things, and the folder has been left with three of them** (2026-09-06, `b17cl`: checkpoint,
entries row and prose landed; the recording and its row did not, because this skill had no step for them):

| piece | where | step |
|---|---|---|
| the confirmed /30,000 number on seed 7 | `runs/<arm>_checkpoint_evals_hof30k.json`, or a fresh `evaluate.py ... one` | 0 |
| the checkpoint + `arch.json`, loaded from the copy | `hallOfFame/<arm>-ckpt<step>/` | 1-3 |
| the row and the prose | `hallOfFame/HOF.md`: entries table, "All admitted" line, ordering paragraph, a section | 4 |
| the recording and its row | `hallOfFame/gifs/<entry>.gif`, the recordings table | 5 |
| the check | `tests/test_hof_entries.py` fails on any entry missing a piece | 6 |

## 0. Confirm on a fresh seed. Not optional, and the seed is the whole point

**First look for the measurement that already exists.** Every batch's chain ends in an `hof30k` pass — 30,000
episodes on seed 7 over the rows that read ≥99 /5,000 — so a candidate handed over by a progress update usually
has its confirmed number on disk already:

```
python3 -c "
import json; d=json.load(open('runs/<arm>_checkpoint_evals_hof30k.json'))
print('seed', d['seed'], 'episodes', d['episodes'])
for r in sorted(d['rows'], key=lambda r: r['step']): print(r['step'], r['perfect_percent'], r['perfect_ci95'], r['perfect_games'])
"
```

`seed 7`, `episodes 30000`, and a row at the step: that row is the confirmed number, its neighbours in the same
list are the basin (below), and there is nothing to re-run — go to step 1. No file, or no row at the step (the
checkpoint never read ≥99 /5,000, or the pass has not run yet — `tools.batch_state <batch>` says whether the
wave's `hof30k` is owed): run the measurement below, or the `hof-remeasure` skill for a whole batch.

**Every candidate you are handed is the maximum of a selection, so it is biased upward and it will
fall.** Measured here:

| | selected | fresh | drop |
|---|---|---|---|
| snek2's top four HOF entries | 98.0-99.0 /500 | 95.9-97.5 /1000 | mean **−1.45 pp** |
| `b5h` @9027584 (max of 2,172 rows) | 99.20 /5000 | 98.8 /2000 | **−0.4 pp** |

```
PYTHONPATH=. python -u evaluate.py <policy> one --step <step> \
  --episodes 30000 --seed 7 --width 1000
```

- **‡ Use a seed the selecting pass did not use.** Close-outs and `hof5000`-style re-measures run
  `--seed 0`. Re-running at seed 0 replays *the same episodes* and confirms nothing — it will
  reproduce the inflated number to the decimal and look like corroboration.
- **30,000 episodes at `--width 1000` takes ~5 min** and gives a ±0.12 pp interval. `one` is
  single-process (sharding splits by checkpoint, not by episodes), so run several candidates as
  parallel processes rather than reaching for `--shards`.
- **A desktop `hof30k` wave is this step done in bulk** — an eval spec with `"episodes": 30000`,
  `"eval_args": ["--label", "hof30k", "--seed", "7"]` and `above:99:hof5000`, as b9's was. Same episodes, same seed, same eval path; its rows are confirmed rates and go straight
  to step 1.
- **The number that goes in `HOF.md` is this one.** Never a stage-A /100, never a graph point, never
  the close-out /500 the candidate was picked by.

### ‡ If the claim is "a record", re-measure the incumbent at the same depth too

**Never rank a fresh deep number against a published shallower one — the incumbent is a selected high
as well.** Measured: the snek2 champion's published 98.73% /3,000 read **98.48% /30,000**
on the same afternoon, same seed, same eval path. Against the published figure the new checkpoint
looked tied (p = 0.26); against the matched re-measure it wins by +0.47 pp (p < 1e-6). Same weights
both times. A cross-depth comparison would have recorded the opposite conclusion.

Re-measuring the incumbent costs one more `one` run. Do it before writing "record" anywhere.

### Choosing between near-tied candidates: take the one in the better basin

A validated tie-break — the mean score of a candidate's **neighbours within ±1M
transitions** predicts its true deep rate *better than its own selected score does* (r = +0.46
against +0.25 on 181 rows). Two `b5h` rows both read 99.20 /5000; their basins did not:

| step | own | neighbours | basin mean | ≥98.5 in basin |
|---|---|---|---|---|
| 6,782,976 | 99.20 | 10 | 98.14 | 4 |
| **9,027,584** | 99.20 | **28** | **98.54** | **17** |

An isolated spike is noise wearing a good number. Promote the region, not the pixel.

### Not a trap, and do not chase it

The `describe()` line differs between the run and your re-measure — `dist 0.001, chase_safe c=0.0
gate=85` instead of the arm's own shaping. **Shaping is reward-side: it changes `avg_reward` and
nothing else.** Verified on the same checkpoint and seed: 1975/2000 both ways, identical avg score,
reward 192.09 against 192.39. `vectorized/config.describe`'s own docstring says so.

## 1. Get the checkpoint, which may not be on this box

A desktop-trained arm's checkpoints exist **only on `the-claw-den`** — `savedPolicies/` is not on the
git bus. Check `ls -d savedPolicies/<policy>` first, and note the box may still hold the arm under its
**pre-rename name** (`p2h-ep8-seed8` there, `b5h-ep8-seed8` here).

```
mkdir -p savedPolicies/hofstage-<short>
rsync -av --files-from=- the-claw-den:Snek/snek3/savedPolicies/<remote-policy>/ \
  savedPolicies/hofstage-<short>/ <<'EOF'
ckpt-<step>.pt
arch.json
EOF
```

**`--files-from` because multi-source rsync over ssh does not take a quoted list** — neither
`'host:a b'` nor `'host:{a,b}'` works; both fail as one absurd path and cost two attempts.

## 2. Copy it in

```
mkdir -p hallOfFame/<policy>-ckpt<step>
cp <source-dir>/ckpt-<step>.pt <source-dir>/arch.json hallOfFame/<policy>-ckpt<step>/
```

Name the directory for the **arm's current name**, not the box's stale one. **`arch.json` is
required** — without it the copy will not load at all, which is deliberate: width and observation era
cannot be guessed from weights (`tools/arch.py`).

## 3. Verify the *copy*, not the original

A wrong or missing `arch.json` reads as a beginner rather than as an error, so load the copy:

```
PYTHONPATH=. python -u evaluate.py hallOfFame/<policy>-ckpt<step> one \
  --step <step> --episodes 500 --seed 11
rm -rf savedPolicies/hofstage-<short>
```

**No staging needed** — `tools/restore.policy_dir` takes any directory holding an `arch.json` beside
the checkpoint, so a `hallOfFame/` path is addressable by `evaluate.py`, `watch.py` and
`record_gif.py` directly. (snek2 staged under a throwaway `savedPolicies/` name because its loader
could not; do not carry that step across.) It must read like a champion.

Then delete the step-1 staging directory. A partial `savedPolicies/` arm left behind looks like a real
one to every other tool.

## 4. Write `HOF.md` — four places, not one

| place | what |
|---|---|
| the entries table | a row in rate order: entry name, algo / net / the knob that made it, **confirmed** rate with the perfect-game count, CI, the /5000 it was selected at, the drop. Bold the record only |
| the "All admitted on 30,000 fresh episodes" line under `## The entries` | add the entry and its date |
| the ordering paragraph below the table | where it sits against its neighbours, with z and p (two-proportion z on the perfect-game counts) — "not distinguishable" when p > 0.05, and say so |
| its own `##` section, newest first, above the record's | what batch and knob it came from, how the protocol found it (screened / re-measured / confirmed counts), the basin (its `hof30k` neighbours within ±1M and their mean; its `hof5000` neighbours' mean against the other entries' basins), the batch's 5,000 → 30,000 drop, whether it is a record — and the step-3 verification count |

State what it was selected out of; that is what lets a later session judge the number. **If it is a record**, the
section takes the `‡ The record:` title, the old record's section is retitled `‡ The previous record:`, and the
recordings table's "— the record" tag moves; `record_gif.py`'s `HOF_RECORD` does **not** move (user's decision,
2026-09-03).

## 5. Record it, and add the recording's row

Every entry has three complete games in `hallOfFame/gifs/`, at the folder's settings so the eras' folders read alike:

```
PYTHONPATH=. python -u record_gif.py hallOfFame/<entry> --tile 20 --colors 32 --out hallOfFame/gifs/<entry>.gif
```

~1 s to capture, a 2-5 MB file, deterministic (greedy policy, seeded food). The printout gives the three game
lengths; put them in the row. Then add the row to the **recordings table** in `HOF.md`, in the same order as the
entries table: `![short](gifs/<entry>.gif)<br>**\`<arm>\`** @<step><br>**<rate>% /30,000**`, and in the second
column what to watch for *against the other recordings* (game length, route shape) — never a claim about the rate,
which three games cannot support. Bump the section's size total.

## 6. Check, then push

```
PYTHONPATH=. python -m pytest -q tests/test_hof_entries.py
```

`test_every_real_entry_has_its_recording_and_both_of_its_rows_in_hof_md` reads the real folder: every entry
directory must have its gif and its two rows, or it names what is missing. Then commit `hallOfFame/` (the entry,
the gif, `HOF.md`) and the doc lines that called it a candidate (`docs/runs.md`'s `Now`, the batch's readings in
`docs/results.md` and `docs/charts.md`) in one commit.

`hallOfFame/` is committed output and `HOF.md` is documentation, so both go up without waiting —
**unless the same commit touches code**, which sends the whole thing back to needing approval.

## What does *not* go in

- A number that only exists at the selection depth.
- A second checkpoint from the same arm a few hundred thousand transitions from one already in, unless
  it is measurably different. It implies a ranking the data cannot support — snek2 declined to promote
  four statistically indistinguishable sweep candidates for exactly this reason.
- Anything you have not loaded from the copy in step 3.
