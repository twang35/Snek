---
name: hof-promote
description: Promote a snek3 checkpoint into hallOfFame/, on a confirmed fresh measurement. Use for "put that checkpoint in the HOF", "promote this one", "add a hall of fame entry", "is this a record", "save the best checkpoint".
---

# Promote a checkpoint to `hallOfFame/`

`hallOfFame/` is outside checkpoint rotation, so an entry survives whatever happens to
`savedPolicies/`. snek2 lost a 17.0% peak permanently for want of this. The bar is not "it scored
well once" — it is **a number that held up on episodes nobody selected on**.

## 0. Confirm on a fresh seed. Not optional, and the seed is the whole point

**Every candidate you are handed is the maximum of a selection, so it is biased upward and it will
fall.** Measured here:

| | selected | fresh | drop |
|---|---|---|---|
| snek2's top four HOF entries (2026-08-20) | 98.0-99.0 /500 | 95.9-97.5 /1000 | mean **−1.45 pp** |
| `b5h` @9027584 (2026-09-01, max of 2,172 rows) | 99.20 /5000 | 98.8 /2000 | **−0.4 pp** |

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
  `"eval_args": ["--label", "hof30k", "--seed", "7"]` and `above:99:hof5000`, as b9's was on
  2026-09-03. Same episodes, same seed, same eval path; its rows are confirmed rates and go straight
  to step 1.
- **The number that goes in `HOF.md` is this one.** Never a stage-A /100, never a graph point, never
  the close-out /500 the candidate was picked by.

### ‡ If the claim is "a record", re-measure the incumbent at the same depth too

**Never rank a fresh deep number against a published shallower one — the incumbent is a selected high
as well.** Measured 2026-09-01: the snek2 champion's published 98.73% /3,000 read **98.48% /30,000**
on the same afternoon, same seed, same eval path. Against the published figure the new checkpoint
looked tied (p = 0.26); against the matched re-measure it wins by +0.47 pp (p < 1e-6). Same weights
both times. A cross-depth comparison would have recorded the opposite conclusion.

Re-measuring the incumbent costs one more `one` run. Do it before writing "record" anywhere.

### Choosing between near-tied candidates: take the one in the better basin

A validated tie-break (2026-09-01) — the mean score of a candidate's **neighbours within ±1M
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

## 4. Write the row, then push

Add a row to `hallOfFame/HOF.md` carrying the **confirmed** rate, its CI, the episode count and the
seed. State what it was selected out of; that is what lets a later session judge the number.

`hallOfFame/` is committed output and `HOF.md` is documentation, so both go up without waiting —
**unless the same commit touches code**, which sends the whole thing back to needing approval.

## What does *not* go in

- A number that only exists at the selection depth.
- A second checkpoint from the same arm a few hundred thousand transitions from one already in, unless
  it is measurably different. It implies a ranking the data cannot support — snek2 declined to promote
  four statistically indistinguishable sweep candidates for exactly this reason.
- Anything you have not loaded from the copy in step 3.
