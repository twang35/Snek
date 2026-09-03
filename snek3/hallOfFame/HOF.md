# Hall of Fame — snek3

The best policies the PyTorch era has produced, preserved as standalone checkpoints so they survive
whatever happens to `savedPolicies/`.

**Why this folder exists.** A training keeps a bounded number of checkpoints, so a long run
eventually deletes its own best one. That has already cost real evidence in snek2 — `b5c-schlongIS`'s
17.0% peak became permanently unmeasurable. Copies here are outside any rotation and are deleted by
nothing.

## The entries

All admitted on 30,000 fresh episodes at **seed 7** — a seed the selecting pass never used; `b5h` and
`b6b` on 2026-09-01, `b9ch` on 2026-09-03. Add one with the [`hof-promote`](../skills/hof-promote/SKILL.md)
skill.

| entry | algo / net | confirmed **/30,000** | 95% CI | selected at | drop |
|---|---|---|---|---|---|
| **`b9ch-lam999-seed4-ckpt47251456`** | PPO, `fc 320`, 4 epochs, **λ 0.999** | **99.30%** (29790) | [99.2, 99.4] | 99.40 /5000 | −0.10 pp |
| `b5h-ep8-seed8-ckpt9027584` | PPO, `fc 320`, 8 epochs | **98.96%** (29687) | [98.84, 99.07] | 99.20 /5000 | −0.24 pp |
| `b6b-fc200x100-seed2-ckpt133120000` | PPO, `fc 200,100`, 4 epochs | **98.73%** (29619) | [98.60, 98.85] | 99.10 /5000 | −0.37 pp |

`b9ch` leads `b5h` by 0.34 pp (z = 4.5, p < 1e-5) and `b5h` leads `b6b` by 0.23 pp (z = 2.60,
p = 0.0094), so the ordering is real rather than a coin.

## ‡ The record: `b9ch` @47251456, 99.30% over 30,000 episodes — 2026-09-03

**The first entry above 99% at depth, and the first to come out of a hyperparameter sweep rather than
a seed batch.** b9 swept GAE λ off b7's `fc 320` / 4-epoch cell; λ 0.999 seed 4 is the arm. It was
found by the protocol working end to end: 1,258 checkpoints screened at 500 episodes, 80 of them
re-measured at 5,000 (`hof5000`), the 16 at ≥99 /5,000 re-measured at 30,000 on seed 7 (`hof30k`, on
the desktop). Its neighbours confirm it is a region and not a pixel: @47267840, @47235072, @47218688
and @47611904 all read **99.20** /30,000 and the arm's 27 `hof5000` neighbours within ±1M average
98.70, against 98.54 for `b5h`'s basin. The 5,000 → 30,000 drop across all 33 confirmed rows was
**−0.14 pp** (the smallest yet; `b5h` dropped −0.24 and snek2's entries −1.45), and 18 of the 33 beat
`b5h`'s 98.96 — six arms of the λ ≥ 0.99 plateau hold a checkpoint above the old record. Only the one
is promoted, per the rule below. The same wave measured `b9cl-lam100-seed4` @19316736 at 99.10 and
`b9cc-lam995-seed3` @18350080 at 99.10; they are not indistinguishable from `b9ch` (z ≈ 2.9) and are
not promoted, because a second λ-plateau entry adds nothing the table does not already say.

**`record_gif.py`'s `HOF_RECORD` names this entry** (commit of 2026-09-03), so `record_gif.py hof` records it and
`record_gif.py --list` marks it `*`.

## ‡ The previous record, and the first time this project beat snek2 on a matched measurement

**`b5h` @9027584 was the record at 98.96% over 30,000 episodes, 2026-09-01 to 2026-09-03.** The comparison that established it
was run the same day, same depth, same seed, same env and same eval path — which matters, because the
snek2 champion's published figure was itself measured shallower and is itself a selected high:

| policy | era | /30,000, seed 7 | 95% CI |
|---|---|---|---|
| **`b5h-ep8-seed8` @9027584** | snek3, PPO | **98.96%** | [98.84, 99.07] |
| `b6b-fc200x100-seed2` @133120000 | snek3, PPO | 98.73% | [98.60, 98.85] |
| `b44a-lowlr7-b29b` @2739000 | snek2, DQN — the champion | **98.48%** | [98.34, 98.62] |

- `b5h` beats the champion by **+0.47 pp, z = 5.16, p < 1e-6**. `b6b` beats it by +0.25 pp, p = 0.010.
- **The champion's own 98.73% / 3,000 fell to 98.48% / 30,000** — a −0.25 pp drop, and the reason
  cross-depth comparisons are not allowed to settle this question. Against the *published* 98.73% the
  same `b5h` measurement reads p = 0.26 and looks like a tie. Same weights, same afternoon; only the
  depth of the baseline differs.
- **`record_gif.py`'s `HOF_RECORD` names this entry**, so `record_gif.py hof` and
  `record_gif.py --list` (which marks it `*`) resolve without arguments. This file stays the
  authority; that constant is a shortcut, and `tests/test_hof_entries.py` fails if the two disagree.

## The admission rule

**Step 0 is a fresh, deep re-measure at an unused seed, and it is not optional.** Every candidate
reaches this folder as the maximum of a selection, so it is biased upward.

| | selected | fresh | drop |
|---|---|---|---|
| snek2's top four entries (2026-08-20) | 98.0-99.0 /500 | 95.9-97.5 /1000 | mean **−1.45 pp** |
| `b5h` first look (2026-09-01) | 99.20 /5000 | 98.8 /2000 | −0.4 pp |
| `b5h` and `b6b` as admitted | 99.20, 99.10 /5000 | 98.96, 98.73 /30000 | −0.24, −0.37 |
| `b9ch` as admitted (2026-09-03) | 99.40 /5000 | 99.30 /30000 | **−0.10** |

All three entries held up far better than snek2's did. That is the *result*, not the process working as
usual — and the likely reason is the next section.

## Why these two, out of 2,172 candidates

Both batches were re-measured whole at 5,000 episodes: **b5 873 rows, b6 1,299**. The picks were not
simply the top rows.

- **Take the basin, not the pixel.** Two `b5h` rows both read 99.20 /5000. Their neighbourhoods within
  ±1M transitions did not: 28 neighbours at mean 98.54 for @9027584 against 10 at 98.14 for
  @6782976. The neighbourhood mean predicts a row's true deep rate **better than its own selected
  score does** (r = +0.46 against +0.25, n = 181), so @9027584 was promoted and @6782976 was not.
- **`b5h`'s record region is early and it is wide** — 65 rows between 4M and 12M transitions averaging
  98.41 /5000, nine of them ≥99%. That is 1.6-4.7% of the arm's 255M budget: the best policy the whole
  49.5-core-hour b5 batch produced was found about ten minutes into a 6.4-hour run.
- **`b6b`'s came at 58% of its budget**, which is characteristic of its batch — b6's champion-level
  rows are back-loaded (16 late against 4 early, p = 0.041) where b5's are front-loaded (21 of 29 in
  the first decile). **That analysis is not yet written into `docs/`** — it exists only here and in the
  `hof5000` JSONs under `runs/`.

## What is not here

- Any number that exists only at the depth it was selected at.
- A second checkpoint from the same arm a few hundred thousand transitions from one already in.
  `b5h`'s basin holds ten rows at ≥99% /5000 and promoting them would imply a ranking the data cannot
  support — snek2 declined to promote four indistinguishable sweep candidates for the same reason.
- Recordings. `record_gif.py` writes to `snek3/gifs/`, not here; `hallOfFame/gifs/` is where a
  recording is copied to be embedded in this file, and it is still empty.

## Running an entry

```
PYTHONPATH=. python -u watch.py hallOfFame/<entry>
PYTHONPATH=. python -u evaluate.py hallOfFame/<entry> one --step <step> --episodes 500 --seed 11
PYTHONPATH=. python -u record_gif.py hallOfFame/<entry>
```

An entry is addressable directly — `tools/restore.policy_dir` takes any directory with an `arch.json`
beside the checkpoint, so nothing has to be staged under `savedPolicies/` to be read.

`arch.json` sits beside every checkpoint and is **required** — width and observation era cannot be
recovered from weights, so a copy without it does not load at all rather than loading wrongly
(`tools/arch.py`). Both entries here are observation era `b09c616`, 30 values, 3 actions.
