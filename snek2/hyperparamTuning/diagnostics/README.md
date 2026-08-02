# Observation-space diagnostics

The instruments behind [`../../claudeFeatureRecommendations.md`](../../claudeFeatureRecommendations.md),
kept because the simulator in `diag.py` is validated against the game and rebuilding that is the
slow part. **Frozen alongside that document** (2026-08-02) — they are preserved so results can be
reproduced or extended, not maintained. They read checkpoints and never write to
`savedPolicies/` or `runs/`, so they cannot disturb a training arm.

Everything runs from `snek2/`, with the scripts' own directory added to the path by each script:

```
cd snek2
PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u hyperparamTuning/diagnostics/<script> [args]
```

| script | args | what it answers |
|---|---|---|
| `simcheck.py` | — | does the simulator agree with `Snake.step()`? Random play, dies fast, early-game only |
| `simcheck2.py` | — | the same, driven by a food-seeking heuristic so it reaches length 87 |
| `diag.py` | `<policy> <step> <episodes> <seed> <out.json>` | failure attribution and observation aliasing |
| `diag2.py` | same | scores candidate observations at the losing decisions |
| `diag3.py` | same | three versions of the tail-reachability test |
| `diag4.py` | same | five versions, including the advanced-tail fix that won |
| `probe.py` | — | dumps boards where two versions of the test disagree, as ASCII grids |
| `analyze.py` | `"<glob of diag.py shards>"` | merges and reports the `diag.py` measurements |
| `score.py` | `"<glob of diag2.py shards>"` | ranks the candidates from `diag2.py` |
| `score4.py` | `"<glob of diag4.py shards>"` | scores the five variants from `diag4.py` |

## Run the validations first

Nothing else here is worth reading if the simulator disagrees with the game, because the failure
attribution replays every loss through it. Both scripts print a mismatch count that must be zero:

```
664 steps, 60 deaths, 60 death calls agreed, 0 mismatches
16173 steps, max length 87, outcomes {'perfect': 0, 'dead': 0, 'starved': 12}, mismatches 0
```

`simcheck2.py` exists because random play dies around step 11 and never reaches the crowded
states that matter. Note its outcome line: a tail-following, food-seeking heuristic starves in 12
of 12 games at a maximum length of 87, so it is a survival reference and not a solver.

The `diag*.py` scripts also assert on every step that their recomputation of `head_with_tail` and
`lg(num_groups)` matches the live observation vector, and report `observed_mismatch` in the
output. That must be zero too, and it will stop being zero the moment `group_obs` changes — which
is the point. Any of the recommended fixes will trip it, and the scripts then need updating to
distinguish "the fix is in" from "the recomputation is stale".

## Sharding

A single process is too slow for enough losses, so the measurements were six processes of 60
episodes with seeds 101-106, merged by the `analyze`/`score` scripts:

```
for i in 1 2 3 4 5 6; do
  PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u hyperparamTuning/diagnostics/diag4.py \
    b8f-disc9975seed2 3149000 60 $((100+i)) /tmp/d4_$i.json > /tmp/d4_$i.log 2>&1 &
done
```

That is 360 episodes, about 1.08M steps, and takes roughly 8-10 minutes on this machine. **The
same seed reproduces the same games**, because the policy is greedy and the only randomness is
food placement — so `diag.py` through `diag4.py` at seeds 101-106 measure an identical set of
games and their results can be compared decision by decision. That is what made the tail-test
comparison a paired one.

These are eval-style processes rather than trainers, so the four-trainer rule does not apply, but
they take six cores. Do not run a full shard set alongside a batch of arms.

## Numbers to reproduce against

From `b8f-disc9975seed2` at checkpoint `3149000`, 360 episodes, 1,076,492 steps:

| measurement | value |
|---|---|
| outcomes | 288 perfect, 72 collisions, **0 starvations** |
| losses attributed to one decision in the last 40 | 70 of 72 |
| lead time from that decision to death | median 2, p90 13, max 29 |
| snake length at the fatal decision | median 83 |
| tail test flags the fatal move — observed / advanced tail / timed | 22.1% / **94.1%** / 92.6% |
| `lg(num_groups)` at those decisions | right 7.4%, wrong 57.4% |
| static free area | right 48.5%, wrong 36.8% |
| food sealed off from every legal move, at length ≥ 50 | 33.9% of steps |

If a re-run diverges from these, suspect the environment changed rather than the scripts — every
number above is tied to the observation space and reward function as of 2026-08-02.

**These numbers cannot be reproduced on `master`, and the failure is silent.** Two later changes
the same day rewrote observation indices 18 and 19, while leaving the vector 20 values wide — so
`b8f-disc9975seed2` still restores without an error and then plays like a beginner (0, 0, 1 over
three episodes, against 90.3% at the time). Check out commit `e4514a8` to reproduce anything above.
`simcheck.py`, `simcheck2.py` and `probe.py` load no checkpoint and still run anywhere.
