# Results — every arm

The canonical arm table. One row per arm, filled in when the arm stops and its stage-B measurement
lands. Config, final numbers, verdict.

*No arms yet.* snek3 has no learning code — `dqn/` is phase 3.

| arm | config | steps | best 500-ep | ≥98%/500 count | sef | verdict |
|---|---|---|---|---|---|---|

## Imported policies

Not arms: snek2 checkpoints converted to torch, kept as reference policies for A/B. They carry
snek2's training, so their numbers say something about **snek3's environment and measurement**, not
about snek3 as a learner.

| policy | source | rows | episodes/row | pooled perfect | snek2's own number |
|---|---|---:|---:|---|---|
| `b44a-import` @2739000 | `../../snek2/hallOfFame/b44a-lowlr7-b29b-ckpt2739000` | 1 | 3,000 | **98.8%** [98.3, 99.1] | 98.73% / 3,000 |
| `b45a-import`, seed 0 | every checkpoint of `../../snek2/savedPolicies/b45a-lowlr8-b29b` | 3,222 | 100 | **97.287%** | 97.291% |
| `b45a-import`, seed 1 | the same, a second food stream | 3,222 | 100 | **97.318%** | 97.291% |

Regenerated rather than committed, in one deterministic command — see
[`../CLAUDE.md`](../CLAUDE.md). The measurements are
[`../runs/b44a-import_phase1.json`](../runs/b44a-import_phase1.json),
[`../runs/b45a-import_checkpoint_evals_ab3222.json`](../runs/b45a-import_checkpoint_evals_ab3222.json)
and `..._ab3222seed1.json`.

**The 3,222-row pass is the phase-2 gate and it is the strongest measurement in the project.** Mean
per-row difference −0.004 pp against a 0.041 pp standard error, and per-row spread 2.30 pp observed
against 2.30 pp predicted by sampling alone — a ratio of 1.00, which leaves nothing for an
implementation difference. The threshold counts are in [`findings.md`](findings.md), along with why
the count of rows at exactly 100/100 disagreed and why that turned out to be a food stream.

**The 0.07 pp gap is two episodes and it is not evidence of anything.** 2964/3000 against
2962/3000, on different food streams, and the two 95% intervals are identical to a tenth of a point.
What *is* evidence is that the conversion is exact upstream of the measurement — see
[`findings.md`](findings.md).

**`avg_reward` is not comparable and `perfect_percent` is.** snek2 trained `b44a` with chase-safe
shaping at `c=0.10` and `FOOD_DISTANCE_REWARD=0`; the measurement above ran under snek3's defaults,
`c=0.0` and `0.001`. A greedy policy's action is an argmax over its own Q-values, so the reward
config cannot change which moves it plays or what it scores — it only changes the number the reward
terms add up to. That is why a reward figure is never the basis of a comparison here.

## Reading this table

- **`best 500-ep`** is the best row of the arm's stage-B file. It is a *selected* high — a record
  claim needs a fresh measurement of the winner at 1,000+ episodes
  ([`invariants.md`](invariants.md) invariant 9).
- **`≥98%/500 count`** is the width of the arm's record region, and it is the more robust number.
  snek2's champions were single lucky rows about as often as they were real plateaus.
- **`sef`** is `strong_eval_fraction`, the share of the arm's stage-A evals at ≥80% perfect.
  **Compare only at a common step horizon.**
- Every snek3 arm runs 100 episodes per stage-A eval and 500 per stage-B row, so nothing in this
  table needs an episode-count correction. A comparison against a **snek2** number does — see
  [`invariants.md`](invariants.md) invariant 8.
