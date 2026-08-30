# Results — every arm

The canonical arm table. One row per arm, filled in when the arm stops and its stage-B measurement
lands. Config, final numbers, verdict.

**Batch b1 — the DDQN baseline at every default, seeds 1-4, 3M steps.** Closed 2026-08-29. No
stage-B column: **no checkpoint in any of the four reached 95/100 in stage A**, so `screen:95`
selects nothing and there is nothing to measure at 500 episodes. The stage-A numbers are the result.

| arm | config | steps | trailing score | peak best30 | best single eval | ≥95/100 | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `b1a-baseline-seed1` | defaults | 3.00M | 92.26 | 42.1% | 49% | 0 | still rising at the cap |
| `b1b-baseline-seed2` | defaults | 3.00M | 92.65 | 58.3% | 70% | 0 | still rising at the cap |
| `b1c-baseline-seed3` | defaults | 3.00M | 92.87 | 56.7% | 68% | 0 | still rising at the cap |
| `b1d-baseline-seed4` | defaults | 3.00M | 94.17 | **81.9%** | **91%** | 0 | still rising at the cap |

**The phase-3 gate (≥90% perfect) is not met, and the batch does not say the learning code is
wrong.** Three separate reasons, in order of how much they matter:

1. **All four arms were cut off mid-climb.** Not one had plateaued: b1a's perfect rate went 20% at
   500k to ~40% at 3M, b1d's 0% to ~80%, both monotonically, and b1d's highest band is its last
   500k. The 3M cap is the binding constraint, not convergence.
2. **The config was never snek2's record config.** snek3's defaults are chase-safe shaping `c=0.0`
   and **IS weights on**; snek2's record is `c=0.10` at **gate 75** with **IS off**, and its own
   batch 28-29 finding is that *the gate is the lever*. b1 is the no-shaping baseline class, which
   in snek2 was also far from records. Gating phase 3 on it was my mistake — the plan's own phase 5
   names the b29/b47-class config, and that is what the gate needed.
3. **The gate's wording does not say which number it means.** snek2's best *pooled equal-effort* was
   90.50% while its headline 98-99% figures are single selected checkpoints at 500 episodes. Read as
   a trailing rate, "≥90% perfect" sits at snek2's absolute ceiling; read as "some checkpoint gets
   there", b1d's 91% single eval already passes.

The perfect-game counter is alive, which is worth confirming rather than assuming
([`invariants.md`](invariants.md) invariant 2 is about exactly this failing silently): b1d's
non-perfect games average **91.5 of 95**, so the arm is dying with three or four squares left, which
is the endgame this task has always been about.


## Batch p0 — the PPO tuning sweep

**15 arms, seed 1, 10M transitions each, all on b2's reward function**, each one knob off a reference
of lr 3e-4 / γ 0.99 / λ 0.98 / entropy 0.01 / fc 320 / 128x128 rollout / 4 epochs / minibatch 256.
Closed 2026-08-29. Seven ran on the laptop, eight on the desktop. **A tuning pass, not a gate** — no
arm is seed-matched to anything, so no row here supports a between-config claim on its own.

| arm | knob | best30 | sd30 | ≥95 evals | stage B: n | best | ≥98 |
|---|---|---:|---:|---:|---:|---:|---:|
| `p0q-ep8` | epochs 8 | **97.2** | 3.0 | 217 | pending | | |
| `p0k-fc200x100` | fc 200,100 | **97.1** | 2.5 | 215 | pending | | |
| `p0g-ent003` | entropy 0.003 | 96.9 | 2.8 | 153 | 153 | 98.6% | 2 |
| `p0n-fc300x100` | fc 300,100 | 96.9 | 2.4 | **233** | pending | | |
| `p0e-lam95` | λ 0.95 | 96.8 | 2.4 | 179 | 179 | 98.2% | **7** |
| `p0o-g995` | γ 0.995 | 96.7 | **1.8** | 166 | pending | | |
| `p0a-lr3e4-g99` | *the reference* | 96.6 | 2.2 | 108 | 108 | 98.4% | 6 |
| `p0j-lr5e4` | lr 5e-4 | 96.5 | 3.6 | 160 | 160 | **99.0%** | **7** |
| `p0m-fc200` | fc 200 | 96.4 | 2.0 | 131 | pending | | |
| `p0i-lr1e4` | lr 1e-4 | 95.0 | 5.0 | 47 | 47 | 97.4% | 0 |
| `p0l-fc500` | fc 500 | 94.7 | 3.2 | 93 | pending | | |
| `p0p-roll64` | rollout 64 | 94.1 | 2.9 | 104 | pending | | |
| `p0f-lam100` | λ 1.0 | 90.8 | 5.2 | 11 | 11 | 96.2% | 0 |
| `p0h-ent03` | entropy 0.03 | 90.6 | 9.1 | 9 | 9 | 94.8% | 0 |
| `p0r-mb1024` | minibatch 1024 | **89.7** | 4.4 | 16 | pending | | |
| `p0b`, `p0c`, `p0d` | lr 1e-3, lr 3e-3, γ 0.9975 | 85.2, 69.9, 81.6 | 7.2, 18.4, 4.7 | 1, 3, 0 | 1, 3, — | 94.4%, 95.8% | 0 |

`p0b`/`p0c`/`p0d` stopped at the 3M cap and are the arms the cap-inversion finding is measured
against; the rest ran 3M and were then resumed to 10M.

### What it establishes

**No winner.** Nine arms inside **0.8 pp** on best30, and three metrics give three orderings of the top
three (best30 → `p0q`; ≥98%/500 count → `p0e`/`p0j`; stage-B peak → `p0j`). At n=1 per config, that is
one number. **p0 hands p1 the reference config unchanged.**

**One axis moved, 7.5 pp, monotonically — gradient steps per transition.** minibatch 1024 (0.25x) 89.7
· reference (1x) 96.6 · epochs 8 (2x) 97.2. **Rollout size is a second axis:** `p0p-roll64` holds the
ratio fixed, halves the rollout, and loses ~2.5 pp.

**Two narrow layers beat one wide one, and 320 is more width than this task needs.** `fc 200,100` 97.1,
`fc 300,100` the most ≥95 checkpoints of any arm at 233, `fc 200` level with the 320 baseline, and
**`fc 500` clearly worse at 94.7**.

### Against DQN, at the same protocol

| | transitions | stage-B measurements | best | ≥98%/500 | density | wall clock per arm |
|---|---:|---:|---:|---:|---:|---:|
| **PPO p0, 7 laptop arms pooled** | 10M | 658 | **99.0%** | **22** | **3.34%** | **~3 min** (7 sharing 14 cores) |
| **DQN b2, 4 seeds pooled** | 18M | 1,135 | 99.2% | 5 | 0.44% | ~7-8 h (16 cores) |

**PPO's record-region density is 7.6x DQN's**, which is the metric
[`../plans/ppo.md`](../plans/ppo.md) §10 pre-registered for this comparison. DQN still holds the higher
single checkpoint.

**The honest depth, and it is the number to quote.** `p0j-lr5e4` @9,469,952 measured **99.0%/500** —
equal to snek2's admitted hall-of-fame record at that depth — and re-measured on a fresh seed at 3,000
episodes: **97.7% [97.1, 98.1]**, a 1.3 pp fall. `p0g-ent003` @8,159,232 fell 98.6% → **96.6%**
[95.9, 97.2]. So:

| policy | 3,000-episode measurement |
|---|---:|
| `b44a-import` @2739000 — snek2's champion, converted | **98.8%** [98.3, 99.1] |
| `p0j-lr5e4` @9469952 — PPO's best | **97.7%** [97.1, 98.1] |

**The champion is still ahead, by 1.1 pp and outside PPO's interval.** And it got there on **2.74M**
transitions against PPO's 9.47M, so on sample efficiency to a champion checkpoint the snek2 DQN lineage
is ~3.5x better. Neither number is a verdict on the algorithms — the champion is a selected best across
snek2's whole history and `p0j` is one arm of a first tuning sweep — but quoting PPO's 99.0%/500 without
this table would be quoting a selected high, which
[`../CLAUDE.md`](../CLAUDE.md) explicitly warns against.

## The PPO gate arm

**`ppo-smoke` — the phase-6b gate, not a batch arm.** 508k transitions at
[`../plans/ppo.md`](../plans/ppo.md) §7's untuned defaults, on the laptop, 2026-08-29. Kept because it
is the first PPO measurement in this project and the DQN comparison below is the reason 6c exists;
it is deliberately outside the p-series and nothing should be seed-matched against it.

| | transitions | avg score | perfect | notes |
|---|---:|---:|---:|---|
| `ppo-smoke`, stage A 100 eps | 508k | 77.6 | 1% | best single eval 3% at 442k |
| `ppo-smoke`, re-measured 500 eps | 508k | **79.55** | **1.2%** [0.6, 2.6] | median 82, **max 95** — perfect games happen |
| `b1a-d`, stage A 100 eps, matched | ~510k | 85.6 - 91.9 | 6 - 34% | b1's step 85,000 x 6 transitions |

**PPO learns this game, and at a matched sample budget it is behind DQN rather than beside it.** One
untuned arm against four tuned-by-nothing DQN seeds, so the gap is a starting point and not a verdict
— but it is the honest headline, and the four diagnostics say where to push:

| diagnostic | at 508k | reading |
|---|---:|---|
| `explained_variance` | **0.90** | the critic is not the problem, which is the risk §8 ranked highest |
| `approx_kl` | 0.002 | tiny |
| `clip_fraction` | 0.03 | **the clip is barely binding at 0.2, so the learning rate is *low*, not high.** The first knob for p0 |
| `entropy` | 1.086 → **0.27** | committing fast against ln 3 = 1.0986. Whether that is premature is p0's second question |

25.7k transitions/s at fc 320 on the laptop with the stage-A queue on, and `step == transitions`
exactly, which is the whole point of PPO's step unit.

![ppo-smoke](../runs/ppo-smoke.png)


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
