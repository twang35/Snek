# Group H: the outlier -- GDI / LBC

**Status: planned 2026-09-16, nothing built.** Group H of [`algorithm-series.md`](algorithm-series.md);
conventions in [`README.md`](README.md). Phase 8, the last row, behind a reproduction gate.

GDI -- Generalized Data-distribution Iteration (Fan & Xiao 2022) -- and its successor LBC -- Learnable
Behavior Control (Fan, Xiao et al. 2023) -- report the highest mean human-normalised scores on Atari.
The idea: treat the *data distribution* the agent learns from as the thing being optimised, by
maintaining a **family of behaviour policies** (a parameterised space of, for example, entropy
temperatures and value mixtures over the learner's outputs) and a **selector** that shifts the
sampling of that family toward the behaviours whose data most improves the learner. LBC frames the
family as a learnable behaviour space and the selector as a bandit over it. The learner underneath is
an off-policy actor-critic with V-trace / Retrace-style corrections.

It is the least reproduced result in the series -- one group, no maintained reference implementation
beyond the authors' -- which is why the row is a **reproduction first and an experiment second**. The
mean is interesting; whether it can be made to exist on this box is the first question.

## 1. The reproduction gate

Before any Snake code is written, the row has to show that the method as published can be
reproduced on **one known Atari game, on the desktop, within one wave's budget**. This is the only row
in the series that runs anything but Snake, and it runs it for one reason: a Snake result from an
implementation that cannot reproduce its own paper's number tells us nothing about GDI, only about
the implementation.

| gate | rule |
|---|---|
| the game | one Atari game the paper reports with a large margin over Rainbow and R2D2 (the plan names it when the implementer has read the tables; candidates are the games where GDI-H3's per-game score is far above the field, not a hard-exploration one) |
| the target | the paper's reported score for that game at a budget the box can afford, scaled: if the paper's 200M-frame number is out of reach in a wave, the gate is that the learning curve at the box's budget sits on the paper's published curve for that game, within the seed spread the paper reports |
| the budget | **one desktop wave** (8 arms' worth of wall-clock). If the reproduction is not on the curve at the end of it, the row is closed as "not reproducible here" and the finding is written -- a real finding, not a failure |
| the code | a standalone harness under `algos/gdi/repro/`, using the authors' code where a licence permits and a clean reimplementation where it does not; it is **not** on the seam and is deleted after the gate (`docs/findings.md` keeps the result). Atari and its dependencies are installed in a separate conda env, never in `snek3` |

If the gate passes, the Snake row proceeds. If it fails, §2 and §3 are not built.

## 2. The row, if the gate passes

### H1 -- GDI / LBC on Snake

| piece | here |
|---|---|
| package | `algos/gdi/`, `NAME` `gdi` |
| the learner | an off-policy actor-critic: the actor is `algos/ppo/net.py`'s (a PPO checkpoint seeds it), the critic a `QNet`; targets by V-trace over sequences from a replay of recent trajectories (`algos/r2d2/replay.py`'s sequence buffer if D2 has landed, a simpler FIFO of sequences if not). The learner reuses what exists; the contribution is above it |
| the behaviour family | a space of behaviour policies derived from the learner's outputs: softmax over a mixture of the actor's logits and the critic's advantages, parameterised by a temperature and a mixing weight -- `SNEK_GDI_FAMILY` names the parameterisation and its grid size (16). Each lane is assigned a point in the family at episode start |
| the selector | LBC's bandit: a sliding-window UCB over family points, rewarded by the **learner's** improvement attributable to data from that point (the paper's proxy is the extrinsic return of the behaviour; the plan uses that first and the improvement proxy as `SNEK_GDI_SELECTOR=improvement` for the ablation) |
| the sidecar | `algo` `gdi`; the checkpoint is the actor, restored as PPO's, so the eval measures the learner's greedy policy. A `--policy-variant family:<i>` measures a behaviour, as `a-return-tail.md` §5 |
| the step | R2D2's: one `collector.step()`, `collect_envs` moves |
| knobs | `SNEK_GDI_*`: the family, the selector, its window and UCB weight, V-trace's ρ̄ and c̄, the learning rates. PPO's and DQN's knobs refused by name |

Tests: with a family of one point the algorithm is plain V-trace actor-critic (a fixture checks the
target against a hand-computed one); the selector converges to the better of two points on a
synthetic problem; a behaviour at temperature → 0 is the greedy actor. Mutants: the importance ratio
unclipped, the selector's window not sliding, the behaviour sampled from the wrong lane's point.

## 3. The batch

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| H1 | 4 seeds of `gdi` | the PPO reference's reward preset and `hist8`; the actor seeded from a PPO `hist8` checkpoint for two seeds, fresh for two | PPO `hist8`, C1 (the other off-policy actor-critic in the series), E2 (the other bandit over behaviours) | stage-B density, `hof5000`, `hof30k`, drawdowns; **the selector's trace** -- which family points it settles on, which is the row's mechanistic reading |
| H1 selector | 4 seeds with the selector replaced by a uniform draw over the family | H1 | H1 | is it the family or the selection |

## 4. Gates

1. The reproduction gate, §1.
2. Smoke; the actor checkpoint restores and watches as a PPO one.
3. The mutation spec kills every mutant.
4. Tuning budget: one laptop wave on the family's grid and the selector's window. The row is not tuned
   beyond that; a method whose headline is "the data distribution is the lever" should not need the
   learner's knobs tuned on this game.

## 5. What would change the plan

- **The gate fails.** The row closes with the finding, and the series ends with Group G's result as
  its last word. This is the likelier outcome and the plan is written so it costs one wave.
- **The gate passes and H1 beats PPO.** The selector's trace says what it found; if it settled on
  high-temperature behaviours early and greedy ones late, that is an exploration schedule the
  entropy rows (C) and Agent57's bandit (E2) can be read against, and it argues for a learned
  behaviour schedule on the incumbent.
- **The gate passes and H1 is level.** The mean HNS was breadth across games, not depth on one, and
  the row closes.
