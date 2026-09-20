# Group H: the outlier -- GDI / LBC

**Status: planned 2026-09-16, nothing built.** Group H of [`algorithm-series.md`](algorithm-series.md);
conventions in [`README.md`](README.md). Phase 9, the last row, behind a reproduction gate.

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
| the game | one Atari game the paper reports with a large margin over Rainbow and R2D2, not a hard-exploration one. From GDI's Tables 8-9 (verified 2026-09-16): **Kung Fu Master** (GDI-H3 1,666,665 against Rainbow 52,181 and R2D2 200,176), with Jamesbond (620,780 / 19,809 / 23,266), Krull (594,540 / 8,741 / 145,285) and Demon Attack (787,985 / 111,185 / 143,665) as the fallbacks. Kung Fu Master first because its margin is the largest and its episodes are short enough for a wave to show a curve |
| the target | the paper's reported score for that game at a budget the box can afford, scaled: the GDI numbers are at 200M frames with 160 environments, LBC's at **1B** frames. If the 200M-frame number is out of reach in a wave, the gate is that the learning curve at the box's budget sits on the paper's published curve for that game, within the seed spread the paper reports |
| the budget | **one desktop wave** (8 arms' worth of wall-clock). If the reproduction is not on the curve at the end of it, the row is closed as "not reproducible here" and the finding is written -- a real finding, not a failure |
| the code | a standalone harness under `algos/gdi/repro/`, using the authors' code where a licence permits and a clean reimplementation where it does not; it is **not** on the seam and is deleted after the gate (`docs/findings.md` keeps the result). Atari and its dependencies are installed in a separate conda env, never in `snek3` |

If the gate passes, the Snake row proceeds. If it fails, §2 and §3 are not built.

## 2. The row, if the gate passes

### H1 -- GDI / LBC on Snake

| piece | here |
|---|---|
| package | `algos/gdi/`, `NAME` `gdi` |
| the learner | an off-policy actor-critic: the actor is `algos/ppo/net.py`'s (a PPO checkpoint seeds it), the critic a `QNet`; targets by V-trace over sequences from a replay of recent trajectories (`algos/r2d2/replay.py`'s sequence buffer if E2 has landed, a simpler FIFO of sequences if not). The learner reuses what exists; the contribution is above it |
| the behaviour family | **GDI's, as written**: π_λ = ε · softmax(A_1 / τ_1) + (1 − ε) · softmax(A_2 / τ_2), λ = (τ_1, τ_2, ε) with 1/τ ∈ [0, 50] at unit steps and ε ∈ [0, 1] at 0.1 -- GDI-I3 shares one advantage head (θ_1 = θ_2), GDI-H3 has two heads under two reward shapings. The plan runs **I3**: one head, so the family is temperatures and a mixing weight over the learner's own advantages, `SNEK_GDI_FAMILY=i3` and its grid. Each lane is assigned a point in the family at episode start. LBC's form (three policies with their own γ and reward shaping, mixed by learned weights) is `SNEK_GDI_FAMILY=lbc` and is the second wave if I3 passes |
| the selector | GDI's: an **ensemble of tile-coding bandits** (7, tiling widths {2, 3, 4}, learning rates {0.05, 0.1, 0.2}, tile offsets uniform), each scoring a family point by its z-normalised extrinsic return plus c · √(log(1 + ΣN) / (1 + N_i)), acting by argmax or a random draw; no sliding window. `SNEK_GDI_SELECTOR=tile` is the paper's; `uniform` is the ablation. The improvement-based reward the earlier draft named is not in either paper and is dropped |
| the learner | GDI's, not a fresh design: policy gradient with **V-trace for V and Retrace for Q**, ρ̄ = c̄ = 1.05, loss scalings V 1.0, Q 10.0, π 10.0, **no entropy regularisation**; AdamW 5e-4 with a 4,000-step warm-up and linear anneal to 0, weight decay 0.01 → 0, β₂ 0.98, ε 1e-6, gradient norm 50; sequences of 80 with a 40-step burn-in and stored recurrent state, batch 64, each sequence replayed twice; an LSTM of 256 over the trunk; the auxiliary forward and inverse dynamics losses. The actor is `algos/ppo/net.py`'s shape plus the LSTM (E1's `RecurrentActorCritic` if it has landed), the critic a `QNet`; the sequence buffer is `algos/r2d2/replay.py`'s. What the paper has that is not replicated: 160 environments (the box's lanes are 32) and the two reward shapings of H3 |
| the sidecar | `algo` `gdi`; the checkpoint is the actor, restored as PPO's, so the eval measures the learner's greedy policy. A `--policy-variant family:<i>` measures a behaviour, as `a-return-tail.md` §5 |
| the step | R2D2's: one `collector.step()`, `collect_envs` moves |
| knobs | `SNEK_GDI_*`: the family and its grid, the selector and its bandit count, V-trace's ρ̄ and c̄, the three loss scalings, the optimiser's warm-up and anneal. Defaults are GDI's Table 5 values above; `SNEK_DISCOUNT` **0.997**, the paper's (`README.md`, "Translating"), with the reference's 0.99 in the PPO-seeded local cell. PPO's and DQN's knobs refused by name |

Tests: with a family of one point the algorithm is plain V-trace actor-critic (a fixture checks the
target against a hand-computed one); the selector converges to the better of two points on a
synthetic problem; a behaviour at temperature → 0 is the greedy actor. Mutants: the importance ratio
unclipped, a bandit's count not incremented on the point it chose, the behaviour sampled from the wrong lane's point.

## 3. The batch

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| H1 | 4 seeds of `gdi` on GDI-I3's settings above, fresh (the paper's form) + 4 seeds with the actor seeded from a PPO `hist8` checkpoint (the local variant) | the PPO reference's reward preset and `hist8` | PPO `hist8`, B1 (the other off-policy actor-critic in the series), F2 (the other bandit over behaviours) | stage-B density, `hof5000`, `hof30k`, drawdowns; **the selector's trace** -- which family points it settles on, which is the row's mechanistic reading |
| H1 selector | 4 seeds with the selector replaced by a uniform draw over the family + 4 seeds with a single fixed point (τ → 0, ε 1: the greedy learner, which is plain V-trace/Retrace actor-critic) | H1 | H1 | is it the family or the selection, and is either worth anything over the learner alone |

## 4. Gates

1. The reproduction gate, §1.
2. Smoke; the actor checkpoint restores and watches as a PPO one.
3. The mutation spec kills every mutant.
4. Tuning budget: one laptop wave on the family's grid and the bandit ensemble's size. The row is not tuned
   beyond that; a method whose headline is "the data distribution is the lever" should not need the
   learner's knobs tuned on this game.

## 5. What would change the plan

- **The gate fails.** The row closes with the finding, and the series ends with Group G's result as
  its last word. This is the likelier outcome and the plan is written so it costs one wave.
- **The gate passes and H1 beats PPO.** The selector's trace says what it found; if it settled on
  high-temperature behaviours early and greedy ones late, that is an exploration schedule the
  entropy rows (C) and Agent57's bandit (F2) can be read against, and it argues for a learned
  behaviour schedule on the incumbent.
- **The gate passes and H1 is level.** The mean HNS was breadth across games, not depth on one, and
  the row closes.
