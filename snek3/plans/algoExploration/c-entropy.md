# Group C: entropy -- Discrete SAC, Revisiting Discrete SAC

**Status: planned 2026-09-16, nothing built.** Group C of [`algorithm-series.md`](algorithm-series.md);
conventions in [`README.md`](README.md). Phase 3 of the running order; waits for A1.

The question: is PPO's advantage on this game the entropy bonus rather than the policy gradient? PPO
here carries an entropy coefficient that anneals 0.01 → 0.001 over the run, and the b27 `hist8` result
was found under it. Soft actor-critic learns a maximum-entropy policy with an automatically tuned
temperature, off-policy, from replay -- so it separates "entropy regularisation" from "on-policy
clipped policy gradient". C2 is the set of fixes that made discrete SAC competitive on Atari, run to
tell whether C1's result was the idea or the implementation.

## 1. What the group shares

| shared piece | decision |
|---|---|
| package | `algos/sac/`, one `algo.py` with `NAME`s `sac` and `sac2`, the second being the first with three knobs at non-default values (§2, C2). One package because C2 *is* C1 plus fixes, and the fixes are flags |
| networks | an actor over `QNet`'s hidden stack producing logits (`algos/ppo/net.py`'s actor, reused), and **two Q critics** of DQN's `QNet` shape plus their target copies. The actor is the checkpoint (`algo.net`); the critics ride in `resume.pt` |
| replay, collection | `algos/dqn/replay.py` and `collect.py` with the fork **off** and the shield off: SAC's exploration is its own entropy, and forking a stochastic policy's replay would put a different distribution under the critics than the actor induces. `SNEK_FORK_*`, epsilon and the shield knobs are refused by name |
| the sidecar | `algo` `sac` or `sac2`; no new head field -- the checkpoint is an actor of `QNet` shape, so `restore` reads it exactly as a PPO checkpoint, argmax over logits |
| the step | DQN's: one `collector.step()`, `collect_envs` moves; `SNEK_COLLECT_ENVS` default 16 so the replay fills at a useful rate |
| knobs | `SNEK_SAC_LEARNING_RATE` (3e-4), `SNEK_SAC_CRITIC_LEARNING_RATE` (3e-4), `SNEK_SAC_BATCH_SIZE` (64), `SNEK_SAC_TARGET_UPDATE_PERIOD` (2000 gradient updates, hard copy) / `SNEK_SAC_TAU` (1.0; 0.005 makes it Polyak), `SNEK_SAC_ALPHA` (`auto`, or a fixed number), `SNEK_SAC_TARGET_ENTROPY_RATIO` (0.98 of log \|A\|), `SNEK_SAC_INIT_ALPHA` (1.0), `SNEK_SAC_ALPHA_LEARNING_RATE` (3e-4), `SNEK_SAC_REPLAY_RATIO` (0.25 gradient steps per move), `SNEK_SAC_N_STEP` (1), `SNEK_SAC_ENTROPY_PENALTY` (0, off), `SNEK_SAC_CRITIC_COMBINE` (`min`), `SNEK_SAC_Q_CLIP` (0, off). `DISCOUNT` and `COLLECT_ENVS` shared. Defaults are C1's paper values (§2b) |

## 2. The rows

### C1 -- Discrete SAC (Christodoulou 2019)

SAC with a categorical policy: the expectation over actions is exact rather than sampled, so the
soft value is Σ_a π(a|s)[min(Q1, Q2)(s, a) − α log π(a|s)], the actor loss is the KL of π against
the softmax of Q/α, and the temperature α is tuned by gradient to a target entropy.

| module | contents |
|---|---|
| `algos/sac/agent.py` | the soft target, the twin-critic loss (Huber, as `algos/dqn/agent.py` argues for a +100 terminal), the actor loss, the α loss on log α, Polyak update of both targets |
| `algos/sac/algo.py` | the seam: `advance()` collects `collect_envs` moves then runs `updates_per_step` updates; `fields()` reports α and the policy entropy; `on_eval` does nothing to a schedule, since SAC has none -- it returns α so the row carries it |
| `algos/sac/net.py` | `build(arch)` returns the actor for restore; the critics are built by the agent |

Tests: with a uniform policy and equal critics the soft value is the hand-computed constant; the α
gradient's sign flips as entropy crosses the target; the actor loss is minimised by the softmax of
Q/α on a two-state example; Polyak at τ = 1 is a hard copy. Mutants: the min over critics replaced by
the mean, the log π term's sign, the `(1 − done)` on the soft target, the entropy target's sign.

### C2 -- Revisiting Discrete SAC (Zhou, Wang, Feng & Zhou 2022; TMLR 2024, arXiv 2209.10081)

The paper finds discrete SAC fails on Atari for two reasons and fixes each -- and **the fixes are not
what an earlier draft of this plan said** (verified 2026-09-16 against the paper's §5 and Table 3 and
the authors' code). The temperature is **fixed at α = 0.05** in every main run, not auto-tuned; the
policy's entropy is stabilised by an **entropy-penalty** on the *change* in entropy between updates;
and the Q overshoot is fixed by **double average Q-learning with a Q-clip**, which is a PPO-style clip
on the critic's update rather than a clip of the target to a running range.

| fix | what it is | knob | `sac` | `sac2` |
|---|---|---|---|---|
| **entropy-penalty** | β · ½ E[(H(π_old) − H(π))²] added to the policy loss, π_old the policy before the update; β swept {0.1, 0.2, 0.5, 1} | `SNEK_SAC_ENTROPY_PENALTY` | 0 | 0.5 |
| **double average Q** | the target is r + γ · avg(Q′₁, Q′₂) instead of the min | `SNEK_SAC_CRITIC_COMBINE` | `min` | `avg` |
| **Q-clip** | the critic loss is max((Q − y)², (Q′ + clip(Q − Q′, −c, c) − y)²), Q′ the target critic; c swept {0.5, 1, 2, 5} | `SNEK_SAC_Q_CLIP` | 0 | 0.5 |
| fixed temperature | α is a constant, no target entropy and no α optimiser | `SNEK_SAC_ALPHA` | `auto` | 0.05 |

`sac2` is `sac` with those values as its defaults; a `sac` spec can set any of them, and a `sac2` spec can
put α back on `auto` for the ablation that asks whether the fixed temperature is itself a fix.

Tests: with `avg` and two equal critics the target equals the `min` target; the Q-clip term equals the
plain squared error when \|Q − Q′\| < c and is the larger of the two branches otherwise; the entropy
penalty is zero when the policy did not move and grows as its square; with `SNEK_SAC_ALPHA` fixed the
α optimiser is not built. Mutants: `avg` computed as a sum, the clip applied to the target instead of
the online-minus-target difference, the penalty using the entropy's sign rather than its square.

## 2b. The papers' settings, and how each lands here

| setting | SAC-Discrete (Christodoulou 2019, Table 2) | Revisiting (Zhou et al., Table 3, both agents) | here |
|---|---|---|---|
| optimiser, lr | Adam 3e-4, all nets | Adam 1e-5 actor and critic, 3e-4 for α when auto | the paper's, per row: C1 3e-4, C2 1e-5 |
| batch | 64 | 64 | 64 |
| replay | 1M; 20k random steps before learning | 1e5 | C1 1M, C2 1e5; prefill 20k moves |
| target | hard copy, "fixed network update frequency 8000" -- **the unit is not stated**; read as agent steps, as Dopamine counts it, that is 2,000 gradient updates at one update per 4 steps (the pseudocode writes Polyak, no τ given) | Polyak τ 0.005 | `SNEK_SAC_TARGET_UPDATE_PERIOD` **2,000** gradient updates for C1 (the knob counts updates, as DQN's does); τ 0.005 for C2 |
| update frequency | 1 gradient step per 4 env steps | 0.1 per step (Tianshou `update-per-step`) | `SNEK_SAC_REPLAY_RATIO` 0.25 for C1, 0.1 for C2 |
| n-step | 1 | 3 | per row |
| temperature | auto, target 0.98 · log \|A\| | fixed 0.05 | per row |
| network | Nature CNN → 512 | 2 × 512 | `fc 320` actor and critics, the reference's trunk; a 2 × 512 cell is the local departure worth one wave if C2 is short |
| critic loss | MSE | (Eq. 17 above) | Huber for C1 (the +100 terminal, as `algos/dqn/agent.py` argues) -- **a stated departure**; C2's clip is on the squared error as the paper writes it |
| discount | 0.99 | 0.99 | 0.99 |
| reward | clipped [−1, 1] | clipped | not clipped |
| budget | 100k agent steps, 5 seeds | 10M env steps, 3 seeds | 50M moves a cell, raised if still rising; C1's paper budget is tiny and is the reason it also runs at F1's 500k-step cap |

**On this game the fixed α is the thing to watch.** α 0.05 was set against a unit reward; here food is 1,
so the entropy term is the paper's relative to a meal and 2,000× smaller relative to the win. The paper
cell keeps 0.05 as written; if the policy goes deterministic before the endgame is learned, the one
tuning wave (§4) is over α ∈ {0.05, 0.2, 1.0}, and `auto` is the comparison.

## 3. The batches

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| C1 | 4 seeds `sac` on the 2019 paper's settings (§2b) + 4 seeds `sac` at F1's 500k-step cap (the paper's own regime, scaled) | the PPO reference's reward preset, `SNEK_OBS_HISTORY=8`, `SNEK_FC_LAYERS=320` | A1 paper (DQN) and PPO's `hist8` table | stage-B density, `hof5000`, `hof30k`, drawdowns; **and the policy entropy trace** beside PPO's, which is in every PPO row already |
| C2 | 4 seeds `sac2` on the 2022 paper's settings (§2b) + 4 seeds `sac2` with `SNEK_SAC_ALPHA=auto` | C1's | C1 | as C1; the fixed-vs-auto α pair says whether the temperature was a fix |
| C2 halves | 4 seeds entropy-penalty only, 4 seeds avg-Q + Q-clip only (the paper's two ablations) | C2's | C1, C2 | which fix did it |

The halves run only if C2 differs from C1 by more than noise (`n=4` resolves ~10 pp, `CLAUDE.md`);
if C1 and C2 are level there is nothing to attribute.

## 4. Gates

1. Smoke for both names; the actor checkpoint restores and watches as a PPO one does.
2. A 500k-move laptop arm of `sac` shows α falling from 1.0 and the entropy approaching the target; a
   temperature that stays at its initial value is the 2019 agent's failure mode and is fixed before the
   row is queued. The same arm of `sac2` logs the entropy-penalty term and the fraction of critic
   samples the Q-clip's second branch wins, both non-zero.
3. The mutation spec kills every mutant.
4. Tuning budget: one laptop wave over α (§2b) for C2 and over `SNEK_SAC_LEARNING_RATE` for C1.

## 5. What would change the plan

- **C1 matches PPO.** Entropy regularisation, not the on-policy update, is what PPO's result rests on.
  That makes A6's Munchausen result (an entropy-regularised value target) the natural comparison, and
  argues for adding a tuned-temperature entropy term to the value rows in B and D.
- **C1 collapses and C2 does not.** The implementation, not the idea; the halves say which fix. This
  is the paper's result and the expected one.
- **Both trail A1.** A stochastic actor is a cost on a deterministic game and the group closes; the
  remaining entropy question is A6's alone.
