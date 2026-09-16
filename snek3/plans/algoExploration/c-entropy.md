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
| knobs | `SNEK_SAC_LEARNING_RATE` (3e-4), `SNEK_SAC_CRITIC_LEARNING_RATE` (3e-4), `SNEK_SAC_BATCH_SIZE` (256), `SNEK_SAC_TAU` (0.005, Polyak), `SNEK_SAC_TARGET_ENTROPY_RATIO` (0.98 of log |A|), `SNEK_SAC_INIT_ALPHA` (1.0), `SNEK_SAC_ALPHA_LEARNING_RATE` (3e-4), `SNEK_SAC_UPDATES_PER_STEP` (1). `DISCOUNT` and `COLLECT_ENVS` shared |

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

### C2 -- Revisiting Discrete SAC (Zhou, Wang, Feng & Zhou 2022)

The paper finds discrete SAC fails on Atari for two reasons and fixes each: the temperature drives
the policy to near-uniform when the target entropy is too high, and the Q estimate overshoots. The
fixes, as knobs on `algos/sac/`:

| fix | knob | default for `sac` | value for `sac2` |
|---|---|---|---|
| **entropy-penalty** -- the target entropy is scaled down and α's loss is clipped so the temperature cannot run away | `SNEK_SAC_TARGET_ENTROPY_RATIO`, `SNEK_SAC_ALPHA_CLIP` | 0.98, off | 0.7, on |
| **double average Q-learning with Q-clip** -- the target uses the *average* of the two critics rather than the min, and the TD target is clipped to a running range of the average | `SNEK_SAC_CRITIC_COMBINE` (`min`/`avg`), `SNEK_SAC_Q_CLIP` | `min`, off | `avg`, on |

`sac2` is `sac` with those values as its defaults; a `sac` spec can set any of them. The paper's
third recommendation, a larger batch, is `SNEK_SAC_BATCH_SIZE` and is set in the spec.

Tests: with `avg` and two equal critics the target equals the `min` target; Q-clip leaves a target
inside the range untouched and clips one outside; the α clip is inert when the loss is inside the
bound. Mutants: `avg` computed as a sum, the clip range read from the online net instead of the
running statistic.

## 3. The batches

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| C1 | 4 seeds of `sac` | the PPO reference's reward preset, `SNEK_OBS_HISTORY=8`, `SNEK_FC_LAYERS=320`; the group's defaults | A1 (DQN) and PPO's `hist8` table | stage-B density, `hof5000`, `hof30k`, drawdowns; **and the policy entropy trace** beside PPO's, which is in every PPO row already |
| C2 | 4 seeds of `sac2` | C1's | C1 | as C1 |
| C2 halves | 4 seeds entropy-penalty only, 4 seeds Q-clip only | C1's | C1, C2 | which fix did it |

The halves run only if C2 differs from C1 by more than noise (`n=4` resolves ~10 pp, `CLAUDE.md`);
if C1 and C2 are level there is nothing to attribute.

## 4. Gates

1. Smoke for both names; the actor checkpoint restores and watches as a PPO one does.
2. A 500k-move laptop arm shows α falling from 1.0 and the entropy approaching the target; a
   temperature that stays at its initial value is the paper's failure mode and is fixed before the row
   is queued.
3. The mutation spec kills every mutant.
4. Tuning budget: one laptop wave over `SNEK_SAC_LEARNING_RATE` and `SNEK_SAC_UPDATES_PER_STEP`.

## 5. What would change the plan

- **C1 matches PPO.** Entropy regularisation, not the on-policy update, is what PPO's result rests on.
  That makes A6's Munchausen result (an entropy-regularised value target) the natural comparison, and
  argues for adding a tuned-temperature entropy term to the value rows in B and D.
- **C1 collapses and C2 does not.** The implementation, not the idea; the halves say which fix. This
  is the paper's result and the expected one.
- **Both trail A1.** A stochastic actor is a cost on a deterministic game and the group closes; the
  remaining entropy question is A6's alone.
