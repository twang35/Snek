# snek3 — PPO

**Status: draft for review, 2026-08-29. Not approved, no code written.** Phase 6 of
[`pytorch-port.md`](pytorch-port.md), which calls it "the actual research".

The whole design question is not "how do you write PPO" — that is ~200 lines — but **how much of the
existing stack a PPO arm can share, so that a PPO-vs-DQN result is a statement about the algorithm
and not about two measurement paths.** The answer is: everything below the algorithm, plus the entire
measurement protocol, unchanged. What has to go is four DQN-specific mechanisms, and one of them
(forking) PPO replaces for free.

## Decisions proposed

| | proposal |
|---|---|
| **Env, observation, rewards** | **byte-identical.** No change to `env/` or `vectorized/` at all |
| **Reward shaping** | **stays, and matters more than it does for DQN.** It is reward-side and algorithm-agnostic. §3 |
| **Forking** | **goes.** A forced action from a cloned state is off-policy by construction. PPO's own stochastic policy does the same job with a correct log-prob. §2 |
| **Epsilon + the shield** | **go.** Replaced by the entropy bonus. The *finding* behind the shield survives as an optional logit mask. §2 |
| **PER, target net, double-Q, n-step, prefill** | **go.** Replaced by a `(T, N)` rollout buffer and GAE |
| **Network** | **two separate towers**, actor `30 -> 320 -> 3` — the same shape and the same initialisers as `QNet`, so the policy function class is identical — and critic `30 -> 320 -> 1`. §6 |
| **Checkpoints** | `ckpt-<step>.pt` holds **the actor only**, so `arch.json` needs no new field and stage B measures the policy exactly as it does for DQN. The critic lives in `resume.pt`. §6 |
| **Eval protocol** | **unchanged.** Stage A 100 episodes on every checkpoint, stage B 500 on every checkpoint at ≥95/100, `screen:95`, the same charts and reports |
| **Eval policy** | **argmax over the logits**, the analogue of DQN's greedy. Sampled evaluation is a later knob |
| **Step unit** | **a PPO step is one transition = one game move.** Every row of both algorithms gains a `transitions` field. §4 |
| **`train.py`** | **one entry point, one eval/checkpoint/report path**, with the algorithm behind a five-method object. §5 |
| **First batch** | **a 2-arm tuning pass before the 4-arm gate.** b1 gated on untuned defaults and answered nothing; do not repeat it. §10 |

---

## 1. What stays identical, and why that is the point

| layer | status |
|---|---|
| `env/`, `vectorized/` — game, 30-value observation era `b09c616`, rewards, both shaping terms, `VecSnake`, `engine.measure` | **untouched** |
| the `policy_fn` seam — `(m, 30) float32 -> (m,) int64` | **untouched.** An argmax over logits has the same shape as an argmax over Q |
| stage A / stage B, 100 / 500 episodes, the ≥95 screen, `MIN_CHECKPOINT_SCORE`, the Wilson interval | **untouched** |
| `runs/<policy>.{md,png}`, `_evals.json`, `run_report`, `progress_chart`, `stage_b_chart`, `compare_results` | **untouched** except one column (§4) |
| `tools/checkpoints.py`, `arch.py`, `restore.py`, `watch.py`, `record_gif.py` | **one line**: `ALGORITHMS['ppo']` in `restore.py`, which that file's comment already anticipates |
| the stage-A queue, the eval workers, the chart window, `live_runs` | **untouched.** `eval_queue` copies the trainer's fields blind — its own comment says it has "no business knowing what an epsilon is" |
| the desktop git bus | **untouched.** A spec carries an arbitrary `env` dict, so `SNEK_ALGO=ppo` needs no daemon change |

That list is the reason this is worth doing here rather than in a fresh directory. A PPO arm and a
DQN arm will be measured by the same instrument, on the same board, with the same reward function, at
the same episode counts — so `docs/protocol.md`'s judging rules apply to the comparison as written,
and `tools/compare_results.py` can put a PPO stage-B pass against b2's directly.

**One pre-existing bug found while reading, and it is not PPO's:** `train.py` builds
`VecSnake(width, seed=...)` and never passes `shaping_discount`, so it stays 1.0 while b2 trains at
γ=0.9975. Potential-based shaping pays `c·(γ·Φ(s') − Φ(s))` and the invariance theorem needs the
*agent's* γ, so b2's shaping is off by `c·(1−γ)·Φ ≈ 2.5e-4` per step — the same order as
`FOOD_DISTANCE_REWARD`, and it biases toward staying alive in a high-Φ state. Small, real, and it
should be fixed for PPO. Fixing it for DQN changes b2's dynamics, so that is a separate decision.

---

## 2. What PPO cannot keep

### Forking goes, and PPO replaces it for free

`dqn/collect.py`'s fork clones a primary lane with `copy_rows` and forces an action **the policy did
not take**. Its docstring states the problem it solves: at ε≈0.003 the buffer holds the consequence
of the chosen action and never of the alternative, so `Q(s, a_good)` is never raised and the argmax
has no reason to flip.

That is a *Q-learning* problem. PPO has no argmax to flip: it samples from π and the gradient moves
probability mass by the sign of the advantage, so both actions at a decision point are tried, with
correctly attributed log-probs, as a matter of course. And the fork's data is unusable to PPO anyway
— the clipped objective's ratio `π(a|s)/π_old(a|s)` assumes the batch was drawn from `π_old`, and a
forced action was not drawn from anything.

So forking is deleted from the PPO path, `SNEK_FORK_*` is rejected rather than ignored when
`SNEK_ALGO=ppo` (the project's rule: an override that quietly does something else is worse than one
that refuses), and the 4x step-count artefact it caused disappears with it.

### The epsilon schedule and the shield go; the finding behind the shield survives

`dqn/schedules.py`' two phases and `dqn/agent.py`'s shield exist together: a mastery-gated ε deadlocks
without the shield, because random endgame moves kill the snake, the buffer fills with trajectories
that never finish a board, and the perfect rate that drives ε stays 0. Four snek2 arms sat there for
942k steps.

PPO has no ε. Exploration is the entropy bonus, and it is a *cost* term rather than a forced
random-action rate, so the deadlock mechanism is much weaker — the policy is free to become
near-deterministic in the endgame while staying stochastic where it is unsure. The replacement is
`SNEK_PPO_ENTROPY_COEF`, fixed for the first batch with an optional linear anneal.

**The shield does have a clean PPO form, and it is a different agent.** Masking the fatal moves
(observation indices 6-8) in the logits before the softmax is standard invalid-action masking, and
the gradients are correct because the policy being optimised *is* the masked policy. But then the
masked policy is what has to be evaluated, or stage A measures a function that was never trained. So
this is **two settings, not three**:

| `SNEK_PPO_ACTION_MASK` | meaning |
|---|---|
| `0` (default for the comparison batch) | no mask anywhere. The clean A/B against b2 |
| `1` | masked at collect **and** at eval. A deliberately different, probably better, artefact |

A collect-only mask is not offered: it would train π_masked and measure π_unmasked. Note that `1`
costs nothing structurally — the mask is derived from the observation, so `greedy_policy_fn` stays a
`(m, 30) -> (m,)` callable and the seam holds.

### PER, the target network, double-Q, n-step windows and the prefill go

All five are off-policy machinery. `dqn/replay.py` (271 lines of sum tree and importance weights) is
replaced by `ppo/rollout.py`, ~120 lines of preallocated `(T, N)` arrays and a backward GAE pass. The
target network and the double-Q split have no analogue; the n-step window is what GAE's λ
generalises; and there is nothing to pre-fill because the first rollout *is* the first batch.

`SNEK_REPLAY_RATIO` also goes, and its job is done by `epochs × rollout / minibatch` — which is worth
naming because the numbers are startling. §4.

---

## 3. Reward shaping stays, and it is more important here

Shaping lives in `vectorized/vec_env.py::_shaping_reward` and `env/game.py`, on the reward, before any
algorithm sees it. Nothing about it is DQN-specific and none of it is touched.

It should also *help more*. DQN gets two things PPO does not: a replay buffer that keeps a rare
terminal +100 transition available for thousands of gradient steps, and prioritised replay that
actively resurfaces it (a perfect game's TD error is enormous). PPO sees each transition in one
rollout, for `epochs` passes, and then never again. Dense reward is the substitute, and both shaping
terms are dense by construction — `chase_safe` flips 2.5-3.6 times per meal for a struggling policy.

Two consequences for the plan:

- **b2's reward config is the one to run**, so the shaping dose is matched: `CHASE_SAFE_SHAPING=0.1`,
  `CHASE_SAFE_GATE=75`, `FOOD_DISTANCE_REWARD=0`.
- **`shaping_discount` must be passed the agent's γ** on the PPO path, or the telescope does not
  close and the shaping is no longer policy-invariant. See the bug in §1.

The one thing to keep in mind is [`docs/invariants.md`](../docs/invariants.md) invariant 1: a perfect
game is identified by its score, never its reward. Nothing in PPO reads the reward to classify an
outcome, so this is a tripwire that stays green rather than work.

---

## 4. The step unit, which is the thing that will be misread

[`docs/findings.md`](../docs/findings.md) already carries the scar: a snek3 DQN step is **four** game
moves and a snek2 step was one, and that confounded the whole b2-vs-b47 interim reading. PPO makes it
worse — a rollout is `envs × T` transitions, so a naive counter would make a PPO "step" 16,384 game
moves.

**Proposal: for PPO, `self.step` counts transitions, so a step is one game move and one buffer row.**
Then a PPO step number is directly comparable to a snek2 step number, and to a snek3 DQN step number
after the documented 4x. And **every eval row of both algorithms gains a `transitions` field**, which
the DQN collector already tracks in `counters['transitions']` — so the ambiguity is removed at the
source instead of being carried in a doc warning.

The eval and checkpoint intervals stay equal by construction, which is `train.py`'s one
non-negotiable interval rule. For PPO they are set as `rollout_transitions × SNEK_PPO_EVAL_ROLLOUTS`
rather than by `SNEK_EVAL_INTERVAL`, so an eval always lands on a rollout boundary and no checkpoint
exists that no screen can select.

### The cost model inverts, and it is worth stating before anyone is surprised

| per transition | DQN (b2) | PPO (envs 128, T 128, 4 epochs, minibatch 256) |
|---|---:|---:|
| gradient steps | 1.0 | **0.016** |
| times each transition is used | 1 | 4 |

DQN does 64x more backprop per transition. Projected — **to be measured in the smoke pass, not
trusted from here** — a PPO arm at b2's transition budget (12M) trains in minutes rather than 2.8 h,
and stage A becomes essentially the entire wall clock: 12M transitions at an eval every 32,768 is 366
evals × 100 episodes = 36,600 episodes, ~6 min streamed. **A PPO arm may cost well under an hour.**

Two things follow. **Arms are cheap, so run seeds and sweeps rather than agonising over defaults** —
this is what makes the tuning pass in §10 affordable. And the honest comparison axis is
**transitions**, with wall clock reported beside it as a separate fact; a PPO arm that matches b2 on
transitions and takes a tenth of the time is a real result, and so is one that needs 10x the
transitions and still finishes first.

---

## 5. Where PPO plugs into `train.py`

`train.py` is 822 lines and roughly 350 of them are the machinery that has to be shared: the arch
sidecar, the checkpoint cadence and its prune/save inversion, stage A in-process and queued, the
queue drain and its four patience constants, resume, the report, the chart, the window, the cap.
Duplicating that into a `train_ppo.py` would fork the measurement path, which is the one thing this
comparison cannot afford.

**Proposal: keep one `train.py` and one `Trainer`, and put the algorithm behind five methods.**

```python
algo.prefill()                                    # DQN fills the replay; PPO returns immediately
algo.advance()            -> (steps, transitions) # one iteration: collect, then learn
algo.on_eval(eval_rows, measured) -> dict         # the schedule's move, plus this row's algo fields
algo.policy_fn                                    # greedy, for engine.measure
algo.net                                          # what a ckpt holds: the Q net, or the actor
algo.state_dict() / load_state_dict()             # what resume.pt holds
```

`dqn/algo.py::DqnAlgo` wraps the existing agent + replay + collector + `dqn/schedules`; nothing inside
those files changes. `ppo/algo.py::PpoAlgo` wraps net + rollout + collector + `ppo/schedules`. The
`Trainer` chooses by `SNEK_ALGO` (default `dqn`) and adds `steps` to `self.step` and `transitions` to
`self.transitions`.

`advance()` returning both counters is what lets DQN keep its exact meaning (steps 1, transitions 4)
while PPO gets `steps == transitions`. The interval test becomes "crossed a multiple of the interval",
which for DQN's increment of 1 reduces to `% == 0` — bit-identical.

**This refactor is the riskiest item in the plan, because it touches the code four b2 arms are running
from.** Two mitigations, both non-negotiable:

- **Prove bit-exactness for DQN before any PPO arm runs.** A fixed-seed 2,000-step arm's
  `_evals.json` must be byte-identical before and after, with `SNEK_EVAL_QUEUE=0` for determinism —
  the same standard the fused-Adam and sum-tree changes were held to.
- **Land it after b2 closes** (~1.4 h of training left at 15:04, then its stage-B wave). Nothing goes
  to the desktop while the daemon may relaunch an arm from a half-refactored tree.

---

## 6. `ppo/`, module by module

| file | ~lines | contents |
|---|---:|---|
| `ppo/net.py` | 110 | `PolicyNet` (30 -> 320 -> 3 logits) and `ValueNet` (30 -> 320 -> 1). `build()` returns the actor. `greedy_policy_fn` = argmax of logits, optionally masked |
| `ppo/rollout.py` | 120 | the `(T, N)` buffer and the backward GAE pass. Pure numpy; no torch |
| `ppo/collect.py` | 130 | N lanes, T steps. Samples, stores `(obs, action, logprob, value, reward, done)`. No forking, no windows, no shield |
| `ppo/agent.py` | 180 | the clipped surrogate, the value loss, the entropy bonus, the epoch/minibatch loop, and the diagnostics |
| `ppo/schedules.py` | 60 | the entropy coefficient and the optional LR anneal, as pure functions — the same stateless shape as `dqn/schedules.py`, for the same resume reason |
| `ppo/algo.py` | 120 | the five-method object §5 drives |

Six decisions inside those worth arguing before they are written.

**Two separate towers, not a shared trunk.** Three reasons and the third is the one that decides it.
The actor is then *exactly* `QNet`'s shape and initialisers, so the comparison is "same policy
function class, different learning rule". `vf_coef` becomes nearly irrelevant, because a value loss
of ~1600 at initialisation (V starts ≈0, true V is ~40 at γ=0.9975) cannot contaminate a policy
gradient it shares no parameters with. And the actor is its own `nn.Module`, so `ckpt-<step>.pt` can
hold `actor.state_dict()` and `checkpoints.load(..., strict=True)` works with no wrapper and no new
`arch.json` field — `arch.py`'s `FIELDS` are "every field is required", so adding one would invalidate
every committed sidecar.

**snek3's initialisers, not orthogonal.** He-normal with Keras' truncation correction on the hidden
layer and `uniform(-0.03, 0.03)` on the head, from `dqn/net.py`. On a 3-logit head that uniform range
gives an almost exactly uniform softmax, which is what PPO's conventional `gain=0.01` head is *for*,
so the convention is already satisfied. `SNEK_PPO_INIT=orthogonal` exists as a fallback if p0 shows
the opening policy matters.

**A dedicated seeded generator for the action sample**, and a second for the minibatch shuffle,
neither shared with the env's food stream — `dqn/collect.py`'s comment on exactly this ("an arm's
decisions would depend on how many food cells were rejected") applies unchanged.

**Learning rate 3e-4, not DQN's 1e-5.** Reusing 1e-5 is a trap worth naming in the doc: PPO takes
~64x fewer gradient steps per transition, so the same LR is ~64x less total parameter movement over
an arm. 3e-4 is the PPO convention and p0 sweeps it.

**Huber on the value loss, with `SNEK_PPO_VALUE_LOSS=mse` available.** Same argument `dqn/agent.py`
gives for its TD loss: a perfect game pays +100 against a typical step's ~0.001, so one terminal
return in a minibatch of 256 dominates a squared error. This is a deliberate deviation from textbook
PPO and it is a knob, not a fact.

**Diagnostics are part of the row, not the log.** Each eval row carries a `ppo` sub-dict — exactly as
a DQN row carries `fork` — with `entropy`, `approx_kl`, `clip_fraction`, `explained_variance`,
`policy_loss`, `value_loss`. These are what distinguish "PPO does not work on this task" from "the
policy collapsed at 200k transitions", and this project's culture is that an unmeasured failure is a
wasted arm. `run_report.EVAL_COLUMNS` gains `entropy_coef` and skips columns absent from every row, so
a DQN report is unchanged and a PPO report has no blank `epsilon` column.

---

## 7. The knobs

Reusing `SNEK_COLLECT_ENVS` for the lane count rather than inventing `SNEK_PPO_ENVS`: it means the
same thing in both algorithms, and the value used is recorded in `runs/<policy>.md` either way.
`SNEK_FORK_*`, `SNEK_INITIAL_EPSILON`, `SNEK_MIN_EPSILON`, `SNEK_GUIDED_FRACTION`, `SNEK_REPLAY_*`,
`SNEK_PRIORITY_EXPONENT`, `SNEK_IS_*`, `SNEK_TARGET_UPDATE_*` and `SNEK_N_STEP_UPDATE` are **rejected
with a named error** under `SNEK_ALGO=ppo`.

| knob | default | notes |
|---|---|---|
| `SNEK_ALGO` | `dqn` | `ppo` selects the whole path. Recorded in `arch.json` as it already is |
| `SNEK_COLLECT_ENVS` | 1 dqn / **128** ppo | lanes. `VecSnake.step` costs 536 us at 1 lane and 950 us at 64, so width is nearly free |
| `SNEK_PPO_ROLLOUT` | 128 | T. `envs × T` = 16,384 transitions per iteration |
| `SNEK_PPO_EPOCHS` | 4 | passes over each rollout |
| `SNEK_PPO_MINIBATCH` | 256 | samples/s peaks around batch 256-1,024 and degrades past it |
| `SNEK_PPO_CLIP` | 0.2 | |
| `SNEK_PPO_GAE_LAMBDA` | 0.98 | higher than the usual 0.95 — see §8 |
| `SNEK_PPO_ENTROPY_COEF` | 0.01 | max entropy on 3 actions is ln 3 = 1.099 |
| `SNEK_PPO_VF_COEF` | 0.5 | near-inert with separate towers |
| `SNEK_PPO_LEARNING_RATE` | 3e-4 | **not** DQN's 1e-5 |
| `SNEK_PPO_TARGET_KL` | 0 (off) | early-stop the epoch loop; reported either way |
| `SNEK_PPO_NORMALIZE_ADV` | 1 | per minibatch |
| `SNEK_PPO_ACTION_MASK` | 0 | 1 masks at collect **and** eval. §2 |
| `SNEK_PPO_VALUE_LOSS` | `huber` | or `mse` |
| `SNEK_PPO_INIT` | `snek2` | or `orthogonal` |
| `SNEK_PPO_EVAL_ROLLOUTS` | 2 | evals (and checkpoints) every this many rollouts |
| `SNEK_DISCOUNT` | 0.9975 for the comparison | shared with DQN. §8 |

---

## 8. Where the risk actually is

| risk | why | what it looks like in the row | mitigation |
|---|---|---|---|
| **the +100 terminal is invisible to GAE** | at γ=0.9975, λ=0.95 the advantage horizon is `1/(1−γλ)` ≈ **19 steps**; the win is ~950 moves from the opening | perfect % flat 0 while avg score climbs | λ 0.98-1.0, and the shaping terms. The critic is the carrier — as it is for 1-step DQN, which reached 98.7% |
| **entropy collapse** | 3 actions, and nothing forces a floor once the bonus is outweighed | `entropy` falling toward 0 with perfect % stuck | the entropy coefficient, annealed rather than fixed if p0 shows it |
| **critic scale** | V is ~40 while a step reward is ~0.001 | `explained_variance` near 0 or negative | separate towers, huber, and `explained_variance` is the direct readout |
| **the refactor** | 350 lines of shared measurement machinery move | a DQN arm that differs from b2 in any digit | the bit-exactness gate in §5 |
| **γ, `PERFECT_GAME_REWARD` coupling** | invariant 6: finishing beats farming only when `W > 1/(1−γ^k)` — 40.5 at γ=0.9975 and k≈10, 10.4 at γ=0.99 | agents that farm and never finish | W=100 clears both, so **γ=0.99 is the safer end** on this axis and the riskier end on the one above. p0 tries both |

The first row is the one to take seriously, and it is the reason the tuning pass exists. The others
are all diagnosable from a single eval row, which is the point of putting the diagnostics there.

---

## 9. Tests, and the mutants that must die

Per `snek3/CLAUDE.md`: fixtures in the same pass as the logic, and **a passing suite is not coverage —
`tools/mutate.py` has to kill the mutants.**

| fixture | pins |
|---|---|
| GAE against a hand-computed 5-step example | the arithmetic, digit for digit |
| GAE with a `done` in the middle | **no advantage crosses an episode boundary.** The bug `dqn/collect.py` records snek2 shipping in its n-step window |
| λ=1, γ=1 | advantage == Monte-Carlo return − value |
| λ=0 | advantage == one-step TD error |
| **log-prob round trip** | recomputing the log-prob of the stored action with unchanged weights reproduces the stored value exactly, so the first minibatch's ratio is exactly 1.0. **The single best PPO bug detector** |
| the clipped objective's **gradient**, not its value | ratio > 1+ε with A>0 contributes zero gradient; ratio < 1−ε with A<0 likewise. A loss-value fixture passes with the wrong branch |
| entropy of a uniform 3-way policy | == ln 3 |
| advantage normalisation | mean 0, sd 1; and a size-1 minibatch does not produce NaN |
| the action mask | a masked action is never sampled, its log-prob is −inf, and it never enters the loss |
| rollout reuse | no transition appears in two rollouts; every transition appears in every epoch exactly once |
| `restore.ALGORITHMS['ppo']` | a PPO ckpt round-trips; a dqn sidecar cannot be loaded as ppo |
| **DQN bit-exactness across the refactor** | a fixed-seed 2,000-step `_evals.json`, byte-identical |
| the rejected knobs | `SNEK_FORK_BRANCHES=4` with `SNEK_ALGO=ppo` raises and names itself |

Mutants, all of which must be killed: drop `(1−done)` from the GAE recursion; drop it from the
bootstrap only; `min` → `max` in the clipped surrogate; drop the sign on the entropy term; use
`logits` where `log_softmax` is needed; reuse the previous rollout's values; skip advantage
normalisation; `t+1` → `t` in the value lookup; γ and λ swapped; drop the `.detach()` on the stored
log-prob. Ten, against phase 0's bar of twelve.

---

## 10. Phases and gates

| # | phase | gate |
|---:|---|---|
| **6a** | the `train.py` seam and `DqnAlgo`. No PPO code | a fixed-seed DQN arm is **byte-identical** before and after. Lands after b2 closes |
| **6b** | `ppo/` plus the fixtures and the mutant spec | suite green, 10 of 10 mutants killed, and a `smoke` arm learns *something* — avg score above the ~4 an untrained policy gets |
| **6c** | **batch p0 — tuning, 2-4 arms, 2M transitions each** | not a gate. Its output is a config: LR, entropy coef, λ, γ. Cheap enough (§4) to run more than four |
| **6d** | **batch p1 — 4 seed-matched arms at b2's transition budget (12M)** | phase 3's bar, restated: **one arm reaches ≥90% perfect in a stage-A eval.** Plus a measured transitions/s and wall clock |
| **6e** | stage B on p1, and the comparison | lead with the **≥98%/500 count** — the width of the record region — against b2's own close-out, and the sign test across the four seeds on `strong_eval_fraction` at a matched transition horizon |

**6c exists because b1 was gated on untuned defaults and answered nothing.** `docs/runs.md` says it
plainly: "b1 ran snek3's bare defaults and that was the wrong batch to gate on", and the five knobs b2
changed were the whole difference. PPO out of the box has *more* untuned knobs than DQN did, and §4
says an arm is cheap. Spending one short batch on a config is the same lesson applied one phase later.

**p1 runs b2's env config**, so the reward function is matched: `SNEK_CHASE_SAFE_SHAPING=0.1`,
`SNEK_CHASE_SAFE_GATE=75`, `SNEK_FOOD_DISTANCE_REWARD=0`, `SNEK_FC_LAYERS=320`, seed N pinned to arm
letter N. That makes p1 seed-matched against b1, b2, b29, b41 and b47.

**What a failure means, pre-registered.** If p1 misses ≥90% on every seed at 12M transitions,
the reading is not "PPO does not work here" — it is "PPO does not work here at DQN's sample budget",
and the follow-up is a single long arm at 50-100M transitions, which §4 says costs hours rather than
days. `docs/runs.md`'s backlog already names Munchausen-DQN and SAC-discrete as the fallbacks.

---

## 11. Open questions for the user

1. **γ for p1: 0.9975 (matches b2) or 0.99 (snek3's default, and safer on invariant 6)?**
   Recommendation: let p0 try both and let p1 use the winner, stating the γ beside every comparison.
2. **Action masking: is the `1` setting in scope at all?** Recommendation: p1 runs unmasked for a
   clean A/B, and a p2 runs masked as a candidate *better agent* rather than as a comparison.
3. **The `train.py` refactor, or a separate `train_ppo.py`?** Recommendation: the refactor, gated on
   byte-identical DQN output. A forked measurement path is the failure mode this whole plan is
   shaped to avoid.
4. **Retro-add `transitions` to DQN eval rows now, while b2 is running?** It is additive and
   `run_report` reads with `.get()`, so old rows stay readable — but it means new b2 rows carry a
   field its earlier rows do not. Recommendation: yes, after b2 closes.

## Backlog, once p1 has run

| idea | prior |
|---|---|
| **A PPO actor warm-started from the snek2 champion** | free: the champion's `30 -> 320 -> 3` weights load into `PolicyNet` unchanged, and Q-values of magnitude ~30 make a near-deterministic opening softmax. Answers "can PPO hold a policy DQN found" separately from "can PPO find one" |
| **Sampled evaluation** | measure π rather than argmax π. A different question, and one line |
| **Adaptive entropy on the eval history** | the `dqn/schedules.py` pattern applied to the entropy coefficient. Only if p0 shows a fixed coefficient is the binding constraint |
| **Wide-and-shallow arms** | PPO's cost is the env, and the env is 196k transitions/s at 1,024 lanes. Nothing in DQN could spend that |
