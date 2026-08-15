# Distributional RL — C51 (categorical DQN)

**Status:** proposed 2026-08-15, **awaiting review.** Nothing is implemented. A throwaway feasibility
probe has already been run against this repo's own environment and specs, and its results are in
[what the probe established](#what-the-probe-already-established) — they remove most of the
implementation risk and change two design choices from what a naive port would have done.

**One line:** replace the scalar `DdqnAgent` with a categorical (C51) agent that predicts a
*distribution* over the return on a fixed atom grid and trains it by cross-entropy, behind a
`SNEK_ALGO=c51` switch, leaving the environment, the rewards, the collector, the shield and the
epsilon schedule untouched.

Reading order for context: the record config this would be measured against is in
[`../hyperparamTuning/runs.md`](../hyperparamTuning/runs.md#record-status), and the reason a
value-representation change is on the table at all is that the post-peak collapse has no established
mechanism — the obvious candidate was ruled out in
[`findings.md`](../hyperparamTuning/findings.md#-falsified-2026-08-14-there-is-no-plasticity-loss--the-collapsed-networks-fit-a-new-target-better-than-their-own-peak).

## Why this, and what the honest prior is

C51's headline result (Bellemare et al. 2017) is a large median improvement on Atari at 200M frames
with pixel inputs and a conv net. **This project is none of those things** — a 30-value hand-built
observation, an MLP of a few tens of thousands of parameters, and useful learning finished by ~1.5M
steps. Transferring the *size* of that result here is not a reasonable expectation, and the plan
should not be sold on it.

The reason to run it anyway is narrower and fits what this project has actually measured:

**1. The problem here is instability, not ceiling.** Nine network shapes never raised the ceiling
(batch 20), and the selected column has been flat at 93-99% since batch 11. What separates a 94.2%
batch from a 97.6% one is whether a strong region *holds*. The most robust empirical claim in the
distributional literature is not higher final score, it is **reduced value-estimate variance and
gentler degradation** — the failure mode this folder keeps documenting as post-peak collapse. And that
collapse now has *no* mechanism: plasticity loss was ruled out directly (a collapsed net fits a new
target **better** than its own peak did), so the value estimate itself is one of the few candidates
left untried.

**2. The returns here are genuinely bimodal exactly where the game is decided.** From a packed
endgame state the return is approximately "win: the terminal +100 discounted" or "die: a small
negative". A scalar Q must regress the *mean* of those two, which is a value no outcome ever pays.
C51 represents both modes. This is the strongest a-priori argument and it is specific to this reward
structure, not a generic appeal to the paper.

**3. It is cheap, because `tf_agents` ships most of it.** `CategoricalDqnAgent`,
`CategoricalQNetwork`, `CategoricalQPolicy` and the distribution projection are all present in the
pinned 0.18.0. The work is a ~90-line subclass, a network builder, four `arch.json` fields, and one
agent-construction switch.

**4. Its loss has bounded gradients by construction**, which is the property `GRADIENT_CLIPPING=10`
was adopted to buy and failed to deliver — and the stated motivation for that arm was *exactly* "the
large terminal reward produces occasional huge gradients"
([falsified](../hyperparamTuning/findings.md), 1 of 3 seeds against 3 of 3). A cross-entropy over a
fixed support cannot produce a large gradient from a large target, because the target is a probability
vector. Clipping tried to fix the symptom outside the loss; this fixes it inside.

**5. It is orthogonal to the whole backlog, and the backlog is running dry at the top.** Every open
item is a reward term, an architecture width or a schedule. `CHASE_SAFE_SHAPING` has now returned
**null twice** — b27 (pooled 85.2 vs 87.9) and b30 (best-30 92.9 vs 93.6) — and batch 20 already
established that architecture never raises the ceiling. Nothing else queued changes the *loss*, so a
result here is not confounded with anything behind it, and nothing above it in the backlog is
currently promising.

**6. There is no prior art in this folder at all.** `c51`, `distributional`, `rainbow` and `quantile`
appear nowhere in the docs or code, and the agent class is hardcoded at `snek2.py:320` with no
`SNEK_*` knob — so unlike every batch since b1, this arm cannot be launched by environment variable
and needs the code change first. That is the reason it has never been tried, not a judgement that it
would not work.

**Falsifiable prior, pre-registered:** ~40% that C51 is a null on `best_perfect30` at a matched
horizon; ~35% that it is a modest improvement in *drawdown* (peak-to-later decline) with a null on
peak; ~15% a real improvement in both; ~10% that it is clearly worse, most likely through the
learning-rate/loss-scale interaction in [risks](#risks-in-the-order-they-are-likely-to-bite).

## What actually changes, mathematically

| | now (`DdqnAgent`) | with C51 |
|---|---|---|
| network head | 3 units, one Q per action | 3 × `num_atoms` logits |
| prediction | `Q(s,a)` scalar | `p(s,a)` over a fixed support `z_1..z_N` |
| target | `r + γ·d·max_a' Q'(s',a')` | project `r + γ·d·z` onto the support, weighted by `p'(s',a*)` |
| loss | element-wise Huber on the TD error | cross-entropy against the projected target |
| greedy action | `argmax_a Q(s,a)` | `argmax_a Σ_i z_i·p_i(s,a)` |
| PER priority | `\|td_error\|` | KL(target ‖ prediction) — see [priority signal](#the-priority-signal-kl-not-cross-entropy) |

Everything else is untouched: the observation, all rewards, `discount` coming from the environment's
time step (0.99 mid, **0.0 on `LAST`**, with agent `gamma=1.0`), the forking collector, the
exploration shield, the epsilon schedule, perfect-game counting, checkpointing, charts, and the eval
pipeline's episode logic.

## What the probe already established

Run 2026-08-15 against `SnakeEnvironment` and this repo's own `ShieldedEpsilonGreedyPolicy` and
`TrajectoryPrioritizedReplayBuffer` (throwaway script, scratchpad, nothing committed). All six are
verified facts, not expectations:

| question | answer |
|---|---|
| does `CategoricalQNetwork` build on this env's specs and match `build_q_net`'s initializers? | **yes** — `encoding_network`'s default kernel init is exactly `VarianceScaling(2.0, fan_in, truncated_normal)`, the same as `dense_layer()`. Head is `RandomUniform(±0.03)` with bias `-0.2` vs our `0.0`, which is a constant shift that cancels in the per-action softmax |
| does the exploration shield work over the categorical greedy policy? | **yes** — `agent.policy.info_spec == ()`, which is what `ShieldedEpsilonGreedyPolicy` requires, and it wrapped and ran |
| is the replay-buffer spec unchanged? | **yes** — `collect_data_spec` is identical, so a c51 arm can even warm-start from a ddqn arm's saved buffer |
| **does upstream's `_loss` honour the IS weights we pass?** | **no — it silently drops them.** `weights` is accepted and never used; skewing them 32× moved our loss and left upstream's bit-identical |
| **does upstream give PER a priority signal?** | **no** — it returns `DqnLossInfo(td_loss=(), td_error=())` with a `TODO(b/127318640)` in the source. `training.py`'s `signal.numpy()` would raise on an empty tuple |
| is a from-scratch reimplementation of the loss correct? | **yes, exactly** — our unweighted single-selection total equals upstream's to 6 decimals (4.706757 both) on the same batch |

Two things the probe changed in the design:

- **The loss must be overridden, not merely subclassed for convenience.** Two upstream defects
  (dropped IS weights, no priority signal) sit in the one method, and both are load-bearing here:
  PER is on in every arm, and `SNEK_IS_WEIGHTS` is a live knob. A vendored copy is not needed — a
  ~35-line reimplementation reproduces upstream's number exactly, which is a better test than
  trusting a copy.
- **The double-vs-single mutation is invisible at initialisation.** `agent.initialize()` syncs the
  target net to the online net, so double and single atom selection returned *identical* losses in
  the probe (4.706757 both). A test for double selection that does not first desync the target
  network passes for the wrong reason and its mutant survives.

Also measured: 200 `agent.train` steps at batch 128 in **1.33 s (150 train-steps/s)** with
`num_atoms=111` on `fc (50,100,50)`, so the loss step is not near the bottleneck (the desktop's arms
run the *whole* loop at ~92 steps/s). Parameter count on that shape: **28,683 vs 11,853** for ddqn.

## The one real design question: the support

C51 needs a fixed `[v_min, v_max]` grid, and **this env's return range is awkward** — the reward
quantum is `FOOD_REWARD=1`, `DEATH_REWARD=-5`, and a perfect game pays `PERFECT_GAME_REWARD=100` on
its terminal step. So:

- The state before the last food has a return of ~**101**, and with γ=0.99 the win bonus is still
  worth 20+ for the last ~160 steps of a won game — the endgame this project cares about most.
- Almost every other state's return sits in roughly **[-5, 25]**.

A grid that covers 101 therefore spends most of its atoms where nothing ever lands. Three ways out,
and the recommendation is the first:

| option | verdict |
|---|---|
| **cover the true range with enough atoms** — `[-6, 105]`, 111 atoms → spacing exactly 1.0 = one food | **recommended.** Atoms are cheap on this net, and `project_distribution` *interpolates* between neighbouring atoms, so the represented **mean is exact** even at coarse spacing — what coarse atoms cost is distributional detail, which is the thing C51 is supposed to buy |
| rescale the rewards so the range is small | **rejected.** `runs.md`'s "explicitly not planned" keeps `FOOD_REWARD`/`DEATH_REWARD`/`STARVE_REWARD`/`PERFECT_GAME_REWARD` fixed *because* they rescale `avg_reward`, which is what `BOOTSTRAP_REWARD_THRESHOLDS` fires on — a rescale would silently move the epsilon schedule and confound the arm |
| a nonlinear (signed-sqrt) support | not now. It is the principled answer to a 1-vs-100 dynamic range and it is ~10 lines, but it is nonstandard, and choosing it before measuring the return distribution would be guessing |

**Phase 0 measures the distribution rather than arguing about it** (below). Two loud guards ship with
the code either way: `v_max ≥ PERFECT_GAME_REWARD + FOOD_REWARD` and `v_min ≤ DEATH_REWARD +
STARVE_REWARD`, both refusing to start unless `SNEK_C51_ALLOW_CLIPPING=1`. Silent clipping of the win
value is exactly the class of failure `arch.json` exists to prevent.

## The priority signal: KL, not cross-entropy

Upstream returns nothing, so this is a free choice. **Use `KL(target ‖ prediction) = CE − H(target)`,
which is what the Rainbow paper uses**, not the raw cross-entropy.

The reason is measurable in the probe: at initialisation the per-example CE ran **4.59-4.71**, a
0.12-wide band around `ln(111) = 4.7095`. Most of that magnitude is the *irreducible* entropy of the
projected target — the projection spreads mass over two atoms, so `H(target) > 0` always — and an
additive near-constant compresses the relative spread that PER's `alpha` exponent acts on. That is
the same defect this repo already documented for `td_loss` as a priority (Huber's log-log slope
against `|δ|` measures 1.92-1.99, so `td_loss` at `alpha=0.6` is an effective exponent near 1.2), and
sharpness is [a variance dial, not a quality dial](../hyperparamTuning/findings.md) — a sharper
signal raised both the ceiling and the death rate.

**One thing is worse here than in the Huber case, which is why this is not a cosmetic choice.**
`td_loss` and `td_error` rank the *same* transitions — Huber is monotone in `|δ|`, top-1000 Jaccard
**1.0000** on 8 of 8 arms, so the signal only ever changed how much mass the top got. CE and KL differ
by `H(target)`, which is **not** monotone in KL, so the two genuinely *reorder* transitions. This
choice can change which experience gets replayed, not just how hard.

`SNEK_PRIORITY_SIGNAL` keeps its two existing values and both map to the KL for a c51 arm, so
**`training.py` needs no change at all** — the subclass returns the per-example KL in *both*
`td_loss` and `td_error`. `SNEK_PRIORITY_SIGNAL=ce` is added as the ablation.

## Design

### Knobs (all default to today's behaviour)

| knob | default | meaning |
|---|---|---|
| `SNEK_ALGO` | `ddqn` | `ddqn` or `c51`. Anything else exits at startup |
| `SNEK_NUM_ATOMS` | 111 | size of the support grid. Ignored unless `ALGO=c51` |
| `SNEK_V_MIN` / `SNEK_V_MAX` | -6.0 / 105.0 | support bounds, guarded as above |
| `SNEK_C51_DOUBLE` | 1 | 1 = action for the target atoms chosen by the **online** net (Rainbow-style double, matching today's `DdqnAgent`); 0 = upstream's target-net selection |
| `SNEK_C51_ALLOW_CLIPPING` | 0 | escape hatch for the two support guards |
| `SNEK_PRIORITY_SIGNAL` | `td_error` | gains `ce` as a third value, meaningful only for c51 |

`SNEK_C51_DOUBLE=1` is the default deliberately: the control is `DdqnAgent`, so upstream's
single-selection C51 would change *two* things at once. The knob keeps the confound measurable
instead of baked in.

### `arch.json` — four new fields, and the trap they close

A c51 checkpoint restored into a ddqn network is the exact failure `policy_arch.py` was written for:
the head is `3·N` wide instead of 3, and `expect_partial()` says nothing. Worse, `v_min`/`v_max` are
**part of the policy** — the greedy action is `argmax Σ z_i p_i`, so restoring correct weights against
the wrong support silently produces a different policy with no shape mismatch anywhere.

So `build_arch` gains `algo`, `num_atoms`, `v_min`, `v_max`; `assert_restorable` compares `algo`
(reading a missing field as `'ddqn'`, so every existing directory and every `hallOfFame/` entry keeps
loading untouched); `assert_config_matches` adds all four to the resume check alongside
`fc_layer_params`; and the support comes **from the file** at eval and watch time, never from the
environment — the same rule that removed the reliance on `SNEK_FC_LAYERS`.

### Edits, file by file

| file | change |
|---|---|
| **`categorical_agent.py`** *(new, ~90 lines)* | `SnekCategoricalDqnAgent(CategoricalDqnAgent)`. Overrides `_loss` (per-example KL in `extra`, `sample_weight=weights` into `aggregate_losses`) and `_next_q_distribution` (online-net argmax when `double`). `n_step_update > 1` raises at construction — the n-step target-support branch is ~10 more lines and no recent arm uses n>1 (n=3 measured null in batch 15, n=5 closed), so it is left out rather than shipped untested |
| **`under_the_hood.py`** | `build_categorical_q_net(num_actions, fc_layer_params, num_atoms)` beside `build_q_net`, wrapping `CategoricalQNetwork` so the two share one construction site and one set of initializers |
| **`policy_arch.py`** | the four fields, the `'ddqn'` default, the two extra assertions |
| **`snek2.py`** | the knobs, the two support guards, an `if algo == 'c51'` branch choosing net + agent, and `algo`/`num_atoms`/support in `run_config` so `runs/<policy>.md` records them |
| **`eval_agent.py`** | `build_eval_agent` dispatches on `arch['algo']`, building the categorical net and agent with the **recorded** support. Covers `eval_checkpoints.py` and `eval_workers.py`, which both call it |
| **`watch.py`** | currently builds its own `DdqnAgent` — a fourth construction site the module docstring in `eval_agent.py` claims does not exist. Route it through `build_eval_agent` rather than adding a second c51 branch |
| **`hyperparamTuning/perDiagnostics/*.py`** | `input_sensitivity_over_time.py`, `per_priorities.py`, `plasticity.py`, `plasticity_probe.py` call `agent._q_network` directly and would get `[B, 3, N]` logits. Add a shared `expected_q(net, obs, arch)` helper and use it, so a c51 arm is measurable rather than a shape error |
| **`training.py`** | **none** — that is the point of returning the signal in both `extra` fields |

### Tests — `tests/test_categorical_agent.py` plus additions to `test_policy_arch.py`

Each row names the mutation it is there to kill, per the "mutate the implementation and confirm a
test fails" rule.

| test | mutant it kills |
|---|---|
| unweighted single-selection loss equals upstream's on a fixed batch | any error in the reimplemented projection/CE — the check the probe already passes |
| skewed IS weights change our total; upstream's is unchanged | dropping `sample_weight=weights` (and it pins *why* the override exists) |
| `extra.td_error` is `[batch]`, finite, ≥ 0, and equals `extra.td_loss` | returning `()` again, or only filling one field, which would break `training.py` |
| KL is ~0 when prediction == target while CE is `H > 0` | reverting the priority to raw CE |
| **with a desynced target net**, the selected atoms are the online argmax's row | switching to target-net selection. **Must desync first** — see the probe finding, the mutant survives otherwise |
| loss at init ≈ `ln(num_atoms)` | a support/atom-count mismatch between net and agent |
| a `discount == 0` transition puts target mass at the reward, not spread over the support | anything that breaks the terminal-bootstrap contract `snake_environment.to_tensor_time_step` is built on |
| `v_max` below `PERFECT_GAME_REWARD` raises unless the escape hatch is set | dropping the clipping guards |
| the shield masks fatal exploration moves over a *categorical* greedy policy | a regression in `ShieldedEpsilonGreedyPolicy`'s `info_spec` contract |
| arch: missing `algo` reads as `ddqn`; c51-vs-ddqn mismatch raises; atoms/support mismatch fails a resume; round trip keeps all four fields | every silent-restore path this repo has already been bitten by twice |

Then the full suite: **24 modules** (up from 23), ~610 tests, 0 failed.

## Phases

| phase | work | gate to the next |
|---|---|---|
| **0** | `perDiagnostics/return_distribution.py`: play greedy episodes with a `hallOfFame/` champion and a mid-skill checkpoint, compute per-state discounted returns at γ=0.99, report percentiles, max, and the share above 25. Minutes to run | picks `v_min`/`v_max`/`num_atoms` from data. Diagnostics-only, so it pushes without review |
| **1** | the edits and tests above, plus mutation checks | full suite green; every mutant fails |
| **2** | `snek2.py smoke` with `SNEK_ALGO=c51 SNEK_MIN_CHECKPOINT_SCORE=0 SNEK_MAX_STEPS=3000`: trains, writes `arch.json` with `algo=c51`, checkpoints, `watch.py smoke` loads it, and a one-checkpoint eval measures it. Record steps/s against a matched ddqn smoke | it runs end to end on both hosts' code path. **Note the eval displaces every `evals/` PNG** (CLAUDE.md) — check what is at the top level first and restore afterwards |
| **3** | pilot: one wave of 4 arms, `{atoms 51, atoms 111} × {lr 1e-5, lr 5e-5}`, one seed each, `SNEK_MAX_STEPS=600000`. A **screen, not a result** — does it learn at all, how fast to the first perfect game, is the loss scale sane at 1e-5 | pick one config. If none reaches its first perfect game by ~300k while the control did, stop and report rather than sweeping |
| **4** | the batch: 4 seeds of the chosen config, **matched to `b24`** so `b24a-d` is the seed-matched control and no control arms need running | the pre-registered criteria below |

### The phase 3/4 launch line

`b24`'s config verbatim, plus the new knobs, one process per seed. The batch prefix is taken at launch
(next free is **`b31`**, but `fc 512` and the four owed `320` seeds are ahead of this in the backlog,
so do not reserve it):

```
cd snek2
SNEK_ALGO=c51 SNEK_NUM_ATOMS=<n> SNEK_V_MIN=<v> SNEK_V_MAX=<v> \
  SNEK_FC_LAYERS=320 SNEK_IS_WEIGHTS=0 SNEK_TARGET_UPDATE_PERIOD=1000 \
  SNEK_DISCOUNT=0.9975 SNEK_FOOD_DISTANCE_REWARD=0 SNEK_FORK_BRANCHES=4 \
  SNEK_MAX_STEPS=2000000 SNEK_SEED=<n> \
  PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u snek2.py b31<x>-c51seed<n> \
  > /tmp/b31<x>.log 2>&1 &
```

`SNEK_FORK_BRANCHES=4` is not optional and nothing in the summary block reports it, so an omission
differs from the control invisibly — the trap b27 nearly fell into. Confirm every arm with
`grep 'hyperparameter override:'` on its log, which should now include `ALGO`, `NUM_ATOMS`, `V_MIN`
and `V_MAX`.

**Host note.** The desktop is full (b28 running, b29 queued) and pushing to `ops` starts real work on
another machine, so it needs its own approval. The laptop has no trainers but is currently running
b30's four close-out evals; the pilot waits for that to finish rather than sharing 14 cores with it.

### Pre-registered criteria for phase 4

**Primary is `best_perfect30` at a matched truncation of 2M**, not `strong_eval_fraction`. That is the
2026-08-14 re-measurement talking: at b24's level `best_perfect30` has sd **0.67** against `sef`'s
**5.59**, so paired at n=4 it resolves ~**3.6 pp** where `sef` resolves ~21.3 pp, and it still has
3.8 pp of headroom. `sef` is reported alongside for continuity. **Never on `peak_trailing`** — all
four b24 arms sit exactly on its 95.00 cap; report the count of trailing-95.00 windows instead
(control: 7 · 22 · 10 · 17).

Secondary, and the one the mechanism actually predicts: **drawdown** — peak `pf30` minus the mean over
the 500k that follows it.

Decisive artifact: **the count of full-length ≥98%/500 rows** out of the close-out → HOF-500 chain.
The control is `1 · 1 · 0 · 0` across b24b/d/c/a. On the desktop that chain is automatic; **on the
laptop both passes are by hand**, as 4 parallel `eval_checkpoints.py` processes with `EVAL_WORKERS` ≥ 4.
Note also that a strong arm can finish with an empty HOF job, since that chain gates at 98 — `b25b`'s
plausible 97.2% was abandoned that way — so a gate-97 pass by hand is worth doing if the close-out
looks strong.

| outcome | reading |
|---|---|
| `best_perfect30` up with 4/4 seeds in the same direction | real. Close out at gate 95, then the HOF-500 pass, then promote anything that holds ≥98%/500 |
| `best_perfect30` null but drawdown better in ≥3/4 seeds | the predicted result. Worth 4 more seeds, since drawdown has never been a primary metric and its between-seed variance is unmeasured |
| both null | **C51 is closed for this task** — and given how strong the literature's prior is, that is a `findings.md` entry in its own right |
| clearly worse | check the learning-rate confound *once* (one arm at the pilot's other rate), then close |

Two power facts the reading has to respect. **`n=4` cannot resolve an effect below ~10 pp on `sef`**
(5 pp needs n≈17-37), which is why the criteria are written as "large effect or clear null" rather
than as a comparison of point estimates. And **pair seed by seed**: seed 2 or 4 has been the best arm
in **18 of 18** config waves (paired p=0.00005), so an unpaired mean comparison mostly measures the
seed draw.

## Risks, in the order they are likely to bite

1. **Learning rate.** Cross-entropy has nothing to do with the Huber TD loss's scale, so `1e-5` —
   tuned for the latter, and already flagged in the backlog as "very conservative" — may be simply
   wrong for c51. This is the most likely way a real effect gets read as a null, which is why phase 3
   screens two rates before any seed set is spent.
2. **PER interaction.** Priorities change meaning even with the KL fix, and `alpha=0.6` was chosen
   against `|td_error|`'s spread. Left at 0.6 for the first batch **on purpose** — one change at a
   time — with `alpha` as the named follow-up if the batch is ambiguous. Note the phase-4 control is
   `IS_WEIGHTS=0`, so the IS-weight path is not even exercised there; the fix matters for the default
   config and for not shipping a knob that silently does nothing.
3. **Head-size confound.** C51 multiplies the head by `num_atoms` — on `fc 320` with 111 atoms that
   is 116.8k parameters against ddqn's 10.9k. The licence to ignore it is batch 20's nine-shape sweep
   ("architecture never raises the ceiling") and the b24/b25/b26 finding that the lift tracks the
   *widest layer*, not the parameter count — but it is a confound, it should be stated in the write-up,
   and it is a second reason to prefer fewer atoms if phase 0 permits.
4. **Support mis-specification.** Covered by the two startup guards and by phase 0.
5. **A reward-derived quantity breaking silently.** This plan touches no reward, but it does touch the
   terminal bootstrap, and the last thing to touch that class cost eight arms 300k+ steps each: the
   perfect-game counter compared a final reward with `PERFECT_GAME_REWARD`, read 0%, and — because
   `training.epsilon_for` uses the trailing perfect rate as its skill signal — **pinned epsilon at its
   ceiling**, so the runs were handicapped as well as mismeasured. The `discount == 0` test above is
   there for that reason, and phase 2's smoke check should confirm `perfect_percent` is non-zero on an
   arm that fills a board.
6. **`tf_agents` 0.18.0 internals.** The subclass depends on `_support`, `_num_atoms`,
   `_next_q_distribution` and `project_distribution`. The equal-to-upstream test is the tripwire: if
   an upgrade changes the projection, that test fails loudly instead of the arm training subtly wrong.

## What this is not

- **Not Rainbow.** No noisy nets, no duelling head, no multi-step. Those are separate arms and
  bundling them would make the result unattributable, which is the mistake batch 10 is still paying for.
- **Not QR-DQN/IQN, yet.** Quantile regression needs no `[v_min, v_max]` at all, which fits this env's
  1-vs-100 reward range better than any fixed grid — and if phase 0 says the support is the binding
  constraint, it is the right follow-up. It is ~150 lines with no `tf_agents` support, against C51's
  ~90 with most of the machinery already shipped, so C51 goes first.
- **Not a reward change.** The support is chosen to fit the existing rewards, never the reverse.
