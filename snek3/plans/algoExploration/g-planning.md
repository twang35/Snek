# Group G: planning -- AlphaZero-style MCTS, MuZero, EfficientZero V2, Muesli

**Status: planned 2026-09-16, nothing built.** Group G of [`algorithm-series.md`](algorithm-series.md);
conventions in [`README.md`](README.md). Phase 7 of the running order, all four rows in one phase.

The question: does planning with the true simulator win outright? The transition is deterministic
except the food spawn, the simulator is exact and cheap, and the fixed-path references already show
that lookahead plays perfect games (100.00% / 30,000). G1 searches with the real game and a learned
policy and value prior; it is the row in the whole series most likely to beat the record. G2-G4 then
ask what it costs to *learn* the model instead of having it, and whether the model helps the update
even without search at play time.

## 1. The seam change this group needs, designed once

A search needs the **board**, not the 26-value observation the policy seam hands over: it has to
clone a state, step it, and read the result. The engine hands `policy_fn` only `obs[rows]`.

| decision | rule |
|---|---|
| the protocol | a policy may be a `StatefulPolicy` (D's `begin(rows, fresh)`) **and** may declare `needs_state = True`. For such a policy the engine calls `act(obs, state)` where `state` is `vec.snapshot(rows)`: the `(bodies, lengths, dirs, foods, steps, last_food_steps, scores)` tuple `VecSnake.set_state` already takes, for those rows. `vectorized/` gains `snapshot` beside `set_state`; both are numpy and it imports no torch |
| where the search runs | inside the policy, on **its own `VecSnake`** of width `rows × simulations`: it `set_state`s every simulation lane from the snapshot, steps the batch, and reads rewards and terminals. The engine's env is never touched. This is what makes the search batched -- every lane's every simulation is one vectorised step -- and it is the reason the search is affordable at all |
| the food | the spawn is stochastic. Search treats it as a chance node with **one sample per simulation** (the simulator's own draw under the policy's private seed), which is the standard MCTS answer and is what the tour reference's safety argument does not need but the value estimate does. `forced_food` on `VecSnake.step` exists for the parity tests and is the hook if a determinised search is wanted for the ablation in §3 |
| the sidecar | `arch.json` gains `search`: `{"type": "mcts", "simulations": 50, "c_puct": 1.25}` for G1, `{"type": "muzero", ...}` for G2-G3, none for G4 (Muesli acts with the network). In the signature. The **network alone** is also loadable from the same checkpoint (§5) |
| training | self-play. The collector *is* the search policy over `collect_envs` lanes, storing per step the observation, the search's visit distribution, and later the game's outcome; the update fits the policy head to the visits and the value head to the outcome. `train.py` is unchanged: `advance()` is one self-play step over the lanes plus the gradient steps the replay ratio allows |
| the step | one self-play step of `collect_envs` lanes: `step_granularity` 1, `advance()` returns `(1, collect_envs)`. **A step here is `simulations` times more expensive than any other row's**, and the cap is set in the spec accordingly |

**Tests for the seam**: `snapshot` then `set_state` on a fresh env reproduces the board to the cell,
including a coiled endgame board from the parity fixtures; a policy that returns the tour's move from
the snapshot alone plays a perfect game through `engine.measure`; the layering probe finds no torch.

## 2. The rows

### G1 -- AlphaZero-style MCTS on the real `Game` (Silver et al. 2017, single-agent form)

PUCT search over the real simulator with a network giving a policy prior and a value per state; the
root's visit distribution is the acting policy and the training target; the game's outcome is the
value target. Single-agent, so no opponent and no minimax: the value is the discounted return-to-go,
and the terminal is death, starvation or the perfect game.

| module | contents |
|---|---|
| `algos/mcts/search.py` | the batched PUCT: a tree per lane as arrays (children by action, N, W, P), selection by PUCT with `c_puct`, expansion of all leaves in one network call, evaluation of the leaf value, backup. Dirichlet noise at the root in training (`SNEK_MCTS_DIRICHLET_ALPHA` 0.3, ε 0.25), none at eval |
| `algos/mcts/net.py` | `QNet`'s trunk, a policy head (logits over 3 actions) and a value head; **the policy head alone is what `restore` builds for the network-only read**, and it is `algos/ppo/net.py`'s actor shape, so a PPO checkpoint can seed it (`SNEK_INIT_FROM`) |
| `algos/mcts/collect.py` | self-play over `collect_envs` lanes: search at every step, act by sampling the visits with temperature `SNEK_MCTS_TEMPERATURE` (1.0 for the first `SNEK_MCTS_TEMP_STEPS` moves of an episode, then 0), store `(obs, visits, reward)`, fill in the value target at episode end (n-step bootstrap from the value head for long games, `SNEK_MCTS_TD_STEPS` 200, since a 2,300-move game's pure Monte-Carlo target is one number per episode) |
| `algos/mcts/agent.py` | cross-entropy of the policy head against the visits, Huber on the value, weight decay |
| `algos/mcts/algo.py` | `mcts`: `build_config` with `SNEK_MCTS_SIMULATIONS` (50), `SNEK_MCTS_C_PUCT` (1.25), the knobs above, `SNEK_MCTS_REPLAY_SIZE` (a game-level buffer, in positions), `SNEK_REPLAY_RATIO`; `policy_fn` is the search at eval simulations with temperature 0 |

**The value target on a game this long.** A perfect game is ~1,000 moves for the champions. With γ =
0.99 the return-to-go is dominated by the next few meals plus the +100 at the end, and the death −5 is
small against it. The plan uses the reward preset unchanged so the row is comparable, and records the
value target's scale in the smoke; if the value head cannot separate a doomed board from a safe one at
that scale, `SNEK_MCTS_VALUE_RESCALE` (R2D2's h) is the knob.

Tests: PUCT on a hand-built two-action tree picks the documented child; the backup adds the value to
every ancestor's W and N exactly once; the root visits sum to `simulations`; a search with `simulations`
= 1 acts on the prior; on a board where one action is an immediate death the visit distribution puts
<5% on it after 50 simulations with a uniform prior and a zero value head (the simulator's terminal is
enough). Mutants: the value backed up without the discount, the child's Q read as W instead of W/N,
Dirichlet noise applied at eval.

### G2 -- MuZero (Schrittwieser et al. 2020)

The same search with a **learned** model: a representation network h(obs) → s, a dynamics network
g(s, a) → (r, s'), and a prediction network f(s) → (π, v). Search runs in latent space; targets are
K-step unrolled (K = 5) from replayed positions, with the reanalyse of stored positions' policy
targets by a fresh search. Read against G1: what learning the model costs when the true one was free.

| module | contents |
|---|---|
| `algos/muzero/model.py` | h, g, f as MLPs over the latent width `SNEK_MUZERO_LATENT` (128); reward and value as categorical over a support (Group A's `Categorical` head at a support sized to the reward), with the h(x) rescaling |
| `algos/muzero/search.py` | G1's `search.py` with the simulator calls replaced by `g`; the same arrays, the same PUCT, min-max normalised Q. One file, a `model` argument -- the real simulator is a model too, and G1's search is written against that interface so G2 does not fork it |
| `algos/muzero/agent.py` | the K-step loss: policy CE, value CE, reward CE at each unrolled step, gradient scaled by 1/K; reanalyse ratio `SNEK_MUZERO_REANALYSE` (1.0) |
| `algos/muzero/algo.py` | `muzero`; `policy_fn` is the search in latent space at eval simulations, `needs_state = False` -- **MuZero acts from the observation alone**, which is what makes it comparable to the network-only rows and is one of the reasons it is in the series |

Tests: the unrolled loss at K = 1 equals a one-step loss; the learned reward on a scripted transition
reaches the true reward on a small fit; the latent search with a *perfect* learned model (a test double
whose `g` calls the simulator) reproduces G1's root visits on the same board. Mutants: the gradient
scale dropped, the unrolled action off by one, the value target bootstrapped from the wrong step.

### G3 -- EfficientZero V2 (Wang et al. 2024)

MuZero for the low-data regime: a self-supervised consistency loss between the predicted latent
g(h(o_t), a) and h(o_{t+1}), a value-prefix head predicting the summed reward over the unroll with
an LSTM, off-policy correction of stale value targets by reanalysing with a fresh search, and in V2
**Gumbel search** (sequential halving with Gumbel noise at the root, which acts well with few
simulations) and search-based value estimation. Read against G2 and F1: whether it is the search or
the sample-efficiency tricks that carry it at low step counts.

| module | contents |
|---|---|
| `algos/muzero/` | the same package with `NAME` `ezv2`: `SNEK_EZ_CONSISTENCY` (2.0) adds the latent consistency loss, `SNEK_EZ_VALUE_PREFIX=1` swaps the reward head for the prefix LSTM, `SNEK_EZ_GUMBEL=1` selects the Gumbel root, `SNEK_MUZERO_SIMULATIONS` 16 (V2's few-simulation regime) |
| `algos/muzero/gumbel.py` | sequential halving over the root's actions with Gumbel-perturbed logits and the completed-Q improved policy as the target. New |

Tests: sequential halving over 3 actions with 16 simulations visits the documented counts; the
consistency loss is zero when g is the identity on a repeated observation; the value prefix on a
constant-reward sequence equals K × r. Mutants: the halving keeping the wrong half, the consistency
target not stop-gradiented, the prefix reset not on episode boundary.

**The budget question G3 is for.** G3 runs at F1's short cap (500k steps) and again at G2's cap. F1
was chosen as the value-based probe of the same regime, so F1 against G3 at the short cap is the
"search or tricks" reading.

### G4 -- Muesli (Hessel et al. 2021)

A policy-gradient method that uses MuZero's one-step model for a regularised policy update rather
than for search at play time: the policy is trained toward a clipped-MPO target built from the
model's action values, with the model, value and policy learned jointly by the MuZero-style unrolled
loss. Acts with the network alone. Read against G2 and against PPO: whether the model improves the
*update* when nothing searches at act time.

| module | contents |
|---|---|
| `algos/muesli/` | `agent.py`: the CMPO target π_CMPO ∝ π exp(clip(advantage / normaliser)), the policy loss (CE to that target plus the PG term with importance weights, from replay with retrace-style correction), the model loss shared with `algos/muzero/agent.py`; `algo.py` `muesli` with `SNEK_MUESLI_CLIP` (1.0), `SNEK_MUESLI_MODEL_WEIGHT`; `policy_fn` is argmax over the policy head, stateless, `needs_state = False` |
| the step | PPO-like: `advance()` collects `collect_envs` moves then updates from a replay of recent sequences |

Tests: the CMPO target with zero advantages is the current policy; the clip is inert inside its bound;
the policy loss gradient points toward the higher-advantage action on a two-action example. Mutants:
the exp taken before the clip, the normaliser dropped, importance weights unclipped.

## 3. The batches

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| G1 | 4 seeds of `mcts`, 50 simulations, measured **both ways** (§5) | the PPO reference's reward preset, `hist8`, `SNEK_FC_LAYERS=320`; `SNEK_INIT_FROM` a PPO `hist8` checkpoint's trunk for two of the seeds, fresh for the other two | PPO `hist8`, the fixed-path references | stage-B density and the depth passes **for the network alone**; the same for the search at eval simulations; steps per perfect game via `tools.fixed_path --policy` beside the references |
| G1 simulations | the best G1 seed's checkpoint measured at 1, 10, 50, 200 simulations | G1 | G1 | the perfect rate as a function of the search budget: where the network alone ends and the search begins to pay |
| G2 | 4 seeds of `muzero` at G1's cap | G1's | G1 | as G1, network-alone and searched |
| G3 | 4 seeds of `ezv2` at 500k, 4 at G2's cap | G2's | F1 at 500k; G2 | as G1 |
| G4 | 4 seeds of `muesli` | G2's | PPO `hist8`, G2 network-alone | as PPO |

**Registered prediction (the agent's, 2026-09-16).** The G1 search at 50 simulations plays perfect
games at a rate above every HOF entry within a fraction of the reference cap, because the simulator
refuses the fatal move the champions die of and the value prior only has to be right about which
branch is *doomed*, not which is optimal. The G1 network alone is level with PPO or below it. G2 trails
G1 searched and matches G1 network-alone; G3 at the short cap beats F1; G4 is level with PPO. The
result that would matter: G1 network-alone *above* PPO, which would say search-generated targets are
a better teacher than the policy gradient for this game even when no search runs at play time.

## 4. Gates

1. The seam change lands first with its tests; every existing checkpoint measures identically.
2. G1 smoke with `SNEK_MCTS_SIMULATIONS=4`; the log reports the search's moves per second, which sets
   the cap for the spec. A search below ~200 lane-moves/s on the laptop is redesigned before it is
   queued (batch the leaf evaluations wider, or cut simulations), since a 2,300-move game at 50
   simulations is the budget item.
3. The mutation specs kill every mutant.
4. G2-G4 wait for G1's closed number.

## 5. The eval decision: the search is not the network

The eval protocol measures a policy network. G1's *agent* is a search; the network alone is a
different, weaker policy, and both are legitimate results with different claims attached. The decision,
in the shape `a-return-tail.md` §5 set:

- **`arch.json` describes the checkpoint; `search` in it says a search is available, not that one is
  used.** The default `restore` read is the **network alone**, so stage A (which feeds nothing here but
  must stay cheap), `watch.py` and every shard behave as for any checkpoint, at ordinary cost.
- **A pass names the read**: `--policy-variant search:50` selects the searched policy at that
  simulation count; the result file carries the variant and is named beside the network-alone one.
  Stage B, `hof5000` and `hof30k` run for both variants of a G1 checkpoint, and the searched passes are
  budgeted at the moves-per-second the smoke measured.
- **The HOF row says which.** A searched record is a record of a different kind -- as the fixed-path
  rows are, and they already carry a "no policy" form -- so the row carries `search:50` in its name and
  the network-alone row, if it qualifies, sits beside it. The steps-per-game column comes from
  `tools.fixed_path --policy` for both.

## 6. What would change the plan

- **G1 searched beats the record.** Expected. The follow-on question is the simulation ladder: how few
  simulations still beat it, which is the cost of the guarantee.
- **G1 network-alone beats PPO.** The teacher, not the search, is the lever; every earlier group's
  best row is worth re-running with search-generated targets (a distillation batch), and G4 becomes
  the most interesting row rather than the least.
- **G1 cannot learn a useful prior at all** (the search plays well but the network stays at the
  policy-gradient level or below). Then the search is doing the work alone, which the fixed-path
  references already do more cheaply, and the group's finding is that planning here is a
  hand-written algorithm's job.
- **G2 matches G1.** The model is learnable from 26 features and the exact simulator bought little;
  G3's tricks and G4's update are then read on their own terms.
