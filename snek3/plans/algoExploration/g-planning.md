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
| the protocol | a policy may be a `StatefulPolicy` (E's `begin(rows, fresh)`) **and** may declare `needs_state = True`. For such a policy the engine calls `act(obs, state)` where `state` is `vec.snapshot(rows)`: the `(bodies, lengths, dirs, foods, steps, last_food_steps, scores)` tuple `VecSnake.set_state` already takes, for those rows. `vectorized/` gains `snapshot` beside `set_state`; both are numpy and it imports no torch |
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
| `algos/mcts/search.py` | the batched PUCT: a tree per lane as arrays (children by action, N, W, P), selection by the AlphaZero pseudocode's rule `(log((N + 19652 + 1) / 19652) + 1.25) · P · √N_parent / (1 + N_child)` (`SNEK_MCTS_C_INIT` 1.25, `SNEK_MCTS_C_BASE` 19652), expansion of all leaves in one network call, evaluation of the leaf value, backup with the discount. Dirichlet noise at the root in training (`SNEK_MCTS_DIRICHLET_ALPHA`, ε 0.25), none at eval. **α scales inversely with the move count**: the papers use 0.3 for chess's ~35 moves, 0.03 for Go's ~250; Snake has 3, so the paper's rule gives **α ≈ 3** (`10 / moves`, the AlphaZero heuristic), and 0.3 is the local value |
| `algos/mcts/net.py` | `QNet`'s trunk, a policy head (logits over 3 actions) and a value head; **the policy head alone is what `restore` builds for the network-only read**, and it is `algos/ppo/net.py`'s actor shape, so a PPO checkpoint can seed it (`SNEK_INIT_FROM`) |
| `algos/mcts/collect.py` | self-play over `collect_envs` lanes: search at every step, act by sampling the visits with temperature `SNEK_MCTS_TEMPERATURE` (1.0 for the first `SNEK_MCTS_TEMP_STEPS` moves of an episode -- the papers' 30 is 30 of ~150 moves; here 300 of ~1,500, the same fraction -- then 0), store `(obs, visits, reward)`, fill in the value target at episode end. **AlphaZero's target is the game outcome**; on a 2,300-move game with a discount that is one number per episode, so the plan uses MuZero's n-step bootstrap from the value head, `SNEK_MCTS_TD_STEPS` 10 (MuZero Atari's), and states the departure -- an AlphaZero cell with the pure outcome (`td_steps` = ∞) runs beside it |
| `algos/mcts/agent.py` | cross-entropy of the policy head against the visits, MSE on the value (the papers'), equal weights, SGD momentum 0.9 with weight decay 1e-4 and the papers' step schedule (`SNEK_MCTS_OPTIMIZER` `sgd`; `adam` is the local variant) |
| `algos/mcts/algo.py` | `mcts`: `build_config` with `SNEK_MCTS_SIMULATIONS` (**800**, AlphaZero's; MuZero Atari's 50 is the second cell), the knobs above, `SNEK_MCTS_REPLAY_SIZE` (the most recent games, in positions -- AlphaZero's window is 1e6 games), `SNEK_REPLAY_RATIO`; `policy_fn` is the search at eval simulations with temperature 0 |

**The value target on a game this long.** A perfect game is ~1,000 moves for the champions. Undiscounted,
as AlphaZero has it, the outcome is the total reward -- ~95 meals plus the +100 win, or the meals so far
minus 5 at a death -- so the value head is asked for the *score*; at MuZero's 0.997 (horizon ~333) the
return-to-go is the next few hundred moves' meals plus most of a win that is near. The plan uses the reward preset unchanged so the row is comparable, and records the
value target's scale in the smoke; if the value head cannot separate a doomed board from a safe one at
that scale, `SNEK_MCTS_VALUE_RESCALE` (R2D2's h, which is also MuZero's) is the knob. Q in the tree is
min-max normalised over the tree as MuZero does, since the papers' ±1 value range is the one thing
their PUCT constants assume.

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
| `algos/muzero/model.py` | h, g, f as MLPs over the latent width `SNEK_MUZERO_LATENT` (128), the latent min-max scaled to [0, 1] as the paper; reward and value as categorical over an **integer support** with the h(x) rescaling (ε 0.001): the paper's 601 atoms over [−300, 300] on Atari; here the rescaled return range is about [−1.5, 10] plus the linear term, so **`SNEK_MUZERO_SUPPORT` 21 atoms over [−10, 10]** brackets it with margin, and the smoke asserts it |
| the gradient | each unrolled step's loss scaled by 1/K and the gradient into the dynamics function's hidden state scaled by ½, as the pseudocode's `scale_gradient` |
| `algos/muzero/search.py` | G1's `search.py` with the simulator calls replaced by `g`; the same arrays, the same PUCT, min-max normalised Q. One file, a `model` argument -- the real simulator is a model too, and G1's search is written against that interface so G2 does not fork it |
| `algos/muzero/agent.py` | the K-step loss (K 5): policy CE, value CE, reward CE at each unrolled step, value weight 0.25 (the Reanalyze setting), TD steps 10, priorities \|ν − z\| with α = β = 1; reanalyse fraction `SNEK_MUZERO_REANALYSE` (0.8, the paper's "80% of updates use a fresh search's policy target") |
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
simulations) and search-based value estimation. Read against G2 and B1: whether it is the search or
the sample-efficiency tricks that carry it at low step counts.

| module | contents |
|---|---|
| `algos/muzero/` | the same package with `NAME` `ezv2`: `SNEK_EZ_CONSISTENCY` (2.0) adds the latent consistency loss, `SNEK_EZ_VALUE_PREFIX=1` swaps the reward head for the prefix LSTM (hidden 512 in the paper; `SNEK_EZ_PREFIX_HIDDEN` 128 over this latent), `SNEK_EZ_GUMBEL=1` selects the Gumbel root, `SNEK_MUZERO_SIMULATIONS` 16 (V2's Atari value), value weight 0.25, policy entropy 5e-3, TD steps 5 with `td_lambda` 0.95, the **mixed value target** (n-step TD for young transitions and early training, the search-based value otherwise; `SNEK_EZ_MIXED_T1` 40k updates, `SNEK_EZ_MIXED_T2` 20k transitions), target net every 400 updates, priorities α = β = 1 |
| `algos/muzero/gumbel.py` | sequential halving over the root's actions with Gumbel-perturbed logits, the `σ(q) = (c_visit + max_b N(b)) · c_scale · q` transform at c_visit 50, c_scale 0.1 (the V2 repo's Atari value, not 1), and the completed-Q improved policy as the target. Snake has 3 actions, so the paper's K = 8 sampled actions is every action: sequential halving over 3 with 16 simulations. New |

Tests: sequential halving over 3 actions with 16 simulations visits the documented counts; the
consistency loss is zero when g is the identity on a repeated observation; the value prefix on a
constant-reward sequence equals K × r. Mutants: the halving keeping the wrong half, the consistency
target not stop-gradiented, the prefix reset not on episode boundary.

**The budget question G3 is for.** G3 runs at B1's short cap (500k steps) and again at G2's cap. B1
was chosen as the value-based probe of the same regime, so B1 against G3 at the short cap is the
"search or tricks" reading.

### G4 -- Muesli (Hessel et al. 2021)

A policy-gradient method that uses MuZero's one-step model for a regularised policy update rather
than for search at play time: the policy is trained toward a clipped-MPO target built from the
model's action values, with the model, value and policy learned jointly by the MuZero-style unrolled
loss. Acts with the network alone. Read against G2 and against PPO: whether the model improves the
*update* when nothing searches at act time.

| module | contents |
|---|---|
| `algos/muesli/` | `agent.py`: the CMPO target π_CMPO ∝ π_prior exp(clip(advantage / σ_adv, −c, c)) with c 1.0 and the advantage normalised by a moving std (β_var 0.99, ε 1e-12), computed **exactly over the 3 actions** (the paper samples 16 actions on Atari; with 3 the expectation is exact, which is also what its large-scale run does); the policy loss = PG term with Retrace advantages (λ 0.95) + λ_CMPO 1.0 × KL to the CMPO target, entropy bonus 0; the model loss shared with `algos/muzero/agent.py` at K 5, value weight 0.25, reward 1.0; a target network at update rate 0.1; AdamW 3e-4 decayed to 0, weight decay 0; `algo.py` `muesli` with `SNEK_MUESLI_CLIP`, `SNEK_MUESLI_KL_WEIGHT`, `SNEK_MUESLI_REPLAY_FRACTION` (0.75), `SNEK_MUESLI_TARGET_RATE`; `policy_fn` is argmax over the policy head, stateless, `needs_state = False` |
| the step | as the paper's: each batch is 96 sequences of 30 steps, **75 % from replay and 25 % fresh** on-policy; `advance()` collects `collect_envs` × 30 moves, appends them to a replay of 6M positions' worth (scaled to the cap), and takes one update per collection |

Tests: the CMPO target with zero advantages is the current policy; the clip is inert inside its bound;
the policy loss gradient points toward the higher-advantage action on a two-action example. Mutants:
the exp taken before the clip, the normaliser dropped, importance weights unclipped.

## 2b. The papers' settings, and how each lands here

| setting | AlphaZero (preprint; pseudocode) | MuZero Atari (App. B, D, F, G; pseudocode) | EfficientZero V2 Atari (Table 3; repo) | Muesli Atari (Table 5) | here |
|---|---|---|---|---|---|
| simulations | 800 | 50 | 16 | -- (acts with the network) | G1 800 and 50; G2 50; G3 16 |
| PUCT | c_init 1.25, c_base 19652 | c1 1.25, c2 19652, min-max Q | Gumbel root, c_visit 50, c_scale 0.1 | -- | the paper's, per row |
| root noise | Dir(α) ε 0.25, α by move count (0.3 chess, 0.03 Go) | Dir(0.25), ε 0.25 | Dir(0.3), ε 0.25 | -- | α ≈ 3 for 3 moves in the paper cell; 0.3 local |
| temperature | ∝ visits for 30 moves, then argmax | 1.0 → 0.5 → 0.25 by training progress, whole game | 1 → 0.5 at 50 % → 0.25 at 75 % | -- | G1 the AlphaZero form (300 moves); G2, G3 the MuZero schedule |
| unroll K, TD steps | -- | 5, 10 | 5, 5 (λ 0.95) | 5, Retrace | per row |
| optimiser | SGD m 0.9, lr 0.2 → 0.0002 in 3 steps, wd 1e-4, batch 4096 | SGD m 0.9, lr 0.05 · 0.1^(step / 350k), wd 1e-4, batch 1024 | SGD m 0.9, lr 0.2, wd 1e-4, grad norm 5, batch 256 | AdamW 3e-4 → 0, wd 0 | the paper's optimiser per row; **batch 256** for all four (the papers' 1024-4096 are for 350 actors' throughput; 256 is EZ's and fits the lanes) |
| value / reward | MSE, ±1 | categorical 601 over ±300, h(x) ε 0.001, value weight 0.25 (Reanalyze) | categorical over ±300 (51 bins in the repo), value 0.25, consistency 2.0, entropy 5e-3 | categorical, value 0.25, reward 1.0 | G1 MSE on the unrescaled return; G2-G4 categorical 21 over ±10 on the rescaled return |
| replay, priority | last 1e6 games, uniform | 125k sequences of 200; \|ν − z\|, α = β = 1 | 1e6 transitions FIFO; α = β = 1 | 6M frames, 75 % replay | scaled to the cap, per row |
| reanalyse | -- | 80 % | 1.0 (repo) | -- | per row |
| target network | -- | acting checkpoint every 1,000 updates | every 400 updates | rate 0.1 | per row |
| discount | 1 (game outcome) | 0.997 | 0.997 | 0.995 | **the paper's, per row**: G1's AlphaZero cell undiscounted (the outcome; its TD-10 cell at MuZero's 0.997), G2 and G3 0.997, G4 0.995 |
| budget | 700k updates, 5,000 actors | 20B frames (Reanalyze 200M), 350 actors | 100k env steps, update-to-data 1 | 200M frames | G1 and G2 the reference's 100M transitions; G3 100k moves, then 500k; G4 50M moves |

## 3. The batches

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| G1 | 4 seeds of `mcts` at **800** simulations (AlphaZero's, §2b: SGD, MSE value, α ≈ 3, TD ∞) + 4 seeds at **50** (MuZero Atari's search budget with the TD-10 target), measured **both ways** (§5) | the PPO reference's reward preset, `hist8`, `SNEK_FC_LAYERS=320`; all fresh, as the papers -- a `SNEK_INIT_FROM` PPO trunk is the local variant and runs only if the fresh cells do not learn a prior | PPO `hist8`, the fixed-path references | stage-B density and the depth passes **for the network alone**; the same for the search at eval simulations; steps per perfect game via `tools.fixed_path --policy` beside the references |
| G1 simulations | the best G1 seed's checkpoint measured at 1, 10, 50, 200, 800 simulations | G1 | G1 | the perfect rate as a function of the search budget: where the network alone ends and the search begins to pay |
| G2 | 4 seeds of `muzero` at G1's cap | G1's | G1 | as G1, network-alone and searched |
| G3 | 4 seeds of `ezv2` at 100k moves (the paper's regime) + 4 at 500k; then 4 at G2's cap if either moved | G2's | B1 at 100k and 500k; G2 | as G1 |
| G4 | 4 seeds of `muesli` | G2's | PPO `hist8`, G2 network-alone | as PPO |

**Registered prediction (the agent's, 2026-09-16).** The G1 search at 50 simulations plays perfect
games at a rate above every HOF entry within a fraction of the reference cap, because the simulator
refuses the fatal move the champions die of and the value prior only has to be right about which
branch is *doomed*, not which is optimal. The G1 network alone is level with PPO or below it. G2 trails
G1 searched and matches G1 network-alone; G3 at the short cap beats B1; G4 is level with PPO. The
result that would matter: G1 network-alone *above* PPO, which would say search-generated targets are
a better teacher than the policy gradient for this game even when no search runs at play time.

## 4. Gates

1. The seam change lands first with its tests; every existing checkpoint measures identically.
2. G1 smoke with `SNEK_MCTS_SIMULATIONS=4`; the log reports the search's moves per second, which sets
   the cap for the spec. A search below ~200 lane-moves/s on the laptop is redesigned before it is
   queued (batch the leaf evaluations wider, or cut simulations), since a 2,300-move game at 800
   simulations is the budget item -- **and 800 is the number most likely to fail this gate.** If it does,
   the 800 cell runs at whatever the gate allows and says so; the 50 cell is the one the paper's Atari
   form supports anyway.
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
