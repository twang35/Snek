# Group F: data efficiency -- BBF

**Status: planned 2026-09-16, nothing built.** Group F of [`algorithm-series.md`](algorithm-series.md);
conventions in [`README.md`](README.md). Phase 1 of the running order, beside A1, because its answer
sizes every later row's budget.

The question: how much compute does any of this need? Every PPO reference arm runs to 100M transitions
over ~8 h with two-thirds of that in stage A. BBF is the value agent tuned for the 100k-step Atari
regime -- a bigger network, periodic parameter resets, and n-step and γ annealed within each reset
cycle -- and its result here says whether Snake at 26 features is anywhere near as expensive as the
series has been budgeting for. It also produces the row the whole Atari "data-efficient" family is
read against, so EfficientZero V2 (G3) has a comparison on this game.

## 1. The row

### F1 -- BBF (Schwarzer et al. 2023, "Bigger, Better, Faster")

Rainbow-style DQN at a high replay ratio, with: a wider network (Impala CNN at width ×4 on Atari, a
2048-wide hidden layer), **shrink-and-perturb resets** of the encoder every `reset_interval` gradient
steps (parameters interpolated half-way toward a fresh initialisation) with the layers after it reset
fully, **n-step annealed** 10 → 3 and **γ annealed** 0.97 → 0.997 exponentially over the first 10k
gradient steps after each reset, weight decay, a self-predictive representation loss (SPR), dueling,
double-Q, **C51 and prioritised replay kept**, no noisy nets, an EMA target network, and no ε-greedy at
all during training. Verified 2026-09-16 against §4 and the authors' `BBF.gin`.

**What transfers and what does not, decided up front:**

| piece | the paper | here |
|---|---|---|
| the network | Impala ×4, 15-layer ResNet, hidden 2048 | wider MLP: `SNEK_FC_LAYERS` at 4× the reference width (1280) with layer norm, the same "bigger" lever without a CNN. The trunk is `QNet`'s hidden stack, so the shape is one knob; a 2048 cell is the tuning wave |
| resets | encoder shrink-and-perturbed 50 % toward a fresh init, the layers after it fully reset, every 40k gradient steps at replay ratio 8 (20k at 2); no resets in the last 100k gradient steps | `algos/bbf/resets.py::shrink_and_perturb(module, alpha, seed)`: θ ← α θ + (1 − α) θ_fresh with α 0.5 on the hidden stack, and a full re-init of the head, every `SNEK_BBF_RESET_INTERVAL` gradient steps (40k); `SNEK_BBF_NO_RESETS_AFTER` (a fraction of the cap). The optimiser state for those parameters is reset with them. New |
| n-step and γ anneal | 10 → 3 and 0.97 → 0.997, exponential, over 10k gradient steps after each reset | `algos/bbf/schedules.py`: from `SNEK_BBF_N_STEP_START` (10) to `SNEK_N_STEP_UPDATE` (3) and from `SNEK_BBF_GAMMA_START` (0.97) to `SNEK_DISCOUNT` (**0.99** here, not 0.997 -- `README.md`, "Translating") over `SNEK_BBF_ANNEAL_STEPS` (10k gradient steps) after each reset. **`algos/dqn/collect.py`'s n-step window becomes settable between steps**; that is the one change to `algos/dqn/`, and it is a setter with a fixture that the window at n is exactly what the constant n produced |
| SPR | weight 5, 5-step latent prediction | **dropped for the first batch.** The self-predictive loss is a representation learner for pixels; on a 26-value vector it is a transition-model auxiliary of unclear value, and it is the one piece with its own network and its own knobs. `SNEK_BBF_SPR` is reserved and a second batch adds it at the paper's weight if F1 is short of the paper's relative gain. This is F1's one stated departure from the paper |
| dueling, double-Q, EMA target, weight decay | on, on, τ 0.005 every update with target-net action selection, AdamW 0.1 | **F1 builds `DuelingTrunk`**, in `algos/rainbow/net.py` where B1 will find it; double-Q is DQN's; `SNEK_TARGET_UPDATE_TAU` 0.005 with period 1; AdamW `SNEK_BBF_WEIGHT_DECAY` 0.1 |
| replay | prioritised, capacity 200k, 2,000 steps before learning | `algos/dqn/replay.py` with PER **on** at its default exponent (the paper keeps Dopamine's prioritised scheme), 200k capacity, 2,000 prefill |
| replay ratio | **8** gradient steps per env step for the headline (2 also reported), batch 32 | `SNEK_REPLAY_RATIO` for 8 gradient steps per move at batch 32 -- the paper's headline setting and the row's point; the 2 cell is the second wave |
| head | C51, 51 atoms | **C51, 51 atoms on [−10, 110]** -- F1 runs in phase 1 beside A1, so the head is built here (`algos/dist/heads.Categorical`, A2's module, written early) and the support is the return-range choice of `a-return-tail.md` §1, not yet the one A2's stability batch settles on; if A2 later moves it, F1's local cell is re-run at A2's. A scalar-head cell reads against A1 without the distribution |
| optimiser | Adam 1e-4, ε 1.5e-4 | the same |
| exploration | ε 0 during training, 0.001 at eval | ε 0, shield off, fork off; eval greedy |
| budget | 100k agent steps, 26 games, many seeds | **100k moves** -- the paper's regime scaled by `README.md`'s rule, and then 500k as the plan's own second cap |

`build_config` is DQN's plus `SNEK_BBF_*`; PPO's knobs refused by name. `algo.py` registers `bbf`.

Tests: shrink-and-perturb at α = 1 is a no-op and at α = 0 equals a fresh init with the given seed;
the head is fully re-initialised while the hidden stack is interpolated; the optimiser's moment buffers
for the reset parameters are zero after a reset; the anneal starts at its start value on the step after
a reset and reaches the target at `anneal_steps`; no reset fires inside the no-reset tail; the n-step
setter produces the constant window's targets when held. Mutants: the reset applied to every layer at
α, the anneal not restarted on a reset, γ annealed the wrong direction, the tail guard dropped.

## 2. The batches

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| F1 | 4 seeds `bbf` at **100k moves** (the paper's regime; replay ratio 8, resets every 40k gradient steps, no resets in the last 12.5 %) + 4 seeds at **500k moves** (~1 % of a reference run; the reset interval and tail scaled with the cap) | A1's reward and history, the paper's values above; no fork, so a counted step is one move | A1's two cells at the same move count, and A1's full run | **onset step** (first eval ≥80%), the perfect rate at the cap, and the stage-B density at the cap against A1's density at the same step. Then `hof5000`/`hof30k` as usual on whatever it produced |
| F1 replay ratio 2 | 4 seeds at 500k moves with `SNEK_REPLAY_RATIO` 2 and resets every 20k (the paper's cheaper setting) + 4 seeds scalar head | F1 | F1 | is the replay ratio the lever; is the distribution |
| F1 long | the best cell held to A1's cap (`SNEK_INIT_FROM` the F1 checkpoints, or simply a second batch at the long cap) | F1 | A1 | whether the data-efficient config also wins at the budget the series has been using, or trades the top for the onset |

## 3. Gates

1. Smoke; the reset fires at least once inside the 5,000-step smoke (`SNEK_BBF_RESET_INTERVAL` set
   low in the smoke spec) and the log line says so.
2. The mutation spec kills every mutant.
3. Tuning budget: one laptop wave on the reset interval and the hidden width (1280 / 2048).

## 4. What F1's number is used for

The series budgets every row at the reference's 100M-transition cap because that is what PPO needed.
F1's onset step against A1's is the measurement that either confirms that or lets every later
value-based row (B, D2, E) queue at a shorter cap with a stated basis. The rule the plan proposes:
**if F1 reaches A1's cap-density within a fifth of A1's steps, the later value rows queue at half the
reference cap and hold on if the curve is still rising**; otherwise they queue at the reference cap.

## 5. What would change the plan

- **F1 matches A1's density at the short cap.** The rule in §4 fires. Also: the resets are worth
  offering to every value row as a knob, since a run that plateaus and drifts is the shape most of this
  project's collapses have.
- **F1 trails A1 at the short cap but wins the long one.** The resets are a late-stability lever, not
  an efficiency one, on this game; the long batch's drawdown count is the evidence.
- **F1 is below A1 everywhere.** The 100k-regime recipe does not transfer off pixels; SPR is tried once
  and the group closes.
