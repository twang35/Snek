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

Rainbow-style DQN at a high replay ratio, with: a wider network (Impala CNN ×4 on Atari), **shrink-and-
perturb resets** of the final layers every `reset_interval` gradient steps (parameters interpolated
toward a fresh initialisation), **n-step annealed** 10 → 3 and **γ annealed** 0.97 → 0.997 over the
first 10k steps after each reset, weight decay, a self-predictive representation loss (SPR), dueling,
no noisy nets, no PER, and a target network with Polyak averaging.

**What transfers and what does not, decided up front:**

| piece | here |
|---|---|
| the network | wider MLP: `SNEK_FC_LAYERS` at 4× the reference width (1280) with layer norm, the same "bigger" lever without a CNN. The trunk is `QNet`'s hidden stack, so the shape is one knob |
| resets | `algos/bbf/resets.py::shrink_and_perturb(module, alpha, seed)`: θ ← α θ + (1 − α) θ_fresh, on the head and the last hidden layer, every `SNEK_BBF_RESET_INTERVAL` gradient steps (40k in the paper; scaled to the run in the smoke). The optimiser state for those parameters is reset with them. New |
| n-step and γ anneal | `algos/bbf/schedules.py`: exponential from `SNEK_BBF_N_STEP_START` (10) to `SNEK_N_STEP_UPDATE` (3) and from `SNEK_BBF_GAMMA_START` (0.97) to `SNEK_DISCOUNT` (0.99 here, not 0.997 -- the horizon argument in `e-exploration.md` §1) over `SNEK_BBF_ANNEAL_STEPS` (10k) after each reset. **`algos/dqn/collect.py`'s n-step window becomes settable between steps**; that is the one change to `algos/dqn/`, and it is a setter with a fixture that the window at n is exactly what the constant n produced |
| SPR | **dropped for the first batch**. The self-predictive loss is a representation learner for pixels; on a 26-value vector it is a transition-model auxiliary of unclear value, and it is the one piece with its own network and its own knobs. `SNEK_BBF_SPR=1` is reserved and a second batch adds it if F1 is short of the paper's relative gain |
| dueling, Polyak, weight decay | **F1 builds `DuelingTrunk`**, in `algos/rainbow/net.py` where B1 will find it: F1 runs in phase 1 and B1 in phase 4, so the module is written here and reused there. `SNEK_TARGET_UPDATE_TAU` 0.005, AdamW `SNEK_BBF_WEIGHT_DECAY` 0.1 |
| replay | `algos/dqn/replay.py` with priorities off (`SNEK_PRIORITY_EXPONENT=0`), replay ratio `SNEK_REPLAY_RATIO` 2 (the paper's 8 at batch 32 is 2 gradient steps per transition at batch 128) |
| head | scalar (the paper's C51 is on for Atari; here it is off for the first batch so F1 reads against A1 without Group A's head, and A2 exists before a second F1 batch would) |
| the step | DQN's |

`build_config` is DQN's plus `SNEK_BBF_*`; PPO's knobs refused by name. `algo.py` registers `bbf`.

Tests: shrink-and-perturb at α = 1 is a no-op and at α = 0 equals a fresh init with the given seed;
the optimiser's moment buffers for the reset parameters are zero after a reset; the anneal starts at
its start value on the step after a reset and reaches the target at `anneal_steps`; the n-step setter
produces the constant window's targets when held. Mutants: the reset applied to every layer, the
anneal not restarted on a reset, γ annealed the wrong direction.

## 2. The batches

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| F1 | 4 seeds of `bbf` at **`SNEK_MAX_STEPS` = 500k counted steps** (2M moves at the fork's four; ~2% of a reference run) | A1's reward and history, the BBF defaults above | A1 at the same step, and A1's full run | **onset step** (first eval ≥80%), the perfect rate at the cap, and the stage-B density at the cap against A1's density at the same step. Then `hof5000`/`hof30k` as usual on whatever it produced |
| F1 long | the same 4 arms held to A1's cap (`SNEK_INIT_FROM` the F1 checkpoints, or simply a second batch at the long cap) | F1 | A1 | whether the data-efficient config also wins at the budget the series has been using, or trades the top for the onset |

## 3. Gates

1. Smoke; the reset fires at least once inside the 5,000-step smoke (`SNEK_BBF_RESET_INTERVAL` set
   low in the smoke spec) and the log line says so.
2. The mutation spec kills every mutant.
3. Tuning budget: one laptop wave on the reset interval and the replay ratio.

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
