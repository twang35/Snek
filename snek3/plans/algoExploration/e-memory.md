# Group E: memory -- recurrent PPO, R2D2

**Status: planned 2026-09-16, nothing built; reviewed 2026-09-30** (E1 declared local-only, E2's update count made the budget's
goal, a feed-forward R2D2 control added, E1's minibatch sized in transitions -- §0, §2, §2b, §3; a second review the same day fixed the sequence geometry to the paper's 40 loss steps inside 80, gave the critic its own GRU, put the previous reward on the seam, and rewrote §5 around the new controls; a third fixed the short-episode rule, derived the stride, split the stored states per tower and relabelled the C51 cell). Group E of [`algorithm-series.md`](algorithm-series.md);
conventions in [`README.md`](README.md). Phase 6 of the running order; waits for Group C, and Group D (the reset probe) runs before it.

The question: does memory over the body matter, given that the 26-value observation summarises the
body rather than showing it? The prior is strong: making the last eight turns visible (`hist8`) was
the largest lever in the project, 49 → 94-95% stage-B density (`docs/findings.md`). A recurrent policy
can carry an arbitrarily long history without widening the observation. E1 is recurrence alone, on
the incumbent, read against PPO; E2 is recurrence on a value agent with R2D2's stored-state and burn-in
machinery, read against E1 and C1 so the gain can be attributed to the memory or to the agent.

## 0. Paper-cell fidelity review (2026-09-23)

The paper cell follows the paper **wherever the game allows** (`README.md`, the paper-fidelity row, widened
2026-09-23); the local cell carries every codebase choice. This group was audited against that rule the
same day. Each item is fixed **before the row's paper cell is queued**; *to confirm* means the plan's
citation does not settle it and the paper or its code must be read first. An empty status is *open*.

| item | paper | here | fix | status |
|---|---|---|---|---|
| E1 paper cell | baselines `ppo2` Atari: 128 steps x 8 envs, 4 minibatches, 4 epochs, lr 2.5e-4 annealed to 0 and clip **held at 0.1** in `baselines/ppo2/defaults.py` (the PPO paper's Atari table anneals clip as 0.1 x α; the code it cites does not), clipped-value MSE, entropy 0.01, value 0.5, grad norm 0.5, Adam eps 1e-5, γ 0.99 and λ 0.95 fixed | b27's `hist8` config (rollout 256, 128 lanes, minibatch 512, huber, the horizon anneal) | **E1 is declared local-only** (the user, 2026-09-30): there is no paper, the reference is Atari code defaults tuned for pixels, and the row's question is memory *against the incumbent*, which a `ppo2`-default cell cannot answer. E1 stays in the series as a datapoint, read against b27's `hist8` table | **decided**: local-only |
| E2 dueling streams | R2D2: dueling over the LSTM with 512-wide streams | "dueling scalar (D1's module)", no width; the module is C1's | 512-wide streams (`SNEK_RAINBOW_STREAM_WIDTH` analogue) | |
| E2 importance weights | max-normalised | `dqn/replay.py`'s mean | `normalization='batch_max'` in the paper cell | |
| E2 prefill | actors fill the replay on their own epsilon ladder | epsilon 1, off the clock | a prefill on the ladder | |
| E2 loss and clip | *to confirm*: squared TD under the value rescaling, gradient-norm clip 40 (Ape-X) | unstated; `dqn/agent.py`'s Huber would be inherited | the paper's, stated in §2b | |
| LSTM width | 512 | the paper's (README widened 2026-09-23) | none | matches the new rule |
| E2 update count | learner updates over 10B frames, with the target period and lr both in updates. **The figures this plan has carried (256 actors at ~260 steps/s, ~5 updates/s, hence ~200k-400k updates) are from memory and do not agree with each other**: at that throughput 10B frames is ~10 hours of wall clock, and the paper trained for days. The paper's throughput section must be read and the count recomputed | §2b's replay-ratio line gave ~7,800 updates at 50M moves | **the update count is the budget's goal** (the user, 2026-09-30; §2b); **200k is the working floor until the paper's number is read**, and that read comes before the benchmark | **decided**; the number *to confirm* |
| E2 short episodes | sequences never cross an episode boundary; the paper does not say what a window at an episode's start burns in | a fixed 40-step burn-in gives a 20-step episode **no loss position at all**, and early Snake episodes are 10-30 steps, so padding and a mask alone would leave the agent nothing to learn from until it survives 40 moves | **the loss-block rule** (§2, replay row): the window is defined by its 40-step loss block, the burn-in is whatever the episode has before it (0 at the episode's start, where the stored state is the exact zero state) | **decided** 2026-09-30 |
| E2 C51 cell's claim | -- | the C51 cell differs from the paper cell in the head **and** the rescaling (off, above), so paper-against-C51 is two recipes, not the distribution alone | relabelled in §3 and §5; a scalar cell with the rescaling off is the control that would isolate the distribution, run only if the C51 cell moves | **decided** 2026-09-30 |
| E2 memory control | the paper's §4 feed-forward ablation: the same agent with the LSTM removed | none; §3 read the paper cell against A1, which differs in every plumbing knob | a `ffr2d2` cell in the first wave (§3), on **the same loss positions, priority mask, moves, updates, batch and sequence length** as the paper cell | **decided** 2026-09-30 |
| E2 sequence geometry | §2.3 and §3 of the paper: sequences of **m = 80** steps overlapping by 40, the first **l = 40 as burn-in**, the update on **the remaining 40** | §2 stored `burn_in + length` = 120 and cut at 80: two geometries in one plan, and the update arithmetic depended on which | **the paper's: 80 stored, 40 burn-in, 40 loss steps, stride 40** (the user, 2026-09-30). ACME's R2D2 stores 40 + 80 + 1 and learns over 80; it is the departure, not followed | **decided** |
| E1 actor-critic sharing | -- (E1 is local-only) | the incumbent's critic is **its own tower** (`algos/ppo/net.py`: the actor stays exactly `QNet`, `vf_coef` cannot leak into the policy gradient, the checkpoint is the actor alone); §2 shared the trunk and GRU between actor and critic | **the critic gets its own GRU tower** (§2), so E1 differs from b27 in the recurrence and nothing else | **decided** 2026-09-30 |
| E2 C51 head under h(x) | R2D2 has no distributional variant; a transformed categorical target has no paper definition | §2 left it implied | **the C51 cell runs `SNEK_R2D2_RESCALE=0`**: its support already covers the game's return range, which is what the rescaling buys the scalar head. Stated in §2b | **decided** 2026-09-30 |
| E2 previous-reward input | the previous reward is an LSTM input; whether raw or under h(x) | the mutant list assumes rescaled | state it in §2b once read | *to confirm* (at +100 a win the scale matters) |
| E2 new-sequence priority | Ape-X actors compute each new sequence's TD priority on their local copy | unstated | computed on the live net at bank time, or max priority stated as a departure | *to confirm* |
| E2 loss reduction | ACME's R2D2: squared TD summed over the loss window, averaged over the batch, max-normalised IS weights | unstated | the paper's, stated in §2b | *to confirm* against the paper and ACME |
| E2 replay span | 4M rows of 2.5B agent steps = 0.16% of the run | 4M rows of 50M moves = 8%: the buffer holds far staler policy data | none, as written; a stated translation | noted |
| the ε ladder knob | per-actor ε held for the run | `SNEK_EPSILON_SCHEDULE` has `eval` and `linear` only; `apex` is not built | in E2's module table (§2) | open |

## 1. The seam change this group needs, designed once

`policy_fn` is `(m, obs_len) -> (m,)` and stateless. The engine (`vectorized/engine.py`) calls it once
per resident job on that job's rows, and **lanes migrate between jobs and reset between episodes**
without telling the policy. A recurrent policy needs, per lane, a hidden state that persists across
steps and is zeroed when the lane's episode restarts. This is the one place the group touches
`vectorized/`, and Group F inherits it.

| decision | rule |
|---|---|
| the protocol | a policy may be a `StatefulPolicy`: a callable with the same `(obs) -> actions` shape **plus** a `begin(rows, fresh, prev_reward)` method the engine calls before the step, where `rows` are the lane indices this call covers, `fresh` is a boolean mask of lanes that were reset since the last step, and `prev_reward` is each lane's reward from its previous step (0 for a fresh lane). **The previous action the policy remembers itself** (it chose it; zero one-hot on a fresh lane); **the previous reward only the engine knows**, and R2D2's net takes both as inputs, so it is part of the seam, on every path -- training, stage A, the shards, `watch.py` and `record_gif.py`. A plain callable has no `begin` and is treated as today. A plain callable has no `begin` and is treated as today. `vectorized/` still imports no torch: the protocol is duck-typed on a method name and the mask is numpy |
| who owns the state | the policy. It keeps the recurrent state indexed by absolute lane row -- `(width, hidden)` for a GRU, **the pair `(h, c)` for an LSTM**, each `(width, hidden)` -- plus the previous action, zeroes the `fresh` rows in `begin`, and reads and writes the rows it is handed. The engine holds no state for it |
| what the engine adds | tracking which rows it reset since the previous policy call (it already knows -- `reset_rows` is its own call) and passing that mask through `begin`. One assignment and one method call in the step loop; `measure_stream` and `measure` unchanged in signature |
| stage A, shards, `watch.py` | all go through `restore.policy_fn_for`, which returns a `StatefulPolicy` for a recurrent sidecar; the shard's episodes-in-lockstep loop is the engine's, so nothing else changes. `watch.py` runs one lane and resets on the game's own boundary |
| the sidecar | `arch.json` gains `recurrent`: `{"type": "gru", "hidden": 128}`; in the signature. A checkpoint with it cannot load into a feed-forward net, and the reverse |
| training | the collectors own their own hidden states in the same way; PPO's `(T, N)` rollout stores the state at each step's start (E1) and the replay stores it per sequence (E2). **A value query must not advance the carried state**: PPO's bootstrap `V(s_T)` at the rollout's end runs the critic's GRU from the carried state and discards the result, and a test pins that the state before and after the query is the same |

**Tests for the seam**: an engine run with a stateful policy that counts steps since reset per lane
agrees with the episode lengths the engine reports; a lane that migrates to a new job starts fresh;
the layering test's probe still finds no torch under `vectorized/`.

## 2. The rows

### E1 -- recurrent PPO

PPO with a GRU between `QNet`'s hidden stack and the actor and critic heads. The rollout is
sequential per lane already (`algos/ppo/collect.py` steps every lane T times), so the collection stores
the hidden state at each step and the update runs the GRU over each lane's T-step chunk from the
stored initial state, with truncated backpropagation over the chunk. Episode boundaries inside a
chunk zero the state, and the `(1 − done)` that gates GAE gates the recurrence too.

| module | contents |
|---|---|
| `algos/ppo/net.py` | `RecurrentActor` and `RecurrentCritic`, **two towers, each trunk → GRU(`hidden`) → head**, as the incumbent's actor and critic are two towers (the module docstring's three reasons: the actor stays `QNet` plus a cell, `vf_coef` cannot leak into the policy gradient, the checkpoint is the actor alone). The recurrent compute doubles; the alternative, one shared GRU, would change the sharing arrangement and the recurrence in the same cell (review, 2026-09-30). Built when `arch['recurrent']` is set; the feed-forward path is untouched |
| `algos/ppo/rollout.py` | **two** state buffers beside the others, the actor's and the critic's, each `(T, N, hidden)` for a GRU and a `(T, N, 2, hidden)` `(h, c)` pair for an LSTM, plus a `(T, N)` `fresh` mask; GAE unchanged |
| `algos/ppo/collect.py` | carries the state across steps, zeroes on `done` |
| `algos/ppo/agent.py` | the epoch loop iterates minibatches of **whole lanes** (sequences), not shuffled transitions, and replays the GRU from the stored state; the losses are the same three statements |
| knobs | `SNEK_PPO_RECURRENT` (0/off; `gru`, `lstm`), `SNEK_PPO_RECURRENT_HIDDEN` (128), `SNEK_PPO_SEQ_MINIBATCH` (lanes per minibatch, **2**: 2 lanes x rollout 256 = 512 transitions, b27's minibatch, so the update count per epoch stays b27's 64 -- see below). Feed-forward PPO is exactly what it was at the defaults, and a fixture asserts a rollout and an update are byte-identical with the knob off |

Tests: a GRU replay from the stored state reproduces the log-probs stored at collection (the ratio is
1 on the first epoch, to tolerance); a `done` inside a chunk zeroes the state; the recurrent net with
`hidden` = 0 is refused rather than silently feed-forward. Mutants: the state not zeroed on `done`, the
minibatch shuffling transitions, the stored state off by one step.

**There is no recurrent-PPO paper, so E1 is local-only** (decided 2026-09-30, §0): its base is b27's `hist8` config and its
control is that table. The design reference is OpenAI baselines' `ppo2` with the `lstm` policy**
(`baselines/common/models.py`, `baselines/ppo2/ppo2.py`): one LSTM of 128 units after the trunk, the
state carried across rollouts, minibatches of whole per-env rollouts (`nenvs // nminibatches` lanes
each) with the `done` mask resetting the state inside a sequence, truncated backpropagation over the
rollout. That is the design above: LSTM 128 (`gru` is the variant), b27's rollout of **256** (`SNEK_PPO_ROLLOUT`;
the code default and baselines' Atari default are 128) as the BPTT length.

**The minibatch is sized in transitions, not in baselines' lane count** (decided 2026-09-30). Baselines' 4 minibatches over
128 lanes would be 32 lanes x 256 = 8,192 transitions and 4 updates an epoch, against b27's 512 and 64: a 16-fold change in
the update count, confounded with the recurrence. So `SNEK_PPO_SEQ_MINIBATCH` is **2 lanes** = 512 transitions and 64 updates
an epoch, the control's. The cost is a gradient from two episodes' worth of lanes per update rather than b27's shuffled 512; if
the smoke shows it unstable the fallback is SB3-contrib's chunking (`RecurrentPPO`: shorter sub-sequences, each with its stored
state), which gives more sequences per 512 transitions at the price of the whole-sequence property the tests pin. The width
tuning wave (§4) is also where 4 lanes is tried.

**The base is `hist8`, not `hist0`.** The question is whether memory adds to what the history window
already gives, since that is the incumbent. The second cell, at `SNEK_OBS_HISTORY=0`, asks whether
recurrence *replaces* the window, and runs in the same wave (§3): its control is **b27's own `hist0`
cell**, which is the same config less the history, not b7's older one.

### E2 -- R2D2 (Kapturowski, Ostrovski, Quan, Munos & Dabney 2019)

Recurrent replay distributed DQN: an LSTM Q-network trained on fixed-length sequences from replay,
each stored with the recurrent state at its start, with a **burn-in** prefix that is replayed to warm
the state before the loss is taken over the rest, n-step double-Q targets, PER with a sequence priority
(η-mix of max and mean absolute TD), and the value rescaling h(x) = sign(x)(√(|x| + 1) − 1) + εx on
**unclipped** rewards. **The paper's head is a dueling scalar head, not a distributional one**, and the
LSTM's input is the trunk's features concatenated with the previous action (one-hot) and the previous
reward. Here it is R2D2 the algorithm, at the box's actor count.

| module | contents |
|---|---|
| `algos/r2d2/net.py` | trunk → concat(previous action one-hot, previous reward) → LSTM(`hidden`) → dueling scalar head (Group C's `DuelingTrunk`, scalar form). `SNEK_R2D2_HEAD=c51` puts C1's dueling C51 head there instead, for the **local** variant that asks whether the memory and the distribution compound |
| `algos/r2d2/replay.py` | a sequence buffer over `algos/dqn/replay.py`'s sum tree: entries are windows **defined by their loss block** (the short-episode rule, §0, decided 2026-09-30): each episode is cut into consecutive loss blocks of `SNEK_R2D2_SEQ_LENGTH − SNEK_R2D2_BURN_IN` = **40** steps from its first step, so every step of every episode is a loss position exactly once; a window is its loss block with **up to 40 burn-in steps prepended, as many as the episode has before the block** -- 40 for every block but the first, 0 for the first, whose stored state is the exact zero state the collector had at the episode's start -- and the stored `(h, c)` is the collector's at the window's first step. Mid-episode this is the paper's geometry exactly: 80 stored, 40 burn-in, 40 loss, consecutive windows overlapping by 40; **the stride is derived, not a knob** (`SNEK_R2D2_SEQ_OVERLAP` is gone, which also settles what "overlap" means at length 160 in §4: the loss block is 120 and the stride 120). At an episode's start the paper is silent and the rule costs nothing, since a zero-step burn-in from a known state is what burn-in is for. **The last block of an episode is short** when the length is not a multiple of 40: stored short, padded at the end with a validity mask, and the loss and the priority sum over valid steps only. A 20-step episode is one window, 0 burn-in, 20 valid loss steps. Rows stored **once**, a window is a `(start, burn_in, length)` triple into them; 1e5 windows are ≤ 4M rows. The n-step target of the last loss steps reads the `n` rows after the block when the episode has them and bootstraps to 0 at a terminal, so a window is drawable once those rows exist or the episode has ended (as `algos/bbf/replay.py` does). Priorities per sequence; the transition buffer is reused for the tree only |
| `algos/r2d2/collect.py` | `algos/dqn/collect.py`'s lanes carrying an LSTM state and the previous action and reward, cutting each episode into 40-step loss blocks from its first step and banking each with its burn-in prefix and stored state (the replay row's rule), never across an episode boundary; **`SNEK_EPSILON_SCHEDULE=apex`, new** in `algos/dqn/schedules.py`: lane i holds `SNEK_INITIAL_EPSILON ** (1 + SNEK_APEX_ALPHA * i / (lanes - 1))` for the run (the ladder is per lane, so it lives beside the collector, not the eval history); **no fork** (a forked lane would need a copied state and a copied sequence prefix; refused by name) |
| `algos/r2d2/agent.py` | burn-in of the first 40 steps under `no_grad` **on both the online and the target net from the same stored state**, then the loss over the last 40: the n-step double-Q target h(Σγ^k r + γ^n h⁻¹(Q_target(s', argmax Q_online(s')))) per step, squared TD **summed over the 40 loss steps and averaged over the 64 sequences**, IS weights max-normalised over the batch (ACME's reduction; *to confirm* against the paper), gradient norm clipped at 40 (Ape-X); the sequence priority η·max + (1 − η)·mean of |TD| over the valid loss steps. **With `SNEK_R2D2_HEAD=c51`** the per-step loss is C51's cross-entropy against the n-step categorical target projected onto the support (`algos/dist/c51.py`, no rescaling, §2b), the same sum-over-steps, mean-over-sequences reduction, and the per-step quantity in the priority mix is that cross-entropy, as Rainbow's priority is (Group C); double-Q's argmax is over the distribution's mean. **`SNEK_R2D2_RECURRENT=0`** replaces the LSTM by a dense 512 + ReLU and skips the burn-in unroll, **and nothing else**: the same 80-step windows, the same last-40 loss positions and validity mask, the same priorities, so the control never trains on a row the paper cell does not |
| knobs | `SNEK_R2D2_HIDDEN` (512), `SNEK_R2D2_SEQ_LENGTH` (80), `SNEK_R2D2_BURN_IN` (40; the loss block and the stride are their difference), `SNEK_R2D2_PRIORITY_ETA` (0.9), `SNEK_R2D2_RESCALE` (1) with `SNEK_R2D2_RESCALE_EPS` (1e-3), `SNEK_R2D2_HEAD` (`scalar`), `SNEK_R2D2_PREV_INPUT` (1: feed previous action and reward), **`SNEK_R2D2_RECURRENT` (1; 0 is the feed-forward control: the LSTM replaced by a dense 512 + ReLU, burn-in and stored state off, sequences kept so the loss window and priorities are the same)**; DQN's names for the rest, at the paper's values (§2b) |
| the step | one `collector.step()`, `collect_envs` moves |

Tests: the rescaling and its inverse compose to the identity; **the burn-in carries no gradient**: after a backward pass from the
loss window, the gradient equals the one from a pass that starts the loss window from the burn-in's output as a constant (a forward-only
`.grad is None` check proves nothing, review 2026-09-30); the feed-forward control's loss touches exactly the paper cell's loss positions on a
hand-built batch with a short episode in it; the sequence priority at η = 1 is
the max and at η = 0 the mean; a stored state replayed over the burn-in matches the state the collector
had at the loss window's start when the weights have not changed; the previous-action input at an
episode's first step is the zero vector. Mutants: burn-in included in the loss, the inverse rescaling
skipped on the target, overlap producing a gap instead, the previous reward fed unrescaled.

## 2b. The papers' settings, and how each lands here

| setting | R2D2 (Table 2 and §2; "missing parameters follow Ape-X") | here |
|---|---|---|
| LSTM | 512, after the conv trunk's 512 features; previous action and reward as extra inputs | **512** (`SNEK_R2D2_HIDDEN`), over `fc 320`; a 256 cell is the tuning wave |
| head | dueling, scalar, 512-wide streams | dueling scalar, C1's module (`algos/rainbow/net.py`) at `stream_width` 512; the C51 head is the local variant |
| sequence, burn-in, overlap | m = 80 with the first l = 40 as burn-in and the update on the remaining 40 (§2.3, §3); adjacent sequences overlap by 40; never across an episode boundary | **80 stored, 40 burn-in, 40 loss steps, stride 40** (§0, decided 2026-09-30). **Note**: ACME's R2D2 stores burn-in + 80 + 1 and learns over 80 steps, twice the paper's rows an update; this plan follows the paper's text and names ACME as the departure. Every row count below is at 40 loss steps |
| n-step | 5, double Q | `SNEK_N_STEP_UPDATE=5` |
| discount | 0.997 | **0.997** (`SNEK_DISCOUNT`); the local cell takes the reference's |
| replay | 4M observations (1e5 part-overlapping sequences); priority exponent 0.9, IS exponent 0.6, η 0.9 | 1e5 windows over 4M rows stored once (26 + 16 history values, action, reward, done: ~45 floats, ~720 MB) **plus the stored `(h, c)` per window** (1e5 x 2 x 512 floats, ~410 MB) and the previous action and reward per row: **~1.2 GB an arm, ~5 GB a 4-arm wave**; halve the windows to 5e4 if the desktop's memory says so; α 0.9, β 0.6 held, η 0.9 |
| batch | 64 sequences | 64 sequences |
| optimiser | Adam 1e-4, ε 1e-3 | Adam 1e-4, ε 1e-3 |
| target | hard copy every 2,500 learner updates | 2,500 |
| value rescaling | h(x) with ε 1e-3; rewards **not** clipped | the same, on the unclipped reward -- this is the one paper in the series whose reward handling transfers as written. **The C51 cell runs it off** (`SNEK_R2D2_RESCALE=0`, §0): a transformed categorical target has no paper definition, and the support [−10, 110] already covers the return range |
| exploration | 256 actors, per-actor ε_i = 0.4^(1 + 7 i / 255) (Ape-X), so ε from 0.4 down to 0.4⁸ ≈ 6.5e-4, held for the run | `collect_envs` 32 lanes with the same formula over i = 0..31 (`SNEK_EPSILON_SCHEDULE=apex`, `SNEK_INITIAL_EPSILON` 0.4, `SNEK_APEX_ALPHA` 7) -- a per-lane ε the collector already has the shape for, since the fork gives lanes different roles today; shield off, fork off |
| replay ratio and **the update count** | the paper states throughput, not a ratio, and **the throughput figures below are from memory and must be read from the paper before the benchmark** (§0): 256 actors at ~260 steps/s against ~5 learner updates/s. At 64 x 40 = 2,560 loss rows an update those give **~0.19 loss rows per observed row**; the often-quoted 0.8 may count all 80 stored rows, or come from a later paper, and is not this plan's figure until sourced. Over 10B frames = 2.5B agent steps the same numbers give ~10 hours of wall clock, which the paper's run time contradicts, so the resulting update count (~200k-400k as carried here) is **unverified**. The target period (2,500) and the lr are in updates, so the update count is what the paper's schedule is written against | **the update count is the goal** (decided 2026-09-30), not the ratio. `SNEK_REPLAY_RATIO` here is *gradient steps per banked transition*, a different quantity: at 0.2-0.8 loss rows per observed row and 64 x 40 = 2,560 loss rows an update, 50M moves is ~4,000-15,600 updates (two to six target copies, 1e-4 lr barely moving), and at 0.8 gradient steps per move it is 40M sequence updates no box can run. **What 200k updates preserves and what it does not**: the target period and the lr are in updates, so the paper's *schedule* survives; the replay intensity does not -- 200k x 2,560 rows over 50M moves is ~10 loss rows a move against the paper's 0.2-0.8, so data is reused 12-50x more and the buffer is correspondingly staler relative to the policy. **That is a deliberate adaptation, stated in the spec's manifest note**, and it is the same in every E2 cell: the paper, `ffr2d2` and C51 cells match on moves, updates, batch and sequence length, so the within-wave readings are clean and only the comparison to the paper's regime carries it. **Before E2 is queued**: benchmark the sequence update *and* the collector separately (`tools/`' 2,000-step rule, 4-arm desktop wave at ~0.7x solo) -- cutting moves saves collection and stage-A time but **cannot cut the cost of 200k learner updates**, so the benchmark has to say which of the two is the limit -- then set `SNEK_REPLAY_RATIO`, `SNEK_MAX_STEPS` and the batch so the cell reaches **at least 200k updates** (the working floor; the paper's own count replaces it once read) inside objective 3's ~24 h, and write the resulting update count, moves, rows a move and ratio into this row and the manifest note. If 200k does not fit even with the learner the whole budget, the batch is the lever and the shortfall is stated |
| actor weight refresh | every 400 environment steps | not applicable: one process, the collector reads the live net |
| frames | 10B, 256 actors | **set by the update count above**, not by a move budget: the move cap is what the benchmark and 200k+ updates give (50M was the placeholder before 2026-09-30); R2D2's algorithmic content does not need the actor count and the box has not got it |

E2's **local** cell keeps everything above and swaps in this codebase's plumbing where it exists: PER
0.6, the eval-driven ε, the shield, the fast target. Because R2D2 has no fork and the per-lane ε ladder
is its own exploration answer, the local cell's difference is smaller than Group A's, and it runs only
if the paper cell trails C1.

## 3. The batches

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| E1 (**local-only**, §0) | 4 seeds `SNEK_PPO_RECURRENT=lstm` (the `ppo2` form, hidden 128, `SNEK_PPO_SEQ_MINIBATCH=2`, two towers) + 4 seeds **the same at `SNEK_OBS_HISTORY=0`** (the no-window cell, moved into the first wave 2026-09-30: recurrence can replace the window without beating it, so it cannot wait on E1 moving; `gru` goes to the tuning wave) | b27's `hist8` PPO config verbatim (the reference; `plans/zigzag-shaping.md` §6 states it) | PPO `hist8` | stage-B density, `hof5000`, `hof30k`, drawdowns; the onset step |
| E1 no-window | in E1's first wave (above) | E1 | **b27's `hist0` cell** (the same config less the history; b7's `hist0` is an older config), PPO `hist8` and E1 | does recurrence replace the window: level with `hist8` from `hist0` is a yes even if the `hist8` cell is level too |
| E2 | three cells, 12 arms (decided 2026-09-30): 4 seeds `r2d2` **paper** (§2b: LSTM 512, scalar dueling head, 5-step, Adam 1e-4, target 2,500, the Ape-X ε ladder, the update count §2b sets) + 4 seeds **`ffr2d2`**, the paper cell with `SNEK_R2D2_RECURRENT=0` (the paper's own §4 feed-forward ablation: the same 80-step windows, last-40 loss positions, priorities, rescaling, ladder, moves, updates and batch, no memory) + 4 seeds `r2d2` with `SNEK_R2D2_HEAD=c51 SNEK_R2D2_RESCALE=0` (C1's head under the memory, §2b) | the b2 preset with step penalty 0.01, `hist8`, **shaping off** (the paper-cell rule of 2026-09-23) | paper against `ffr2d2`; paper against C1 paper and A1 paper for shape; the C51 cell against the paper cell | as E1. **Memory alone is paper minus `ffr2d2`**: the only pair that differs in the LSTM and nothing else (A1 paper differs in γ, n-step, optimiser, ε, replay, target and rescaling, so it cannot carry that reading). **The C51 cell against the paper cell is two recipes**, the categorical head without the rescaling against the scalar head with it (§0); it is not the distribution alone. If it moves, a scalar cell with `SNEK_R2D2_RESCALE=0` is the control that separates the two, and runs then |
| E2 local | 4 seeds paper with the codebase's PER, ε and target | E2 paper | E2 paper | only if E2 paper trails C1 |

## 4. Gates

1. The seam change lands first, with its tests, and every existing checkpoint still measures to the
   same numbers (a fixed-seed `evaluate.py ... one` on a HOF entry before and after, byte-identical rows).
2. Smoke for both; `watch.py` on a recurrent checkpoint plays a whole game with the state carried.
3. The mutation specs kill every mutant.
4. Tuning budget: one laptop wave each on the hidden width (E1 128 / 256; E2 512 / 256) and, for E2, the
   sequence length (80 / 160, Agent57's; at 160 with burn-in 40 the loss block and stride are 120, by the replay row's rule).

## 5. What would change the plan

(Rewritten 2026-09-30 around the controls: memory in E2 is read from the paper cell against `ffr2d2`, never against C1 or A1.)

- **E1 beats `hist8`.** Memory beyond eight moves matters; the no-window cell says whether the
  window was a proxy for it, and every later row (F, G's policy prior) is offered the recurrent cell.
- **E1 `hist8` is level and E1 `hist0` is level with it.** Recurrence replaces the window without
  improving on it: the eight-move history and a 128-cell memory carry the same information for this
  policy class. Worth knowing, and a reason to keep the window (it is cheaper).
- **E2 paper beats `ffr2d2`.** The memory helps under the value agent's machinery -- most likely the
  stored-state replay lets it learn from long endgames the on-policy rollout truncates. F1 and F2 are
  built on E2 as planned. **E2 beating C1 says nothing about memory by itself**: they differ in every
  plumbing knob; it says R2D2's recipe beats Rainbow's here. Likewise the C51 cell beating the paper
  cell says the categorical recipe beats the rescaled scalar one under the memory, not that the
  distribution is the lever, until the rescaling-off scalar control runs.
- **E2 paper is level with `ffr2d2`, and E1 is level.** No benefit from **these** recurrent
  configurations at **these** budgets -- a 128-cell GRU over b27's rollout and a 512 LSTM over
  40-step loss windows at ~10 rows a move -- not a proof that the 26 features plus eight moves are
  sufficient statistics. The failures are not a memory problem the series can reach, and the plan for F
  notes that its base (E2) is a null.
