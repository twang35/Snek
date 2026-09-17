# Group D: memory -- recurrent PPO, R2D2

**Status: planned 2026-09-16, nothing built.** Group D of [`algorithm-series.md`](algorithm-series.md);
conventions in [`README.md`](README.md). Phase 5 of the running order; waits for Group B.

The question: does memory over the body matter, given that the 26-value observation summarises the
body rather than showing it? The prior is strong: making the last eight turns visible (`hist8`) was
the largest lever in the project, 49 → 94-95% stage-B density (`docs/findings.md`). A recurrent policy
can carry an arbitrarily long history without widening the observation. D1 is recurrence alone, on
the incumbent, read against PPO; D2 is recurrence on a value agent with R2D2's stored-state and burn-in
machinery, read against D1 and B1 so the gain can be attributed to the memory or to the agent.

## 1. The seam change this group needs, designed once

`policy_fn` is `(m, obs_len) -> (m,)` and stateless. The engine (`vectorized/engine.py`) calls it once
per resident job on that job's rows, and **lanes migrate between jobs and reset between episodes**
without telling the policy. A recurrent policy needs, per lane, a hidden state that persists across
steps and is zeroed when the lane's episode restarts. This is the one place the group touches
`vectorized/`, and Group E inherits it.

| decision | rule |
|---|---|
| the protocol | a policy may be a `StatefulPolicy`: a callable with the same `(obs) -> actions` shape **plus** a `begin(rows, fresh)` method the engine calls before the step, where `rows` are the lane indices this call covers and `fresh` is a boolean mask of lanes that were reset since the last step. A plain callable has no `begin` and is treated as today. `vectorized/` still imports no torch: the protocol is duck-typed on a method name and the mask is numpy |
| who owns the state | the policy. It keeps a `(width, hidden)` array indexed by absolute lane row, zeroes the `fresh` rows in `begin`, and reads and writes the rows it is handed. The engine holds no state for it |
| what the engine adds | tracking which rows it reset since the previous policy call (it already knows -- `reset_rows` is its own call) and passing that mask through `begin`. One assignment and one method call in the step loop; `measure_stream` and `measure` unchanged in signature |
| stage A, shards, `watch.py` | all go through `restore.policy_fn_for`, which returns a `StatefulPolicy` for a recurrent sidecar; the shard's episodes-in-lockstep loop is the engine's, so nothing else changes. `watch.py` runs one lane and resets on the game's own boundary |
| the sidecar | `arch.json` gains `recurrent`: `{"type": "gru", "hidden": 128}`; in the signature. A checkpoint with it cannot load into a feed-forward net, and the reverse |
| training | the collectors own their own hidden states in the same way; PPO's `(T, N)` rollout stores the state at each step's start (D1) and the replay stores it per sequence (D2) |

**Tests for the seam**: an engine run with a stateful policy that counts steps since reset per lane
agrees with the episode lengths the engine reports; a lane that migrates to a new job starts fresh;
the layering test's probe still finds no torch under `vectorized/`.

## 2. The rows

### D1 -- recurrent PPO

PPO with a GRU between `QNet`'s hidden stack and the actor and critic heads. The rollout is
sequential per lane already (`algos/ppo/collect.py` steps every lane T times), so the collection stores
the hidden state at each step and the update runs the GRU over each lane's T-step chunk from the
stored initial state, with truncated backpropagation over the chunk. Episode boundaries inside a
chunk zero the state, and the `(1 − done)` that gates GAE gates the recurrence too.

| module | contents |
|---|---|
| `algos/ppo/net.py` | `RecurrentActorCritic`: trunk → GRU(`hidden`) → actor head and critic head. Built when `arch['recurrent']` is set; the feed-forward path is untouched |
| `algos/ppo/rollout.py` | a `(T, N, hidden)` state buffer beside the others and a `(T, N)` `fresh` mask; GAE unchanged |
| `algos/ppo/collect.py` | carries the state across steps, zeroes on `done` |
| `algos/ppo/agent.py` | the epoch loop iterates minibatches of **whole lanes** (sequences), not shuffled transitions, and replays the GRU from the stored state; the losses are the same three statements |
| knobs | `SNEK_PPO_RECURRENT` (0/off; `gru`, `lstm`), `SNEK_PPO_RECURRENT_HIDDEN` (128), `SNEK_PPO_SEQ_MINIBATCH` (lanes per minibatch, 16). Feed-forward PPO is exactly what it was at the defaults, and a fixture asserts a rollout and an update are byte-identical with the knob off |

Tests: a GRU replay from the stored state reproduces the log-probs stored at collection (the ratio is
1 on the first epoch, to tolerance); a `done` inside a chunk zeroes the state; the recurrent net with
`hidden` = 0 is refused rather than silently feed-forward. Mutants: the state not zeroed on `done`, the
minibatch shuffling transitions, the stored state off by one step.

**There is no recurrent-PPO paper; the reference is OpenAI baselines' `ppo2` with the `lstm` policy**
(`baselines/common/models.py`, `baselines/ppo2/ppo2.py`): one LSTM of 128 units after the trunk, the
state carried across rollouts, minibatches of whole per-env rollouts (`nenvs // nminibatches` lanes
each) with the `done` mask resetting the state inside a sequence, truncated backpropagation over the
rollout. That is the design above, so D1's paper cell *is* the plan: LSTM 128 (`gru` is the local
variant), the reference's rollout of **256** (b27's `SNEK_PPO_ROLLOUT`, `docs/runs.md`; the code default and
baselines' Atari default are 128) as the BPTT length, and baselines' 4 minibatches over 128 lanes = 32 lanes a minibatch (the reference's 256-transition
minibatches are a transition count and do not apply to whole-lane sequences). SB3-contrib's `RecurrentPPO` (256 hidden, chunked minibatches) is the other common form and
is not followed, because its chunking breaks the whole-sequence property the tests pin.

**The base is `hist8`, not `hist0`.** The question is whether memory adds to what the history window
already gives, since that is the incumbent. A second cell at `SNEK_OBS_HISTORY=0` asks whether
recurrence *replaces* the window; it runs only if the first cell moves.

### D2 -- R2D2 (Kapturowski, Ostrovski, Quan, Munos & Dabney 2019)

Recurrent replay distributed DQN: an LSTM Q-network trained on fixed-length sequences from replay,
each stored with the recurrent state at its start, with a **burn-in** prefix that is replayed to warm
the state before the loss is taken over the rest, n-step double-Q targets, PER with a sequence priority
(η-mix of max and mean absolute TD), and the value rescaling h(x) = sign(x)(√(|x| + 1) − 1) + εx on
**unclipped** rewards. **The paper's head is a dueling scalar head, not a distributional one**, and the
LSTM's input is the trunk's features concatenated with the previous action (one-hot) and the previous
reward. Here it is R2D2 the algorithm, at the box's actor count.

| module | contents |
|---|---|
| `algos/r2d2/net.py` | trunk → concat(previous action one-hot, previous reward) → LSTM(`hidden`) → dueling scalar head (F1's `DuelingTrunk`, scalar form). `SNEK_R2D2_HEAD=c51` puts B1's dueling C51 head there instead, for the **local** variant that asks whether the memory and the distribution compound |
| `algos/r2d2/replay.py` | a sequence buffer over `algos/dqn/replay.py`'s sum tree: entries are `(burn_in + length)`-step windows with the stored initial state; priorities per sequence. New; the transition buffer is reused for the tree only |
| `algos/r2d2/collect.py` | `algos/dqn/collect.py`'s lanes carrying an LSTM state and the previous action and reward, cutting sequences at `SNEK_R2D2_SEQ_LENGTH` with overlap `SNEK_R2D2_SEQ_OVERLAP`, never across an episode boundary; **no fork** (a forked lane would need a copied state and a copied sequence prefix; refused by name) |
| `algos/r2d2/agent.py` | burn-in replay under `no_grad`, the n-step double-Q target on the remainder, the rescaling and its inverse, the sequence priority |
| knobs | `SNEK_R2D2_HIDDEN` (512), `SNEK_R2D2_SEQ_LENGTH` (80), `SNEK_R2D2_BURN_IN` (40), `SNEK_R2D2_SEQ_OVERLAP` (40), `SNEK_R2D2_PRIORITY_ETA` (0.9), `SNEK_R2D2_RESCALE` (1) with `SNEK_R2D2_RESCALE_EPS` (1e-3), `SNEK_R2D2_HEAD` (`scalar`), `SNEK_R2D2_PREV_INPUT` (1: feed previous action and reward); DQN's names for the rest, at the paper's values (§2b) |
| the step | one `collector.step()`, `collect_envs` moves |

Tests: the rescaling and its inverse compose to the identity; the burn-in leaves parameters without
gradient (a fixture checks `.grad` is None after a burn-in-only pass); the sequence priority at η = 1 is
the max and at η = 0 the mean; a stored state replayed over the burn-in matches the state the collector
had at the loss window's start when the weights have not changed; the previous-action input at an
episode's first step is the zero vector. Mutants: burn-in included in the loss, the inverse rescaling
skipped on the target, overlap producing a gap instead, the previous reward fed unrescaled.

## 2b. The papers' settings, and how each lands here

| setting | R2D2 (Table 2 and §2; "missing parameters follow Ape-X") | here |
|---|---|---|
| LSTM | 512, after the conv trunk's 512 features; previous action and reward as extra inputs | **512** (`SNEK_R2D2_HIDDEN`), over `fc 320`; a 256 cell is the tuning wave |
| head | dueling, scalar, 512-wide streams | dueling scalar (F1's module); the C51 head is the local variant |
| sequence, burn-in, overlap | 80, 40, 40; never across an episode boundary | 80, 40, 40 |
| n-step | 5, double Q | `SNEK_N_STEP_UPDATE=5` |
| discount | 0.997 | **0.997** (`SNEK_DISCOUNT`); the local cell takes the reference's |
| replay | 4M observations (1e5 part-overlapping sequences); priority exponent 0.9, IS exponent 0.6, η 0.9 | 1e5 sequences (4M rows at 26+16 values ≈ 700 MB per box; halve it to 5e4 if the desktop's memory says so); α 0.9, β 0.6 held, η 0.9 |
| batch | 64 sequences | 64 sequences |
| optimiser | Adam 1e-4, ε 1e-3 | Adam 1e-4, ε 1e-3 |
| target | hard copy every 2,500 learner updates | 2,500 |
| value rescaling | h(x) with ε 1e-3; rewards **not** clipped | the same, on the unclipped reward -- this is the one paper in the series whose reward handling transfers as written |
| exploration | 256 actors, per-actor ε_i = 0.4^(1 + 7 i / 255) (Ape-X), so ε from 0.4 down to 0.4⁸ ≈ 6.5e-4, held for the run | `collect_envs` 32 lanes with the same formula over i = 0..31 (`SNEK_EPSILON_SCHEDULE=apex`, `SNEK_INITIAL_EPSILON` 0.4, `SNEK_APEX_ALPHA` 7) -- a per-lane ε the collector already has the shape for, since the fork gives lanes different roles today; shield off, fork off |
| replay ratio | ~0.8 replays per observation | `SNEK_REPLAY_RATIO` for 0.8 samples per transition |
| actor weight refresh | every 400 environment steps | not applicable: one process, the collector reads the live net |
| frames | 10B, 256 actors | 50M moves a cell, raised if still rising -- the ordinary budget, not the paper's; R2D2's algorithmic content does not need the actor count and the box has not got it |

D2's **local** cell keeps everything above and swaps in this codebase's plumbing where it exists: PER
0.6, the eval-driven ε, the shield, the fast target. Because R2D2 has no fork and the per-lane ε ladder
is its own exploration answer, the local cell's difference is smaller than Group A's, and it runs only
if the paper cell trails B1.

## 3. The batches

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| D1 | 4 seeds `SNEK_PPO_RECURRENT=lstm` (the `ppo2` reference form, hidden 128) + 4 seeds `gru` | b27's `hist8` PPO config verbatim (the reference; `plans/zigzag-shaping.md` §6 states it) | PPO `hist8` | stage-B density, `hof5000`, `hof30k`, drawdowns; the onset step |
| D1 no-window | 4 seeds of the better cell at `SNEK_OBS_HISTORY=0` | D1 | PPO `hist0` (b7) and D1 | does recurrence replace the window; only if D1 moved |
| D2 | 4 seeds `r2d2` **paper** (§2b: LSTM 512, scalar dueling head, 5-step, Adam 1e-4, target 2,500, the Ape-X ε ladder) + 4 seeds `r2d2` with `SNEK_R2D2_HEAD=c51` (B1's head under the memory) | A1's reward and history | D1, A1 paper and B1 paper | as D1; the scalar cell against A1 is memory alone, the C51 cell against B1 is memory on the stack |
| D2 local | 4 seeds paper with the codebase's PER, ε and target | D2 paper | D2 paper | only if D2 paper trails B1 |

## 4. Gates

1. The seam change lands first, with its tests, and every existing checkpoint still measures to the
   same numbers (a fixed-seed `evaluate.py ... one` on a HOF entry before and after, byte-identical rows).
2. Smoke for both; `watch.py` on a recurrent checkpoint plays a whole game with the state carried.
3. The mutation specs kill every mutant.
4. Tuning budget: one laptop wave each on the hidden width (D1 128 / 256; D2 512 / 256) and, for D2, the
   sequence length (80 / 160, Agent57's).

## 5. What would change the plan

- **D1 beats `hist8`.** Memory beyond eight moves matters; the D1 no-window cell says whether the
  window was a proxy for it, and every later row (E, G's policy prior) is offered the GRU.
- **D1 is level and D2 beats B1.** The memory needs the value agent's machinery -- most likely the
  stored-state replay lets it learn from long endgames the on-policy rollout truncates. E1 and E2 are
  built on D2 as planned.
- **Neither moves.** The 26 features plus eight moves are sufficient statistics for this policy class,
  the failures are not a memory problem, and the plan for E notes that its base (D2) is a null.
