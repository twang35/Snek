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
| knobs | `SNEK_PPO_RECURRENT` (0/off; `gru`), `SNEK_PPO_RECURRENT_HIDDEN` (128), `SNEK_PPO_SEQ_MINIBATCH` (lanes per minibatch, 16). Feed-forward PPO is exactly what it was at the defaults, and a fixture asserts a rollout and an update are byte-identical with the knob off |

Tests: a GRU replay from the stored state reproduces the log-probs stored at collection (the ratio is
1 on the first epoch, to tolerance); a `done` inside a chunk zeroes the state; the recurrent net with
`hidden` = 0 is refused rather than silently feed-forward. Mutants: the state not zeroed on `done`, the
minibatch shuffling transitions, the stored state off by one step.

**The base is `hist8`, not `hist0`.** The question is whether memory adds to what the history window
already gives, since that is the incumbent. A second cell at `SNEK_OBS_HISTORY=0` asks whether
recurrence *replaces* the window; it runs only if the first cell moves.

### D2 -- R2D2 (Kapturowski, Ostrovski, Quan, Munos & Dabney 2019)

Recurrent replay distributed DQN: an LSTM Q-network trained on fixed-length sequences from replay,
each stored with the recurrent state at its start, with a **burn-in** prefix that is replayed to warm
the state before the loss is taken over the rest, n-step targets, PER with a sequence priority
(η-mix of max and mean absolute TD), and the value rescaling h(x) = sign(x)(√(|x| + 1) − 1) + εx.
Here it is R2D2 the algorithm, at the box's actor count, on B1's Rainbow head.

| module | contents |
|---|---|
| `algos/r2d2/net.py` | trunk → LSTM(`hidden`) → B1's dueling C51 head (or the scalar head, `SNEK_R2D2_HEAD=scalar`, for the ablation) |
| `algos/r2d2/replay.py` | a sequence buffer over `algos/dqn/replay.py`'s sum tree: entries are `(burn_in + length)`-step windows with the stored initial state; priorities per sequence. New; the transition buffer is reused for the tree only |
| `algos/r2d2/collect.py` | `algos/dqn/collect.py`'s lanes carrying an LSTM state, cutting sequences at `SNEK_R2D2_SEQ_LENGTH` with overlap `SNEK_R2D2_SEQ_OVERLAP`; **no fork** (a forked lane would need a copied state and a copied sequence prefix; refused by name) |
| `algos/r2d2/agent.py` | burn-in replay under `no_grad`, the n-step target on the remainder, the rescaling and its inverse, the sequence priority |
| knobs | `SNEK_R2D2_HIDDEN` (256), `SNEK_R2D2_SEQ_LENGTH` (80), `SNEK_R2D2_BURN_IN` (40), `SNEK_R2D2_SEQ_OVERLAP` (40), `SNEK_R2D2_PRIORITY_ETA` (0.9), `SNEK_R2D2_RESCALE` (1), `SNEK_R2D2_HEAD` (`c51`); DQN's names for the rest |
| the step | one `collector.step()`, `collect_envs` moves |

Tests: the rescaling and its inverse compose to the identity; the burn-in leaves parameters without
gradient (a fixture checks `.grad` is None after a burn-in-only pass); the sequence priority at η = 1 is
the max and at η = 0 the mean; a stored state replayed over the burn-in matches the state the collector
had at the loss window's start when the weights have not changed. Mutants: burn-in included in the
loss, the inverse rescaling skipped on the target, overlap producing a gap instead.

## 3. The batches

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| D1 | 4 seeds `SNEK_PPO_RECURRENT=gru` | b27's `hist8` PPO config verbatim (the reference; `plans/zigzag-shaping.md` §6 states it) | PPO `hist8` | stage-B density, `hof5000`, `hof30k`, drawdowns; the onset step |
| D1 no-window | 4 seeds at `SNEK_OBS_HISTORY=0` | D1 | PPO `hist0` (b7) and D1 | does recurrence replace the window; only if D1 moved |
| D2 | 4 seeds of `r2d2` | A1's reward and history; B1's head and optimiser | D1 and B1 | as D1 |
| D2 scalar head | 4 seeds `SNEK_R2D2_HEAD=scalar` | D2 | A1 and D2 | recurrence on a plain DQN, so the gain over A1 is memory alone |

## 4. Gates

1. The seam change lands first, with its tests, and every existing checkpoint still measures to the
   same numbers (a fixed-seed `evaluate.py ... one` on a HOF entry before and after, byte-identical rows).
2. Smoke for both; `watch.py` on a recurrent checkpoint plays a whole game with the state carried.
3. The mutation specs kill every mutant.
4. Tuning budget: one laptop wave each on the hidden width and, for D2, the sequence length.

## 5. What would change the plan

- **D1 beats `hist8`.** Memory beyond eight moves matters; the D1 no-window cell says whether the
  window was a proxy for it, and every later row (E, G's policy prior) is offered the GRU.
- **D1 is level and D2 beats B1.** The memory needs the value agent's machinery -- most likely the
  stored-state replay lets it learn from long endgames the on-policy rollout truncates. E1 and E2 are
  built on D2 as planned.
- **Neither moves.** The 26 features plus eight moves are sufficient statistics for this policy class,
  the failures are not a memory problem, and the plan for E notes that its base (D2) is a null.
