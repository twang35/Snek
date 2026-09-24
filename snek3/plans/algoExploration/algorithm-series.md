# The algorithm series: which learning algorithms to test against Snake, and in what order

**Status: agreed 2026-09-16, nothing built.** This is the ordering and the reasoning only. Each group has
its implementation plan beside this file -- [`README.md`](README.md) indexes them and holds the conventions
every plan leans on: `a-return-tail.md`, `b-entropy.md`, `c-value-stack.md`, `d-data-efficiency.md`,
`e-memory.md`, `f-exploration.md`, `g-planning.md`, `h-gdi.md`.

The question behind the series is not "which algorithm has the highest Atari score". Atari
human-normalised scores mix three regimes -- 200M-frame single-agent runs, 100k-step data-efficiency
runs, and multi-billion-frame distributed runs with hundreds of actors -- and the two things Atari
rewards hardest, sparse exploration and pixel perception, are the two things Snake here does not have.
The question is **which property of a learning algorithm matters for this game**, and the series is
ordered so that each result changes what the next one is expected to show.

## 0. What every group runs, and how long (the user's three objectives, 2026-09-20)

| objective | rule |
|---|---|
| **1. Explore the paper's algorithm** | four arms **as close to the paper's settings as possible** -- its loss and reduction, priorities and importance weights, clipping, optimiser and ε, batch, replay and prefill, target, temperature, schedule and its shape, and every layer width with a counterpart -- translated to this game only where a setting has no meaning here (`README.md`, "Translating a paper's setting"). **This may read worse than the tuned PPO reference, and that is accepted**: the paper cell is the row's headline and it is run as written even when a gate has shown it will fail (b41's 0.98 entropy target) |
| **2. Local tweaks** | four arms with the settings this codebase has found to matter -- the replay plumbing (PER, batch 128, the fast target copy), a target or temperature scaled to three actions, the fork and shield where the algorithm can take them -- to see whether they make a **significant** improvement. `n=4` resolves ~10 pp; smaller differences are noise |
| **3. Time** | **four arms in about 8 hours, at most about 24**, on the desktop's 4-arm waves. Some algorithms need more and are the exception; the usual levers are the update ratio, N, the cap and the lane count, sized from a 2,000-step solo benchmark (`tools/`'s rule: an 8-arm desktop wave runs each arm at ~0.15x the solo rate, a 4-arm wave at ~0.7x) |

So a new paper algorithm's first batch is **4 paper + 4 local**, one wave of two cells, and the later rows of a
group (a paper's own fixes, an ablation) follow the same shape. Where the plan below says "paper cell" and
"local cell", this is what it means.

## 1. What the game is, and what that asks of an algorithm

| property of Snake here | where it is established | the algorithmic axis it points at |
|---|---|---|
| the observation is 26 hand-built values, not the board (`obs26-20260907`) | `docs/environment.md` | the state is **partially observed**: the body layout is summarised, not seen. Memory across steps (recurrence) may recover what the features drop |
| the champions die of one rare fatal move late in a 1,000-2,300 step game, not of a noisy return | `docs/findings.md` (food-sealed pockets, tail-following orbits) | the **tail of the return distribution**, not its mean, is what separates 98% from 100%. Distributional and risk-sensitive value heads target exactly this |
| the transition is deterministic except the food spawn, and the simulator is exact and cheap | `env/game.py`, `vectorized/` | **planning against the true simulator** is available and a learned world model is the wrong version of it. The fixed-path references (100.00% / 30,000, `hallOfFame/HOF.md`) already show lookahead wins here |
| the food is always reachable and the tour proves it | the Hamiltonian-cycle reference | there is **no hard-exploration problem**. Intrinsic-reward agents answer a question this game does not ask, and are in the series to confirm that, not to win |
| two boxes, 8 trainers each, 4 seeds a cell (`docs/protocol.md`) | root `CLAUDE.md` | **sample cost per run** bounds how many rows the series can afford, so the data-efficiency probe runs early |
| PPO is the incumbent and the snek3 default (`algos/ppo/`): **snek3's record is 99.89% /30k** (`b32g`, warm-started from a `hist8` hold) and its plain `hist8` cell reads 99.81% (`b27t`); snek2's 98.7% is the historical mark under the old 30-value observation, not a number any row is read against | `hallOfFame/HOF.md`, `docs/runs.md` b27 | **snek3's PPO is the control** every row is read against -- b27's `hist8` table, on the same 26-value observation, reward preset and eval protocol. snek2 is frozen, its observation era is gone, and its champion no longer loads, so it cannot serve |

The value-based side has its own control already: `algos/dqn/` is in snek3 behind the same algo seam as
`algos/ppo/`, so vanilla DQN is the first row to run, not the first to build.

## 2. The rows, grouped by the question each answers

Removed from the candidate list in review: **Ape-X** (Rainbow plus distributed replay; the algorithmic
content is already in the Rainbow row, the rest is actor count neither box has), **Dreamer V3** (a world
model for pixels and continuous control, on a 26-vector observation with an exact simulator), and
**SAC-BBF** (continuous-action). Added: **QR-DQN** (the missing step between C51 and IQN), **vanilla DQN**
(the value-family control), **Munchausen** (the ingredient Beyond the Rainbow credits most, one loss
term), **AlphaZero-style MCTS on the real simulator** (the planning row that fits the game), and
**recurrent PPO** (so a recurrence result can be separated from a value-based one).

### Group A -- does modelling the return tail help a rare-fatal-move game

| order | row | what it isolates, read against the row above it |
|---|---|---|
| A1 | **DQN** (double, as `algos/dqn/` already is) | the value-family control on the snek3 observation and reward |
| A2 | **C51** | the return distribution itself, on a fixed 51-atom support |
| A3 | **QR-DQN** | dropping the fixed support: the same distribution fitted by quantile regression |
| A4 | **IQN** | sampling the quantile fractions each step; and the **risk-sensitive acting** it allows, run as its own arm (act on the low quantiles only) |
| A5 | **FQF** | learning where the fractions go |
| A6 | **Munchausen** (M-DQN, then M-IQN on the best of A2-A5) | the entropy-regularised bootstrap, separated from the distribution |

The ladder is one knob per rung on purpose. Without A3, a C51-to-IQN gap cannot be attributed. **A2 must
be stable on this reward scale before A3-A5 mean anything:** snek2's C51 was unstable enough that a
win-reward shrink was tried and falsified (`snek2/hyperparamTuning/findings.md`, 2026-08-16), and that
was on the 30-value observation with a different reward configuration. The A4 risk-sensitive arm is the
one property in this whole family that a scalar critic cannot have, and it is the single most direct
test of the "one rare fatal move" diagnosis.

### Group B -- does entropy regularisation on a value agent match PPO's

| order | row | what it isolates |
|---|---|---|
| B1 | **Discrete SAC** | maximum-entropy value learning; whether PPO's advantage here is the entropy bonus rather than the policy gradient |
| B2 | **Revisiting Discrete SAC** | the fixes (entropy-penalty scaling, double average Q clipping) that made it competitive on Atari; whether the B1 result was the idea or the implementation |

### Group C -- does the strongest single-box value stack beat the incumbent

| order | row | what it isolates |
|---|---|---|
| C1 | **Rainbow** | C51 plus double, dueling, prioritised replay, n-step and noisy nets. Whether the classic stack of value tricks, on this game, closes the gap to PPO |
| C2 | **Beyond the Rainbow** | Rainbow's stack rebuilt around IQN, Munchausen, an IMPALA-style trunk and vectorised collection. The practical ceiling of the value family on one box |

C is read against A: if C2 beats C1 by about what A4 plus A6 beat A2 by, the stack adds nothing beyond
its parts. If it beats it by more, the trunk or the collection is doing work worth isolating.

### Group D -- is the late drift a plasticity problem

**Re-planned 2026-09-20.** BBF was in phase 1 as the budget probe (how many steps does any of this need); Group A answered that empirically (onset at 0.3-1.8M steps, batches queued at 2-3M), and BBF's lever, a replay ratio of 8, buys data efficiency with compute on a game whose simulator is free. What Group A left open is the other end of the curve: every value cell reached 88-94% and then drifted or never held. BBF's resets are a late-plasticity mechanism, so the row is now that probe.

| row | what | the question it answers |
|---|---|---|
| D1 | **BBF's shrink-and-perturb resets** as a knob on A6's best cell (`SNEK_RESET_*`, `algos/dqn/resets.py`) | whether periodically pulling the trunk toward a fresh initialisation, and re-initialising the head, lets a value net hold what it reaches. Read against the cell without resets (b40), on hold and drawdowns rather than onset. If it holds, the knob is offered to E2 and re-run on C's best |

### Group E -- does memory over the body matter given 26 features

| order | row | what it isolates |
|---|---|---|
| E1 | **Recurrent PPO** | recurrence alone, on the incumbent. The `hist8` result (making the last eight turns visible was the largest lever in the project, `docs/findings.md`) is the prior that says memory should matter |
| E2 | **R2D2** | recurrence on a value agent, with the burn-in and stored-state machinery. Read against E1 and C1: whether the gain is the memory or the agent |

E2 is R2D2 the algorithm -- recurrent replay with stored states and burn-in -- not R2D2 the 256-actor
deployment. Run it at the actor count the box has.

### Group F -- does exploration machinery matter when there is no exploration problem

| order | row | what it isolates |
|---|---|---|
| F1 | **NGU** | episodic-novelty intrinsic reward on R2D2. Kept in the series to *confirm* it changes nothing, and to see whether the intrinsic term hurts a game where the shortest route is already the right one |
| F2 | **Agent57** | NGU plus a meta-controller over the exploration/exploitation family. Read against F1 and E2: whether the meta-controller recovers what NGU's intrinsic term costs |

Both are built on E2, so they cannot run before it.

### Group G -- does planning with the true simulator win outright

| order | row | what it isolates |
|---|---|---|
| G1 | **AlphaZero-style MCTS on the real `Game`** | search with an exact model, a learned policy and value prior. This is the row most likely to beat the PPO record (99.89% /30k), and the row whose result the fixed-path references most nearly predict |
| G2 | **MuZero** | the same search with a *learned* model. Read against G1: what learning the model costs when the true one was free |
| G3 | **EfficientZero V2** | MuZero's data-efficient form. Read against G2 and D1: whether the search or the sample-efficiency tricks carry it at low step counts |
| G4 | **Muesli** | the policy-gradient relative: MuZero's model used for a regularised policy update instead of search at act time. Whether the model helps the *update* even when there is no search at play time |

G1 has a wrinkle the others do not: **the eval protocol measures a policy network, and G1's agent is a
search**. The plan for G1 has to decide whether the reported number is the network acting alone (the
distilled policy), the search acting with a fixed budget, or both, and the HOF has to say which. The
fixed-path rows in the HOF already carry a "no policy" reference form, so there is a precedent.

### Group H -- the outlier

| order | row | what it isolates |
|---|---|---|
| H1 | **GDI / LBC** | a generalised data-distribution optimisation over a learned behaviour family; the highest reported mean on Atari and the least reproduced. Kept because the mean is interesting. Budget it as a reproduction first: if the reference result cannot be reproduced on a known Atari game within one wave, the row is closed as "not reproducible here", which is itself the finding |

## 3. The ordering, and why

Reading order is by group above; **running order** keeps each group together and runs a group only once
the groups it is read against have closed. The order is chosen to make the progression legible, not to
finish fastest: a row that could run earlier on its dependencies alone still waits for its group.

| phase | rows | why here |
|---|---|---|
| 1 | **A1** DQN | already built, and the control for every value row |
| 2 | **A2 → A3 → A4 → A5**, then **A6** | the distributional ladder, one rung at a time, with A2 stabilised on this reward before A3 starts. A4's risk-sensitive arm is the series' most direct test of the diagnosis in §1 |
| 3 | **B1 → B2** discrete SAC | one change to a value agent that exists, read against A1 and PPO |
| 4 | **C1 → C2** Rainbow, Beyond the Rainbow | read against A, so they wait for A. C2 also wants A6's Munchausen result |
| 5 | **D1** BBF's resets | the reset probe on A6's best cell: whether the late drift every A cell showed is a plasticity problem. Placed after C so its answer is read beside the strongest value stack, and offered to E2 |
| 6 | **E1** recurrent PPO, then **E2** R2D2 | the memory question in one phase. E1 is the cheaper form with the `hist8` prior behind it and is read against PPO; E2 is read against E1 and C1, the value stack it is built on |
| 7 | **F1 → F2** | built on E2; the expected result is a null, so they run once the value rows they are read against have closed |
| 8 | **G1**, then **G2 → G3**, **G4** | the planning group in one phase. G1 is the row most likely to beat the record and needs the eval-protocol decision in Group G made first; the learned-model rows are read against G1's number |
| 9 | **H1** | a reproduction gate first, then the row if it passes |

Two ordering rules that hold across phases: **no row starts until the row it is read against has a
closed stage-B number** (`docs/protocol.md`), and **a row that fails to stabilise is closed as a
finding, not retuned indefinitely** -- the sweep history (`docs/sweep.md`) is the precedent for how much
tuning one row deserves before the answer is "not on this game".

## 4. What every row reports

Each row is measured under the standing protocol so the rows are comparable to each other and to PPO:
the same 26-value observation, the same reward configuration as the current PPO reference, the same
stage-A and stage-B evals, 4 seeds a cell, the true perfect rate at depth and the number of drawdowns
as the deciding numbers rather than the best single eval. Each row's design plan adds only what is
specific to it: G1's search-versus-network decision, A4's risk-sensitive acting arm, H1's reproduction
gate.

**Each row emulates its paper (decided 2026-09-16).** The first cell of every row is the paper's own
configuration -- optimiser, learning rate, batch, replay, target period, exploration schedule, head
sizes -- translated to Snake only where a setting cannot transfer (reward clipping, the discount, frame
budgets, the CNN trunk), by the rules in [`README.md`](README.md) "Translating a paper's setting to
Snake". This codebase's own additions to value learning -- the fork, the exploration shield, the
eval-driven epsilon, the fast target copy -- are **off** in that cell and on in a second, *local* cell,
so the gap between an algorithm and PPO is not confounded with the gap between the paper's plumbing and
snek3's. Every plan carries a "paper settings" table with the verified values and their sources.

The series' own summary lives in `docs/findings.md` when rows close, one line per row against PPO; this
plan is the ordering and does not carry results.
