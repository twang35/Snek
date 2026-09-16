# The algorithm series: which learning algorithms to test against Snake, and in what order

**Status: agreed 2026-09-16, nothing built.** This is the ordering and the reasoning only. Each row
becomes its own design plan when its turn comes; implementation is not in scope here.

The question behind the series is not "which algorithm has the highest Atari score". Atari
human-normalised scores mix three regimes -- 200M-frame single-agent runs, 100k-step data-efficiency
runs, and multi-billion-frame distributed runs with hundreds of actors -- and the two things Atari
rewards hardest, sparse exploration and pixel perception, are the two things Snake here does not have.
The question is **which property of a learning algorithm matters for this game**, and the series is
ordered so that each result changes what the next one is expected to show.

## 1. What the game is, and what that asks of an algorithm

| property of Snake here | where it is established | the algorithmic axis it points at |
|---|---|---|
| the observation is 26 hand-built values, not the board (`obs26-20260907`) | `docs/environment.md` | the state is **partially observed**: the body layout is summarised, not seen. Memory across steps (recurrence) may recover what the features drop |
| the champions die of one rare fatal move late in a 1,000-2,300 step game, not of a noisy return | `docs/findings.md` (food-sealed pockets, tail-following orbits) | the **tail of the return distribution**, not its mean, is what separates 98% from 100%. Distributional and risk-sensitive value heads target exactly this |
| the transition is deterministic except the food spawn, and the simulator is exact and cheap | `env/game.py`, `vectorized/` | **planning against the true simulator** is available and a learned world model is the wrong version of it. The fixed-path references (100.00% / 30,000, `hallOfFame/HOF.md`) already show lookahead wins here |
| the food is always reachable and the tour proves it | the Hamiltonian-cycle reference | there is **no hard-exploration problem**. Intrinsic-reward agents answer a question this game does not ask, and are in the series to confirm that, not to win |
| two boxes, 8 trainers each, 4 seeds a cell (`docs/protocol.md`) | root `CLAUDE.md` | **sample cost per run** bounds how many rows the series can afford, so the data-efficiency probe runs early |
| PPO is the incumbent at 98.7% (snek2) and is the snek3 default (`ppo/`) | `hallOfFame/HOF.md` | PPO is the **control** every row is read against, on the same observation, reward and eval protocol |

The value-based side has its own control already: `dqn/` is in snek3 behind the same algo seam as
`ppo/`, so vanilla DQN is the first row to run, not the first to build.

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
| A1 | **DQN** (double, as `dqn/` already is) | the value-family control on the snek3 observation and reward |
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

### Group B -- does the strongest single-box value stack beat the incumbent

| order | row | what it isolates |
|---|---|---|
| B1 | **Rainbow** | C51 plus double, dueling, prioritised replay, n-step and noisy nets. Whether the classic stack of value tricks, on this game, closes the gap to PPO |
| B2 | **Beyond the Rainbow** | Rainbow's stack rebuilt around IQN, Munchausen, an IMPALA-style trunk and vectorised collection. The practical ceiling of the value family on one box |

B is read against A: if B2 beats B1 by about what A4 plus A6 beat A2 by, the stack adds nothing beyond
its parts. If it beats it by more, the trunk or the collection is doing work worth isolating.

### Group C -- does entropy regularisation on a value agent match PPO's

| order | row | what it isolates |
|---|---|---|
| C1 | **Discrete SAC** | maximum-entropy value learning; whether PPO's advantage here is the entropy bonus rather than the policy gradient |
| C2 | **Revisiting Discrete SAC** | the fixes (entropy-penalty scaling, double average Q clipping) that made it competitive on Atari; whether the C1 result was the idea or the implementation |

### Group D -- does memory over the body matter given 26 features

| order | row | what it isolates |
|---|---|---|
| D1 | **Recurrent PPO** | recurrence alone, on the incumbent. The `hist8` result (making the last eight turns visible was the largest lever in the project, `docs/findings.md`) is the prior that says memory should matter |
| D2 | **R2D2** | recurrence on a value agent, with the burn-in and stored-state machinery. Read against D1 and B1: whether the gain is the memory or the agent |

D2 is R2D2 the algorithm -- recurrent replay with stored states and burn-in -- not R2D2 the 256-actor
deployment. Run it at the actor count the box has.

### Group E -- does planning with the true simulator win outright

| order | row | what it isolates |
|---|---|---|
| E1 | **AlphaZero-style MCTS on the real `Game`** | search with an exact model, a learned policy and value prior. This is the row most likely to beat 98.7%, and the row whose result the fixed-path references most nearly predict |
| E2 | **MuZero** | the same search with a *learned* model. Read against E1: what learning the model costs when the true one was free |
| E3 | **EfficientZero V2** | MuZero's data-efficient form. Read against E2 and F1: whether the search or the sample-efficiency tricks carry it at low step counts |
| E4 | **Muesli** | the policy-gradient relative: MuZero's model used for a regularised policy update instead of search at act time. Whether the model helps the *update* even when there is no search at play time |

E1 has a wrinkle the others do not: **the eval protocol measures a policy network, and E1's agent is a
search**. The plan for E1 has to decide whether the reported number is the network acting alone (the
distilled policy), the search acting with a fixed budget, or both, and the HOF has to say which. The
fixed-path rows in the HOF already carry a "no policy" reference form, so there is a precedent.

### Group F -- how much compute does any of this need

| order | row | what it isolates |
|---|---|---|
| F1 | **BBF** | the data-efficiency probe: a value agent tuned for 100k steps (bigger net, reset schedules, annealed n-step and gamma). Whether Snake at 26 features needs anywhere near the steps the incumbent uses |

F1 is early in the ordering (§3) because its answer sizes every later row's budget.

### Group G -- does exploration machinery matter when there is no exploration problem

| order | row | what it isolates |
|---|---|---|
| G1 | **NGU** | episodic-novelty intrinsic reward on R2D2. Kept in the series to *confirm* it changes nothing, and to see whether the intrinsic term hurts a game where the shortest route is already the right one |
| G2 | **Agent57** | NGU plus a meta-controller over the exploration/exploitation family. Read against G1 and D2: whether the meta-controller recovers what NGU's intrinsic term costs |

Both are built on D2, so they cannot run before it.

### Group H -- the outlier

| order | row | what it isolates |
|---|---|---|
| H1 | **GDI / LBC** | a generalised data-distribution optimisation over a learned behaviour family; the highest reported mean on Atari and the least reproduced. Kept because the mean is interesting. Budget it as a reproduction first: if the reference result cannot be reproduced on a known Atari game within one wave, the row is closed as "not reproducible here", which is itself the finding |

## 3. The ordering, and why

Reading order is by group above; **running order** is by what each result unlocks and what it costs.

| phase | rows | why here |
|---|---|---|
| 1 | **A1** DQN, **F1** BBF | A1 is already built and is the control for every value row. F1 says how many steps every later row needs. Both are cheap and neither depends on anything |
| 2 | **A2 → A3 → A4 → A5**, then **A6** | the distributional ladder, one rung at a time, with A2 stabilised on this reward before A3 starts. A4's risk-sensitive arm is the series' most direct test of the diagnosis in §1 |
| 3 | **D1** recurrent PPO, **C1 → C2** discrete SAC | both are one change to an agent that exists. D1 in particular is the cheapest row with a strong prior behind it, and its result decides how much D2 is worth |
| 4 | **E1** AlphaZero MCTS | the row most likely to beat the record. It needs the eval-protocol decision in Group E made first, which is why it is not in phase 1 despite being independent of every other row |
| 5 | **B1 → B2** Rainbow, Beyond the Rainbow | read against A, so they wait for A. B2 also wants A6's Munchausen result |
| 6 | **D2** R2D2 | waits for D1 (is memory worth a second, heavier form) and B1 (the value stack it is built on) |
| 7 | **E2 → E3**, **E4** | the learned-model rows wait for E1's number, which is what they are read against |
| 8 | **G1 → G2** | built on D2; run last among the value rows because the expected result is a null |
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
specific to it: E1's search-versus-network decision, A4's risk-sensitive acting arm, H1's reproduction
gate.

The series' own summary lives in `docs/findings.md` when rows close, one line per row against PPO; this
plan is the ordering and does not carry results.
