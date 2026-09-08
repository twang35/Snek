# Short-term memory: showing the policy its last four moves

**Written 2026-09-07, revised the same day after review. Phase 1 ran the same day and found the
feature addresses a failure the best checkpoints do not have (see "Phase 1 result"). Built anyway the
same evening, at the user's request, to measure it: section 4 is as built, and batch b27 runs it.** The question asked was: give
snek a view of its last four moves, cheaper than four one-hot triples (12 inputs), and is there a
better approach altogether. The answer is in section 2; the rest is what it costs and how to test it.

**Decided in review (2026-09-07):**

| decision | |
|---|---|
| encoding | option B — two bits per move, read off the body, `SNEK_OBS_HISTORY` gates the block |
| purpose | **fewer zigzags, and through them a higher perfect rate.** Not aliasing loops; section 3 is written for that |
| convention | the new inputs are *descriptive*, not "1 is good"; `docs/environment.md` says so |
| `ended_by` | goes into the stage-B row permanently, alongside this work |
| **phase 1 result** | **the best checkpoints starve; they do not collide.** 257 of 280 failures across b17cl and b10ck at 30,000 games each are starves, nearly all an exact closed loop of the snake following itself while the food sits reachable. History cannot break a loop of period 8-98, and the policy is confident on every lap. **Sections 4-5 are not built.** The measurement and its numbers: `docs/findings.md`, "The best checkpoints fail by starving in a closed loop"; pictures `charts/deaths/` |
| order | **investigation first, as a gate.** Phase 1 is sections 3 and 3b; nothing in section 4 is built until phase 1 has said how the best checkpoints die |
| now | **built and queued as b27** (2026-09-07 evening): depths 0 / 4 / 8 × 8 seeds on b26's base with step penalty 0.01, 100M. Prediction on record: no effect on the perfect rate, since the dominant failure is a loop of period length + 1. The reward-side follow-up (b26's step penalty) runs alongside |

**Two predictions, registered before the data.** The user expects the best checkpoints to die mostly by
**starving**; this plan's first draft assumed **collisions**. Phase 1 is designed to say which, and
the answer decides the feature: history addresses the traps a snake builds for itself, and does
nothing for a snake that cannot get to its food.

| if phase 1 finds | then |
|---|---|
| mostly forced collisions, with zigzag-laid walls | build section 4, run section 5 |
| mostly forced collisions, walls laid straight | zigzag is not the mechanism; history is the wrong feature. Write it up, stop |
| mostly food-inaccessible deaths (starves and sealed-food collisions, below) | a different problem — waiting for the tail to free the food. History does not help; the plan closes and a new one opens. **← this is what phase 1 found: 96% and 87%** |

## 0. The one-line recommendation

**Do not add a history buffer. Read the last moves off the body.** The four moves that put the head
where it is *are* the shape of the first five body cells behind it, so the feature is a pure function
of the board like every other one of the 30. That removes every piece of new state — nothing to reset,
snapshot, or keep in parity between the two env implementations — and it costs one gather per move.
Encode each move as two bits, `[turned left, turned right]`, forward being `(0, 0)`: **8 inputs for
four moves**, gated by one env key so a batch sweeps it as a one-knob sweep against the current
vector.

## 1. What the 12 features would cost, and where the cost actually is

The concern was that 12 inputs is inefficient. Compute-wise it is nothing: the first layer is
`30 -> 320 -> 3` and grows to `42 -> 320 -> 3`, adding 3,840 weights to a network of ~10.9k — a third more first-layer weights, no measurable time. The costs are
elsewhere:

| cost | why |
|---|---|
| **it changes the MDP** | every checkpoint restores against the vector it trained on. A longer vector stops every HOF checkpoint and every converted snek2 champion from loading — `docs/invariants.md` §3. Not a reason not to do it, but the new block must be **optional and off by default** so the current era stays loadable |
| **two implementations** | `env/observations.py` is the reference and `vectorized/vec_env.py` the fast copy; `tests/test_vec_parity.py` holds them equal over 18,053 states. Any new block is written twice and pinned by that test |
| **new per-game state** (only if done as an action buffer) | a ring of the last four actions has to be reset with the game, threaded through `Game.snapshot`/`restore_snapshot`, and kept in lockstep in `VecEnv._reset_rows`. Every one of those is a place the two implementations can drift. **Section 2 avoids all of it** |
| **a block at constant length would need an era bump** | not the case here; a longer vector fails `load_state_dict(strict=True)` loudly. But `arch.json` must record the history depth so `tools/restore.py` rebuilds the env the checkpoint expects |

So the 12-vs-8-vs-4 question is the least important one. The important one is *derived state vs
stored state*.

## 2. The options, and why body-derived two-bit wins

The move made at step *t* is the turn between the body's direction into cell *k+1* and its direction
into cell *k*. With `START_SEGMENTS = 4` the snake begins as five cells in a straight line, so at
step 0 the body reads "forward, forward, forward, forward" — exactly what a history buffer
initialised to forward would say. When the body is shorter than the depth asked for (never, at depth
4), the missing moves read forward.

| option | inputs / 4 moves | keeps | verdict |
|---|---|---|---|
| **A. one-hot triple per move** (the proposal) | 12 | exact sequence | fine, but 4 of the 12 are the complement of the other 8 |
| **B. `[turned left, turned right]` per move, forward = `(0,0)`** | **8** | exact sequence | **recommended.** Same information as A; a ReLU layer reads "went straight" from the two zeros through its bias |
| C. one signed scalar per move, `-1 / 0 / +1` | 4 | exact sequence | compact, and left/right *are* geometrically opposite so the ordering is meaningful — but it breaks the vector's `[0, 1]` convention, and the net must learn that 0 is a category not a midpoint |
| D. summaries: net rotation over the window, count of turns, steps since last turn | 2–3 | lossy | cheapest, but it *prejudges* what memory is for (circling). Try this as the follow-up ablation once B shows an effect, not first |
| E. two decaying traces, `t = d*t + 1{turned}` | 2 | soft, infinite horizon | elegant, uninterpretable, and one more knob (`d`) to sweep |
| F. recurrent actor (GRU) | 0 new inputs | learned | **not this.** Breaks the `policy_fn: (m, OBS_LEN) -> (m,)` seam the eval engine and both trainers are built on, breaks "the actor is exactly `QNet`", and PPO's rollout would need sequence batching and hidden-state storage. A different project |
| G. stack the last four 30-vectors | 120 | the situation, not the actions | expensive, and consecutive observations do not name the action taken between them |

**Derived from the body (B) versus a stored ring buffer (any of A–E):** identical values in every
reachable state, so this is purely an implementation choice — and the derived form has no reset path,
no snapshot field, no parity trap, and no vectorised buffer to keep in step with autoreset. It is also
the honest description of the feature: it is a *body-shape* feature, not memory.

## 3. What the feature is for: zigzags

The motivation is **zigzagging** — the snake alternating left and right turns through open space
rather than travelling straight or along an edge. The hypothesis, the same one indices 23-25 were
added on, is that a zigzag carves the free space into pockets that are harder to fill later, and that
the cost lands at the end of the game as a missed perfect. A memoryless policy cannot see that it is
zigzagging: each step's 30 values describe the board ahead, and a left after a right looks the same as
a left after a left. Four moves of history are exactly what makes the pattern visible to it.

Two consequences for how this is judged:

- **The headline metric is the perfect rate, the same as every batch.** History is *enabling*: it lets
  the policy tell a zigzag from a straight run, and nothing else rewards one over the other. If the
  perfect rate does not move, the feature failed, however the paths look.
- **The mechanism check is a zigzag measure, taken from the same games.** Without it a gain cannot be
  attributed to straighter paths rather than to noise or to something else the extra inputs enable.

The measure, computed from the action sequence of each eval episode:

| measure | definition |
|---|---|
| turn density | turns per step |
| **reversal rate** | a left within *k* steps of a right or vice versa, per step; `k = 2` is the zigzag proper |
| mean straight run | steps between turns |

**The evals do not currently record any of this, nor how a game ended.** A stage-B row carries scores
and the perfect rate; `VecEnv.step`'s `info` carries `died / starved / perfect` per game and
`vectorized/engine.py` drops it; the actions are never kept. So the first step, before any observation
change, is a read-only diagnostic on b25's ladder top over 5,000 episodes: the three measures above,
split by outcome (perfect / collision / starve), and the reversal rate **as a function of board fill**.
That tells us two things the sweep needs — whether the champion zigzags at all, and whether it does so
where the hypothesis says the damage is done (the crowded endgame) or in the open early board where it
costs nothing. It is also the baseline the treatment arms are read against.

### 3b. The failures: was there a zigzag in the last 200 steps before death?

Zigzagging does not kill by itself, so the question that matters is the converse: **when a strong
checkpoint does die, was a zigzag involved?** Three things make this answerable, and two make it easy
to answer wrongly.

**Finding deaths is not the hard part.** At 98.7% perfect, 5,000 episodes hold ~65 failures and the
vectorised engine plays that in minutes; 20,000 episodes give ~250, which is enough for every
comparison below. **Reproducing them is free**, because the measured policy is the argmax and the
game is deterministic given its food sequence: record `(seed, food cells in order, actions)` for every
episode that ends in `died` or `starved`, and each one replays exactly to any step, board and all.
That is a small trace hook in `vectorized/engine.py` writing failures only — perfect games are kept as
the summary measures of section 3, not as traces.

**The two ways to get it wrong:**

| trap | why | the control |
|---|---|---|
| **the endgame is crowded** | turn density rises with board fill in *every* game, so "the last 200 steps before death had many reversals" is true of the last 200 steps of a perfect game too | compare each death window against windows from perfect games **at the same board fill**, not against the early board |
| **the death itself is a turn** | the last few moves into a dead end are forced turns, and a 200-step window ending at death is dominated by the trap, not by what caused it | split the window at the **point of no return** (below) and ask the zigzag question of the steps *before* it |

**The point of no return.** For a collision death, replay the episode and walk back from the last step
to the last one at which the chosen move still left the head able to reach the tail — index 9-14's
`can reach tail` for the action taken, or a flood fill on the replayed board. Everything after it is
the trap closing; everything before it is where the trap was *built*. Then the sharp version of the
question: **the pocket that killed the snake is walled by its own body; which body segments form that
wall, and what turns laid them down?** Segment *k* behind the head was placed *k* steps ago, so the
wall's segments index straight into the action sequence, and the answer is a number: the reversal rate
of the moves that built the fatal wall, against the reversal rate of the moves that built the rest of
the body. If zigzagging is the mechanism, that number is high and the matched-fill control is not. If
the wall was laid by long straight runs, zigzagging is not what kills this policy and history is the
wrong feature — a finding worth as much as the other outcome.

**Three kinds of death, not two.** A collision is not always a trap. If the food sits sealed in a
pocket of its own — index 29 at 0, or more generally the region the fatal move enters holds the food
and no way out — then eating it and dying the next step is a *choice*: nothing forced the snake in,
and had it declined it would have circled until the starve budget ran out. That death is a starve
with a different ending, and counting it as a collision would credit the wrong mechanism. So the
outcome split is:

| category | test on the replay | root cause |
|---|---|---|
| **forced collision** | at the point of no return the alternative that kept the tail reachable did **not** contain the food; the snake walled itself in | path planning — the case history could help |
| **sealed-food collision** | the fatal pocket contains the food, or the fatal move (or one within a few steps of it) eats; the safe alternative would have meant not eating | food inaccessible — the snake traded its life for the meal |
| **starve** | 500 steps without eating | food inaccessible — it declined the trade, or never found a way |

Sealed-food collisions and starves are then reported **together as "food inaccessible"** against
forced collisions, since that is the split the user's prediction and this plan's disagree on. Within
food-inaccessible, keep the sub-split: how often the snake takes the fatal meal versus waits it out is
its own finding about the reward, since death costs -5 and the food pays +1.

Starves get the same trace but a different question, since nothing trapped the head: is the snake
circling (a repeated `(head, head_dir)` cycle — the aliasing case of the first draft) or reaching but
never entering the food's region? Those are two different features, and neither is move history
unless the cycle is four moves long.

**What gets produced:** one table — deaths by the three categories above, with the pre-trap reversal rate, the fatal-wall
reversal rate, and the matched-fill perfect-game rate beside them — and a **contact sheet** of the
failures: the final 200 steps of each as one still, head path coloured by time, wall segments marked,
so the numbers can be checked by eye on the dozens of games there are. `record_gif.py` and
`Game.snapshot` already do the replay and the drawing; the sheet is a layout on top of them.

`ended_by` and the three path measures then go into the stage-B row permanently, so the close-out of
every future batch carries them.

## Phase 1 result (2026-09-07)

Both HOF leaders, 30,000 episodes each, every failure replayed (`tools/death_trace.py`,
`tools/death_analyze.py`, `tools/death_sheet.py`; the contact sheets are
[`../charts/deaths/b17cl-30k-s11-failures.png`](../charts/deaths/b17cl-30k-s11-failures.png) and
[`../charts/deaths/b10ck-30k-s11-failures.png`](../charts/deaths/b10ck-30k-s11-failures.png)).

| | b17cl | b10ck |
|---|---:|---:|
| failures / 30,000 | 153 | 127 |
| **starved** | **147** | **110** |
| sealed-food collision | 5 | 17 |
| forced collision | 1 | 0 |
| starves that are an exact `(cell, heading)` cycle | 127 | 70 |
| … with period = length + 1 (the snake one gap behind its own tail) | 32 | 37 |
| safe approach existed on this share of starve-window steps | 26% | 15% |
| logit margin over the declined safe approach, median | 2.5 | 3.0 |
| argmax changed by forcing the starve-budget input to 0 | 1.7% | 6.1% |
| pre-window reversal rate minus matched-fill control | +2 pp | +3 pp |

What the pictures show: a serpentine or ring **around an empty room with the food inside**, orbiting
(b10ck's shape), and the endgame loop **past a food sealed in a one- or two-cell pocket** beside the
path (both). The snake is confident in the loop, the food is reachable, and the observation repeats
exactly every lap except for a starve-budget value compressed into the top of its range.

**Verdict on this plan:** zigzags are not what kills these policies, and a four-move history cannot
break a loop whose period is the snake's length. Not built. The "tail-chasing local optimum" that
indices 26-28 were added to price is real and is *the* failure mode — but that block only fires when
the head lands on the cell the tail vacates this step (period = length), and the dominant loop runs one
cell behind (period = length + 1), which it does not see.

**Where a new plan starts:** the reward. Starving costs −0.5, a death −5, and there is no per-step
cost, so an uncertain approach is worse in expectation than an orbit — the loop is what the reward
asks for. Candidates, in order: a starve penalty of the same order as a death; a per-step cost; a
linear rather than lg-compressed starve-budget input (or steps-since-food). All three are judged on
the same trace tooling: the starve count falls and the sealed-food count does not rise.

## 4a. As built (2026-09-07)

| piece | where | what differs from the design |
|---|---|---|
| knob | `env/constants.py` `OBS_HISTORY`, `OBS_LEN = 26 + 2N`, `OBS_BLOCKS` gains `('move_history', 2N)` | as designed |
| era | `OBS_ERA = 'obs26-20260907' + '-histN'` | **the depth rides in the era, not in a new `arch.json` field** — every field of the sidecar is required, so a new one would have invalidated b26's sidecars. `tools/sidecar_env.py` parses it |
| reference | `env/observations.py` `move_history_obs(body_positions, depth)`; `get_observations` gains `body_positions=None` | the grid has no body order, so the game passes `snake.get_positions()` |
| vectorised | `vectorized/vec_env.py` `move_history_bits`, `REL` (the inverse of `TURN`), `DIRCODE` | as designed; elementwise parity over the 18,053 states at depth 4 |
| entry points | `tools/shard.py`, `evaluate.py`, `watch.py`, `record_gif.py`, `tools/death_trace.py` call `sidecar_env.adopt_from_argv` before importing the env | **new.** The design said "restore sets the env key"; restore runs after the env is imported, so the setting has to happen at the top of the process |
| shared eval workers | `tools/scheduler.py` `wave_obs_history`: workers get the wave's depth; a wave that mixes depths gets none and its arms run stage A in-process | **new, and the one real constraint**: one worker process serves every arm on the box and one width. A depth sweep puts one depth per wave of eight |
| tests | `tests/test_move_history.py`: values on hand-built bodies, the vectorised form against the reference on driven lanes, the era and sidecar, the scheduler rule, and the layout + parity suites re-run at depth 4 from a subprocess | |

## 4. The build, if section 3 says go

| step | where | what |
|---|---|---|
| 1 | `env/constants.py` | `OBS_HISTORY = _env('OBS_HISTORY', 0, int)`; `OBS_BLOCKS` appends `('move_history', 2 * OBS_HISTORY)` when nonzero; `OBS_LEN` becomes `observation_length()`. Off by default, so the current era is untouched and every checkpoint still loads |
| 2 | `env/observations.py` | `move_history_obs(body_cells, head_dir, depth)` — the reference, from the head and the `depth + 1` cells behind it. Appended at the end, per the "new blocks go on the end" rule |
| 3 | `vectorized/vec_env.py` | the same from `self.body` and `self.hp`, a `(n, depth+1)` gather on the circular buffer, `TURN`'s inverse to name the turn between consecutive directions |
| 4 | `tests/` | `test_observation_layout.py` pins the block's range; `test_observations.py` pins the values on a hand-drawn body (straight, L, U, spiral, and a snake shorter than the depth); `test_vec_parity.py` runs its existing 18,053 states at depth 4 as well as 0 |
| 5 | `tools/arch.py`, `tools/restore.py` | `arch.json` gains `obs_history`; restore sets the env key from it before building the env, and refuses a mismatch with a clear message. Existing sidecars without the field read as 0 — the one time a missing field is allowed, since every committed sidecar is depth 0 |
| 6 | `docs/environment.md`, `docs/invariants.md` | the table row and the era note |

Two things stay exactly as they are: `dqn/agent.py`'s exploration shield reads `block_ranges()` by
name and is unaffected by an appended block, and the frozen diagnostic scripts index by position and
are unaffected for the same reason.

## 5. The experiment

One knob, dense, four seeds, on the b25 base — the protocol every sweep here has used:

| arm | `SNEK_OBS_HISTORY` | inputs |
|---|---|---|
| control | 0 | 30 |
| | 2 | 34 |
| | 4 | 38 |
| | 8 | 46 |

Judged on the b25 numbers (true rate at depth, drawdowns), plus the section-3 path measures at the
close-out, so an improvement can be attributed to straighter paths rather than assumed — and so a
*flat* perfect rate with a *falling* reversal rate is read as "it stopped zigzagging and that was not
the problem", which is a finding too.

**One optional fifth arm, worth discussing:** the other lever for fewer zigzags is a small per-turn
penalty in the reward, which needs no observation change and no new era. Running it beside the history
arms would say whether the policy needs to *see* its zigzags or merely to be charged for them. It is a
reward change, so it is a different knob and does not belong in the depth sweep proper; it is listed
here so the comparison is not forgotten. Depth is the real
unknown; the encoding is not, which is why the encoding is fixed and the depth is swept. If depth 4
or 8 wins, the follow-up ablation is option D — replace the eight bits with net rotation and turn
count and see whether the summary keeps the gain.

## 6. Open questions

The review settled the purpose, the encoding, the convention and `ended_by` (top of file). Left open:

- **The zigzag threshold.** `k = 2` for a reversal is the natural definition; whether `k = 3` or a
  run-length view says something different is for the diagnostic to show.
- **How many failures are enough.** Section 3b assumes ~250 from 20,000 episodes; if the fatal-wall
  measure separates cleanly at 60 the run can stop early, and if it does not separate at 250 the
  answer is "no".
- **The turn-penalty arm** (section 5): run it beside the depth sweep, or hold it for a later batch.
- **Endgame-only history.** If the diagnostic shows zigzags only matter late, a cheaper variant gates
  the block on board fill. Premature until the diagnostic runs.
