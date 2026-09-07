# Short-term memory: showing the policy its last four moves

**Written 2026-09-07, revised the same day after review. Not started.** The question asked was: give
snek a view of its last four moves, cheaper than four one-hot triples (12 inputs), and is there a
better approach altogether. The answer is in section 2; the rest is what it costs and how to test it.

**Decided in review (2026-09-07):**

| decision | |
|---|---|
| encoding | option B — two bits per move, read off the body, `SNEK_OBS_HISTORY` gates the block |
| purpose | **fewer zigzags, and through them a higher perfect rate.** Not aliasing loops; section 3 is written for that |
| convention | the new inputs are *descriptive*, not "1 is good"; `docs/environment.md` says so |
| `ended_by` | goes into the stage-B row permanently, alongside this work |
| implementation | **not yet.** The plan is agreed; the build waits for a go |

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

Starves get the same trace but a different question, since nothing traps the head: is the snake
circling (a repeated `(head, head_dir)` cycle — the aliasing case of the first draft) or reaching but
never entering the food's region? Those are two different features, and neither is move history
unless the cycle is four moves long.

**What gets produced:** one table — deaths by cause, with the pre-trap reversal rate, the fatal-wall
reversal rate, and the matched-fill perfect-game rate beside them — and a **contact sheet** of the
failures: the final 200 steps of each as one still, head path coloured by time, wall segments marked,
so the numbers can be checked by eye on the dozens of games there are. `record_gif.py` and
`Game.snapshot` already do the replay and the drawing; the sheet is a layout on top of them.

`ended_by` and the three path measures then go into the stage-B row permanently, so the close-out of
every future batch carries them.

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
