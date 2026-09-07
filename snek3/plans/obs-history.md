# Short-term memory: showing the policy its last four moves

**Written 2026-09-07, for discussion.** Nothing here is built. The question asked was: give snek a
view of its last four moves, cheaper than four one-hot triples (12 inputs), and is there a better
approach altogether. The answer this plan argues for is in section 2; the rest is what it costs and
how to test it.

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

## 3. What the feature is for, and what to measure before trusting it

A history of moves helps a memoryless policy in exactly one way: **it breaks observation aliasing.**
When the 30 values are identical around a cycle the policy makes the same choice at each lap and
circles until the starve budget runs out. Four moves of history make the laps distinguishable only
when the cycle is short, and any effect shows up as **fewer starve deaths**, not fewer collisions.

**The evals do not currently record how a game ended.** A stage-B row carries scores and the perfect
rate; `VecEnv.step`'s `info` carries `died / starved / perfect` per game, and `vectorized/engine.py`
drops it. So the first step, before any observation change, is a read-only diagnostic:

1. Play the current best checkpoint (b25's ladder top) for 5,000 episodes and split the endings:
   perfect, wall/body collision, starve. Also count **repeated `(head, head_dir, obs)` states within
   an episode** as a direct measure of aliasing loops.
2. **If starve deaths are near zero, history cannot move the number** and the plan stops here with
   that finding written into `docs/findings.md`. If they are a meaningful share of the losses, go on.

This also decides whether to carry `ended_by` into the stage-B row permanently. It costs nothing and
would have answered this question already.

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

Judged on the b25 numbers (true rate at depth, drawdowns), plus the section-3 death split at the
close-out so an improvement can be attributed to fewer starves rather than assumed. Depth is the real
unknown; the encoding is not, which is why the encoding is fixed and the depth is swept. If depth 4
or 8 wins, the follow-up ablation is option D — replace the eight bits with net rotation and turn
count and see whether the summary keeps the gain.

## 6. Open questions for the discussion

- **Is aliasing the failure mode you have seen?** Section 3 measures it, but if the motivation is
  something else — smoother paths, fewer zigzags in the endgame — the feature is the same but the
  metric is not, and the plan should say which.
- **Convention.** Every existing input reads "1 is good or safe". These eight are descriptive, not
  evaluative. Fine, but it should be written down in `environment.md` so the next reader does not
  hunt for the polarity.
- **Should `ended_by` go into the stage-B row regardless?** Recommended yes; it is one field and it
  would have shortcut this whole discussion.
