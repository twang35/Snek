# Potential-based reward shaping on realised chase-safety

**Status:** approved 2026-08-11, **not implemented**. Deferred on purpose — the user wants to finish
several more batches first, and batches 21-23 are mid-ladder on PER importance-sampling β.

**One line:** replace the food-distance shaping — already off since batch 17 — with a potential-based
term on "are the head, the food and the tail in one region", the one property that still separated the
record checkpoints from each other after everything else was controlled for.

Reading order for context: the marker comes from
[`../hyperparamTuning/findings.md`](../hyperparamTuning/findings.md#-what-the-record-checkpoints-do-differently-they-find-food-in-the-endgame-and-that-is-nearly-all-of-it),
the tool that measured it is
[`../hyperparamTuning/perDiagnostics/behaviour_profile.py`](../hyperparamTuning/perDiagnostics/behaviour_profile.py).

## Why this, and why now

Two established results point at it from opposite directions.

**Removing the food-distance shaping was the first non-null in six batches** (batch 16, `sef` +11.35 pp
at a matched 1.25M). The term subtracted 0.001 on any ordinary move that increased **Manhattan**
distance to the food. Manhattan distance ignores the body, so in a packed endgame the only safe move
is often *around* the snake and increases it every step. The penalty fired on correct play. It was also
one-sided — it only ever subtracted — which is not potential-based, and an asymmetric per-step penalty
changes which policy is optimal. That is a mechanism for the batch-16 result, which `findings.md`
currently records without one.

**But removing it may have cost something.** `findings.md` carries an untested worry: the modal failure
since batch 16 is *failing to go get reachable food*, so the removal may have bought consistency and
paid for it in starvations. Nobody measured it either way. This term is a replacement guidance signal
aimed at exactly that failure, without the defect that made the old one punish correct endgame play.

## The pre-check that already passed

A binary potential is worthless if the flag is almost always 1 — the trap observation index 29 fell
into, which sits at 1 in 99.95% of states and whose weights are therefore untrained. It is not. Share
of chosen moves keeping head, food and tail in one region, from the `behaviour_profile` runs behind the
champion-vs-mediocre finding (seeds 21 and 22, averaged):

| policy | 10-49 | 50-84 | 85-94 | 95-99 |
|---|---|---|---|---|
| b17b @1190k (best on record) | 0.92 | 0.75 | 0.55 | 0.41 |
| b18b @1588k | 0.93 | 0.68 | 0.47 | 0.21 |
| b23b, mid-run | 0.86 | 0.50 | 0.30 | 0.12 |
| b20a-d peaks | 0.78-0.85 | 0.33-0.47 | 0.12-0.24 | 0.05-0.09 |

**Wide dynamic range, and it separates elite from mediocre at every length.** So the potential flips
often and the term is not degenerate.

**One consequence for the design, and it is the main open risk.** The shaping signal is largest where Φ
is near 0.5, which is the **mid-game (50-84)**. Through 85-99 the flag is 0 most of the time even for
the record checkpoints, so the term goes quiet exactly where perfect games are won. That may be
correct — the mid-game is where the packing that decides the endgame happens — but see the graded
fallback under [Risks](#risks).

## The form

```
F(s, s') = c * (gamma * Phi(s') - Phi(s)),    Phi(s) = 1 if head, food and tail share one region
```

Potential-based shaping (Ng, Harada and Russell 1999) **leaves the optimal policy unchanged for any
bounded Φ**. That property is the reason to prefer it here rather than any hand-tuned bonus: the marker
it rests on is correlational, so a term that *could* move the optimum would be a real risk. This one
cannot. It can only make learning faster or slower.

Two conditions carry the guarantee, and both are easy to break silently:

- **`Φ(terminal) = 0`.** Otherwise the invariance fails.
- **γ must be the training discount**, not 1.0. With γ = 1 the sum telescopes in undiscounted terms
  while the agent discounts, leaving a residual per-step bonus for being chase-safe — a different and
  unprincipled term.

---

## The edits — five files, ~70 lines

### 1. `state_helpers.py` — the state-level potential

New function beside `group_obs`, reusing `count_groups` and `get_adjacent_groups`:

```python
def chase_safe_state(grid, head_pos, tail_pos, current_food):
    """1 when the head, the food and the tail all sit in one open region of `grid`.

    The *state* form of the per-action flag at observation indices 15-17. `group_obs` asks
    "would this move leave them in one region"; this asks "are they in one region now", which
    is what a potential function needs. Kept separate rather than folded into `group_obs`,
    which short-circuits fatal moves and needs a post-move grid per action.
    """
    if current_food == 'no food':
        return 0
    cols = grid.shape[1]
    regions, _ = count_groups(grid)
    food_bit = 1 << ((current_food.position[1] + 1) * cols + (current_food.position[0] + 1))
    escape = (get_adjacent_groups(regions, cols, head_pos)
              & get_adjacent_groups(regions, cols, tail_pos))
    return 1 if any(regions[index] & food_bit for index in escape) else 0
```

The head and tail cells are occupied, so they get `get_adjacent_groups`; the food cell is open, so it
gets containment. That mirrors `group_obs` exactly — see its comment on why the food test is
containment and not adjacency.

### 2. `snake_constants.py` — the knob

Beside `FOOD_DISTANCE_REWARD`, and read from the environment for the same documented reason:
`Snake.step` runs in the parallel env worker processes, and `from snake_constants import *` binds a
copy at import, so an assignment in the parent would never reach a worker.

```python
DEFAULT_CHASE_SAFE_SHAPING = 0.0
CHASE_SAFE_SHAPING = float(os.environ.get('SNEK_CHASE_SAFE_SHAPING',
                                          DEFAULT_CHASE_SAFE_SHAPING))
```

Default 0.0, so every existing arm and every historical number is unaffected.

### 3. `Snake.py` — the term itself

`Game.__init__` gains `discount=1.0`, stored as `self.shaping_discount`, plus
`self.chase_safe_potential = 0.0`. `reset()` sets the potential from the opening board.

The block goes at the **end of `step`**, beside the distance shaping. That position is load-bearing: by
then `_rebuild_grid()` has run, so `self.grid` is the post-move board, and every branch that can set
`self.finished` has already run.

```python
# Potential-based shaping (Ng, Harada and Russell 1999): F = c * (gamma * Phi(s') - Phi(s)).
# This form leaves the optimal policy unchanged for any bounded Phi, which is the reason to
# prefer it here -- the marker it rests on is correlational, so a term that could move the
# optimum would be a real risk. Phi(terminal) must be 0 or the invariance breaks.
if CHASE_SAFE_SHAPING:
    if self.finished:
        new_potential = 0.0
    else:
        new_potential = float(chase_safe_state(self.grid, self.head.tile_pos,
                                               self.tail.tile_pos, self.current_food))
    reward += CHASE_SAFE_SHAPING * (self.shaping_discount * new_potential
                                    - self.chase_safe_potential)
    self.chase_safe_potential = new_potential
```

Three things this gets right by construction:

- **On an eating step the food has already respawned** before `_rebuild_grid()`, so Φ(s′) is measured
  against the **new** food. This is the bug the old distance term needed a whole exclusion for, and it
  is the reason to compute Φ from state rather than reuse the per-action flag at `obs[15 + a]` — which
  is free but is computed against the food that was just eaten.
- **On a death the head is off the board**, so the grid is unusable. The `finished` branch never
  touches it.
- **A perfect game pays −c** at the winning step, because Φ(terminal) = 0. Required by the theory, and
  negligible against `PERFECT_GAME_REWARD = 100`.

`if CHASE_SAFE_SHAPING:` is a **performance gate only**. `count_groups` draws no randomness, so
skipping it cannot shift the food stream, and at c = 0 the term adds exactly 0.0 either way. So 0.0
remains a clean ablation in the same sense `FOOD_DISTANCE_REWARD=0` is.

### 4. `snake_environment.py` — thread the discount

One line: pass `discount=discount` into `Game(...)`. The env already receives the true discount.

Threading it rather than re-reading `SNEK_DISCOUNT` inside `snake_constants`, because `snek2.py` holds
that default (`tuned('DISCOUNT', 0.99)`) and a second copy of it would drift silently. The three
`hyperparamTuning/diagnostics/` scripts construct `Game` directly and run with shaping off, so the
1.0 default never reaches a shaped arm.

### 5. `snek2.py` — record it

Two lines, matching what `FOOD_DISTANCE_REWARD` already gets: the `hyperparameter override:` startup
print, and the `run_config` entry so `runs/<policy>.md` shows what the arm actually ran with.

---

## Tests

`tests/test_reward_shaping.py` already has the `build_game(shaping)` reload harness — needed because
`Snake.py` does `from state_helpers import *` and `state_helpers` does `from snake_constants import *`,
so both bind their own copy of the constant — and the `play()` fixed-action-sequence helper. Both
extend directly.

| test | what it pins |
|---|---|
| default is 0.0, and the knob reaches `Snake` | the wiring, mirroring the two existing knob tests |
| **c = 0 leaves every reward and every food position byte-identical** | the regression protecting all existing arms |
| Φ 1→0 costs exactly `−c` | sign and magnitude |
| Φ 0→1 pays exactly `+c·γ` | that it is two-sided, unlike the old term |
| Φ held at 1 costs exactly `c·(γ−1)` per step | that γ is wired and not defaulted to 1.0 |
| shaped rewards over an episode sum to the γ-weighted telescope | the invariance property, end to end |
| Φ = 0 on all three endings: death, starvation, perfect game | the condition the theory requires |
| `SnakeEnvironment(discount=x)._game.shaping_discount == x` | the thread-through, the silent-breakage risk |

One more in `tests/test_observation_spec.py`:

- **`chase_safe_state` on the post-move board agrees with `group_obs`'s `obs[15 + a]`** for every legal
  non-eating action across the existing hand-built fixtures. This pins that the potential means the
  same thing as the input the policy reads. Two documented exceptions the test must exclude: the eating
  move, where `group_obs` substitutes `head_with_tail`, and the tail-follow move, where `group_obs`
  has a special case for the vacated cell that a real post-move grid does not need.

**Then mutate and confirm a named test fails** — flip the sign, drop the γ, drop the `Φ(terminal) = 0`
branch, swap `head_groups & tail_groups` for `head_groups` alone, and drop the `+1` grid offset in
`food_bit`. Check the failure *type* as well: a `TypeError` means a stale test, not passing code.

## Phase 0 — two measurements before any code is worth writing

**Base rate and flip rate of Φ per length band.** A ~20-line scratchpad script: 20 greedy episodes with
one hall-of-fame checkpoint and one b20 peak, reporting Φ's mean and its 0↔1 transition rate. This sets
`c` on a principle instead of a guess — pick `c` so the summed absolute shaping between two meals is
**≲ 20-30% of `FOOD_REWARD = 1.0`**. Prior is c ≈ 0.1; the flip rate decides.

**Cost.** Time 2,000 steps with the term on and off. Expect ~+15% on observation build — one more
`count_groups` against the three `group_obs` already runs — and no measurable change in steps/second,
since an 11x observation speedup once moved eval wall clock by approximately nothing. If it is worse
than ~3%, say so before proposing arms.

## Validation design — pre-registered

4 laptop arms, seeds 1-4, batch 23's launch block plus `SNEK_CHASE_SAFE_SHAPING=<c>` and
`SNEK_MAX_STEPS=2500000`. `SNEK_FOOD_DISTANCE_REWARD=0` **stays** — batch 17 onward runs shaping off,
and mixing two shaping terms with opposite intent would answer nothing.

**Seed-matched against batch 23 as the control.** b23 is the current config and tracks b18 nearly
seed-for-seed, so it is a valid control at no new compute cost. Read the pairing **seed by seed**:
seeds 2 and 4 produce the best arm in 18 of 18 config waves (+5.41 pp at 550k), so a wave-mean against
a wave-mean would be confounded by seed.

| measure | source | what it decides |
|---|---|---|
| `sef` at a matched horizon | `runs/<policy>_evals.json` | whether it worked |
| p90 steps per meal at 95-99, `headroom_p10` | `behaviour_profile.py` | whether it worked *for the stated reason* |
| starvation share | `point_of_no_return.py` | the open `findings.md` item on what batch 16's removal cost |

If `sef` moves and p90 does not, the outcome is real and the story is wrong. If p90 drops toward the
records' 5-13 and `sef` does not move, the marker is a symptom rather than a cause. **Both outcomes are
informative**, which an outcome-only test cannot promise — that is the point of fixing the mechanism
check in advance.

## Risks

- **The marker is correlational.** n = 7 checkpoints, ~18 tested correlations, causal direction
  unresolved. A well-packed snake is chase-safe *because* it is well packed.
- **n = 4 cannot resolve an effect below ~10 pp.** The same config has produced 62.5 and 18.0 here. A
  null means "not large", not "no effect".
- **The signal lands in the mid-game, not the endgame.** **Strengthened 2026-08-14**, and the risk is
  smaller than it reads: Φ sitting near 0 through 95-99 is the board genuinely having no safe meal —
  eating the reachable food leaves the head no legal move in **54%** of losses
  ([`findings.md`](../hyperparamTuning/findings.md#-retracted-2026-08-14-the-positions-are-trapped--geom-counts-routes-that-eat-and-die)).
  So a graded *distance*-to-food potential is the wrong fallback; it would pull the snake onto meals
  that kill it. If a graded version is wanted, grade the **region the head and tail share**, as below.
  If the arms come back null and Phase 0
  confirms Φ ≈ 0 through 85-99, the follow-up is a **graded** potential: the share of open cells in the
  region the head and tail share, which `count_groups` already returns as a bitmask, so it costs one
  `bin(region).count('1')`. That is a different hypothesis — packing, not reachability — and it should
  not be built alongside the binary version.
- **`avg_reward` shifts slightly**, and the bootstrap epsilon phase thresholds on it. The term
  telescopes, so the shift is bounded by about `c` per episode against ~95 food points, far smaller
  than the distance term's documented shift. `avg_score` is a food count and is unaffected.

## What this deliberately does not touch

**No observation change** — so no `OBS_ERA` bump, no `arch.json` change, and every hall-of-fame
checkpoint still loads. **No change to the starve budget**, which is the alternative that was
considered and rejected for this slot: it would alter the MDP and make every historical number
incomparable, including the 97.6% record and all 3,712 measured rows. Potential-based shaping leaves
the task and the optimal policy unchanged, so it costs nothing from the measurement history.

Nothing in `savedPolicies/`, `runs/`, `evals/` or `hallOfFame/`.

## Order of work when this is picked up

Phase 0 measurements → the five edits → the tests → the mutation pass → report the diff and the
measured cost, uncommitted. The arm launch is a **separate** approval, because it starts real training.
