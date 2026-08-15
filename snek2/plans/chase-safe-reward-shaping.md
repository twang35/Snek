# Potential-based reward shaping on realised chase-safety

**Status:** approved 2026-08-11. **Phase 0 done and the code shipped 2026-08-14** — `c = 0.10`, Variant B
(length-gated at 85), knobs `SNEK_CHASE_SAFE_SHAPING` and `SNEK_CHASE_SAFE_GATE`, default off. See
[implementation notes](#-implementation-notes-2026-08-14) for the measured cost and the mutation pass. The hold has expired: batches 20-26 are all closed and both hosts
are idle, so this is the top backlog item and the design below is ready to execute. **Revised 2026-08-14** in four places, each marked **‡**: the control moved from
batch 23 to batch 24, the potential has to survive a fork, Phase 0 shrank because
`behaviour_profile.py` already answered half of it, and `c` now has an arithmetic rule rather than a
prior.

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

**Wide dynamic range, and it separates elite from mediocre at every length**, so the potential is not
degenerate. ~~So the potential flips often.~~ **‡ That inference was wrong and Phase 0 falsified it:** a
base rate near 0.5 says nothing about how often the value *changes*, and the records turn out to flip only
**0.21-0.63 times per meal**. Base rate and flip rate are independent, which is why `c` needed measuring
at all — see [Phase 0 results](#-phase-0-results-2026-08-14).

**One consequence for the design, and it is the main open risk.** The shaping signal is largest where Φ
is near 0.5, which is the **mid-game (50-84)**. Through 85-99 the flag is 0 most of the time even for
the record checkpoints, so the term goes quiet exactly where perfect games are won. That may be
correct — the mid-game is where the packing that decides the endgame happens — but see the graded
fallback under [Risks](#risks).

## ‡ What it could move, and by which route — added 2026-08-14

Three routes, in descending order of how likely they are:

| route | what moves | prior |
|---|---|---|
| **Denser credit assignment** → the strong region gets wider | `best_perfect30`, `sef`, pooled, and therefore the *number* of hall-of-fame tickets | **most likely.** Every effect this project has found — batches 16, 21-24 — moved this axis and left the peak alone |
| **Better mid-game packing** → the endgame arrives winnable more often | the per-episode perfect rate at convergence, i.e. the ceiling | **the one that would be new.** Supported by the 92%/77%/5% packing separation, but causally unproven, and Φ is quiet through 85-99 |
| Null, or slower learning from a badly scaled `c` | nothing, or variance | real. The term is a per-step signal on a correlational marker |

**Why a hall-of-fame checkpoint follows from the first route rather than the third.** A new record needs
one checkpoint at ≥98% over 500 fresh episodes. Batch 24 turned **199** ≥97%/100 checkpoints into 9 that
held ≥97%/500 and 2 at 98.0% — and the arm that produced none (`b24a`) is the arm with the lowest
`best_perfect30`, not the one with the lowest peak. So the odds of a record are set by how wide and how
dense the strong region is, which is exactly what routes 1 and 2 widen and what `best_perfect30`
measures.

**One theory point that is easy to state backwards.** "Potential-based shaping leaves the optimal policy
unchanged" is a statement about the exact solution of the MDP. Under function approximation and a 3M-step
budget the policy actually *found* does change — that is the entire reason to add the term. The theorem
buys safety (it cannot make the thing being optimised worse), not inertness.

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

## The edits — five files, ~80 lines

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

#### ‡ The potential is per-episode state, so `restore_snapshot` has to rebuild it — added 2026-08-14

**`Game.snapshot()` does not carry `chase_safe_potential`, and the branch envs are a reused pool.**
`ForkingCollector._fork` pops an env off `self._pool`, calls `restore_from_snapshot`, and `_retire`
puts it back — so without a fix a forked branch computes its first `F` against the potential left
behind by whatever branch episode last used that env. That injects an arbitrary constant of up to
`±c` into the branch's return and **breaks the telescope on branch streams**, which is where the
invariance guarantee lives.

**This is live, not hypothetical.** The base config below is batch 24's, and it runs
`FORK_BRANCHES=4` with `fork_prob=0.5` at `fork_min_length=85` — so the corruption would land on
roughly every other endgame fork, i.e. exactly the transitions this term exists to improve.

The fix is one line at the end of `restore_snapshot`, beside its `_rebuild_grid()`:

```python
self.chase_safe_potential = float(chase_safe_state(self.grid, self.head.tile_pos,
                                                   self.tail.tile_pos, self.current_food))
```

**Recompute rather than add a snapshot field.** Φ is a pure function of grid, head, tail and food,
all of which the snapshot already restores exactly, so recomputing is byte-identical to carrying the
value — and it leaves `GameSnapshot`, `validate_snapshot` and `test_game_snapshot.py` untouched. The
cost is one `count_groups` per fork, against one per step.

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
| **‡ `restore_snapshot` leaves the potential equal to Φ of the restored board** — assert it against a game deliberately left holding the wrong value first | the fork gap above, which no other test can see |
| **‡ a restored branch's first shaped reward equals the parent's next shaped reward** | the same gap, end to end, on the path the collector actually takes |

One more in `tests/test_observation_spec.py`:

- **`chase_safe_state` on the post-move board agrees with `group_obs`'s `obs[15 + a]`** for every legal
  non-eating action across the existing hand-built fixtures. This pins that the potential means the
  same thing as the input the policy reads. Two documented exceptions the test must exclude: the eating
  move, where `group_obs` substitutes `head_with_tail`, and the tail-follow move, where `group_obs`
  has a special case for the vacated cell that a real post-move grid does not need.

**Then mutate and confirm a named test fails** — flip the sign, drop the γ, drop the `Φ(terminal) = 0`
branch, swap `head_groups & tail_groups` for `head_groups` alone, and drop the `+1` grid offset in
`food_bit`. Check the failure *type* as well: a `TypeError` means a stale test, not passing code.

## ‡ Phase 0 — one measurement, because half of it is already done

**The base rate is measured.** The table above is `behaviour_profile.py` output, and it already shows Φ
is non-degenerate with a wide per-band range. Nothing needs re-running for it.

**Only the flip rate is missing, and it is what sets `c`.** A ~20-line scratchpad script: a copy of
`behaviour_profile.py`'s harness with the per-action read at `obs[15 + a]` replaced by a
`chase_safe_state` call on the live grid.

**‡ Weighted toward the near-endgame, revised 2026-08-14.** The original 20 episodes on two checkpoints
gives only ~100 meal intervals in the top band, which is the band the batch is *for*. Instead:

| | |
|---|---|
| checkpoints | **`b24d` @1342k**, **`b18b` @1588k**, **`b20d` @3000k** — the same trio the packing finding uses, so the numbers sit beside its table |
| episodes | **60 each on identical food streams** (seeds 201/202), matching `endgame_packing.py`'s protocol |
| bands | 50-84, **85-89, 90-94, 95-97, 98-99** — split at the top, where the old 85-94 / 95-99 pair is too coarse |
| per band | Φ mean · **flips per meal** · **flips per 100 steps** · steps per meal · **share of the episode's steps** |

**Three of those columns are load-bearing.** `flips per meal` is the calibration denominator, because a
real 1.0 arrives once per meal. `flips per 100 steps` has to be reported beside it, or a weak policy's
86-226 steps per meal at 95-99 inflates its per-meal count purely through meal *duration* rather than
flip frequency. And **share of steps** is the number that decides the gate question below: it says how
much of a global `c`'s dose lands where no help is wanted.

### ‡ Why the flip rate is the quantity that sets `c` — and not the base rate

**The whole point of the potential form is that the total return is fixed, so `c` cannot be read off the
return.** The discounted sum telescopes *exactly*: `Σ γᵗ F_t = −c·Φ(s₀)`, a constant that depends on the
opening board and not on the policy. That is the invariance. It also means the term's size is invisible in
the return — a c of 0.01 and a c of 10 change the objective by the same constant.

**What `c` actually scales is the per-transition reward, which is what a 1-step DQN learns from.** The
real reward is 0 on all but ~95 of a ~1,500-step episode, so on a step where Φ flips, `±c` is the *entire*
reward signal for that transition. So the calibration has to compare `c` against `FOOD_REWARD = 1.0` per
**meal interval** — the natural denominator, since that is how often a real 1.0 arrives.

**Three magnitudes, which is why only the flip rate matters:**

| event | shaped reward | over an episode |
|---|---|---|
| Φ flips 0→1 | `+c·γ` | the dominant term |
| Φ flips 1→0 | `−c` | the dominant term |
| Φ **held** at 1 | `c·(γ−1)` = **−0.0025c**/step | −0.24c over 1,500 steps at γ=0.9975 |
| Φ held at 0 | exactly 0 | nothing |

So a held potential is nearly free and the base rate — the thing `behaviour_profile.py` already measured
— tells us almost nothing about scale. **Flips are the entire budget**, hence:

```
c = 0.25 / (flips per meal interval, in the band where that rate is highest),  clamped to [0.02, 0.10]
```

At 2-3 flips per meal that is c ≈ 0.10 (the old prior); at 5-6, c ≈ 0.05. Use the **worst band** rather
than the episode mean, so the term cannot dominate the food signal anywhere; then sanity-check the other
direction, that at the *low* band the per-meal shaping is still ≥ ~2% of `FOOD_REWARD`, or the term is
inert where it is supposed to act.

**Why two checkpoints and not one.** `b24d` is the regime the batch is trying to reach and `b20d` is the
one it starts from, so their flip rates bracket what a shaped arm will actually experience — early
training resembles `b20d`, late training `b24d`. Calibrating on the higher rate bounds the dose; reporting
both says how much the dose fades as the arm improves.

**The measurement is circular, and benignly so.** Φ's flip rate is a property of the *policy*, and these
two policies trained without the term — so a shaped arm's rate will differ. The direction is favourable:
locally the term pays `+cγ` for entering chase-safety and `−c` for leaving it, so a policy that responds
to it flips *less*, and the term self-attenuates as the arm gets better. Phase 0 therefore measures an
upper bound on the dose, which is the safe side to be wrong on.

**The interaction that may matter more than the reward scale: PER.** Priorities are `|TD error|^α` with
α=0.6 and IS off in this base, so adding `±c` to the reward on flip steps raises those transitions'
priority directly. If `c` is comparable to a typical endgame `|TD error|`, the term's largest effect could
be *which transitions get replayed* rather than what the targets say. That is plausibly the mechanism
rather than a side effect — and it is checkable after the fact with
[`per_priorities.py`](../hyperparamTuning/perDiagnostics/per_priorities.py), which is worth doing whichever
way the batch lands.

**Report the flip rates, the chosen `c` and both sanity checks before writing any arm launch**, so the
value rests on a principle rather than a prior.

### ‡ Phase 0 results 2026-08-14

**`c = 0.10`, and Variant B is confirmed.**

Ran with [`perDiagnostics/chase_safe_potential.py`](../hyperparamTuning/perDiagnostics/chase_safe_potential.py),
60 episodes per checkpoint (30 each on seeds 201/202), payloads in
[`perDiagnostics/results/`](../hyperparamTuning/perDiagnostics/results/README.md). Φ was validated first
against the observation the policy already reads — `obs[15 + a]` on the post-move board, **4,460
agreements and 0 disagreements** under the `b24d` greedy policy across every band including coiled
endgame boards, with three mutations each producing disagreements.

**Genuine flips per meal** — genuine excludes the one mandatory terminal `Φ→0` per episode and, for the
gated form, the one gate crossing, since neither is a signal a policy can act on:

| band | `b24d` (98.0%) | `b18b` (97.6%) | `b20d` (~47%) | Φ base rate, records vs `b20d` |
|---|---|---|---|---|
| 10-49 | 0.38 | 0.25 | 0.58 | 0.88 / 0.92 vs 0.77 |
| 50-84 | 0.52 | 0.63 | **1.44** | 0.72 / 0.67 vs 0.34 |
| 85-89 | 0.27 | 0.41 | **2.52** | 0.52 / 0.46 vs **0.15** |
| 90-94 | 0.45 | 0.21 | **2.72** | 0.37 / 0.49 vs **0.09** |
| 95-97 | 0.33 | 0.38 | **3.61** | 0.33 / 0.24 vs **0.05** |
| 98-99 | **0.02** | **0.00** | 0.04 | 0.20 / — vs 0.01 |

| | `b24d` | `b18b` | `b20d` |
|---|---|---|---|
| share of steps at length ≥85 | 0.109 | 0.108 | **0.412** |
| share of flips at length ≥85 | 0.144 | 0.141 | 0.324 |
| **c, variant A** (global) | 0.591 | 0.617 | **0.203** |
| **c, variant B** (gate 85) | 0.805 | 0.870 | **0.097** |

**Four results, and three of them were not what this plan predicted.**

1. **The deepest endgame is confirmed unshapeable.** At 98-99 the genuine rate is **0.00-0.04 per meal**
   — 57 of `b24d`'s 59 flips there and **all 56** of `b18b`'s are the terminal `Φ→0`. A band can show 40
   flips per 100 steps and carry no signal at all, which is exactly why the decomposition was added. The
   structural-zero prediction holds; a gate above ~95 would shape nothing.
2. **Φ is far more static than the prior assumed.** The plan expected 2-3 flips per meal; the records
   read **0.21-0.63**, five to ten times lower. Read alone that is the "inert" branch of this file's own
   decision table.
3. **‡ But the dose is self-attenuating, and that is the result.** `b20d` — the regime training spends
   most of its steps in — flips **2.5-3.6 times per endgame meal** and spends **41.2%** of its steps at
   length ≥85, against the records' 10.8%. So the gated term fires ~**35 times per episode** for a
   struggling policy and ~**4.6** for a record one. It is loud exactly where the arm is bad at the
   endgame and goes quiet as it gets good — the benign feedback this plan predicted, now measured rather
   than assumed. The "inert" reading applies to the *converged* policy, where potential-based shaping is
   provably neutral anyway.
4. **‡ Correction: gating does *not* buy a larger `c`.** The Variant B section above argued it would,
   assuming mid-game flip traffic was consuming the budget. It is not — for the policy that binds the
   calibration, the flips are *concentrated* at ≥85 (2.5-3.6 per meal against 1.44 in the mid-game), so
   gating raises the rate per shaped meal and **halves** `c`, 0.203 → 0.097. The gate's benefit is the
   one the user asked for and nothing more: it leaves the early game untouched.

**So `c = 0.10`**, from `b20d`'s gated rate of 2.567 genuine flips per meal (`0.25 / 2.567 = 0.097`).
Calibrating on the highest rate bounds the dose, per the rule above, and it lands on both the clamp
ceiling and this plan's original prior of 0.1 by two independent routes.

**One honesty note for reading the outcome.** At c = 0.10 the gated term contributes ~3.4 of total
|shaping| per episode against ~95 of food, so **a null would be ambiguous between "wrong idea" and "dose
too small"**. The follow-up if the batch is null *and* shows no harm is a dose ladder at 0.05 / 0.20, not
a different potential — and the 25% budget would still permit up to ~0.20 on `b20d`'s rate.

**A bonus result worth its own line.** Φ's base rate at length 85-94 is **0.52 / 0.46 for the records
against 0.145 / 0.091 for `b20d`** — a 3-5× separation at exactly the band where the packing gap opens.
That is independent support that the quantity is worth shaping, measured on state rather than on chosen
moves.

### ‡ Variant B — a length-gated Φ, if the signal is wanted only in the near-endgame

Added 2026-08-14, from the observation that early-game competence needs no help. **`c` is a single global
scalar, so it cannot be aimed at a length band** — calibrating it on the endgame only makes the term
louder *everywhere*, and if the endgame flip rate is the lower one that means over-dosing the mid-game to
make the endgame audible. The way to aim the term is to change **Φ**, not `c`:

```
Phi(s) = 1 if snake_len >= GATE and head, food and tail share one region, else 0
```

**This is still potential-based and the guarantee is untouched** — the theorem holds for *any* bounded
function of state, and a length gate is one. Two properties it gains:

- **Total discounted shaping becomes exactly 0.** The telescope is `−c·Φ(s₀)`, and Φ(s₀) = 0 because the
  snake starts at length 5. Nothing to offset at all.
- ~~**A larger `c` fits the same safety budget**, because the mid-game's flip traffic no longer competes
  for it.~~ **‡ Falsified by Phase 0, 2026-08-14.** The flips are *concentrated* at ≥85 for the policy
  that binds the calibration, not in the mid-game, so gating **halves** `c` (0.203 → 0.097). The gate's
  only benefit is leaving the early game untouched — which is the point, but it costs signal strength per
  flip rather than buying it. See [Phase 0 results](#-phase-0-results-2026-08-14).

**One hard constraint on how deep the gate can usefully sit.** Φ is **structurally 0 at length 99** — one
free cell cannot hold head, food and tail — and near-structurally 0 at 98, where indices 18-20 take over
the job. So **the last two or three meals cannot be shaped by this potential at any `c`**. The shapeable
near-endgame is roughly **85-94**, which is also exactly where the packing gap opens (92% / 77% / 5% at
90-94, ten meals before the end). A gate at **85** is therefore the candidate; a gate at 95 would buy
almost nothing.

**`GATE = 85` also matches `FORK_MIN_LENGTH = 85`**, which this base config already runs at
`FORK_BRANCHES=4` — so the shaped transitions are the same ones the collector already oversamples. That
compounds the dose, which is the intent, and is a second reason to keep `c` conservative.

**The risk that argues against gating, stated fairly.** The packing finding shows policies *arrive* at
90-94 fragmented rather than arriving less often, so the behaviour that separates them happens **upstream
of the band where it becomes visible**, and nobody has pinned how far upstream. If the causal horizon is
length 70, a gate at 85 shapes only the symptom. Variant A (no gate) hedges that by shaping everywhere and
accepting a mid-game dose; Variant B bets that 85+ is early enough.

**Pick one, do not split the batch.** n=2 per variant resolves nothing. Phase 0's step-share and per-band
flip columns are what decide it: if a large share of steps sits below 85 *and* the mid-game carries most of
the flips, gate it; if flips are concentrated at 85+ anyway, the gate is redundant and A is simpler.

**Cost.** Time 2,000 steps with the term on and off. Expect ~+15% on observation build — one more
`count_groups` against the three `group_obs` already runs — and no measurable change in steps/second,
since an 11x observation speedup once moved eval wall clock by approximately nothing. If it is worse
than ~3%, say so before proposing arms.

## ‡ Validation design — pre-registered, and the control moved to batch 24

**The base config is batch 24's, not batch 23's.** When this was written b23 was the best config on
record; b24 has since beaten it by **+12.2 pooled on all four seeds** with a new record (`b24d` @1342k,
98.0%/500), and b25/b26 established that the lift tracks capacity rather than width. Shaping the weaker
base would answer a question about a config nobody will run again. So: **`fc 320`, IS off, target 1000,
disc 0.9975, 3M steps** — b24 exactly — plus the one new knob.

4 arms, seeds 1-4, **`b27a-d`**:

```
SNEK_FC_LAYERS=320 SNEK_IS_WEIGHTS=0 SNEK_TARGET_UPDATE_PERIOD=1000 \
  SNEK_DISCOUNT=0.9975 SNEK_FOOD_DISTANCE_REWARD=0 \
  SNEK_CHASE_SAFE_SHAPING=<c> SNEK_MAX_STEPS=3000000 SNEK_SEED=<n> \
  PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u snek2.py b27<x>-chasesafeseed<n> \
  > /tmp/b27<x>.log 2>&1 &
```

`SNEK_FOOD_DISTANCE_REWARD=0` **stays** — batch 17 onward runs the distance shaping off, and mixing two
shaping terms with opposite intent would answer nothing.

**Seed-matched against `b24a-d` as the control**, at no new compute cost: same code era, same
observation, same 3M horizon, four finished seeds with close-outs and an HOF-500 already measured. Read
the pairing **seed by seed** — seeds 2 and 4 produce the best arm in 18 of 18 config waves (+5.41 pp at
550k), so a wave-mean against a wave-mean is confounded by seed.

**Run it on the desktop.** `the-claw-den` is idle, four trainers fit, and the queued path chains
`training → close-out (top20) → HOF-500` automatically — which for a batch whose whole question is
"does the ceiling move" is the difference between a result and a result three days later. The laptop's
four slots are free too and would work, but every close-out would then be launched by hand. Queueing is
a **separate approval**, per `CLAUDE.md`: pushing to `ops` starts real work on another machine.

**One eval-era note.** `fc 320` is a checkpoint era: `arch.json` records it and every eval rebuilds the
recorded net from the sidecar, so nothing has to be passed by hand — but any checkpoint copied out of
`savedPolicies/` must take `arch.json` with it.

| measure | source | what it decides |
|---|---|---|
| **‡ `best_perfect30` at a matched horizon, seed by seed** | `runs/<policy>_evals.json` | **the primary — target and test both.** On the perfect rate itself, control at 95.3-96.7 with 3.8 pp of headroom; it ordered b24's HOF outcomes 4 of 4; and against a b24-class control it resolves **~3.6 pp** paired at n=4, ~6× sharper than `sef` |
| `sef` at a matched horizon | `runs/<policy>_evals.json` | reported alongside for continuity — but at this level its paired sd is 15.35, so it resolves only ~21 pp and cannot decide this batch |
| **‡ count of full-length ≥98%/500 rows** | the auto-HOF chain | **the decisive artifact** — a new hall-of-fame checkpoint is the outcome this is for. Control: 1 · 1 · 0 · 0 across b24b/d/c/a |
| pooled eq-effort at gate 95 | close-out `_checkpoint_evals.json` | consolidation, the axis every effect found here has moved |
| **‡ NOT `peak_trailing`** | — | capped at 95 and all four controls sit on 95.00, so it cannot register a gain. Report the *count* of trailing-95.00 windows instead (b24: 7 · 22 · 10 · 17) |
| realised chase-safety and p90 steps per meal at 95-99, `headroom_p10` | `behaviour_profile.py` | whether it worked *for the stated reason* — the term should raise the behaviour it shapes |
| **‡ one-piece share at length 90-94** | `endgame_packing.py` | the largest per-policy separation on record (92% / 77% / 5%). If chase-safety rises and packing does not, the term bought the marker without the property |
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
  not be built alongside the binary version. **‡ The length-gated Variant B above is the cheaper first
  response to this risk**, since it keeps the same Φ and only changes where it is non-zero.
- **‡ The graded packing potential now has its own case, and it still must not be built alongside this.**
  The 2026-08-14 packing finding — one-piece share at length 90-94 of **92% / 77% / 5%** for `b24d` /
  `b18b` / `b20d` — is a larger separation than realised chase-safety's, and it is upstream: fragmenting
  the board is *why* the food lands where eating it kills. So the fallback below is no longer only a
  fallback; it is a competing first choice. It stays a **separate batch** because running both terms at
  once would leave a moved `sef` unattributable, which is the confound that already cost this project
  batch 10. The order given here — binary chase-safety first — follows the user's request and the
  approved plan; if the graded version is preferred instead, swap them rather than merging them.
- **‡ The result is now coupled to `fc 320`.** Shaping is measured on top of batch 24's capacity, so a
  null means "no large effect on this base", not "no effect". That is the right trade — the base is the
  strongest on record and its controls are already paid for — but the interaction is untested and a
  positive result should be re-run on `50,100,50` before it is called a property of the term.
- **`avg_reward` shifts slightly**, and the bootstrap epsilon phase thresholds on it. **‡ Corrected
  2026-08-14 — the earlier "bounded by about `c` per episode" was wrong by ~5×**, because it applied the
  telescope to the wrong sum. The *discounted* sum telescopes exactly to `−c·Φ(s₀)`; `avg_reward` is
  **undiscounted**, where the residual is `−c·[(1−γ)(T−1) + 1]` — at γ=0.9975 that is −1.1c on a 50-step
  episode and **−4.75c** on a 1,500-step one, so −0.47 at c=0.1 against an `avg_reward` of ~160. Still
  negligible for the thresholds (which are 2 to 20, and reached while episodes are short and the residual
  is ~−0.11), and far smaller than the distance term's documented shift. `avg_score` is a food count and
  is unaffected.

## What this deliberately does not touch

**No observation change** — so no `OBS_ERA` bump, no `arch.json` change, and every hall-of-fame
checkpoint still loads. **No change to the starve budget**, which is the alternative that was
considered and rejected for this slot: it would alter the MDP and make every historical number
incomparable, including the 97.6% record and all 3,712 measured rows. Potential-based shaping leaves
the task and the optimal policy unchanged, so it costs nothing from the measurement history.

Nothing in `savedPolicies/`, `runs/`, `evals/` or `hallOfFame/`.

## ‡ Implementation notes 2026-08-14

Shipped as described, with two departures worth recording and one thing the mutation pass caught.

**The gate became a knob** (`SNEK_CHASE_SAFE_GATE`, default 85) rather than a constant, because the
batches queued alongside this vary it. The length test runs *before* the flood fill, which is why the
gated form is nearly free.

**`Game.__init__` gained `discount=1.0`** and `SnakeEnvironment` passes its own through. The frozen
diagnostics build a `Game` directly and get 1.0, which never reaches a shaped arm because they all run
with shaping off.

**Measured cost**, 6,000 steps including one observation build each — the plan asked for this and set
~3% end to end as the threshold to report back on:

| | seconds | vs off |
|---|---|---|
| shaping off | 0.304 | — |
| c=0.1, **gate 85** | 0.308 | **+1.3%** |
| c=0.1, ungated | 0.363 | +19.1% |

The +19.1% is the predicted fourth `count_groups` and it is the honest worst case. The gated figure is
measured under random play, which never reaches length 85, so it is really the cost of the length
comparison; the true gated cost is about `0.11 × 19% ≈ 2%` of the observation path for a policy that
spends 10.9% of its steps above the gate — and less than that end to end, since the observation build is
a fraction of a training step. Inside the threshold.

**24 tests, and the mutation pass found a vacuous one.** Six mutations — flipped sign, dropped γ, dropped
`Φ(terminal)=0`, dropped the `restore_snapshot` recompute, dropped the gate, dropped the `+1` padding
offset — each failed a named test with an `AssertionError`. But **swapping `head_groups & tail_groups`
for `head_groups` alone failed nothing**, because in both sealed-pocket fixtures the head does not
neighbour the pocket either, so the two rules agree. Fixed by adding `HEAD_ONLY_TRAP`: a full row of body
seals row 0, the head sits on the junction touching both regions, the tail touches only the lower one, so
food in row 0 must read 0 and the head-only rule reads 1. That mutation now fails exactly one test. This
is the same trap `CLAUDE.md` records for `group_obs`, and it recurred on the first attempt.

Full suite: **566 tests, 0 failed.** End-to-end smoke run under the exact batch config confirmed the
override line prints both knobs and `runs/<policy>.md` records them.

## Order of work when this is picked up

1. **Phase 0** — the flip-rate script, and `c` from the rule above. Cheap, read-only, no code change.
2. **The five edits**, including the `restore_snapshot` line the fork gap needs.
3. **The tests**, then the mutation pass — flip the sign, drop the γ, drop the Φ(terminal) branch, drop
   the `restore_snapshot` line, and confirm a *named* test fails each time, checking the failure type.
4. **The cost measurement** — 2,000 steps with the term on and off. Φ is a genuine fourth
   `count_groups` per step: `food_space_obs` avoided being one on purpose, and `group_obs`' three calls
   are on per-action post-move grids, so none of them is reusable for the state-level Φ. Expect ~+15% on
   observation build and no measurable change in steps/second. **If it is worse than ~3% end to end, say
   so before proposing arms.**
5. **Report the diff and the measured cost, uncommitted** — the code rule in `CLAUDE.md`.

The arm launch is a **separate** approval, and queueing it on the desktop is a second one, because
pushing to `ops` starts real work on another machine.
