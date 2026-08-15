# Potential-based reward shaping on realised chase-safety

**Status:** approved 2026-08-11, **not implemented**. The hold has expired — batches 20-26 are all
closed and both hosts are idle as of 2026-08-14, so this is the top backlog item and the design below
is ready to execute. **Revised 2026-08-14** in four places, each marked **‡**: the control moved from
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

**Wide dynamic range, and it separates elite from mediocre at every length.** So the potential flips
often and the term is not degenerate.

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

**Only the flip rate is missing, and it is what sets `c`.** A ~20-line scratchpad script: 20 greedy
episodes on **`b24d` @1342k** (the 98.0%/500 record, in `hallOfFame/`) and **`b20d` @3000k** (the
mediocre reference the packing and entrapment findings both use), reporting Φ's 0↔1 transition count
**per meal interval** by length band. Both checkpoints are already local and both diagnostics that use
them run at `discount=0.9975`, so this is a copy of `behaviour_profile.py`'s harness with the
per-action read replaced by `chase_safe_state`.

**The rule that turns that number into `c`.** Target summed |shaping| between two meals at **≲ 25% of
`FOOD_REWARD = 1.0`**, and a flip costs `c` while a held Φ costs only `c·(1−γ)` = 0.0025c per step at
γ = 0.9975 — negligible, ~1/4 of the old distance penalty even summed over a 1,500-step episode. So
flips dominate and

```
c = 0.25 / mean flips per meal interval,   clamped to [0.02, 0.10]
```

At 2-3 flips per meal that is c ≈ 0.10 (the old prior); at 5-6 it is c ≈ 0.05. **Report the number and
the resulting `c` before writing any arm launch**, so the value is on a principle rather than a guess.

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
  not be built alongside the binary version.
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
