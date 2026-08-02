# Observation space: what the champion's failures say, and what to change

**Frozen snapshot, 2026-08-02.** This is a record of one investigation on one day, not a
maintained document. Nothing here is expected to be kept in sync with the code, and later
sessions should treat every number as "measured then" rather than "true now". Live tuning state
belongs in [`hyperparamTuning/`](hyperparamTuning/); conclusions worth carrying
forward should be copied into `findings.md` rather than edited here.

Everything below was measured on the **post-audit environment** (the 2026-08-01 boundary, after
six env bugs were fixed) using the current champion `b8f-disc9975seed2` at checkpoint
`3149000`, greedy, over **360 episodes / 1,076,492 steps**.

**Status, added the same day.** Several of the recommendations were implemented immediately after
this was written; the document itself is left exactly as measured.

| recommendation | outcome |
|---|---|
| advance `tail_pos` | done — moved that same checkpoint 80.0% → **90.3%** with no retraining |
| fix the starve observation and add length | done — split into a `[0, 1]` budget and an explicit `snake_len / PERFECT_SCORE` |
| set `discount = 0.0` on terminal steps | done — the final transition of every episode now trains toward the reward instead of `reward + 0.9975 * V(terminal)` |
| drop the `game_over` input | done, and only safe *after* the discount fix, exactly as this document argued |
| add "is the food reachable" | done, and stricter than proposed here: head, food **and** tail in one region, so it names a reachable meal rather than merely a visible one |
| zero the region values on a fatal move | done, raised by the user after reading the food-reachable fix: a wall move happened to read 0 already, but a self-collision did not — `head_with_tail` was 1 on **5,289 of 14,642** body-collision moves in real play, describing a snake that does not survive the move that produced it |
| everything else | not implemented |

The vector is **23 values** now, so the checkpoints this document was measured on fail to load. Use
commit `e4514a8` for those — and note that for part of the same day the count was coincidentally
back at 20 while two indices meant different things, in which window those checkpoints loaded
silently and played like beginners. Details in the 2026-08-02 subsections of
[`hyperparamTuning/findings.md`](hyperparamTuning/findings.md); the "input 18 reaches 8.97" and
"input 19 is always 0" observations below are now historical.

---

## Summary of recommendations

| change | confidence | measured effect | cost |
|---|---|---|---|
| Advance `tail_pos` to the **post-move** tail, which depends on whether the move eats | **measured** | flags 94.1% of the fatal moves, vs 22.1% today; never points the wrong way | one extra argument, no runtime cost |
| Set `discount = 0.0` on terminal steps **before** touching input 19 | mechanism certain, effect unmeasured | death currently pays `−5 + 0.9975·V(terminal)` instead of `−5` | 1 line |
| Add snake length (or free-cell count) | **measured gap** | no length signal exists at all for length ≥ 50 | free, 1 value |
| Add time-aware reachable area, per action | **measured** | 92.6% right / 0% wrong; median 2 cells for the fatal branch vs 99 | ~200 µs/step as written |
| Add "is the food reachable", the slot already reserved in `observation_spec` | **measured gap** | food sealed off from every legal move on 33.9% of late steps | ~free |
| Rescale input 18 | theory only | it reaches 8.97 while every other input is ≤ 3.17 | free |
| Remove `lg(num_groups)` (3 values) | **measured** | right 7.4% / **wrong 57.4%** at the fatal decisions | frees 3 inputs |
| Landing-cell degree | marginal | 61.8% right / 8.8% wrong | free |
| Static free area per action | **rejected** | 48.5% right / 36.8% wrong — a coin flip | — |
| Straight-ahead corridor rays | **rejected** | 35.3% / 32.4% — no signal | — |
| Cell parity / board colour | **rejected on theory** | every orthogonal move flips colour, so all three actions land on the same one | — |

The single most valuable change is not a new observation. It is a correction to an existing one —
`group_obs` uses one tail position for a tail that has usually moved — and it was found by
measuring which observation would have distinguished the move that lost each game from a move
that survived.

---

## How this was measured

### The setup

The champion policy played 360 episodes across six processes (seeds 101-106), greedy, in the
current environment. It won 288 — **80.0%**, consistent with the 82% that `eval_checkpoints`
measured for this checkpoint over 100 episodes, since a 360-episode binomial interval here is
about ±4 points.

At every step, for each of the three actions, the diagnostic recomputed facts the policy is
**not** given: the free area the head would land in, whether the food would still be reachable,
the degree of the landing cell, and time-aware versions of all of these.

### The failure attribution

Every lost episode was replayed through an independent simulator. At each of its last 40
decisions, for each legal move, a tail-following heuristic tried to survive 60 more steps. The
**latest decision at which the move the policy chose loses and some alternative survives** is
the pinpointed mistake. The facts about the board at that moment then say which observation
would have flagged it.

### What the attribution does and does not prove

It identifies **avoidable death**, not a lost win. The survival heuristic ranks moves by
"keeps the tail reachable, then maximises area" and therefore *avoids food*, which is legitimate
for a survival test and is why it should not be mistaken for a solver — driven as a food-seeking
player it starved in 12 of 12 games at a maximum length of 87. A branch the heuristic survives
was definitely survivable; a branch it fails may still have been survivable by a better player.
So "some alternative survived" is solid, and "no alternative survived" is weak.

It also shows that a distinction is **representable**, not that a network will learn it. Every
percentage below is an upper bound on what adding the feature can buy.

### Validation of the instruments

Two checks, because every number in this document depends on them:

- **The simulator agrees with the game.** Body, food and death compared step by step: 664 steps
  and 60 deaths under random play (60 of 60 death calls agreed), then 16,173 steps up to length
  87 under a food-seeking heuristic. **Zero mismatches.** Random play dies around step 11, which
  is why the second, longer check exists — the first never reached the crowded states the
  attribution runs in.
- **The recomputed observation is the real observation.** The recomputation of
  `head_with_tail` and `lg(num_groups)` was compared against the live observation vector on
  every legal action of all 1,076,492 steps. **Zero mismatches.** An early version disagreed on
  13.9% of steps; the cause was that `group_obs` computes the flag for fatal moves too, where
  the prospective head sits inside a wall and the test is meaningless.

---

## What the failures actually look like

| outcome | episodes | share |
|---|---|---|
| perfect | 288 | 80.0% |
| collision | 72 | 20.0% |
| **starved** | **0** | **0%** |

| score when the game was lost | episodes |
|---|---|
| below 50 | 5 |
| 50-79 | 31 |
| 80-89 | 21 |
| 90-94 | 15 |

Half of all losses happen at score 80 or better, and 15 died within five food of winning. The
policy is not failing to learn the game; it is failing in the endgame.

| how the loss arrived | count |
|---|---|
| zero safe moves available on the final step | 67 of 72 |
| one safe move available on the final step | 5 of 72 |
| chose a move the safety flags already called fatal | 2 of 70 |
| pinpointed to one decision within the last 40 | **70 of 72** |
| no survivable branch anywhere in that window | 2 of 72 |

| steps between the fatal decision and the end | count |
|---|---|
| 0 (the final decision) | 2 |
| 1-3 | 39 |
| 4-10 | 18 |
| 11 or more | 11 |

Median 2, p90 13, **max 29**. Snake length at the fatal decision: median **83**, range 11-97.

### Why this points at representation rather than horizon

At `DISCOUNT=0.9975` the effective horizon is about 400 steps. **Every fatal decision was within
29 steps of death**, and 41 of 70 within three. The information needed to avoid these deaths is
comfortably inside the horizon the value function already spans, and the deaths are not
credit-assignment failures at long range. They are cases where the inputs at the moment of
choice do not distinguish the fatal move from the safe one.

That matters for what to do about it: a longer discount, n-step returns, or a bigger buffer
address a problem these measurements do not find. Two of 72 losses were doomed more than 40
steps out; those are the planning failures, and they are 3% of the total.

---

## Finding: `head_with_tail` asks a question that goes silent in the endgame

### The mechanism

`group_obs` in [`state_helpers.py`](state_helpers.py) computes "can the head still reach the
tail" like this:

```python
head_groups = get_adjacent_groups(regions, cols, new_head_pos)
tail_groups = get_adjacent_groups(regions, cols, tail_pos)
if len(head_groups & tail_groups) > 0 or tuple(new_head_pos) == tail_pos:
    head_with_tail = 1
```

For the head, adjacency is the right query: after the move the head's own cell is occupied, so
the only way to ask what it can reach is to look at its neighbours.

For the tail it is the wrong query, for two reasons that turned out to be one.

**The tail position is never advanced.** `group_obs` is handed a single `tail_pos` and uses it
for every action, but the tail's position after the move depends on whether the move eats. On an
ordinary step the tail vacates, so the real post-move tail is the segment *ahead* of it. On a
step that eats, `add_segment` refills the cell and the tail does not move at all. One argument
cannot be right for both, and it is right only in the eating case.

**And the cell it does use is queried by adjacency.** `update_grid` **frees** the tail cell,
which was audit fix number 6, so the cell is open and belongs to a region. In a coiled endgame it
is typically surrounded on all four sides by the snake's own body, making it a **singleton region
with no open neighbours**. `get_adjacent_groups` returns the empty set, the intersection is empty,
and the flag reads 0 no matter where the head goes.

There is an unhappy interaction with the audit here. Before fix number 6 the tail cell stayed
occupied; after it, the cell is open and the natural query changes. The fix supplied new
information that the unchanged query throws away, precisely when the snake is coiled. That is
consistent with the standing open question in `findings.md` — that nothing trained after the
audit has beaten a pre-audit checkpoint — though it is a hypothesis that fits, not a proven
cause.

### The measurement

Four versions, all scored on the same 360 games. "Flags" means the version marked a surviving
move as tail-reachable and the move the policy took as not, at the 68 losing decisions where the
chosen move was legal. The right-hand column counts disagreements with the time-aware walk over
all 1,992,387 legal actions.

| version of the test | flags the fatal move | both true | both false | wrong way | differs from time-aware |
|---|---|---|---|---|---|
| **observed** — what is in the vector today | 15 — **22.1%** | 4 | **49** | 0 | 33,049 |
| **holding** — also count the region containing the cell | 63 — 92.6% | 4 | 1 | 0 | 23,173 |
| **advanced tail** — use the post-move tail | 64 — **94.1%** | 4 | 0 | 0 | **20,032** |
| both of those together | 64 — 94.1% | 4 | 0 | 0 | 20,032 |
| **timed** — body cells open as they vacate | 63 — 92.6% | 5 | 0 | 0 | 0 |

In 49 of 68 fatal decisions the current flag is 0 for *both* options. It never lies — zero wrong
in every row — it simply goes dark exactly where games are decided. Over the 462,274 steps at
length ≥ 50, the fraction where the test claims *no* legal move keeps the tail reachable is
0.66% observed, against 0.12% for the holding variant and 0.05% timed.

**Advancing the tail subsumes the holding patch.** The two disagree on 3,381 of 1,992,387 legal
actions and every one of those 3,381 is a case the advanced tail catches and holding misses, so
the union of the two is byte-identical to the advanced tail alone. There is no reason to carry
both. It also lands closer to the time-aware walk than holding does, and at the losing decisions
it edges out that walk, 64 against 63 — so the expensive breadth-first version is not needed for
this feature at all.

One decision from the run, at length 87: the branch the policy took had **10** free cells and
the branch that survived had **2**. Time-aware, they are 10 and 99. The tiny gap opens as the
body passes; the roomy pocket never does.

### The fix

`group_obs` needs one more input — the position of the segment ahead of the tail, which is
`self.tail.front_segment.tile_pos` and is already maintained by `add_segment`. Then, per action:

```python
# tail_pos is right only when the move eats, because add_segment refills that cell
post_move_tail = tail_pos if eats_food else ahead_of_tail_pos
tail_groups = get_adjacent_groups(regions, cols, post_move_tail)
if len(head_groups & tail_groups) > 0 or tuple(new_head_pos) == tail_pos:
    head_with_tail = 1
```

`eats_food` is already computed inside `update_grid`, and `regions` is already computed, so this
costs nothing measurable. Guard `front_segment` being `None` at length 1, which cannot happen
with `START_SEGMENTS = 4` but is one line.

**Keep the `new_head_pos == tail_pos` special case, and note it compares against the old tail.**
Following your own tail is always safe, and no region test can see that move, because the cell
ends up holding the head. A first measurement that dropped this clause while advancing the tail
looked catastrophic — 1,481 spurious disagreements in 15,700 actions — and the entire difference
was that one move.

### Why it works

Tail-reachability is not an arbitrary heuristic; it carries a survival guarantee. If a path
exists through free space from the head to its own tail, the snake can survive indefinitely by
following that path, because the tail retreats ahead of the head as it advances. Eating breaks
the guarantee (the snake grows), and it is a guarantee about survival rather than winning — but
it is the reason this feature is the right one to have, and the reason a *correct*
implementation of it should dominate area-style heuristics.

The current implementation tests a proxy for that property which fails in exactly the
configuration where the property is the only thing keeping the snake alive. The fix restores the
question the feature was meant to ask.

That the cheap fix matches a full time-aware breadth-first walk — 94.1% against 92.6% — says
something useful about how much look-ahead this feature needs. Asking about the tail's position
*one step into the future* captures essentially all of the value of simulating the whole vacate
schedule, because the head only ever has to reach the tail, and the tail is only ever one step
ahead of where the snake is going.

---

## Finding: the agent cannot see its own length after 50

`steps_until_starve` is the only input that depends on length:

```python
# MAX_STEPS_BEFORE_STARVE_SIZE_MULTIPLIER is 10
max_steps_until_starve = min(snake_len * MAX_STEPS_BEFORE_STARVE_SIZE_MULTIPLIER, 500)
return [log2plus1(max(100, max_steps_until_starve) - (current_step - last_food_step))]
```

Measured at zero elapsed steps:

| snake length | value |
|---|---|
| 5 | 6.6582 |
| 10 | 6.6582 |
| 20 | 7.6511 |
| 40 | 8.6475 |
| 49 | 8.9396 |
| **50 through 99** | **8.9687 — one single value** |

Both ends are clipped: the floor of 100 flattens everything below length 10, and the cap of 500
flattens everything from 50 up. **For the entire second half of every game the observation
carries no information about the snake's length**, and what information it does carry is
entangled with time since the last food.

The only other length-dependent inputs are the three perfect-game flags, which fire on about one
step per won episode.

### Why it matters theoretically

Two reasons, and the second is the one that connects to this project's central problem.

The policy reason is aliasing: a memoryless network must map identical inputs to identical
Q-values, so two states that differ only in length get one compromise policy. Correct play at
length 20 (chase food) and at length 90 (thread the coil) are not the same, so a compromise is
wrong in both.

The value-function reason is worse. The remaining return depends enormously on length — at score
80 there are 15 food and a +100 terminal bonus left; at score 20 there are 75 food and a bonus
some 2,000 steps further away, which at 0.9975 is discounted by roughly an order of magnitude.
A value function that cannot see length is **fitting a target that depends on a variable it
cannot observe**, so the residual TD error is irreducible rather than shrinking with training.
That is noise injected into every gradient, forever, and it is a plausible contributor to the
instability and seed-to-seed spread this investigation keeps measuring. Adding length is free —
`snake_len` is already a parameter of `get_observations`.

---

## Finding: the terminal bootstrap is never cut off

This was found while checking whether input 19 (`game over`) could be deleted as a dead input.
It cannot, and the reason is a defect elsewhere.

`SnakeEnvironment.to_tensor_time_step` sets the discount unconditionally:

```python
return TimeStep(step_type=...,
                reward=...,
                discount=convert_to_tensor(self._discount, dtype=np.float32),
                observation=...)
```

Every step gets `0.9975`, including `StepType.LAST`. The tf-agents convention is the opposite:
`ts.termination()` sets `discount = 0.0`, and that zero is the *only* mechanism that stops the
bootstrap. Confirmed in the installed framework:

- `dqn_agent._loss`: `discounts = gamma * next_time_steps.discount`, with `gamma` defaulting to
  1.0 and `snek2.py` not overriding it, so all discounting comes from the time step.
- `common.compute_td_targets`: `rewards + discounts * next_q_values`.
- `valid_mask = tf.cast(~time_steps.is_last(), tf.float32)` masks transitions whose **current**
  step is terminal — the boundary transitions — not the bootstrap off a terminal next state.

So the final transition of every episode is trained toward
`reward + 0.9975 · max_a Q_target(terminal observation)` rather than toward `reward`.

### Why this matters

For scale: at score 80, with the +100 win bonus roughly 400 steps away, `V` is plausibly in the
tens. A death whose target is `−5 + 0.9975·V(terminal)` is therefore not a penalty at all unless
the network has independently learned that terminal observations are worth ~0.

It clearly has learned much of that, or the policy would suicide immediately, and it does not.
Input 19 is a perfectly reliable indicator, and the network has every incentive to key on it.
Three consequences remain:

- Capacity and gradient budget are being spent learning something the framework enforces for
  free, from roughly one example per 2,700 steps of experience.
- Early in training, before that is learned, deaths are under-penalised — and early training is
  exactly when arms in this project collapse.
- The −5 death penalty is diluted by whatever residual value the network assigns to terminal
  states, which weakens precisely the signal needed to avoid the traps measured above.

**Ordering matters.** Input 19 is 0 in 100% of the states the policy acts in, which makes it look
like an obvious deletion. Deleting it while the discount is wrong would remove the only signal
that terminal states are worthless. Fix the discount first; then the input is genuinely
redundant and can go.

The fix is one line — `0.0` when `step_type` is `LAST`, `self._discount` otherwise — and it
changes the training target, so it invalidates comparisons across the change just as the audit
did.

---

## Finding: `lg(num_groups)` points the wrong way

The region count occupies three of the twenty inputs. At the 68 fatal decisions, taking "fewer
regions is better" as the rule: **right 5, wrong 39, tied 24** — right 7.4%, wrong 57.4%.

The surviving branch usually leaves *more* regions than the fatal one. That inverts the
intuition the feature was built on, and the explanation is the same one that sinks static area
below: what kills the snake is not splitting the free space, it is entering the wrong piece.
Splitting is normal and often correct.

Three inputs that are anti-correlated with survival at the decisions that matter are worth
removing, both to stop misleading the network and to make room for observations that measured
well.

---

## Finding: the food observations describe a target that is often sealed off

Six of the twenty inputs describe the food through **Manhattan distance**, which ignores the
body entirely. Measured over the run:

| | steps | share |
|---|---|---|
| food unreachable from every legal move | 180,719 | 16.8% of all steps |
| same, at length ≥ 50 | 156,529 | **33.9% of late steps** |

For a third of late-game steps, six inputs are pointing at a target that no move can reach, and
three of them assert that a move gets "closer" to it.

`SnakeEnvironment.observation_spec` already reserves the slot:

```python
head_with_food_obs = 0      # head is in same group as food
```

Someone intended this and disabled it. The regions are already computed for `group_obs`, so
turning it on is close to free.

Two honest caveats. There were **zero starvations** in 360 games, so this is not killing the
policy today — it has learned to stall through sealed-food periods. And the effect on the
perfect rate is unmeasured; the strong claim here is only that six inputs are actively wrong a
third of the time in the endgame, which is a plausible drag on learning speed rather than a
demonstrated one. The stronger version of the change is a path distance to the food (a breadth
-first distance through free space, once per step rather than once per action, ~60 µs) which
would replace a misleading number with a correct one instead of adding a flag beside it.

---

## Finding: input scales and near-dead inputs

Occupancy and range of all twenty inputs, over 3,660 states:

| index | contents | non-zero | range |
|---|---|---|---|
| 0, 2, 4 | food is closer, per action | 39.6-51.5% | 0 .. 1 |
| 1, 3, 5 | 1/(distance+1), per action | 100% | 0.053 .. 1 |
| 6-8 | move is safe, per action | 63.4-73.1% | 0 .. 1 |
| 9, 11, 13 | tail reachable, per action | 61.5-76.6% | 0 .. 1 |
| 10, 12, 14 | lg(region count), per action | 100% | 1.0 .. 3.170 |
| 15-17 | move wins the game, per action | ~0% | 0 .. 0 |
| 18 | lg(steps until starving) | 100% | 1.0 .. **8.969** |
| 19 | episode is over | **0%** | 0 .. 0 |

Input 18 is roughly three times the magnitude of the next largest input and nine times the
binary flags, feeding a first layer of 50 units with no input normalisation anywhere in
`build_q_net`. With a shared weight initialisation, the largest-magnitude input dominates early
gradients and effectively sets its own learning rate relative to the others. Dividing by
`log2(501)` is free and costs no information.

Inputs 15-17 fire on about one step per won episode. They are not useless — they are the only
place the network can learn "this move wins", and they cannot be derived from the food inputs
because the network does not know its own length. That is another argument for adding length
rather than for deleting these.

---

## Candidates that measured badly

Recorded because a plausible-sounding feature that measurably does not work is worth as much as
one that does. All scored on the same 68 decisions, "right" meaning the candidate ranked a
surviving move above the one that lost:

| candidate | right | tie | wrong | median, fatal vs surviving branch |
|---|---|---|---|---|
| time-aware tail reachable | 92.6% | 7.4% | 0% | 0 vs 1 |
| time-aware reachable area | 92.6% | 7.4% | 0% | 2 vs 99 |
| time-aware walk depth | 92.6% | 7.4% | 0% | 3 vs 83 |
| landing-cell degree | 61.8% | 29.4% | 8.8% | 1 vs 1 |
| **static free area** | 48.5% | 14.7% | **36.8%** | 2 vs 1 |
| **straight-ahead corridor length** | 35.3% | 32.4% | **32.4%** | 0 vs 1 |

**Static free area is a coin flip, and I expected it to be the headline recommendation.** Among
the 53 decisions where both branches agreed on the tail flag, the surviving branch had more room
in 26 and less in 19. Area minus length is about −82 for both branches — at length 83 the free
area is a handful of cells either way, so "is there room for my body" carries no information in
the endgame at all.

There is a reason, and it is visible in the policy's own behaviour: when the areas of two safe
moves differ by 5 cells or more, the champion enters the **smaller** area 62.7% of the time.
That is not a flaw. Cleaning up a small pocket while the tail is still adjacent to it is correct
play; leaving it for later is how it gets sealed. So a feature that says "prefer more room" is
advising against a winning strategy, which is why raw area scores near chance and the time-aware
version — where the pocket that reopens scores 99 and the pocket that does not scores 2 —
scores 92.6%.

Board parity is rejected on theory rather than measurement: every orthogonal move flips the
checkerboard colour of the head, so all three actions land on the same colour and a parity input
carries no per-action signal whatsoever. Its value in a Hamiltonian-cycle framing is real, but
it has to enter as part of a cycle representation, not as a per-move feature.

---

## Aliasing: what the vector provably cannot distinguish

Grouping all 1,076,492 steps by their exact observation vector:

| | exact 20-value vector | geometry only (input 18 dropped) |
|---|---|---|
| distinct classes | 655,243 | 54,968 |
| steps sharing a vector with another state | 54.9% | **98.1%** |
| snake-length spread within a class, median | 9 | **34** |
| same, p90 | 28-32 | 70-72 |
| same, max | 77-83 | **93** |
| reachable-area spread within a class, median | 8 | 28-32 |
| same, p90 | 27-30 | 69-71 |

Read the right-hand column as: strip the starve clock, and the same nineteen numbers describe
board states whose snake lengths differ by 34 on the median and by as much as 93. The starve
clock is doing most of the work of telling states apart, and it is a clock — it counts steps
since the last food, resetting on every food, rather than describing the board.

Within a single exact-vector class the reachable free area still differs by a median of 8 cells.
That is the sense in which the missing information is provable: these are states the network
cannot tell apart, and they differ materially in the quantity that decides whether the next move
is fatal.

---

## Speculative ideas with a high ceiling

None of these is supported by the measurements above; they are listed because the ceiling is
plausibly higher than anything in the confident tier.

**Frame stacking or recurrence.** The network is memoryless, and its only phase cue is a clock
that resets on every food. Stacking the last two to four observation vectors is a small change
and would let the network infer where its tail is heading and whether space is opening or
closing, which is the dynamic quantity the time-aware features approximate statically. Standard
practice in DQN, cheap, and it composes with everything else here.

**Hamiltonian-cycle scaffolding.** A perfect game *is* a Hamiltonian path over the board, and
known 100% snake solutions follow a cycle. An input for "does this move follow a precomputed
cycle", or a shaping term for adherence, hands the agent the endgame structure instead of asking
it to rediscover it. Highest ceiling of anything in this document and the largest change; the
risk is that it teaches cycle-following rather than play, and that the cycle constraint is
wasteful early when greedy food-chasing is optimal.

**Potential-based shaping on the corrected tail flag.** Give up tail contact and pay a small
cost immediately, rather than only at the death that follows 2-13 steps later. Expressed as a
difference of potentials it is policy-invariant in theory. In practice this is where reward
hacking appears, and this project has been burned by shaping before — the old
`FOOD_DISTANCE_REWARD` fired on 96.8% of food-eating steps.

**The time-aware walk as a bitwise dilation.** If the time-aware area proves worth its cost,
the set-based implementation measured here (66 µs per action, 198 µs for three, five times the
entire current observation vector) is not the way to ship it. `count_groups` already does flood
fill as bitmask dilation at 5.7 µs; the same trick applies with a schedule — unmask each body
segment's bit as its vacate time arrives, then dilate. One round per time step on a ~144-bit
integer.

---

## What this predicts about reaching 100%

**Not 100% from observations alone. Roughly 95% looks reachable; the last few points need
something other than features.**

The arithmetic: 94.1% of the losing decisions are flaggable by the corrected tail test, 5.9% read
the same on both branches for every candidate tried, and 2 of 72 losses were doomed more than 40
steps before the end. At today's 20% loss rate, a network that learned the corrected flag
perfectly would leave something like 1.5-2% of games lost. Reaching zero from there requires
either explicit search at decision time, a cycle-following endgame, or a safety layer that vetoes
moves — none of which are observation changes.

Two caveats keep this from being a forecast. Making a distinction representable is not the same
as a network learning it, so 94.1% is a ceiling and not a prediction. And the fatal decisions
sit at median length 83, in states reached only after a long correct game, which is the thinnest
part of the training distribution — the network sees far fewer of those states than mid-game
ones, however good the features are.

---

## Cost budget

Measured at body length 75:

| operation | cost |
|---|---|
| the entire current observation vector | 39.3 µs |
| one `count_groups` flood fill | 5.7 µs |
| one time-aware walk (set-based) | 66.0 µs |
| three time-aware walks | 197.9 µs |

The corrected tail flag, snake length, food reachability, the rescale and the input removals are
all free or nearly so. Only the time-aware area has a real price, and at ~200 µs against a
training step of roughly 1.3 ms it is a ~15% throughput cost — worth paying only if the free
changes do not deliver, and worth reimplementing as a bitmask dilation first.

---

## How to test this without wasting a batch

Any change here breaks comparability exactly as the 2026-08-01 audit did. Both the observation
changes and the terminal-discount fix alter the MDP, so every existing figure becomes a
different environment's figure.

The free changes should go in as **one batch, not a sequence**: the corrected tail flag, snake
length, the rescale, removing `lg(num_groups)`, the food-reachability flag, and the terminal
discount. Batch 9 measured an 18-point spread between two seeds of the same config, which is
wider than any effect worth detecting, so a single arm cannot measure any of this — two seeds
minimum, and the comparison is against a re-baselined batch on the same code rather than against
anything already recorded.

If the batch does not help, the ablation order suggested by the evidence is: corrected tail flag
first (largest measured effect), terminal discount second (certain mechanism, unknown
magnitude), length third (measured gap, theoretical argument), and the rest are cleanups.

---

## Appendix: reproduction

The diagnostics were scratch scripts in the session scratchpad and were not preserved. What
matters is reproducible from the descriptions above; the pieces are:

- Play a fixed checkpoint greedily, recording per-action facts recomputed from `game.grid`,
  `game.head.tile_pos`, `game.tail.tile_pos` and `game.snake.get_positions()`.
- A standalone simulator: `new_head` from `CURRENT_DIRECTION_MAPS`; the move is fatal if it
  leaves the board or lands on `body[:-1]`, or on all of `body` when it eats; the tail vacates
  unless the move eats. Validate it against `Snake.step()` before trusting it.
- Attribution: buffer the last 40 states, and for each one test each legal move by rolling a
  tail-following heuristic forward 60 steps.
- The time-aware walk: breadth-first from the new head, where a cell holding body segment `i`
  (0 = head) is passable to a walker arriving at time `d` when `d >= len(body) - i`.
- Aliasing: hash the rounded observation vector, and track the min and max of snake length and
  reachable area within each class.

Every headline number in this document came from six shards of 60 episodes each, and the
per-shard spreads were tight enough that the aggregates are not driven by one run.
