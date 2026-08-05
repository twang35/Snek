# Archive: superseded findings

Findings moved out of `findings.md` on 2026-08-04, in two groups: work on observation
vectors this project has since replaced (the vector was 20, then 21, 23, 26 and is now 30),
and per-batch config results from batches 1-8 that later batches settled or overturned.

**Historical record — not meant to be read into context.** Everything still load-bearing was
condensed into [`../findings.md`](../findings.md) rather than moved; see
[`README.md`](README.md) for what that means.

## Environment audit 2026-08-01: observations and rewards both changed

**Nothing measured before 2026-08-01 is comparable to anything measured after it.** Six
environment bugs were fixed in one pass. Two of them change what the agent sees, one changes
what it is paid, and one changes how long it may go without food — so every arm in
`completedRuns.md`, every number in this file, and the `hallOfFame` entries were produced by a
*different* MDP than the one that runs now. Re-baseline before comparing anything across that
line.

| # | fix | changes | measured effect of the bug |
|---|---|---|---|
| 1 | `reset()` built a `(10, 10)` unbordered grid; `step()` builds `(12, 12)` with walls | observation | 0 of 500 first-observations differed numerically; latent `IndexError` |
| 2 | Distance shaping compared the eaten food to its replacement | reward | fired on 96.8% of eat steps (n=4825), making `FOOD_REWARD` 0.999 |
| 3 | `SCREENTILES` were floats, so `random.randint(0, 9.0)` | neither | `DeprecationWarning` on 3.10, `TypeError` on 3.12+ |
| 4 | `last_food_step` set before the step counter incremented | starve budget | every budget was short by one step |
| 5 | `CLOSER_TO_FOOD_REWARD_*` unused | neither | dead code |
| 6 | `update_grid` never freed the tail tile | **observation** | `num_groups` wrong on 12.1% of steps, 40.0% past score 80 |

#### #6 is the one that matters

`group_obs` simulates each candidate move to report how fragmented the free space is and
whether the head can still reach its tail. It marked the prospective head cell as occupied but
never freed the tail cell, so connectivity was computed as though the tail were a wall and any
region reachable only *through* the tail read as sealed off. Measured against the 92% policy
over 20,814 steps:

| score | steps | `head_with_tail` differs | `num_groups` differs |
|---|---|---|---|
| 0-19 | 3,543 | 1.1% | 2.8% |
| 20-39 | 6,826 | 0.6% | 4.5% |
| 40-59 | 5,110 | 0.7% | 15.9% |
| 60-79 | 3,299 | 1.6% | 14.8% |
| **80-99** | 2,036 | **3.5%** | **40.0%** |

All 258 `head_with_tail` flips were `0 → 1` — the feature only ever claimed the snake was
trapped when it was not, never the reverse. The error was therefore a **pessimism bias**, and
it concentrated in the endgame, where free space is a thin channel and one cell is much more
likely to be the bridge between two regions.

The fix has to be conditional: on a move that eats, `add_segment()` refills the tile the tail
came from, so it stays occupied. Verified over 39,684 steps — 5,068 eat steps always kept the
old tail tile as body, 34,616 non-eat steps always freed it. That conditional is why the
original line was commented out rather than missing, so this was a parked trade-off, not an
oversight.

#### But fix #6 was incomplete, and the same conditional was missed elsewhere

Freeing the tile was necessary and not sufficient. `group_obs` is handed a single `tail_pos` and
uses it for all three actions, yet the post-move tail is the **segment ahead of it** on an
ordinary step and that same cell only on a step that eats — the identical eat/no-eat distinction
that fix #6 turned on. Worse, the cell it does use is queried by *adjacency*, and now that the
cell is free it is usually a singleton region with no open neighbours in a coiled endgame, so the
flag reads 0 whatever the head does.

Measured 2026-08-02 over 360 episodes of the champion, at the 68 decisions that actually lost
games:

| version of the tail test | flags the losing move | wrong way |
|---|---|---|
| as shipped | 15 — **22.1%** | 0 |
| `tail_pos` advanced to the post-move tail | 64 — **94.1%** | 0 |
| a full time-aware walk, for reference | 63 — 92.6% | 0 |

The flag never lies; it goes silent, and it goes silent exactly where games are decided. Note the
direction is the same **pessimism bias** as bug #6 itself, from the same cause one layer up.

Full write-up in [`../claudeFeatureRecommendations.md`](../../claudeFeatureRecommendations.md),
including a ranked scoring of every other candidate observation, the terminal-discount defect, and
the absent length signal. Instruments in [`diagnostics/`](../diagnostics/). Both are frozen at
2026-08-02.

#### Fixed 2026-08-02, and it was worth 10 points with no retraining

`group_obs` now takes both tail positions and picks between them per action: the tail advances to
the cell ahead of it, except on a move that eats, where `add_segment()` refills the tile it came
from and the tail does not move. The observation width is unchanged at 20, so old checkpoints
still load.

| `b8f-disc9975seed2` @3149000, greedy, 360 episodes | perfect | 95% CI |
|---|---|---|
| broken `head_with_tail` | 288/360 — 80.0% | 75.9-84.1 |
| **fixed** | **325/360 — 90.3%** | **87.2-93.3** |

**+10.3 points on a policy that was never trained with the corrected feature**, and the intervals
do not overlap. That direction is the opposite of the audit, which cost old policies ~10 points,
and the asymmetry has a mechanism: the audit changed what the features *said*, while this changed
whether one of them said anything at all. The policy had already learned to trust
`head_with_tail = 1`; the flag was simply mute in coiled endgames, and unmuting it lets behaviour
the network already had fire where it previously could not.

Where the remaining 35 losses sit is the more interesting part:

| score at the loss | broken | fixed |
|---|---|---|
| below 50 | 5 | 1 |
| 50-79 | 31 | 12 |
| 80-89 | 21 | 8 |
| **90-94** | **15** | **14** |

Mid-game deaths collapsed; deaths within five food of a win did not move at all. The endgame
residual is a different problem from the one this fixed, which is what
`claudeFeatureRecommendations.md` predicted when it put the ceiling from this feature near 95%
rather than 100%.

**This is a second comparability boundary.** Everything measured between 2026-08-01 and this fix
was produced by a different MDP than the one that runs now, so batch 9's figures and the 80.0%
baseline above are both historical. Re-baseline before comparing.

## The starve observation, 2026-08-02: split in two, and the vector is now 21 wide

The third boundary of the day, and the first one that **changes the width of the observation**,
so unlike the previous two it breaks checkpoint loading outright.

`steps_until_starve` was a single value, `log2plus1(budget - elapsed)`, and it was doing three
jobs badly at once.

| problem | detail |
|---|---|
| length was invisible | the budget is `min(10 * len, 500)`, so **every length from 50 to 99 gave the identical 8.9687** at equal elapsed steps |
| it was the only length signal | so the second half of every game had no phase information at all, while fatal decisions sit at median length 83 |
| the scale was an outlier | it reached **8.97** where every other input was at or below 3.17, with no input normalisation in `build_q_net` |
| it drove the game rule too | `Snake.step()` starved on `steps_until_starve(...)[0] <= 0`, so rescaling the observation would have moved the starvation threshold |

Now there are two observations and a separated rule:

- `starve_budget(len)` and `steps_until_starve(...)` return plain step counts, and the rule reads
  those. The observation no longer shares a function with the rule.
- Index 18 is `log2plus1(remaining) / log2plus1(500)`, in `[0, 1]`, keeping the log compression —
  10 versus 20 steps of budget is worth reacting to, 400 versus 410 is not. Scaled by the maximum
  budget rather than each snake's own, so one value always means the same number of steps.
- Index 19 is `snake_len / PERFECT_SCORE`, linear, because 80 versus 90 segments matters at least
  as much as 20 versus 30. It also supersedes the `remaining_spaces` slot `observation_spec` had
  reserved and disabled, open cells being the complement of length.

**The starvation rule is provably unmoved**: `log2plus1(x) <= 0` exactly when `x <= 0`, checked as
a unit test across the boundary for five lengths, and confirmed over 14,984 steps of real play
with 9 starvations and **0 disagreements** with the old formula. Starvation still fires at 500
elapsed for long snakes and at `10 * len` for short ones.

The largest value anywhere in the vector is now **3.585** rather than 8.97. The remaining outlier
is `lg(num_groups)` at indices 10, 12 and 14 — which
[`../claudeFeatureRecommendations.md`](../../claudeFeatureRecommendations.md) recommends removing
rather than rescaling, having measured it as right 7.4% and wrong 57.4% at the decisions that lose
games.

**Effect on the perfect rate is unmeasured, and cannot be measured by transfer.** Unlike the
`head_with_tail` fix there is no policy-that-never-saw-it to re-run, because a policy trained on
the old value cannot read the new one. This needs a training arm to evaluate.

## The terminal bootstrap, 2026-08-02: never cut off, now cut off

The fourth change of the day, and the only one that alters the **training target** rather than the
observation. `SnakeEnvironment.to_tensor_time_step` set `discount = self._discount` for every step
type, including `StepType.LAST`. tf-agents' own `ts.termination()` sets `0.0`, and that zero is the
only mechanism that stops the bootstrap:

- `dqn_agent._loss`: `discounts = gamma * next_time_steps.discount`
- `common.compute_td_targets`: `rewards + discounts * next_q_values`
- `DdqnAgent` is built without `gamma` in `snek2.py`, so `gamma` is **1.0** and the time step
  carried all of the discounting
- the `valid_mask` in that loss drops transitions whose *current* step is LAST — not the bootstrap
  off a terminal *next* state

So every episode's final transition was trained toward `reward + 0.9975 * V(terminal)` instead of
`reward`. Verified directly on a real terminal step: the target was `-4.932` where the reward is
`-5.0`, and is now exactly `-5.000`. That demo used a freshly initialised network, so `V(terminal)`
was only `+0.068`; on a trained network, with a `+100` win bonus a few hundred steps out, it is
plausibly in the tens, and the `-5` death penalty was being diluted by that much.

**Two consequences beyond the arithmetic.**

The `game_over` observation is gone, taking the vector from 21 values back to 20. It was 1 only in
a terminal observation, which no policy ever acts on, so it was a constant 0 input — but it could
not simply be deleted while the bootstrap existed, because it was the only signal the network could
use to learn that terminal states are worth nothing. That is now nothing's job.

**The n-step falsification needs re-testing.** `to_n_step_transition` composes
`r_t + g*d_t*r_{t+1} + g^2*d_t*d_{t+1}*r_{t+2} + ...`, and those per-step `d` values are the only
truncation at an episode boundary. With a terminal `d` of 0.9975 the sum runs straight through the
end of the episode and into whatever follows it in the buffer. The `n=2` and `n=3` arms that
falsified n-step returns were therefore measuring something other than n-step returns. This does
not mean n-step helps — it means the experiment was not clean, so the entry in the falsified
section is downgraded to open rather than reversed.

**Also unmeasurable by transfer**, for a sharper reason than the starve change: this one alters
what the agent is trained toward, not what it sees, so a policy already trained cannot show
anything about it either way. It needs an arm.

## Safe to chase the food, 2026-08-02: three values, and the vector is 23

The fifth change of the day, and the first that *adds* a signal rather than fixing one. Per action,
1 when the head, the food and the tail all land in **one** region after the move — a route to the
food and a route back out. It fills the `head_with_food_obs` slot that `observation_spec` has had
reserved and disabled since before the audit, with a stricter test than that slot was named for.

The gap it closes: the food values say which way the food is, `head_with_tail` says an escape route
exists, and nothing tied the two together, so a policy had no way to distinguish a reachable meal
from one that seals it in. Relevant given that the food is sealed off from every legal move on
**33.9% of late steps**.

One region, not three reachable things. The head can neighbour two regions at once, and taking the
food through one while the only escape is through the other is exactly the trap — so the test is
`head_groups & tail_groups`, intersected with the region holding the food. Two special cases earn
their keep, both caught by dedicated tests after a mutation check:

| case | why it needs handling |
|---|---|
| the move eats | the food's cell becomes the head, belongs to no region, and would always read 0 — the move the flag exists to encourage |
| the move follows the tail | no region test sees the tail from the cell it is vacating, so intersecting would veto a move that is safe by construction |

Measured over 15,316 steps of heuristic play: the flag is 1 on **32.5%** of action slots, at least
one action is chaseable on 43.2% of steps but only **20.6% of late steps**, and there were zero
cases of it claiming safety where `head_with_tail` was 0. So it is informative rather than
degenerate, and conservative exactly where the board is crowded.

**It is free.** `get_observation()` costs 39.1 µs against 39.3 µs before the observation existed,
because it is computed inside `group_obs` and shares the flood fill — which is ~46% of the cost of
an observation and would otherwise have run three more times per step.

Effect on the perfect rate is unmeasured and needs an arm, like the two changes above it.

#### Fatal moves are zeroed, not hypothesised

A follow-up the same day, after the user asked directly: for a move that runs into a wall or the
snake's own body, what happens to `head_with_tail`, `lg(num_groups)` and safe-to-chase? Measured
answer, before this fix — a wall move could never read 1 (an off-board cell's only on-board
neighbour is the vacated head cell, which `update_grid` never clears, so no region test finds
anything past it), but a **body** collision had no such accident protecting it: the new head cell
is still on-board and can see whatever real regions sit beside it, so `head_with_tail` read 1 on
**5,289 of 14,642** body-collision moves in a sweep of real games — a hypothetical answer about a
snake that does not survive the move that produced it.

`group_obs` now checks legality first, using the same test `body_and_wall_collisions` already
makes — food, open, or the tail-follow special case — and short-circuits a fatal move to zero
before running the flood fill at all, rather than computing what a survivor would have found.
This is strictly safer than the accident it replaces: the wall case kept its existing zero, and the
body case lost a number that was never meaningful.

**It is also faster, not merely safer.** Skipping `count_groups` for every fatal action means the
measured cost of `get_observation()` **dropped to 28.7 µs**, against the 39.1 µs the safe-to-chase
addition itself had already been shown to add for free — a fatal move now costs nothing at all
rather than a full flood fill whose answer got discarded.

15 of 44 tests catch the exact regression of removing this gate, confirmed by mutation test: a
version restored to "compute regardless of legality" fails a third of the suite. One persisted
test covers walls specifically, built on a full-size board rather than a hand fixture, since the
point is the real board's edges. The body-collision sweep — 14,642 moves, re-measured after the
fix at zero non-zero flags, against 5,289 before it — was ad-hoc verification rather than a stored
test, since the fixtures already updated for the eat/no-eat conditional cover that case by
construction (every action they name as fatal now asserts `0`, not a hypothetical count).

#### `lg(num_groups)` was on the wrong scale, and the rest of the vector was audited too

The user asked for two things after this: fix the region-count scale, and check whether anything
else in the 23-value vector shares the problem.

`lg(num_groups)` (indices 10, 12, 14) had no normalization at all — it was `log2plus1(group_count)`
with nothing dividing it down, sitting on a vector where every other input is `[0, 1]`. Measured
directly: **1.0** where every other observation's ceiling is **1.0**, against `lg(num_groups)`
reaching **~4.4** — the same scale mismatch the starve observation had before it was split earlier
the same day.

The fix mirrors the starve pattern: `MAX_GROUPS_FOR_SCALE = 16` in `snake_constants.py`,
`GROUPS_OBS_SCALE = log2(MAX_GROUPS_FOR_SCALE + 1)` in `state_helpers.py`, and
`num_groups = log2plus1(group_count) / GROUPS_OBS_SCALE`. Unlike the starve budget, this cap is
not a game rule — nothing clamps the true region count, so a sufficiently fragmented board could
push the normalized value slightly past 1.0. The cap itself needed measuring rather than deriving,
since no closed-form formula bounds how many regions a single connected snake body can carve out
of a 10x10 board:

| method | result |
|---|---|
| heuristic play, 180 episodes, 422,608 candidate moves (the same per-action lookahead `group_obs` computes) | never exceeded **13** |
| hand-built adversarial body — a comb of 5 full-height teeth, each notched to split its corridor into 3 pieces, 70 of 100 cells | reached **13** |

Both independent methods topped out at the same number, so 16 was chosen for real headroom without
compressing the range that matters for everyday values.

**The rest of the vector was audited on request, and nothing else needed changing:**

| indices | what | range |
|---|---|---|
| 0-5 | food: is-closer flag, `1/(distance+1)` | `[0, 1]` by construction — reciprocal of a distance ≥ 0 |
| 6-8, 15-17, 18-20 | binary flags (safe, safe-to-chase, wins) | `{0, 1}` exactly |
| 9, 11, 13 | `head_with_tail` | `{0, 1}` exactly |
| 21 | starve budget, already split and scaled this same day | `[0, 1]`, both ends reached exactly |
| 22 | `snake_len / PERFECT_SCORE` | `[0, 1]` by construction — length is bounded by the board |

Confirmed directly rather than only reasoned about: a sweep of 18,450 steps recorded the min and
max of every one of the 23 indices, and the largest value anywhere in the vector is now **exactly
1.0**, the smallest **exactly 0.0** — the region-count values were the only outlier.

47 tests now, 3 new ones dedicated to this: the scale constant's formula matches the starve
pattern's (`GROUPS_OBS_SCALE == log2(MAX_GROUPS_FOR_SCALE + 1)`, not merely close to it), the
scaled value hits exactly 1.0 at the design cap and exceeds it just past it, and every measured
region count from the 422,608-move sweep stays in `[0, 1]` with real margin under the cap. Two
mutations confirmed these catch what they are meant to: lowering the cap below the measured
ceiling, and reverting the scale formula to drop its `+1`.

#### Hugging a wall or body, 2026-08-02: three more values, and the vector is 26

The user's own idea, not one this document proposed: for each action, 1 when the post-move head
has a wall or body segment immediately to its left *or* right, 0 when both sides are open or the
move is fatal. **This is a hypothesis, not a finding** - the intent is to let a policy learn to
travel along a wall or the edge of an existing pocket rather than through the middle of open space
next to one, since cutting through the middle can turn one large pocket into two smaller ones that
are each harder to use later. Nothing here measures whether that actually happens; it is recorded
as a design rationale, to be judged the same way `head_with_food_obs` was - by what a trained arm
does with it, not by intuition.

"Left" and "right" are relative to the heading the move leaves the head facing, found by looking
`CURRENT_DIRECTION_MAPS` up a second time - the same table that answers "what is my new heading"
for the action itself answers "what is 90 degrees off that heading" when asked of the new heading
rather than the old one. Checked against the grid *after* the move (the same one `group_obs`
already builds for the region values), not before it, so a cell the tail is vacating this step
reads as open rather than blocked - the one case that distinguishes the two is narrow (the left or
right cell has to be exactly the tail's current position), but it is exactly the same kind of
staleness the tail-advancing fix corrected in `head_with_tail` two fixes earlier the same day, so
it is treated the same way here on principle rather than only after it was shown to matter.

A fatal move reads 0 regardless of the geometry beside it, matching the group-value convention
established earlier the same day. Computed inside `group_obs` to reuse its legality gate and its
post-move grid rather than duplicating either - it needs neither the flood fill nor the regions
those exist for, so this is a reuse of infrastructure rather than a third thing sharing genuinely
shared work.

Measured over 18,450 steps of heuristic play: the flag is 1 on **37.9%** of action slots (**40.0%**
of slots at length ≥ 50), and the full vector's range is unaffected - every value still spans
exactly `[0, 1]`. Cost is unaffected too: two extra grid lookups per legal action, no flood fill,
measured at 27.2 µs at length 75 against 28.7 µs before this addition existed - within run-to-run
noise, not a measurable increase.

55 tests now, 8 new. Four are hand-built boards proving the geometry (open board hugs nothing;
running along the left edge hugs on the forward move and is fatal on the move that would leave the
board; a placed body segment is hugged whether reached head-on or from an adjacent angle; both
sides blocked still reads a single 1, not 2); one proves the fatal-move override directly, using
the same obstacle that would have read 1 if the move survived; two are a matched pair proving the
post-move-grid decision - the same board reads 0 when the move does not eat (tail vacates, cell
reads open) and 1 when it does (tail stays, cell reads blocked); one sweeps every fixture in the
section checking the fatal-move invariant holds throughout. Four mutations confirmed each design
decision is load-bearing: checking the old grid instead of the new one, using the pre-move heading
instead of the post-move one, requiring both sides blocked instead of either, and checking only one
side - each is caught by a different subset of the fixtures.

Effect on the perfect rate is unmeasured and needs an arm, like every other observation change
today.

#### Rewards are now exact

Before, an eaten food usually paid 0.999 and a death usually cost 5.001. All four terminal
outcomes are now exactly their constants, verified by playing until each occurred:

| outcome | reward | episodes checked |
|---|---|---|
| ate food | `1.0` | 400 games, only value seen |
| died | `-5.0` | 400 games, only value seen |
| starved | `-0.5` | 398 starve episodes |
| perfect game | `100` | 199, with the threshold patched to score 5 to reach the branch |

The shaping penalty now applies only to an ordinary surviving move, so it can no longer
contaminate a terminal reward.

#### What this cost: about 10 points, measured properly

Both batch-8 arms were re-run through the full `top20` close-out on the new environment,
2026-08-02. The selector is deterministic on an arm's graph history, so it picked the **same
checkpoints** as the original close-outs — 52 for `b8f`, 20 for `b8d` — which makes this a
matched comparison at 100 episodes a side, free of the selection confound that makes pooled
rates otherwise incomparable.

| arm | ckpts | old pooled | new pooled | change | per-ckpt median | improved |
|---|---|---|---|---|---|---|
| `b8f-disc9975seed2` | 52 | 66.3% | **54.4%** | **-11.9 pts** | -10.0 | 3 of 52 |
| `b8d-disc995clip` | 20 | 60.4% | **51.2%** | **-9.1 pts** | -7.0 | 1 of 20 |

**Systematic, not noise**: 49 of 52 and 19 of 20 checkpoints got worse, and the two arms lost a
similar amount despite different configs — consistent with a uniform off-distribution tax rather
than anything config-specific. Worst single checkpoint fell 28 points; the best gained 3.

**Best checkpoint, and the ranking moved:**

| arm | old best | new best |
|---|---|---|
| `b8f` | 88.9% @2816000 | **82.0% @3149000** |
| `b8d` | 76.0% @5027000 | **73.0% @2539000** |

The old champion `2816000` reads 73% now, while `3149000` — 84.3% before, third-ranked — is the
best on this environment. So *which* checkpoint is best is environment-dependent, and a
hall-of-fame entry is only the record for the environment that measured it.

**Cross-arm ordering survived**, which is the reassuring part for the investigation: `b8f` still
beats `b8d` on pooled (54.4 vs 51.2) and on best checkpoint (82 vs 73), same as before (66.3 vs
60.4, 88.9 vs 76.0). The gap narrowed but never changed sign. The audit shifted both arms down
by a similar amount rather than scrambling their order.

**This is not evidence the fixes are harmful.** These policies were trained to read two features
whose meaning changed underneath them, so a drop is the expected cost of being off-distribution.
Whether the corrected observations *train* better is a different question, untested, and what
batch 9 will start to answer. An earlier 30-episode spot check put the champion at 21/30 and this
supersedes it — 100 episodes on 72 checkpoints rather than 30 on one.

## Measured: `b6b-alpha06` is the best *bet*, `b4c` still has the best *ceiling*

Every batch 5/6 arm was measured with the `top10` rule of the time — its ten best surviving
checkpoints at 100 greedy episodes each, 1000 episodes per arm. Pooled:

| arm | eff exp | pooled perfect (1000 eps) | 95% CI | best ckpt | worst ckpt | graph said |
|---|---|---|---|---|---|---|
| `b6b-alpha06` | ~1.2 | **24.5%** (245/1000) | 21.9-27.3% | 36% @1455k | 12% | 21.7% |
| `b6a-alpha04` | ~0.8 | 8.1% (81/1000) | 6.6-10.0% | 13% @514k | 2% | 14.3% |
| `b5d-schlongTDE` | ~0.8 | 6.6% (66/1000) | 5.2-8.3% | 12% @1160k | 2% | 10.7% |
| `b5c-schlongIS` | ~1.6 corr | 2.1% (21/1000) | 1.4-3.2% | 6% @2239k | 0% | 17.0% |

`b6b` is **3x the next best arm with non-overlapping intervals** — the first
non-marginal separation since `b4c`.

### But it did not beat `b4c`, and the honest comparison matters

`b4c`'s earlier measurement used **4** checkpoints, so comparing its pooled number to
`b6b`'s 10-checkpoint pool is unfair to `b6b` — a deeper pool reaches into weaker
checkpoints. Taking each arm's **top 4 by the same a priori criterion** (smoothed graph
rate, before any 100-episode measurement):

| arm | top-4 pooled | 95% CI | best single checkpoint |
|---|---|---|---|
| `b4c-schlongper` | **31.8%** (127/400) | 27.4-36.5% | **51%** @869k |
| `b6b-alpha06` | 21.8% (87/400) | 18.0-26.1% | 36% @1455k |

So `b4c`'s config really does produce better policies **when it survives**. `b6b` did not
beat it. What `b6b` beats it on is *reliability*, and once the death rate is priced in the
ranking flips:

| config | eff exp | ceiling when it survives | survival | expected value |
|---|---|---|---|---|
| `b4c` (alpha 0.8 `td_loss` no IS) | ~1.6 | 31.8% | 1 of 3 | **~10.6%** |
| `b6b` (alpha 0.6 `td_loss` no IS) | ~1.2 | 24.5% | 1 of 1 | **~24.5%** |

**`b6b`'s config is the better bet by roughly 2x, despite the lower ceiling.** Its survival
is n=1, so that number is soft and the next thing to do is seed it 2-3 more times. But this
is the first config here that looks both good *and* repeatable, and repeatability has been
the binding constraint all along.

## Retracted: the `theSchlong` PER config is not reliably better — it is a coin flip

`b4c-schlongper` (alpha 0.8, `td_loss`, no IS) produced the best result on record and
this document previously called that config established. **Batch 5 ran it twice more and
both repeats died permanently.**

| arm | config | outcome | died at | best 30-eval perfect |
|---|---|---|---|---|
| `b4c-schlongper` | alpha 0.8, `td_loss`, no IS | **survived**, 51% measured | — (dipped to 10.1 @ 203k) | 34.0% |
| `b5a-schlong` | identical | **dead** | 272k | 10.0% |
| `b5b-schlong2` | identical | **dead** | 246k | 7.7% |

All three hit a crisis in the same 200-270k window. `b4c` bottomed at trailing 10.1 and
climbed back to 62+ by 320k; the other two flatlined at 0.0 and stayed there for 1.6M
and 1.7M further steps. So the config does not produce a better policy — it produces a
**~1-in-3 lottery ticket**, and the one time it paid out it paid out very well.

This is the third premature conclusion in this investigation, and the pattern is now
unmistakable: **every result here that rested on a single seed has failed to replicate.**

## Reverting either factor alone survives the crisis

Both single-factor variants are alive past the step where both exact repeats died:

| arm | reverted | final step | worst dip in 180-300k | best 30-eval pf | **measured** |
|---|---|---|---|---|---|
| `b5c-schlongIS` | IS weights back on | 2.31M | 62.1 — barely noticed it | 17.0% @211k | **2.1%** |
| `b5d-schlongTDE` | `abs(td_error)` signal | 2.07M | 22.8 — dipped and recovered | 10.7% @410k | **6.6%** |

Both survived the window that killed the exact repeats, so reverting either factor does
buy stability. **But neither policy is any good**, and `b5c` — the arm that sailed through
the crisis most smoothly — measured **worst of all four arms at 2.1%**.

So the earlier reading here, that IS weights "look like the strongest stabilizer" and that
adding them during the port was not a mistake, needs splitting in two:

- **About stability, it holds.** `b5c` never dropped below 62 in the crisis window.
- **About quality, it is wrong.** IS correction cancels the prioritization it is correcting,
  and with it most of the benefit. 2.1% is barely above the ~1% committed baseline.

Caveat on `b5c`'s number specifically: it ran 2M steps past its peak, so its 17.0% peak
checkpoint had already been evicted and only weak survivors remained to measure. Its true
ceiling is somewhere above 2.1% and unrecoverable. That is a measurement failure caused by
letting the arm run, not purely a property of the config — see
[`runs.md`](../runs.md) on checkpoint retention.

---

## The headline: 51% perfect games, measured

The four best `b4c-schlongper` checkpoints were reloaded and evaluated over **100
greedy episodes each** with [`eval_checkpoints.py`](../../eval_checkpoints.py). Full
results in `runs/b4c-schlongper_checkpoint_evals.json`.

| checkpoint | perfect % over 100 eps | 95% CI | mean score | median score |
|---|---|---|---|---|
| **869000** | **51.0%** | **41.3-60.6%** | 83.4 | **95.0** |
| 942000 | 29.0% | 21.0-38.5% | 74.3 | 84.0 |
| 775000 | 25.0% | 17.5-34.3% | 75.4 | 76.5 |
| 162000 | 22.0% | 15.0-31.1% | 51.9 | 67.5 |

**Checkpoint 869000 wins more than half the games it plays.** Its median score is 95 of
a possible 95 — over half its episodes are perfect wins — and the mean of 83.4 is
dragged down by a minority of early deaths (min 13).

These numbers were produced by an earlier version of `eval_checkpoints.py` that stopped
at the Nth finished episode, which **truncates in-flight episodes and so biases the rate
downward** (perfect games are the longest episodes). The harness now runs whole rounds
instead. The direction of the bias means 51% is a floor rather than an overstatement, but
these four are worth re-measuring when the machine is free.

This **vindicates the premise the investigation started from.** The brief said the
config should reach ~50% perfect games around 1M iterations. The committed config
reaches ~1%; with `theSchlong`'s PER restored, 51% arrives at 869k steps. The premise
was right, and three PER changes were the regression.

The methodological lesson from this measurement — that selecting checkpoints off the
graph overestimates, by up to 24 points here — is in
[`hyperparamTuning.md`](../hyperparamTuning.md), since it is about how to measure rather
than about the policy.

## What `theSchlong` did differently: three PER changes, none validated

`theSchlong` (the 2022 code in the repo root, read-only) once reached a **76%
perfect-game rate**. Diffing it against `snek2` was the most productive single action in
this investigation — a working reference beats a guess.

**What the 76% was**, since it was got wrong twice: per the human, **a single checkpoint
that got very lucky**, not a mean over a run and not a sustained level. So it was never
a known-good target to reproduce; it was evidence that the config had a high ceiling and
a wide spread. That prediction is exactly what `b4c-schlongper` then showed.

**Rewards, grid size, network shape and the epsilon ladder are byte-identical**, so
scores compare directly across the two. `Snake.py` and `snake_environment.py` differ only
cosmetically (a policy-name overlay, a moved print, `set_display`/`get_score` helpers).
What changed is the replay buffer, and it changed **three ways at once**:

| aspect | `theSchlong` | `snek2` default | why it was changed |
|---|---|---|---|
| alpha | **0.8** | 0.6 | "0.6 against raw TD error is the usual choice" |
| priority signal | **`td_loss`** (element-wise Huber) | `abs(td_error)` | Huber shrinks small errors, widening spread → effective exponent ~1.6 |
| IS weights | **none at all** | mean-normalized, beta 0.4→1.0 | prioritizing without IS correction is biased |

Every one is defensible in isolation, and **every one was validated only at 30k steps** —
far below the ~250k line where anything becomes judgeable. The measurement that justified
them is therefore worthless for this question.

There is also a mechanical interaction: **IS weights partially cancel prioritization**,
because high-priority samples get downweighted gradients. So `alpha=0.6 + IS` is a much
gentler intervention than `alpha=0.8 + no IS`. All three changes push the same direction —
weaker effective prioritization.

`SNEK_PRIORITY_SIGNAL` and `SNEK_IS_WEIGHTS` now make the old behaviour reachable. Both
default to current behaviour.

## Restoring that PER config roughly triples the sustained perfect rate

`b4c-schlongper` reverted all three changes together and beat every other arm on every
measure:

| measure | `b4c-schlongper` | best of everything else |
|---|---|---|
| best 30-eval window | **34.0%** (851-880k) | 16.7% (`b1a-base`) |
| best single eval | **80%** | 40% |
| cumulative over run | **11.06%** | 5.89% (`b1a-base`) |
| peak avg score | **92.0** of 95 | 87.5 (`b1a-base`) |
| evals at >=50% perfect | **41** | 0 |

It held 50-60% repeatedly across 700k-1000k, where no other arm has produced a single
eval above 40%. Its block trajectory shows both the win and the cost:

| steps | score mean | perfect mean |
|---|---|---|
| 100-150k | 74.2 | 12.8 |
| 150-300k | 40.3 → 19.3 | **severe collapse**, score to ~19 |
| 350-400k | 70.4 | 11.0 |
| 700-750k | 75.5 | 13.2 |
| 850-900k | **79.0** | **32.2** |
| 950-1000k | 74.2 | 23.2 |
| 1000-1050k | 65.7 | 11.2 |

**Higher-performing and higher-variance.** It survived a near-total collapse around 250k,
recovered, then climbed for 600k steps to a level nothing else approached.

Three caveats that matter:

- **n=1.** This domain has produced 62.5-vs-18.0 from one config. Repeating this 2-3x is
  batch 5's first priority.
- **It would have been killed at 300k.** Its collapse was deep enough that the
  investigation's own stop criteria would have ended the run before its best 600k steps.
  That is a standing argument for longer horizons.
- **Three changes were reverted together**, so which one carries the gain is unknown. The
  knobs to separate them exist; that is batch 5 priorities 2 and 3.

## Uniform sampling helps a little; the relationship is not monotonic

Removing prioritization entirely (`PRIORITY_EXPONENT=0.0`) beat the committed
`alpha=0.6 + IS` default but landed at about a third of `b4c`'s rate:

| config | best perfect-30 |
|---|---|
| alpha 0.8 + `td_loss` + no IS (`b4c`) | **34.0%** |
| alpha 0 + 500k buffer (`b4b`) | 9.3% |
| alpha 0 (`b4a`) | 8.7% |
| alpha 0.6 + `abs(td_error)` + IS (default, `b1a`/`b2a`) | 16.7% / 7.0% |

So "how much prioritization" is not a dial that improves in one direction. Maximum
aggressive prioritization is best, none is middling, and the committed halfway setting is
not reliably better than none. The most likely explanation is the IS interaction above:
the default's IS weights cancel much of what its alpha asks for, giving the worst of both.

Prior evidence, now in context: PER at alpha=0.6 measured *worse* than uniform over 3
seeds at 30k steps (last-5 avg 46.7 sd 10.6, vs **60.1 sd 4.0** for alpha=0), and
alpha=0.8-with-Huber measured worse still. The 30k horizon makes all of that unreliable,
and the long-horizon result reverses the alpha=0.8 part of it.

## The committed config reaches ~1% at 1M steps, not ~50%

`b2a-base2` ran to 999k on committed defaults — the horizon where ~50% was expected:

| steps | score mean | perfect mean |
|---|---|---|
| 150-200k | 69.6 | **5.2** (its best) |
| 500-550k | 64.9 | 3.8 |
| 750-800k | 66.6 | 3.0 |
| 950-1000k | 64.3 | **1.1** |

Best window over the whole run: 7.0%. This is the measurement that motivated diffing
against `theSchlong`, and it is now explained — the PER changes.

## The perfect-game celebration was throttling the best arms

`Snake.render()` marks a perfect game with a **blocking** `pygame.time.wait()`, and
`PERFECT_GAME_WAIT_MS` defaulted to **5000**. Every training eval runs its first episode on
the *displayed* environment, so any eval whose first episode was a win stalled for 5
seconds — against roughly 5 seconds of actual training per 1000-step eval interval.

The cost scaled with how good the arm was:

| arm quality | share of evals stalling | wasted per eval | penalty |
|---|---|---|---|
| ~40% perfect (`b7f`) | ~40% | ~2.0s | **~40% slower** |
| ~10% perfect | ~10% | ~0.5s | ~10% slower |
| dead (0% perfect) | 0% | 0s | none |

So the mechanism **penalised exactly the arms worth running** and rewarded nothing. Now
`SNEK_PERFECT_WAIT_MS`, default 500ms, recorded in each run's `run_config`.

**This partly explains the step-count gap previously attributed entirely to episode
length.** Dead arms reaching 1.7-2M steps while good arms reached 1.0-1.3M in the same wall
clock was read as "a dead policy ends episodes instantly". That is still the main effect,
but a slice of it was the winner's 5-second pause. **Step-based comparisons are unaffected**
— every arm's eval series is indexed by step, not time — but any wall-clock or steps/second
comparison across arms of different quality was biased, and runs before this fix are not
comparable on wall clock to runs after it.

`eval_checkpoints.py` had the same problem, worse: it stalls the *whole round*, because
`parallel_env.step()` does not return until every worker has stepped, so one visible win
froze all 10 workers. Measured on a 45%-perfect checkpoint, 20 episodes took **92.0s at
5000ms against 79.7s at 400ms** — 13% on a small run, more at full scale. It now defaults to
`EVAL_PERFECT_WAIT_MS=400`.

### The visible eval window looks broken and is not

Two behaviours that look like a crashed eval, both cosmetic — **no eval has ever been
truncated.** All 11 completed eval files hold exactly 10 checkpoints x 100 episodes and
every log reaches its final `wrote` line, with the only exceptions being
`OSError: Bad file descriptor` from multiprocessing connection cleanup *after* the results
are written.

- **The window stops mid-game and vanishes.** A round ends when every worker has finished
  one episode. `ParallelPyEnvironment` steps all envs together, so a worker that finishes
  early keeps being stepped and auto-resets into fresh episodes that are **not counted**.
  The visible worker is therefore usually part-way through a throwaway game when the round
  ends, and the process exits after the last checkpoint, closing the window wherever it had
  got to.
- **It used to freeze for seconds at a time.** The 5000ms blocking wait above, during which
  no `pygame.event.pump()` runs, so macOS marks the window unresponsive.

## Speed pass 2026-08-01: rendering was the bottleneck, not the code

Profiled and fixed in order of measured cost. Everything below is behaviour-preserving —
observations and rewards are byte-identical across 7,976 records of fixed-seed play, and the
9 live tests still pass.

| what | before | after | on the real workload |
|---|---|---|---|
| **Eval window** (worker 0 rendered) | 6050 us/step | 163 us/step | 30-ep eval **70.1s → 14.0s (5.0x)** |
| **Training eval window** (every eval) | 15.6s/episode | 1.3s/episode | ~14s saved per eval at champion skill |
| `get_pos` used `np.add` on int pairs | 8.3M calls, 11.7s | removed entirely | game+obs **1791 → 161 us/step (11.1x)** |
| Flood fill recursed and re-checked `is_open` | 2.0M redundant numpy lookups | iterative, no re-check | included above |
| **Flood fill rewritten as bitwise dilation** | 0.899s in `count_groups` | 0.155s | game+obs **161 → 54 us/step**, 33x overall |
| Eval inference ran eager | 1421 us/call | 208 us/call (6.8x) | ~0 end to end (not the bottleneck) |
| `copy.deepcopy` on the grid | 0.056s | `.copy()` | negligible, 0.3% of profile |

**The lesson is the ordering.** Two optimizations that looked large in isolation — 11.1x on
the observation path and 6.8x on inference — moved the eval wall clock by roughly nothing,
because `ParallelPyEnvironment` steps every worker together and waits for the slowest, and
the slowest was the one drawing a window at 37x the cost of a headless step. Profile the
critical path, not the code that looks hot.

**Training can no longer draw at all**, and `SNEK_DISPLAY_EVAL` is gone with the environment
it controlled — see the section below. `eval_checkpoints.py` keeps `EVAL_RENDER=1`; that and
`watch.py`, which renders in its own process and follows a live arm's newest checkpoint, are
the only ways to see a game.

#### Deleting the display path made every eval episode parallel

Training used to play the first of its ten eval episodes alone, on a second environment in the
main process, because pygame allows one display per process and the workers are separate
processes — so that was the only episode that *could* be drawn. Nothing else used it. Once
watching moved to `watch.py` the constraint was gone, and with it the serial episode, the extra
environment, and both display switches.

| eval shape | per eval, champion skill |
|---|---|
| 1 serial + 9 parallel | 5.95s (1.67s serial + 4.28s round) |
| **10 parallel** | **4.55s** |

**~24% of an eval**, 3 reps each with a tight spread. The tenth worker costs only ~0.27s: a
round ends with its slowest episode, and slowest-of-ten is barely worse than slowest-of-nine.
Statistically unchanged — still ten independent greedy episodes, and the round already counted
only each worker's first.

End to end on four arms it measured 588 → 675 steps/s (+15%) at 32-46k steps, but treat that
loosely: arm ranges overlap and these 4-arm benchmarks vary ~8% run to run, which is why the
isolated per-eval figure is the one to quote. Early training also understates it, because short
episodes make the serial episode cheap.

**This is the third thing that only existed because of the window** — after the eval script's
lockstep visible worker and training's per-eval render cost. Turning rendering off did not just
save its own cost, it invalidated a chunk of surrounding design.

#### Two corrections to the first version of this section

**The mechanism is not the same in both places.** In `eval_checkpoints.py` the visible worker
sits *inside* the `ParallelPyEnvironment`, so it genuinely paced the other nine. Training is
different: `compute_avg_return()` runs the displayed episode alone on a single env and *then*
calls the parallel batch, so the window never gated a worker — the cost was simply additive.
The original text described training as if it had the lockstep problem. It did not.

**`SNEK_DISPLAY_EVAL` did nothing when first added, and the "10k steps 83s → 56s" figure
was noise.** `compute_avg_return()` called `set_display(True)` unconditionally at the start of
every eval, overriding the flag, so both of those runs rendered. What that timing actually
measured was learning-speed variance: first-eval scores across smoke runs ranged 0.1 to 39.1,
and episode length tracks skill, so two runs of the same code differ by far more than the flag
would. The flag now reads `snake_constants.DISPLAY_EVAL`, which both ends consult.

The honest measurement is per-episode with a fixed policy, which has no variance to hide in:
one champion eval episode costs **15.6s rendered against 1.3s headless** (2835 steps at 5493us
against 2654 steps at 484us). Scale that by evals-per-run rather than trusting an end-to-end
smoke timing.

#### Why a window costs 5.2ms a frame and cannot be optimised away

Profiling `render()` against a real window, with the snake short enough that sprite count is
not a factor:

| component | us per frame |
|---|---|
| `pygame.display.update()` | **5164** |
| `pygame.font.Font(None, n)` built 3x per frame | 366 |
| `all.draw()`, `all.clear()`, `event.pump()`, cached font renders | 2-4 each |
| whole `render()` | 5300 |

It is the flip — a round trip to the macOS window server — and the game flips once per game
step. Dirty-rect updates do not help, because the cost is per flip and not per pixel. The
fonts are now cached, which is worth 7% of a windowed frame and rather more of a headless one,
but the only real levers are flipping less often or flipping in another process. `watch.py`
does the latter and caps its own frame rate.

#### How the connectivity fill got to 33x

`count_groups` now represents the open cells as a single Python int and grows a region by
smearing it one cell in each direction — `region | region<<1 | region>>1 | region<<cols |
region>>cols`, masked back to open cells, until it stops growing. Each round is a handful of
operations on a ~144-bit int no matter how many cells the region holds, against a
cell-at-a-time walk paying interpreter overhead per cell per neighbour.

The wall ring `_rebuild_grid()` pads the grid with is what makes it safe. Shifting by one
crosses a row boundary, but the first and last columns are always wall, so no open bit ever
sits where it could wrap, and the mask discards it anyway. The same ring keeps the vertical
shifts in range. Regions come back as bitmasks instead of sets of coordinates, so
`get_adjacent_groups` tests membership with one `&` rather than searching a set per region.

Output is byte-identical across 7,976 records of fixed-seed play, and region *ordering*
changed, which nothing reads — callers want the count and whether two cells share a region.

**Where it stands:** no single hotspot is left. `group_obs` is ~46% of observation cost,
down from ~84%, and the rest is spread across `Snake.move`, `_rebuild_grid` and
`get_adjacent_groups`. Further work has little to aim at, and little to win: the whole
game+observation path is now well under 10% of a step, so the bottleneck is TF inference.

## The mechanism: `td_loss` doubles the effective priority exponent

`common.element_wise_huber_loss` uses delta 1.0, so for `|td_error| < 1` — which is
most transitions once a policy is decent — `td_loss = 0.5 * td_error^2`. Priorities are
then raised to alpha. Squaring inside and exponentiating outside **compounds**:

| priority signal | alpha | effective exponent on `\|td_error\|` | arms | outcome |
|---|---|---|---|---|
| `td_loss` | 0.8 | **~1.6** | `b4c`, `b5a`, `b5b` | 1 of 3 survived |
| `td_error` | 0.8 | ~0.8 | `b5d-schlongTDE` | alive, recovered from a 243k dip |
| `td_loss` | 0.8, IS on | ~1.6, corrected | `b5c-schlongIS` | alive, barely dipped |

The three "PER changes" recovered from `theSchlong` were never independent. `td_loss` and
alpha 0.8 multiply into an extreme exponent, and dropping IS weights removes the only
thing correcting the resulting bias. `b4c` was not running aggressive prioritization —
it was running prioritization roughly twice as sharp as any alpha value anyone intended
to test, uncorrected.

That reframes the whole line of investigation: **alpha 0.8 was never the config under
test.** The nominal value and the effective value differ by 2x whenever
`PRIORITY_SIGNAL=td_loss`, which makes every `td_loss` arm incomparable to its alpha
label. Treat the effective-exponent column above as the real independent variable.

### Sharpness is a variance dial, not a quality dial

> **Weakened by batch 7.** Seeding eff ~1.2 four times gave **2 deaths of 4** (`b7b` at
> 1162k, `b7c` at 573k), against eff ~1.6's 2 of 3. At these sample sizes 50% and 33% are
> not meaningfully different, so the claim that lower sharpness is *safer* no longer has
> support. What survives is the narrower claim that eff ~1.6 has the higher **ceiling**.
>
> Both eff ~1.2 deaths arrived late — 573k and 1162k — well past where the eff ~1.6 arms
> died (246k, 272k). So lower sharpness may **delay** death rather than prevent it, which
> would make measured "survival" partly an artefact of how long an arm is run. Any future
> survival rate quoted here needs a fixed step horizon attached.

The prediction made before batch 6 ran was that alpha 0.4 with `td_loss` (~0.8 effective)
survives and alpha 0.6 (~1.2) is marginal. The first half held. **The second half was
wrong, and the correction is more interesting than the prediction.**

Ranked by best 30-eval perfect rate, with the near-death-and-recovered count:

| arm | eff exp | best 30-eval pf | peak trailing | near-death recoveries | fate |
|---|---|---|---|---|---|
| `b4c-schlongper` | ~1.6 | **34.0%** @880k | 86.4 | 2 | survived, 51% measured |
| `b6b-alpha06` | ~1.2 | **21.7%** @1467k | 81.3 | 3 | running, still oscillating |
| `b5c-schlongIS` | ~1.6 corr | 17.0% @211k | 80.7 | 1 | running, declining 2M steps |
| `b6a-alpha04` | ~0.8 | 14.3% @372k | 82.8 | 1 | running, flat and stable |
| `b5d-schlongTDE` | ~0.8 | 10.7% @410k | 76.7 | 1 | running, stable |
| `b5a-schlong` | ~1.6 | 10.0% @84k | 76.5 | 1 | **died** 272k |
| `b5b-schlong2` | ~1.6 | 7.7% @129k | 74.7 | 1 | **died** 246k |

Among arms that survived, the **ceiling rises monotonically with effective exponent**:
~1.6 gives 34.0%, ~1.2 gives 21.7%, ~0.8 gives 14.3% and 10.7%. The **risk of permanent
death rises with it too** — 2 of the 3 arms at ~1.6 died outright. And the top two arms
are precisely the two with the most near-death excursions.

So prioritization sharpness buys **variance**, and variance buys both the high ceiling and
the absorbing failure. It is a risk/return dial:

| eff exponent | behaviour | **measured** perfect rate | death risk |
|---|---|---|---|
| ~0.8 | tame, flat, boring | 6.6-8.1% | none seen |
| ~1.2 | violent oscillation, always recovers | **24.5%** | none seen yet |
| ~1.6 | oscillation that can become absorbing | 31.8% | 2 of 3 |
| ~1.6, IS-corrected | tame despite nominal sharpness | **2.1%** | none seen |

The measured column (1000 episodes per arm, 400 for `b4c`) replaces the graph-derived
figures an earlier version used, and it changes two things. The gap between ~0.8 and ~1.2
is **3x, much larger than the graphs suggested**, and the IS-corrected arm came **last**,
not mid-table — so "IS weights are the strongest stabilizer" is true about *stability* and
false about *quality*. Stability bought by IS correction appears to cost most of the
performance.

**Retracted from an earlier version of this section: the claim that there is a "cliff
between ~0.8 and ~1.2."** There is no cliff. `b6b` at ~1.2 crossed the supposed cliff and
became the second-best arm on record. What actually separates ~1.2 from ~1.6 is whether
the oscillation's low excursions are absorbing, and that looks like a dice roll rather
than a threshold.

### Retracted: `b6b`'s crash was not permanent capability loss

An earlier version of this section said `b6b-alpha06` suffered a "crash with permanent
capability loss" that "never regained a quarter of its 79.3 peak." **Both claims were
wrong.** It has gone to near-zero and fully recovered *twice*:

| block | mean trailing | min | mean perfect |
|---|---|---|---|
| 0-200k | 36.5 | **0.0** | 2.6% |
| 200-400k | 14.5 | 4.4 | 0.0% |
| 400-600k | 24.7 | 8.5 | 1.9% |
| 600-800k | 60.3 | 39.4 | 6.2% |
| 800-1000k | 71.3 | 61.6 | 10.4% |
| 1000-1200k | 66.4 | 47.0 | 3.9% |
| 1200-1400k | 23.7 | **0.9** | 2.8% |
| 1400-1600k | 61.8 | 39.8 | 10.5% |
| 1600-1800k | 61.3 | 32.1 | **13.3%** |

It is a **very long-period oscillator** whose perfect-game trend is *rising* across the
oscillations. Judging it required more than a million steps of patience, and every read
before ~600k would have been wrong.

Two rules come out of this. First, the death criterion is not "reached 0.0" — `b6b` hit
0.0 in its first block and went on to 21.7%. It is **stayed pinned at 0.0 for hundreds of
thousands of steps**, which is what `b5a`/`b5b` did for 1.7M+. Second, an oscillator's
period can exceed 1M steps, so a several-hundred-thousand-step flat stretch is not
evidence of a settled level. This is the fourth premature conclusion in this
investigation and the third to involve reading a trough as an ending.
## An outlier eval is signal, not luck — and smoothing destroys it

**Terminology:** these checkpoints were briefly called "lucky" in this document. That was
wrong and the word is gone. They are **outliers**, and measurement shows they are genuinely
better than their neighbours rather than lucky draws from the same policy.


This is the most useful measurement result in the investigation, and it **falsified the
assumption `eval_checkpoints.py` was originally built on**. The script ranked checkpoints
by perfect rate smoothed over a centred 10-eval window, on the reasoning that a single
10-episode eval reaching 70-80% must be a lucky draw. Measuring both selections against
the truth says otherwise:

| selection rule | pooled measured | 95% CI | episodes |
|---|---|---|---|
| **raw single eval** (outliers) | **41.3%** | 35.9-47.0% | 300 |
| smoothed region rate | 27.1% | 24.0-30.6% | 700 |

Non-overlapping: the checkpoints picked as outliers are **14 points better**,
not worse. Correlation against the 100-episode measurement across the ten checkpoints:

| predictor | correlation with true rate |
|---|---|
| raw single 10-episode eval | **+0.64** |
| smoothed region rate | **-0.40** |

Smoothing is not merely weaker here, it is **anti-predictive**. The binomial says why: if a
policy's true rate were 27%, the chance a 10-episode eval shows 7+ perfect games is
**0.006**. A high single eval is strong evidence about *that checkpoint*. Averaging it with
its neighbours describes the *region* instead — and as the next section shows, the region
is a poor proxy for the checkpoint.

**Consequence: `b6b-alpha06` and `b6a-alpha04` were measured with the smoothed-heavy
selector and are therefore underestimates.** Their 24.5% and 8.1% are not comparable to
anything measured since.

They also **cannot be fixed by re-measuring**. `b6a`'s best graph point in 1415 evals is 50%
and `b6b` has exactly two above 50%, so under the current thresholds `b6a` yields nothing and
`b6b` yields two checkpoints. The alpha comparison needs new seeds, not new measurements.

### Refinement: a floor, and a must-measure line

**Superseded 2026-07-31.** The thresholds described below were **>=80% mandatory with a cap of
10**; they are now **>=90% mandatory with a cap of 20 and a 60% floor**. The reasoning in this
section still explains *why* a two-tier rule exists; only the numbers moved. See "the tiers
moved" at the end.

The ranking above says *which* checkpoints to prefer. It says nothing about how far down the
list to go, and the answer turns out to matter as much:

| single eval | rule | why |
|---|---|---|
| **>=80%** | always measure, even past the 10-slot target | 8+ perfect in 10 is the strongest available signal; a slot limit is no reason to drop one |
| **60-70%** | fill remaining slots, best first | the real candidate band for most arms |
| **<=50%** | never measure | 100 episodes buys precision about a checkpoint that was never going to be the arm's best |

Because a graph point is 10 episodes, `perfect_percent` only takes values 0, 10, … 100, so
these thresholds are coarser than they read: `>=80%` is {80, 90, 100} and the fill band is
exactly {60, 70}.

**The distribution is extremely skewed, which is what makes the floor worth having.** Across
all 30 arms run so far, 22 have **never produced a single eval above 50%** in thousands of
evals. Effort concentrates on very few arms:

| arm | evals | points at >=80% | points at 60-70% |
|---|---|---|---|
| `b8f-disc9975seed2` | 1757 | **16** (3 at 90%) | 101 |
| `b8d-disc995clip` | 2065 | 4 | 50 |
| `b7f-disc995seed3` | 1058 | 1 | 34 |
| `b4c-schlongper` | 1097 | 1 | 18 |
| 22 others | — | **0** | 0-3 |

So the same 10-checkpoint budget was previously spending 10 evals on arms with no candidate
at all, and capping `b8f` at 10 when it has 16 checkpoints that each cleared 80%.

#### The tiers moved: >=90% mandatory, cap 20, 60% floor

The >=80% rule was right in shape and wrong in scale, and the arms outgrew it within a day. Once
`b8f` reached 3M steps it presented **109** checkpoints at >=80% — roughly seven hours of
evaluation, on an arm that was still training:

| arm | picked now (>=90%, cap 20) | under >=80%, cap 10 |
|---|---|---|
| `b8f-disc9975seed2` | **32** | 109 |
| `b8d-disc995clip` | **20** | 33 |
| `b7f-disc995seed3` | 20 | 10 |

Two measured results justify the narrowing rather than mere cost. Across 88 checkpoints, **90%
and 80% graph points had indistinguishable mean true rates** (57.9% vs 58.6%), so the wide
mandatory tier was buying volume, not information — while the five **100%** points were the only
group with a distinct and higher floor (64-73%). And within the high band the *surrounding* rate
predicts better than the graph value (+0.48 vs +0.10), so a capped fill tier ordered by region
rate loses little.

**The change is not uniformly cheaper**, which is worth stating plainly: a weak arm now gets
*more* attention (`b7f` 10 → 20) because the cap doubled and the fill band widened to include
80%. The saving is concentrated where the cost actually was.


## Two measurement caveats that change how numbers here should be read

#### Pooled rates only compare when the selection rule matches

`b4c-schlongper` measured three ways:

| selection | pooled | note |
|---|---|---|
| 4 hand-picked | 31.8% /400 | the original measurement |
| 3 outliers + 7 smoothed | 31.4% /1000 | agrees closely |
| 3 clusters of 3 + 1 | **26.2%** /1000 | *lower by construction* |

The cluster run is not a disagreement — 6 of its 10 picks are deliberately the weaker
neighbours, so it measures the spike-vs-neighbour gap rather than the config's level.
**`b4c`'s level is ~31%.** Never compare pooled numbers produced by different selection
rules.

The 2026-07-30 thresholds make this worse, not better, and deliberately so: the checkpoint
count itself now varies per arm (16 for `b8f`, 1 for `b8e`) and the population is truncated at
50%. Pooling over 16 checkpoints and over 1 are not the same statistic. **Use best checkpoint
for cross-arm comparison from here on**, and read pooled only as a within-arm consistency
check — a config whose best and pooled figures are close is producing a strong *region*, which
is the property the project is actually chasing.

#### 100 episodes is a weaker instrument than its interval implies

Checkpoint 869000 — frozen weights, greedy policy — has been measured three separate
times:

| run | rate | 95% CI |
|---|---|---|
| 4-ckpt hand-picked | **51%** | 41.3-60.6 |
| outliers+smoothed | 42% | 32.8-51.8 |
| clusters | **32%** | 23.7-41.7 |
| **pooled** | **41.7%** | **36.2-47.3** |

A 19-point spread on identical weights, roughly 2.8 sigma at the extremes — more than
binomial noise comfortably explains, so either the Wilson interval understates the real
variance or something differs between runs that has not been identified. Either way:

- **The published 51% was the high draw of three.** Use **41.7% over 300 episodes**.
- **The best checkpoint found is 851000, not 869000** (40-44% across two measurements).
- Prefer several hundred episodes, or repeat a measurement, before treating any single
  100-episode figure as settled.

### Update 2026-07-30: 20 repeat measurements say the instrument is fine

The `b4c` @869000 case above now looks like the outlier rather than the rule. Twenty checkpoints
were measured twice on the same day, at 100 episodes each:

| arm | steps measured 2+ times | mean spread | max spread |
|---|---|---|---|
| `b8f-disc9975seed2` | 34 | 4.8 | 11 |
| `b8d-disc995clip` | 17 | 5.0 | 12 |
| **combined** | **51** | **4.8** | **12** |

**Mean spread 4.8 points and 47 of 51 within ±10** — comfortably inside binomial expectation, with
no systematic direction. Independent spot checks agree too: the 92% champion read 25/30 and the 88%
one 85/20 during hall-of-fame restore verification.

So a single 100-episode figure is usable at ±10, and the earlier warning should be read as "one
checkpoint once behaved strangely" rather than a property of the measurement. The practical advice
that survives: prefer pooled figures over many checkpoints for *comparing configs*, because that
is where the interval genuinely shrinks (±1.3 at 6300 episodes).


## Filter, not ranker — but the *surrounding* rate does rank, at n=88

The graph value **stops carrying information once it is high**, while the region rate around it
turns out to carry quite a lot. From 88 checkpoints measured on 2026-07-30 (the largest sample in
the project):

| correlation with measured rate | n=88 | n=26 (earlier same day) |
|---|---|---|
| graph single eval | **+0.10** | -0.09 |
| **surrounding rate** | **+0.48** | -0.03 |

| graph point | n | mean measured | range |
|---|---|---|---|
| **100%** | 5 | **67.2%** | **64-73** |
| 90% | 16 | 57.9% | 22-82 |
| 80% | 67 | 58.6% | 33-88 |

Three separate readings, in order of how much weight they carry:

**The surrounding rate is a genuine ranking signal (+0.48).** This **reverses the -0.40 recorded
above** and the "smoothing is anti-predictive" language that went with it. The reconciliation is
that the two measurements answer different questions: the -0.40 compared *selecting* on smoothed
rate against selecting on raw, across a wide range; the +0.48 asks whether, *among* checkpoints
that already spiked to >=80%, the region rate predicts which is best. It does. The selector's
existing surrounding-rate tiebreak is doing real work rather than breaking ties arbitrarily.

**The graph value itself still does not rank (+0.10).** 90% and 80% points are indistinguishable
(57.9% vs 58.6% mean), and both span ~50 points. So the filter/ranker split holds: use >=80% to
decide *what* to measure, and do not trust the order. **Measuring the whole tier remains correct**
— there is no way to know in advance which 80% point is the 88% and which is the 33%.

**A 100% graph point is a different signal — now n=9 and it held.** A targeted run on 2026-07-31
measured four more, at 80%, 83%, 81% and 73%:

| graph point | n | measured range | mean |
|---|---|---|---|
| **100%** | **9** | **64-83%** | **72.5%** |
| 90% | 16 | 22-82% | 57.9% |
| 80% | 67 | 33-88% | 58.6% |

**Every 100% point measured so far is at least 64%**, across two arms and nine checkpoints, while
90% and 80% points both reach down into the 20s and 30s. That is the only graph value in this
project with a usable floor, and it is 15 points above the next tier on the mean.

**Act on it: measure 100% points first.** They are rare — `b8f` produced 9 in 3.5M steps — so this
costs almost nothing and reliably finds a top-decile checkpoint. It does *not* find the best one:
the 88% champion came from an 80% graph point, and the best of these four was 83%.

None of this overturns the +0.64 finding above, which was measured across a wider spread of graph
values; range restriction attenuates every correlation computed here.
## Batch 9: the discount candidates win different things, and neither is settled

First batch trained on the post-audit environment. 2x `0.9975` against 2x `0.995`, shared base
alpha 0.6 / `td_loss` / no IS, close-outs measured at 3.4-3.6M steps while the arms were still
training.

| arm | discount | best ckpt | top-3 | pooled | outcome |
|---|---|---|---|---|---|
| `b9a-disc9975a` | 0.9975 | 65.0% @1735k | 64.3% | **54.9%** /2000 | survived |
| `b9b-disc9975b` | 0.9975 | — | — | — | **dead**, peaked at 328k |
| `b9c-disc995a` | 0.995 | 52.0% @2603k | 51.3% | 38.0% /2000 | survived |
| `b9d-disc995b` | 0.995 | **70.0%** @2544k | **66.3%** | 42.4% /1700 | survived |

**Expected value favours `0.995`, because it survives.** Mean top-3 across both seeds is **58.8%
for 0.995** against **32.2% for 0.9975** — the dead arm contributes zero, and that gap is larger
than anything the surviving arms differ by. This is the same shape the pre-audit data showed, where
0.995 went 3 of 3 and 0.9975 1 of 2, which is mild evidence the audit did not change the underlying
dynamics.

**Consistency favours `0.9975`.** `b9a` pools 54.9% with 18 of 20 checkpoints above 40%, against
42.4% and 38.0% for the 0.995 seeds. If a run survives, 0.9975 gives a wider good region.

**Seed spread exceeds the effect.** The two 0.995 seeds are 18 points apart on best checkpoint
(70% vs 52%) — more than any difference between the values. At n=2 that is expected, and it is why
this stays open rather than becoming a finding. Seeds 3 and 4 of each value would separate them.

### Nothing trained after the audit beats a pre-audit checkpoint yet

Batch 9's best is **70%**. `b8f`'s `3149000`, trained on the *old* observation space, re-measures
at **82%** on this one, at a comparable horizon (its best sat at 3.15M; batch 9 was measured at
3.4-3.6M).

So the corrected observations have not yet produced a better policy than the buggy ones did. That
is one batch, two configs, and arms that had not finished, so it is not a verdict — but it is the
opposite of what the fixes were meant to buy, and worth stating plainly rather than waiting for a
batch that agrees. The audit's justification was always correctness rather than performance: a
feature that claims the snake is trapped when it is not was wrong regardless of what it scored.

### Two process notes from this batch

**A partial close-out is not a small version of a complete one.** `b9d` at 12 of 17 checkpoints had
a best of 49% and looked like the weakest arm in the batch; its final five checkpoints contained its
top three, including the batch's best at 70%. I reported the interim ranking and it was wrong.

**The best checkpoint and the trailing peak are in different places.** `b9a`'s best is at 1735k
against a trailing peak at 3277k; `b9d`'s best is at 2544k against a trailing peak at 1232k. In
both directions, so trailing average is not a proxy for where the good checkpoints are.


## `DISCOUNT=0.995` matches the best ceiling and removes the death risk

The strongest result in the investigation, and the first one where the mechanism was
predicted before the run rather than reconstructed after it.

At `DISCOUNT=0.99` the effective horizon is ~100 steps, while a perfect game on the 9x9
grid runs several hundred. The `PERFECT_GAME_REWARD` was therefore discounted into
near-irrelevance — the value function could barely see the thing the whole project is
optimising for. Raising the discount to 0.995 was listed as the top untested candidate for
exactly that reason.

Three seeds, all with `PRIORITY_EXPONENT=0.6 PRIORITY_SIGNAL=td_loss IS_WEIGHTS=0`, all
measured with the outlier-top10 rule so every number below is comparable:

| arm | discount | best ckpt | top-3 pooled | all-10 pooled | survived |
|---|---|---|---|---|---|
| `b7f-disc995seed3` | **0.995** | **51%** @860k | **48.0%** (42.4-53.6) | 38.8% | yes |
| `b4c-schlongper` | 0.99 | 50% @869k | 46.7% (41.1-52.3) | 37.1% | **1 of 3 seeds** |
| `b7e-disc995seed2` | **0.995** | 39% @334k | 34.7% (29.5-40.2) | 29.5% | yes |
| `b7d-discount995` | **0.995** | 26% @1330k | 22.7% (18.3-27.7) | 16.4% | yes |
| `b7a-a06seed2` | 0.99 | 19% @1822k | 18.3% (14.4-23.1) | 12.0% | yes |

### The gain is reliability, and the ceiling claim would have been wrong

`b7f` (51%) and `b4c` (50%) are a **dead heat** — overlapping intervals on best checkpoint
*and* on top-3 pooled. So 0.995 did not raise the peak; it reproduced it. An earlier draft
of this section was going to claim a new record by comparing `b7f`'s 38.8% against `b4c`'s
previously published 31.4%. **That comparison was invalid**: the two used different
selection rules, and re-measuring `b4c` under the matched rule moved it to 37.1%. The
entire apparent 7-point gain was the selector, not the config. See the selection-rule
caveat below — it caught a false headline within a day of being written down.

What did change is the death rate:

| config | mean level across seeds | survival | expected value |
|---|---|---|---|
| `DISCOUNT=0.995` | 28.2% | **3 of 3** | **28.2%** |
| `b4c` config, eff ~1.6 | 37.1% | 1 of 3 | 12.4% |
| same config at 0.99 (`b7a`) | 12.0% | 2 of 4 | 6.0% |

**~2.3x the expected value of the best previous config**, purely from reaching the same
ceiling without discarding two runs in three. Every earlier lever traded ceiling against
reliability; this is the first to escape that tradeoff.

Secondary evidence that the peak has headroom: at its best checkpoint `b7f` averages **88.8
of a maximum 95**, against `b4c`'s 84.9. Even its failures end closer to a perfect game.

**Two caveats.** Survival is established only to **~1.1M steps** — `b7e` and `b7f` were
stopped at 1.28M and 1.06M while their 0.99 siblings died at 1162k and 573k, so they have
not outlived the danger window by much. And `avg_reward` is **not comparable** across a
discount change, since the discount rescales the reward; compare perfect rates only.


## The discount has an optimum near 0.995, not a monotone benefit

`DISCOUNT=0.995` was such a clear win over 0.99 that the obvious next move was more of it.
**More is worse.** Both `DISCOUNT=0.999` seeds died, and neither reached a decent level
first:

| discount | eff horizon | peak trailing | outcome |
|---|---|---|---|
| 0.99 | ~100 | 88.8 | 12.0% measured, dies 2 of 4 seeds |
| **0.995** | ~200 | 92.6 | **38.8% measured, 3 of 3 survived** |
| **0.9975** | ~400 | **89.4** | **best-ever 47.7% best-30, but 1 of 2 seeds** |
| 0.999 | ~1000 | 63.1 / 31.8 | **dead 2 of 2** (at 452k and 398k) |

`b8b-disc999seed2` never produced a single perfect game across 1.41M steps. The prediction
recorded before launch was that longer horizons grow bootstrapped targets and 0.999 might
destabilise rather than help — that is what happened, and for once it was unambiguous at
n=2 with nothing to wait out.

The shape makes sense given the mechanism. At 0.99 the ~100-step horizon is shorter than a
perfect game, so the terminal bonus is invisible; at 0.999 the ~1000-step horizon exceeds
episode length, so the value function is bootstrapping over a horizon longer than the task
and the targets stop being well conditioned. 0.995 sits close to actual episode length,
which is the point.

**Practical consequence: stop sweeping the discount upward.** Anything above 0.9975 is
answered. But the optimum's *location* is now genuinely open between 0.995 and 0.9975, which
was not the case when this section was written.

### Update 2026-07-30: 0.9975 holds the record, measured

`b8f-disc9975seed2` leads on every column, and unlike the first version of this section the
headline figures are now **measured** rather than graph windows:

| arm | discount | best measured ckpt | pooled | best 30-eval pf | ckpts at >=80% |
|---|---|---|---|---|---|
| **`b8f-disc9975seed2`** | **0.9975** | **88.0%** | **59.2%** /6300 | **69.3%** | **63** |
| `b8d-disc995clip` | 0.995 + clip | 80.0% | 58.4% /2500 | 50.0% | 25 |
| `b7f-disc995seed3` | 0.995 | 51% | 38.8% /1000 | 44.0% | 1 |
| `b8c-disc9975` | 0.9975 | not measured | — | 14.7% | 0 |

The **63-vs-1 gap in checkpoints above 80%** against `b7f` says `b8f` sustains a strong region
rather than spiking through one, which is the property "consistent perfect rate" actually names.
Its pooled 59.2% over 6300 episodes is the most solid number in the project.

The one thing that keeps 0.9975 from being settled is **survival: it is 1 of 2**. `b8c` ran the
identical config and declined monotonically to a stop. So 0.9975 has the best measured ceiling and
an unproven survival record, on one seed each way.

**Next step is seeds 3 and 4 at 0.9975**, not more 0.995. 0.995 is already at 3 of 3 and more
seeds would only re-confirm it, while 0.9975 could be either the new optimum or a coin flip and
two seeds decide which. Run them past 2.5M steps — that is where both records were found.

## Protocol detail moved from hyperparamTuning.md, 2026-08-04

Superseded selector thresholds, the rejected early-abandonment analysis, and the
incremental-save mechanics. The live protocol doc kept the rules; these are the workings.

#### Why not abandon weak checkpoints early

The intuitive alternative is to start every checkpoint at 100 episodes and cut it once it looks
weak. It was simulated and **it barely helps**. Cutting anything below 12/20 at the 20-episode
mark is statistically safe — of the 157 checkpoints that finished at >=80%, the expected number
wrongly cut is 0.24, or **0.15%** — but it saves only **14%** of the episodes.

The reason is the shape of the selected population, which is a tight blob rather than a few good
runs among junk:

| final rate | share of selected checkpoints |
|---|---|
| below 60% | 9.7% |
| 60-69% | 32.9% |
| 70-79% | 40.7% |
| 80% or better | 16.8% |

Only a tenth of the population finishes below 60%, and a gate lenient enough to keep an 80%
checkpoint keeps nearly everything above 60% as well — at 20 episodes the noise band is ±17pp, so
a safe gate has to sit right where the population already is. Pushing the gate to 14/20 buys
1.44x but starts losing 3.2% of the >=80% set. Screening wins because it economises on the many
*mediocre* checkpoints, which is where the time actually is, rather than on the few bad ones.

Worth knowing when reading any of these numbers: of the 8.8pp spread between checkpoints in a
100-episode measurement, **26% is pure coin-flip noise**, and the winner's curse is real — batch
10's headline 93/100 shrinks to a posterior mean of **87.2%** once you account for it being the
max of ~300 noisy measurements.

#### Why the thresholds moved from >=80%/cap 10 to >=90%/cap 20

The first version measured **everything** at >=80% with no upper bound, which does not scale
on a good arm. Once `b8f-disc9975seed2` reached 3M steps it presented **109** such checkpoints —
about seven hours of evaluation — and it was still training:

| arm | checkpoints picked now | under the old rule |
|---|---|---|
| `b8f-disc9975seed2` | **32** | 109 |
| `b8d-disc995clip` | **20** | 33 |
| `b7f-disc995seed3` | 20 | 10 |

Measurement justified the narrowing: across 88 checkpoints, 90% and 80% graph points had
**indistinguishable** mean true rates (57.9% vs 58.6%), so an unbounded 80% tier was buying
volume rather than information. The 100% points were the only genuinely distinct group.

Note the change is not uniformly cheaper. A **weak** arm now gets *more* attention — `b7f` goes
from 10 to 20 — because the cap doubled and the fill band widened to include 80%. The saving is
concentrated where the cost was: strong arms with hundreds of high checkpoints.

**Most arms have nothing above the floor,** so the selector refuses them outright with a message
naming their best graph point. That is the intended outcome: 22 of the 30 arms run so far never
produced a single eval above 50%. It also means `b6a-alpha04` cannot be re-measured at all and
`b6b-alpha06` yields only 2 checkpoints.

**These rule changes break pooled-rate comparability.** Pooled now averages over a checkpoint
count that varies by arm (32 vs 20 vs 1) and over a population truncated at 60%, so a strong
arm's pooled figure is not computed the same way as a weak one's. Compare **best checkpoint**
across arms, and treat pooled as a within-arm consistency read.

#### Results are saved incrementally

The output JSON is rewritten after **every** checkpoint, via `.partial` + `os.replace` so a
reader never sees a half-written file. An interrupted run therefore keeps everything measured up
to that point, and the payload carries `complete: false` until the final checkpoint lands —
check that field before treating a file as an arm's full measurement.

This matters because these runs are long: 20 checkpoints is over an hour, and a 63-checkpoint
run took four. The earlier version wrote once at the end, so any interruption discarded all of
it — the numbers would survive in the log, but nothing machine-readable.

**Measure the whole >=80% tier; do not trust its order.** 26 high-eval checkpoints measured on
2026-07-30 show the graph value carries no ranking signal once it is high — correlation with the
true rate is **-0.09** across both arms, and it flips sign between them (+0.66 / -0.57). The
three 90% points in that sample measured 39%, 21% and 44%, the worst three of `b8f`'s sixteen.
Range restriction explains part of it (everything here is 70-90%), so this coexists with the
+0.64 below rather than replacing it. The practical consequence is the important part: **there is
no way to tell in advance which >=80% checkpoint is the 63% and which is the 21%**, so measure
all of them rather than truncating the list.

**Outlier evals are not luck — those checkpoints really are better.** This was measured,
not assumed, and it reversed an earlier version of this protocol that ranked by smoothed
rate on the theory that a 70-80% single eval had to be a fluke:

| evidence | result |
|---|---|
| correlation with true 100-episode rate | raw single eval **+0.64**, smoothed **-0.40** |
| pooled measurement by selection rule | raw **41.3%**, smoothed 27.1% (non-overlapping) |
| outlier vs the checkpoints 1000 steps either side | outlier won **3 of 3**, by 9.0 / 11.5 / 27.5 points |
| P(10-episode eval shows 7+ perfect \| true rate 27%) | **0.006** |

So a spike is evidence about *that checkpoint*, and smoothing averages it into a statement
about the *region* — which is a different and less useful thing. Ranking on the surrounding
rate systematically picked worse checkpoints.

**Adjacent steps are allowed through on purpose.** 1000 training steps is enough to move the
perfect rate by tens of points — one measured triple reads 8% / 35% / 7% — so neighbouring
checkpoints are separate policies rather than repeat samples of one.

**Budget ~37 seconds per 100-episode checkpoint** with `EVAL_WORKERS=10` and two arms in
parallel, which is ~50% of a 14-core machine. A 20-checkpoint run is ~12 minutes; a 660-checkpoint
arm is ~7 hours flat, or ~2 hours with `EVAL_SCREEN_EPISODES=20`.

**Do not lower `EVAL_WORKERS` to save CPU — it does the opposite.** Measured seconds per episode
on one checkpoint: **1.03 at 2 workers, 0.33 at 10, 0.30 at 20**. TensorFlow's thread pool costs
about a core whether the batch it is handed has 2 rows or 20, so a small worker count pays full
inference overhead for a fraction of the work; 2 workers is 3x slower *and* worse per unit of CPU.
This document previously said throughput was core-bound past ~10 workers, and the batch-10
close-out was launched at 2 workers on that basis — it ran 2.8x slower than it needed to. If a run
has to be made gentler on the machine, run fewer arms at once, not fewer workers.

**Prefer a worker count that divides `EVAL_EPISODES`.** Episodes round up to a whole round, so 12
workers turn a 100-episode request into 108 and those rows no longer match the rest of the arm.

XLA (`jit_compile=True`) is *worse* here — 0.38 s/episode against 0.32 — and pinning TensorFlow to
one thread makes no reliable difference. Neither is used.

An arm whose mandatory tier exceeds the cap costs proportionally more: `b8f` at 32 is ~20 minutes.

Results land in `runs/<policy>_checkpoint_evals<suffix>.json` with a Wilson 95%
confidence interval. Several copies can run at once on different arms — give each its own
`EVAL_OUT_SUFFIX` or they overwrite each other, then merge.

**An interrupted close-out is resumable.** `EVAL_RESUME=1` with the identical command skips every
checkpoint the output file already holds at full length and measures the rest, keeping the
`top20` selection metadata that an explicit step list would lose. A checkpoint that was only
part-measured is redone rather than topped up, so no run has to pool two summaries of one
checkpoint. This is also the safe way to change `EVAL_WORKERS` mid-close-out: kill it, relaunch
with `EVAL_RESUME=1`, lose only the checkpoint that was in flight.

**Always give an exploratory run its own throwaway `EVAL_OUT_SUFFIX`, even a 30-second one.**
The first write happens at the first round of the first checkpoint, not at the end, and it
overwrites whatever was already at that path unconditionally — a CPU-load probe killed after
seeing an unexpectedly large checkpoint count is enough to do it. This is exactly how a
246-checkpoint close-out (`b10d`'s full breakdown, the day of the batch-10 close-out) got
destroyed: a calibration run reused the real `_top20` suffix and was killed before completing,
but not before its first in-flight write landed. `eval_checkpoints.py` now keeps one rolling
`<path>.previous` backup of the last *complete* result at each path before overwriting it —
recover with a plain file copy if this happens again — but that is the safety net, not a
reason to skip the distinct suffix.

**Checkpoint retention bounds all of this.** A checkpoint is written every 1000 steps, so
`max_to_keep` is a rolling window measured in millions of steps, and an arm run past that
window **deletes the checkpoint behind its best number**.

`max_to_keep` was 1000 — a 1M-step window — which cost real evidence: three of batch 5/6's
four arms outran it, and `b5c-schlongIS`'s 17.0% peak at 211k became unmeasurable once the
arm passed 1.28M steps, leaving a best surviving region worth only 7.0%. **It is now
10000**, a 10M-step window, at ~188 KB per checkpoint (~1.8 GB per policy at full depth).
The legacy `train*/` dirs run 9.7 MB per checkpoint because they predate moving the replay
buffer out of the checkpointer, so they would be ~97 GB at this depth — do not resume those
under the new setting without checking disk.

Two habits still apply. Close an arm out at its horizon: past peak, the marginal training
step is worth less than the checkpoint it evicts. And `top20` filters to surviving
checkpoints, so it degrades gracefully instead of failing on a deleted step.

**Each eval process opens one visible window** (worker 0 renders, the rest are
headless), so four parallel evals give four games to watch. The rendering worker is
slower, which only means its round takes longer — episodes are i.i.d. across workers, so
which worker produced one carries no information.

**Episodes are collected in whole rounds**, one per worker, rather than by stopping at
the Nth finished episode. Stopping mid-flight discards the episodes still running, and
**perfect games are the longest episodes there are**, so truncation drops them
preferentially and biases the measured rate *downward*. `EVAL_EPISODES` is rounded up to
a whole number of rounds for this reason. The first published measurement (51% at
`b4c-schlongper` 869000) used the truncating version, so it is if anything an
underestimate.

**Never quote a graph peak as a policy's perfect-game rate**, but do use it to *choose*
what to measure. A graph point is only 10 episodes, so the number itself is unusable — a
70% point measured 40%, and a 40-50% point measured 12-17%. What the spike reliably tells
you is *which checkpoint is worth 100 episodes*, and there it beats every smoothed
alternative (+0.64 vs -0.40 correlation). Use the graph to select, `eval_checkpoints.py`
to quote.

**Every arm ends with checkpoint evals.** Run `eval_checkpoints.py <arm> top20` and compare
*those* numbers across arms. Comparing graph peaks across arms compounds the error once per
arm, and it demonstrably misranks: `b5c-schlongIS` is 2nd of its batch by graph window and
**last by measurement** (17.0% vs 2.1%).

**Compare pooled rates only when the selection rule matches.** `b4c-schlongper` pools 31.4%
under outlier+smoothed selection and 26.2% under cluster selection — not a contradiction,
because 6 of the 10 cluster picks are deliberately the weaker neighbours. Its level is
~31%. A pooled number is only meaningful alongside the rule that produced it.

**Repeat a measurement before trusting it.** Checkpoint 869000, frozen weights and a greedy
policy, measured 51%, 42% and 32% on three separate 100-episode runs — a 19-point spread,
about 2.8 sigma, wider than binomial noise comfortably explains. Its pooled figure over 300
episodes is 41.7%. Treat a lone 100-episode result as provisional even though its Wilson
interval looks tight.
Choose the candidates by trailing-window rate rather than by single-eval peak, and expect
the measured value to come in below the window that selected it.

Three of the run-comparison metrics above need care:

- **Always use a trailing window for perfect %, never a single eval.** One eval is
  10 episodes, so the metric moves in 10-point jumps. Use the last 30 for a coarse
  read, and compare it against the previous 30 rather than an absolute threshold.
- **Score stops proxying the objective late in a run.** `b1a-base` recovered from a
  collapse to its old score while its perfect rate kept falling to a fifth of its
  peak. Score is right early, when perfect % sits at 0 for tens of thousands of
  steps and cannot separate two configs at all. Past ~250k steps, or after any
  collapse, read perfect % directly and treat score as context only.
- **A large drawdown is not disqualifying.** A config that swings wildly but reaches
  a high perfect-game rate beats a placid one that plateaus low — `b1a-base` versus
  `b2a-base2` is exactly that contrast. `b4c-schlongper` is the extreme case: it
  collapsed to score ~19 and went on to be the best arm in the investigation.