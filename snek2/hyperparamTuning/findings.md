# Findings

What this investigation has actually established, organized by topic rather than by
the order it was discovered. Companion to [`runs.md`](runs.md) (what is next),
[`completedRuns.md`](completedRuns.md) (per-arm outcomes), and
[`failureModes.md`](failureModes.md) (the degradation patterns).

Don't re-litigate anything here without new evidence. Do re-read the falsified
section before proposing an epsilon or buffer experiment.

The observation space has its own write-up:
[`../claudeFeatureRecommendations.md`](../claudeFeatureRecommendations.md) covers what the
champion's 72 losses turn on and which candidate observations measured well or badly, with the
instruments in [`diagnostics/`](diagnostics/). It is a **frozen artifact from 2026-08-02** — read
it as evidence from that date, not as current state.

| finding | status |
|---|---|
| **Observations and rewards changed 2026-08-01 — nothing before that line is comparable** | **breaking**, 6 env bugs fixed |
| The audit cost old policies ~10 points, and cross-arm ordering survived it | **measured**, 72 ckpts matched |
| **Observations changed twice on 2026-08-02** — two more boundaries after the audit's | **breaking**, and the second changes the vector width |
| **Earlier checkpoints load on `master` and play like beginners** — the width matches, the meanings changed; use `e4514a8` | **breaking**, and **silent** |
| **Terminal steps carried a non-zero discount**, so every episode's last transition bootstrapped off the terminal state | **fixed** 2026-08-02 |
| The n-step falsification was measured with returns that leak across episode ends | **re-opened**, not overturned |
| **Fixing `head_with_tail` moved the champion 80.0% → 90.3% with no retraining** | **measured**, 360 eps a side, intervals disjoint |
| Audit fix #6 was incomplete: `group_obs` used one tail position for a tail that has usually moved | **fixed** 2026-08-02 |
| **Terminal steps never carry `discount = 0`**, so death trains toward `−5 + 0.9975·V(terminal)` | mechanism confirmed, effect unmeasured |
| Nothing in the observation vector distinguishes snake lengths 50 to 99 | **measured**, it is a single value |
| On the new env, `0.995` has the better expected value and `0.9975` the steadier single arm | **open**, n=2 each |
| Nothing trained *after* the audit yet beats a pre-audit checkpoint re-measured on the new env | **open**, 1 batch |
| **The record is 92% perfect games** (`b8f-disc9975seed2` @2816k, `DISCOUNT=0.9975`) | **measured**, 92/100 episodes, **old env** |
| **An arm has a lifetime: peak ~2.5-3M steps, dead by ~7M** | **established**, 2 arms followed to the end |
| **The horizon was the binding constraint** — records live past 2.5M, old arms stopped at ~1.06M | **established** |
| A 100% single graph eval is the only graph value with a usable floor | **measured**, 9 of 9 above 64% |
| 100-episode measurement reproduces within binomial noise | **established**, 51 repeats, mean spread 4.8 |
| Checkpoints below score 40 are no longer written, so a dead arm cannot evict good ones | **fixed** 2026-08-01 |
| `DISCOUNT=0.995` matches the then-best ceiling (51%) and survives 3 of 3 seeds | **measured**, ~2.3x expected value |
| Prefer top-3 pooled over best-of-10; the max of 10 noisy measurements is upward-biased | **established** |
| `b6b-alpha06` (alpha 0.6, `td_loss`, no IS) is the best *bet* at 0.99: 24.5% over 1000 eps | **measured**, n=1 seed, and an **underestimate** |
| `b4c` has the best *ceiling*: ~31%, but survives only 1 of 3 | **measured**, 1400 eps |
| A high single 10-episode eval predicts a good checkpoint; smoothing is anti-predictive | **established**, +0.64 vs -0.40 |
| Policy quality changes materially within 1000 training steps | **established**, 3 of 3 clusters, up to 27 points |
| `b4c`'s best checkpoint is 851000 (~40%), not 869000 | **measured** |
| The published "51% at 869000" was the high draw of three measurements; pooled 41.7% | **corrected** |
| Restoring `theSchlong`'s PER roughly triples the perfect rate | **retracted** — did not replicate, 0 of 2 |
| That config is a coin flip: 1 of 3 seeds survives, the rest die permanently | **established**, n=3 |
| `td_loss` + alpha 0.8 + no IS is effectively alpha 1.6 | **established** — it is arithmetic |
| Sharpness is a variance dial: higher ceiling *and* higher death risk | **weakened** — eff 1.2 dies 2 of 4, eff 1.6 dies 2 of 3 |
| No prioritization setting tested so far survives reliably | **established**, 7 seeds across two sharpness levels |
| `DISCOUNT=0.9975` holds the record but is 1 of 2 on survival | **open** — needs seeds 3 and 4 |
| Higher discount is monotonically better | **falsified** — 0.999 died 2 of 2 |
| `GRADIENT_CLIPPING=10` on 0.995 helps | **falsified** — 1 of 3 seeds, no ceiling gain |
| The 5s perfect-game pause slowed good arms ~40% and biased wall-clock comparisons | **fixed**, now 500ms |
| Evals looked truncated but never were: 11 of 11 complete at 10 ckpts x 100 eps | **verified** |
| There is a stability "cliff" between eff 0.8 and 1.2 | **retracted** — `b6b` crossed it and thrived |
| Reverting *either* factor alone survives the crisis | **established**, n=1 each |
| The committed config reaches ~1% at 1M steps | **established** |
| Degradation after 236-312k is systemic across configs | **established**, 5 arms |
| Epsilon reaching 0.0 causes the collapse | **falsified** |
| A larger replay buffer prevents the collapse | **not settled** |
| n-step returns help | **open** — the n=2 and n=3 arms ran with returns that leaked across episode ends |

---

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

Full write-up in [`../claudeFeatureRecommendations.md`](../claudeFeatureRecommendations.md),
including a ranked scoring of every other candidate observation, the terminal-discount defect, and
the absent length signal. Instruments in [`diagnostics/`](diagnostics/). Both are frozen at
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
[`../claudeFeatureRecommendations.md`](../claudeFeatureRecommendations.md) recommends removing
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

## Policy quality changes materially within 1000 training steps

Evaluating each high-single-eval checkpoint **together with the checkpoints immediately
either side of it** — 100 episodes each — settles whether "this checkpoint is good" can be
distinguished from "this part of the run is good". It can:

| cluster | centre | neighbours at +/-1000 | centre advantage |
|---|---|---|---|
| 851000 (`b4c`) | **40.0%** | 28.5% | **+11.5 points** |
| 869000 (`b4c`) | **32.0%** | 23.0% | **+9.0 points** |
| 970000 (`b4c`) | **35.0%** | 7.5% | **+27.5 points** |
| 2806000 (`b8f`) | **80.0%** | 74.0% | **+6.0 points** |

Pooled over the first three, centres measure 35.7% (CI 30.5-41.2) against neighbours' 19.7%
(CI 16.7-23.0) — non-overlapping, and the effect is in the same direction in **4 of 4** clusters.

The `b8f` cluster is the weakest confirmation and the most informative one. Its graph values read
80% / **100%** / 70% and measured 74% / **80%** / 74%, so the centre still won — but by 6 points
with overlapping intervals, on an arm where *every* checkpoint in the region is strong. **The
advantage shrinks as the surrounding region improves**, which is what you would expect if the
spike reflects a genuinely better policy rather than a measurement artefact: there is less room
above a 74% neighbourhood than above a 7.5% one.

The 970000 cluster is the extreme case: **969000 measures 8%, 970000 measures 35%, 971000
measures 7%.** Those are 100-episode measurements, so **1000 training steps can gain or
lose 27 points of perfect-game rate.** Training is far more non-stationary at the
checkpoint level than this investigation previously assumed, and adjacent checkpoints are
not interchangeable samples of one policy.

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

## Checkpoint-to-checkpoint variance is large, and it is not sampling noise

Within `b6b`'s 1455-1464k cluster — 9000 train steps end to end, checkpoints that should
be nearly identical policies — measured rates at 100 episodes each:

| ckpt | 1455k | 1456k | 1461k | 1462k | 1463k | 1464k |
|---|---|---|---|---|---|---|
| perfect % | **36** | 25 | 24 | **16** | 24 | 31 |

**A 20-point spread across 9000 steps.** At 100 episodes each these are real differences,
not sampling error. Consequences:

- **One checkpoint does not characterise a policy region.** Evaluating a single checkpoint
  from this cluster would have yielded anywhere from 16% to 36% depending on the draw.
- **Pool across several checkpoints** for any number that gets compared across arms. This
  is why `top20` deliberately allows adjacent picks: spacing them out hides exactly this.
- The published **51% for `b4c` at 869000 is one checkpoint**, so it is the top of a
  distribution like this one, not the config's level. `b4c`'s pooled 31.8% is the fairer
  figure.

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
[`runs.md`](runs.md) on checkpoint retention.

---

## The headline: 51% perfect games, measured

The four best `b4c-schlongper` checkpoints were reloaded and evaluated over **100
greedy episodes each** with [`eval_checkpoints.py`](../eval_checkpoints.py). Full
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
[`hyperparamTuning.md`](hyperparamTuning.md), since it is about how to measure rather
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

## Falsified: epsilon reaching 0.0 does not cause the collapse

The hypothesis was that the epsilon ladder's last rung (`epsilon.assign(0.0)` once
`avg_reward > 100`) makes the collect policy fully greedy, turning the replay buffer into
a closed loop on the policy's own behaviour. The evidence looked strong: `b1a-base` was
the only arm that reached 0.0 and the only arm that collapsed, with a mechanism and a
timing that fit (0.0 at 92k, collapse at 265k, ~173k apart — about the time to flush a
100k buffer at ~800-step episodes).

Three arms settled it, all past the judgeable horizon:

| arm | epsilon regime | outcome |
|---|---|---|
| `b3b-epsfloor2` | floored at 0.001 from 147k | peaked 305k, declined 71 → 52 and 7.0% → 3.3% |
| `b3a-epsfloor` | floored at 0.001 from 267k | peaked ~300k, declined 74 → 61 and 8.6% → 1.3% |
| `b3c-buf500k` | **fully greedy at 0.0** from 282k | did not break in the predicted window; later died at 750k for unrelated reasons |
| `b4c-schlongper` | fully greedy from 121k | the best arm in the investigation |

The prediction failed in both directions. **The caveat recorded when the hypothesis was
proposed turned out to be the entire signal:** reaching 0.0 requires `avg_reward > 100`,
so only *strong* runs get there — "reached 0.0" and "was good enough to collapse from a
height" are entangled and the correlation cannot separate them.

Two things worth carrying forward:

- **That correlation was as strong as this domain produces** — one arm at 0.0, one arm
  collapsed, same arm, specific mechanism, timing that fit. It was still wrong. With n=1
  arms and a stated confound, a mechanism that "fits the timing" adds no evidence.
- **The test was still worth running.** It cost one knob and three arms that were going to
  run anyway, closed the question, and incidentally produced batch 3's best arm and the
  natural experiment that settled it.

`MIN_EPSILON` stays in the code — it defaults to 0.0 and changes nothing unless set, and
knowing epsilon 0.0 is *safe* is a useful result. Do not add a knob for the last-rung
threshold.

### Related: when each arm's epsilon treatment actually started

`MIN_EPSILON` only changes behaviour at the last rung, and crossing `avg_reward > 100` is
uncommon and late. **One crossing is all it takes**, because the ladder is a one-way
ratchet — a single eval over 100 pins epsilon permanently, and a later score drop never
raises it back.

| policy | first `avg_reward > 100` | epsilon after |
|---|---|---|
| `b4c-schlongper` | 121k | 0.0 |
| `b1a-base` | 92k (18 evals over) | 0.0 |
| `b3b-epsfloor2` | 147k | 0.001 (floored) |
| `b3a-epsfloor` | 267k | 0.001 (floored) |
| `b3c-buf500k` | 282k | 0.0 |
| `b4a-uniform` | 425k | 0.0 |
| `b4b-unifbuf500k` | 290k | 0.0 |
| `b2a-base2` | never (peaked 99.1) | 0.001 |

This is why a floored arm can be indistinguishable from an unfloored one for its first
few hundred thousand steps, and why `b3a-epsfloor` spent 267k steps as an accidental
baseline repeat.

## Not settled: whether a larger replay buffer helps

`REPLAY_BUFFER_MAX_LENGTH=500000` was tested twice with opposite results:

| arm | buffer | sampling | outcome |
|---|---|---|---|
| `b3c-buf500k` | 500k | PER alpha 0.6 | flattest curve in the investigation, then **died completely at 750k** |
| `b4b-unifbuf500k` | 500k | uniform | steadiest arm, healthy at 1.23M, but only 9.3% |

The difference between them is prioritization, not buffer size, which points the same way
as the `b4c` result. A 500k buffer with uniform sampling is stable-but-low; with PER it
died. Neither is evidence that buffer size is the lever, so **the diversity-squeeze
mechanism described in [`completedRuns.md`](completedRuns.md) is still an untested
hypothesis**, not a finding. `REPLAY_BUFFER_MAX_LENGTH=1000000` is in the backlog at low
priority.

## Re-opened 2026-08-02: n-step returns were never cleanly tested

| policy | steps | peak score (at) | best perfect-30 | 1st perfect |
|---|---|---|---|---|
| `b1c-nstep3` | 1.14M | 76.0 (255k) | 1.7% | 206k |
| `b2b-nstep2` | 580k | 74.6 (140k) | 0.7% | 121k |

Both peaked *below* every baseline, both then declined for hundreds of thousands of steps,
and both sat at zero perfect games in their trailing windows. Two arms ordered by n giving
the same shape looked like a trend rather than noise, and this overturned a still earlier read
that n=3 had "the best trajectory of the batch" — true through 200k, false afterwards.

**That conclusion is withdrawn, because the mechanism it tested was broken.** Terminal steps
carried a non-zero discount until 2026-08-02, and `to_n_step_transition` composes

```
r_t + g*d_t*r_{t+1} + g^2*d_t*d_{t+1}*r_{t+2} + ...
```

where those per-step `d` values are the **only** thing that truncates the sum at an episode
boundary. At `d = 0.9975` on a terminal step, an n-step return keeps accumulating past the end of
the episode into whatever sits next to it in the replay buffer. Both arms above were therefore
trained on returns that mix episodes together, which is a fair explanation for peaking below every
1-step baseline.

This is **not** evidence that n-step helps. It is a retraction of the evidence that it does not.
If a slot is ever spare, `n=2` on the fixed environment is a cheap re-test — and unlike the
original pair it would be measuring n-step returns.

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

## The record is **92% perfect games**, and the horizon was the binding constraint

Final close-out measurement 2026-08-01, with the same arms' earlier measurements below for the
trajectory — which is the whole story of this section:

| arm | when | ckpts | best ckpt | top-3 | pooled | 95% CI |
|---|---|---|---|---|---|---|
| `b8f-disc9975seed2` | **close-out, 5.47M** | 52 | **92.0%** @2816k | **86.7%** | **66.3%** /5200 | 65.1-67.6 |
| `b8d-disc995clip` | close-out, 11.64M | 20 | 76.0% @5027k | 72.7% | 60.4% /2000 | 58.2-62.5 |
| `b8f` | mid-run, 2.65M | 63 | 88.0% @2581k | 82.7% | 59.2% /6300 | 57.9-60.4 |
| `b8d` | mid-run, 2.93M | 25 | **80.0%** @2538k | 74.7% | 58.4% /2500 | 56.5-60.3 |
| `b8f` | mid-run, 1.78M | 16 | 63.0% @1618k | 60.3% | 46.5% /1600 | 44.1-48.9 |
| `b8d` | mid-run, 2.08M | 10 | 62.0% @1688k | 58.7% | 48.3% /1000 | 45.2-51.4 |
| `b7f-disc995seed3` | final, 1.06M | 10 | 51% @860k | 48.0% | 38.8% /1000 | — |
| `b4c-schlongper` | final, 1.06M | 10 | 50% @869k | 46.7% | 37.1% /1000 | — |

**Pooled figures are only comparable within one selector.** The close-out rows used the current rule
(all >=90%, fill to 20 from >=60%); the mid-run rows used the earlier >=80% rule. A more selective
set has a higher pooled rate by construction, so `b8f`'s 59.2% → 66.3% is partly the selector. The
**best-checkpoint column is comparable throughout**, and there the record went 51% → 88% → **92%**.

**The pooled column carries the claim.** 59.2% over 6300 episodes has a ±1.3 interval, so this is
not a best-of-N artefact: it is 20 points above the pooled figure that stood the same morning and
non-overlapping with it. `b8f` has 35 of 63 checkpoints at >=60%.

**The two configs stay tied on pooled** (overlapping intervals) with `b8f` ahead on best. The
champion checkpoint is preserved in [`../hallOfFame/`](../hallOfFame/README.md).

### The late-checkpoint hypothesis: confirmed for supply, mixed for quality

The previous version of this section flagged as speculative that "the horizon may have been
truncating the best checkpoints of good arms". Re-measurement supports it, but not uniformly, and
the distinction matters:

| | corr(step, measured) | 1.0-1.8M | 2.2-2.6M | 2.6-3.0M |
|---|---|---|---|---|
| `b8f` | **+0.61** | ~45% | **64.5%** | 63.6% |
| `b8d` | **-0.11** | 59.5% | 60.3% | 54.0% |

**What is solid is the supply of good checkpoints, not per-checkpoint quality.** In thirteen
hours `b8f` went from 16 checkpoints at >=80% to **63**, and `b8d` from 4 to 25. Both arms' best
checkpoints sit at ~2.55M, and every previous record-holder was stopped at 1.06M — before that
region existed.

**Per-checkpoint quality rises with steps for `b8f` (+0.61) and not for `b8d` (-0.11)**, whose
late band is slightly worse. So "train longer" is not a law. Note also that this correlation is
computed only over checkpoints that already cleared the 80% filter, which restricts the range and
understates any true relationship.

The counter-evidence from before still stands: `b7d` ran to 1.60M at 0.995 and peaked at 26%,
`b7a` reached 2.00M with a 19% ceiling. Long runs do not rescue a mediocre arm.

**Practical rule: do not stop a healthy arm at ~1M steps.** Both records came from territory the
old horizon forbade.

### The horizon has an upper bound too: peak ~2.5-3M, dead by ~7M

Followed to the end, both arms traced the same four-phase arc. `b8d` ran to **11.6M steps** — the
longest run in the project by more than 2x — and died:

| phase | steps | `b8f` perfect (per 1M) | `b8d` perfect (per 1M) |
|---|---|---|---|
| climb | 0-2M | 17.2% → 30.1% | 6.8% → 15.4% |
| **peak** | **~2.5-3M** | **40.9%** | **27.4%** |
| decline | 3-6M | 18.6% → 7.4% → 10.1% | 14.6% → 11.9% → 0.3% |
| death | 7M+ | — | **0.0%** for 4.5M steps |

Both arms' best measured checkpoints (2581k, 2538k) and best 30-eval windows (2828k, 2671k) fall in
the peak band. `b8d`'s last perfect game was at 5496k, 6.1M steps before it was still running.

**So the practical horizon is ~3-3.5M steps**, not 1M and not unlimited. The ~8.5M steps `b8d` spent
after its peak produced nothing measurable. That the decline ends in death rather than a plateau
also means a past-peak arm is not merely unproductive — it is on its way to zero.

#### Corollary: a sudden jump in step rate is a symptom of death

`b8d` advanced **7.3M steps in ~24 hours** while `b8f` managed 1.9M on the same machine. Almost all
of that gap is that **a dead policy plays very short episodes** — the snake dies immediately — so it
burns training steps several times faster than a competent one.

Never read step rate as progress. This is the same confound that once made eval cost look like a
config difference, and it now has a second use: an arm that suddenly starts advancing much faster
than its sibling is probably dying, not accelerating.

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

## Falsified: `GRADIENT_CLIPPING=10` does not buy stability

Clipping went in as a cheap independent stability aid on top of `DISCOUNT=0.995`, on the
reasoning that the 10.0 terminal reward produces occasional huge gradients and that clipping
them would prevent the catastrophic drops. After three seeds it is **1 of 3**, against **3 of
3** for plain 0.995:

| arm | peak trailing | best 30-eval pf | best measured | outcome |
|---|---|---|---|---|
| `b8d-disc995clip` | **86.9** | **50.0%** | **80.0%** (58.4% pooled) | peaked ~2.7M, declining at 3.48M |
| `b8e-clipseed2` | 85.9 | 21.3% | 32.0% (1 ckpt) | faded; stopped at 1.16M |
| `b8g-clipseed3` | 77.0 | 30.0% | **none >50%** | dead; stopped at 3.43M |

**It was briefly this file's headline, off `b8d` at 163k steps.** That reading — "the fastest
riser on record", 36.0% best-30 by 163k against `b7f`'s 699k — was wrong twice over. `b8d`'s
own early window was followed by a near-total collapse (0.4% mean perfect across 300-600k) and
everything durable came after 600k, so it was not a head start. And the two seeds that followed
did not reproduce it.

**The "raises the ceiling" escape hatch is now closed too.** `b8d` measured 62.0% best / 48.3%
pooled, which looked like a unique ceiling gain — until `b8f` measured **63.0% / 46.5% without
clipping**, with overlapping intervals. Re-measurement 13 hours later widened the gap the other
way: **`b8f` 88.0% / 59.2% against `b8d` 80.0% / 58.4%**, still tied on pooled but with the
non-clipped arm ahead on ceiling. Clipping shows **no measured benefit and a worse survival
record**. Do not adopt it.

Recording the process error, because it is the recurring one: that ceiling claim was written
while `b8d` was measured and `b8f` was not, off the arm that happened to finish first. A
two-arm comparison graded from one arm is not a comparison. Wait for both.

## An arm recovered from 1.2M steps at zero — and then died anyway

`b8g-clipseed3` sets both records at once, which is why it is worth its own section:

| block | mean trailing | mean perfect |
|---|---|---|
| 0-300k | 52.7 | 8.7% |
| 600-1800k | **1.7 - 14.7** | **0.0%** |
| **2100-2400k** | **63.7** | **4.3%** |
| 2700-3600k | **0.0** | 0.0% |

**The recovery.** 1.2M steps near zero, then back to 63.7 trailing and a 4.3% perfect rate. The
previous record was ~400k steps. Any stop rule that would have killed this arm at 1M steps —
including the one this project used for most of its life — was wrong on this case.

**The death.** It then collapsed and spent its final 900k pinned at 0.0. So the recovery bought
nothing in the end, and an arm that has completed a recovery arc can still be finished.

The rule that survives both halves: **read `zero_since` against the current step, and require
both a long pinned stretch and no recovery in progress.** `b8g` would satisfy that at 2625k
onward and would not at 1M. Two prior errors in this file — calling `b6b` permanently damaged
and calling `b7b` merely oscillating — were the two directions of getting this wrong, and
`b8g` is the case that contains both.

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

## Engineering facts worth not rediscovering

- **Importance-sampling weights must stay mean-normalized.** cpprb normalizes by the
  largest weight in the whole buffer, so raw batch weights average 0.087 at beta=0.4 and
  0.0027 at beta=1.0 — a silent 11x-370x cut to the learning rate that worsens as beta
  anneals. `normalize_is_weights()` fixes this; don't remove it. (Applies only when
  `IS_WEIGHTS=1`.)
- **`legacy.Adam` is not faster here** despite TF's M1/M2 warning: 0.809 ms/step vs 0.721
  ms for the modern optimizer. Ignore the warning.
- **Throughput is ~230-240 steps/s** for one run on an idle machine, and roughly holds up
  with 4 runs sharing 14 cores. That affects wall-clock only, not learning per step.
- **cpprb is ~2.4x faster than `PyUniformReplayBuffer`** with no measured learning cost.
- **The "upgrade to Gymnasium" warning is inert.** It costs a few log lines and the
  upgrade is unavailable; do not propose it.
