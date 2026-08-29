# Invariants

Nine facts about this game and its measurement that are cheap to state and expensive to rediscover.
Every one is carried from snek2, and every one is written in the cost of learning it there.

**This is not a findings file.** Nothing here is a result about a hyperparameter — those live in
[`findings.md`](findings.md), which starts empty because snek3 inherits none of them. These are
properties of the problem and of the instrumentation.

---

## 1. A perfect game is identified by its score, never by its reward

`is_perfect_score(score)` is the single definition, and every counter goes through it.

**A reward is a sum of terms**, so anything derived from it breaks silently the moment a term is
added. When snek2 shipped potential-based chase-safe shaping, it paid `−c·Φ(s)` at the winning step —
as potential-based shaping must — so a perfect game paid **99.9 instead of 100** and all three
counters, which compared `final_reward == PERFECT_GAME_REWARD`, read **0%**.

Eight arms across two hosts then trained blind for 300k+ steps. And they trained *handicapped*, not
merely mismeasured: see invariant 2. The tell was in their own reports — `max_score` read `95/95`,
and 95 *is* a filled board.

**Identify outcomes from state, not from reward.** Keep one definition and an AST tripwire that fails
if a reward comparison reappears.

## 2. `perfect_percent` is not only a report — it feeds the training

The epsilon refinement schedule is driven by the trailing perfect rate, so breaking the measurement
changes the exploration. In the bug above, epsilon stayed pinned at its 0.0125 ceiling instead of
annealing for the whole 300k steps.

Anything that reads a metric back into the run is a **feedback loop, not a readout**. That is also
why the asynchronous self-eval is a measured follow-up rather than a default — it would put this loop
on a lag.

## 3. `arch.json` travels with the weights, and `obs_era` is the field that matters

A checkpoint restores whenever the observation vector is the same *length*; nothing checks that the
values still mean what they meant.

On 2026-08-02 snek2's vector was briefly back at its original 20 values with two indices repurposed.
**Every hall-of-fame checkpoint restored with no warning and played like a beginner** — the champion
went from 90.3% to scoring 0, 0, 1. The specific trap is an input that was previously *constant*:
`game_over` sat at 0 in every state a policy acts in, so its weights were unconstrained, and the
index it occupied now carried board-fill.

Torch's `load_state_dict(strict=True)` catches a width mismatch. **Only the era marker catches a
change of meaning at constant length.** Bump `OBS_ERA` in one place whenever the vector's meaning
changes, and copy `arch.json` with any checkpoint — into `hallOfFame/`, or rsynced to the desktop —
or it will not load.

## 4. 1 means good or safe throughout the observation, and new blocks append at the end

The order is chronological rather than logical, and that is deliberate: diagnostic scripts index the
vector by hardcoded position, so inserting a block silently repoints every one of them.

Two caveats on the newest blocks. A *fatal* move reads 1 at indices 26-28 — the flag only asks "is
this the tail's cell", so combine it with 6-8, which are the only place legality is stated. And index
29 sits at 1 in **99.95%** of states, so it is nearly a constant and **its weights are not
meaningfully trained** — the same shape as the `game_over` trap in invariant 3.

**Not every block is a per-action triple.** Food-space, starve and board-fill are single values, so
never assume index arithmetic in threes.

## 5. An input that fires in 0.01% of states cannot be trained, however informative it looks

`perfect_game_move`, indices 18-20, is nonzero in **0.000-0.025% of states** — measured over 12,000
greedy states on two arms. It returns `[0, 0, 0]` unless the snake is exactly one food short, so the
"this move wins" flag exists only on the single step before a win. Forcing it to 1 moved `Q` by
**+0.53** on one arm and **−0.94** on another — the wrong sign on the arm that won 92% of its games.

**No policy in snek2 ever learned to win from that input.** The ones that win do it through
board-fill, index 22, which is **rank 1 of 30** by saliency in every arm measured.

Two consequences: never credit an endgame result to indices 18-20, and **do not try to fix an endgame
by adding an input that only fires in the endgame's last step.** Measure occupancy before adding a
block.

## 6. `PERFECT_GAME_REWARD` and `DISCOUNT` cannot be tuned independently

A terminal reward is a potential, not a prize, and it has a threshold. With `k` steps per meal and
`f = γ^k`, one meal of progress is worth `f^(m−1)·[W(1−f) − 1]`, so progress only raises value when

```
W > 1 / (1 − γ^k)
```

At `DISCOUNT=0.9975` and this game's 7-12 steps per meal that is **34-58**. `PERFECT_GAME_REWARD=100`
clears it by 2-3×. snek2's batch 33 cut the win to 10, missed by 3-6×, and the agents **correctly
learned to avoid finishing.**

**So lowering the win requires lowering γ to match.**

## 7. Rendering costs ~5.2 ms a frame, and it is not our drawing code

The game flips once per game step, and the flip is a round trip to the OS window server. Everything
`render()` actually does is 2-4 us.

So **training never draws** — the dummy video driver is selected unconditionally — and `watch.py` and
`record_gif.py` render in their own processes. A process that will not draw must **select the dummy
driver, not merely skip drawing**: `Game.__init__` calls `set_mode()` regardless, and `reset()` blits
and flips, so a real driver with drawing disabled opens a window, paints it white once and never
touches it again. That looks like a broken window rather than an absent one.

Related, and the reason `env/` is the only package allowed to import pygame: **never call bare
`pygame.init()`.** It starts every subsystem including `pygame.mixer`, which opens a real CoreAudio
stream per process — 10 idle workers drove `coreaudiod` to 15% CPU. Init subsystems by name.

## 8. A rate compares across a change in episodes per eval; a threshold crossing does not

Fewer episodes means more noise. More noise **raises** a maximum and **raises** a threshold-crossing
fraction, so **the arm measured with fewer episodes always looks better than it is.**

- **Safe across the boundary:** banded mean perfect rate. A smaller sample of the same true rate has
  the same expectation.
- **Not safe:** `best_perfect30`, `max_single_eval` (maxima over a noisy statistic), and
  `strong_eval_fraction` — which matters most, because it is the primary metric. At a true rate of
  0.50, P(≥80% perfect) is 0.055 on 10 episodes and 0.006 on 20: a **9.3× ratio**. It shrinks to
  1.08× at a true rate of 0.80 and inverts slightly by 0.90, so the bias is huge through the middle
  band where arms spend most of a run and vanishes once an arm is genuinely strong. No point estimate
  resolves it, because an arm's figure is a mixture over its own trajectory.

snek3 runs 100 episodes throughout, so this only bites when comparing against a snek2 number. Correct
it exactly rather than arguing about it: an eval that scored `k` of 20 would, at 10 episodes, have
scored `X ~ Hypergeometric(20, k, 10)` and counts as strong iff `X ≥ 8`.

## 9. This domain is very noisy

The same config has produced **62.5 and 18.0**. Never conclude from a single run; repeat promising
configs 2-3 times. **n=4 cannot resolve an effect below ~10 pp.**

A related selection effect governs record claims. **The maximum over a set of measured checkpoints is
a selected high**: snek2's 99.0%/500 champion re-measured at **97.5% over 1,000** fresh episodes, and
its four best hall-of-fame entries fell a mean **1.4 pp** on re-measurement. Stage B of the eval
protocol ranks candidates; it does not certify one. A record needs a fresh measurement of the single
winner.

---

## And one process rule

**Never delete `runs/` or `hallOfFame/`.** A wrongly kept file costs a few KB; a wrongly deleted one
costs a training run. `hallOfFame/` exists specifically because a `max_to_keep` rotation destroyed
evidence once — snek2's `b5c-schlongIS` 17.0% peak — and snek3 drops the rotation entirely, since a
policy-only checkpoint is ~45 KB.
