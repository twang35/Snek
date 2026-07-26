# Run status, results, and queue

Companion to [`hyperparamTuning.md`](hyperparamTuning.md) (the protocol) and
[`charts.md`](charts.md) (the graphs). This is the file that changes constantly;
the protocol rarely does.

## What is currently running

Update this section whenever runs start or stop — a future session reads it to
know what is in flight and might have been terminated.

| policy | config change | started | status |
|---|---|---|---|
| `train` | human-started, committed defaults | before this work | **stopped by the human ~15:15**, freeing a slot. Do not restart it; it is theirs |
| `b1a-base` | control: committed defaults | 14:15 | running. Score flat at ~69 since ~50k, but perfect-game rate is still climbing (1.7 -> 3.3), so keep going |
| `b1c-nstep3` | `N_STEP_UPDATE=3` | 14:15 | running. Slowest arm but the only one with strong score momentum (+10.3 over last 20 evals at 201k). Cap ~18:15, then label "promising but too slow" if still short of the others |
| `b2a-base2` | baseline repeat #2 | 15:10 | running. Needed to establish baseline spread |
| `b2b-nstep2` | `N_STEP_UPDATE=2` | 15:16 | running. Took the slot the human freed. Launched now rather than later so it shares machine conditions with `b1c-nstep3` (n=3) and the two are directly comparable |
| `b1b-tgt200` | `TARGET_UPDATE_PERIOD=200` | 14:15 | **stopped 15:10 at ~104k.** Score rising only slowly (+1.8), perfect rate flat (1.0 -> 1.0), and its hypothesis was already answered. Resume with `SNEK_TARGET_UPDATE_PERIOD=200` if worth revisiting |

That is 4 trainers, i.e. the budget is full — do not launch more until one stops.
Logs for this batch are in `/Users/tony_wang/.claude/jobs/f3cb1855/tmp/b1{a,b,c}.log`,
which is job-scoped and will not survive; the durable record is
`runs/<policy>_evals.json`, so analyse from there.

If these were killed before ~120k steps, they can be resumed simply by relaunching
with the same policy name and the same `SNEK_*` overrides (the overrides are *not*
persisted in the checkpoint — relaunching without them silently changes the config
mid-run, which would invalidate the arm).

---

## Prior findings carried in from earlier sessions

These are already established; don't re-litigate them without new evidence.

- **Prioritized replay at alpha=0.6 measured *worse* than uniform** over 3 seeds
  at 30k steps: last-5 avg 46.7 (sd 10.6) vs **60.1 (sd 4.0)** for `alpha=0`.
  Uniform was also far more consistent. At alpha=0.8 with Huber-loss priorities it
  was worse still. Plausible reason: the reward is already dense and shaped
  (`FOOD_DISTANCE_REWARD` on every step), and PER's advantage is largest with
  sparse rewards, so over-sampling high-error transitions mostly adds variance.
  **Unresolved:** that was only 30k steps. PER may pay off closer to 1M. Worth one
  long-horizon retest.
- `alpha=0.6` is nevertheless the committed default, chosen deliberately. Treat it
  as the baseline to beat, not as known-good.
- **Importance-sampling weights must stay mean-normalized.** cpprb normalizes by
  the largest weight in the whole buffer, so raw batch weights average 0.087 at
  beta=0.4 and 0.0027 at beta=1.0 — a silent 11x-370x cut to the learning rate
  that worsens as beta anneals. `normalize_is_weights()` fixes this; don't remove
  it.
- **Priorities come from `|td_error|`, not `td_loss`.** Huber is quadratic below
  |e|=1 so it shrinks small errors, widening its spread; feeding it in gave an
  effective exponent near |e|^1.6 instead of |e|^0.6 and measurably hurt learning.
- **`legacy.Adam` is not faster here** despite TF's M1/M2 warning: measured 0.809
  ms/step vs 0.721 ms for the modern optimizer. Ignore that warning.
- **Throughput is ~230-240 steps/s** for a single run on an idle machine. Expect
  substantially less with 4 runs sharing 14 cores; that affects wall-clock only,
  not learning per step.

---

## Completed runs

### Batch 1 — interim, all still running

Compared at **matched step 83000**, because wall-clock progress is not comparable
between arms (see the eval-cost confound below). Objective metrics first.

| policy | key change | trailing-20 perfect % | 1st perfect | last-5 score | curve mean | max drawdown | peak |
|---|---|---|---|---|---|---|---|
| `b1a-base` | none (control) | **1.5** | 44000 | **66.0** | 52.4 | 19.2 | 78.6 |
| `b1b-tgt200` | `TARGET_UPDATE_PERIOD=200` | 0.5 | **33000** | 59.2 | **53.4** | 27.4 | 76.9 |
| `b1c-nstep3` | `N_STEP_UPDATE=3` | 0.0 | never | 26.3 | 16.5 | 18.6 | 38.1 |

Perfect-game rate is still tiny and very coarse at this horizon — a trailing-20
average of 1.5% is three perfect games across 200 episodes. It is not yet a
reliable way to separate these arms, which is exactly why score is the workhorse
metric this early. Do not pick a winner on perfect % at 83k steps.

Charts for every arm live in [`charts.md`](charts.md), kept separate so this file
stays readable.

**Everything below is n=1 and this domain is very noisy. Treat as hypotheses to
test, not conclusions.** The single most useful next action is repeating the
baseline, because nothing here can be interpreted without knowing its spread.

#### The big surprise: perfect games arrive ~20x earlier than expected

The premise of this investigation was that perfect games need ~1M iterations to
reach ~50%. `b1a-base` produced its **first 10% perfect-game eval at ~44k steps**,
and `b1b-tgt200` at **~33k**. Both plateau in avg_score around 65-75 out of 95 by
40k.

That is a large enough discrepancy to be worth chasing before tuning anything
else, since it changes what "good" means. Candidate explanations, untested:

1. The code changed materially since the 1M-iteration experience was formed — the
   replay buffer is now cpprb prioritized rather than `PyUniformReplayBuffer`, and
   the collect policy runs under `tf.function`. Neither *should* change learning
   per step, but PER changes the sample distribution, so it might.
2. Lucky seed. Entirely plausible at n=1 given the documented 62.5-vs-18.0 spread.
3. The human's long-running `train` policy carries 1.3M steps of history under the
   older code, so its trajectory isn't directly comparable to a fresh run.

Worth asking the human whether ~10% perfect at 44k is genuinely faster than they
have seen, since they have the historical context that these logs don't.

#### `TARGET_UPDATE_PERIOD=200`: hypothesis not supported, but interesting anyway

The prediction was smoother curves and smaller drawdowns. It got the **opposite**
on the drawdown metric (27.4 vs 19.1) and a lower last-5. What it did do is learn
**much faster early** — roughly score 55 by 15k steps where the baseline needed
~25k — and reach its first perfect game sooner.

So the frequent-target-update theory of catastrophic forgetting looks wrong, at
least at this horizon, while "less frequent target updates learn faster early"
looks worth pursuing. Try 50 and 500 to see whether there's a trend or this is
noise.

#### `N_STEP_UPDATE=3`: slower per step, but the best trajectory of the batch

At matched steps it looks like the clear loser, and an early read of this batch
called it exactly that. Judged on trajectory instead it is the most interesting arm
in the batch. Score by 40k-step block:

| steps | mean score | max |
|---|---|---|
| 0-40k | 5.1 | 16.2 |
| 40-80k | 26.9 | 38.1 |
| 80-120k | 26.8 | 35.4 |
| 120-160k | 37.4 | 66.3 |
| 160-200k | 49.7 | 70.1 |
| 200-240k | 57.3 | 67.5 |

Near-monotonic, and at 201k it was still gaining +10.3 per 20 evals while
`b1a-base` had gone flat at ~69. It has produced no perfect game yet, which is the
one real mark against it.

So n-step is not "worse", it is **slower but more consistently improving** — which
is arguably closer to what this investigation wants than a config that sprints to a
plateau. Whether it overtakes the baseline is an open question; that is what the
4-hour cap is for. If it plateaus below ~70 it is merely slow; if it keeps climbing
past that it is the most promising lead so far.

Lesson recorded: **do not judge an arm at matched steps alone.** Matched-step
comparison is right for fairness but blind to momentum, and momentum was the
signal that mattered here.

#### Neither arm actually showed catastrophic forgetting

At this horizon both arms oscillate roughly +/-10 around a high plateau rather
than collapsing. That is qualitatively different from the collapses seen in the
earlier short 30k runs, which in hindsight were probably the rising phase plus
seed variance rather than forgetting.

This also exposes a **flaw in the max-drawdown metric**: it cannot tell noisy
oscillation around a good plateau apart from a genuine collapse. `b1b` scores
worse on it purely for oscillating at a high level. A better forgetting metric
would be something like the largest sustained drop — a drawdown that persists over
several consecutive evals rather than one bad eval. Worth implementing before
leaning on drawdown again.

#### Confound found: eval cost scales with policy quality

A better policy eats more food, so its episodes are longer, so its 10-episode eval
takes longer in wall-clock. `b1c-nstep3` reached 161k steps while the other two
were at ~68k in the same elapsed time, purely because it was worse and its evals
were cheap.

Consequences: never compare arms by wall-clock or by "where they got to"; always
compare at matched steps. And a batch of arms will drift apart in step count, so
plan to stop them by step count rather than by time.

---

## Planned queue, and what each is expected to show

Ordered by expected value. Revise freely as results land — this is a plan, not a
commitment.

### Batch 1 (first, 3 slots, target ~120k steps)

1. **`b1a-base`** — committed defaults, as a control under the same machine load
   as its batch mates. Not expected to be good; expected to be the *reference*.
   Needed because throughput and contention vary between batches, so cross-batch
   comparison is weaker than within-batch.
2. **`b1b-tgt200`** — `TARGET_UPDATE_PERIOD=200`. **Highest-prior change.** The
   default of 8 gradient steps between target-network syncs is extremely frequent;
   standard DQN uses hundreds to thousands. A target that chases the online
   network is a classic cause of oscillation and forgetting, which is exactly the
   reported symptom. Expect: visibly smoother curve and smaller drawdowns, even if
   peak score is similar.
3. **`b1c-nstep3`** — `N_STEP_UPDATE=3`. The food reward is immediate but the
   perfect-game bonus is terminal and extremely sparse, so credit has to crawl
   back one step per gradient update. n-step returns propagate it ~3x faster.
   Expect: earlier onset of learning (relevant to the "nothing until 40k" effect)
   and better late-game behaviour.

### Batch 2 — revised after batch 1's interim results

4. **`b2a-base2` / `b2b-base3`: repeat the baseline 2-3x.** Now the top priority,
   not an afterthought. Batch 1 says the baseline reaches ~70 avg score and 10%
   perfect games by ~44k, which contradicts the premise of the whole exercise.
   Until the baseline's spread is known, no arm can be judged against it. These
   also test whether the early perfect games reproduce.
5. **`N_STEP_UPDATE=2`** — running now as `b2b-nstep2`. n=3 is the batch's best
   trajectory but too slow to reach a perfect game; n=2 tests whether the stability
   survives at a faster rate. If n=2 lands between n=1 and n=3 on both speed and
   steadiness, that is a clean monotonic trend and the knob is real rather than
   noise. Worth trying n=5 afterwards to find where it turns over.
6. **`TARGET_UPDATE_PERIOD=50` and `=500`.** Batch 1 hinted that longer periods
   learn faster early even though they didn't reduce drawdown. Two more points
   establish whether that's a trend or noise.
6. **`LEARNING_RATE=1e-4`** *only after* the target-update fix. Currently 1e-5 is
   very conservative; the in-code comment already suggests 1e-4. With a stable
   target it may train several times faster. On its own with `TARGET_UPDATE_PERIOD=8`
   it would probably make instability worse, so the order matters.
7. **`DISCOUNT=0.995` or `0.999`.** At 0.99 the effective horizon is ~100 steps,
   but a perfect game is several hundred steps long, so the terminal bonus is
   discounted into near-irrelevance. Raising it should make the perfect-game reward
   actually reachable by the value function — plausibly the single most relevant
   change for the *stated* end goal, though also a known source of instability, so
   it pairs naturally with the target-update fix.
8. **Soft target updates**, `TARGET_UPDATE_TAU=0.005` with
   `TARGET_UPDATE_PERIOD=1`. An alternative to (2) that some find smoother still.
9. **`PRIORITY_EXPONENT=0.0` at long horizon.** Confirms or overturns the 30k-step
   finding at the scale that actually matters.
10. **`GRADIENT_CLIPPING=10`.** Cheap insurance against the loss spikes that tend
    to accompany forgetting events.
11. **`FC_LAYERS=128,128`.** Capacity. Lower prior: 20 inputs and 3 actions is a
    small problem, so capacity is unlikely to be the binding constraint before the
    stability issues above are fixed.
12. **Longer epsilon exploration.** The decay ladder in `maybe_update_epsilon()` is
    driven by reward thresholds and steps down once per eval, reaching 0.1 fairly
    early. If forgetting correlates with epsilon getting small, a floor is worth
    testing. Note this schedule is coupled to `eval_interval`, which is a latent
    confound if that interval is ever changed.

### Explicitly not planned

- Reward changes — they'd break comparability of `avg_score` with every recorded
  run.
- Reverting to `PyUniformReplayBuffer` — cpprb is ~2.4x faster with no measured
  learning cost, so cheaper experiments come from keeping it.
