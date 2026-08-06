# What is running, and what is next

The live file. Everything here is current state and forward plan — results and
conclusions live elsewhere so this stays short enough to actually keep accurate.

| file | contents |
|---|---|
| `runs.md` (this file) | what is running, what to run next |
| [`completedRuns.md`](completedRuns.md) | every arm that has finished: config, final numbers, verdict |
| [`findings.md`](findings.md) | what is established, what has been falsified |
| [`failureModes.md`](failureModes.md) | the four ways a policy degrades, and how to tell them apart |
| [`hyperparamTuning.md`](hyperparamTuning.md) | the protocol: metrics, how to judge, how to launch |
| [`charts.md`](charts.md) | progress graph per arm |

## The current record: 96%, and it runs on `master`

`b11b-obs30seed2` @855000 scored **96/100** (CI 90.2-98.4), found by batch 11's close-out on
2026-08-04, and preserved in [`../hallOfFame/`](../hallOfFame/README.md). It is the first record in
a while that loads on `master` as it stands, because batch 11 is the only batch trained on the
current 30-value vector.

**Corrected for the winner's curse it is ~94%.** It is the maximum of 204 full-length measurements
in its own arm. Shrinking it against a Beta prior fitted on `b11b`'s 104 *unselected* graph-100%
rows gives 94.0% — a much smaller haircut than the previous record's, because this measurement is
100 episodes rather than 30. Details and the reason the unselected tier is the only clean prior:
[`completedRuns.md`](completedRuns.md#a-new-best-measured-checkpoint-b11b-855000-96100).

**The previous record, `b10d-disc995seed4` @1815000 at 95/100, no longer runs.** The two
observations added 2026-08-03 took the vector to 30 values, so it and every other batch-10
checkpoint fail to restore; **`450e66e` is the last commit with the 26-value vector.** That 95% was
measured mid-run over 30 episodes and shrinks to ~87%, and its own arm's close-out found 93%
@1695000 over a full 100 — treat those two as one policy family measured twice rather than a record
and a near-miss. Full batch results:
[`archive/batches1-10.md`](archive/batches1-10.md).

## Reference: the 2026-08-03 changes, and what they froze

Batch 10's own write-up moved to
[`archive/batches1-10.md`](archive/batches1-10.md) once batch 13 made batch 11 the baseline.
The two subsections below stayed here because they are not about batch 10 — they describe the eval
protocol and the observation vector every current arm runs on.

## Evals got ~10x cheaper on 2026-08-03

Two changes, both measured rather than assumed:

| change | effect |
|---|---|
| `EVAL_WORKERS` 2 → 10 | **2.8x** faster per checkpoint (103s → 37s) at *lower* CPU per episode |
| three-stage close-out (default) | **~2.4x** fewer episodes, with every graph-100% checkpoint fully measured |

The close-out protocol is now: every checkpoint whose training graph point was **100%** gets the
full 100 episodes immediately (uncapped — 47/142/7/146 across batch 10's arms), everything else
selected gets 20, and the best `EVAL_CONFIRM_COUNT` **of those screened** get 80 more.

**The 100% tier is coverage, not a champion shortlist.** Graph-100% checkpoints average 73.0%
against 71.2% for graph-90% ones, but the 90% tier holds the higher maximum (93% vs 89%) and
produced *all four* arms' best checkpoint, being ~4x larger. Finding the best checkpoint is the
confirm stage's job.

**`EVAL_CONFIRM_COUNT` is 100, raised from 30 on 2026-08-03.** At 30 the arm's true best non-100%
checkpoint reached the confirm set only **57%** of the time on b10d's 369 candidates — a coin flip
on the headline number. 100 takes that to **97%** and only moves the cost from 2.71x to 2.20x.

The worker-count finding is a correction: this project had concluded eval throughput was
core-bound past ~10 workers and dropped batch 10's close-out to 2 workers to hit a ~50% CPU
target. That made it 2.8x slower *and* cost more CPU per episode, because TensorFlow's thread
pool costs about a core whether its batch has 2 rows or 20. **To be gentler on the machine, run
fewer arms, not fewer workers.**

**Early abandonment was rejected in 2026-08-03 and shipped on 2026-08-05**, and the reversal is
about *which rule*, not new data. The rejected version cut a checkpoint once its running rate
**looked weak** — a predictive rule, which needs a safety margin to avoid discarding something that
would recover, and safe margins saved only 14% against batch 10's 937 measurements.
`EVAL_MIN_ACHIEVABLE=85` is instead an **arithmetic** rule: stop only when the remaining episodes
cannot carry the checkpoint to 85% even if all of them are perfect. No margin is needed, because
nothing that would reach the bar can be cut.

That the population is a tight blob between 60% and 80% is what *made* it work rather than what
killed it: a gate at 85% sits above the whole blob, so nearly all of it is out of contention early.
Measured on batch 13's first 505 full-length rows, full-length work drops to **70%** — 439 of them
were already arithmetically out before their 100th episode. Details in
[`hyperparamTuning.md`](hyperparamTuning.md#measuring-a-policy-properly-eval_checkpointspy).

A close-out is also resumable now (`EVAL_RESUME=1`), which is what made switching the worker
count mid-run cost only the checkpoint in flight rather than the 333 already measured.

## The environment changed again on 2026-08-03: two new observations

A fourth environment (‡‡‡ in [`completedRuns.md`](completedRuns.md)). The vector went from **26
values to 30**, in two steps.

#### Indices 26-28: following the tail

Three values, one per action, set to **0** when the move lands the head on the cell the tail is
vacating — the tail-chasing move — and 1 otherwise. 1 is good, per the project convention.

One wart of that direction: a **fatal** move also reads 1, since the flag only answers "is this the
tail's cell". Nothing is lost — indices 6-8 mark fatal moves, so the three cases are still
recoverable — but a 1 here on its own does not mean "this move is fine".

**The hypothesis.** Tucking in directly behind its own tail is safe forever and makes no
progress, and it is the same move that closes a pocket down one cell at a time from behind.
Keeping a cell of slack — pushing a bubble of open space along ahead of the tail rather than
closing on it — leaves room to manoeuvre later. Nothing in the vector named that move before:
indices 6-8 call it safe (it is), 9-14 call the tail reachable (it is), and 23-25 fire on it
exactly as they fire on travelling along a wall. Unvalidated — a hypothesis about what the
feature lets a policy express, not a measured effect.

Measured live so it is at least not a dead input, over 24k states of random-but-legal play: the
flag is available on **4.0%** of states, spread across all three actions (380/310/262). The rate
rises with snake length over the range random play reaches (3.9% at length 4-7, 5.7% at 8-11),
which is the direction the hypothesis wants, but random play cannot grow a snake far enough to
say more — that needs a trained policy on this vector, which does not exist yet.

#### Index 29: how cramped the food's space is

One value, not per action — the first observation to describe the food rather than the snake. **0**
when the food is sealed into a single cell, 0.5 when its open region is exactly two cells, 1 for
anything roomier or when there is no food. 1 is safe, per the project convention.

**The hypothesis.** Food sealed into a one- or two-cell pocket cannot be taken by approaching it.
The snake has to wait, or work elsewhere, until its own tail vacates a cell and opens the pocket —
push a bubble of space round to it. Nothing in the vector could express that: `food_observations`
gives direction and distance, which point straight at an unreachable meal, and `safe_to_chase_food`
collapses to 0 without distinguishing "sealed in one cell" from "reachable but the exit is bad".

**Polarity was flipped to the convention deliberately**, after briefly shipping the reverse. The
sign costs nothing in itself — the first layer absorbs it exactly (`w(1-x) + b` = `(-w)x + (b + w)`,
both unconstrained, no weight decay anywhere here) — so the whole vector reading one way is worth
more than the alternative.

The alternative had one advantage, now given up: the common case would sit at **0**. As shipped,
this input reads 1 in **99.95%** of states, which makes it very nearly a constant — it acts much
like a second bias, collecting gradient on almost every sample while carrying information on very
few. That is the shape of the `game_over` trap this project has already been bitten by. The effect
on learning is modest, since a bias absorbs a constant either way, but **do not assume this index's
weights were meaningfully trained** if it is ever repurposed.

**This is a rare-event feature, and how rare depends entirely on length.** Over 48,635 states of
random-but-legal play it left 1 only 14 times — 0.03% — because random play cannot grow a snake past
length ~11. Against a self-avoiding random-walk body as a proxy for a real one:

| snake length | rate below 1 (cramped) |
|---|---|
| 10 | 0.4% |
| 30 | 2.2% |
| 50 | 9.6% |
| 60 | ~26% |

So it is nearly silent early and common in the endgame — which is where an 80%-plus policy spends
its decisive moves, filling an 81-cell board. Treat those figures as a floor: real snakes coil more
tightly than a random walk. Unvalidated either way.

#### Consequences of both

Batch 10's checkpoints no longer load on `master`; `450e66e` is the last commit with a 26-value
vector. The "config vs environment" question `completedRuns.md` raises for batch 10 is now
permanently unanswerable in its original form — nothing isolates whether the 2026-08-02 fixes or
that seed cluster produced the 95%, and the environment has moved on again.

Batch 10's close-outs finished before this mattered, but the hazard is worth keeping in mind: an
eval process works only because it loaded the matching observation code at startup, so `EVAL_RESUME=1`
against a batch-10 arm would now build a 30-value network and fail to restore. **Batch 10's measured
numbers are final and cannot be extended.**

## Batch 11 is closed out — it is the control for batch 12

**Batch 11 stopped 2026-08-04 09:09; its close-outs finished the same day.** Four seeds of batch
10's config on the 30-value vector, and a null result exactly as its own pre-registration predicted.
Full write-up, including two unpredicted differences that are *not* findings:
[`completedRuns.md`](completedRuns.md#batch-11--the-same-config-on-the-30-value-vector-no-significant-difference).

| arm | final step | best-30 perfect | best-30 @3.185M | best ckpt (100 ep) | graph-100% tier | drawdown |
|---|---|---|---|---|---|---|
| `b11b-obs30seed2` | 3.56M | **91.7%** @873k | **91.7%** | **96%** @855k | **81.0%** (n=104) | 18.0 pp |
| `b11a-obs30seed1` | 3.19M | 85.7% @678k | 85.7% | 94% @671k | 79.5% (n=48) | **42.4 pp** |
| `b11d-obs30seed4` | 3.59M | 78.3% @3468k | 76.3% | 88% @3507k | 69.3% (n=40) | 5.6 pp |
| `b11c-obs30seed3` | 3.23M | 73.0% @1718k | 73.0% | 87% @1706k | 69.0% (n=23) | 18.0 pp |

**All three comparisons against batch 10 came out the same way — +4 to +5 pp, none significant:**

| metric | batch 10 | batch 11 | difference | exact p |
|---|---|---|---|---|
| best-30 @3.185M (pre-registered) | 76.2% | 81.7% | +5.4 pp | 0.243 |
| graph-100% tier rate | 70.6% | 74.7% | +4.1 pp | 0.143 |
| best checkpoint | 86.8% | 91.2% | +4.5 pp | 0.157 |

The two new observations are kept under the stated decision rule (keep unless clearly worse). The
agreement across metrics makes a real positive effect more likely than zero, but n=4 cannot separate
+5 pp from +1 pp — which is the whole argument for the next batch being wider.

`b11b` holds both records now: the highest best-30 (91.7%) and the best measured checkpoint (96%).
Both are n=1 of eight arms with heavily overlapping batch means — high-water marks, not findings.

## Batch 12 is abandoned at ~1M of 2.5M — it is the negative result batch 13 is built on

`b12a-eps002seed1`, `b12b-eps002seed2`, `b12c-eps002seed3`, `b12d-eps002seed4`, stopped
2026-08-04. All four cleared the pre-registered abandon condition together, so the batch was
called early rather than run to its horizon. Charts and per-arm readings in
[`charts.md`](charts.md#batch-12--the-epsilon-rewrite-and-the-deadlock-it-found).

Note for anyone stopping an arm: `SIGTERM` and `SIGINT` are both swallowed by the trainer — there
is no signal handler in `training.py` — so it takes `SIGKILL`. Checkpoints and `_evals.json` are
rewritten every 1000 steps, so at most a partial interval is lost, but copy the four
`_evals.json` files aside first if they hold the only record of something.

### ‡ The new schedule deadlocks. All four arms are failing, 4/4, at ~1M steps

Epsilon descends out of bootstrap correctly and then **pins at the refine ceiling 0.05 forever**,
because the refine phase descends on mastery that the level of exploration it is holding prevents
the agent from acquiring.

| arm | step | trailing | pf30 now | eps | pinned at 0.05 for | evals with a perfect game |
|---|---|---|---|---|---|---|
| `b12a` | 1.02M | 55.5 | 0.0% | 0.05 | 686k steps | 41 / 1022 |
| `b12b` | 0.95M | 57.8 | 0.0% | 0.05 | **942k steps** | **0 / 954** |
| `b12c` | 0.92M | 60.4 | 0.0% | 0.05 | 409k steps | 8 / 918 |
| `b12d` | 1.02M | 61.7 | 0.0% | 0.05 | 455k steps | 32 / 1021 |

Against the control at the same step, on the pre-registered primary metric:

| | `strong_eval_fraction` @1.01M | trailing @1.01M | pf30 @1.01M |
|---|---|---|---|
| batch 11 | 25.2 / 30.5 / 0.0 / 8.2% | 84.5-88.5 | 14.0-82.3% |
| batch 12 | **0.0% ×4** | **54.9-61.7** | **0.0% ×4** |

**The numbers above are greedy, and the comparison is clean.** Evals run `agent.policy`, which
TF-Agents builds as `GreedyPolicy`; only `agent.collect_policy` is the `EpsilonGreedyPolicy` that
`epsilon` feeds. Verified two ways — `_setup_policy` in the installed `dqn_agent`, and empirically
with `epsilon=1.0`, where `agent.policy` returns one action on a fixed observation across 60 calls
while `collect_policy` returns all three. So there is **no exploration tax on the metric**: pf30 = 0
is a real property of the greedy policy, and batch 11 and batch 12 are measured the same way.

**The mechanism is a learning deadlock, not a measurement one.** At eps 0.05, 3.3% of *collected*
actions are random, and a random move with a long snake is usually fatal — so the replay buffer
fills with trajectories that die before the endgame and the agent never sees the states a perfect
game is made of. The greedy policy therefore never masters the endgame, greedy pf30 stays 0,
`refine_epsilon(0, top=0.05, floor=0.002)` returns exactly `top`, and the collection distribution
never improves. The loop closes through the *policy*:

```
eps 0.05 → 3.3% random collected actions → buffer lacks endgame states
        → greedy policy cannot finish → pf30 = 0 → refine returns the ceiling → (repeat)
```

Batch 11's crude ladder always escaped because it descended on step count, which no policy can
suppress.

**The descent is also far too shallow to escape even with luck.** 0% → 6.3% pf30 (`b12a`'s
best-ever window) moves epsilon only 0.0500 → 0.0388. Meaningful relief needs 20-40% pf30, which
is exactly what 0.05 makes impossible: `pf30=10%` → 0.0334, `20%` → 0.0224, `40%` → 0.0100.

**Sustained high epsilon degrades a policy that was already working.** `b12a` read greedy trailing
**87.0** and pf30 6.3% at step 214k, then decayed to 55.5 over the next 800k steps at the same
epsilon. All four peak at 81-87 by step 214-479k and then decline. Both numbers are greedy, so this
is the learned policy getting worse, not a measurement artefact.

This clears the pre-registered abandon condition (a >10 pp drop on the primary metric) 4/4 with a
mechanism understood analytically, at 1M of the 2.5M horizon. Running to 2.5M would spend ~5 h
confirming a deadlock that is provable from the code.

The design flaw is general: *any* purely mastery-gated schedule deadlocks if its ceiling sits above
the exploration level at which mastery is achievable.

### The fix: shield the exploration move, not the schedule

Rather than making epsilon decay faster, attack what makes exploration expensive. In a *guided*
episode the epsilon coin's random move is drawn from the moves that do not kill the snake this
step, instead of uniformly from all three. `shielded_policy.py`, wired in as the collect policy.

| decision | value | why |
|---|---|---|
| what is shielded | **the epsilon draw only** | see below — this is the whole design |
| the greedy argmax | **never shielded** | it must eat the -5 and learn |
| `SNEK_GUIDED_FRACTION` | 0.5 | half of refinement-phase episodes |
| when it engages | at the bootstrap handover | nothing to protect while the snake is short |
| `INITIAL_EPSILON` | **unchanged**, 0.4 | the early ladder is the part that works |
| handover | **0.05 → 0.0125** | two rungs added below; see the smoke result |
| guaranteed-descent envelope | **not added** | judge the lower ceiling on its own first |

**Only exploration is shielded, never the greedy action.** Overriding a fatal *greedy* move would
mean `Q(s, a_fatal)` never gets updated toward `DEATH_REWARD` in the states where the network is
wrong, so those values would drift on generalisation alone — and evals run unshielded, so the arm
would walk into walls it was never allowed to learn about. Shielding exploration only removes the
tax while keeping every death the policy earns itself.

**The mask was already in the observation.** Indices 6-8 are "is the move safe (not body or wall)",
per action, and `state_helpers.body_and_wall_collisions` already handles the case a naive check gets
wrong: the cell the tail is vacating is safe to enter. So the shield needs **no environment change
and no new game logic**, just `obs[6:9]`.

**It is one step deep, deliberately.** Snake's hard problem is sealing itself into a region it
cannot escape, and that is untouched — an arm still has to learn it. All this removes is "the coin
flipped and the snake drove into its own body".

**The shield turns off if an arm collapses**, because `guided_fraction_for` is stateless in the same
way `epsilon_for` is: one rule, "shielded iff refining". An arm back in the bootstrap band is
relearning to survive, which is where dying is informative.

**Verified before launch.** 19 tests in `tests/test_shielded_policy.py`, all 9 mutants of the mask
and schedule logic caught; 237 tests total, 0 failures.

### ‡ The shield alone is not enough — this is why the handover moved too

A smoke run at the batch-12 config, `SEED=1` so it pairs with `b12a`, shield on, handover still at
0.05. Mean trailing score per 50k band:

| band | smoke shielded | `b12a` unshielded | `b11a` near-zero eps |
|---|---|---|---|
| 200-250k | 79.6 | 83.8 | 84.9 |
| 300-350k | 83.3 | 77.5 | 89.5 |
| 350-400k | **82.8** | **74.2** | **90.9** |
| pf30 @350k | 0.3% | 0.0% | **19.0%** |
| trailing gained per 100k | 4.7 | negative | 11.1 |

**What the shield fixed:** the decay. `b12a` peaked at 214k and fell 83.8 → 74.2; the shielded arm
was still rising at the same step. That is the failure mode that killed batch 12, and it is gone.

**What it did not fix:** perfect games. 2 perfect-game evals in 355 against `b12a`'s 41 by the same
step, and it plateaus at trailing ~83 where the perfect rate is ~0. The curve is steeply nonlinear —
trailing 83 → ~0% perfect, trailing 91 → 19% — so plateauing 8 points low costs everything.

**Why.** A one-step mask prevents blunders, not *self-trapping*, and in a near-full board almost any
deviation seals a region a few moves later. So the collect policy still never finishes a board, the
buffer still holds no trajectories that eat the last ~10 food, and the greedy policy still cannot
learn them. Perfect games are measured greedy, so this was never exploration killing the eval — it
is the buffer missing completed endgames. **The shield makes exploration survivable without making
the endgame completable.**

Hence the handover drop to 0.0125: 0.83% forced non-greedy per step against 3.3%, close to the
regime batch 11 proved. Keep the shield anyway — it costs nothing and removes the decay.

Three things this write-up got wrong on the way, all worth remembering. The perfect rate was
believed to be measured under epsilon, making the controller read its own noise — it is not, evals
are greedy, so the proposed greedy probe episodes were **rejected as solving nothing**. A `-1e9`
masked logit made the boxed-in fallback look redundant, because `tf.random.categorical` shifts each
row by its maximum and so samples an all-masked row uniformly by accident; `-inf` makes the fallback
load-bearing and testable. And the shield's one-step depth was flagged as an acceptable limitation
when it is in fact the binding one.

## Nothing is training and nothing is evaluating. The epsilon question is closed

Batch 13 (`b13a`-`b13d`) ran 2026-08-05 to 3.39M / 3.70M / 3.67M / 3.51M, stopped healthy after
10.4 h, and is fully measured. Full write-up in
[`completedRuns.md`](completedRuns.md#batch-13--the-epsilon-rewrite-plus-the-exploration-shield-an-exact-null).

| | result |
|---|---|
| 350k abandon condition | **passed 4/4** — `b13b` hit trailing 92.4 / pf30 72.3% by 350k |
| epsilon at stop | 0.0023-0.0050 — the refinement phase reached its intended range |
| best ckpt vs batch 11 | 89.5% vs 91.2%, -1.8 pp, p = 0.875 |
| graph-100% tier vs batch 11 | 74.6% vs 74.7%, **-0.1 pp, p = 1.000** |
| `best_perfect30` vs batch 11 | 82.2% vs 82.2%, **+0.0 pp, p = 1.000** |
| `strong_eval_fraction` vs batch 11 | 19.5% vs 17.5%, +2.0 pp |
| post-peak drawdown | 20.0 vs 21.0 pp, p = 0.875 |
| new hall-of-fame entry | `b13d-shieldseed4-ckpt986000` at **95%**, 2nd best on record |

**The deadlock is gone and the outcome is unchanged.** Three batches on epsilon have produced one
deadlock (12) and one exact null (13). Keep the floor and the anti-ratchet, which were always
justified on mechanism; **do not spend more arms on exploration level.** Recorded in
[`findings.md`](findings.md#now-measured-2026-08-05-exploration-was-tested-in-both-directions-and-neither-helped).

**The shield stays on by default** at `GUIDED_FRACTION=0.5`. It is confounded with the handover
change here and unproven at 0.0125 — but it demonstrably fixed batch 12's decay at 0.05, costs
nothing measurable, and `SNEK_GUIDED_FRACTION=0` reproduces the unshielded behaviour exactly if it
ever needs isolating.

### What batch 13 leaves for the next batch

The epsilon axis is closed, so the next batch should spend its four slots on something else. The
binding constraint has not moved: **n=4 resolves ~10-15 pp on `best_perfect30` and ~7 pp on
`strong_eval_fraction`**, and the per-seed spread in this batch was -9.4 to +12.3 pp on an effect
that is genuinely zero. Candidates are in the backlog below, unchanged — `SNEK_LEARNING_RATE=1e-4`
is still the highest-value untested knob.

One thing worth doing first and cheaply: **batch 13's arms are four fresh seeds on the current
config and environment**, so pooled with batch 11 they give **n=8 for the baseline** rather than a
treatment group. That is the widest baseline this project has had, and it is what any future knob
test should be compared against.

**Honest statement of the bet.** The rewrite's original premise — that batches 10-11 running 96.8%
of steps at epsilon exactly 0 was itself a defect — has one falsified prediction against it and no
evidence for it. The part that survives is the *ratchet* being a real defect. So the null
hypothesis here is that b11's near-zero regime is simply correct and every version of this schedule
is a wash at best. 0.0125 is a bet placed to find that out cheaply.

### Why shorter and wider, and what it actually buys

**Three of batch 11's four arms peaked before 1.8M**, then spent 1.5-2.5M further steps getting
worse, so a 10-hour run spent most of its compute past the point of interest. Capping near 2.5M
costs about a third per arm and buys seeds instead — and seed count is the binding constraint.

Be precise about the gain, because an earlier version of this section was not: **n=12 detects ~5 pp
only on a low-variance metric.** On `best_perfect30` it is 8.7 pp, and 5 pp there would need ~37
arms per group, which is not reachable. On `strong_eval_fraction` n=12 gives 5.9 pp and n=8 gives
7.2 pp. See the table in
[`hyperparamTuning.md`](hyperparamTuning.md#the-primary-metric-strong_eval_fraction-the-share-of-an-arms-evals-at-80).

Open question the cap does not answer: whether arms that peak at ~700k would have gone higher with
more steps, or whether the peak is the ceiling. `b11d` peaked at 3468k and was the only arm still
near its peak when stopped, so a 2.5M cap risks truncating that kind of arm — worth letting one or
two run on past the cap once the wave's comparison data is in hand.

### Later candidates

Deferred pending the above. The gate on all four is the same and it is not a code change any more —
it is that **n=4 cannot see an effect this size**, which batch 11 has now demonstrated three
different ways. Running any of these at n=4 would produce another unreadable batch.

| change | why | gate |
|---|---|---|
| second discount value (`0.9975`, or an interior point like `0.996`) | batch 9 left 0.995 vs 0.9975 unsettled; whether that survives the fourth environment is unknown | needs the wider design |
| `SNEK_LEARNING_RATE=1e-4` | the highest-value untested knob | needs the wider design |
| eff exponent ~1.4 (`td_loss` alpha 0.7) at 0.995 | `b4c` and `b7f` tie on ceiling; sharpness may still add on top of the discount | needs the wider design |
| best config + `REPLAY_BUFFER_MAX_LENGTH=500000` | `b4b` beat `b4a` slightly, so diversity may stack | needs the wider design |
| partial IS correction (beta < 1) | full correction cost `b5c` almost everything (2.1%); partial may keep stability without the cost | needs a new knob |
| anything aimed at the post-peak decline | both batch-8 arms peaked at ~2.5-3M and fell away; nothing tried so far addresses *why* | needs a mechanism first |

**Seed count is the binding constraint, not the number of knobs tried.** Six single-seed
conclusions in this document have been overturned or weakened — most recently gradient
clipping, which looked like batch 8's headline twice before failing at 1 of 3. Nothing goes in
[`findings.md`](findings.md) as established without n=3, which is why batch 10 spent all four
slots on one value rather than splitting them.

## Standing backlog

Untested, ordered by expected value. Rationale for the ones that need it follows the
table.

| change | targets | prior |
|---|---|---|
| `LEARNING_RATE=1e-4` | training speed | high, but order it after a stability fix |
| `TARGET_UPDATE_PERIOD=50` / `500` | early learning speed | medium — 2 points to test a hinted trend |
| `TARGET_UPDATE_TAU=0.005`, period 1 | smoothness (soft target updates) | medium |
| `FC_LAYERS=128,128` | capacity | low |
| ~~epsilon ladder *shape*~~ | exploration schedule | **done 2026-08-04** — rewritten, needs measuring |
| `REPLAY_BUFFER_MAX_LENGTH=1000000` | experience diversity | low — the 500k result was ambiguous |

**`LEARNING_RATE=1e-4` — only after a stability fix.** 1e-5 is very conservative and
the in-code comment already suggests 1e-4. With a stable target it may train several
times faster; on its own with `TARGET_UPDATE_PERIOD=8` it would probably make
instability worse. The order matters.

**`TARGET_UPDATE_PERIOD=50` and `=500`.** Batch 1 hinted that longer periods learn
faster early even though they didn't reduce drawdown. Two more points establish
whether that is a trend or noise. Note `b1b-tgt200` was stopped at 104k, well short of
the ~250k horizon, so that hint is weak evidence.

**Epsilon ladder shape — rewritten 2026-08-04, and it is now the highest-value untested
change on this page.** The old ladder ran **96.8% of batches 10-11's training steps at epsilon
exactly 0.0**, bottoming out at median step 15000 while 7 of 8 arms were still at 0% perfect
games, because its rungs were calibrated to `avg_reward` values a non-winning policy clears.
Replaced with two phases — `avg_reward` bootstrap to `INITIAL_EPSILON`/8, then a geometric
descent driven by the trailing-30 perfect rate — neither a ratchet, floor 0.002, and exactly 0
rejected at startup. Design and the measured diagnosis:
[`hyperparamTuning.md`](hyperparamTuning.md#the-epsilon-schedule--rewritten-2026-08-04-and-it-breaks-curve-comparability).

Two consequences for planning. **Learning curves are no longer comparable to batches 1-11** —
checkpoints still load, but every earlier arm trained greedily from step ~15k, so graph shapes
are not like-for-like. And because the refinement phase is a pure function of current skill, a
declining arm now automatically explores more (`b11a` would have gone 0.0020 → 0.0087 across its
42pp drawdown), which makes this the first change in the backlog aimed at the post-peak decline
rather than at the ceiling.

## Explicitly not planned

- **Reward changes** — they would break comparability of `avg_score` with every run
  recorded so far.
- **Reverting to `PyUniformReplayBuffer`** — cpprb is ~2.4x faster with no measured
  learning cost, so cheaper experiments come from keeping it.
- **An LR schedule** — no evidence of optimization instability; degradation is gradual
  in every arm, not spiky.
- **Adding more epsilon knobs** — the schedule was rewritten 2026-08-04 and already exposes
  `INITIAL_EPSILON` and `MIN_EPSILON`; the thresholds, windows and the 80% target are constants
  in `training.py` on purpose. Measure the new schedule before making any of them tunable, or
  the next batch will vary four things at once.
- **Setting `SNEK_MIN_EPSILON=0`** — rejected at startup. See
  [`findings.md`](findings.md#scope-of-that-falsification-added-2026-08-04-it-was-never-about-the-descent-rate)
  for why the batch-3 result does not license it.
- **`N_STEP_UPDATE=5`** — n=2 and n=3 both peak below baseline and then decline, so
  the trend already points the wrong way.
- **Resuming any arm from batch 10 or earlier** — every checkpoint on record from before
  batch 11 was trained on an observation vector this project has since changed (20, 23 or 26
  values against the 30 batch 11 trained on), so none of them load; see
  [`../hallOfFame/README.md`](../hallOfFame/README.md#the-entries-below-predate-2026-08-02-and-do-not-run-on-master).
  **Batch 11 is now the only resumable batch.** Its four arms were stopped healthy at 3.19-3.59M,
  and `b11d` was still near its peak, so resuming *that* arm is the one case worth considering —
  it is also the open question the 2M cap below would truncate.

### Batch bookkeeping

Each batch keeps its **description** — why it is shaped that way, what each arm isolates,
what outcome would mean what — in this file for as long as any of its arms is running.
When the last arm of a batch stops, move that description and its results to
[`completedRuns.md`](completedRuns.md) and delete it here.

The reason to keep the description live rather than only the status table: the design
rationale is what tells a future session whether a surprising result is informative or
just an arm that was never going to answer anything.

Verify what's actually running with `pgrep -fl "python -u snek2.py"`. Not
`grep "[s]nek2.py"` — git telemetry `curl` processes carry `snek2/snek2.py` in their
payload and inflate the count.
