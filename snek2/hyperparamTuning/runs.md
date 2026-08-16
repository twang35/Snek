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

## ⚠ Batches 27 and 30 were relaunched — the perfect-game counter was reward-based (2026-08-14)

**Read this before anything below it.** Every perfect-game counter identified a win by comparing the
episode's final reward with `PERFECT_GAME_REWARD`, and the chase-safe term shifts that reward by `−c`. So
`perfect_percent` read **0 for every eval of b27a-d and b30a-d** while the arms were filling boards from
step 9k, and because `training.epsilon_for` takes the trailing perfect rate as its skill signal, **epsilon
stayed pinned at 0.0125** — the refinement ceiling — instead of annealing. Full account, numbers and fix:
[`findings.md`](findings.md#-a-perfect-game-was-identified-by-its-final-reward-and-the-shaping-term-silenced-every-counter).

**The fix shipped as `b72a5a84`** — counting moves to `state_helpers.is_perfect_score(score)`, and
`tests/test_perfect_game_counting.py` pins it (10 tests, both mutants caught, suite 23 modules / 596 tests /
0 failed). Deployed to the desktop the same evening; both boxes now run it.

| batch | void arms | live arms | where |
|---|---|---|---|
| **27** | `b27a-d`, killed at 309-326k | **`b27e-h`** (seeds 1-4), launched 21:31 | desktop, priority 10, close-outs and HOF-500 auto-chained |
| **30** | `b30a-d`, killed at 137-139k | **`b30e-h`** (seeds 1-4), launched 21:40 | laptop, one chart window on `--arms b30` |

**Fresh policies, not resumes.** The void arms' weights and buffers carry 320k steps trained under an
exploration schedule that was never going to descend, which is the one thing a seed-matched comparison
against b24/b25 cannot absorb.

**Batch 30's checkpoints were deleted and the wave relaunched a second time** (21:40), because the first
relaunch's chart window opened with **eight panels**: the viewer's arm registry admitted anything inside a
12 h TTL, and the killed a-d arms were 71 minutes old. `savedPolicies/b30[a-h]` (~330 MB) and the tmp
registry are gone; **`runs/b30a-d*` is kept** — those graphs, reports and eval series are the measured
record of the counter bug, committed at `1cd5a03b`, and `charts/` holds its own copies by design. Say so if
you want them removed as well. The registry rule is now liveness-based
([the mechanism](../../CLAUDE.md#rendering-is-off-by-default--use-watchpy-to-see-a-game)); the desktop was
checked and needs no change, since its daemon passes explicit PNG paths and reads no registry.

**Progress update, 2026-08-15 08:46 — b27 and b30 are done; b28 is running; the fix held throughout.**
All twelve fixed-counter arms trained with a real perfect rate and a descending epsilon — the two things
b27a-d could not produce in 320k steps — so the counter fix is confirmed end to end. Where each batch
landed (full numbers and graphs in [`charts.md`](charts.md)):

- **b27e-h (desktop): done at 2M, closed out — a null.** Pooled mean **85.2** (eq-effort, gate 95) vs the
  b24 control's ~87.9 (a shade *below*), and **0 of 4** seeds produced a ≥98%/500 checkpoint — best `b27h`
  **97.5%** — against the control's **two** 98.0%/500 records. `c=0.10` on `fc 320` did not reproduce the
  record, let alone beat it.
- **b30e-h (laptop): done at 2M and closed out at 15:05 — also a null, and by the same margin as b27.**
  Training was a dead heat (matched ≤2M best-30 **92.9 vs 93.6**, `sef` **56.9 vs 58.6** against the b25-r2
  control), and the close-out agrees: pooled equal-effort **83.3 mean** (84.3 / 84.3 / 83.8 / 81.0) against
  b25-r2's ~86.1, so **~2.8 behind its seed-matched control** — the same direction and size as b27's 85.2 vs
  87.9 on `fc 320`. **Two architectures, two nulls-or-worse for `c=0.10`.** Ten checkpoints reached ≥98% at
  *100* episodes (6 / 3 / 1 / 0), best 99.0% — but that is not the ≥98%/**500** gate. **The HOF-500 pass has
  since run and is empty**: every one of its ten candidates was abandoned under gate 98 before 500 episodes,
  best partial `b30e` @651k **96.1% at 285 episodes**, then `b30g` 95.3% and `b30f` 90.9%. So the
  shaping×architecture 2×2 is complete and `c=0.10` produced **no record-tier checkpoint on either net** —
  the write-up is in [`completedRuns.md`](completedRuns.md) and
  [`findings.md`](findings.md#-chase-safe-reward-shaping-is-null-to-negative-at-c010--two-architectures-agree).
- **b28a-d (desktop): running, 262-275k of 2M (~13%), all four healthy** — epsilons off the ceiling
  (0.003-0.005), no zero stretch. The `c=0.20` dose rung, and a dead heat with the control this early
  (matched ≤275k best-30 **56.9 vs 57.4**). Since both `c=0.10` batches came back null, **b28 is the arm
  that decides "wrong idea" vs "dose too small to see."**
- **b29 (gate 75): queued** behind b28 on the desktop. **Laptop: `b31a-d` was stopped at 538-569k with no
  close-out and the laptop now holds `b32a-d` (C51 + Adam `epsilon`, 1M).** The reason for both is the
  same measurement — C51's churn is the learning rate, not C51
  ([`findings.md`](findings.md#-the-c51-arms-chaos-is-the-learning-rate-not-c51--and-the-rate-is-high-because-c51-needs-it)).

**Timing.** b28a-d reach 2M in ~5.5 h at ~92 steps/s. Desktop memory sits well inside the band (4 trainers
+ forked self-evals). Check the desktop with `git show origin/ops-status:status.json`.

## C51 pilot — closed at 600k, and it handed off to batch `b31` by itself (2026-08-15)

Distributional RL, phase 3 of
[`../plans/distributional-c51.md`](../plans/distributional-c51.md). The implementation is committed
(`245cf914`).

| | |
|---|---|
| wave A, 15:06 | `c51pilot-lr1e5seed{1,2}`, `c51pilot-lr5e5seed{1,2}` |
| wave B, 16:41 | `c51pilotB-lr1e4seed{1,2}`, `c51pilotB-lr25e4seed{1,2}` |
| config | b25's verbatim (`fc 200,100,100`, `IS_WEIGHTS=0`, `TARGET_UPDATE_PERIOD=1000`, `DISCOUNT=0.9975`, `FORK_BRANCHES=4`, no food-distance shaping) plus `ALGO=c51`, 51 atoms over `[-5, 120]`. Seeds matched across all four rates |
| cap | **600k steps** — a screen, not a result |
| launchers | `launch_c51_pilot.sh` (wave A, which waited out b30's close-out), then the generic `launch_c51_wave.sh` (wave B) |

**What it is asking**, in order: does a categorical agent learn this task at all; how many steps to its
first perfect game against b25's ~9k; is the loss scale sane at 1e-5 (a cross-entropy starts at
`ln 51 ≈ 3.93`, where the Huber TD loss starts near 0, so the same learning rate is not obviously the same
step size). **The gate to phase 4 is one learning rate.**

**Eight trainers on a 14-core laptop is deliberate**, and measured before launching: ~2.3 GB per arm
(0.4 GB parent + 1.9 GB across 11 forked self-eval workers) against 36 GB of RAM, and a swap-in rate of
244 pages per 20 s, so the cost is throughput — roughly half the steps/s per arm — and not paging. It is
also the one place this project's "never more than 4 trainers" rule is knowingly suspended; it was the
user's call, for this screen only.

### The handoff to `b31` ran unattended, and worked

**`launch_c51_batch.sh b31` fired at 20:09**, ~3h20m after it was armed: all eight pilot arms reached the
600k cap, it picked `5e-5`, launched `b31a-d` at 2M, regenerated the tables below and pushed the result as
`c45e8a4f` — with nobody watching any of it. Written this way because cron jobs
in this tool are session-only and fire only while the REPL is idle, so nothing scheduled can be relied on
once the session closes — a detached `nohup` can.

| step | what it does | what it refuses to do |
|---|---|---|
| wait | polls until no `snek2.py c51pilot` trainer remains (one substring covers both waves) | it excludes `chart_viewer` from that `pgrep`, or it would wait forever on the window rather than the arms |
| slots | waits for the laptop's 4-trainer limit to be free | if something else is still training after 6 h it **exits without launching** rather than breaking the limit |
| pick | [`pick_c51_lr.py`](pick_c51_lr.py)'s pre-registered rule — mean `best_perfect30` at a **common horizon**, then `sef`, then `peak_trailing` | if fewer than two rates have usable data it refuses, and the launcher falls back to `5e-5` so a batch still starts |
| launch | `b31a-d`, 4 seeds, 2M, the chosen rate, otherwise identical to the pilots — so `b25a-d` is the seed-matched control | staggers the four by 5 s rather than leaning on the chart viewer's claim lock with nobody watching |
| docs | regenerates the marked region in this file and `charts.md`, then commits and pushes | a push failure is logged, not retried — the commit is local and recoverable |

**Three guards in the picker exist because a dry run got the answer wrong.** At an early horizon every arm
reads `best_perfect30` 0.0 *and* `sef` 0.0, and a two-level rule then picked whichever rate came out of a
dict first — it chose the slowest rate over the fastest, so `peak_trailing` is now the third key. An arm
that dies early no longer sets everyone's horizon (it would have judged seven healthy arms at 13k of 600k).
And an arm with no eval series is excluded and named rather than counted as a zero, so a failed launch
cannot vote against its own rate.

<!-- C51-PILOT-STATUS:BEGIN -->
*Generated by `pick_c51_lr.py` at 2026-08-15 20:09, when the last pilot arm stopped — the numbers below are read straight off the eval series, and the prose around this block is hand-written.*

**Compared at a common horizon of 600k steps**, the lowest final step any arm reached, because both metrics accumulate over an arm's own evals and a longer arm would otherwise win on horizon alone.

| lr | seeds | mean best-30 | mean `sef` | mean peak trail |
|---|---|---|---|---|
| 5e-05 **← chosen** | 2 | 69.5 | 12.6 | 92.42 |
| 1e-05 | 2 | 56.5 | 3.6 | 89.89 |
| 0.0001 | 2 | 39.0 | 5.3 | 88.19 |
| 0.00025 | 2 | 4.0 | 0.0 | 68.79 |

| arm | lr | seed | step | best-30 | `sef` | peak trail | first perfect |
|---|---|---|---|---|---|---|---|
| `c51pilot-lr1e5seed1` | 1e-05 | 1 | 600k | 85.3 | 7.3 | 93.56 | 141k |
| `c51pilot-lr5e5seed2` | 5e-05 | 2 | 600k | 71.7 | 13.0 | 93.30 | 20k |
| `c51pilot-lr5e5seed1` | 5e-05 | 1 | 600k | 67.3 | 12.1 | 91.54 | 15k |
| `c51pilotB-lr1e4seed2` | 0.0001 | 2 | 600k | 66.3 | 10.6 | 90.80 | 46k |
| `c51pilot-lr1e5seed2` | 1e-05 | 2 | 600k | 27.7 | 0.0 | 86.22 | 92k |
| `c51pilotB-lr1e4seed1` | 0.0001 | 1 | 600k | 11.7 | 0.0 | 85.58 | 8k |
| `c51pilotB-lr25e4seed1` | 0.00025 | 1 | 600k | 5.7 | 0.0 | 70.82 | 49k |
| `c51pilotB-lr25e4seed2` | 0.00025 | 2 | 600k | 2.3 | 0.0 | 66.76 | 59k |

**Chosen: `5e-05`** — best_perfect30 69.5 against 56.5 for the next rate (1e-05).

**Batch `b31` launched at 2026-08-15 20:09** on this rate, 4 seeds, 2M cap, `fc 200,100,100`, otherwise b25's config — so `b25a-d` is the seed-matched control.
<!-- C51-PILOT-STATUS:END -->

**`SNEK_CHART_VIEWER=0` on all four, one window opened by hand.** `chart_viewer.batch_prefix` groups only
`b<n><letters>-` names, so four `c51pilot-*` arms would open four windows; the launcher uses the
`--glob`/`--watch` form an eval wave already uses. The pilot deliberately does **not** claim `b31` — `fc 512`
and the four owed `320` seeds are ahead of C51 in the backlog below.

## Batch 32 — Adam's `epsilon` on C51, `lr 1e-4`, two reference values — running on the laptop (2026-08-15)

**Does `epsilon` separate C51's learning speed from its churn?** Four arms to **1M**, `lr 1e-4`
throughout, everything else b25's config plus `ALGO=c51`:

| arms | `SNEK_ADAM_EPSILON` | source of the value |
|---|---|---|
| `b32a`, `b32b` (seeds 1, 2) | **1.5e-4** | Dopamine's published Rainbow config |
| `b32c`, `b32d` (seeds 1, 2) | **3.125e-4** | Dopamine's published C51 config |
| *control — already on disk* | 1e-7 (Keras default) | `c51pilotB-lr1e4seed1/2`, same config, 600k |

**The control is not in this wave**, which is why four arms buy a three-way comparison. The two pilot
arms at `lr 1e-4` ran this config at the framework default, so **seeds 1 and 2 are reused deliberately**
and the comparison is paired at a 600k horizon with the extra 400k as free information. Paired matters:
the seed decides which arm in a wave wins in 18 of 18 measured waves.

**Why `lr 1e-4` and not the pilot's chosen `5e-5`.** It is where the defect is largest while the rate
still learns — churn **0.117-0.245** against the ddqn control's 0.033-0.058, never settling, yet seed 2
reached best-30 66.3 still rising at 599k. At `2.5e-4` the arm is broken outright and a working fix could
be invisible under whatever else is wrong.

**Judge it on churn and drawdown depth, not `best_perfect30`.** Within-rate seed spread at `1e-4` is
**54.6 pp**, so at n=2 per side the score resolves nothing — the same trap the pilot's rate screen fell
into. The readout is
[`perDiagnostics/c51_stability.py`](perDiagnostics/c51_stability.py) at `--end 600000` against
`c51pilotB-lr1e4seed1/2`. What each outcome means:

- **Churn drops toward the ddqn floor and drawdowns shrink, learning speed holds** → the mechanism is
  confirmed and `epsilon` becomes a standing part of the c51 config. The follow-up is then re-running the
  rate screen with it, since the pilot's whole rate ranking was measured at 1e-7.
- **Churn drops but so does learning speed** → it is acting as a smaller learning rate in disguise, which
  the pilot already showed costs the peak. `tests/test_adam_epsilon.py` pins that it *should not* — a
  well-driven gradient keeps >98% of its step — so this outcome would mean the gradients here are far
  smaller than assumed, which is itself worth knowing.
- **Nothing moves** → falsified in 4 arms, and the remaining candidates are the exploration-schedule
  ratchet and n-step returns, in that order.

**Timing.** ~1.7 h to 1M at four arms. `1e-7` is still the default, so nothing else in the project
changes; the knob is recorded per-arm in `runs/<policy>.md`.

## Batch 31 — C51 at `lr 5e-5`, 2M — **stopped at 538-569k, no close-out** (2026-08-15)

Launched 20:09 by the pilot's handoff, **killed 23:10 at the user's call** after
[`c51_stability.py`](perDiagnostics/c51_stability.py) showed the chaos was the learning rate rather than
C51, which made a 2M run at a rate chosen under the old reading not worth the slots. **No close-out was
run** — deliberately, not pending.

Reached 538-569k in 2h44m, all four healthy (no zero stretch), best-30 **21.0 / 53.3 / 66.7 / 71.7** — a
**50.7 pp spread at one config**, which is the n=4 noise problem restated rather than a result. Graphs in
[`charts.md`](charts.md); the arms are in [`completedRuns.md`](completedRuns.md) as void.

## Batch 27 / 28 / 29 — potential-based chase-safe shaping (b28 running, b29 queued — desktop)

**The shaping shipped 2026-08-14; batch 27 is on its second launch** (`b27e-h` — a-d were void, see the
banner above). `Snake.step` now adds
`c·(γΦ(s′) − Φ(s))` where **Φ = 1 iff the head and the tail share a free region that also contains the
food, and the snake is at least `SNEK_CHASE_SAFE_GATE` long** — the length gate is variant B of the plan.
Potential-based, so by Ng/Harada/Russell the optimal policy is unchanged for any bounded Φ and any `c`;
what changes is the gradient the agent sees on the way there, and the PER priorities of the transitions
that flip Φ. Everything is in [`../plans/chase-safe-reward-shaping.md`](../plans/chase-safe-reward-shaping.md).

All twelve arms are **b24's config plus the one knob** — `fc 320`, `IS_WEIGHTS=0`, `td_error`,
`TARGET_UPDATE_PERIOD=1000`, `DISCOUNT=0.9975`, `FORK_BRANCHES=4`, no food-distance shaping, seeds 1-4 —
so **`b24a-d` is the seed-matched control at zero extra compute**. The cap is **2M**, not b24's 3M: b24's
best HOF-500 checkpoints land at 1.03-1.39M and 2.86M, and 2M buys three batches for the price of two.

| batch | `c` | gate | priority | what a lift here means |
|---|---|---|---|---|
| **b27** (done — null) | 0.10 | 85 | 10 | the calibrated dose — one quarter of a meal's reward spread over the ~2.5 genuine flips a struggling policy makes per endgame meal. **Result: pooled 85.2 vs control ~87.9, 0 of 4 ≥98%/500. No lift.** |
| **b28** (running) | **0.20** | 85 | 20 | the dose ladder. b27 *was* null, so this separates "wrong idea" from "too small to see" — the one ambiguity a single-dose test cannot resolve |
| **b29** (queued) | 0.10 | **75** | 30 | the gate ladder. 75 starts shaping ~10 meals earlier, where the packing decisions that *create* the endgame are made |

**`c = 0.10` is measured, not assumed** ([Phase 0](findings.md#-measured-the-chase-safe-potential-is-nearly-static-for-a-record-policy-and-busy-for-a-bad-one)):
Φ flips **~35 times an episode for `b20d`** and **~4.6 for the records**, so the term self-attenuates as a
policy improves — it is a scaffold that fades. At 0.10 the total shaping over an episode is ~0.25 of a
single meal's reward for a struggling policy, and the discounted sum telescopes to exactly
`−c·Φ(s₀)` = **0** here, since the opening board (length 5) is below any gate.

**Two things Phase 0 established that the design had wrong.** Gating does *not* buy a larger `c` — flips
are concentrated at length ≥85 for the binding policy, so gating **halves** the calibrated dose
(0.203 → 0.097). And **Φ almost never *changes* at length 98-99** — 0.00-0.04 genuine flips per meal — so
the last two or three meals get no gradient from this quantity at any `c`, which is exactly where
[the starvation finding](findings.md) says the losses are. So b27 is a test of whether *reaching* length
96-98 more reliably raises the perfect rate, not a fix for the final meals.

**Corrected 2026-08-14: "no flips at 98-99" is not "Φ is 0 there", and reading it that way is what let the
counter bug ship.** This section used to say Φ was *structurally 0* at length 99. Measured on a constructed
full-tour board, **Φ(s) = 1** on the pre-win board: with one cell free the food occupies it, the head's only
legal move is into it, and in the tail-chasing endgame the tail borders it too, so head, food and tail do
share a region. What is 0 at length 99 is the **per-action flag** at `obs[15 + a]`, which is computed on the
*post-move* board — after the winning move there is no food and no free cell
([the Phase 0 note](findings.md#the-record-checkpoint-specifically) is about that flag). The consequence is
concrete: the winning transition **is** shaped, paying exactly `−c`, which is the value that stopped every
perfect-game counter from recognising a win.

**Judge these on `best_perfect30` and the count of ≥98%/500 checkpoints, not on peak trailing.** Peak is
capped at 95.00 and all four b24 arms already sit on the cap
([why](findings.md#-peak-trailing-is-a-saturated-metric--it-is-capped-at-95-and-four-arms-already-sit-on-the-cap)),
so it cannot move even if a shaped arm plays perfectly. `sef` alongside, at the matched 2M horizon.

**The implementation caught a live defect the design would have shipped.** The potential is per-episode
state, and `Game.snapshot()` does not carry it — with `FORK_BRANCHES=4` forking at length ≥85, every
endgame fork would have started its shaping from the *parent's* stale Φ. `restore_snapshot()` now
recomputes it. 24 tests in `tests/test_reward_shaping.py` pin the behaviour, each verified to fail against
a mutated implementation.

Desktop status (2026-08-15): **b28a-d running** (4 trainers, the box's cap), **b29 queued** behind them,
each with close-out and HOF-500 auto-chained. b27e-h finished and closed out; ~6 h per batch at ~92 steps/s.
Check with `git show origin/ops-status:status.json`.

**b30e-h is fully done** — finished at the 2M cap, then closed out (killed mid-run and resumed with
`EVAL_RESUME=1`) and HOF-500 re-measured on the laptop on 2026-08-15, both null (0 of 10 checkpoints
≥98%/500). The shaping×architecture 2×2 is complete; see the progress banner above and
[`completedRuns.md`](completedRuns.md#batch-30--chase-safe-shaping-on-fc-200100100-c010-null-and-it-completes-the-shapingarchitecture-22).

## Batches 20-26 are closed — where their descriptions went

All seven batches finished and closed out, so per the bookkeeping rule at the end of this file their
descriptions moved to [`completedRuns.md`](completedRuns.md):

| batch | change | verdict | design + results |
|---|---|---|---|
| **26** | **fc `100,100`** under IS-off | **does not carry the lift** — pooled 79.2, only +3.5 over the control, at **more** parameters than b24's `320`. The arm that showed the lift is width, not size | [write-up](completedRuns.md#batch-26--fc-100100-under-is-off-the-shallow-shape-does-not-carry-the-lift) |
| **25** | **fc `200,100,100`** under IS-off | **the lift replicates** — pooled 86.0, +10.3, all 4 seeds; peak unmoved. No hall entry: the auto chain's gate-98 abandoned every candidate, `b25b` @911k still 97.2% at 392 episodes | [write-up](completedRuns.md#batch-25--fc-200100100-under-is-off-the-lift-replicates-at-a-second-shape--but-no-record) |
| **24** | **fc `320`** under IS-off | **first architecture result + a new record** — pooled **87.9**, +12.2 over the control, all 4 seeds (ceiling unmoved). HOF-500: `b24d` @1342k **98.0%/500**, the new record | [write-up](completedRuns.md#batch-24--fc-width-320-under-is-off-the-first-architecture-result-and-a-new-record) |
| **23** | IS β annealed **0→0.1** | **the best point on the β ladder** — pooled **75.7**, +20.7 over the control, higher on all 4 seeds | [write-up](completedRuns.md#batch-23--β-annealed-001-the-best-point-on-the-β-ladder-near-the-no-is-extreme) |
| **22** | IS **off** (`SNEK_IS_WEIGHTS=0`) | **dead heat with β→0.1** — pooled 75.7. The consolidation gain saturates by β→0.1 | [write-up](completedRuns.md#batch-22--is-off-a-dead-heat-with-β01--the-consolidation-gain-saturates) |
| **21** | partial IS (β→**0.5**) | beats the β→1.0 control (pooled 64.3 vs 55.0, 3/4 seeds), well short of no-IS | [write-up](completedRuns.md#batch-21--partial-is-β05-beats-the-β10-control-still-far-behind-no-is) |
| **20** | `FC_LAYERS`, **nine shapes** | **architecture never raises the ceiling**; capacity binds only below ~0.55× | [the sweep's design](completedRuns.md#batch-20--the-design-of-the-nine-shape-sweep-complete-2026-08-12) |

**‡ The width result now has three shapes behind it, and "it tracks capacity" is retracted.** Pooled lift
over the b22 control orders by **widest layer**, not by size: `320` **+12.2** at 0.94× the control's
parameters, `200,100,100` **+10.3** at 3.09×, `100,100` **+3.5** at 1.14×. The smallest of the four nets
wins and the second-largest gets almost nothing, so parameter count is not even monotone with the result
([finding](findings.md#-corrected-2026-08-14-the-is-off-architecture-lift-tracks-the-widest-layer-not-the-parameter-count)).
**The architecture arm this implies is `fc 512` under the b24 config** — a wider first layer, not a bigger
net — and it is now the strongest remaining consolidation direction after the shaping batches.

**The β ladder is the live result to build on.** Gradient concentration ESS/N walks down it — **β→1.0
≈1.0** (batch 20's control, near-uniform) → **β→0.5 ≈0.86** (b21) → **β→0.1** (b23, effective exponent
α·(1−β)=0.54) → **IS off ≈0.38** (b22) → b18's no-IS ≈0.21 — and pooled climbs monotonically with it:
**55.0 → 64.3 → {75.7, 75.7} → ~78.8**, then **flattens at the bottom**. Most of the consolidation is
bought by β→0.1; going all the way to IS-off adds nothing measurable. **The ceiling is unmoved throughout**
(peak 94.4-94.9), so the ladder buys time-near-the-ceiling, not a higher one.

`b23b` holds five full-length checkpoints ≥95/100 around 777k (best 97/100). It was **not** a
hall-of-fame candidate after re-measurement — the selected @777k reads 92.4% over 500 fresh episodes, the
worst of its own cluster. The `b23b` 217-242k collapse was investigated in place and is **not** an escape
from a local minimum; all four seeds make the same level shift
([`findings.md`](findings.md#-falsified-a-drawdown-is-not-how-a-policy-escapes-a-local-minimum)).

## Record status

**NEW RECORD, 2026-08-13: `b24d-fc320noisseed4` @1342000 — 98.0% over 500 fresh episodes** (490/500,
CI **96.4-98.9**). It edges the prior record, `b18b-tgt1000seed2` @1588000 at 97.6%/700 (CI 96.1-98.5), on
the point estimate — intervals overlap, so it is a narrow lead, **taken as confirmed under the folder's
500-episode standard** (we are not re-running for the 700-vs-500 difference). It is the first record to come
from a non-default architecture (`fc 320`, not `50,100,50`). `b24b-fc320noisseed2` @2860000 ties it at
98.0%/500. Both promoted to [`../hallOfFame/`](../hallOfFame/README.md), copies verified to load and play.

**Like `b18b` before it, `b24d` @1342k is a checkpoint that did *not* shrink** — it read 97.0/100 in the
close-out and **rose** to 98.0%/500 on re-measurement, the genuine-region signature. That is the exception,
not the rule: of the batch's 199 ≥97%/100 checkpoints only 9 held ≥97%/500, and `b24a`'s two 100%/100 highs
produced zero survivors. Selection inflates a /100 high by ~5-6 pp, which is why every HOF add must clear the
500-episode re-measurement first (see [`../hallOfFame/`](../hallOfFame/README.md)).

**The prior record, `b18b` @1588000 (97.6%/700), still stands as the deepest-measured strong checkpoint** and
was itself the first selected high that did not shrink (98/100 → 97.4%/500, a 0.6 pp change, against `b17b`
99→94.2, `b15b` 97→93.0, `b14a` 96→93.5, `b11b` 96→~94).

**But `b18b` @1588k is a narrow peak, not a strong region.** `@1578000` — 10k steps earlier — reads
**91.6%**, 5.8 pp lower. The old rule survives every better result: a selected row describes the checkpoint
the screen liked, never its neighbourhood — and the same caution applies to `b24d` @1342k, whose neighbours
have not been position-sampled. Any regional claim still needs a position-chosen sample.
Full derivation of that error: [`archive/runs-archive.md`](archive/runs-archive.md); measurement
caveats: [`findings.md`](findings.md#three-measurement-caveats).

### Max progression across batches

Best checkpoint per batch, at most three each, newest first. **Two columns, because they say
different things:** `selected` is the close-out's own best row and reads high by construction;
`re-measured` is a later independent sample and is the only column that can be compared across
batches. `*trunc*` means no full-length row survived the gate, so the figure is shorter and noisier.

| batch | change | best selected | re-measured |
|---|---|---|---|
| **24** | **fc `320`** under IS-off | 100% @1633k /100 (`b24a`) · 100% @2126k /100 (`b24c`) · 99% @1031k /100 (`b24b`) | **98.0% /500** ← **new record** (`b24d` @1342k) · 98.0% /500 (`b24b` @2860k) · 97.4% /500 (`b24c` @2982k) |
| **20** | `FC_LAYERS` shapes (**all 9 closed**) | 92% @1470k *trunc* (`b20ah`, 100,200,100) · 91% @1435k *trunc* · 91% @2935k *trunc* (`b20g`) | **none reached full length** — 0 of 36 arms |
| **19** | standard PER + IS | 91% @1536k *trunc* · 77% @1485k *trunc* · 76% @937k *trunc* | **none reached full length** |
| **18** | `TARGET_UPDATE_PERIOD` 1000 | **98% @1588k** · 97% @1601k · 96% @1289k | **97.6% /700** ← **record** · 94.7% /700 · 85.4% /500 |
| 17 | forked endgame collection | 99% @1248k · 99% @1205k · 98% @1231k | **94.24% /5120** · region grid 84% |
| 16 | `FOOD_DISTANCE_REWARD=0` | 93% @913k *trunc* · 92% @1203k · 85% @979k *trunc* | — |
| 15 | `N_STEP_UPDATE=3` | 97% @3245k · 95% @4697k · 91% @3671k | **93.0% /300** |
| 14 | `DISCOUNT=0.9975` | 96% @3702k · 93% @2559k · 90% @2261k | **93.5% /200** |
| 13 | eps handover + shield | 95% @986k · 92% @3367k · 91% @1166k | — |
| 12 | eps handover 0.05 | *deadlocked, not measured* | — |
| 11 | the 30-value vector | 96% @855k · 94% @671k · 88% @3507k | **~94%** (shrunk) |
| 10 ‡‡ | `DISCOUNT=0.995` | 93% @1695k · 90% @1501k · 85% @2344k | 74.9% /66000 pooled |
| ≤9 | earlier environments | `b8f` 92%, `b9d` 70%, `b7f` 51%, `b4c` 50% | not comparable |

**Read the re-measured column and the story is short: 94.2% for a year of batches, then 97.6%.**
Batches 11-19 all train on the same 30-value vector so they are comparable to each other; batch 10
(‡‡) and everything below it are earlier environments where the same checkpoint scores differently,
which is why those rows are not a trend line.

**The selected column is nearly flat from batch 11 on** — 93-99% in every batch that produced a
full-length row — which is exactly why it cannot be used to judge progress. Batch 17's three 98-99%
rows re-measured to 94.2%; batch 18's 98% re-measured to 97.6%. Same selected number, 3.4 pp apart in
reality.

**Outstanding, highest-value, in order:**

1. **`CHASE_SAFE_SHAPING`: b27 and b30 are closed and both null at `c=0.10` (0 records on either net); b28
   (`c=0.20`) and b29 (gate 75) are the live question.** Design, Phase 0 measurement and implementation notes:
   [`../plans/chase-safe-reward-shaping.md`](../plans/chase-safe-reward-shaping.md); the batch description is
   at the top of this file. The next decision point is b28's close-out against `b24a-d` — whether a doubled
   dose does what `c=0.10` did not.
2. **`fc 512` under the b24 config is now the strongest untested architecture arm.** b25/b26 turned the
   width result into an ordering on the *widest layer* — 320 → +12.2, 200 → +10.3, 100 → +3.5, 50 → 0 —
   with parameter count not even monotone
   ([finding](findings.md#-corrected-2026-08-14-the-is-off-architecture-lift-tracks-the-widest-layer-not-the-parameter-count)).
   `512` is the only cheap way to ask whether that ordering keeps going or has a knee, and it needs no new
   control: `b24a-d` is seed-matched. Order it behind the shaping batches, which own the queue.
3. **4 more `320` seeds are still owed before the width×IS-off interaction is called established.** b24's
   +12.2 is at the n=4 sign-test floor (p=0.0625). b25/b26 answered a different question (which property
   carries the lift) and do not firm up the `320` gap itself.
4. **Consider a position-chosen grid around `b18b` @1588000.** The record is a narrow peak, so the
   open question is whether `TARGET_UPDATE_PERIOD=1000` produces a better *region* or just got one
   lucky checkpoint. A blind every-10k grid over 1.55-1.62M would settle it, and it is the same test
   that deflated `b17b`'s apparent region to 84%.
5. **The 11 batch-18 checkpoints at exactly 95.0%** were excluded from the 500-episode sweep, which
   took ">95%" literally. ~9 minutes of eval if a fuller picture of the region is wanted.

## Closed batches (11-19)

One line each; **batches 20-26 are in the table above**, with write-up links. Full write-ups and
per-seed numbers in [`completedRuns.md`](completedRuns.md), superseded detail in
[`archive/runs-archive.md`](archive/runs-archive.md).

| batch | change | verdict |
|---|---|---|
| **19** | standard PER (`td_error` + IS on, β→1.0) | **falsified** — worse on all 5 pooled metrics, 4/4 seeds, p=0.125. Drawdown 55.5 → 8.8, but at a lower level |
| 18 | `TARGET_UPDATE_PERIOD` 8 → 1000 | primary moved: 102k faster to pf30≥40%, 4/4 seeds; drawdown improved. **Close-out done** — tightest eq-effort spread of any batch (74.8-81.9), 20 rows ≥95% |
| 17 | `FORK_BRANCHES=4` forked endgame collection | null on config (one seed carried it), **project record on `b17b`**; dose ~60% of design |
| 16 | `FOOD_DISTANCE_REWARD=0` | **first non-null** — `sef` +11.35 pp at 1.25M; consolidation not ceiling. Needs replication |
| 15 | `N_STEP_UPDATE=3` | falsified on speed — 128k slower; evals null |
| 14 | `DISCOUNT=0.9975`, `GUIDED_FRACTION=0.8` | null vs 13 |
| 13 | eps handover 0.0125 + shield 0.5 | null vs 11 on five metrics |
| 12 | eps handover 0.05 | deadlocked, abandoned 4/4 |
| 11 | the 30-value vector itself | +4-5 pp vs 10, not significant |

**The binding constraint is seed variance, not ideas** — n=4 resolves nothing below ~10 pp, and peak
trailing reads 93.8-95.0 flat across batches 11-23. Nothing has raised the ceiling; batches 16 and 21-23
raised how much of the *time* an arm sits near it. **Two things have moved peak trailing downward** —
batch 19's full IS correction (94.16, 4/4 seeds) and batch 20's 0.29× net (93.75, 4/4) — so the invariance
is breakable, just not yet upward. **Fifteen batches of optimiser, PER and architecture knobs have not
raised the ceiling once** — peak trailing reads 94.84-95.00 across `50,100,50`, `100,100`,
`200,100,100` and `320` — which is the argument behind the reward-shaping batches now running (b27-b29).

## Standing backlog

Untested, ordered by expected value. Rationale for the ones that need it follows the
table.

| change | targets | prior |
|---|---|---|
| ~~`CHASE_SAFE_SHAPING`~~ — potential-based shaping on head/food/tail in one region | endgame food-finding, the modal failure since batch 16 | **shipped and running 2026-08-14** as **b27** (`c` 0.10, gate 85), **b28** (`c` 0.20) and **b29** (gate 75) — see the batch description at the top of this file and the [plan](../plans/chase-safe-reward-shaping.md) |
| **Free space in one piece** — as a graded potential and/or an observation | endgame packing, which is what decides whether the food lands somewhere edible | **new 2026-08-14**, and the largest per-policy separation on record: one-piece share at length 90-94 is **92% / 77% / 5%** for `b24d` / `b18b` / `b20d` ([findings](findings.md#-the-packing-property-the-records-keep-their-free-space-in-one-piece-and-it-separates-them-by-87-points)). `count_groups` already runs every step, so both forms are nearly free. **Correlational, n=3 checkpoints** |
| **`SNEK_ALGO=c51`** — distributional RL, a categorical value head instead of a scalar Q | post-peak collapse, which has no established mechanism now that plasticity loss is ruled out | **proposed 2026-08-15, awaiting review** — [plan](../plans/distributional-c51.md). The first change to the *loss* rather than a reward, width or schedule, and the only backlog item that needs a code change before it can be launched by env var. A feasibility probe already ran: the shield, the buffer spec and the network all work unchanged, and `tf_agents`' own C51 loss **silently drops IS weights and returns no PER priority**, both of which the plan fixes. Honest prior: ~40% null on `best_perfect30` |
| `LEARNING_RATE=1e-4` | training speed | high, but order it after a stability fix |
| ~~`TARGET_UPDATE_PERIOD`~~ | early learning speed, target stability | **closed as batch 18** — primary moved, 4/4 seeds; see [`completedRuns.md`](completedRuns.md) |
| `TARGET_UPDATE_TAU=0.005`, period 1 | smoothness (soft target updates) | medium |
| ~~`FC_LAYERS=128,128`~~ | capacity | **closed by batch 20**, which swept nine shapes rather than one: none raised the ceiling — [findings](findings.md#network-shape-the-sweep-is-complete--nine-shapes-and-architecture-never-raises-the-ceiling). Width under IS-off is the one loose end, and batches 24/25 are on it |
| **`FC_LAYERS=512`** under the b24 config | consolidation — how far the widest-layer ordering goes | **new 2026-08-14, and the top architecture item.** b24/b25/b26 order the lift by widest layer (320 → +12.2, 200 → +10.3, 100 → +3.5) with parameter count non-monotone ([finding](findings.md#-corrected-2026-08-14-the-is-off-architecture-lift-tracks-the-widest-layer-not-the-parameter-count)). `b24a-d` is already the seed-matched control, so the arm costs four trainings and nothing else |
| ~~epsilon ladder *shape*~~ | exploration schedule | **done 2026-08-04** — rewritten, needs measuring |
| `REPLAY_BUFFER_MAX_LENGTH=1000000` | experience diversity | low — the 500k result was ambiguous |

**`LEARNING_RATE=1e-4` — only after a stability fix.** 1e-5 is very conservative and
the in-code comment already suggests 1e-4. With a stable target it may train several
times faster; on its own with `TARGET_UPDATE_PERIOD=8` it would probably make
instability worse. The order matters.

**Epsilon ladder shape — rewritten 2026-08-04, and still never measured in isolation.** Every batch from
13 on has run the new ladder, so it is baked into all of them and no arm separates it from the config it
shipped with; that is why it stays on this page even though the code change is long done.
The old ladder ran **96.8% of batches 10-11's training steps at epsilon
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

- ~~**Reward changes**~~ — **retracted 2026-08-07, and the stated reason was wrong.** The claim was
  that changing a reward breaks comparability of `avg_score`. It does not: `avg_score` is a count of
  food eaten, so it and every eval metric derived from it are on the same scale whatever the rewards
  are. Only `avg_reward` changes scale. **Retracting it paid off immediately**: `FOOD_DISTANCE_REWARD`
  became tunable and batch 16's ablation of it is the first non-null in six batches. `FOOD_REWARD`,
  `DEATH_REWARD`, `STARVE_REWARD` and `PERFECT_GAME_REWARD` remain fixed, since those *do* rescale
  `avg_reward` enough to move the bootstrap epsilon thresholds a long way — but the batch-16 result is
  a standing argument for looking at the rest of the reward function rather than at another optimiser
  knob. See [`hyperparamTuning.md`](hyperparamTuning.md#available-knobs).
- **Plasticity interventions — resets, ReDo, shrink-and-perturb, L2-to-init** — **closed 2026-08-14
  before being tried.** These are the standard response to the shape of this project's curves, and the
  mechanism they target is measurably absent: across 9 arms dormancy *falls* from a fresh-net control,
  centred srank ends at 95-99% of it, and a direct fit-a-new-target probe reads **0.96-1.52× a fresh
  net** with a paired 3M change of −0.021 to +0.022. `b20d` collapses 80.3 → 42.7 while its probe fit
  **rises**. See
  [`findings.md`](findings.md#-falsified-2026-08-14-there-is-no-plasticity-loss--the-collapsed-networks-fit-a-new-target-better-than-their-own-peak).
  The one real ageing signature is weight growth (1.4-2.7× init) with movement decay (3-10×), nearly all
  of it inside the first 500k — a **shrinking effective step size**, whose fixes are weight decay or a
  larger late learning rate, not resets. That variant is not planned either while `LEARNING_RATE` and
  `CHASE_SAFE_SHAPING` are ahead of it in the backlog.
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
- **`N_STEP_UPDATE=5`** — batch 15 measured n=3 and the predicted mechanism is absent: it reached
  pf30 ≥ 40% **128k later** than its control, 3 of 4 seeds slower, with level a null. The whole case
  for a larger n was that the effect scales with n, so its absence at n=3 is the worst possible sign
  for n=5. Contamination was never the obstacle either — 0.53% of targets at n=3, 1.06% at n=5 — so
  there is no cost to blame the null on. **Closed unless the propagation story is revived by
  something other than n.**
- **Resuming any arm from batch 10 or earlier** — every checkpoint on record from before
  batch 11 was trained on an observation vector this project has since changed (20, 23 or 26
  values against the 30 batch 11 trained on), so none of them load; see
  [`../hallOfFame/README.md`](../hallOfFame/README.md#the-entries-below-predate-2026-08-02-and-do-not-run-on-master).
  **Batches 11 onward are all resumable** — 11-15 share the 30-value vector. The arms worth resuming
  are the ones stopped mid-climb: **`b15a` and `b15d`, both still gaining in their final 500k band at
  5.5-6.0M**, then `b14c`, `b11d` and `b13c`. Resuming needs `SNEK_MAX_STEPS` raised above the arm's
  current step or it exits immediately.

### Batch bookkeeping

Each batch keeps its **description** — why it is shaped that way, what each arm isolates,
what outcome would mean what — in this file for as long as any of its arms is running.
When the last arm of a batch stops, move that description and its results to
[`completedRuns.md`](completedRuns.md) and delete it here.

The reason to keep the description live rather than only the status table: the design
rationale is what tells a future session whether a surprising result is informative or
just an arm that was never going to answer anything.

**Verify what is running on both hosts — neither check sees the other**, so a count is meaningless
without naming the box:

| host | check |
|---|---|
| laptop (4 trainers max) | `pgrep -fl "python -u snek2.py"` |
| desktop `the-claw-den` | `git show origin/ops-status:status.json` — read `counts`, `running`, and the `iso` heartbeat |

Not `grep "[s]nek2.py"` on the laptop — git telemetry `curl` processes carry `snek2/snek2.py` in their
payload and inflate the count. And `git fetch` first: `git show origin/ops-status:...` serves whatever
was last fetched, which on 2026-08-12 was **17 hours stale** and showed four finished evals as still
running.
