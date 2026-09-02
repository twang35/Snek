# Runs — current state and forward plan

**Newest at the top, in every doc in this directory.** Current state first, then what is next,
then how we got here. A batch that closes is written above the batch before it, and a new finding
goes directly under `## Established` in [`findings.md`](findings.md).

## Now

**Batch b8 — "what fixes b4's collapse" — is training on the desktop, wave 1 of 2, ~70% done at
2026-09-01 14:03.** Eight arms at 100M transitions each: entropy **0.003** x seeds 1-4
(`b8a`-`b8d`) and the **0.01 -> 0.001 anneal** x seeds 1-4 (`b8e`-`b8h`). 2 h 49 m in, ~7,000
steps/s per arm, 69-75M done, so wave 1 lands around 15:15 and its stage B follows automatically.
Wave 2 (`target_KL` 0.02, λ 0.95) and b8's stage B are queued behind it. **Nothing is running on the
laptop.**

Read at b8's own horizon against b4's control truncated to the same 71M — **both treatments are
marginally ahead and neither is resolved.** n=4 cannot resolve an effect below ~10 pp, and best30 is
a peak statistic that is still rising in every arm:

| group | n | best30 | range | sef |
|---|---:|---:|---|---:|
| b4 control, entropy 0.01, truncated to 71M | 8 | 96.69 | 95.8-97.3 | 85.8 |
| b8 entropy 0.003 | 4 | 96.98 | 96.5-97.4 | 88.2 |
| b8 entropy anneal 0.01 -> 0.001 | 4 | 96.93 | 95.9-98.0 | 87.1 |

`b8g-entanneal-seed3` at best30 **98.0** is already above every b4 arm's final best30 at twice the
budget, which is the single most encouraging number in the batch and also exactly the kind of
maximum-over-8 that this project keeps having to retract. Wait for stage B.

**‡ One caveat on b8's design, now that b7 has closed: it holds `fc (200,100)`, which b7 has just
shown to be the *wrong* shape.** b8 is still the right experiment — it is asking what fixes b4's
drawdowns, and b4 is `fc (200,100)`, so the control has to match — but a stability knob that helps
here has to be re-confirmed on `fc (320,)` before it goes into a champion attempt.

## Just closed: b7, the fc-layout sweep — `fc (320,)` wins

**All four waves and all four stage-B passes closed 2026-09-01**, 32 arms in ~11 h on the desktop.
Pooled 10.9% of 28,006 stage-B rows in the ≥98%/500 record region, and the spread across layouts is
3.4x:

| layout | ≥98%/500 | | layout | ≥98%/500 |
|---|---:|---|---|---:|
| **`fc (320,)`** | **17.3%** | | `fc (200,100,50)` | 10.8% |
| `fc (200,100)` | 11.8% | | `fc (160,160)` | 8.3% |
| `fc (100,200,100)` | 11.6% | | `fc (300,100)` | 6.8% |
| `fc (100,100)` | 11.3% | | `fc (400,200)` | 5.1% |

**Every `fc 320` seed beats every seed of five of the seven other layouts** (exact Mann-Whitney
p=0.029, the floor at 4-vs-4). This **inverts b3's single-seed ranking**, which put `fc 300,100`
first and `fc 320` last of those three and is what queued b4 and b7 in the first place. Per-arm
numbers in [`results.md`](results.md), the reading and the two retractions in
[`findings.md`](findings.md), 64 charts in [`charts.md`](charts.md).

**It also changes the primary metric for this kind of question.** `strong_eval_fraction` ranks b7's
layouts *backwards* (Spearman −0.79 across the eight layout means); the stage-A ≥98% rate ranks them
right (+0.80). Both are free from the same eval history — see [`findings.md`](findings.md).

## The b8 design, as queued

**"What fixes b4's collapse".** 4 stability knobs x 4 seeds at 100M, holding b4's config fixed
(`fc (200,100)`, 8 epochs, b2's reward, seeds 1-4) so exactly one knob moves per group, with **b4
itself as the control** at 8 seeds — no control arms queued. Priorities 60/70 put both waves behind
every b7 wave, which is why wave 1 only started once b7's fourth close-out was done.

| wave | knob | why it, and not something else |
|---:|---|---|
| **1** | entropy **0.003** | the only knob whose stability signal in b3 is monotone in both directions: the share of post-competence evals below 80% perfect ran 2.9% at 0.003, 12.2% at the 0.01 reference, **45.6%** at 0.03 |
| **1** | entropy **0.01 → 0.001** | the anneal in `ppo/schedules.py`, **never used by an arm**. Explore early, commit late; it completes exactly at this cap |
| 2 | **`target_KL` 0.02** | **never exercised** — `epochs_run` is 8 in all 97,656 of b4's recorded updates. 0.02 binds on the tail and not the body: b4's `approx_kl` is 0.0035 median and 0.0079 at p95, but **0.023 at p99 and 0.514 at worst**, 146x the median |
| 2 | **λ 0.95** | b3's two λ arms had the best drawdown profile in the sweep — 0.0% below 50% *and* below 80%, mean 93.9 against the reference's 88.2 — at one seed each |

**Why this and not a second fc sweep at a smaller learning rate.** `lr 1e-4` is the one knob b3
measured as peaked, and it came out worse on the very axis the consistency argument is about: 14.5% of
its post-competence evals below 80% against the reference's 12.2%, best30 95.0 against 96.6, **zero**
≥98%/500 rows against 6, and the latest competence onset in the sweep. Smaller lr bought slower
learning, not steadier learning. It is n=1, so it is not settled — but b7 is already resolving the
network axis, and the collapse is the larger unexplained effect.

**Both new knobs were smoke-tested before queueing**, because a silently-ignored knob costs four arms:
the anneal reads 0.0091 at 10% of a 0.01→0.001 run, which is the linear value, and `target_KL` at a
deliberate 0.001 stops the epoch loop after 1 of 8 epochs with `stopped_early=True`.

**Batch b4 closed on the desktop 2026-08-31**: pooled **7.3%** of stage-B rows at ≥98%/500 against
b6's 12.9% and b5's 9.6%, and best30 **97.0-97.9** against 97.8-98.5 for both. **The arm built from
b3's two best single knobs is the weakest of the three 8-seed batches** — shape and epochs interact
negatively, and a one-knob-at-a-time sweep licenses no stacking. That is the finding b7 was built on.
Its charts were imported and redrawn 2026-09-01, and a laptop `hof5000` pass re-measured its 274
≥98.5% candidates at 5,000 episodes: **one row clears the snek2 champion and none reaches 99%**
([`results.md`](results.md)). **‡ The b5-vs-b6 half of that comparison did not survive the same
re-measure** — the two are identical at 5,000 episodes and b5 is ahead on champion-level rows; see
[`findings.md`](findings.md).

**Before that — 2026-08-30 17:34 — three things closed and both boxes went idle.**

| what | where | outcome |
|---|---|---|
| **batch b6** stage B, 8 arms | laptop | done in 226.1 min, status 0. Pooled **12.8%** of rows ≥98%/500 |
| **batch b5** stage B, 8 arms | desktop | done in 222.6 min, status 0. Pooled **9.6%**; best row **100.0%/500** |
| **the parallelism sweep**, 11 waves | desktop | all 53 jobs done. **The eval side answered; the training side did not** |

Full per-arm numbers and the b5/b6 comparison are in [`results.md`](results.md); the charts are in
[`charts.md`](charts.md); the sweep is in [`findings.md`](findings.md).

**b6 leads b5 on the pooled headline but the sign test is a coin (5 of 8, p≈0.29→0.73), and rank 1 is
a tie at 98.5 best30.** More importantly the two batches differ in **two** knobs — `fc (200,100)` + 4
epochs against `fc (320,)` + 8 epochs — so this is not the network-shape test
[`results.md`](results.md) and this file have both been calling for. **b4 ran that test on 2026-08-31
and b7 is sweeping the axis outright** — see the top of this section.

### Next, in the order the evidence argues for

**The sweep after b8 is designed: [`../plans/hyperparam-sweep.md`](../plans/hyperparam-sweep.md)**
(2026-09-01) -- batches b9-b17, one knob each at four seeds on b7's `fc (320,)` base at 50M, with
b7aa-b7ad as the shared control; the machine-readable grid is `plans/hyperparam-sweep.json` and
`tools/sweep_specs.py` expands a batch into specs. Nothing from it is queued yet.

1. ~~**One batch varying only the network**, matched epochs and matched budget.~~ **Done: b7, closed
   2026-09-01.** `fc (320,)` won and b3's ranking inverted. What this opens, in order:
   **a champion attempt at `fc (320,)` + 4 epochs**, which no batch has yet run at length — b5 was
   `fc 320` at *8* epochs and b7 was 4 epochs at only 50M, and the two best cells of the grid have
   never been in the same arm; and the **DQN** half of the same sweep, since `dqn/net.py` takes the
   same `fc_layers` config, which would say whether the shape effect is PPO's or the task's.
2. **Re-confirm whatever b8 finds on `fc (320,)`.** b8 holds b4's `fc (200,100)` because b4 is its
   control, and b7 has since shown that shape to be the weaker one, so a stability knob that helps b8
   is a candidate rather than a result until it is re-run on the winning shape.
3. **Re-run the worker sweep with long waves.** 3.2-minute waves cannot resolve it — see
   [`findings.md`](findings.md). ~30 min per wave is what the earlier hand-measurement used.
4. **Sweep arm count**, which the queue cannot express: `_dispatch` takes pending jobs in priority
   order up to `max_trainers` regardless of priority value, so any group of 8+ trainer jobs launches
   as exactly 8. It needs a `runtime.json` commit between waves, or a real sweep job type.

### Next for PPO

**‡ Rewritten 2026-09-01, because b7 refuted the paragraph that was here.** It read "two hidden
layers beat every single-layer width tried", from b3's one-seed-each densities — `fc 300,100` 9.0%,
`fc 200,100` 7.9%, `fc 320` 5.6%, `fc 500` 3.2%, `fc 200` 0.8%. At four seeds and 50M those three
reverse: `fc 320` **17.3%**, `fc 200,100` 11.8%, `fc 300,100` 6.8%. What survives from the old
paragraph is the capacity half — width past 320 hurts, and `fc (400,200)` is now last of eight — and
the convenient fact that **`fc 320` is snek2's shape, carried across so a champion's weights convert,
and every batch in both eras has used it.** `dqn/net.py` takes the same `fc_layers` config, so the
same sweep is one batch away for DQN and has never been run — see [`findings.md`](findings.md).

**The follow-up wave, designed and not yet launched** — push the axis that moved rather than resample
the flat ones: epochs 12 and 16, minibatch 128, rollout 256, and `fc 200,100` + epochs 8 as the one
interaction worth a slot. Depth belongs to a "better agent" batch rather than to a seed-matched
comparison, which has to hold the network at 320 to stay matched against b2.

## The b2-era plan — superseded, kept for the reasoning

**Written 2026-08-29, before b3, b5, b6 and b4 ran.** Item 3 below describes a 4-arm b4 holding
`fc 320` fixed, which was never run: the `b4` name was reused on 2026-08-30 for the network-shape
test, and *that* is the b4 that closed on 2026-08-31. What survives here is why each item was
queued, which is worth more than the schedule was.

1. **Read b2 against b1 and against b29/b41/b47.** A b2-vs-b29 difference smaller than the
   b29-vs-b41 process-noise gap is noise, not a port regression — snek2 ran that config three times
   precisely to have the yardstick.
2. **Phase 6 — `ppo/`.** The reason snek3 exists, and the design is
   [`../plans/ppo.md`](../plans/ppo.md). **Phases 6a, 6b and 6c are all closed** — the algorithm seam
   is in `train.py` with three fixed-seed DQN arms byte-identical across it, `ppo/` is written and
   tested (122 fixtures, 14 of 14 mutants killed), and batch b3 has run 15 arms. Deployed to the
   desktop 2026-08-29 once b2's stage-B wave published. **6d — batch b4 — is next**, at **18M**
   transitions to match b2 (3M counted steps x 6 transitions per step; the plan's 12M was wrong).

   **‡ Two claims made from the 6b gate arm are withdrawn, and both were withdrawn by b3.** The gate
   arm was 508k transitions on snek3's *unshaped* defaults, and neither conclusion survived a shaped
   arm at 20x the budget:

   - **"PPO is behind DQN rather than beside it"** — withdrawn. Matched on transitions *and* on
     reward function, the two ranges top out at the same number (96.9 best30), and on the ≥98%/500
     count PPO is ~10x denser. The gate arm's gap was the reward function, not the algorithm.
   - **"`clip_fraction` 0.03 says the learning rate is low"** — falsified outright. Raising it to
     1e-3 and 3e-3 both made things *worse* (85.2 and 69.9 best30 at 3M, the latter at sd 18.4), and
     1e-4 was worse than 1e-3. The learning rate is peaked at the default and a low clip fraction did
     not mean what I read into it.

   The gate arm's chart stays at [`../runs/ppo-smoke.png`](../runs/ppo-smoke.png) as a record of
   what an unshaped PPO arm does. It is a gate arm, not a p-series arm.

3. **Batch b4 — the seed-matched gate batch.** 4 arms, seeds 1-4, **18M transitions**, b2's env
   config, **fc 320 held fixed** so the comparison is seed-matched against b1, b2, b29, b41 and b47.
   **b3 hands it the reference config unchanged**, because b3 found no winner — which makes b4 a
   cleaner comparison than the plan expected rather than a blocked one. Phase 3's ≥90% bar is already
   cleared by b3, so b4's job is the comparison, not the gate.

**The stage-A queue is next after b2 and the numbers are now measured rather than projected.** Stage A
is **66%** of an arm's 8.1 h (not 90%), and streaming recovers **3.3-3.4x** of it (not 5.7x) — see
[`findings.md`](findings.md). Cutting episodes does not work: 4x fewer buys 1.6x, because the cost is
lane drain. Every way of recovering it makes the epsilon schedule's feedback lag
([`invariants.md`](invariants.md) invariant 2), so the lag must be **bounded** rather than left to
float. b1 is the baseline any such change is measured against.

| an arm at 3M counted steps | training | stage A | total |
|---|---:|---:|---:|
| as b2 runs today | 2.79 h | 5.33 h | **8.1 h** |
| + the two bit-exact fixes (landed) | 2.33 h | 5.33 h | 7.7 h |
| + a bounded eval queue, 2 workers per 4 trainers | 2.33 h | ~0 | **2.3 h** |

The queue's arithmetic closes: 4 trainers at 299 st/s demand 1.20 checkpoints/s, and one streamed
worker supplies 0.54-0.89, so **two workers serve four trainers** — six processes on the desktop's 16
cores.

## Backlog

One line per idea, with a prior. A design that is settled enough to implement gets a file in
[`../plans/`](../plans/) and a row here.

| idea | prior |
|---|---|
| **PPO** | [`../plans/ppo.md`](../plans/ppo.md) — **phases 6a and 6b closed 2026-08-29; batch b3 is next.** No longer a backlog item. The reason snek3 exists. On-policy and wide, so it is the algorithm that actually exploits a 196k env-steps/s vectorised env, where DQN's replay ratio caps the loop at ~4,000 steps/s |
| **Batched or asynchronous self-eval** | **the next change. 8.1 h an arm becomes ~2.3 h, measured.** The win is keeping the lanes full, so a queue drained by streaming workers gets it; the drained shape is the whole cost and cutting episodes does not touch it. Cost is a lag on the epsilon schedule — **bound it**, do not let queue depth set it |
| **Replay ratio < 1** | ~~the only way past ~4,000 agent steps/s~~ **do not use this to reproduce snek2.** Ratio 1.0 already matches snek2's 1 gradient step per transition; lowering it makes snek3 *less* data-efficient than snek2 ever was. It remains a real dynamics knob, worth 2x at batch 512, but it is not a comparability fix — `SNEK_MAX_STEPS` is |
| **Drop observation indices 10/12/14** | ~1.5x on the observation build. Region enumeration is 33% of the connectivity cost and those three indices are its only consumers. Batch 45 reached 99% with them in, so this is a cost question |
| **Munchausen-DQN, SAC-discrete** | the discrete off-policy actor-critic options, if PPO underperforms. **TD3 does not apply** — it is continuous-action and this task has three discrete actions |

## How we got here — the closed phases

History, kept because each phase gate is a claim someone may want to re-check. Nothing here is
current state.

**Batch b2 — b29's record config on the torch stack, seeds 1-4, 3M steps. Closed 2026-08-29**, all
four arms and the stage-B wave `done`; results on the `results` branch, unread into
[`results.md`](results.md). Launched on the desktop 2026-08-29 08:09. This is the phase-3 gate re-run on the configuration snek2 actually set
records with; b1 ran snek3's bare defaults and that was the wrong batch to gate on.

| knob | snek3 default | b2 = b29 |
|---|---|---|
| `SNEK_IS_WEIGHTS` | 1 | **0** |
| `SNEK_TARGET_UPDATE_PERIOD` | 8 | **1000** |
| `SNEK_DISCOUNT` | 0.99 | **0.9975** |
| `SNEK_FOOD_DISTANCE_REWARD` | 0.001 | **0** |
| `SNEK_CHASE_SAFE_SHAPING` | 0.0 | **0.1** |
| `SNEK_CHASE_SAFE_GATE` | 85 | **75** |
| `SNEK_FC_LAYERS` | 320 | 320 |

**Five knobs differ, not the two b1's write-up first suggested.** The target-update period and the
discount are substantial algorithmic differences, and they were nearly missed by reading the results
summary instead of snek2's own b47 spec. Read the spec.

Seed N is pinned to arm letter N, so every arm is seed-matched against b29a-d, b41a-d, b47a-d **and**
snek3's own b1a-d. Budget ~7 h an arm; the desktop auto-queues one stage-B wave for the batch.

### ‡ Interim reading at 0.36-0.41M of 3M — the phase-3 gate is met; the lead over b47 was a units artefact

Read 2026-08-29 09:13, ~1 h in, at **106 counted steps/s an arm** (the 290 st/s in the log excludes the
self-eval; stage A is ~2/3 of the wall clock). ETA ~16:00. Both tables are 100-episode graph evals, so
b2 and b47 are on the same instrument and the counts are directly comparable.

**‡ Corrected 2026-08-29: a snek3 counted step is four game moves and a snek2 step was one.** See
[`findings.md`](findings.md). Every b2 step number below is 4x a b47 step number in game moves,
buffer rows and gradient steps alike, so **the "b2 leads b47 on every seed" headline compares b2 at
1.4-1.6M transitions against b47 at 0.34-0.39M.** The b47 column is truncated to the same *counter*
value, which is not the same work. Data efficiency is identical in both eras (1 gradient step per
transition), so nothing here is a learning-rate difference — it is a budget difference.

| seed | step | b2 ≥95 | b2 ≥98 | b2 best30 | b47 ≥95 | b47 ≥98 | b47 best30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.36M | 0 | 0 | 85.9 | 0 | 0 | 34.1 |
| 2 | 0.37M | 7 | 1 | 92.7 | 0 | 0 | 67.0 |
| 3 | 0.34M | 0 | 0 | 81.0 | 0 | 0 | 55.0 |
| 4 | 0.39M | **52** | **17** | **96.8** | 0 | 0 | 68.3 |

b47 at its own furthest point — 1.38-1.63M, where snek2 froze mid-batch — had best30 82.2 / 90.5 /
**96.0** / 83.6 and ≥95 counts 0 / 8 / 173 / 0.

**‡ "b2d at 0.39M has already passed b47c's best30 at 1.63M" is withdrawn.** b2d at 0.39M counted
steps had done **1.56M** transitions and 1.56M gradient steps; b47c at 1.63M had done 1.63M and 1.63M.
The two are at matched values on both axes, so the correct statement is **b2d matches b47c at matched
work**, not that it beats it at a quarter of the budget. The matched-work comparison is the one to
make from here: read b2 at 4x b47's step number.

**The phase-3 gate (≥90% perfect) is met on both readings and by two arms.** b2d's trailing-30 perfect
rate peaks at 96.8% and it has 52 single evals at ≥95/100 including a 100; b2b is at 92.7% with 7. b1
never reached one such eval in 3M steps, so the five knobs are the whole difference — which is snek2's
own batch-28/29 finding, reproduced.

**‡ Same shape as b47, different carrier.** One arm carries the batch and two produce nothing ≥95, as
in b47 (carrier seed 3) and b41 (also seed 3) and b29 (seed 2). b2's carrier is seed 4, so it is now 1
of 4 for every seed and the carrier is a coin, not a property of the seed. **Do not read b2a or b2c as
a regression**: snek2 saw the same on the same config.

**What is not yet answered is phase 5**, which needs a ≥98%/500 *region* from the stage-B wave rather
than these 100-episode rows. b2d's 17 rows ≥98/100 and 7 ≥99/100 make a region plausible and not
assured — snek2's winner's-curse drops on this instrument were −2.6, −3.2 and −6.2 pp. And the snek2
side of that comparison is **b29's own close-out, not b47's**: b47 was frozen at 69-81% of its 2M cap
and never closed out, so no b47 ≥98%/500 data exists.

**b1-vs-b2 is unaffected by the correction** — both arms ran the same collector at the same ratio, so
they share the 4x and their step axes are directly comparable. "The five knobs are the whole
difference" stands.

**b1 is closed** ([`results.md`](results.md)): four arms at 3M, peak perfect 42.1 / 58.3 / 56.7 /
81.9%, **no checkpoint anywhere at 95/100**, every arm still climbing at its cap. Its stage-B wave
has run and published 0 rows an arm, which is the honest measurement rather than a failure.

**Phases 0-4 are closed and batch b1 has run.** `env/`, `vectorized/`, the measurement engine,
checkpoint I/O, the eval wave, the charts, the viewers, `dqn/`, `train.py` and the desktop daemon are
all in, and the box runs snek3 rather than snek2. **The phase-3 gate is still open** — see Now.

**Phase 0 — the two env implementations agree.** 36,000 states × 30 observation indices, **0
mismatches**, across a growth regime (24,000 states, 49 episodes, lengths to 60) and a coiled endgame
regime (12,000 states, 280 episodes, 26 perfect games), with rewards, terminations, both shaping
terms and the win path in parity too. **17 of 17 hand-made mutants killed.**

**Phase 1 — the snek2 champion plays in torch.** `b44a-lowlr7-b29b-ckpt2739000` converted and
measured **98.8% perfect over 3,000 episodes** against snek2's 98.73%, inside the ±0.6 pp gate. The
conversion itself is exact rather than close: on 12,864 states the two networks' Q-values differ by
at most 2.7e-5 on values of magnitude ~30.6, and the **argmax is identical on every state**, so the
policies are the same function. `watch.py` plays it and `record_gif.py` records it.

**Phase 2 — the flat protocol reproduces the tiered one.** All **3,222** checkpoints of
`b45a-lowlr8-b29b` converted and measured, against snek2's own close-out: mean per-row difference
**−0.004 pp** on a 0.041 pp standard error, and observed spread / predicted spread **1.00**. 14
minutes on four shards. Three findings came out of it, including a **5.7x correction to the cost of
stage A** — see [`findings.md`](findings.md).
