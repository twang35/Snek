# Runs — current state and forward plan

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

## Now

**Batch b2 — b29's record config on the torch stack, seeds 1-4, 3M steps.** All four on the desktop,
launched 2026-08-29 08:09. This is the phase-3 gate re-run on the configuration snek2 actually set
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

## Next, in order

1. **Read b2 against b1 and against b29/b41/b47.** A b2-vs-b29 difference smaller than the
   b29-vs-b41 process-noise gap is noise, not a port regression — snek2 ran that config three times
   precisely to have the yardstick.
2. **Phase 6 — `ppo/`.** The reason snek3 exists. The design is drafted in
   [`../plans/ppo.md`](../plans/ppo.md) and is **not approved yet**.

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
| **PPO** | [`../plans/ppo.md`](../plans/ppo.md) — drafted 2026-08-29, awaiting review. The reason snek3 exists. On-policy and wide, so it is the algorithm that actually exploits a 196k env-steps/s vectorised env, where DQN's replay ratio caps the loop at ~4,000 steps/s |
| **Batched or asynchronous self-eval** | **the next change. 8.1 h an arm becomes ~2.3 h, measured.** The win is keeping the lanes full, so a queue drained by streaming workers gets it; the drained shape is the whole cost and cutting episodes does not touch it. Cost is a lag on the epsilon schedule — **bound it**, do not let queue depth set it |
| **Replay ratio < 1** | ~~the only way past ~4,000 agent steps/s~~ **do not use this to reproduce snek2.** Ratio 1.0 already matches snek2's 1 gradient step per transition; lowering it makes snek3 *less* data-efficient than snek2 ever was. It remains a real dynamics knob, worth 2x at batch 512, but it is not a comparability fix — `SNEK_MAX_STEPS` is |
| **Drop observation indices 10/12/14** | ~1.5x on the observation build. Region enumeration is 33% of the connectivity cost and those three indices are its only consumers. Batch 45 reached 99% with them in, so this is a cost question |
| **Munchausen-DQN, SAC-discrete** | the discrete off-policy actor-critic options, if PPO underperforms. **TD3 does not apply** — it is continuous-action and this task has three discrete actions |
