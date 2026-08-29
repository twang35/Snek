# Runs — current state and forward plan

**Batch b1 is running on both hosts.** snek3 closed **phases 0 through 4** of
[`../plans/pytorch-port.md`](../plans/pytorch-port.md) on 2026-08-28: `env/`, `vectorized/`, the
measurement engine, checkpoint I/O, the eval wave, the charts, the viewers, `dqn/`, `train.py` and
the desktop daemon are all in, and the box now runs snek3 rather than snek2.

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

**Batch b1 — the DDQN baseline at every default, 3M steps, only `SNEK_SEED` varying.** One batch
serves two gates: phase 3 wants any arm at **90% perfect**, phase 5 wants a four-seed region
comparison against snek2's b47 class.

| arm | seed | host | launched |
|---|---:|---|---|
| `b1a-baseline-seed1` | 1 | laptop | 2026-08-28 23:26 |
| `b1b-baseline-seed2` | 2 | desktop | 2026-08-28 23:45 |
| `b1c-baseline-seed3` | 3 | desktop | 2026-08-28 23:45 |
| `b1d-baseline-seed4` | 4 | desktop | 2026-08-28 23:45 |

Split across both hosts deliberately, so the four arms run **in parallel** rather than serialised
behind the desktop's wave barrier. Budget **~7 h** each with stage A on. The desktop auto-queues
stage B for its three when they finish; `b1a`'s has to be launched by hand or its checkpoints
rsync'd across, since the git bus carries specs and not weights.

**`SNEK_COLLECT_ENVS` is at its default 1 (4 lanes with the fork branches) on all four.** 16 lanes
measures 1.9x faster, but raising it on only the three desktop arms would confound the seed
comparison, and the hours are in stage A anyway — a 2x training speed-up moves an arm 8%. It waits
for a batch where all four arms can take it.

## Next, in order

The phase table in [`../plans/pytorch-port.md`](../plans/pytorch-port.md) §10 is the plan; this is
only what is immediately next.

1. **Close b1.** Any arm at 90% closes the phase-3 gate; all four together are phase 5's comparison.
   Stage B on every checkpoint that screened at ≥95/100.
2. **Phase 6 — `ppo/`.** The reason snek3 exists.

**Decide the stage-A batching question before phase 6, not after.** Stage A at 16.9 ep/s is ~90% of
an arm's wall clock and the 5.7x is recoverable, but every way of recovering it makes the epsilon
schedule's feedback lag ([`invariants.md`](invariants.md) invariant 2). b1 is the baseline any such
change has to be measured against.

## Backlog

One line per idea, with a prior. A design that is settled enough to implement gets a file in
[`../plans/`](../plans/) and a row here.

| idea | prior |
|---|---|
| **PPO** | the reason snek3 exists. On-policy and wide, so it is the algorithm that actually exploits a 196k env-steps/s vectorised env, where DQN's replay ratio caps the loop at ~4,000 steps/s |
| **Batched or asynchronous self-eval** | **worth 5.7x on an arm, and measured, not projected.** ~5 h an arm becomes ~1 h. The win is keeping the lanes full rather than overlapping with training, so batching K checkpoints into one `measure_stream` call gets nearly all of it. Cost either way is a K-interval lag on the epsilon schedule, which is a feedback loop rather than a readout — pre-register it |
| **Replay ratio < 1** | the only way past ~4,000 agent steps/s for DQN, and a learning-dynamics change rather than a free win. Pre-register it |
| **Drop observation indices 10/12/14** | ~1.5x on the observation build. Region enumeration is 33% of the connectivity cost and those three indices are its only consumers. Batch 45 reached 99% with them in, so this is a cost question |
| **Munchausen-DQN, SAC-discrete** | the discrete off-policy actor-critic options, if PPO underperforms. **TD3 does not apply** — it is continuous-action and this task has three discrete actions |
