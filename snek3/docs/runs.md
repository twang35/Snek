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

**Nothing is running.** Batch b1 closed 2026-08-29: all four seeds ran their full 3M steps on
schedule, and the desktop published cleanly. The numbers are in [`results.md`](results.md).

| | |
|---|---|
| laptop | idle |
| desktop `the-claw-den` | idle. `snek3-runner` owns the box; `b1-stageb` is `failed` in its ledger |

**The phase-3 gate is not met and b1 cannot settle it.** Peak trailing perfect rate by seed: 42.1 /
58.3 / 56.7 / **81.9%**, against a gate of 90. But **no arm had plateaued** — every one was still
climbing at its cap — and **no checkpoint in any arm reached 95/100**, so stage B had nothing to
select even before its command failed. Read b1 as a rate-of-climb measurement, not a verdict.

Two defects to clear before the next batch, in this order:

1. **`launch.py` builds a stage-B command `evaluate.py` cannot parse** — it passes several policies
   and `--selector`, and the real signature is `evaluate.py <policy> [selector]`. Every wave exits 2.
   Fix, and add the fixture that hands the built argv to the real parser
   ([`findings.md`](findings.md)).
2. **The phase-3 gate needs to name its number.** "≥90% perfect" is a trailing rate at snek2's
   ceiling or a single checkpoint b1d already cleared, depending on how it is read.

## Next, in order

1. **Re-run the gate on the b29/b47-class config**, not on bare defaults: chase-safe shaping
   `c=0.10` at **gate 75**, IS **off**, fc 320. snek2's own batch 28-29 result is that the gate is
   the lever, and b1's defaults have shaping at 0.0 with IS on. Give it **6M steps** — b1d was still
   rising at 3M — or accept that a 3M arm measures the climb.
2. **Phase 6 — `ppo/`.** The reason snek3 exists.

**Decide the stage-A batching question before phase 6, not after.** Stage A at 16.9 ep/s is ~90% of
an arm's wall clock and the 5.7x is recoverable, but every way of recovering it makes the epsilon
schedule's feedback lag ([`invariants.md`](invariants.md) invariant 2). b1 is the baseline any such
change is measured against, and a 6M-step arm makes the cost twice as visible.

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
