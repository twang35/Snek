# Runs — current state and forward plan

**Nothing is running.** snek3 closed **phases 0 and 1** of
[`../plans/pytorch-port.md`](../plans/pytorch-port.md) on 2026-08-28. `env/`, `vectorized/`, the
measurement engine, checkpoint I/O and the two viewers are in; there is still no learning code.

**Phase 0 — the two env implementations agree.** 36,000 states × 30 observation indices, **0
mismatches**, across a growth regime (24,000 states, 49 episodes, lengths to 60) and a coiled endgame
regime (12,000 states, 280 episodes, 26 perfect games), with rewards, terminations, both shaping
terms and the win path in parity too. **17 of 17 hand-made mutants killed.**

**Phase 1 — the snek2 champion plays in torch.** `b44a-lowlr7-b29b-ckpt2739000` converted and
measured **98.8% perfect over 3,000 episodes** against snek2's 98.73%, inside the ±0.6 pp gate. The
conversion itself is exact rather than close: on 12,864 states the two networks' Q-values differ by
at most 2.7e-5 on values of magnitude ~30.6, and the **argmax is identical on every state**, so the
policies are the same function. `watch.py` plays it and `record_gif.py` records it.

## Now

| | |
|---|---|
| laptop | idle |
| desktop `the-claw-den` | idle for snek3. snek2's daemon still owns the box until the snek3 runner is deployed (phase 4) |

## Next, in order

The phase table in [`../plans/pytorch-port.md`](../plans/pytorch-port.md) §10 is the plan; this is
only what is immediately next.

1. **Phase 2 — the eval wave.** `vectorized/shard.py`, `vectorized/wave.py`, the `steps:<file>`
   selector, `run_report`, `arch` and the charts. Then a **3,222-row A/B** against snek2's own
   `b45a-lowlr8-b29b_checkpoint_evals_vec.json`, converting every checkpoint of that arm. ~2 h of
   compute, and the only way to know the flat protocol agrees with the tiered one it replaces.
2. **Phase 3 — `dqn/`.** DDQN + PER + the epsilon schedule + the forking collector + the shield.
   Gate: one arm at ≥90% perfect, ≥1,500 agent steps/s with the self-eval off.
3. **Phase 4 — `desktop/`.**

## Backlog

One line per idea, with a prior. A design that is settled enough to implement gets a file in
[`../plans/`](../plans/) and a row here.

| idea | prior |
|---|---|
| **PPO** | the reason snek3 exists. On-policy and wide, so it is the algorithm that actually exploits a 196k env-steps/s vectorised env, where DQN's replay ratio caps the loop at ~4,000 steps/s |
| **Asynchronous self-eval** | the next 8x on training wall clock: ~2 h an arm becomes ~20-30 min. Cost is a one-interval lag on the epsilon schedule, which is a feedback loop rather than a readout |
| **Replay ratio < 1** | the only way past ~4,000 agent steps/s for DQN, and a learning-dynamics change rather than a free win. Pre-register it |
| **Drop observation indices 10/12/14** | ~1.5x on the observation build. Region enumeration is 33% of the connectivity cost and those three indices are its only consumers. Batch 45 reached 99% with them in, so this is a cost question |
| **Munchausen-DQN, SAC-discrete** | the discrete off-policy actor-critic options, if PPO underperforms. **TD3 does not apply** — it is continuous-action and this task has three discrete actions |
