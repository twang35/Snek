# Runs — current state and forward plan

**Nothing is running.** snek3 is in phase 0 of
[`../plans/pytorch-port.md`](../plans/pytorch-port.md): the environment and the measurement engine,
with no learning code yet.

## Now

| | |
|---|---|
| laptop | idle |
| desktop `the-claw-den` | idle for snek3. snek2's daemon still owns the box until the snek3 runner is deployed (phase 4) |

## Next, in order

The phase table in [`../plans/pytorch-port.md`](../plans/pytorch-port.md) §10 is the plan; this is
only what is immediately next.

1. **Phase 1 — the champion transfer.** Convert `snek2/hallOfFame/b44a-lowlr7-b29b-ckpt2739000` to a
   torch `state_dict` and measure it. Gate: **98.7% ± 0.6 pp over 3,000 episodes.** This validates
   the env, the observation, the policy path and the eval engine with no training code at all.
2. **Phase 2 — the eval wave**, then a 3,222-row A/B against snek2's own
   `b45a-lowlr8-b29b_checkpoint_evals_vec.json`.
3. **Phase 3 — `dqn/`.** DDQN + PER + the epsilon schedule + the forking collector + the shield.

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
