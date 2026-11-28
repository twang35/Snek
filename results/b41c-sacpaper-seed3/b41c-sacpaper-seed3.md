# b41c-sacpaper-seed3

step **789,000** · 789 evals · trailing **0.05** · peak **24.11** @1,000 · sef **0.0** · best30 **0.0** @789,000

**Below threshold since step 133,000** — 656,000 steps ago. Not a verdict; arms have recovered from longer.

## Config

| | |
|---|---|
| algo | sac |
| collect_envs | 16 |
| discount | 0.99 |
| eval_interval | 1000 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| init_from | None |
| max_steps | 3125000 |
| min_checkpoint_score | 40.0 |
| sac_adam_epsilon | 1e-08 |
| sac_alpha | auto |
| sac_alpha_learning_rate | 0.0003 |
| sac_batch_size | 64 |
| sac_critic_combine | min |
| sac_critic_learning_rate | 0.0003 |
| sac_entropy_penalty | 0.0 |
| sac_init_alpha | 1.0 |
| sac_learning_rate | 0.0003 |
| sac_n_step | 1 |
| sac_prefill | 20000 |
| sac_priority_exponent | 0.0 |
| sac_q_clip | 0.0 |
| sac_replay_buffer_max_length | 1000000 |
| sac_replay_ratio | 0.25 |
| sac_target_entropy | 1.07664 |
| sac_target_entropy_ratio | 0.98 |
| sac_target_update_period | 2000 |
| sac_tau | 1.0 |
| seed | 3 |
| torch_threads | 1 |

![b41c-sacpaper-seed3](b41c-sacpaper-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 46.88 | 24.11 | 0.0 | 77.0 | 43.103 | 0.0 |  |
| 2000 | 29.99 | 16.31 | 0.0 | 73.0 | 27.652 | 0.0 |  |
| 3000 | 2.91 | 17.04 | 0.0 | 39.0 | 2.326 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 778000 | 0.03 | 0.04 | 0.0 | 1.0 | -4.973 | 0.0 |  |
| 779000 | 0.05 | 0.04 | 0.0 | 1.0 | -4.953 | 0.0 |  |
| 780000 | 0.07 | 0.04 | 0.0 | 1.0 | -4.933 | 0.0 |  |
| 781000 | 0.09 | 0.05 | 0.0 | 1.0 | -4.913 | 0.0 |  |
| 782000 | 0.01 | 0.04 | 0.0 | 1.0 | -4.993 | 0.0 |  |
| 783000 | 0.04 | 0.04 | 0.0 | 1.0 | -4.963 | 0.0 |  |
| 784000 | 0.06 | 0.05 | 0.0 | 1.0 | -4.943 | 0.0 |  |
| 785000 | 0.06 | 0.05 | 0.0 | 2.0 | -4.943 | 0.0 |  |
| 786000 | 0.04 | 0.05 | 0.0 | 1.0 | -4.963 | 0.0 |  |
| 787000 | 0.04 | 0.04 | 0.0 | 1.0 | -4.963 | 0.0 |  |
| 788000 | 0.05 | 0.05 | 0.0 | 2.0 | -4.953 | 0.0 |  |
| 789000 | 0.08 | 0.05 | 0.0 | 2.0 | -4.923 | 0.0 |  |
