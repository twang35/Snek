# b13af-mb64-seed2

step **50,003,968** · 3052 evals · trailing **92.68** · peak **94.42** @44,990,464 · sef **91.3** · best30 **97.6** @44,924,928

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 16384 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 50003968 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_anneal_fraction | 1.0 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 64 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 2 |
| torch_threads | 1 |

![b13af-mb64-seed2](b13af-mb64-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.39 | 1.39 | 0.0 | 4.0 | -1.72 | 0.0 |  |
| 32768 | 13.8 | 7.6 | 5.0 | 28.0 | 8.8 | 0.0 |  |
| 49152 | 30.65 | 15.28 | 13.0 | 56.0 | 25.65 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.9 | 92.51 | 55.0 | 95.0 | 188.785 | 96.0 |  |
| 49840128 | 93.01 | 92.58 | 41.0 | 95.0 | 181.79 | 90.0 |  |
| 49856512 | 94.1 | 92.51 | 44.0 | 95.0 | 188.94 | 96.0 |  |
| 49872896 | 93.08 | 92.28 | 19.0 | 95.0 | 183.85 | 92.0 |  |
| 49889280 | 84.1 | 92.31 | 0.0 | 95.0 | 168.04 | 85.0 |  |
| 49905664 | 92.01 | 92.34 | 4.0 | 95.0 | 179.795 | 89.0 |  |
| 49922048 | 93.61 | 92.71 | 14.0 | 95.0 | 186.55 | 94.0 |  |
| 49938432 | 94.16 | 92.73 | 46.0 | 95.0 | 189.0 | 96.0 |  |
| 49954816 | 93.83 | 92.55 | 64.0 | 95.0 | 185.595 | 93.0 |  |
| 49971200 | 94.34 | 92.34 | 73.0 | 95.0 | 187.19 | 94.0 |  |
| 49987584 | 92.12 | 92.66 | 12.0 | 95.0 | 182.935 | 92.0 |  |
| 50003968 | 94.07 | 92.68 | 71.0 | 95.0 | 184.795 | 92.0 |  |
