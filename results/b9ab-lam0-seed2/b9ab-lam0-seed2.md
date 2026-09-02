# b9ab-lam0-seed2

step **50,003,968** · 3052 evals · trailing **66.37** · peak **93.73** @21,741,568 · sef **11.4** · best30 **85.1** @28,868,608

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
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.0 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 1.0 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 2 |
| torch_threads | 1 |

![b9ab-lam0-seed2](b9ab-lam0-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.12 | 1.12 | 0.0 | 8.0 | 0.575 | 0.0 |  |
| 32768 | 15.54 | 8.33 | 0.0 | 42.0 | 12.655 | 0.0 |  |
| 49152 | 16.35 | 20.64 | 1.0 | 63.0 | 11.845 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 64.73 | 66.69 | 43.0 | 95.0 | 61.81 | 2.0 |  |
| 49840128 | 68.99 | 66.83 | 41.0 | 95.0 | 67.2 | 3.0 |  |
| 49856512 | 64.03 | 66.32 | 33.0 | 95.0 | 60.115 | 1.0 |  |
| 49872896 | 62.69 | 66.24 | 28.0 | 95.0 | 59.86 | 2.0 |  |
| 49889280 | 64.14 | 66.24 | 37.0 | 95.0 | 60.225 | 1.0 |  |
| 49905664 | 66.15 | 66.32 | 38.0 | 95.0 | 62.19 | 1.0 |  |
| 49922048 | 67.09 | 66.31 | 41.0 | 95.0 | 66.25 | 4.0 |  |
| 49938432 | 66.41 | 66.33 | 33.0 | 95.0 | 64.665 | 3.0 |  |
| 49954816 | 63.68 | 66.49 | 33.0 | 95.0 | 61.845 | 3.0 |  |
| 49971200 | 65.85 | 66.67 | 37.0 | 95.0 | 63.97 | 3.0 |  |
| 49987584 | 64.49 | 66.28 | 34.0 | 95.0 | 62.61 | 3.0 |  |
| 50003968 | 61.49 | 66.37 | 33.0 | 80.0 | 56.58 | 0.0 |  |
