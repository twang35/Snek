# b10as-g91-seed3

step **50,003,968** · 3052 evals · trailing **90.88** · peak **93.68** @14,041,088 · sef **4.1** · best30 **77.5** @37,109,760

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.91 |
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
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 9.2 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b10as-g91-seed3](b10as-g91-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.02 | 0.02 | 0.0 | 1.0 | -0.57 | 0.0 |  |
| 32768 | 1.78 | 0.9 | 0.0 | 7.0 | 1.28 | 0.0 |  |
| 49152 | 13.35 | 9.67 | 0.0 | 32.0 | 10.87 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.55 | 90.94 | 63.0 | 95.0 | 155.73 | 64.0 |  |
| 49840128 | 92.68 | 90.82 | 66.0 | 95.0 | 153.87 | 62.0 |  |
| 49856512 | 90.92 | 90.84 | 48.0 | 95.0 | 151.115 | 61.0 |  |
| 49872896 | 92.39 | 90.92 | 48.0 | 95.0 | 155.525 | 64.0 |  |
| 49889280 | 89.31 | 90.95 | 42.0 | 95.0 | 139.555 | 51.0 |  |
| 49905664 | 91.16 | 90.95 | 61.0 | 95.0 | 144.39 | 54.0 |  |
| 49922048 | 88.64 | 90.85 | 27.0 | 95.0 | 135.9 | 48.0 |  |
| 49938432 | 89.11 | 90.76 | 46.0 | 95.0 | 127.37 | 39.0 |  |
| 49954816 | 88.32 | 90.75 | 36.0 | 95.0 | 127.62 | 40.0 |  |
| 49971200 | 88.98 | 90.78 | 41.0 | 95.0 | 117.29 | 29.0 |  |
| 49987584 | 90.64 | 90.85 | 65.0 | 95.0 | 125.96 | 36.0 |  |
| 50003968 | 89.3 | 90.88 | 57.0 | 95.0 | 123.625 | 35.0 |  |
