# b10aq-g91-seed1

step **50,003,968** · 3052 evals · trailing **91.71** · peak **94.1** @32,309,248 · sef **9.4** · best30 **85.5** @30,441,472

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
| seed | 1 |
| torch_threads | 1 |

![b10aq-g91-seed1](b10aq-g91-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 4.87 | 4.87 | 0.0 | 23.0 | 4.37 | 0.0 |  |
| 32768 | 32.92 | 43.2 | 0.0 | 83.0 | 30.845 | 0.0 |  |
| 49152 | 66.94 | 38.81 | 24.0 | 95.0 | 64.51 | 1.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.96 | 91.19 | 35.0 | 95.0 | 170.025 | 78.0 |  |
| 49840128 | 93.32 | 90.99 | 23.0 | 95.0 | 175.315 | 83.0 |  |
| 49856512 | 94.26 | 91.0 | 84.0 | 95.0 | 175.35 | 82.0 |  |
| 49872896 | 93.55 | 90.94 | 20.0 | 95.0 | 172.605 | 80.0 |  |
| 49889280 | 93.36 | 90.91 | 18.0 | 95.0 | 167.485 | 75.0 |  |
| 49905664 | 94.71 | 91.42 | 90.0 | 95.0 | 184.755 | 91.0 |  |
| 49922048 | 94.08 | 91.85 | 75.0 | 95.0 | 168.205 | 75.0 |  |
| 49938432 | 94.31 | 92.07 | 71.0 | 95.0 | 177.39 | 84.0 |  |
| 49954816 | 93.83 | 91.17 | 13.0 | 95.0 | 179.85 | 87.0 |  |
| 49971200 | 94.47 | 91.06 | 90.0 | 95.0 | 177.55 | 84.0 |  |
| 49987584 | 94.51 | 91.63 | 86.0 | 95.0 | 178.585 | 85.0 |  |
| 50003968 | 93.06 | 91.71 | 61.0 | 95.0 | 169.175 | 77.0 |  |
