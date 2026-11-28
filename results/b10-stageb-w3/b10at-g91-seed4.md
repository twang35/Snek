# b10at-g91-seed4

step **50,003,968** · 3052 evals · trailing **92.12** · peak **93.78** @32,620,544 · sef **7.3** · best30 **84.7** @32,669,696

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
| seed | 4 |
| torch_threads | 1 |

![b10at-g91-seed4](b10at-g91-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.84 | 2.84 | 0.0 | 9.0 | 0.81 | 0.0 |  |
| 32768 | 2.91 | 2.88 | 0.0 | 17.0 | 2.365 | 0.0 |  |
| 49152 | 29.45 | 25.79 | 1.0 | 86.0 | 26.52 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.97 | 92.4 | 53.0 | 95.0 | 160.13 | 68.0 |  |
| 49840128 | 93.23 | 92.42 | 67.0 | 95.0 | 164.37 | 72.0 |  |
| 49856512 | 91.19 | 92.48 | 8.0 | 95.0 | 167.305 | 77.0 |  |
| 49872896 | 93.98 | 92.17 | 74.0 | 95.0 | 173.08 | 80.0 |  |
| 49889280 | 93.48 | 92.18 | 59.0 | 95.0 | 163.625 | 71.0 |  |
| 49905664 | 92.16 | 92.17 | 8.0 | 95.0 | 158.325 | 67.0 |  |
| 49922048 | 93.26 | 92.23 | 37.0 | 95.0 | 157.435 | 65.0 |  |
| 49938432 | 93.99 | 92.2 | 86.0 | 95.0 | 163.14 | 70.0 |  |
| 49954816 | 93.42 | 92.22 | 61.0 | 95.0 | 156.6 | 64.0 |  |
| 49971200 | 91.65 | 92.09 | 48.0 | 95.0 | 142.89 | 52.0 |  |
| 49987584 | 91.17 | 92.09 | 43.0 | 95.0 | 140.42 | 50.0 |  |
| 50003968 | 92.57 | 92.12 | 41.0 | 95.0 | 152.765 | 61.0 |  |
