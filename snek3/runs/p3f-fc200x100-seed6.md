# p3f-fc200x100-seed6

step **57,753,600** · 3519 evals · trailing **94.04** · peak **94.48** @42,860,544 · sef **90.0** · best30 **97.5** @45,596,672

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 16384 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 6 |
| fc_layers | (200, 100) |
| graph_eval_episodes | 100 |
| max_steps | 400000000 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 6 |
| torch_threads | 1 |

![p3f-fc200x100-seed6](p3f-fc200x100-seed6.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 12.57 | 12.57 | 1.0 | 44.0 | 8.335 | 0.0 |  |
| 32768 | 20.39 | 16.48 | 1.0 | 42.0 | 15.435 | 0.0 |  |
| 49152 | 28.49 | 20.48 | 9.0 | 49.0 | 23.535 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 57475072 | 94.45 | 93.35 | 75.0 | 95.0 | 188.475 | 95.0 |  |
| 57491456 | 91.33 | 93.87 | 9.0 | 95.0 | 180.38 | 90.0 |  |
| 57507840 | 94.08 | 93.13 | 10.0 | 95.0 | 191.09 | 98.0 |  |
| 57524224 | 94.43 | 93.6 | 60.0 | 95.0 | 189.45 | 96.0 |  |
| 57540608 | 94.95 | 93.91 | 90.0 | 95.0 | 192.955 | 99.0 |  |
| 57556992 | 93.65 | 93.91 | 4.0 | 95.0 | 187.675 | 95.0 |  |
| 57573376 | 94.9 | 94.08 | 88.0 | 95.0 | 191.91 | 98.0 |  |
| 57589760 | 94.88 | 94.13 | 83.0 | 95.0 | 192.885 | 99.0 |  |
| 57622528 | 94.74 | 93.8 | 80.0 | 95.0 | 191.75 | 98.0 |  |
| 57638912 | 93.95 | 93.69 | 31.0 | 95.0 | 188.925 | 96.0 |  |
| 57655296 | 94.72 | 94.02 | 84.0 | 95.0 | 190.735 | 97.0 |  |
| 57753600 | 94.94 | 94.04 | 89.0 | 95.0 | 192.945 | 99.0 |  |
