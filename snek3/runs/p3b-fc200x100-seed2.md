# p3b-fc200x100-seed2

step **61,702,144** · 3759 evals · trailing **93.34** · peak **94.55** @51,150,848 · sef **94.6** · best30 **97.4** @48,201,728

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
| seed | 2 |
| torch_threads | 1 |

![p3b-fc200x100-seed2](p3b-fc200x100-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 10.36 | 10.36 | 2.0 | 19.0 | 5.36 | 0.0 |  |
| 32768 | 29.46 | 22.72 | 8.0 | 53.0 | 24.46 | 0.0 |  |
| 49152 | 36.41 | 26.14 | 8.0 | 62.0 | 31.41 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 61407232 | 94.0 | 93.39 | 7.0 | 95.0 | 190.965 | 98.0 |  |
| 61423616 | 93.43 | 93.3 | 62.0 | 95.0 | 183.385 | 91.0 |  |
| 61440000 | 93.09 | 93.34 | 15.0 | 95.0 | 187.07 | 95.0 |  |
| 61456384 | 93.56 | 93.12 | 18.0 | 95.0 | 187.54 | 95.0 |  |
| 61472768 | 92.4 | 93.11 | 18.0 | 95.0 | 182.445 | 91.0 |  |
| 61489152 | 93.73 | 93.12 | 73.0 | 95.0 | 181.695 | 89.0 |  |
| 61505536 | 93.54 | 93.28 | 71.0 | 95.0 | 180.6 | 88.0 |  |
| 61521920 | 91.79 | 93.35 | 15.0 | 95.0 | 174.825 | 84.0 |  |
| 61571072 | 94.36 | 93.2 | 76.0 | 95.0 | 186.305 | 93.0 |  |
| 61587456 | 93.82 | 93.32 | 73.0 | 95.0 | 181.83 | 89.0 |  |
| 61603840 | 92.93 | 93.3 | 71.0 | 95.0 | 176.915 | 85.0 |  |
| 61702144 | 94.99 | 93.34 | 94.0 | 95.0 | 192.95 | 99.0 |  |
