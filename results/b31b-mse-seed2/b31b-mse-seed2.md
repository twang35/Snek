# b31b-mse-seed2

step **100,007,936** · 3052 evals · trailing **94.55** · peak **94.8** @85,491,712 · sef **97.4** · best30 **99.7** @85,491,712

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 32768 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 100007936 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_anneal_fraction | 0.5 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_discount_final | 0.999 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | 0.001 |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.95 |
| ppo_gae_lambda_final | 0.999 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 16.8 |
| ppo_horizon_final | 500.3 |
| ppo_learning_rate | 0.00025 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 512 |
| ppo_normalize_adv | True |
| ppo_rollout | 256 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 32768 |
| ppo_value_loss | mse |
| ppo_vf_coef | 0.5 |
| seed | 2 |
| torch_threads | 1 |

![b31b-mse-seed2](b31b-mse-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 13.7 | 13.7 | 0.0 | 27.0 | 8.858 | 0.0 |  |
| 65536 | 33.83 | 28.0 | 5.0 | 64.0 | 29.04 | 0.0 |  |
| 98304 | 34.68 | 29.33 | 1.0 | 59.0 | 29.644 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 94.55 | 94.59 | 50.0 | 95.0 | 192.329 | 99.0 |  |
| 99680256 | 95.0 | 94.64 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 99713024 | 94.71 | 94.63 | 66.0 | 95.0 | 192.438 | 99.0 |  |
| 99745792 | 94.37 | 94.61 | 32.0 | 95.0 | 192.105 | 99.0 |  |
| 99778560 | 94.36 | 94.62 | 31.0 | 95.0 | 192.08 | 99.0 |  |
| 99811328 | 94.99 | 94.63 | 94.0 | 95.0 | 192.719 | 99.0 |  |
| 99844096 | 93.74 | 94.63 | 13.0 | 95.0 | 190.422 | 98.0 |  |
| 99876864 | 94.33 | 94.65 | 28.0 | 95.0 | 192.095 | 99.0 |  |
| 99909632 | 93.75 | 94.67 | 30.0 | 95.0 | 190.445 | 98.0 |  |
| 99942400 | 94.49 | 94.62 | 44.0 | 95.0 | 192.221 | 99.0 |  |
| 99975168 | 94.36 | 94.6 | 31.0 | 95.0 | 192.092 | 99.0 |  |
| 100007936 | 93.73 | 94.55 | 27.0 | 95.0 | 190.465 | 98.0 |  |
