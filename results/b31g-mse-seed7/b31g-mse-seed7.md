# b31g-mse-seed7

step **100,007,936** · 3052 evals · trailing **94.72** · peak **94.91** @99,188,736 · sef **97.4** · best30 **99.8** @70,647,808

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
| seed | 7 |
| torch_threads | 1 |

![b31g-mse-seed7](b31g-mse-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 0.3 | 0.3 | 0.0 | 4.0 | -0.296 | 0.0 |  |
| 65536 | 6.4 | 3.35 | 1.0 | 19.0 | 4.702 | 0.0 |  |
| 98304 | 17.37 | 14.18 | 4.0 | 34.0 | 13.463 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.78 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 99680256 | 95.0 | 94.78 | 95.0 | 95.0 | 193.77 | 100.0 |  |
| 99713024 | 95.0 | 94.78 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 99745792 | 95.0 | 94.78 | 95.0 | 95.0 | 193.768 | 100.0 |  |
| 99778560 | 95.0 | 94.78 | 95.0 | 95.0 | 193.768 | 100.0 |  |
| 99811328 | 94.59 | 94.78 | 54.0 | 95.0 | 192.362 | 99.0 |  |
| 99844096 | 94.45 | 94.77 | 40.0 | 95.0 | 192.226 | 99.0 |  |
| 99876864 | 94.66 | 94.76 | 61.0 | 95.0 | 192.43 | 99.0 |  |
| 99909632 | 95.0 | 94.76 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 99942400 | 95.0 | 94.76 | 95.0 | 95.0 | 193.774 | 100.0 |  |
| 99975168 | 94.5 | 94.76 | 45.0 | 95.0 | 192.23 | 99.0 |  |
| 100007936 | 93.67 | 94.72 | 15.0 | 95.0 | 190.404 | 98.0 |  |
