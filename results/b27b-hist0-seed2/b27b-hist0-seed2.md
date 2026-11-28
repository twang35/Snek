# b27b-hist0-seed2

step **100,007,936** · 3052 evals · trailing **94.54** · peak **94.82** @68,648,960 · sef **95.7** · best30 **99.2** @69,337,088

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
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 2 |
| torch_threads | 1 |

![b27b-hist0-seed2](b27b-hist0-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 1.39 | 1.39 | 0.0 | 8.0 | 0.839 | 0.0 |  |
| 65536 | 4.41 | 2.9 | 1.0 | 51.0 | 3.759 | 0.0 |  |
| 98304 | 51.74 | 26.37 | 17.0 | 88.0 | 46.599 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 94.89 | 94.56 | 89.0 | 95.0 | 191.62 | 98.0 |  |
| 99680256 | 94.37 | 94.56 | 32.0 | 95.0 | 192.052 | 99.0 |  |
| 99713024 | 94.76 | 94.5 | 71.0 | 95.0 | 192.488 | 99.0 |  |
| 99745792 | 94.67 | 94.51 | 62.0 | 95.0 | 192.398 | 99.0 |  |
| 99778560 | 94.45 | 94.53 | 58.0 | 95.0 | 190.187 | 97.0 |  |
| 99811328 | 94.27 | 94.5 | 58.0 | 95.0 | 190.967 | 98.0 |  |
| 99844096 | 94.32 | 94.51 | 58.0 | 95.0 | 191.055 | 98.0 |  |
| 99876864 | 95.0 | 94.55 | 95.0 | 95.0 | 193.711 | 100.0 |  |
| 99909632 | 95.0 | 94.53 | 95.0 | 95.0 | 193.725 | 100.0 |  |
| 99942400 | 94.63 | 94.55 | 62.0 | 95.0 | 191.357 | 98.0 |  |
| 99975168 | 95.0 | 94.53 | 95.0 | 95.0 | 193.72 | 100.0 |  |
| 100007936 | 94.94 | 94.54 | 89.0 | 95.0 | 192.663 | 99.0 |  |
