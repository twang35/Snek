# b30h-gl1-seed8

step **100,007,936** · 3052 evals · trailing **94.72** · peak **94.92** @74,743,808 · sef **92.5** · best30 **99.9** @90,243,072

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
| ppo_discount_final | 1.0 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | 0.001 |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.95 |
| ppo_gae_lambda_final | 1.0 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 16.8 |
| ppo_horizon_final | inf |
| ppo_learning_rate | 0.00025 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 512 |
| ppo_normalize_adv | True |
| ppo_rollout | 256 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 32768 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 8 |
| torch_threads | 1 |

![b30h-gl1-seed8](b30h-gl1-seed8.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 8.74 | 8.74 | 0.0 | 29.0 | 5.919 | 0.0 |  |
| 65536 | 35.12 | 27.58 | 0.0 | 69.0 | 30.344 | 0.0 |  |
| 98304 | 34.32 | 21.53 | 8.0 | 53.0 | 29.246 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.72 | 95.0 | 95.0 | 193.768 | 100.0 |  |
| 99680256 | 94.63 | 94.71 | 58.0 | 95.0 | 192.405 | 99.0 |  |
| 99713024 | 95.0 | 94.71 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 99745792 | 95.0 | 94.71 | 95.0 | 95.0 | 193.768 | 100.0 |  |
| 99778560 | 95.0 | 94.71 | 95.0 | 95.0 | 193.771 | 100.0 |  |
| 99811328 | 93.62 | 94.7 | 18.0 | 95.0 | 190.355 | 98.0 |  |
| 99844096 | 94.17 | 94.7 | 47.0 | 95.0 | 190.895 | 98.0 |  |
| 99876864 | 95.0 | 94.73 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 99909632 | 94.95 | 94.73 | 90.0 | 95.0 | 192.718 | 99.0 |  |
| 99942400 | 95.0 | 94.73 | 95.0 | 95.0 | 193.765 | 100.0 |  |
| 99975168 | 94.55 | 94.73 | 50.0 | 95.0 | 192.323 | 99.0 |  |
| 100007936 | 94.48 | 94.72 | 43.0 | 95.0 | 192.248 | 99.0 |  |
