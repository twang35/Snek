# b30b-gl1-seed2

step **100,007,936** · 3052 evals · trailing **94.82** · peak **94.87** @81,657,856 · sef **96.5** · best30 **99.8** @71,794,688

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
| seed | 2 |
| torch_threads | 1 |

![b30b-gl1-seed2](b30b-gl1-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 13.43 | 13.43 | 1.0 | 32.0 | 9.028 | 0.0 |  |
| 65536 | 39.05 | 28.09 | 12.0 | 69.0 | 33.968 | 0.0 |  |
| 98304 | 35.2 | 30.22 | 1.0 | 65.0 | 30.249 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 94.64 | 94.81 | 59.0 | 95.0 | 192.371 | 99.0 |  |
| 99680256 | 94.49 | 94.8 | 44.0 | 95.0 | 192.22 | 99.0 |  |
| 99713024 | 94.92 | 94.81 | 87.0 | 95.0 | 192.639 | 99.0 |  |
| 99745792 | 95.0 | 94.83 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 99778560 | 95.0 | 94.83 | 95.0 | 95.0 | 193.765 | 100.0 |  |
| 99811328 | 95.0 | 94.83 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 99844096 | 94.72 | 94.82 | 67.0 | 95.0 | 192.446 | 99.0 |  |
| 99876864 | 95.0 | 94.82 | 95.0 | 95.0 | 193.776 | 100.0 |  |
| 99909632 | 94.94 | 94.82 | 89.0 | 95.0 | 192.671 | 99.0 |  |
| 99942400 | 95.0 | 94.83 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 99975168 | 94.76 | 94.82 | 71.0 | 95.0 | 192.486 | 99.0 |  |
| 100007936 | 95.0 | 94.82 | 95.0 | 95.0 | 193.773 | 100.0 |  |
