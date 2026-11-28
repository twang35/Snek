# b30d-gl1-seed4

step **100,007,936** · 3052 evals · trailing **94.81** · peak **94.9** @68,288,512 · sef **96.4** · best30 **99.8** @68,386,816

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
| seed | 4 |
| torch_threads | 1 |

![b30d-gl1-seed4](b30d-gl1-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 4.49 | 4.49 | 0.0 | 9.0 | 1.427 | 0.0 |  |
| 65536 | 10.02 | 7.25 | 0.0 | 29.0 | 6.057 | 0.0 |  |
| 98304 | 19.95 | 14.69 | 1.0 | 42.0 | 15.096 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.76 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 99680256 | 95.0 | 94.78 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 99713024 | 94.44 | 94.8 | 39.0 | 95.0 | 192.17 | 99.0 |  |
| 99745792 | 95.0 | 94.78 | 95.0 | 95.0 | 193.773 | 100.0 |  |
| 99778560 | 95.0 | 94.78 | 95.0 | 95.0 | 193.775 | 100.0 |  |
| 99811328 | 94.69 | 94.84 | 64.0 | 95.0 | 192.414 | 99.0 |  |
| 99844096 | 95.0 | 94.81 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 99876864 | 95.0 | 94.84 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 99909632 | 94.69 | 94.81 | 64.0 | 95.0 | 192.421 | 99.0 |  |
| 99942400 | 94.46 | 94.82 | 41.0 | 95.0 | 192.182 | 99.0 |  |
| 99975168 | 94.61 | 94.82 | 56.0 | 95.0 | 192.375 | 99.0 |  |
| 100007936 | 95.0 | 94.81 | 95.0 | 95.0 | 193.764 | 100.0 |  |
