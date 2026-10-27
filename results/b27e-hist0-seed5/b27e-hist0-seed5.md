# b27e-hist0-seed5

step **100,007,936** · 3052 evals · trailing **94.68** · peak **94.86** @73,072,640 · sef **95.7** · best30 **99.4** @92,274,688

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
| seed | 5 |
| torch_threads | 1 |

![b27e-hist0-seed5](b27e-hist0-seed5.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 3.61 | 19.74 | 0.0 | 52.0 | 2.729 | 0.0 |  |
| 65536 | 43.73 | 27.74 | 0.0 | 74.0 | 38.942 | 0.0 |  |
| 98304 | 39.93 | 30.79 | 10.0 | 71.0 | 34.863 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 94.9 | 94.68 | 89.0 | 95.0 | 191.558 | 98.0 |  |
| 99680256 | 95.0 | 94.68 | 95.0 | 95.0 | 193.724 | 100.0 |  |
| 99713024 | 94.83 | 94.66 | 86.0 | 95.0 | 190.474 | 97.0 |  |
| 99745792 | 95.0 | 94.63 | 95.0 | 95.0 | 193.728 | 100.0 |  |
| 99778560 | 94.91 | 94.62 | 89.0 | 95.0 | 191.605 | 98.0 |  |
| 99811328 | 95.0 | 94.65 | 95.0 | 95.0 | 193.725 | 100.0 |  |
| 99844096 | 94.67 | 94.65 | 62.0 | 95.0 | 192.402 | 99.0 |  |
| 99876864 | 94.97 | 94.68 | 92.0 | 95.0 | 192.655 | 99.0 |  |
| 99909632 | 94.13 | 94.65 | 8.0 | 95.0 | 191.869 | 99.0 |  |
| 99942400 | 94.38 | 94.67 | 64.0 | 95.0 | 191.113 | 98.0 |  |
| 99975168 | 94.62 | 94.67 | 57.0 | 95.0 | 192.352 | 99.0 |  |
| 100007936 | 95.0 | 94.68 | 95.0 | 95.0 | 193.735 | 100.0 |  |
