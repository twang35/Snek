# b27x-hist8-seed24

step **100,007,936** · 3052 evals · trailing **94.59** · peak **94.87** @62,554,112 · sef **96.4** · best30 **99.7** @62,226,432

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
| seed | 24 |
| torch_threads | 1 |

![b27x-hist8-seed24](b27x-hist8-seed24.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 6.42 | 6.42 | 0.0 | 18.0 | 2.518 | 0.0 |  |
| 65536 | 21.0 | 13.71 | 0.0 | 46.0 | 16.531 | 0.0 |  |
| 98304 | 23.33 | 16.92 | 8.0 | 41.0 | 18.346 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 93.75 | 94.54 | 10.0 | 95.0 | 190.535 | 98.0 |  |
| 99680256 | 94.48 | 94.52 | 43.0 | 95.0 | 192.259 | 99.0 |  |
| 99713024 | 94.63 | 94.6 | 58.0 | 95.0 | 192.41 | 99.0 |  |
| 99745792 | 94.39 | 94.59 | 34.0 | 95.0 | 192.118 | 99.0 |  |
| 99778560 | 95.0 | 94.54 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 99811328 | 95.0 | 94.6 | 95.0 | 95.0 | 193.77 | 100.0 |  |
| 99844096 | 95.0 | 94.56 | 95.0 | 95.0 | 193.775 | 100.0 |  |
| 99876864 | 94.54 | 94.54 | 49.0 | 95.0 | 192.318 | 99.0 |  |
| 99909632 | 95.0 | 94.57 | 95.0 | 95.0 | 193.777 | 100.0 |  |
| 99942400 | 95.0 | 94.6 | 95.0 | 95.0 | 193.776 | 100.0 |  |
| 99975168 | 94.02 | 94.56 | 44.0 | 95.0 | 190.764 | 98.0 |  |
| 100007936 | 95.0 | 94.59 | 95.0 | 95.0 | 193.769 | 100.0 |  |
