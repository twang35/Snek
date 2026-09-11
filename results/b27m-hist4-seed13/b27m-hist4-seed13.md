# b27m-hist4-seed13

step **100,007,936** · 3052 evals · trailing **94.66** · peak **94.91** @57,606,144 · sef **96.3** · best30 **99.8** @91,914,240

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
| seed | 13 |
| torch_threads | 1 |

![b27m-hist4-seed13](b27m-hist4-seed13.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 0.15 | 0.15 | 0.0 | 2.0 | -1.024 | 0.0 |  |
| 65536 | 0.06 | 0.1 | 0.0 | 2.0 | -0.489 | 0.0 |  |
| 98304 | 3.92 | 1.38 | 0.0 | 30.0 | 2.336 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.59 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 99680256 | 94.62 | 94.61 | 57.0 | 95.0 | 192.391 | 99.0 |  |
| 99713024 | 95.0 | 94.61 | 95.0 | 95.0 | 193.762 | 100.0 |  |
| 99745792 | 95.0 | 94.67 | 95.0 | 95.0 | 193.753 | 100.0 |  |
| 99778560 | 94.81 | 94.64 | 76.0 | 95.0 | 192.566 | 99.0 |  |
| 99811328 | 95.0 | 94.65 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 99844096 | 94.93 | 94.63 | 88.0 | 95.0 | 192.691 | 99.0 |  |
| 99876864 | 93.32 | 94.62 | 10.0 | 95.0 | 190.093 | 98.0 |  |
| 99909632 | 95.0 | 94.65 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 99942400 | 94.61 | 94.69 | 56.0 | 95.0 | 192.373 | 99.0 |  |
| 99975168 | 95.0 | 94.66 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 100007936 | 94.15 | 94.66 | 10.0 | 95.0 | 191.915 | 99.0 |  |
