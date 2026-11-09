# b30c-gl1-seed3

step **100,007,936** · 3052 evals · trailing **94.42** · peak **94.96** @89,489,408 · sef **96.5** · best30 **99.9** @89,423,872

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
| seed | 3 |
| torch_threads | 1 |

![b30c-gl1-seed3](b30c-gl1-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 3.29 | 3.29 | 0.0 | 12.0 | 0.801 | 0.0 |  |
| 65536 | 15.13 | 17.74 | 0.0 | 35.0 | 11.471 | 0.0 |  |
| 98304 | 24.77 | 14.03 | 1.0 | 48.0 | 20.248 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 94.9 | 94.41 | 85.0 | 95.0 | 192.629 | 99.0 |  |
| 99680256 | 92.7 | 94.39 | 21.0 | 95.0 | 186.31 | 95.0 |  |
| 99713024 | 95.0 | 94.47 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 99745792 | 93.42 | 94.44 | 10.0 | 95.0 | 190.157 | 98.0 |  |
| 99778560 | 93.81 | 94.4 | 30.0 | 95.0 | 190.546 | 98.0 |  |
| 99811328 | 94.59 | 94.42 | 54.0 | 95.0 | 192.361 | 99.0 |  |
| 99844096 | 94.61 | 94.39 | 56.0 | 95.0 | 192.378 | 99.0 |  |
| 99876864 | 95.0 | 94.42 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 99909632 | 94.55 | 94.41 | 50.0 | 95.0 | 192.322 | 99.0 |  |
| 99942400 | 95.0 | 94.45 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 99975168 | 93.93 | 94.44 | 41.0 | 95.0 | 190.703 | 98.0 |  |
| 100007936 | 94.64 | 94.42 | 59.0 | 95.0 | 192.363 | 99.0 |  |
