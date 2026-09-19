# b30g-gl1-seed7

step **100,007,936** · 3052 evals · trailing **94.41** · peak **94.93** @58,294,272 · sef **95.1** · best30 **99.9** @58,327,040

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
| seed | 7 |
| torch_threads | 1 |

![b30g-gl1-seed7](b30g-gl1-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 0.59 | 0.59 | 0.0 | 5.0 | -0.007 | 0.0 |  |
| 65536 | 7.29 | 3.94 | 0.0 | 22.0 | 5.36 | 0.0 |  |
| 98304 | 19.27 | 9.05 | 6.0 | 36.0 | 14.744 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 94.55 | 94.42 | 50.0 | 95.0 | 192.319 | 99.0 |  |
| 99680256 | 94.05 | 94.43 | 11.0 | 95.0 | 190.77 | 98.0 |  |
| 99713024 | 94.17 | 94.38 | 12.0 | 95.0 | 191.938 | 99.0 |  |
| 99745792 | 95.0 | 94.42 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 99778560 | 94.29 | 94.39 | 24.0 | 95.0 | 192.063 | 99.0 |  |
| 99811328 | 94.09 | 94.38 | 44.0 | 95.0 | 190.773 | 98.0 |  |
| 99844096 | 94.63 | 94.37 | 58.0 | 95.0 | 192.355 | 99.0 |  |
| 99876864 | 95.0 | 94.42 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 99909632 | 94.32 | 94.41 | 27.0 | 95.0 | 192.043 | 99.0 |  |
| 99942400 | 94.11 | 94.43 | 46.0 | 95.0 | 190.843 | 98.0 |  |
| 99975168 | 93.53 | 94.38 | 43.0 | 95.0 | 188.263 | 96.0 |  |
| 100007936 | 95.0 | 94.41 | 95.0 | 95.0 | 193.761 | 100.0 |  |
