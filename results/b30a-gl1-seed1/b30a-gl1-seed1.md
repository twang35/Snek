# b30a-gl1-seed1

step **100,007,936** · 3052 evals · trailing **94.85** · peak **94.96** @95,944,704 · sef **96.0** · best30 **99.9** @95,944,704

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
| seed | 1 |
| torch_threads | 1 |

![b30a-gl1-seed1](b30a-gl1-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 7.32 | 7.32 | 1.0 | 19.0 | 2.308 | 0.0 |  |
| 65536 | 27.56 | 24.86 | 1.0 | 57.0 | 22.836 | 0.0 |  |
| 98304 | 28.66 | 17.99 | 9.0 | 52.0 | 23.651 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.75 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 99680256 | 95.0 | 94.81 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 99713024 | 94.08 | 94.78 | 36.0 | 95.0 | 190.776 | 98.0 |  |
| 99745792 | 95.0 | 94.75 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 99778560 | 95.0 | 94.75 | 95.0 | 95.0 | 193.766 | 100.0 |  |
| 99811328 | 95.0 | 94.75 | 95.0 | 95.0 | 193.765 | 100.0 |  |
| 99844096 | 95.0 | 94.81 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 99876864 | 95.0 | 94.78 | 95.0 | 95.0 | 193.78 | 100.0 |  |
| 99909632 | 94.53 | 94.8 | 48.0 | 95.0 | 192.297 | 99.0 |  |
| 99942400 | 95.0 | 94.78 | 95.0 | 95.0 | 193.774 | 100.0 |  |
| 99975168 | 95.0 | 94.83 | 95.0 | 95.0 | 193.77 | 100.0 |  |
| 100007936 | 95.0 | 94.85 | 95.0 | 95.0 | 193.769 | 100.0 |  |
