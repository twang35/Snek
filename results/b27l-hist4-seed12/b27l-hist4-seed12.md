# b27l-hist4-seed12

step **100,007,936** · 3052 evals · trailing **94.66** · peak **94.88** @56,557,568 · sef **96.5** · best30 **99.7** @58,916,864

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
| seed | 12 |
| torch_threads | 1 |

![b27l-hist4-seed12](b27l-hist4-seed12.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 0.09 | 0.09 | 0.0 | 2.0 | -0.46 | 0.0 |  |
| 65536 | 1.08 | 0.59 | 0.0 | 8.0 | 0.441 | 0.0 |  |
| 98304 | 15.35 | 5.51 | 1.0 | 33.0 | 11.344 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 94.13 | 94.62 | 8.0 | 95.0 | 191.897 | 99.0 |  |
| 99680256 | 94.51 | 94.65 | 46.0 | 95.0 | 192.273 | 99.0 |  |
| 99713024 | 95.0 | 94.65 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 99745792 | 95.0 | 94.67 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 99778560 | 95.0 | 94.67 | 95.0 | 95.0 | 193.763 | 100.0 |  |
| 99811328 | 95.0 | 94.67 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 99844096 | 95.0 | 94.66 | 95.0 | 95.0 | 193.77 | 100.0 |  |
| 99876864 | 94.17 | 94.63 | 12.0 | 95.0 | 191.949 | 99.0 |  |
| 99909632 | 95.0 | 94.66 | 95.0 | 95.0 | 193.766 | 100.0 |  |
| 99942400 | 95.0 | 94.65 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 99975168 | 94.37 | 94.65 | 53.0 | 95.0 | 191.14 | 98.0 |  |
| 100007936 | 95.0 | 94.66 | 95.0 | 95.0 | 193.767 | 100.0 |  |
