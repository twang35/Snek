# b31a-mse-seed1

step **100,007,936** · 3052 evals · trailing **94.67** · peak **94.85** @66,682,880 · sef **97.6** · best30 **99.7** @64,225,280

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
| ppo_value_loss | mse |
| ppo_vf_coef | 0.5 |
| seed | 1 |
| torch_threads | 1 |

![b31a-mse-seed1](b31a-mse-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 6.82 | 6.82 | 0.0 | 19.0 | 2.119 | 0.0 |  |
| 65536 | 30.65 | 23.38 | 0.0 | 59.0 | 25.776 | 0.0 |  |
| 98304 | 29.17 | 24.54 | 12.0 | 59.0 | 24.121 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.52 | 95.0 | 95.0 | 193.779 | 100.0 |  |
| 99680256 | 95.0 | 94.54 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 99713024 | 95.0 | 94.54 | 95.0 | 95.0 | 193.779 | 100.0 |  |
| 99745792 | 94.31 | 94.52 | 26.0 | 95.0 | 192.043 | 99.0 |  |
| 99778560 | 94.55 | 94.51 | 50.0 | 95.0 | 192.287 | 99.0 |  |
| 99811328 | 94.95 | 94.61 | 90.0 | 95.0 | 192.685 | 99.0 |  |
| 99844096 | 95.0 | 94.59 | 95.0 | 95.0 | 193.783 | 100.0 |  |
| 99876864 | 95.0 | 94.58 | 95.0 | 95.0 | 193.771 | 100.0 |  |
| 99909632 | 95.0 | 94.55 | 95.0 | 95.0 | 193.775 | 100.0 |  |
| 99942400 | 95.0 | 94.61 | 95.0 | 95.0 | 193.776 | 100.0 |  |
| 99975168 | 94.43 | 94.65 | 38.0 | 95.0 | 192.203 | 99.0 |  |
| 100007936 | 95.0 | 94.67 | 95.0 | 95.0 | 193.77 | 100.0 |  |
