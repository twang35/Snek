# b34e-rp50-seed5

step **100,007,936** · 3052 evals · trailing **94.75** · peak **94.93** @95,682,560 · sef **98.5** · best30 **99.8** @95,485,952

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
| init_from | None |
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

![b34e-rp50-seed5](b34e-rp50-seed5.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 4.03 | 4.03 | 0.0 | 16.0 | 2.652 | 0.0 |  |
| 65536 | 5.72 | 4.88 | 0.0 | 18.0 | 4.065 | 0.0 |  |
| 98304 | 12.0 | 7.25 | 0.0 | 27.0 | 9.165 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 94.29 | 94.74 | 24.0 | 95.0 | 192.068 | 99.0 |  |
| 99680256 | 95.0 | 94.73 | 95.0 | 95.0 | 193.774 | 100.0 |  |
| 99713024 | 94.58 | 94.74 | 53.0 | 95.0 | 192.355 | 99.0 |  |
| 99745792 | 95.0 | 94.74 | 95.0 | 95.0 | 193.77 | 100.0 |  |
| 99778560 | 95.0 | 94.76 | 95.0 | 95.0 | 193.771 | 100.0 |  |
| 99811328 | 94.47 | 94.72 | 42.0 | 95.0 | 192.25 | 99.0 |  |
| 99844096 | 95.0 | 94.72 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 99876864 | 94.19 | 94.71 | 14.0 | 95.0 | 191.916 | 99.0 |  |
| 99909632 | 94.49 | 94.74 | 44.0 | 95.0 | 192.215 | 99.0 |  |
| 99942400 | 94.35 | 94.7 | 30.0 | 95.0 | 192.125 | 99.0 |  |
| 99975168 | 95.0 | 94.73 | 95.0 | 95.0 | 193.772 | 100.0 |  |
| 100007936 | 95.0 | 94.75 | 95.0 | 95.0 | 193.776 | 100.0 |  |
