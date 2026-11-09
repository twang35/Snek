# b34h-rp50-seed8

step **100,007,936** · 3052 evals · trailing **94.39** · peak **94.92** @84,475,904 · sef **98.5** · best30 **99.8** @84,082,688

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
| seed | 8 |
| torch_threads | 1 |

![b34h-rp50-seed8](b34h-rp50-seed8.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 0.56 | 0.56 | 0.0 | 4.0 | 0.008 | 0.0 |  |
| 65536 | 0.76 | 0.66 | 0.0 | 4.0 | 0.207 | 0.0 |  |
| 98304 | 1.41 | 0.91 | 0.0 | 8.0 | 0.854 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 93.51 | 94.42 | 11.0 | 95.0 | 190.243 | 98.0 |  |
| 99680256 | 94.59 | 94.47 | 54.0 | 95.0 | 192.37 | 99.0 |  |
| 99713024 | 93.9 | 94.46 | 33.0 | 95.0 | 190.635 | 98.0 |  |
| 99745792 | 94.0 | 94.46 | 43.0 | 95.0 | 190.687 | 98.0 |  |
| 99778560 | 95.0 | 94.44 | 95.0 | 95.0 | 193.768 | 100.0 |  |
| 99811328 | 94.48 | 94.45 | 43.0 | 95.0 | 192.216 | 99.0 |  |
| 99844096 | 94.31 | 94.46 | 26.0 | 95.0 | 192.079 | 99.0 |  |
| 99876864 | 94.61 | 94.44 | 56.0 | 95.0 | 192.382 | 99.0 |  |
| 99909632 | 95.0 | 94.46 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 99942400 | 95.0 | 94.47 | 95.0 | 95.0 | 193.772 | 100.0 |  |
| 99975168 | 94.4 | 94.48 | 36.0 | 95.0 | 191.088 | 98.0 |  |
| 100007936 | 92.27 | 94.39 | 20.0 | 95.0 | 186.977 | 96.0 |  |
