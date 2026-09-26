# b27g-hist0-seed7

step **100,007,936** · 3052 evals · trailing **94.44** · peak **94.74** @79,134,720 · sef **95.2** · best30 **98.8** @79,233,024

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
| seed | 7 |
| torch_threads | 1 |

![b27g-hist0-seed7](b27g-hist0-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 14.07 | 14.07 | 0.0 | 37.0 | 10.575 | 0.0 |  |
| 65536 | 25.92 | 20.0 | 0.0 | 45.0 | 21.275 | 0.0 |  |
| 98304 | 29.24 | 24.82 | 8.0 | 51.0 | 24.193 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.47 | 95.0 | 95.0 | 193.721 | 100.0 |  |
| 99680256 | 93.7 | 94.48 | 55.0 | 95.0 | 188.438 | 96.0 |  |
| 99713024 | 94.37 | 94.44 | 32.0 | 95.0 | 192.103 | 99.0 |  |
| 99745792 | 95.0 | 94.47 | 95.0 | 95.0 | 193.732 | 100.0 |  |
| 99778560 | 93.95 | 94.45 | 58.0 | 95.0 | 189.683 | 97.0 |  |
| 99811328 | 94.28 | 94.45 | 58.0 | 95.0 | 191.013 | 98.0 |  |
| 99844096 | 95.0 | 94.45 | 95.0 | 95.0 | 193.735 | 100.0 |  |
| 99876864 | 94.63 | 94.44 | 58.0 | 95.0 | 192.361 | 99.0 |  |
| 99909632 | 95.0 | 94.44 | 95.0 | 95.0 | 193.728 | 100.0 |  |
| 99942400 | 94.39 | 94.43 | 59.0 | 95.0 | 191.124 | 98.0 |  |
| 99975168 | 94.13 | 94.44 | 8.0 | 95.0 | 191.872 | 99.0 |  |
| 100007936 | 94.59 | 94.44 | 66.0 | 95.0 | 191.325 | 98.0 |  |
