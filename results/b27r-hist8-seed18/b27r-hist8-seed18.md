# b27r-hist8-seed18

step **100,007,936** · 3052 evals · trailing **94.53** · peak **94.86** @82,804,736 · sef **92.7** · best30 **99.7** @67,567,616

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
| seed | 18 |
| torch_threads | 1 |

![b27r-hist8-seed18](b27r-hist8-seed18.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 3.25 | 3.25 | 1.0 | 7.0 | 2.384 | 0.0 |  |
| 65536 | 13.82 | 8.54 | 4.0 | 24.0 | 9.223 | 0.0 |  |
| 98304 | 21.1 | 12.72 | 2.0 | 38.0 | 16.114 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.61 | 95.0 | 95.0 | 193.776 | 100.0 |  |
| 99680256 | 95.0 | 94.61 | 95.0 | 95.0 | 193.779 | 100.0 |  |
| 99713024 | 92.9 | 94.54 | 6.0 | 95.0 | 188.691 | 97.0 |  |
| 99745792 | 93.67 | 94.49 | 17.0 | 95.0 | 190.372 | 98.0 |  |
| 99778560 | 95.0 | 94.51 | 95.0 | 95.0 | 193.771 | 100.0 |  |
| 99811328 | 95.0 | 94.54 | 95.0 | 95.0 | 193.777 | 100.0 |  |
| 99844096 | 93.62 | 94.54 | 8.0 | 95.0 | 190.404 | 98.0 |  |
| 99876864 | 95.0 | 94.54 | 95.0 | 95.0 | 193.774 | 100.0 |  |
| 99909632 | 94.54 | 94.52 | 49.0 | 95.0 | 192.282 | 99.0 |  |
| 99942400 | 95.0 | 94.52 | 95.0 | 95.0 | 193.779 | 100.0 |  |
| 99975168 | 94.13 | 94.52 | 8.0 | 95.0 | 191.911 | 99.0 |  |
| 100007936 | 94.66 | 94.53 | 61.0 | 95.0 | 192.434 | 99.0 |  |
