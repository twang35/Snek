# b34g-rp50-seed7

step **100,007,936** · 3052 evals · trailing **94.65** · peak **94.9** @4,096,000 · sef **98.2** · best30 **99.8** @71,172,096

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
| seed | 7 |
| torch_threads | 1 |

![b34g-rp50-seed7](b34g-rp50-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 0.13 | 0.13 | 0.0 | 2.0 | -0.598 | 0.0 |  |
| 65536 | 0.75 | 0.44 | 0.0 | 2.0 | -2.394 | 0.0 |  |
| 98304 | 0.69 | 0.52 | 0.0 | 3.0 | -4.002 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 94.61 | 94.65 | 56.0 | 95.0 | 192.387 | 99.0 |  |
| 99680256 | 93.77 | 94.76 | 13.0 | 95.0 | 190.55 | 98.0 |  |
| 99713024 | 93.79 | 94.69 | 26.0 | 95.0 | 190.532 | 98.0 |  |
| 99745792 | 95.0 | 94.72 | 95.0 | 95.0 | 193.778 | 100.0 |  |
| 99778560 | 94.18 | 94.69 | 13.0 | 95.0 | 191.955 | 99.0 |  |
| 99811328 | 95.0 | 94.65 | 95.0 | 95.0 | 193.78 | 100.0 |  |
| 99844096 | 95.0 | 94.69 | 95.0 | 95.0 | 193.772 | 100.0 |  |
| 99876864 | 95.0 | 94.69 | 95.0 | 95.0 | 193.771 | 100.0 |  |
| 99909632 | 93.54 | 94.64 | 4.0 | 95.0 | 190.247 | 98.0 |  |
| 99942400 | 95.0 | 94.66 | 95.0 | 95.0 | 193.777 | 100.0 |  |
| 99975168 | 94.3 | 94.65 | 25.0 | 95.0 | 192.08 | 99.0 |  |
| 100007936 | 95.0 | 94.65 | 95.0 | 95.0 | 193.777 | 100.0 |  |
