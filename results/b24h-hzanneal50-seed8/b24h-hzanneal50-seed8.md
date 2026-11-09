# b24h-hzanneal50-seed8

step **200,015,872** · 6104 evals · trailing **94.69** · peak **94.81** @142,704,640 · sef **98.3** · best30 **99.1** @129,662,976

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
| max_steps | 200015872 |
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

![b24h-hzanneal50-seed8](b24h-hzanneal50-seed8.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 21.46 | 21.46 | 3.0 | 40.0 | 16.307 | 0.0 |  |
| 65536 | 56.06 | 43.28 | 2.0 | 95.0 | 52.663 | 1.0 |  |
| 98304 | 50.15 | 36.76 | 13.0 | 95.0 | 46.117 | 1.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 199655424 | 95.0 | 94.67 | 95.0 | 95.0 | 193.732 | 100.0 |  |
| 199688192 | 94.81 | 94.69 | 76.0 | 95.0 | 192.546 | 99.0 |  |
| 199720960 | 95.0 | 94.69 | 95.0 | 95.0 | 193.737 | 100.0 |  |
| 199753728 | 95.0 | 94.72 | 95.0 | 95.0 | 193.726 | 100.0 |  |
| 199786496 | 95.0 | 94.73 | 95.0 | 95.0 | 193.733 | 100.0 |  |
| 199819264 | 94.72 | 94.72 | 67.0 | 95.0 | 192.45 | 99.0 |  |
| 199852032 | 94.38 | 94.71 | 71.0 | 95.0 | 190.118 | 97.0 |  |
| 199884800 | 95.0 | 94.72 | 95.0 | 95.0 | 193.74 | 100.0 |  |
| 199917568 | 95.0 | 94.72 | 95.0 | 95.0 | 193.731 | 100.0 |  |
| 199950336 | 95.0 | 94.71 | 95.0 | 95.0 | 193.733 | 100.0 |  |
| 199983104 | 94.74 | 94.71 | 84.0 | 95.0 | 190.477 | 97.0 |  |
| 200015872 | 94.23 | 94.69 | 18.0 | 95.0 | 191.973 | 99.0 |  |
