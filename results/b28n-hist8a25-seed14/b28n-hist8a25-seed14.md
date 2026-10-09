# b28n-hist8a25-seed14

step **200,015,872** · 6104 evals · trailing **94.72** · peak **94.96** @184,713,216 · sef **98.0** · best30 **99.9** @184,942,592

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
| ppo_anneal_fraction | 0.25 |
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
| seed | 14 |
| torch_threads | 1 |

![b28n-hist8a25-seed14](b28n-hist8a25-seed14.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 3.68 | 3.68 | 0.0 | 12.0 | 0.716 | 0.0 |  |
| 65536 | 21.86 | 17.21 | 0.0 | 48.0 | 17.642 | 0.0 |  |
| 98304 | 26.1 | 14.89 | 0.0 | 50.0 | 21.237 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 199655424 | 95.0 | 94.72 | 95.0 | 95.0 | 193.778 | 100.0 |  |
| 199688192 | 94.47 | 94.7 | 42.0 | 95.0 | 192.211 | 99.0 |  |
| 199720960 | 95.0 | 94.71 | 95.0 | 95.0 | 193.772 | 100.0 |  |
| 199753728 | 94.45 | 94.75 | 40.0 | 95.0 | 192.186 | 99.0 |  |
| 199786496 | 94.53 | 94.73 | 48.0 | 95.0 | 192.315 | 99.0 |  |
| 199819264 | 94.11 | 94.72 | 6.0 | 95.0 | 191.893 | 99.0 |  |
| 199852032 | 95.0 | 94.72 | 95.0 | 95.0 | 193.78 | 100.0 |  |
| 199884800 | 94.59 | 94.7 | 54.0 | 95.0 | 192.327 | 99.0 |  |
| 199917568 | 94.49 | 94.71 | 44.0 | 95.0 | 192.232 | 99.0 |  |
| 199950336 | 95.0 | 94.73 | 95.0 | 95.0 | 193.773 | 100.0 |  |
| 199983104 | 94.15 | 94.72 | 45.0 | 95.0 | 190.888 | 98.0 |  |
| 200015872 | 95.0 | 94.72 | 95.0 | 95.0 | 193.778 | 100.0 |  |
