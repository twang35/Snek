# b28o-hist8a25-seed15

step **200,015,872** · 6104 evals · trailing **94.6** · peak **94.92** @138,346,496 · sef **98.2** · best30 **99.8** @138,346,496

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
| seed | 15 |
| torch_threads | 1 |

![b28o-hist8a25-seed15](b28o-hist8a25-seed15.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 3.17 | 3.17 | 0.0 | 14.0 | 0.377 | 0.0 |  |
| 65536 | 24.34 | 22.13 | 0.0 | 51.0 | 19.673 | 0.0 |  |
| 98304 | 32.74 | 17.96 | 13.0 | 50.0 | 27.671 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 199655424 | 94.39 | 94.56 | 34.0 | 95.0 | 192.167 | 99.0 |  |
| 199688192 | 95.0 | 94.57 | 95.0 | 95.0 | 193.783 | 100.0 |  |
| 199720960 | 95.0 | 94.61 | 95.0 | 95.0 | 193.778 | 100.0 |  |
| 199753728 | 94.69 | 94.6 | 64.0 | 95.0 | 192.478 | 99.0 |  |
| 199786496 | 95.0 | 94.63 | 95.0 | 95.0 | 193.78 | 100.0 |  |
| 199819264 | 94.1 | 94.57 | 5.0 | 95.0 | 191.842 | 99.0 |  |
| 199852032 | 92.74 | 94.56 | 5.0 | 95.0 | 188.456 | 97.0 |  |
| 199884800 | 95.0 | 94.63 | 95.0 | 95.0 | 193.78 | 100.0 |  |
| 199917568 | 94.21 | 94.61 | 16.0 | 95.0 | 191.95 | 99.0 |  |
| 199950336 | 94.76 | 94.6 | 71.0 | 95.0 | 192.502 | 99.0 |  |
| 199983104 | 94.21 | 94.58 | 16.0 | 95.0 | 191.952 | 99.0 |  |
| 200015872 | 95.0 | 94.6 | 95.0 | 95.0 | 193.778 | 100.0 |  |
