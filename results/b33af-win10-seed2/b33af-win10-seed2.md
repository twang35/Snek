# b33af-win10-seed2

step **50,003,968** · 1526 evals · trailing **94.42** · peak **94.69** @28,344,320 · sef **90.2** · best30 **99.4** @45,285,376

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
| max_steps | 50003968 |
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
| seed | 2 |
| torch_threads | 1 |

![b33af-win10-seed2](b33af-win10-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 12.9 | 12.9 | 0.0 | 31.0 | 8.279 | 0.0 |  |
| 65536 | 37.36 | 27.91 | 3.0 | 64.0 | 32.509 | 0.0 |  |
| 98304 | 33.47 | 23.18 | 1.0 | 57.0 | 28.446 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 95.0 | 94.5 | 95.0 | 95.0 | 193.754 | 100.0 |  |
| 49676288 | 94.55 | 94.5 | 50.0 | 95.0 | 192.321 | 99.0 |  |
| 49709056 | 94.39 | 94.51 | 34.0 | 95.0 | 192.156 | 99.0 |  |
| 49741824 | 93.43 | 94.48 | 8.0 | 95.0 | 190.15 | 98.0 |  |
| 49774592 | 94.42 | 94.51 | 37.0 | 95.0 | 192.141 | 99.0 |  |
| 49807360 | 95.0 | 94.44 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 49840128 | 95.0 | 94.42 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 49872896 | 94.41 | 94.51 | 36.0 | 95.0 | 192.178 | 99.0 |  |
| 49905664 | 93.26 | 94.45 | 16.0 | 95.0 | 188.99 | 97.0 |  |
| 49938432 | 93.87 | 94.44 | 24.0 | 95.0 | 190.593 | 98.0 |  |
| 49971200 | 94.42 | 94.42 | 37.0 | 95.0 | 192.132 | 99.0 |  |
| 50003968 | 95.0 | 94.42 | 95.0 | 95.0 | 193.764 | 100.0 |  |
