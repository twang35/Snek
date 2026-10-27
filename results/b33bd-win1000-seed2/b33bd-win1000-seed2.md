# b33bd-win1000-seed2

step **50,003,968** · 1526 evals · trailing **94.6** · peak **94.92** @18,350,080 · sef **78.2** · best30 **99.8** @18,612,224

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

![b33bd-win1000-seed2](b33bd-win1000-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 13.35 | 13.35 | 0.0 | 26.0 | 8.863 | 0.0 |  |
| 65536 | 38.17 | 30.43 | 12.0 | 83.0 | 33.313 | 0.0 |  |
| 98304 | 35.06 | 24.21 | 3.0 | 61.0 | 30.071 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.58 | 94.59 | 53.0 | 95.0 | 192.338 | 99.0 |  |
| 49676288 | 94.29 | 94.63 | 24.0 | 95.0 | 192.015 | 99.0 |  |
| 49709056 | 95.0 | 94.65 | 95.0 | 95.0 | 193.756 | 100.0 |  |
| 49741824 | 94.25 | 94.61 | 20.0 | 95.0 | 192.02 | 99.0 |  |
| 49774592 | 95.0 | 94.61 | 95.0 | 95.0 | 193.754 | 100.0 |  |
| 49807360 | 95.0 | 94.61 | 95.0 | 95.0 | 193.754 | 100.0 |  |
| 49840128 | 95.0 | 94.62 | 95.0 | 95.0 | 193.757 | 100.0 |  |
| 49872896 | 95.0 | 94.63 | 95.0 | 95.0 | 193.755 | 100.0 |  |
| 49905664 | 95.0 | 94.63 | 95.0 | 95.0 | 193.752 | 100.0 |  |
| 49938432 | 93.14 | 94.58 | 20.0 | 95.0 | 188.871 | 97.0 |  |
| 49971200 | 95.0 | 94.58 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 50003968 | 95.0 | 94.6 | 95.0 | 95.0 | 193.77 | 100.0 |  |
