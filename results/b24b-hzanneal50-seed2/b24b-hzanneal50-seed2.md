# b24b-hzanneal50-seed2

step **200,015,872** · 6104 evals · trailing **94.6** · peak **94.85** @120,324,096 · sef **97.9** · best30 **99.5** @120,061,952

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
| seed | 2 |
| torch_threads | 1 |

![b24b-hzanneal50-seed2](b24b-hzanneal50-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 2.31 | 2.31 | 1.0 | 7.0 | -1.411 | 0.0 |  |
| 65536 | 14.53 | 8.42 | 6.0 | 30.0 | 9.704 | 0.0 |  |
| 98304 | 26.25 | 14.36 | 6.0 | 56.0 | 21.244 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 199655424 | 94.63 | 94.78 | 58.0 | 95.0 | 192.37 | 99.0 |  |
| 199688192 | 94.39 | 94.75 | 59.0 | 95.0 | 191.12 | 98.0 |  |
| 199720960 | 93.64 | 94.7 | 10.0 | 95.0 | 189.377 | 97.0 |  |
| 199753728 | 94.58 | 94.67 | 53.0 | 95.0 | 192.312 | 99.0 |  |
| 199786496 | 94.33 | 94.61 | 66.0 | 95.0 | 190.058 | 97.0 |  |
| 199819264 | 93.87 | 94.7 | 10.0 | 95.0 | 190.597 | 98.0 |  |
| 199852032 | 95.0 | 94.73 | 95.0 | 95.0 | 193.733 | 100.0 |  |
| 199884800 | 94.22 | 94.72 | 56.0 | 95.0 | 190.948 | 98.0 |  |
| 199917568 | 94.46 | 94.67 | 56.0 | 95.0 | 191.198 | 98.0 |  |
| 199950336 | 94.59 | 94.63 | 54.0 | 95.0 | 192.322 | 99.0 |  |
| 199983104 | 94.64 | 94.65 | 59.0 | 95.0 | 192.369 | 99.0 |  |
| 200015872 | 94.63 | 94.6 | 58.0 | 95.0 | 192.365 | 99.0 |  |
