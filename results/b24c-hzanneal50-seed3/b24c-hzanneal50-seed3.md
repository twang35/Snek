# b24c-hzanneal50-seed3

step **200,015,872** · 6104 evals · trailing **94.52** · peak **94.83** @129,204,224 · sef **97.6** · best30 **99.3** @128,942,080

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
| seed | 3 |
| torch_threads | 1 |

![b24c-hzanneal50-seed3](b24c-hzanneal50-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 0.02 | 0.02 | 0.0 | 1.0 | -2.848 | 0.0 |  |
| 65536 | 0.43 | 0.23 | 0.0 | 3.0 | -0.122 | 0.0 |  |
| 98304 | 21.22 | 11.96 | 5.0 | 42.0 | 17.142 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 199655424 | 94.21 | 94.64 | 16.0 | 95.0 | 191.942 | 99.0 |  |
| 199688192 | 93.81 | 94.66 | 10.0 | 95.0 | 190.508 | 98.0 |  |
| 199720960 | 94.83 | 94.64 | 78.0 | 95.0 | 192.554 | 99.0 |  |
| 199753728 | 93.11 | 94.59 | 11.0 | 95.0 | 187.804 | 96.0 |  |
| 199786496 | 94.64 | 94.58 | 59.0 | 95.0 | 192.371 | 99.0 |  |
| 199819264 | 94.63 | 94.57 | 58.0 | 95.0 | 192.356 | 99.0 |  |
| 199852032 | 94.95 | 94.56 | 90.0 | 95.0 | 192.669 | 99.0 |  |
| 199884800 | 94.25 | 94.58 | 56.0 | 95.0 | 190.936 | 98.0 |  |
| 199917568 | 95.0 | 94.57 | 95.0 | 95.0 | 193.712 | 100.0 |  |
| 199950336 | 94.74 | 94.59 | 69.0 | 95.0 | 192.473 | 99.0 |  |
| 199983104 | 94.13 | 94.56 | 8.0 | 95.0 | 191.855 | 99.0 |  |
| 200015872 | 94.03 | 94.52 | 8.0 | 95.0 | 190.763 | 98.0 |  |
