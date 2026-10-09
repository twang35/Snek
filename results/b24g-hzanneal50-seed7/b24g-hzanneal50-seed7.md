# b24g-hzanneal50-seed7

step **200,015,872** · 6104 evals · trailing **94.36** · peak **94.81** @170,819,584 · sef **96.3** · best30 **99.3** @160,235,520

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
| seed | 7 |
| torch_threads | 1 |

![b24g-hzanneal50-seed7](b24g-hzanneal50-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 0.01 | 0.01 | 0.0 | 1.0 | -4.991 | 0.0 |  |
| 65536 | 7.84 | 3.92 | 0.0 | 23.0 | 4.81 | 0.0 |  |
| 98304 | 18.13 | 8.66 | 2.0 | 39.0 | 13.906 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 199655424 | 94.69 | 94.57 | 64.0 | 95.0 | 192.428 | 99.0 |  |
| 199688192 | 95.0 | 94.58 | 95.0 | 95.0 | 193.737 | 100.0 |  |
| 199720960 | 94.82 | 94.58 | 90.0 | 95.0 | 189.55 | 96.0 |  |
| 199753728 | 94.86 | 94.61 | 86.0 | 95.0 | 191.541 | 98.0 |  |
| 199786496 | 94.56 | 94.59 | 82.0 | 95.0 | 188.275 | 95.0 |  |
| 199819264 | 92.74 | 94.52 | 13.0 | 95.0 | 184.416 | 93.0 |  |
| 199852032 | 94.71 | 94.51 | 82.0 | 95.0 | 189.377 | 96.0 |  |
| 199884800 | 94.11 | 94.5 | 6.0 | 95.0 | 191.817 | 99.0 |  |
| 199917568 | 93.22 | 94.45 | 9.0 | 95.0 | 188.944 | 97.0 |  |
| 199950336 | 92.65 | 94.35 | 3.0 | 95.0 | 185.33 | 94.0 |  |
| 199983104 | 94.11 | 94.43 | 22.0 | 95.0 | 188.831 | 96.0 |  |
| 200015872 | 94.15 | 94.36 | 10.0 | 95.0 | 191.865 | 99.0 |  |
