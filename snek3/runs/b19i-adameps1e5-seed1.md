# b19i-adameps1e5-seed1

step **6,701,056** · 405 evals · trailing **90.78** · peak **93.98** @3,096,576 · sef **37.3** · best30 **90.4** @4,554,752

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 16384 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 50003968 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-05 |
| ppo_anneal_fraction | 1.0 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 1 |
| torch_threads | 1 |

![b19i-adameps1e5-seed1](b19i-adameps1e5-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 19.35 | 30.84 | 1.0 | 39.0 | 16.137 | 0.0 |  |
| 32768 | 45.1 | 34.69 | 5.0 | 84.0 | 40.128 | 0.0 |  |
| 49152 | 34.75 | 34.75 | 10.0 | 70.0 | 29.673 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 6455296 | 86.95 | 89.07 | 28.0 | 95.0 | 168.675 | 83.0 |  |
| 6471680 | 86.96 | 88.61 | 2.0 | 95.0 | 163.693 | 78.0 |  |
| 6488064 | 90.25 | 88.87 | 44.0 | 95.0 | 175.96 | 87.0 |  |
| 6504448 | 91.99 | 89.3 | 53.0 | 95.0 | 181.681 | 91.0 |  |
| 6520832 | 90.85 | 90.65 | 55.0 | 95.0 | 176.551 | 87.0 |  |
| 6537216 | 89.61 | 90.63 | 44.0 | 95.0 | 173.285 | 85.0 |  |
| 6553600 | 87.15 | 90.46 | 51.0 | 95.0 | 163.89 | 78.0 |  |
| 6569984 | 89.49 | 90.97 | 43.0 | 95.0 | 174.194 | 86.0 |  |
| 6586368 | 89.63 | 91.12 | 43.0 | 95.0 | 173.349 | 85.0 |  |
| 6668288 | 90.08 | 90.89 | 44.0 | 95.0 | 174.776 | 86.0 |  |
| 6684672 | 90.09 | 90.68 | 29.0 | 95.0 | 176.783 | 88.0 |  |
| 6701056 | 91.07 | 90.78 | 39.0 | 95.0 | 179.746 | 90.0 |  |
