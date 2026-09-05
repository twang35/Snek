# b19p-adameps1e8-seed4

step **376,832** · 23 evals · trailing **50.39** · peak **54.33** @344,064 · sef **0.0** · best30 **0.0** @376,832

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
| ppo_adam_epsilon | 1e-08 |
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
| seed | 4 |
| torch_threads | 1 |

![b19p-adameps1e8-seed4](b19p-adameps1e8-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.33 | 0.33 | 0.0 | 2.0 | -0.623 | 0.0 |  |
| 32768 | 16.49 | 13.94 | 1.0 | 34.0 | 11.954 | 0.0 |  |
| 49152 | 25.01 | 12.67 | 4.0 | 45.0 | 19.978 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 196608 | 58.05 | 47.68 | 12.0 | 95.0 | 54.487 | 1.0 |  |
| 212992 | 74.69 | 35.36 | 25.0 | 95.0 | 82.047 | 10.0 |  |
| 229376 | 67.74 | 48.74 | 17.0 | 95.0 | 65.822 | 2.0 |  |
| 245760 | 77.43 | 53.0 | 31.0 | 95.0 | 83.277 | 8.0 |  |
| 262144 | 78.89 | 38.71 | 33.0 | 95.0 | 87.34 | 10.0 |  |
| 278528 | 80.77 | 51.83 | 34.0 | 95.0 | 90.397 | 11.0 |  |
| 294912 | 81.44 | 41.76 | 48.0 | 95.0 | 87.061 | 7.0 |  |
| 311296 | 82.4 | 31.78 | 52.0 | 95.0 | 91.092 | 10.0 |  |
| 327680 | 83.16 | 47.07 | 36.0 | 95.0 | 92.655 | 11.0 |  |
| 344064 | 83.62 | 54.33 | 4.0 | 95.0 | 98.257 | 16.0 |  |
| 360448 | 83.28 | 44.82 | 56.0 | 95.0 | 94.914 | 13.0 |  |
| 376832 | 81.7 | 50.39 | 38.0 | 95.0 | 88.46 | 8.0 |  |
