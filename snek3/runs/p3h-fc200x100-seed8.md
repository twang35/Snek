# p3h-fc200x100-seed8

step **215,105,536** · 13125 evals · trailing **92.92** · peak **94.62** @63,324,160 · sef **95.5** · best30 **97.9** @162,856,960

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 16384 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 6 |
| fc_layers | (200, 100) |
| graph_eval_episodes | 100 |
| max_steps | 400000000 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 8 |
| torch_threads | 1 |

![p3h-fc200x100-seed8](p3h-fc200x100-seed8.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 16.12 | 16.12 | 0.0 | 39.0 | 12.695 | 0.0 |  |
| 32768 | 38.11 | 29.83 | 9.0 | 69.0 | 33.605 | 0.0 |  |
| 49152 | 34.56 | 25.34 | 8.0 | 65.0 | 29.56 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 214859776 | 84.81 | 92.36 | 3.0 | 95.0 | 170.425 | 87.0 |  |
| 214876160 | 90.66 | 92.18 | 8.0 | 95.0 | 180.435 | 91.0 |  |
| 214892544 | 89.01 | 93.8 | 7.0 | 95.0 | 177.745 | 90.0 |  |
| 214908928 | 88.17 | 94.01 | 7.0 | 95.0 | 174.87 | 88.0 |  |
| 214925312 | 93.08 | 93.96 | 1.0 | 95.0 | 187.105 | 95.0 |  |
| 214941696 | 92.7 | 93.65 | 7.0 | 95.0 | 185.595 | 94.0 |  |
| 214958080 | 92.18 | 92.95 | 7.0 | 95.0 | 183.22 | 92.0 |  |
| 214974464 | 91.44 | 92.3 | 3.0 | 95.0 | 184.425 | 94.0 |  |
| 215056384 | 92.51 | 93.58 | 1.0 | 95.0 | 186.445 | 95.0 |  |
| 215072768 | 88.11 | 93.03 | 7.0 | 95.0 | 103.14 | 19.0 |  |
| 215089152 | 94.33 | 93.01 | 57.0 | 95.0 | 190.255 | 97.0 |  |
| 215105536 | 94.21 | 92.92 | 19.0 | 95.0 | 191.175 | 98.0 |  |
