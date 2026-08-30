# p3c-fc200x100-seed3

step **58,507,264** · 3566 evals · trailing **94.09** · peak **94.63** @31,506,432 · sef **92.5** · best30 **98.2** @31,342,592

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
| seed | 3 |
| torch_threads | 1 |

![p3c-fc200x100-seed3](p3c-fc200x100-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 18.75 | 18.75 | 2.0 | 35.0 | 13.795 | 0.0 |  |
| 32768 | 31.14 | 27.8 | 9.0 | 53.0 | 26.185 | 0.0 |  |
| 49152 | 30.03 | 24.39 | 15.0 | 47.0 | 25.075 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 58245120 | 94.15 | 94.18 | 10.0 | 95.0 | 192.155 | 99.0 |  |
| 58261504 | 92.43 | 94.11 | 9.0 | 95.0 | 185.415 | 94.0 |  |
| 58277888 | 94.42 | 94.11 | 49.0 | 95.0 | 190.39 | 97.0 |  |
| 58294272 | 94.43 | 94.17 | 59.0 | 95.0 | 190.445 | 97.0 |  |
| 58310656 | 95.0 | 94.13 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 58327040 | 92.53 | 94.11 | 3.0 | 95.0 | 186.555 | 95.0 |  |
| 58343424 | 93.01 | 94.1 | 9.0 | 95.0 | 187.035 | 95.0 |  |
| 58359808 | 94.53 | 94.12 | 75.0 | 95.0 | 189.55 | 96.0 |  |
| 58376192 | 93.59 | 94.11 | 14.0 | 95.0 | 189.605 | 97.0 |  |
| 58392576 | 93.62 | 94.07 | 26.0 | 95.0 | 187.645 | 95.0 |  |
| 58408960 | 94.0 | 94.15 | 18.0 | 95.0 | 189.925 | 97.0 |  |
| 58507264 | 92.52 | 94.09 | 10.0 | 95.0 | 187.54 | 96.0 |  |
