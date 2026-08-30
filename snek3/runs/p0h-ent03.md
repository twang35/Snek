# p0h-ent03

step **10,010,624** · 611 evals · trailing **85.29** · peak **94.16** @4,947,968 · sef **36.7** · best30 **90.6** @7,602,176

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
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 10000000 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_entropy_coef | 0.03 |
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
| seed | 1 |
| torch_threads | 1 |

## Resumes

Resumed at 3,014,656

![p0h-ent03](p0h-ent03.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 16.62 | 29.77 | 1.0 | 39.0 | 14.32 | 0.0 |  |
| 32768 | 46.69 | 34.08 | 12.0 | 81.0 | 41.78 | 0.0 |  |
| 49152 | 34.66 | 33.77 | 14.0 | 71.0 | 29.705 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 9830400 | 89.52 | 84.52 | 43.0 | 95.0 | 175.585 | 87.0 |  |
| 9846784 | 92.13 | 84.52 | 44.0 | 95.0 | 182.175 | 91.0 |  |
| 9863168 | 92.79 | 84.76 | 24.0 | 95.0 | 186.815 | 95.0 |  |
| 9879552 | 85.09 | 84.68 | 27.0 | 95.0 | 160.21 | 76.0 |  |
| 9895936 | 84.0 | 84.78 | 26.0 | 95.0 | 161.11 | 78.0 |  |
| 9912320 | 92.63 | 85.13 | 35.0 | 95.0 | 186.655 | 95.0 |  |
| 9928704 | 91.05 | 86.66 | 43.0 | 95.0 | 179.105 | 89.0 |  |
| 9945088 | 88.07 | 85.09 | 28.0 | 95.0 | 173.14 | 86.0 |  |
| 9961472 | 72.81 | 84.6 | 24.0 | 95.0 | 135.99 | 64.0 |  |
| 9977856 | 83.52 | 84.67 | 8.0 | 95.0 | 161.625 | 79.0 |  |
| 9994240 | 88.02 | 85.07 | 28.0 | 95.0 | 174.085 | 87.0 |  |
| 10010624 | 84.92 | 85.29 | 28.0 | 95.0 | 166.01 | 82.0 |  |
