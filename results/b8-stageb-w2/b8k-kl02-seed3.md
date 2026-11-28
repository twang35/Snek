# b8k-kl02-seed3

step **100,007,936** · 6104 evals · trailing **92.16** · peak **94.41** @58,540,032 · sef **87.9** · best30 **97.3** @13,385,728

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
| fc_layers | (200, 100) |
| graph_eval_episodes | 100 |
| max_steps | 100007936 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 8 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.02 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b8k-kl02-seed3](b8k-kl02-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 17.24 | 17.24 | 0.0 | 35.0 | 12.6 | 0.0 |  |
| 32768 | 29.44 | 23.34 | 7.0 | 48.0 | 24.62 | 0.0 |  |
| 49152 | 29.81 | 25.5 | 11.0 | 46.0 | 24.855 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99827712 | 91.09 | 92.15 | 31.0 | 95.0 | 173.81 | 84.0 |  |
| 99844096 | 93.26 | 92.12 | 12.0 | 95.0 | 183.17 | 91.0 |  |
| 99860480 | 93.03 | 91.66 | 20.0 | 95.0 | 177.83 | 86.0 |  |
| 99876864 | 92.65 | 91.64 | 5.0 | 95.0 | 177.495 | 86.0 |  |
| 99893248 | 93.59 | 91.76 | 59.0 | 95.0 | 181.465 | 89.0 |  |
| 99909632 | 92.08 | 91.96 | 1.0 | 95.0 | 178.87 | 88.0 |  |
| 99926016 | 92.86 | 91.96 | 5.0 | 95.0 | 184.715 | 93.0 |  |
| 99942400 | 91.02 | 91.89 | 3.0 | 95.0 | 181.925 | 92.0 |  |
| 99958784 | 93.56 | 92.07 | 5.0 | 95.0 | 180.125 | 88.0 |  |
| 99975168 | 91.8 | 91.79 | 1.0 | 95.0 | 175.425 | 85.0 |  |
| 99991552 | 93.81 | 91.93 | 11.0 | 95.0 | 185.62 | 93.0 |  |
| 100007936 | 94.94 | 92.16 | 92.0 | 95.0 | 191.86 | 98.0 |  |
