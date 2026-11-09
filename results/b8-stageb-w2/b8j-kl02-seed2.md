# b8j-kl02-seed2

step **100,007,936** · 6104 evals · trailing **92.39** · peak **94.46** @60,145,664 · sef **85.2** · best30 **96.4** @18,546,688

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
| seed | 2 |
| torch_threads | 1 |

![b8j-kl02-seed2](b8j-kl02-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 3.4 | 3.4 | 1.0 | 15.0 | -1.6 | 0.0 |  |
| 32768 | 19.57 | 11.48 | 4.0 | 40.0 | 14.57 | 0.0 |  |
| 49152 | 26.43 | 16.47 | 9.0 | 51.0 | 21.43 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99827712 | 93.1 | 92.3 | 55.0 | 95.0 | 176.5 | 85.0 |  |
| 99844096 | 94.15 | 92.63 | 74.0 | 95.0 | 185.87 | 93.0 |  |
| 99860480 | 94.15 | 92.35 | 69.0 | 95.0 | 185.87 | 93.0 |  |
| 99876864 | 94.49 | 92.92 | 65.0 | 95.0 | 187.25 | 94.0 |  |
| 99893248 | 93.44 | 92.93 | 69.0 | 95.0 | 182.085 | 90.0 |  |
| 99909632 | 88.72 | 92.77 | 38.0 | 95.0 | 148.2 | 62.0 |  |
| 99926016 | 94.56 | 92.36 | 83.0 | 95.0 | 188.36 | 95.0 |  |
| 99942400 | 92.97 | 92.73 | 70.0 | 95.0 | 177.41 | 86.0 |  |
| 99958784 | 94.18 | 92.55 | 70.0 | 95.0 | 185.9 | 93.0 |  |
| 99975168 | 92.89 | 92.87 | 49.0 | 95.0 | 182.53 | 91.0 |  |
| 99991552 | 94.22 | 92.87 | 82.0 | 95.0 | 183.86 | 91.0 |  |
| 100007936 | 93.12 | 92.39 | 67.0 | 95.0 | 174.44 | 83.0 |  |
