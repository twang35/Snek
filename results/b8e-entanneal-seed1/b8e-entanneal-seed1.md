# b8e-entanneal-seed1

step **100,007,936** · 6104 evals · trailing **93.76** · peak **94.42** @4,046,848 · sef **87.1** · best30 **97.0** @4,063,232

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
| ppo_entropy_coef_final | 0.001 |
| ppo_epochs | 8 |
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

![b8e-entanneal-seed1](b8e-entanneal-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 11.07 | 11.07 | 2.0 | 25.0 | 6.925 | 0.0 |  |
| 32768 | 30.03 | 25.46 | 3.0 | 58.0 | 25.615 | 0.0 |  |
| 49152 | 24.34 | 17.7 | 1.0 | 52.0 | 19.745 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99827712 | 94.07 | 93.6 | 62.0 | 95.0 | 186.83 | 94.0 |  |
| 99844096 | 94.93 | 93.61 | 90.0 | 95.0 | 191.85 | 98.0 |  |
| 99860480 | 94.48 | 93.74 | 64.0 | 95.0 | 189.365 | 96.0 |  |
| 99876864 | 94.58 | 93.8 | 81.0 | 95.0 | 187.385 | 94.0 |  |
| 99893248 | 94.55 | 93.71 | 76.0 | 95.0 | 190.52 | 97.0 |  |
| 99909632 | 93.92 | 93.76 | 36.0 | 95.0 | 188.85 | 96.0 |  |
| 99926016 | 94.03 | 93.85 | 73.0 | 95.0 | 186.925 | 94.0 |  |
| 99942400 | 94.21 | 93.79 | 68.0 | 95.0 | 187.06 | 94.0 |  |
| 99958784 | 93.58 | 93.84 | 1.0 | 95.0 | 188.465 | 96.0 |  |
| 99975168 | 93.39 | 93.8 | 3.0 | 95.0 | 188.32 | 96.0 |  |
| 99991552 | 94.83 | 93.8 | 86.0 | 95.0 | 191.75 | 98.0 |  |
| 100007936 | 93.29 | 93.76 | 66.0 | 95.0 | 171.49 | 80.0 |  |
