# b9bk-lam95-seed1

step **50,003,968** · 3052 evals · trailing **93.63** · peak **94.49** @30,965,760 · sef **93.7** · best30 **97.1** @31,227,904

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
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.95 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 16.8 |
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

![b9bk-lam95-seed1](b9bk-lam95-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 10.01 | 10.01 | 0.0 | 40.0 | 8.79 | 0.0 |  |
| 32768 | 59.57 | 37.91 | 25.0 | 80.0 | 55.695 | 0.0 |  |
| 49152 | 44.73 | 33.61 | 14.0 | 78.0 | 39.775 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.04 | 94.05 | 6.0 | 95.0 | 185.075 | 93.0 |  |
| 49840128 | 93.47 | 94.04 | 16.0 | 95.0 | 185.46 | 93.0 |  |
| 49856512 | 94.68 | 93.73 | 87.0 | 95.0 | 187.71 | 94.0 |  |
| 49872896 | 94.12 | 93.64 | 63.0 | 95.0 | 188.145 | 95.0 |  |
| 49889280 | 92.8 | 93.72 | 67.0 | 95.0 | 176.875 | 85.0 |  |
| 49905664 | 91.09 | 93.77 | 6.0 | 95.0 | 172.09 | 82.0 |  |
| 49922048 | 93.27 | 94.04 | 33.0 | 95.0 | 183.27 | 91.0 |  |
| 49938432 | 92.92 | 93.93 | 73.0 | 95.0 | 176.995 | 85.0 |  |
| 49954816 | 93.47 | 94.01 | 16.0 | 95.0 | 185.505 | 93.0 |  |
| 49971200 | 92.42 | 93.97 | 10.0 | 95.0 | 179.48 | 88.0 |  |
| 49987584 | 93.7 | 93.88 | 38.0 | 95.0 | 186.685 | 94.0 |  |
| 50003968 | 90.93 | 93.63 | 16.0 | 95.0 | 172.925 | 83.0 |  |
