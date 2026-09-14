# b9bv-lam97-seed4

step **50,003,968** · 3052 evals · trailing **94.27** · peak **94.56** @30,162,944 · sef **94.0** · best30 **97.8** @30,162,944

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
| ppo_gae_lambda | 0.97 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 25.2 |
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

![b9bv-lam97-seed4](b9bv-lam97-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.63 | 0.63 | 0.0 | 4.0 | -0.5 | 0.0 |  |
| 32768 | 8.72 | 21.14 | 1.0 | 34.0 | 6.285 | 0.0 |  |
| 49152 | 28.56 | 24.25 | 8.0 | 48.0 | 23.56 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.9 | 94.19 | 20.0 | 95.0 | 188.92 | 96.0 |  |
| 49840128 | 94.66 | 94.19 | 68.0 | 95.0 | 190.63 | 97.0 |  |
| 49856512 | 94.13 | 94.15 | 8.0 | 95.0 | 192.135 | 99.0 |  |
| 49872896 | 94.94 | 94.22 | 89.0 | 95.0 | 192.945 | 99.0 |  |
| 49889280 | 95.0 | 94.21 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49905664 | 94.93 | 94.25 | 88.0 | 95.0 | 192.935 | 99.0 |  |
| 49922048 | 94.39 | 94.24 | 74.0 | 95.0 | 187.375 | 94.0 |  |
| 49938432 | 93.68 | 94.22 | 76.0 | 95.0 | 180.74 | 88.0 |  |
| 49954816 | 94.55 | 94.21 | 81.0 | 95.0 | 189.57 | 96.0 |  |
| 49971200 | 93.98 | 94.28 | 77.0 | 95.0 | 183.03 | 90.0 |  |
| 49987584 | 93.79 | 94.28 | 72.0 | 95.0 | 182.84 | 90.0 |  |
| 50003968 | 94.35 | 94.27 | 80.0 | 95.0 | 187.38 | 94.0 |  |
